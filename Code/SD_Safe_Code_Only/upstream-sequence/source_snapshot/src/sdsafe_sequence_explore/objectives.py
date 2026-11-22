"""Target-aligned, fully mask-aware losses for fine_sequence_explore_v1 (Lane T).

Semantics that must not drift:

* A strict-final positive episode bag is weak episode-level supervision; it is never a
  per-position malice/unauthorized-action label.
* Prefix-state positives are goal-prefix-state positives, not universal unsafe-action
  ground truth.
* ``-1`` labels are unknown/masked and never become negative targets.
* Padding, E context rows and empty CT bags never contribute a negative loss.
* E context remains available to causal encoders and to inference; it is not a
  supervised endpoint in the CT primary objective.
* Class weights/factors are applied exactly once and the actual supervised
  positive/negative mass is reported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch
from torch.nn import functional as F

MISSING = -1
POLICIES = ("L0", "L1", "L2", "L3", "L4")
LENGTH_WEIGHTINGS = ("endpoint", "episode")
MIL_REDUCTIONS = ("max", "logsumexp")


@dataclass(frozen=True)
class ObjectiveConfig:
    # Pointwise endpoint supervision (L0/L1/L2/L3 masks, or explicitly named L4 control).
    endpoint_policy: str | None = "L1"
    endpoint_weight: float = 1.0
    length_weighting: str = "endpoint"
    endpoint_pos_weight: float | None = None
    # Direct strict-final episode MIL over causal CT outputs.
    mil_weight: float = 0.0
    mil_reduction: str = "max"
    mil_beta: float = 1.0
    mil_pos_weight: float | None = None
    # Auxiliary separate-head supervision (L1 endpoint or prefix-state).
    aux_weight: float = 0.0
    aux_target: str = "policy"  # "policy" or "prefix_state"
    aux_policy: str = "L1"
    aux_pos_weight: float | None = None
    # Bounded TRAIN-only ranking extension; disabled by default.
    ranking_weight: float = 0.0
    ranking_margin: float = 1.0

    def __post_init__(self) -> None:
        if self.endpoint_policy is not None and self.endpoint_policy not in POLICIES:
            raise ValueError(f"unknown endpoint policy {self.endpoint_policy!r}")
        if self.aux_policy not in POLICIES:
            raise ValueError(f"unknown aux policy {self.aux_policy!r}")
        if self.aux_target not in ("policy", "prefix_state"):
            raise ValueError("aux_target must be 'policy' or 'prefix_state'")
        if self.length_weighting not in LENGTH_WEIGHTINGS:
            raise ValueError("bad length_weighting")
        if self.mil_reduction not in MIL_REDUCTIONS:
            raise ValueError("bad mil_reduction")
        for name in ("endpoint_weight", "mil_weight", "aux_weight", "ranking_weight"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("endpoint_pos_weight", "mil_pos_weight", "aux_pos_weight"):
            value = getattr(self, name)
            if value is not None and float(value) <= 0:
                raise ValueError(f"{name} must be positive when set")
        if self.mil_reduction == "logsumexp" and float(self.mil_beta) <= 0:
            raise ValueError("mil_beta must be positive")
        if float(self.ranking_margin) < 0:
            raise ValueError("ranking_margin must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObjectiveConfig":
        return cls(**dict(data))


def _supervised(labels: torch.Tensor, valid_mask: torch.Tensor, policy_mask: torch.Tensor | None) -> torch.Tensor:
    labels = labels.to(torch.int64)
    mask = valid_mask.to(torch.bool) & (labels >= 0)
    if policy_mask is not None:
        mask = mask & ~policy_mask.to(torch.bool)
    return mask


def endpoint_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    policy_mask: torch.Tensor | None = None,
    *,
    length_weighting: str = "endpoint",
    pos_weight: float | None = None,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Pointwise BCE over honest endpoint masks.

    ``labels`` is int8/int64 with -1 for unknown/masked rows.  Padding and policy-masked
    rows are excluded.  Unknown is never relabelled 0.  ``length_weighting`` is explicit:
    ``endpoint`` averages every supervised endpoint; ``episode`` first averages within an
    episode and then across episodes so long sequences cannot silently dominate.
    """
    if logits.shape != labels.shape or logits.shape != valid_mask.shape:
        raise ValueError("logits/labels/valid_mask shapes must match")
    if length_weighting not in LENGTH_WEIGHTINGS:
        raise ValueError("bad length_weighting")
    supervised = _supervised(labels, valid_mask, policy_mask)
    targets = labels.to(torch.float32).clamp_min(0.0)
    positive = targets > 0
    class_weight_value = 1.0 if pos_weight is None else float(pos_weight)
    class_weight = torch.where(positive, torch.full_like(targets, class_weight_value), torch.ones_like(targets))
    sample_weight_used = sample_weight is not None
    if sample_weight is None:
        weights = torch.ones_like(targets)
    else:
        weights = sample_weight.to(logits.dtype)
        supervised = supervised & torch.isfinite(weights)
    raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    class_weighted = raw * class_weight
    weighted = class_weighted * weights
    count = int(supervised.sum().item())
    active_episodes = int(supervised.any(dim=1).sum().item()) if logits.ndim == 2 else 0
    if count == 0:
        loss = weighted.sum() * 0.0
        denominator = 0
        normalization = "endpoint_mean" if length_weighting == "endpoint" else "episode_mean_of_endpoint_means"
    elif length_weighting == "endpoint":
        loss = weighted[supervised].mean()
        denominator = count
        normalization = "endpoint_mean"
    elif length_weighting == "episode":
        per_episode_sum = weighted.masked_fill(~supervised, 0.0).sum(dim=1)
        counts = supervised.sum(dim=1)
        episode_means = per_episode_sum / counts.clamp_min(1)
        active = counts > 0
        loss = episode_means[active].mean()
        denominator = int(active.sum().item())
        normalization = "episode_mean_of_endpoint_means"
    else:  # pragma: no cover - validated
        raise ValueError(length_weighting)
    positive_endpoints = int((positive & supervised).sum().item())
    negative_endpoints = count - positive_endpoints
    effective = class_weight * weights
    positive_weight_mass = float(effective[supervised & positive].sum().item())
    negative_weight_mass = float(effective[supervised & ~positive].sum().item())
    sample_weight_mass = float(weights[supervised].sum().item())
    return loss, {
        "loss": float(loss.detach().item()),
        "supervised_endpoints": count,
        "positive_endpoints": positive_endpoints,
        "negative_endpoints": negative_endpoints,
        "active_episodes": active_episodes,
        "positive_mass": positive_weight_mass,
        "negative_mass": negative_weight_mass,
        "class_weight_applied": class_weight_value != 1.0,
        "class_weight_positive": class_weight_value,
        "sample_weight_used": bool(sample_weight_used),
        "sample_weight_effective_mass": sample_weight_mass,
        "raw_bce_sum_unweighted": float(raw[supervised].sum().item()),
        "class_weighted_bce_sum": float(class_weighted[supervised].sum().item()),
        "sample_weighted_bce_sum": float(weighted[supervised].sum().item()),
        "loss_denominator": denominator,
        "loss_normalization": normalization,
        "length_weighting": length_weighting,
        "pos_weight": None if pos_weight is None else class_weight_value,
        "mass_definition": "positive_mass/negative_mass are class-weight x sample-weight endpoint masses, not counts",
    }


def mil_eligible(valid_mask: torch.Tensor, ct_mask: torch.Tensor | None) -> torch.Tensor:
    """Real CT inference positions eligible for episode pooling."""
    eligible = valid_mask.to(torch.bool)
    if ct_mask is not None:
        eligible = eligible & ct_mask.to(torch.bool)
    return eligible


def mil_pool(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    ct_mask: torch.Tensor | None = None,
    *,
    reduction: str = "max",
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool per-position CT logits into one raw episode score.

    Returns ``(pooled, eligible_mask, empty_bag)``.  Empty bags pool to ``-inf`` and are
    reported as ``empty_bag=True``; they are never assigned a negative target.
    """
    if reduction not in MIL_REDUCTIONS:
        raise ValueError("bad MIL reduction")
    if float(beta) <= 0:
        raise ValueError("MIL beta must be positive")
    eligible = mil_eligible(valid_mask, ct_mask)
    empty = ~eligible.any(dim=1)
    # -inf is the exact no-evidence sentinel: padded/E/non-CT positions contribute
    # zero softmax mass and cannot overflow under multiplication by beta.
    masked = logits.masked_fill(~eligible, float("-inf"))
    if reduction == "max":
        pooled = masked.max(dim=1).values
    else:
        counts = eligible.sum(dim=1).to(logits.dtype)
        # Normalized LSE: constant logits stay constant regardless of bag length.
        pooled = (
            torch.logsumexp(masked * float(beta), dim=1) - torch.log(counts.clamp_min(1.0))
        ) / float(beta)
    pooled = torch.where(empty, torch.full_like(pooled, float("-inf")), pooled)
    return pooled, eligible, empty


def mil_bag_loss(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    ct_mask: torch.Tensor | None,
    bag_label: torch.Tensor,
    *,
    reduction: str = "max",
    beta: float = 1.0,
    pos_weight: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Episode MIL BCE against the strict-final bag label.

    Positive bags provide weak supervision only; they are not per-position labels.
    Empty CT bags are excluded from the loss and counted explicitly.  Evaluation must
    still keep them in the denominator with no alarm (``-inf`` score).
    """
    pooled, eligible, empty = mil_pool(logits, valid_mask, ct_mask, reduction=reduction, beta=beta)
    trainable = ~empty
    if bool(trainable.any()):
        pw = None if pos_weight is None else torch.as_tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
        per_bag = F.binary_cross_entropy_with_logits(
            pooled[trainable], bag_label.to(torch.float32)[trainable], reduction="none", pos_weight=pw
        )
        loss = per_bag.mean()
    else:
        loss = logits.sum() * 0.0
    labels = bag_label.to(torch.float32)
    pos_bags = int(((labels > 0) & trainable).sum().item())
    neg_bags = int(((labels <= 0) & trainable).sum().item())
    class_weight_value = 1.0 if pos_weight is None else float(pos_weight)
    return loss, {
        "loss": float(loss.detach().item()),
        "trainable_bags": int(trainable.sum().item()),
        "empty_bags": int(empty.sum().item()),
        "positive_bags": pos_bags,
        "negative_bags": neg_bags,
        "positive_mass": float(class_weight_value * pos_bags),
        "negative_mass": float(neg_bags),
        "class_weight_applied": class_weight_value != 1.0,
        "class_weight_positive": class_weight_value,
        "mass_definition": "MIL masses are bag-level class weights, not endpoint counts",
        "reduction": reduction,
        "beta": float(beta),
        "eligible_ct_positions": int(eligible.sum().item()),
    }


def _episode_prefix_conflicts(batch: Any) -> int:
    prefix = getattr(batch, "prefix_state", None) if not isinstance(batch, Mapping) else batch.get("prefix_state")
    bag = getattr(batch, "bag_label", None) if not isinstance(batch, Mapping) else batch.get("bag_label")
    valid = getattr(batch, "padding_mask", None) if not isinstance(batch, Mapping) else batch.get("padding_mask")
    ct = getattr(batch, "ct_mask", None) if not isinstance(batch, Mapping) else batch.get("ct_mask")
    if prefix is None or bag is None or valid is None:
        return 0
    positive_prefix = (prefix.to(torch.int64) > 0) & valid.to(torch.bool)
    if ct is not None:
        positive_prefix = positive_prefix & ct.to(torch.bool)
    has_prefix = positive_prefix.any(dim=1)
    final_negative = bag.to(torch.float32) <= 0
    return int((has_prefix & final_negative).sum().item())


def aux_endpoint_loss(
    aux_logits: torch.Tensor | None,
    batch: Any,
    config: ObjectiveConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Auxiliary L1 endpoint or prefix-state loss on a separate head."""
    if aux_logits is None:
        return torch.zeros((), dtype=torch.float32), {"loss": 0.0, "enabled": False}
    valid = getattr(batch, "padding_mask")
    ct = getattr(batch, "ct_mask", None)
    if config.aux_target == "prefix_state":
        prefix = getattr(batch, "prefix_state")
        exact = getattr(batch, "exact")
        if prefix is None or exact is None:
            raise ValueError("prefix_state aux target needs prefix_state/exact in the batch")
        labels = prefix.to(torch.int64)
        mask = ~(valid.to(torch.bool) & exact.to(torch.bool))
        if ct is not None:
            # CT is the primary output view; E remains causal context, not an aux target.
            mask = mask | ~ct.to(torch.bool)
        loss, stats = endpoint_bce(
            aux_logits,
            labels,
            valid,
            mask,
            length_weighting="endpoint",
            pos_weight=config.aux_pos_weight,
        )
        stats["target"] = "prefix_state"
    else:
        labels = batch.policy_labels[config.aux_policy]
        masks = batch.policy_masks[config.aux_policy]
        loss, stats = endpoint_bce(
            aux_logits,
            labels,
            valid,
            masks,
            length_weighting="endpoint",
            pos_weight=config.aux_pos_weight,
        )
        stats["target"] = f"policy_{config.aux_policy}"
    stats["conflict_episodes"] = _episode_prefix_conflicts(batch)
    stats["enabled"] = True
    return loss, stats


def bag_ranking_loss(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    ct_mask: torch.Tensor | None,
    bag_label: torch.Tensor,
    *,
    margin: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Bounded pairwise margin ranking on episode-pooled max logits (TRAIN only).

    The caller is responsible for supplying TRAIN-only scores/labels.  No DEV/SELECT
    score may select training negatives.  This is an optional, explicitly named
    extension, not a claimed partial-AUC/CVaR estimator.
    """
    pooled, _, empty = mil_pool(logits, valid_mask, ct_mask, reduction="max")
    labels = bag_label.to(torch.float32)
    positive = (labels > 0) & ~empty
    negative = (labels <= 0) & ~empty
    if not bool(positive.any()) or not bool(negative.any()):
        return logits.sum() * 0.0, {"loss": 0.0, "pairs": 0}
    pos_scores = pooled[positive].unsqueeze(1)
    neg_scores = pooled[negative].unsqueeze(0)
    pair_loss = F.relu(float(margin) - pos_scores + neg_scores)
    loss = pair_loss.mean()
    return loss, {"loss": float(loss.detach().item()), "pairs": int(pair_loss.numel()), "margin": float(margin)}


def compute_objective(
    config: ObjectiveConfig,
    logits: torch.Tensor,
    batch: Any,
    aux_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compose the configured endpoint/MIL/aux/ranking terms exactly once."""
    valid = getattr(batch, "padding_mask")
    total = logits.sum() * 0.0
    stats: dict[str, Any] = {"terms": {}}
    if config.endpoint_policy is not None and float(config.endpoint_weight) > 0:
        labels = batch.policy_labels[config.endpoint_policy]
        masks = batch.policy_masks[config.endpoint_policy]
        term, term_stats = endpoint_bce(
            logits,
            labels,
            valid,
            masks,
            length_weighting=config.length_weighting,
            pos_weight=config.endpoint_pos_weight,
        )
        total = total + float(config.endpoint_weight) * term
        term_stats["coefficient"] = float(config.endpoint_weight)
        term_stats["loss_contribution"] = float(config.endpoint_weight) * float(term_stats["loss"])
        term_stats["units"] = "endpoint_positions"
        stats["terms"]["endpoint"] = term_stats
    if float(config.mil_weight) > 0:
        bag = getattr(batch, "bag_label", None)
        if bag is None:
            raise ValueError("MIL objective needs bag_label")
        term, term_stats = mil_bag_loss(
            logits,
            valid,
            getattr(batch, "ct_mask", None),
            bag,
            reduction=config.mil_reduction,
            beta=config.mil_beta,
            pos_weight=config.mil_pos_weight,
        )
        total = total + float(config.mil_weight) * term
        term_stats["coefficient"] = float(config.mil_weight)
        term_stats["loss_contribution"] = float(config.mil_weight) * float(term_stats["loss"])
        term_stats["units"] = "episode_bags"
        stats["terms"]["mil"] = term_stats
    if float(config.aux_weight) > 0:
        term, term_stats = aux_endpoint_loss(aux_logits, batch, config)
        if aux_logits is None:
            raise ValueError("aux_weight > 0 requires a model with aux_head=True")
        total = total + float(config.aux_weight) * term
        term_stats["coefficient"] = float(config.aux_weight)
        term_stats["loss_contribution"] = float(config.aux_weight) * float(term_stats["loss"])
        term_stats["units"] = "endpoint_positions"
        stats["terms"]["aux"] = term_stats
    if float(config.ranking_weight) > 0:
        bag = getattr(batch, "bag_label", None)
        if bag is None:
            raise ValueError("ranking objective needs bag_label")
        term, term_stats = bag_ranking_loss(
            logits,
            valid,
            getattr(batch, "ct_mask", None),
            bag,
            margin=config.ranking_margin,
        )
        total = total + float(config.ranking_weight) * term
        term_stats["coefficient"] = float(config.ranking_weight)
        term_stats["loss_contribution"] = float(config.ranking_weight) * float(term_stats["loss"])
        term_stats["units"] = "episode_pairs"
        stats["terms"]["ranking"] = term_stats
    endpoint = stats["terms"].get("endpoint", {})
    mil = stats["terms"].get("mil", {})
    aux = stats["terms"].get("aux", {})
    stats["mass_summary"] = {
        "endpoint_positive_weight_mass": float(endpoint.get("positive_mass", 0.0)),
        "endpoint_negative_weight_mass": float(endpoint.get("negative_mass", 0.0)),
        "aux_positive_weight_mass": float(aux.get("positive_mass", 0.0)),
        "aux_negative_weight_mass": float(aux.get("negative_mass", 0.0)),
        "mil_positive_bag_weight_mass": float(mil.get("positive_mass", 0.0)),
        "mil_negative_bag_weight_mass": float(mil.get("negative_mass", 0.0)),
        "note": "endpoint/aux positions and MIL bags have different units; do not sum masses across terms",
    }
    stats["loss"] = float(total.detach().item())
    stats["conflict_episodes"] = _episode_prefix_conflicts(batch)
    return total, stats


__all__ = [
    "MISSING",
    "POLICIES",
    "ObjectiveConfig",
    "endpoint_bce",
    "mil_eligible",
    "mil_pool",
    "mil_bag_loss",
    "aux_endpoint_loss",
    "bag_ranking_loss",
    "compute_objective",
]
