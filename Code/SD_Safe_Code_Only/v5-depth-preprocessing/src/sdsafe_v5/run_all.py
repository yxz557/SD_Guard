"""End-to-end driver: fit the pre-registered arms, evaluate DEV, write the report.

Nothing is selected on DEV.  TRAIN-only selection uses group fold 0 (never fitted).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import ART, BASELINE, FOLDS, TRAIN_SEEDS, V2_BASELINE
from .common import atomic_write_json, json_load
from .train_eval import (
    auc, build, confusions, evaluate, fpr_threshold, group_folds, load_split, train_arm,
)

ARMS = ("C", "E", "F")
TAGS = ("ho", "full")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261111


def fit_all() -> dict[str, Any]:
    runs = []
    for arm in ARMS:
        for seed in TRAIN_SEEDS:
            runs.append(train_arm(arm, seed, (1, 2, 3), "ho"))
            runs.append(train_arm(arm, seed, (0, 1, 2, 3), "full"))
    atomic_write_json(ART / "runs.json", {"runs": runs, "arms": list(ARMS), "tags": list(TAGS),
                                          "seeds": list(TRAIN_SEEDS)})
    return {"runs": runs}


def heldout_selection() -> dict[str, Any]:
    """TRAIN-only selection: episode AUC/FN on the never-fitted fold-0 groups."""
    data = load_split("train")
    folds = group_folds(data["group_id"].tolist())
    held = np.isin(folds, [0])
    episodes = np.asarray(data["episode_id"], dtype=object)[held]
    labels = np.asarray(data["episode_label"], dtype=bool)[held]
    rows_held = np.asarray(data["row_index"])[held]
    out = {}
    from .train_eval import score_rows

    for arm in ARMS:
        per_seed = []
        for seed in TRAIN_SEEDS:
            logits = score_rows(arm, seed, "ho", "train")
            sel = np.isin(np.asarray(data["row_index"]), rows_held)
            uniq, inv = np.unique(episodes, return_inverse=True)
            ep = np.full(uniq.shape[0], -np.inf)
            np.maximum.at(ep, inv, logits[sel].astype(np.float64))
            per_seed.append(ep)
        score = np.mean(np.stack(per_seed), axis=0)
        eq = np.searchsorted(np.unique(episodes), episodes)
        ep_labels = np.zeros(uniq.shape[0], dtype=bool)
        for e, lab in zip(episodes, labels):
            ep_labels[np.searchsorted(uniq, e)] = bool(lab)
        thr = fpr_threshold(ep_labels, score)
        cm = confusions(ep_labels, score, thr)
        out[arm] = {
            "episodes": int(uniq.shape[0]),
            "positives": int(ep_labels.sum()),
            **cm,
            "fnr": cm["fn"] / max(cm["tp"] + cm["fn"], 1),
            "fpr": cm["fp"] / max(cm["fp"] + cm["tn"], 1),
            "auc": auc(ep_labels, score),
            "threshold": thr,
        }
    return out


def _bootstrap(y: np.ndarray, base: np.ndarray, cand: np.ndarray, groups: np.ndarray,
               thr_base: float, thr_cand: float) -> dict[str, Any]:
    uniq = sorted(set(groups.tolist()))
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    d_fn, d_fp, d_fnr, d_fpr = [], [], [], []
    valid = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        rows = np.concatenate([index[uniq[i]] for i in pick])
        yb = y[rows]
        if yb.sum() == 0 or (~yb).sum() == 0:
            continue
        valid += 1
        mb = confusions(yb, base[rows], thr_base)
        mc = confusions(yb, cand[rows], thr_cand)
        d_fn.append(mc["fn"] - mb["fn"])
        d_fp.append(mc["fp"] - mb["fp"])
        d_fnr.append(mc["fn"] / max(mc["tp"] + mc["fn"], 1) - mb["fn"] / max(mb["tp"] + mb["fn"], 1))
        d_fpr.append(mc["fp"] / max(mc["fp"] + mc["tn"], 1) - mb["fp"] / max(mb["fp"] + mb["tn"], 1))

    def pack(v):
        a = np.asarray(v, dtype=np.float64)
        return {
            "mean": float(a.mean()),
            "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
            "p_le_0": float((a <= 0).mean()),
        }

    return {
        "replicates_requested": BOOTSTRAP_REPLICATES,
        "replicates_valid": valid,
        "seed": BOOTSTRAP_SEED,
        "d_fn": pack(d_fn),
        "d_fp": pack(d_fp),
        "d_fnr": pack(d_fnr),
        "d_fpr": pack(d_fpr),
    }


def dev_evaluation() -> dict[str, Any]:
    data = load_split("dev")
    episodes = np.asarray(data["episode_id"], dtype=object)
    uniq = np.unique(episodes)
    ep_label = np.zeros(uniq.shape[0], dtype=bool)
    for i, e in enumerate(episodes):
        ep_label[np.searchsorted(uniq, e)] = bool(data["episode_label"][i])
    groups = np.asarray(
        [data["group_id"][np.searchsorted(episodes, e)] for e in uniq], dtype=object
    )
    base_npz = np.load(V2_BASELINE, allow_pickle=True)
    base_ids = np.asarray(base_npz["episode_id"], dtype=object)
    base_scores = np.asarray(base_npz["ensemble_score"], dtype=np.float64)
    base_labels = np.asarray(base_npz["labels"], dtype=bool)
    base_groups = np.asarray(base_npz["group_id"], dtype=object)
    base_map = {str(e): float(s) for e, s in zip(base_ids, base_scores)}

    variants: dict[str, Any] = {}
    for arm in ARMS:
        for tag in ("ho", "full"):
            per_seed = []
            from .train_eval import score_rows

            for seed in TRAIN_SEEDS:
                logits = score_rows(arm, seed, tag, "dev")
                ep = np.full(uniq.shape[0], -np.inf)
                inv = np.searchsorted(uniq, episodes)
                np.maximum.at(ep, inv, logits.astype(np.float64))
                per_seed.append(ep)
            score = np.mean(np.stack(per_seed), axis=0)
            thr = fpr_threshold(ep_label, score)
            cm = confusions(ep_label, score, thr)
            variants[f"{arm}:{tag}"] = {
                "episodes": int(uniq.shape[0]),
                "positives": int(ep_label.sum()),
                **cm,
                "fnr": cm["fn"] / max(cm["tp"] + cm["fn"], 1),
                "fpr": cm["fp"] / max(cm["fp"] + cm["tn"], 1),
                "auc": auc(ep_label, score),
                "threshold": thr,
                "score": score.tolist(),
                "labels": ep_label.tolist(),
                "episode_id": uniq.tolist(),
                "per_seed_auc": [auc(ep_label, s) for s in per_seed],
            }
    # baseline replay on the same subset (sanity: the accepted numbers must reappear on
    # the full universe, which the covered subset does not claim)
    base_vec = np.asarray([base_map[str(e)] for e in base_ids], dtype=np.float64)
    thr_b = fpr_threshold(base_labels, base_vec)
    cm_b = confusions(base_labels, base_vec, thr_b)
    baseline_full = {
        "episodes": int(base_labels.size),
        **cm_b,
        "fnr": cm_b["fn"] / max(cm_b["tp"] + cm_b["fn"], 1),
        "fpr": cm_b["fp"] / max(cm_b["fp"] + cm_b["tn"], 1),
        "auc": auc(base_labels, base_vec),
        "threshold": thr_b,
    }
    covered = set(uniq.tolist())
    mask_cov = np.asarray([str(e) in covered for e in base_ids])
    thr_bc = fpr_threshold(base_labels[mask_cov], base_vec[mask_cov])
    cm_bc = confusions(base_labels[mask_cov], base_vec[mask_cov], thr_bc)
    baseline_covered = {
        "episodes": int(mask_cov.sum()),
        **cm_bc,
        "fnr": cm_bc["fn"] / max(cm_bc["tp"] + cm_bc["fn"], 1),
        "fpr": cm_bc["fp"] / max(cm_bc["fp"] + cm_bc["tn"], 1),
        "auc": auc(base_labels[mask_cov], base_vec[mask_cov]),
        "threshold": thr_bc,
    }

    # paired bootstrap of F:full against C:full and against the accepted baseline, on the
    # covered episodes only (fixed thresholds, whole groups resampled)
    bootstrap = {}
    cf = variants.get("C:full")
    ff = variants.get("F:full")
    if cf and ff:
        for other in ("C:full", "F:full", "E:full"):
            pass
        # F vs C
        y = ep_label
        fscore = np.asarray(ff["score"])
        cscore = np.asarray(cf["score"])
        bootstrap["F_vs_C"] = _bootstrap(
            y, cscore, fscore, groups, float(cf["threshold"]), float(ff["threshold"])
        )
        escore = np.asarray(variants["E:full"]["score"])
        bootstrap["E_vs_C"] = _bootstrap(
            y, cscore, escore, groups, float(cf["threshold"]), float(variants["E:full"]["threshold"])
        )
        # F vs accepted baseline on the covered episodes
        base_cov = np.asarray([base_map[str(e)] for e in uniq], dtype=np.float64)
        bootstrap["F_vs_accepted_base"] = _bootstrap(
            y, base_cov, fscore, groups, float(baseline_covered["threshold"]), float(ff["threshold"])
        )

    # coverage-limited full-universe proxy with an explicit label-free stitching
    fallback = {}
    if ff is not None:
        covered_mask = mask_cov
        new_mean = float(np.mean(np.asarray(ff["score"])))
        new_std = float(np.std(np.asarray(ff["score"])) + 1e-9)
        old_missing = base_vec[~covered_mask]
        old_mean = float(np.mean(old_missing))
        old_std = float(np.std(old_missing) + 1e-9)
        stitched = np.empty(base_vec.size, dtype=np.float64)
        score_map = {str(e): s for e, s in zip(uniq, np.asarray(ff["score"]))}
        for i, e in enumerate(base_ids):
            if str(e) in score_map:
                stitched[i] = (score_map[str(e)] - new_mean) / new_std
            else:
                stitched[i] = (base_vec[i] - old_mean) / old_std
        thr_s = fpr_threshold(base_labels, stitched)
        cm_s = confusions(base_labels, stitched, thr_s)
        fallback = {
            "variant": "F:full + accepted-base fallback on nothink/uncovered episodes",
            "episodes": int(base_labels.size),
            **cm_s,
            "fnr": cm_s["fn"] / max(cm_s["tp"] + cm_s["fn"], 1),
            "fpr": cm_s["fp"] / max(cm_s["fp"] + cm_s["tn"], 1),
            "auc": auc(base_labels, stitched),
            "threshold": thr_s,
            "stitching": "label-free affine z-scoring per covered/missing part; NOT a candidate",
            "covered": int(covered_mask.sum()),
            "missing": int((~covered_mask).sum()),
        }
    return {
        "universe": "captured common-DEV subset (cot/ctx40k/agentlab)",
        "variants": variants,
        "baseline_full_2028": baseline_full,
        "baseline_covered_subset": baseline_covered,
        "bootstrap": bootstrap,
        "fallback_full_universe": fallback,
        "accepted_reference": BASELINE,
    }


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "eval").mkdir(parents=True, exist_ok=True)
    preflight = {}
    for split in ("train", "dev"):
        preflight[split] = build(split)
    atomic_write_json(ART / "manifest.json", {"build": preflight, "arms": list(ARMS)})
    print("built datasets", json.dumps(preflight))
    fitted = fit_all()
    print("fitted", len(fitted["runs"]), "runs")
    selection = heldout_selection()
    print("heldout selection", json.dumps(selection, indent=2))
    dev = dev_evaluation()
    report = {
        "heldout_selection": selection,
        "dev": dev,
        "runs": fitted["runs"],
        "preflight": preflight,
    }
    atomic_write_json(ART / "eval" / "report.json", report)
    slim = json.loads(json.dumps(report))
    for entry in slim["dev"]["variants"].values():
        entry.pop("score", None)
    atomic_write_json(ART / "eval" / "report_slim.json", slim)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("fn", "fp", "auc", "fnr", "fpr")}
                      for k, v in dev["variants"].items()}, indent=2))
    print("BASELINE covered", json.dumps({k: v for k, v in dev["baseline_covered_subset"].items()}))
    print("BASELINE full", json.dumps({k: v for k, v in dev["baseline_full_2028"].items()}))
    print("fallback", json.dumps(dev["fallback_full_universe"]))


if __name__ == "__main__":
    main()
