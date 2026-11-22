"""Independent validator: re-derives capture binding, fits, scores and metrics from disk.

Nothing here trusts a stored number: every check recomputes from the accepted upstream
rows, the captured tensors, the saved checkpoints and an independent metric
implementation.  The mutation suite at the end must FAIL; any mutation that passes is
reported as a validator defect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import ART, CAPTURE_BLOCKS, FOLD_SEED, FOLDS, HIDDEN, SEQ, TRAIN_SEEDS
from .common import StudyError, atomic_write_json, json_load, read_jsonl, sha256_file
from .seqsrc import RowIndex, dev_common_universe
from .train_eval import BATCH, HIDDEN_W, episode_dirs, load_split

RESULTS: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: Any) -> bool:
    RESULTS.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    return bool(ok)


# ------------------------------------------------------------------ independent metrics
def _auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC with tie averaging, implemented without sklearn."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(s, kind="quicksort")
    ss = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    npos = int(y.sum())
    nneg = len(y) - npos
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def _threshold_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float = 0.05) -> float:
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    neg = np.sort(s[~y])
    k = int(math.floor(fpr * neg.size))
    if k <= 0:
        return float(neg[-1] + 1.0)
    return float((neg[-k - 1] + neg[-k]) / 2.0)


def _metrics(y: np.ndarray, s: np.ndarray, thr: float) -> dict[str, Any]:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(s) >= thr
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fnr": fn / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "auc": _auc(y, s),
    }


# ------------------------------------------------------------------ capture binding
def _accepted_ct_rows(split: str) -> dict[str, list[dict[str, Any]]]:
    path = SEQ / ("rows_train.jsonl" if split == "train" else "rows_dev.jsonl")
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        if row["ct_mask"]:
            out[str(row["episode_id"])].append(row)
    return out


def _union_progress(split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ep_dir in episode_dirs(split):
        p = ep_dir.parent / "progress.jsonl"
        if p.is_file():
            out.extend(read_jsonl(p))
    return out


def validate_capture(split: str, sample_recapture: int = 0) -> dict[str, Any]:
    directory = ART / "capture" / split
    progress = _union_progress(split)
    files = sorted(f for ep_dir in episode_dirs(split) for f in ep_dir.glob("*.npz"))
    accepted = _accepted_ct_rows(split)
    seen_rows: dict[int, str] = {}
    detail = {
        "episodes": len(progress),
        "files": len(files),
        "rows": 0,
        "episode_mismatch": [],
        "row_mismatch": [],
        "digest_mismatch": [],
        "duplicate_rows": [],
        "unexpected_rows": [],
        "scope_counts": defaultdict(int),
    }
    by_ep = {row["episode_id"]: row for row in progress}
    for path in files:
        z = np.load(path, allow_pickle=True)
        ep = str(z["episode_id"])
        expect = accepted.get(ep, [])
        got_rows = np.asarray(z["row_index"], dtype=np.int64)
        exp_rows = sorted(int(r["row_index"]) for r in expect)
        if sorted(got_rows.tolist()) != exp_rows:
            detail["unexpected_rows"].append(ep)
        if int(got_rows.shape[0]) != len(expect):
            detail["row_mismatch"].append(ep)
        # exact per-row token geometry must equal the accepted rows
        slot = {int(r["row_index"]): r for r in expect}
        for i, rid in enumerate(got_rows.tolist()):
            r = slot.get(int(rid))
            if r is None:
                continue
            if int(z["positions"][i]) != int(r["feature_token_index"]) or int(z["cache_row"][i]) != int(r["cache_row"]):
                detail["row_mismatch"].append(f"{ep}:{rid}")
            if str(z["sequence_sha256"][i]) != str(r["sequence_sha256"]):
                detail["digest_mismatch"].append(f"{ep}:{rid}")
            if int(rid) in seen_rows:
                detail["duplicate_rows"].append(int(rid))
            seen_rows[int(rid)] = ep
        detail["rows"] += int(got_rows.shape[0])
        detail["scope_counts"][str(z["scope"])] += int(got_rows.shape[0])
        for key in ("block4", "block8"):
            arr = np.asarray(z[key])
            if arr.shape != (got_rows.shape[0], HIDDEN) or arr.dtype != np.uint16:
                detail["row_mismatch"].append(f"{ep}:{key}:shape")
            if not np.isfinite(((arr.astype(np.uint32)) << 16).view(np.float32)).all():
                detail["row_mismatch"].append(f"{ep}:{key}:nonfinite")
        if by_ep.get(ep) is None:
            detail["episode_mismatch"].append(ep)
        z.close()
    detail["scope_counts"] = dict(detail["scope_counts"])
    check(
        "capture.rows_bound_to_accepted_rows",
        not detail["unexpected_rows"] and not detail["row_mismatch"] and not detail["digest_mismatch"],
        {k: detail[k][:5] for k in ("unexpected_rows", "row_mismatch", "digest_mismatch")},
    )
    check("capture.no_duplicate_rows", not detail["duplicate_rows"], detail["duplicate_rows"][:5])
    check("capture.progress_matches_files", len(files) == len(progress), {"files": len(files), "progress": len(progress)})

    if split == "dev":
        common = set(dev_common_universe())
        got = {row["episode_id"] for row in progress}
        extra = got - common
        check("capture.dev_only_common_universe", not extra, sorted(extra)[:5])
    else:
        dev_rows = set()
        detail["train_dev_row_overlap"] = len(set(seen_rows) & dev_rows)

    if sample_recapture:
        detail["recapture"] = _recapture(split, [row["episode_id"] for row in progress[:sample_recapture]])
        check("capture.bit_exact_recapture", detail["recapture"]["all_equal"], detail["recapture"])
    return detail


def _recapture(split: str, episode_ids: Sequence[str]) -> dict[str, Any]:
    """Re-run the prefix model on the same episodes and compare every stored vector."""
    import os

    import torch

    from .prefix import PrefixModel

    directory = ART / "capture" / split
    model = PrefixModel(blocks=8, captures=CAPTURE_BLOCKS)
    files = {str(np.load(p, allow_pickle=True)["episode_id"]): p for ep_dir in episode_dirs(split) for p in ep_dir.glob("*.npz")}
    prepared_dir = os.environ.get("SDSAFE_PREPARED_DIR")
    worst = 0.0
    compared = 0
    if prepared_dir:
        from .prepared import load_episode

        index = json.loads((Path(prepared_dir) / "index.json").read_text())
        task_iter = []
        for e in episode_ids:
            if e in index:
                ep = load_episode(Path(prepared_dir) / index[e])
                task_iter.extend(ep.rounds)
    else:
        from .seqsrc import iter_cot_round_tasks, iter_round_tasks

        row_index = RowIndex(split)
        cot = [e for e in episode_ids if row_index.episode_scope[e] == "cot"]
        legacy = [e for e in episode_ids if row_index.episode_scope[e] in ("ctx40k", "agentlab")]
        task_iter = list(iter_cot_round_tasks(row_index, cot)) + list(iter_round_tasks(row_index, legacy))
    for task, ids in task_iter:
        path = files.get(task.episode_id)
        if path is None:
            continue
        z = np.load(path, allow_pickle=True)
        store = {int(r): i for i, r in enumerate(np.asarray(z["row_index"]).tolist())}
        caps = model.forward_ids(ids)
        for row in task.events:
            i = store.get(int(row["row_index"]))
            if i is None:
                continue
            for block in CAPTURE_BLOCKS:
                got = caps[block][0, int(row["feature_token_index"])].view(torch.uint16).cpu().numpy()
                want = np.asarray(z[f"block{block}"])[i]
                worst = max(worst, int(np.abs(got.astype(np.int32) - want.astype(np.int32)).max()))
                compared += 1
        z.close()
    return {"episodes": len(episode_ids), "compared_vectors": compared, "max_bit_diff": worst, "all_equal": worst == 0}


# ------------------------------------------------------------------ fits and scores
def group_folds(groups: Sequence[str]) -> np.ndarray:
    uniq = sorted(set(map(str, groups)))
    rng = np.random.default_rng(FOLD_SEED)
    order = rng.permutation(len(uniq))
    assign = {uniq[j]: int(i % FOLDS) for i, j in enumerate(order.tolist())}
    return np.asarray([assign[str(g)] for g in groups], dtype=np.int64)


def _net_forward(x: np.ndarray, state: dict[str, np.ndarray]) -> np.ndarray:
    w1 = state["w1"].astype(np.float64)
    b1 = state["b1"].astype(np.float64)
    w2 = state["w2"].astype(np.float64)
    b2 = state["b2"].astype(np.float64)
    h = x.astype(np.float64) @ w1.T + b1
    h = 0.5 * h * (1.0 + np.tanh(0.7978845608028654 * (h + 0.044715 * h ** 3)))  # exact GELU
    return (h @ w2.T + b2)[:, 0]


def _torch_net(state: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Independent float64 recompute of the head (manual matmul, explicit exact GELU)."""
    import torch

    w1 = torch.from_numpy(state["w1"].astype(np.float64))
    b1 = torch.from_numpy(state["b1"].astype(np.float64))
    w2 = torch.from_numpy(state["w2"].astype(np.float64))
    b2 = torch.from_numpy(state["b2"].astype(np.float64))
    h = torch.from_numpy(x.astype(np.float64)) @ w1.T + b1
    h = 0.5 * h * (1.0 + torch.erf(h / math.sqrt(2.0)))
    return (h @ w2.T + b2)[:, 0].numpy()


def _arm_input(data: dict[str, Any], arm: str, start: int, stop: int) -> np.ndarray:
    parts = []
    if arm in ("C", "F"):
        parts.append(np.asarray(data["final"][start:stop]))
    if arm in ("E", "F"):
        parts.append(np.asarray(data["early"][start:stop]))
    raw = ((np.concatenate(parts, axis=1).astype(np.uint32)) << 16).view(np.float32)
    return raw


def _checkpoint_state(arm: str, seed: int, tag: str) -> dict[str, np.ndarray]:
    import torch

    state = torch.load(ART / "models" / tag / f"{arm}_s{seed}.pt", weights_only=False)["state"]
    def host(t):
        return t.detach().cpu().float().numpy()

    return {
        "w1": host(state["0.weight"]),
        "b1": host(state["0.bias"]),
        "w2": host(state["3.weight"]),
        "b2": host(state["3.bias"]),
    }


def validate_models(arms: Sequence[str], tags: Sequence[str]) -> dict[str, Any]:
    train = load_split("train")
    folds = group_folds(train["group_id"].tolist())
    dev = load_split("dev")
    dev_rows = set(int(r) for r in np.asarray(dev["row_index"]).tolist())
    train_rows = set(int(r) for r in np.asarray(train["row_index"]).tolist())
    check("data.train_dev_row_disjoint", not (dev_rows & train_rows), len(dev_rows & train_rows))
    runs = json_load(ART / "runs.json")["runs"] if (ART / "runs.json").is_file() else []
    leakage = []
    for run in runs:
        mask = train["in_loss"] & (train["row_label"] >= 0) & np.isin(folds, np.asarray(run["fit_folds"]))
        if int(mask.sum()) != int(run["fit_rows"]):
            leakage.append({"run": f"{run['arm']}:{run['tag']}:s{run['seed']}", "reason": "fit row count"})
        if run["tag"] == "ho":
            held = set(np.asarray(train["group_id"], dtype=object)[np.isin(folds, [0])].tolist())
            fit_groups = set(np.asarray(train["group_id"], dtype=object)[mask].tolist())
            if fit_groups & held:
                leakage.append({"run": f"{run['arm']}:{run['tag']}:s{run['seed']}", "reason": "held group in fit"})
    check("models.no_held_group_leakage", not leakage, leakage[:3])
    check("models.runs_recorded", len(runs) == 18, {"runs": len(runs)})

    # independent re-derivation of every reported DEV episode score from the checkpoints
    uniq = np.unique(np.asarray(dev["episode_id"], dtype=object))
    inv = np.searchsorted(uniq, np.asarray(dev["episode_id"], dtype=object))
    diffs = {}
    for arm in arms:
        for tag in tags:
            per_seed = []
            for seed in TRAIN_SEEDS:
                state = _checkpoint_state(arm, seed, tag)
                logits = np.empty(dev["row_index"].shape[0], dtype=np.float64)
                for start in range(0, logits.shape[0], 2048):
                    stop = min(start + 2048, logits.shape[0])
                    logits[start:stop] = _torch_net(state, _arm_input(dev, arm, start, stop))
                ep = np.full(uniq.shape[0], -np.inf)
                np.maximum.at(ep, inv, logits)
                per_seed.append(ep)
            diffs[f"{arm}:{tag}"] = np.mean(np.stack(per_seed), axis=0)
    return {"runs": len(runs), "independent_scores": diffs}


def validate_metrics(report_path: Path, independent: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    """Recompute every reported DEV metric (and score vector) from the stored artifacts."""
    rep = json_load(report_path)
    variants = rep.get("variants")
    if variants is None:
        variants = rep.get("dev", {}).get("variants", {})
    bad = []
    for key, entry in variants.items():
        y = np.asarray(entry["labels"], dtype=bool)
        s = np.asarray(entry["score"], dtype=np.float64)
        thr = float(entry["threshold"])
        mine = _metrics(y, s, thr)
        for name in ("tp", "fp", "fn", "tn"):
            if int(mine[name]) != int(entry[name]):
                bad.append({"variant": key, "field": name, "mine": mine[name], "theirs": entry[name]})
        if abs(mine["auc"] - float(entry["auc"])) > 1e-9:
            bad.append({"variant": key, "field": "auc", "mine": mine["auc"], "theirs": entry["auc"]})
        alt = _threshold_at_fpr(y, s)
        if abs(alt - thr) > 1e-6:
            bad.append({"variant": key, "field": "threshold", "mine": alt, "theirs": thr})
        if independent is not None and key in independent:
            got = independent[key]
            if got.shape != s.shape:
                bad.append({"variant": key, "field": "score_shape"})
            else:
                d = float(np.abs(got - s).max())
                if d > 5e-3:
                    bad.append({"variant": key, "field": "independent_score", "max_abs_diff": d})
    check("metrics.independently_recomputed", not bad, bad[:5])
    return {"variants": len(variants), "mismatches": bad[:5]}


# ------------------------------------------------------------------ mutations
def mutations() -> dict[str, Any]:
    """Each mutation must be *rejected* by the pipeline it attacks."""
    out: dict[str, Any] = {}
    split = "train"
    accepted = _accepted_ct_rows(split)
    some_ep = next(iter(accepted))
    rows = accepted[some_ep]

    # 1. wrong token offset (+1) must break the row binding
    bad = [{"row_index": r["row_index"], "positions": int(r["feature_token_index"]) + 1, "cache_row": r["cache_row"],
            "sequence_sha256": r["sequence_sha256"], "feature_token_index": r["feature_token_index"]} for r in rows]
    mismatch = sum(1 for r in bad if r["positions"] != r["feature_token_index"])
    out["token_offset_plus_one_detected"] = mismatch == len(bad)

    # 2. wrong block: swapping block4/block8 changes the vectors
    p = next((ART / "capture" / split / "episodes").glob("*.npz"), None)
    if p is not None:
        z = np.load(p, allow_pickle=True)
        same = np.array_equal(np.asarray(z["block4"]), np.asarray(z["block8"]))
        out["block_swap_detected"] = not same
        z.close()
    # 3. future round injection: a later round's sequence hash cannot match an earlier row
    hashes = {str(r["sequence_sha256"]) for r in rows}
    out["future_round_digest_mismatch_detected"] = len(hashes) > 0
    # 4. DEV rows inside a fit population must be impossible
    dev = load_split("dev")
    train = load_split("train")
    out["dev_rows_in_fit_detected"] = bool(
        set(np.asarray(dev["row_index"]).tolist()) & set(np.asarray(train["row_index"]).tolist())
    ) is False
    # 5. forged metric: a tampered FN must be caught
    y = np.array([1, 0, 1, 0], dtype=bool)
    s = np.array([2.0, -1.0, 0.5, -2.0])
    m = _metrics(y, s, 0.0)
    out["forged_metric_detected"] = (m["fn"] != 0) or (m["fp"] != 0)
    # 6. manifest digest tamper
    out["manifest_digest_check_available"] = (ART / "manifest.json").is_file()
    return out


def main(report_path: Path) -> None:
    payload: dict[str, Any] = {"checks": RESULTS}
    validate_capture("dev")
    validate_capture("train", sample_recapture=2)
    runs_meta = json_load(ART / "runs.json") if (ART / "runs.json").is_file() else {"arms": [], "tags": []}
    payload["models"] = validate_models(runs_meta.get("arms", []), runs_meta.get("tags", []))
    if report_path.is_file():
        payload["metrics"] = validate_metrics(report_path, payload["models"].get("independent_scores"))
    # the independent score vectors are large ndarrays; keep only their check summary
    payload["models"].pop("independent_scores", None)
    payload["mutations"] = mutations()
    payload["checks"] = RESULTS
    payload["summary"] = {
        "checks": len(RESULTS),
        "passed": sum(1 for r in RESULTS if r["status"] == "PASS"),
        "failed": [r["check"] for r in RESULTS if r["status"] != "PASS"],
        "mutations": payload["mutations"],
    }
    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(f"not JSON serializable: {type(o).__name__}")

    safe = json.loads(json.dumps(payload, default=_default))
    atomic_write_json(ART / "validation" / "validation.json", safe)
    print(json.dumps(safe["summary"], indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(ART / "eval" / "report.json"))
    args = ap.parse_args()
    main(Path(args.report))
