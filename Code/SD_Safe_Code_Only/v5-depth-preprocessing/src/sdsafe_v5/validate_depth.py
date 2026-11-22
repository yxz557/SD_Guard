"""Independent validator for the v5 depth stage.

Deliberately does **not** reuse the pipeline's metric functions or scoring code:

* bf16 decode uses ``sdsafe_v5.loader.bf16_bits_to_f32`` (pinned by known bit patterns);
* metrics and the FPR threshold are re-implemented here in float64 numpy, and every formal
  checkpoint is re-scored by an independent float64 numpy forward pass from the raw bf16
  matrices (the pipeline scores in torch float32);
* a **non-vacuous** ``forged_metric`` mutation edits a stored metric value in a shadow copy
  of the real report; the comparison guard must fail on it (the v4 validator's dummy never
  tampered a stored number and is not used here).

Every mutation reports ``detected: bool``.  The run exits non-zero if any formal check fails
or any required mutation is not detected.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import ART, HIDDEN, TRAIN_SEEDS, V2_DEV_ARRAYS
from .common import atomic_write_json, json_load, sha256_file
from .loader import bf16_bits_to_f32

DS = Path(os.environ.get("SDSAFE_DEPTH_DATASET", "/dev/shm/sdsafe-depth-v5/dataset"))
EVAL = ART / "eval"
OUT = ART / "validation"
PREP = Path("/root/sdsafe-compute/v5-depth/prepared")


# ---------------------------------------------------------------- independent metrics
def _auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(np.int64)
    s = np.asarray(s, dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    sv = allv[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rp = ranks[:pos.size].sum()
    return float((rp - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def _threshold_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float = 0.05) -> float:
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    neg = np.sort(s[~y])[::-1]
    k = int(np.floor(fpr * neg.size))
    if k <= 0:
        return float(neg[0] + 1e-9)
    if neg[k - 1] == neg[k]:
        return float(np.nextafter(neg[k], np.inf))
    return float((neg[k - 1] + neg[k]) / 2.0)


def _conf(y, s, thr):
    y = np.asarray(y).astype(bool)
    p = np.asarray(s, dtype=np.float64) >= thr
    return {"tp": int((p & y).sum()), "fp": int((p & ~y).sum()),
            "fn": int((~p & y).sum()), "tn": int((~p & ~y).sum())}


def _metrics(y, s, thr):
    c = _conf(y, s, thr)
    return {**c, "fnr": c["fn"] / max(c["tp"] + c["fn"], 1),
            "fpr": c["fp"] / max(c["fp"] + c["tn"], 1), "auc": _auc(y, s)}


# ---------------------------------------------------------------- data access
def _meta(split: str) -> dict[str, np.ndarray]:
    z = np.load(DS / split / "meta.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


BLOCKS = {"C": (("final", "final"),), "Craw": (("final", "final"),),
          "H": (("final", "final"), ("block16", "b16")),
          "I": (("final", "final"), ("block24", "b24")),
          "J": (("final", "final"), ("block16", "b16"), ("block24", "b24")),
          "K": (("final", "final"), ("block16", "b16"), ("block24", "b24"))}


def _standardized(split: str, arm: str, stats: dict[str, Any], rows: np.ndarray | None = None) -> np.ndarray:
    specs = BLOCKS[arm]
    meta = _meta(split)
    n = meta["row_index"].shape[0]
    idx = np.arange(n) if rows is None else np.asarray(rows, dtype=np.int64)
    out = np.empty((idx.size, len(specs) * HIDDEN), dtype=np.float64)
    for j, (fname, skey) in enumerate(specs):
        mat = np.load(DS / split / f"{fname}_bf16.npy", mmap_mode="r")
        raw = bf16_bits_to_f32(np.asarray(mat[idx], dtype=np.uint16)).astype(np.float64)
        if arm != "Craw":
            st = stats[skey]
            raw = (raw - np.asarray(st["mu"], dtype=np.float64)) / np.asarray(st["sigma"], dtype=np.float64)
        out[:, j * HIDDEN:(j + 1) * HIDDEN] = raw
    return out


def _erf(x: np.ndarray) -> np.ndarray:
    import torch

    return torch.special.erf(torch.from_numpy(x)).numpy()


def _numpy_scores(x: np.ndarray, state: dict[str, np.ndarray], arch: str = "mlp") -> np.ndarray:
    if arch == "gated_lowrank":
        n = x.shape[0]
        parts = x.reshape(n, 3, HIDDEN)
        g = state["gates"]
        sp = np.log1p(np.exp(-np.abs(g))) + np.maximum(g, 0.0)
        z = None
        for i in range(3):
            zi = parts[:, i, :] @ state["W"][i] + state["b"][i]
            z = sp[i] * zi if z is None else z + sp[i] * zi
        h = 0.5 * z * (1.0 + _erf(z / math.sqrt(2.0)))
        return (h @ state["w2"].reshape(-1, 1) + state["b2"]).reshape(-1)
    h = x @ state["w1"].T + state["b1"]
    # exact GELU (torch nn.GELU default = erf, not the tanh approximation)
    h = 0.5 * h * (1.0 + _erf(h / math.sqrt(2.0)))
    return (h @ state["w2"].T + state["b2"]).reshape(-1)


def _state(arm: str, tag: str, seed: int) -> tuple[dict[str, np.ndarray], str]:
    import torch

    ck = torch.load(ART / "models_depth" / tag / f"{arm}_s{seed}.pt", weights_only=False)
    if ck.get("arch") == "gated_lowrank":
        st = {k: v.double().cpu().numpy() for k, v in ck["params"].items()}
        st["w2"] = ck["w2"]["weight"].double().cpu().numpy()
        st["b2"] = ck["w2"]["bias"].double().cpu().numpy()
        return st, "gated_lowrank"
    st = ck["state"]
    return {"w1": st["0.weight"].double().cpu().numpy(), "b1": st["0.bias"].double().cpu().numpy(),
            "w2": st["3.weight"].double().cpu().numpy(), "b2": st["3.bias"].double().cpu().numpy()}, "mlp"


def _episode_max(s: np.ndarray, inv: np.ndarray, size: int) -> np.ndarray:
    out = np.full(size, -np.inf, dtype=np.float64)
    np.maximum.at(out, inv, np.asarray(s, dtype=np.float64))
    return out


def check_formal_models(report: dict[str, Any]) -> dict[str, Any]:
    stats = json_load(ART / "dataset" / "train_stats.json")
    dev = _meta("dev")
    uniq, inv = np.unique(np.asarray(dev["episode_id"], dtype=object), return_inverse=True)
    y = np.zeros(uniq.size, dtype=bool)
    y[inv] = np.asarray(dev["episode_label"], dtype=bool)
    folds = np.load(ART / "dataset" / "folds.npy")
    tr = _meta("train")
    ho_rows = np.flatnonzero(folds == 0)
    ho_uniq, ho_inv = np.unique(np.asarray(tr["episode_id"], dtype=object)[ho_rows], return_inverse=True)
    ho_lab = np.zeros(ho_uniq.size, dtype=bool)
    ho_lab[ho_inv] = np.asarray(tr["episode_label"], dtype=bool)[ho_rows]
    checks = []
    arms = [a for a in ("C", "Craw", "H", "I", "J", "K") if f"{a}:full" in report["dev_variants"]]
    for arm in arms:
        xd = _standardized("dev", arm, stats["blocks"])
        xt = _standardized("train", arm, stats["blocks"], rows=ho_rows)
        for tag in ("ho", "full"):
            key = f"{arm}:{tag}"
            if key not in report["dev_variants"]:
                continue
            stored = report["dev_variants"][key]
            per = []
            per_ho = []
            for seed in TRAIN_SEEDS:
                st, arch = _state(arm, tag, seed)
                per.append(_episode_max(_numpy_scores(xd, st, arch), inv, uniq.size))
                per_ho.append(_episode_max(_numpy_scores(xt, st, arch), ho_inv, ho_uniq.size))
            ens = np.mean(np.stack(per), axis=0)
            thr = _threshold_at_fpr(y, ens)
            m = _metrics(y, ens, thr)
            checks.append({
                "arm": arm, "tag": tag, "kind": "dev",
                "stored": {k: stored.get(k) for k in ("tp", "fp", "fn", "tn", "auc", "threshold")},
                "independent": m,
                # float32 torch scoring vs float64 numpy can move one boundary episode;
                # a delta of <=1 per cell is accepted and reported, >1 is a failure.
                "confusion_delta": {k: int(int(stored.get(k, -10**9)) - m[k]) for k in ("tp", "fp", "fn", "tn")},
                "confusion_match": all(abs(int(stored.get(k, -10**9)) - m[k]) <= 1 for k in ("tp", "fp", "fn", "tn")),
                "auc_match": abs(float(stored.get("auc", -1)) - m["auc"]) < 1e-7,
                "threshold_match": abs(float(stored.get("threshold", -1)) - thr) < 1e-5,
            })
            ens_h = np.mean(np.stack(per_ho), axis=0)
            thr_h = _threshold_at_fpr(ho_lab, ens_h)
            mh = _metrics(ho_lab, ens_h, thr_h)
            sh = stored.get("heldout_fold0", {})
            checks.append({
                "arm": arm, "tag": tag, "kind": "heldout_fold0",
                "stored": {k: sh.get(k) for k in ("tp", "fp", "fn", "tn", "auc")},
                "independent": mh,
                "confusion_delta": {k: int(int(sh.get(k, -10**9)) - mh[k]) for k in ("tp", "fp", "fn", "tn")},
                "confusion_match": all(abs(int(sh.get(k, -10**9)) - mh[k]) <= 1 for k in ("tp", "fp", "fn", "tn")),
                "auc_match": abs(float(sh.get("auc", -1)) - mh["auc"]) < 1e-7,
            })
        del xd, xt
    return {"checks": checks,
            "passed": all(c["confusion_match"] and c["auc_match"] for c in checks),
            "n_checks": len(checks)}


# ---------------------------------------------------------------- capture/source checks
def check_capture(shards_per_split: int = 60, recapture: int = 2) -> dict[str, Any]:
    from .prepared import load_episode

    out: dict[str, Any] = {}
    for split, expect_eps in (("train", 8229), ("dev", 2028)):
        d = ART / "capture" / split / "episodes"
        files = sorted(d.glob("*.npz"))
        idx = json.loads((PREP / split / "index.json").read_text())
        progress = [json.loads(l) for l in (ART / "capture" / split / "progress.jsonl").read_text().splitlines() if l.strip()]
        committed = {r["episode_id"] for r in progress if r.get("npz_sha256")}
        problems: list[str] = []
        sample = files[:: max(1, len(files) // shards_per_split)][:shards_per_split]
        rows_full = 0
        for f in sample:
            z = np.load(f, allow_pickle=True)
            n = int(z["row_index"].shape[0])
            if not np.array_equal(np.asarray(z["positions"]), np.asarray(z["token_end"])):
                problems.append(f"{f.name}: position != token_end")
            if np.asarray(z["block16"]).shape != (n, HIDDEN) or np.asarray(z["block24"]).shape != (n, HIDDEN):
                problems.append(f"{f.name}: block shape")
            eid = str(z["episode_id"])
            if eid not in idx:
                problems.append(f"{f.name}: episode not in prepared index")
            else:
                ep = load_episode(PREP / split / idx[eid])
                by_round = {int(r.round_id): r for r, _ in ep.rounds}
                for rid in sorted(set(np.asarray(z["round_id"]).tolist())):
                    want = set(np.asarray(z["sequence_sha256"])[np.asarray(z["round_id"]) == rid].tolist())
                    if want != {by_round[int(rid)].sequence_sha256}:
                        problems.append(f"{f.name}: round {rid} digest mismatch")
            z.close()
        full_uniqueness = False
        if len(files) == expect_eps:
            for f in files:
                z = np.load(f, allow_pickle=True)
                ids = np.asarray(z["row_index"]).tolist()
                rows_full += len(ids)
                if len(set(ids)) != len(ids):
                    problems.append(f"{f.name}: duplicate rows inside shard")
                z.close()
            full_uniqueness = True
        out[split] = {"shards": len(files), "expected_episodes": expect_eps,
                      "committed_progress": len(committed), "sample_checked": len(sample),
                      "sample_ok": not problems, "problems": problems[:20],
                      "full_row_uniqueness_checked": full_uniqueness,
                      "rows_full": rows_full if full_uniqueness else None}
    # deterministic recapture of committed TRAIN episodes, bitwise
    if recapture:
        import torch
        from . import CAPTURE_BLOCKS
        from .prefix import PrefixModel

        picked = sorted((ART / "capture" / "train" / "episodes").glob("*.npz"))[:recapture]
        idx = json.loads((PREP / "train" / "index.json").read_text())
        if picked:
            prefix = PrefixModel(blocks=24, captures=tuple(sorted(CAPTURE_BLOCKS)))
            diffs = []
            for f in picked:
                z = np.load(f, allow_pickle=True)
                ep = load_episode(PREP / "train" / idx[str(z["episode_id"])])
                for r, ids in ep.rounds:
                    cap = prefix.forward_positions(ids, r.positions)
                    mask = np.asarray(z["round_id"]) == r.round_id
                    for b in CAPTURE_BLOCKS:
                        a = cap[b].view(torch.uint16).cpu().numpy()
                        bb = np.asarray(z[f"block{b}"])[mask]
                        diffs.append({"episode": str(z["episode_id"]), "round": int(r.round_id),
                                      "block": int(b), "bitwise_equal": bool(np.array_equal(a, bb)),
                                      "max_abs_bitdelta": int(np.abs(a.astype(np.int32) - bb.astype(np.int32)).max())})
                z.close()
            out["recapture"] = diffs
            out["recapture_ok"] = all(x["bitwise_equal"] for x in diffs)
            prefix.close()
            del prefix
            torch.cuda.empty_cache()
    else:
        out["recapture_ok"] = None
    # ---- cross-check block 8 against the already-accepted v4 shards (same 4090)
    v4dir = Path("/root/sdsafe-compute/v4-gpu/artifacts/capture/train/episodes")
    if v4dir.is_dir():
        import hashlib
        import torch
        from .prepared import load_episode as _le
        from .prefix import PrefixModel as _PM

        idx = json.loads((PREP / "train" / "index.json").read_text())
        checks = []
        files = sorted(v4dir.glob("*.npz"))[:400:120]
        prefix = _PM(blocks=24, captures=(8, 16, 24))
        for f in files:
            z = np.load(f, allow_pickle=True)
            eid = str(z["episode_id"])
            v5f = ART / "capture" / "train" / "episodes" / (hashlib.sha256(eid.encode()).hexdigest()[:24] + ".npz")
            if not v5f.is_file() or eid not in idx:
                z.close()
                continue
            ep = _le(PREP / "train" / idx[eid])
            for r, ids in ep.rounds:
                if int(r.round_id) not in set(np.asarray(z["round_id"]).tolist()):
                    continue
                cap = prefix.forward_positions(ids, r.positions)
                a = cap[8].view(torch.uint16).cpu().numpy()
                b = np.asarray(z["block8"])[np.asarray(z["round_id"]) == r.round_id]
                checks.append({"episode": eid, "round": int(r.round_id),
                               "bitwise_equal": bool(np.array_equal(a, b)),
                               "max_abs_bitdelta": int(np.abs(a.astype(np.int32) - b.astype(np.int32)).max())})
            z.close()
        prefix.close()
        del prefix
        torch.cuda.empty_cache()
        out["v4_block8_crosscheck"] = checks
        out["v4_block8_ok"] = bool(checks) and all(c["bitwise_equal"] for c in checks)
    # ---- cross-check block16/24 against a complete 36-block forward (short episode)
    try:
        import torch
        from .prepared import load_episode as _le2
        from .prefix import PrefixModel as _PM2

        idx = json.loads((PREP / "train" / "index.json").read_text())
        v5files = sorted((ART / "capture" / "train" / "episodes").glob("*.npz"))
        picked = None
        for f in v5files:
            z = np.load(f, allow_pickle=True)
            if int(np.asarray(z["seq_len"]).max()) <= 1200:
                picked = (f, str(z["episode_id"]), {k: np.asarray(z[k]) for k in z.files})
                z.close()
                break
            z.close()
        if picked is not None:
            f, eid, shard = picked
            ep = _le2(PREP / "train" / idx[eid])
            full = _PM2(blocks=36, captures=(16, 24))
            checks = []
            for r, ids in ep.rounds:
                cap = full.forward_positions(ids, r.positions)
                mask = shard["round_id"] == r.round_id
                for b in (16, 24):
                    a = cap[b].view(torch.uint16).cpu().numpy()
                    bb = shard[f"block{b}"][mask]
                    checks.append({"episode": eid, "round": int(r.round_id), "block": b,
                                   "bitwise_equal": bool(np.array_equal(a, bb)),
                                   "max_abs_bitdelta": int(np.abs(a.astype(np.int32) - bb.astype(np.int32)).max())})
            full.close()
            del full
            torch.cuda.empty_cache()
            out["full36_crosscheck"] = checks
            out["full36_ok"] = bool(checks) and all(c["bitwise_equal"] for c in checks)
    except Exception as error:  # noqa: BLE001
        out["full36_crosscheck"] = f"{type(error).__name__}: {error}"
        out["full36_ok"] = False
    return out


# ---------------------------------------------------------------- guards
def guard_bf16() -> bool:
    bits = np.asarray([0x3F80, 0xBF80, 0x4000], dtype=np.uint16)
    good = bf16_bits_to_f32(bits)
    bad = bits.astype(np.float32)
    return bool(np.allclose(good, [1.0, -1.0, 2.0]) and not np.allclose(good, bad))


def guard_endpoint(z: dict[str, Any]) -> bool:
    return bool(np.array_equal(np.asarray(z["positions"]), np.asarray(z["token_end"])))


def guard_layer(z: dict[str, Any]) -> bool:
    a = bf16_bits_to_f32(np.asarray(z["block16"]))
    b = bf16_bits_to_f32(np.asarray(z["block24"]))
    return bool(np.abs(a - b).max() > 0.05)


def guard_row_order(z: dict[str, Any]) -> bool:
    r = np.asarray(z["row_index"])
    return bool(np.all(np.diff(r) > 0))


def guard_cohort(eids: list[str], allowed: set[str]) -> bool:
    return all(e in allowed for e in eids)


def guard_oof(fit_groups: set[str], held_groups: set[str]) -> bool:
    return len(fit_groups & held_groups) == 0


def guard_model_sha(prov: dict[str, Any], expected: dict[str, str]) -> bool:
    return all(prov["model_shards"].get(k) == v for k, v in expected.items())


def guard_manifest(manifest: dict[str, Any], root: Path) -> bool:
    for rel, rec in manifest.get("artifacts", {}).items():
        if "sha256" not in rec:
            continue
        p = root / rel
        if not p.is_file() or sha256_file(p) != rec["sha256"]:
            return False
    return True


def check_report_metrics(report: dict[str, Any], independent: dict[str, Any],
                         tol_int: int = 1, tol_auc: float = 1e-7) -> list[str]:
    """Compare values *stored in the report under test* to the independent recompute.

    The stored side is read from the report argument (not from the validator's own copy of
    the report), so tampering a stored metric in a shadow copy is actually detected.
    """
    bad = []
    for c in independent["checks"]:
        key = f"{c['arm']}:{c['tag']}"
        entry = report.get("dev_variants", {}).get(key, {})
        stored = entry if c["kind"] == "dev" else entry.get("heldout_fold0", {})
        ind = c["independent"]
        for k in ("tp", "fp", "fn", "tn"):
            if stored.get(k) is not None and abs(int(stored[k]) - int(ind[k])) > tol_int:
                bad.append(f"{c['arm']}:{c['tag']}:{c['kind']} {k} stored={stored[k]} independent={ind[k]}")
        if stored.get("auc") is not None and abs(float(stored["auc"]) - ind["auc"]) > tol_auc:
            bad.append(f"{c['arm']}:{c['tag']}:{c['kind']} auc stored={stored['auc']} independent={ind['auc']}")
    return bad


def mutations(report: dict[str, Any], independent: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    z = None
    for f in sorted((ART / "capture" / "train" / "episodes").glob("*.npz")):
        zz = np.load(f, allow_pickle=True)
        z = {k: zz[k] for k in zz.files}
        zz.close()
        break
    if z is None:
        raise RuntimeError("no capture shards for mutation tests")
    # 0. bf16 conversion
    out["bf16_conversion_detected"] = guard_bf16()
    # 1. wrong endpoint
    zb = dict(z)
    zb["positions"] = np.asarray(z["positions"]) + 1
    out["wrong_endpoint_detected"] = not guard_endpoint(zb)
    out["endpoint_guard_accepts_clean"] = guard_endpoint(z)
    # 2. wrong layer / swapped block
    zl = dict(z)
    zl["block24"] = z["block16"]
    out["wrong_layer_detected"] = not guard_layer(zl)
    out["layer_guard_accepts_clean"] = guard_layer(z)
    # 3. row order
    zr = dict(z)
    r = np.asarray(z["row_index"]).copy()
    if len(r) > 1:
        r[0], r[1] = r[1], r[0]
    zr["row_index"] = r
    out["row_order_detected"] = not guard_row_order(zr)
    out["row_order_guard_accepts_clean"] = guard_row_order(z)
    # 4. illegal cohort
    allowed_dev = set(np.asarray(np.load(V2_DEV_ARRAYS, allow_pickle=True)["episode_id"], dtype=object).tolist())
    allowed_train = set(json.loads((PREP / "train" / "index.json").read_text()))
    train_eps = [str(z["episode_id"])]
    out["illegal_cohort_detected"] = not guard_cohort(train_eps + ["forged::not-an-episode"], allowed_train)
    out["cohort_guard_accepts_clean"] = guard_cohort(train_eps, allowed_train)
    out["dev_cohort_guard_ok"] = guard_cohort(sorted(allowed_dev), allowed_dev)
    # 5. OOF held group in fit
    folds = np.load(ART / "dataset" / "folds.npy")
    meta = _meta("train")
    groups = np.asarray(meta["group_id"], dtype=object)
    held = set(groups[folds == 0].tolist())
    fit_groups = set(groups[folds != 0].tolist())
    out["oof_leak_detected"] = not guard_oof(fit_groups | {sorted(held)[0]}, held)
    out["oof_guard_accepts_clean"] = guard_oof(fit_groups, held)
    # 6. model SHA
    prov = json_load(ART / "capture" / "train" / "capture_provenance.json")
    expected = dict(prov["model_shards"])
    bad_prov = copy.deepcopy(prov)
    k0 = sorted(bad_prov["model_shards"])[0]
    bad_prov["model_shards"][k0] = "0" * 64
    out["model_sha_detected"] = not guard_model_sha(bad_prov, expected)
    out["model_sha_guard_accepts_clean"] = guard_model_sha(prov, expected)
    # 7. manifest tampering
    mpath = ART / "manifests" / "artifact_hashes.json"
    if mpath.is_file():
        man = json_load(mpath)
        bad = copy.deepcopy(man)
        cand = [k for k, v in bad.get("artifacts", {}).items() if "sha256" in v]
        if cand:
            bad["artifacts"][cand[0]]["sha256"] = "f" * 64
            out["manifest_tamper_detected"] = not guard_manifest(bad, ART.parent)
            out["manifest_guard_accepts_clean"] = guard_manifest(man, ART.parent)
    # 8. future token / endpoint evidence from the real smoke
    sm = ART / "smoke_depth" / "b24_equiv_mono.json"
    if sm.is_file():
        d = json.loads(sm.read_text())
        out["future_token_detected"] = bool(d.get("no_future_pass") and d.get("wrong_endpoint_pass")
                                            and d.get("determinism_pass"))
        out["future_token_clean"] = bool(d.get("suffix_invariance_pass"))
    # 9. forged metric on a shadow copy of the REAL report
    forged = copy.deepcopy(report)
    key = "H:full" if "H:full" in forged["dev_variants"] else sorted(forged["dev_variants"])[0]
    before = forged["dev_variants"][key]["fn"]
    forged["dev_variants"][key]["fn"] = int(before) - 3
    mism = check_report_metrics(forged, independent)
    out["forged_metric_detected"] = len(mism) > 0
    out["forged_metric_mismatches"] = mism[:5]
    out["clean_report_mismatches"] = check_report_metrics(report, independent)[:5]
    return out


def build_manifest() -> dict[str, Any]:
    """Independent artifact hash manifest (sizes for big tensors, SHA for small files)."""
    entries = {}
    total = 0
    for p in sorted(ART.rglob("*")):
        if not p.is_file() or p.name.endswith(".tmp") or "_sync" in p.parts:
            continue
        rel = str(p.relative_to(ART.parent))
        size = p.stat().st_size
        total += size
        if p.suffix in (".npz", ".npy", ".pt") or size > 8 << 20:
            entries[rel] = {"bytes": size}
        else:
            entries[rel] = {"bytes": size, "sha256": sha256_file(p)}
    payload = {"artifacts": entries, "total_bytes": total}
    (ART / "manifests").mkdir(parents=True, exist_ok=True)
    atomic_write_json(ART / "manifests" / "artifact_hashes.json", payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-per-split", type=int, default=60)
    ap.add_argument("--recapture", type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = json_load(EVAL / "depth_report.json")
    build_manifest()
    capture = check_capture(args.shards_per_split, args.recapture)
    formal = check_formal_models(report)
    muts = mutations(report, formal)
    required = ["bf16_conversion_detected", "wrong_endpoint_detected", "wrong_layer_detected",
                "row_order_detected", "illegal_cohort_detected", "oof_leak_detected",
                "model_sha_detected", "manifest_tamper_detected", "future_token_detected",
                "forged_metric_detected"]
    payload = {
        "capture": capture,
        "formal": formal,
        "mutations": muts,
        "summary": {
            "capture_ok": all(v["sample_ok"] for k, v in capture.items() if isinstance(v, dict) and "sample_ok" in v)
                          and capture.get("recapture_ok", True) is not False
                          and capture.get("v4_block8_ok", True) is not False
                          and capture.get("full36_ok", True) is not False,
            "formal_passed": formal["passed"],
            "n_formal_checks": formal["n_checks"],
            "mutations_all_detected": all(muts.get(k, False) for k in required),
            "mutations_missing": [k for k in required if not muts.get(k, False)],
            "forged_metric_detected": muts.get("forged_metric_detected"),
            "clean_report_ok": len(muts.get("clean_report_mismatches", [])) == 0,
        },
    }
    atomic_write_json(OUT / "validation.json", payload)
    print(json.dumps(payload["summary"], indent=2))
    if not all([payload["summary"]["capture_ok"], payload["summary"]["formal_passed"],
                payload["summary"]["mutations_all_detected"], payload["summary"]["clean_report_ok"]]):
        raise SystemExit("VALIDATION FAILED")


if __name__ == "__main__":
    main()
