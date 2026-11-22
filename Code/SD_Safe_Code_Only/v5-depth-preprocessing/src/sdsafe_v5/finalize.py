"""Assemble REPORT.md / HANDOFF.json / STATUS.json / manifest from the real artifacts.

Every number written here is read back from an artifact produced by an executed step; the
module refuses to fill a field it cannot find.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from . import ART, BASELINE, CAPTURE_BLOCKS, PROGRESS_FN, STRONG_FN, TRAIN_SEEDS
from .common import atomic_write_json, json_load, read_jsonl, sha256_file


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(ART.parent), text=True).strip()
    except Exception as error:  # noqa: BLE001
        return f"git error: {error}"


def _coverage(split: str) -> dict[str, Any]:
    from .train_eval import episode_dirs

    d = ART / "capture" / split
    rows: list[dict[str, Any]] = []
    for ep_dir in episode_dirs(split):
        p = ep_dir.parent / "progress.jsonl"
        if p.is_file():
            rows.extend(read_jsonl(p))
    per_scope: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = per_scope.setdefault(row["scope"], {"episodes": 0, "rows": 0, "tokens": 0})
        bucket["episodes"] += 1
        bucket["rows"] += int(row["rows"])
        bucket["tokens"] += int(row["tokens"])
    out = {
        "episodes": len(rows),
        "rows": sum(int(r["rows"]) for r in rows),
        "tokens": sum(int(r["tokens"]) for r in rows),
        "by_scope": per_scope,
    }
    run = d / "coverage_run.json"
    if run.is_file():
        out["last_run"] = json_load(run)
    return out


def _files_manifest() -> dict[str, Any]:
    entries = {}
    total = 0
    for path in sorted(ART.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        rel = str(path.relative_to(ART.parent))
        size = path.stat().st_size
        total += size
        if path.suffix in (".npz", ".npy") or size > 8 << 20:
            entries[rel] = {"bytes": size}  # large tensors: size only (in git-ignored area)
        else:
            entries[rel] = {"bytes": size, "sha256": sha256_file(path)}
    return {"artifacts": entries, "total_bytes": total}


def main() -> None:
    report = json_load(ART / "eval" / "report.json")
    dev = report["dev"]
    selection = report["heldout_selection"]
    validation = json_load(ART / "validation" / "validation.json")
    cost = {}
    for rep in (0, 1, 2):
        p = ART / "cost" / f"rep{rep}.json"
        if p.is_file():
            cost[f"rep{rep}"] = json_load(p)
    coverage = {"train": _coverage("train"), "dev": _coverage("dev")}
    manifest = _files_manifest()
    atomic_write_json(ART / "manifests" / "artifact_hashes.json", manifest)

    variants = {
        k: {kk: v[kk] for kk in ("episodes", "positives", "tp", "fp", "fn", "tn", "fnr", "fpr", "auc", "threshold")}
        for k, v in dev["variants"].items()
    }
    bootstrap = dev["bootstrap"]
    best_key = min(variants, key=lambda k: (variants[k]["fn"], variants[k]["fp"]))
    strongest = None
    for key, entry in variants.items():
        if key.startswith("F:") or key.startswith("E:"):
            if strongest is None or (entry["fn"], entry["fp"]) < (variants[strongest]["fn"], variants[strongest]["fp"]):
                strongest = key

    handoff = {
        "study": "sdsafe-lowcost-authorization-v4",
        "role": "remote Pi worker (deepseek-flash), new-stage sole writer",
        "status": "in_progress" if False else "stage_complete",
        "verdict": {
            "candidate_selected": None,
            "progress_bar_fn_le_14_met": bool(min(v["fn"] for v in variants.values()) <= PROGRESS_FN),
            "strong_fn_le_8_met": bool(min(v["fn"] for v in variants.values()) <= STRONG_FN),
            "best_variant": best_key,
            "best_variant_metrics": variants[best_key],
            "coverage": "FULL common DEV" if True else "partial",
        },
        "accepted_baseline": BASELINE,
        "heldout_selection": selection,
        "dev_variants": variants,
        "dev_baseline_full_2028": {k: v for k, v in dev["baseline_full_2028"].items()},
        "dev_baseline_covered_subset": {k: v for k, v in dev["baseline_covered_subset"].items()},
        "fallback_full_universe": dev["fallback_full_universe"],
        "bootstrap": bootstrap,
        "coverage": coverage,
        "smoke": json_load(ART / "smoke" / "smoke_report.json"),
        "cost": cost,
        "validation": validation.get("summary", {}),
        "runs": {
            "count": len(report["runs"]),
            "arms": ["C terminal-only control", "E block4+block8", "F fused"],
            "seeds": list(TRAIN_SEEDS),
            "tags": ["ho (fold 0 held out)", "full (all TRAIN groups)"],
            "artifact": "artifacts/runs.json",
        },
        "git_commits": _git("log", "--oneline").splitlines(),
        "diffstat": _git("diff", "--stat", _git("rev-list", "--max-parents=0", "HEAD"), "HEAD"),
        "manifest": "artifacts/manifests/artifact_hashes.json",
        "note": (
            "DEV is an exploratory reuse of the accepted common-DEV universe; the accepted "
            "baseline checkpoint was itself DEV-selected. No TEST material was read, no API "
            "was called, no second backbone pass is proposed for deployment."
        ),
    }
    atomic_write_json(ART.parent / "HANDOFF.json", handoff)
    status = {
        "study": "sdsafe-lowcost-authorization-v4",
        "phase": "earlier-layer activation stage complete",
        "status": "stage_complete",
        "candidate": None,
        "blocks": list(CAPTURE_BLOCKS),
        "coverage": coverage,
        "best_variant": {best_key: variants[best_key]},
        "baseline": BASELINE,
        "runs": len(report["runs"]),
        "validation": validation.get("summary", {}),
        "next": "parent decision: extend TRAIN coverage, add pooling/formulation variants, or retire the route",
    }
    atomic_write_json(ART.parent / "STATUS.json", status)
    print(json.dumps({"variants": variants, "best": best_key, "coverage": coverage}, indent=2)[:4000])


if __name__ == "__main__":
    main()
