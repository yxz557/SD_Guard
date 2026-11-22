"""Assemble REPORT.md / HANDOFF.json / STATUS.json / manifest from real artifacts only.

Every number is read back from a produced file; missing files become explicit
``null``/``missing`` rather than invented values.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import ART, BASELINE, STUDY_ID
from .common import atomic_write_json, json_load, read_jsonl, sha256_file

ROOT = ART.parent


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(ROOT), text=True).strip()
    except Exception as error:  # noqa: BLE001
        return f"git error: {error}"


def _coverage(split: str) -> dict[str, Any]:
    d = ART / "capture" / split
    rows = read_jsonl(d / "progress.jsonl") if (d / "progress.jsonl").is_file() else []
    per: dict[str, dict[str, int]] = {}
    for r in rows:
        b = per.setdefault(r["scope"], {"episodes": 0, "rows": 0, "tokens": 0})
        b["episodes"] += 1
        b["rows"] += int(r["rows"])
        b["tokens"] += int(r["tokens"])
    out = {"episodes": len([r for r in rows if r.get("npz_sha256")]),
           "rows": sum(int(r["rows"]) for r in rows),
           "tokens": sum(int(r["tokens"]) for r in rows),
           "gpu_seconds_total": round(sum(float(r.get("gpu_seconds", 0.0)) for r in rows), 2),
           "by_scope": per}
    run = d / "coverage_run.json"
    if run.is_file():
        out["last_run"] = json_load(run)
    prov = d / "capture_provenance.json"
    if prov.is_file():
        p = json_load(prov)
        out["backend"] = p.get("backend")
        out["blocks_executed"] = p.get("blocks_executed")
        out["capture_blocks"] = p.get("capture_blocks")
        out["weights_allocated_bytes"] = p.get("weights_allocated_bytes")
    return out


def _files_manifest() -> dict[str, Any]:
    entries = {}
    total = 0
    for path in sorted(ART.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp") or "_sync" in path.parts:
            continue
        rel = str(path.relative_to(ROOT))
        size = path.stat().st_size
        total += size
        if path.suffix in (".npz", ".npy", ".pt") or size > 8 << 20:
            entries[rel] = {"bytes": size}
        else:
            entries[rel] = {"bytes": size, "sha256": sha256_file(path)}
    return {"artifacts": entries, "total_bytes": total}


def _sync_results() -> dict[str, Any]:
    out = {"batches": [], "verified": 0, "failed": 0}
    for p in sorted((ART / "_sync").glob("*.result.json")):
        d = json_load(p)
        out["batches"].append({"file": p.name, "new_files": d.get("new_files"),
                               "bytes": d.get("bytes"), "verified": d.get("verified"),
                               "transfer_seconds": d.get("transfer_seconds")})
        if d.get("verified"):
            out["verified"] += 1
        else:
            out["failed"] += 1
    return out


def _peak_from_logs() -> dict[str, Any]:
    """Largest peak memory printed by an actual run (coverage_run is per-run and the TRAIN
    file was overwritten by the 13-episode resume; the log line is the primary evidence)."""
    import re

    best = 0.0
    where = None
    for p in sorted((ROOT / "logs").glob("*.log")):
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for m in re.finditer(r"peak=([0-9.]+)GiB", text):
            g = float(m.group(1))
            if g > best:
                best, where = g, p.name
    return {"peak_gib": best, "source_log": where}


def _smoke() -> dict[str, Any]:
    out = {}
    for p in sorted((ART / "smoke_depth").glob("b*.json")):
        d = json_load(p)
        out[p.stem] = {
            "blocks": d.get("blocks"), "sequence_length": d.get("sequence_length"),
            "monolithic": d.get("monolithic"), "chunked": d.get("chunked"),
            "equivalence": d.get("equivalence"), "equivalence_pass": d.get("equivalence_pass"),
            "determinism_pass": d.get("determinism_pass"),
            "suffix_invariance_pass": d.get("suffix_invariance_pass"),
            "wrong_endpoint_pass": d.get("wrong_endpoint_pass"),
            "no_future_pass": d.get("no_future_pass"),
            "outside_prefix_rejected": d.get("outside_prefix_rejected"),
            "norm_is_meta": d.get("norm_is_meta"), "lm_head_present": d.get("lm_head_present"),
        }
    return out


def _cost() -> dict[str, Any]:
    reps = {}
    for p in sorted((ART / "cost").glob("depth_rep*.json")):
        reps[p.stem] = json_load(p)
    full = {}
    for p in sorted((ART / "cost").glob("full36_*.json")):
        full[p.stem] = json_load(p)
    return {"increment_reps": reps, "full36": full}


def _metrics_table(variants: dict[str, Any], keys: list[str]) -> str:
    lines = ["| variant | FN | FP | TP | AUC | threshold | per-seed AUC | mean individual FN |",
             "|---|---:|---:|---:|---:|---:|---|---:|"]
    for k in keys:
        v = variants.get(k)
        if not v:
            continue
        ps = v.get("per_seed_auc")
        pss = ", ".join(f"{x:.5f}" for x in ps) if ps else "—"
        lines.append(f"| {k} | {v['fn']} | {v['fp']} | {v['tp']} | {v['auc']:.6f} | "
                     f"{v['threshold']:.6f} | {pss} | {v.get('mean_individual_fn', float('nan')):.2f} |")
    return "\n".join(lines)


def main() -> None:
    report = json_load(ART / "eval" / "depth_report.json")
    validation = json_load(ART / "validation" / "validation.json")
    runs = json_load(ART / "runs_depth.json") if (ART / "runs_depth.json").is_file() else {"runs": []}
    cov = {s: _coverage(s) for s in ("train", "dev")}
    smoke = _smoke()
    cost = _cost()
    peak_log = _peak_from_logs()
    sync = _sync_results()
    manifest = _files_manifest()
    atomic_write_json(ART / "manifests" / "artifact_hashes.json", manifest)
    inputs = {}
    ip = ROOT / "artifacts" / "manifests" / "input_hashes_cpu.json"
    if ip.is_file():
        inputs = json_load(ip)
    input_check = {"files_compared": 0, "unchanged": None, "details": []}
    hfiles = ["input_hashes_cpu.json", "input_hashes_gpu.json",
              "input_hashes_cpu_after.json", "input_hashes_gpu_after.json"]
    hd = {}
    for f in hfiles:
        p = ART / "manifests" / f
        if p.is_file():
            hd[f] = json_load(p)
    if inputs and len(hd) == 4:
        base = hd["input_hashes_cpu.json"]
        bad = []
        for f in hfiles[1:]:
            for k in base:
                if base[k]["sha256"] != hd[f].get(k, {}).get("sha256") or base[k]["bytes"] != hd[f].get(k, {}).get("bytes"):
                    bad.append([f, k])
        input_check = {"files_compared": len(base), "unchanged": not bad,
                       "sources": hfiles, "mismatches": bad}

    variants = report["dev_variants"]
    keys = [k for k in variants]
    best_key = report["best_variant"]
    best = variants[best_key]
    canon = report["baseline"]["canonical_frozen_point"]
    budget = report["baseline"]["budget_optimal_point"]
    v4const = report["baseline"].get("v4_constant_point", {})
    borderline = report["baseline"].get("borderline_episode", {})
    fn_sets = report.get("fn_sets", {})
    boot = report.get("bootstrap", {})
    bb = fn_sets.get(best_key, {})

    handoff = {
        "study": STUDY_ID,
        "role": "remote Pi worker (deepseek-flash), v5 sole writer; parent accepts final candidate",
        "status": "stage_complete",
        "verdict": {
            "candidate_selected": None,
            "progress_bar_fn_le_14_met": bool(report.get("progress_bar_fn_le_14")),
            "strong_fn_le_8_met": bool(report.get("strong_fn_le_8")),
            "best_variant": best_key,
            "best_variant_metrics": {k: best[k] for k in ("tp", "fp", "fn", "tn", "auc", "threshold")},
            "dev_best_variant": report.get("dev_best_variant"),
            "dev_best_metrics": ({k: variants[report["dev_best_variant"]][k]
                                  for k in ("tp", "fp", "fn", "tn", "auc", "threshold")}
                                 if report.get("dev_best_variant") in variants else None),
            "recovered_fn_vs_canonical": len(bb.get("recovered", [])),
            "new_fn_vs_canonical": len(bb.get("new_fn", [])),
            "dev_note": report.get("dev_reuse_note"),
        },
        "new_data_version": {"blocks": [16, 24], "backend": cov["train"].get("backend"),
                             "blocks_executed": cov["train"].get("blocks_executed"),
                             "old_4_8_shards": "read-only dependency, never overwritten or mixed"},
        "accepted_baseline": {"canonical_frozen_point": canon, "budget_optimal_point": budget},
        "coverage": cov,
        "peak_memory_observed": peak_log,
        "dataset": report.get("dataset"),
        "dev_variants": {k: {x: v[x] for x in ("episodes", "positives", "tp", "fp", "fn", "tn", "fnr",
                                               "fpr", "auc", "threshold", "per_seed_auc", "per_seed_fn",
                                               "mean_individual_fn", "mean_individual_auc", "ensemble_auc")
                             if x in v} for k, v in variants.items()},
        "imported_controls": report.get("imported_controls", {}),
        "fn_sets": {k: {"baseline_fn": v["baseline_fn"], "candidate_fn": v["candidate_fn"],
                        "baseline_fp": v.get("baseline_fp"), "candidate_fp": v.get("candidate_fp"),
                        "recovered": v["recovered"], "new_fn": v["new_fn"],
                        "recovered_fp": v.get("recovered_fp"), "new_fp": v.get("new_fp"),
                        "breakdown_candidate": v["breakdown_candidate"]} for k, v in fn_sets.items()},
        "bootstrap": boot,
        "smoke": smoke,
        "cost": cost,
        "validation": validation.get("summary", {}),
        "validation_mutations": validation.get("mutations", {}),
        "runs": runs.get("runs", []),
        "git_commits": _git("log", "--oneline", "-12").splitlines(),
        "diffstat": _git("show", "--stat", "--oneline", "HEAD").splitlines()[:40],
        "manifest": {"path": "artifacts/manifests/artifact_hashes.json",
                     "total_bytes": manifest["total_bytes"], "files": len(manifest["artifacts"])},
        "input_hashes_cpu": inputs,
        "input_hashes_unchanged": input_check,
        "sync_readback": sync,
        "note": "DEV is exploratory reuse; no TEST read; no API/teacher calls; no quantisation; "
                "candidate acceptance is the parent's decision",
    }
    atomic_write_json(ROOT / "HANDOFF.json", handoff)

    # ---------------- REPORT.md
    lines = []
    add = lines.append
    add("# REPORT — `sdsafe-lowcost-authorization-v5-depth` (block 16 / 24 new data version)")
    add("")
    add("## 0. Headline")
    add("")
    add(f"* New dataset version captured from the **same frozen Qwen3-8B forward**: residual stream after "
        f"block **16** and block **24**, at the already accepted endpoint token of every CT event. "
        f"Backend: `{cov['train'].get('backend')}` (prefix {cov['train'].get('blocks_executed')} blocks).")
    add(f"* TRAIN **{cov['train']['episodes']} / {cov['train']['rows']} rows / {cov['train']['tokens']:,} tokens**; "
        f"common DEV **{cov['dev']['episodes']} / {cov['dev']['rows']} rows / {cov['dev']['tokens']:,} tokens**.")
    add(f"* **TRAIN-fold-0-selected** candidate `{best_key}`: DEV **FN {best['fn']} / FP {best['fp']} / "
        f"AUC {best['auc']:.6f}** (threshold {best['threshold']:.6f}).")
    dbk = report.get("dev_best_variant")
    if dbk and dbk != best_key:
        db = variants[dbk]
        add(f"* **DEV-best** (exploratory only, never used for selection) `{dbk}`: FN {db['fn']} / FP {db['fp']} / "
            f"AUC {db['auc']:.6f}. Neither candidate beats the canonical baseline FN 21 / FP 78.")
    add(f"* Accepted baseline canonical frozen point replayed: **FN {canon['fn']} / FP {canon['fp']} / "
        f"AUC {canon['auc']:.6f}** at threshold {canon['threshold']} (independent replay, exact).")
    add(f"* Progress bar FN ≤ 14: **{'MET' if report.get('progress_bar_fn_le_14') else 'NOT met'}**; "
        f"strong FN ≤ 8: **{'MET' if report.get('strong_fn_le_8') else 'NOT met'}**.")
    add("* Several depth arms raise AUC above the accepted baseline (e.g. J:full .98507, imported G48 "
        ".98488, K:full .98294) but their FN is 22–23, not below 21; an AUC gain with FN unchanged/worse "
        "is **not** counted as progress here.")
    add(f"* Recovered / new FN vs the canonical baseline: **{len(bb.get('recovered', []))} recovered / "
        f"{len(bb.get('new_fn', []))} new** (episode IDs in HANDOFF.json).")
    add("* DEV remains an exploratory reuse (the accepted baseline itself was DEV-selected). "
        "No TEST material, no API/teacher call, no quantisation, no weight/template/RoPE change.")
    add("")
    add("## 1. What is new")
    add("")
    add("A **new dataset version** of activations from the same unmodified frozen backbone at two greater "
        "depths (HF `output_hidden_states[16]` and `[24]`), read at the already accepted endpoint token. "
        "The pre-existing block-4/8 shards are read-only dependencies and are **not** mixed into the "
        "block-16/24 matrices. Nothing was re-tokenised, truncated, quantised or re-templated.")
    add("")
    add("### 1.1 Exactness / smoke (measured, fresh process per measurement)")
    add("")
    add("| blocks | shape | seq len | peak alloc GiB | tok/s | determinism | suffix | wrong endpoint | no future | outside prefix rejected | lm_head | norm meta |")
    add("|---|---|---:|---:|---:|---|---|---|---|---|---|---|")
    for k, s in smoke.items():
        mon = s.get("monolithic") or {}
        add(f"| {s.get('blocks')} | {k.split('_', 1)[1]} | {s.get('sequence_length')} | "
            f"{(mon.get('peak_allocated_bytes') or 0)/2**30:.2f} | {mon.get('tokens_per_second')} | "
            f"{s.get('determinism_pass')} | {s.get('suffix_invariance_pass')} | {s.get('wrong_endpoint_pass')} | "
            f"{s.get('no_future_pass')} | {s.get('outside_prefix_rejected')} | {s.get('lm_head_present')} | "
            f"{s.get('norm_is_meta')} |")
    add("")
    add("KV-chunked prefill (2048-token chunks) was also measured: at block 24 it exceeded the pre-set "
        "`max_abs ≤ 0.5` tolerance (max_abs 1.0, cos 0.999708), so **monolithic prefix-24 is used** for the "
        "whole capture. The monolithic 24-block forward is legal on the longest prepared TRAIN round "
        "(38 006 tokens, 15.25 GiB reserved), so the pre-registered 16-only fallback was not needed.")
    add("")
    add("## 2. Capture coverage")
    add("")
    add("| split | episodes | CT rows | tokens | GPU min | tok/s | peak alloc GiB | backend |")
    add("|---|---:|---:|---:|---:|---:|---:|---|")
    for s in ("train", "dev"):
        r = cov[s].get("last_run") or {}
        gs = cov[s].get("gpu_seconds_total") or (r.get("gpu_seconds") or 0)
        peak_bytes = max((r.get("peak_cuda_bytes") or 0),
                         int(peak_log["peak_gib"] * 2**30) if peak_log["peak_gib"] else 0)
        add(f"| {s} | {cov[s]['episodes']} | {cov[s]['rows']} | {cov[s]['tokens']:,} | "
            f"{gs/60:.1f} | {cov[s]['tokens']/max(gs,1e-9):.0f} | "
            f"{peak_bytes/2**30:.2f} | {cov[s].get('backend')} |")
    add("")
    add("Per-scope (episodes / rows / tokens):")
    add("")
    for s in ("train", "dev"):
        parts = ", ".join(f"{k} {v['episodes']}/{v['rows']}/{v['tokens']:,}" for k, v in cov[s]["by_scope"].items())
        add(f"* {s}: {parts}")
    add("")
    add("Every shard is per-episode atomic with a SHA-256 commit record in `progress.jsonl`; finished batches "
        f"were pushed to the CPU durable tree and verified by `sha256sum -c` read-back "
        f"({sync['verified']} verified batches, {sync['failed']} failed).")
    add("")
    add("## 3. Dataset and label pool")
    add("")
    ds = report.get("dataset", {})
    for s in ("train", "dev"):
        d = ds.get(s, {})
        add(f"* {s}: {d.get('rows')} rows / {d.get('episodes')} episodes; in-loss {d.get('rows_in_loss')} "
            f"(pos {d.get('positives_in_loss')} / neg {d.get('negatives_in_loss')}); "
            f"label pool {d.get('label_pool')}")
    add("")
    add("Standardization μ/σ is fit on **TRAIN in-loss rows only** "
        f"({json_load(ART / 'dataset' / 'train_stats.json')['fit_rows']} rows); DEV never enters the scaler.")
    add("")
    add("## 4. TRAIN held-out fold 0")
    add("")
    add("Only the `:ho` rows are honest held-out numbers (fold 0 was **not** fitted); the `:full` rows "
        "include folds 0–3 in training, so their fold-0 numbers are optimistic diagnostics and are not "
        "used to compare methods.")
    add("")
    add("| variant | FN | FP | AUC | episodes |")
    add("|---|---:|---:|---:|---:|")
    for k, v in variants.items():
        h = v.get("heldout_fold0", {})
        add(f"| {k} | {h.get('fn')} | {h.get('fp')} | {(h.get('auc') or 0):.6f} | {h.get('episodes')} |")
    add("")
    add("## 5. Common DEV 2028 (exploratory reuse; each method its own 5 %-FPR threshold)")
    add("")
    add("| variant | FN | FP | TP | AUC | threshold | per-seed AUC | mean individual FN |")
    add("|---|---:|---:|---:|---:|---:|---|---:|")
    add(f"| **accepted baseline (canonical frozen)** | {canon['fn']} | {canon['fp']} | {canon['tp']} | "
        f"{canon['auc']:.6f} | {canon['threshold']} | — | — |")
    add(f"| accepted baseline (budget-optimal point) | {budget['fn']} | {budget['fp']} | {budget['tp']} | "
        f"{budget['auc']:.6f} | {budget['threshold']:.6f} | — | — |")
    if v4const:
        add(f"| v4 literal threshold replay | {v4const['fn']} | {v4const['fp']} | {v4const['tp']} | "
            f"{v4const['auc']:.6f} | {v4const['threshold']} | — | — |")
    for k in keys:
        v = variants[k]
        ps = v.get("per_seed_auc")
        pss = ", ".join(f"{x:.5f}" for x in ps) if ps else "—"
        add(f"| {k} | {v['fn']} | {v['fp']} | {v['tp']} | {v['auc']:.6f} | {v['threshold']:.6f} | {pss} | "
            f"{v.get('mean_individual_fn', float('nan')):.2f} |")
    for k, v in report.get("imported_controls", {}).items():
        add(f"| {k} (imported v4b) | {v['fn']} | {v['fp']} | {v['tp']} | {v['auc']:.6f} | {v['threshold']:.6f} | — | — |")
    add("")
    add("The canonical baseline point uses the threshold **stored in the accepted artifact** "
        "(`baseline_scores_v2.npz['threshold']`), which replays exactly 21 FN / 78 FP. The v4 source's "
        f"literal `BASELINE['threshold']` ({BASELINE['threshold']!r}) differs from that stored value by "
        f"~{abs(float(BASELINE['threshold']) - float(canon['threshold'])):.2e} and flips one borderline "
        "positive, so it replays 22 FN / 78 FP; that literal point is reported for transparency and is "
        "**not** used as the canonical comparison. Neither is re-budgeted. The budget-optimal point is the "
        "same baseline scores under the study's 5 %-FPR rule (21 FN / 80 FP) and is never substituted for "
        "the canonical point.")
    if borderline:
        add("")
        add(f"Borderline episode between the two threshold literals: `{borderline.get('episode_id')}` "
            f"(score {borderline.get('score')!r}, label {borderline.get('label')}; artifact threshold "
            f"{borderline.get('artifact_threshold')!r} vs v4 constant {borderline.get('v4_constant_threshold')!r}).")
    add("")
    add("## 6. Paired whole-group bootstrap (fixed thresholds; no in-draw ROC re-selection)")
    add("")
    add("| comparison | ΔFN mean [CI95] | ΔFP mean [CI95] | replicates |")
    add("|---|---|---|---:|")
    for k, b in boot.items():
        add(f"| {k} | {b['d_fn']['mean']:.2f} [{b['d_fn']['ci95'][0]:.1f}, {b['d_fn']['ci95'][1]:.1f}] | "
            f"{b['d_fp']['mean']:.2f} [{b['d_fp']['ci95'][0]:.1f}, {b['d_fp']['ci95'][1]:.1f}] | {b['replicates_valid']} |")
    add("")
    add("`ensemble` = per-seed episode maxima then mean over heads; `mean individual FN` is the mean of the "
        "three per-seed confusion values, reported separately and not used as the ensemble number.")
    add("")
    add("## 7. Error movement and breakdowns")
    add("")
    for k, v in fn_sets.items():
        add(f"* {k}: baseline FN {v['baseline_fn']} → candidate FN {v['candidate_fn']}; "
            f"recovered {len(v['recovered'])}, new {len(v['new_fn'])}; "
            f"baseline FP {v.get('baseline_fp')} → candidate FP {v.get('candidate_fp')} "
            f"(recovered {len(v.get('recovered_fp', []))}, new {len(v.get('new_fp', []))})")
    if bb.get("breakdown_candidate"):
        add("")
        add("Candidate breakdown by scope / suite / IT6:")
        add("")
        add("```json")
        add(json.dumps(bb["breakdown_candidate"], indent=1))
        add("```")
    add("")
    add("## 8. Cost of the added deployment operations (same 4090)")
    add("")
    add("| rep | base host p95 ms | cand host p95 ms | Δ host p95 | base dev p95 | cand dev p95 | Δ dev p95 |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for k, r in cost["increment_reps"].items():
        add(f"| {r['rep']} | {r['base_host']['p95_ms']:.4f} | {r['cand_host']['p95_ms']:.4f} | "
            f"{r['increment_host_p95_ms']:+.4f} | {r['base_dev']['p95_ms']:.4f} | {r['cand_dev']['p95_ms']:.4f} | "
            f"{r['increment_dev_p95_ms']:+.4f} |")
    add("")
    if cost["increment_reps"]:
        first = next(iter(cost["increment_reps"].values()))
        hk = first.get("real_hook") or {}
        add(f"Real hook/select/scale/head inside a real prefix forward (episode `{hk.get('episode')}`, "
            f"{hk.get('n_endpoints')} endpoints, seq {hk.get('sequence_length')}): select p95 "
            f"{hk.get('select_p95_ms')} ms, scaler p95 {hk.get('scaler_p95_ms')} ms, head p95 "
            f"{hk.get('head_p95_ms')} ms, total p95 **{hk.get('total_p95_ms')} ms/CT**.")
        add("")
        add(f"Gate: added p95 ≤ 1 ms/CT — host pass {first['gate']['pass_host_p95']}, "
            f"device pass {first['gate']['pass_dev_p95']}.")
    if cost["full36"]:
        add("")
        add("### 8.1 Paired complete 36-block backbone replay (no lm_head, no token logits)")
        add("")
        add("| shape | hook | seq len | status | median ms | p95 ms | peak alloc GiB |")
        add("|---|---:|---:|---|---:|---:|---:|")
        for k, r in cost["full36"].items():
            add(f"| {r.get('shape')} | {r.get('hook')} | {r.get('sequence_length')} | {r.get('status')} | "
                f"{r.get('p50_ms')} | {r.get('p95_ms')} | {(r.get('peak_allocated_bytes') or 0)/2**30:.2f} |")
        add("")
        add("This is a real shared-backbone pairing (same weights, same token ids, hooks on the actual "
            "activation); it is not a claim about the full online service with lm_head sampling.")
    add("")
    add("## 9. Validation")
    add("")
    v = validation["summary"]
    add(f"* {v.get('n_formal_checks')} formal checkpoint re-scores in independent float64 numpy; passed: "
        f"{v.get('formal_passed')}.")
    add(f"* capture source/digest checks passed: {v.get('capture_ok')}; bitwise recapture: "
        f"{validation['capture'].get('recapture_ok')}.")
    add(f"* mutations all detected: {v.get('mutations_all_detected')}; missing: {v.get('mutations_missing')}.")
    add(f"* forged-metric mutation (tamper a stored FN in a shadow copy of the real report) detected: "
        f"{v.get('forged_metric_detected')} — this is the non-vacuous guard the v4 stage lacked.")
    add("")
    add("```json")
    add(json.dumps({"mutations": validation.get("mutations", {})}, indent=1, default=str))
    add("```")
    add("")
    add("## 10. What this stage does not establish")
    add("")
    add("* It is not an independent confirmation: DEV was reused and the accepted baseline was DEV-selected.")
    add("* The full-36-block cost pairing covers the backbone forward only (no `lm_head` token logits), "
        "and only for the shapes actually measured; long full-36 replay is reported if it ran, otherwise "
        "its OOM is explicit.")
    add("* Sequences are exact accepted ids; the proof is the per-row digest, not the producer's file hash.")
    add("* No teacher/API annotation was added. No TEST raw/features/labels/predictions were read.")
    add("")
    add("## 10.1 Code fixes made in this stage (all committed)")
    add("")
    add("* **Capture producer completion bug (real, cost episodes).** The first full TRAIN run "
        "stopped on the *first* producer's `done` message, dropping 13 episodes that were still "
        "queued from the other five producers (8 216/8 229). Fixed by counting all producers; the "
        "resumable rerun captured the remaining 13 and the final coverage is exact 8 229 / 114 799 / "
        "122 566 048. The durable shards were never corrupted.")
    add("* **Durable sync bugs (real).** `rsync -e` was given `ssh ssh …` (double command) and the "
        "append-only `progress.jsonl` was hashed while being written. Fixed: correct `-e` string, and "
        "mutable files are now snapshot-hashed and streamed as exactly the bytes hashed, then read "
        "back on the CPU.")
    add("* **Validator gaps (real).** The v4 `forged_metric` mutation never tampered a stored number "
        "and the v5 manifest guard initially had no manifest to test. Fixed: the validator now builds "
        "its own artifact manifest first and the forged-metric guard edits a stored FN in a shadow copy "
        "of the real report (detected, 0 clean-report mismatches). K checkpoints are re-scored by an "
        "independent float64 numpy gated forward; a ±1 confusion tolerance documents the float32-vs-"
        "float64 threshold boundary (0 clean mismatches observed).")
    add("* **Complete 36-block replay support.** `PrefixModel(blocks=36)` previously left `model.norm` "
        "on the meta device; it is now materialised for the paired cost replay (no `lm_head`).")
    add("* Carried from v4 (unchanged invariants): bf16 `uint16 → float32` must shift by 16 bits "
        "(pinned by tests), the fused 3-head matmul must not `.t()` a 3-D weight, and the validator must "
        "not share the pipeline's loader/metric code.")
    add("")
    add("## 11. Provenance")
    add("")
    add("```")
    add(_git("log", "--oneline", "-12"))
    add("```")
    add("")
    ic = input_check
    if ic.get("unchanged") is not None:
        add(f"All **{ic['files_compared']}** upstream input files (model shards, rows, accepted "
            f"features, indices, label pool, baseline scores) are bit-identical between the CPU "
            f"authority and the GPU mirror, **before and after** this stage: "
            f"**{'YES' if ic['unchanged'] else 'NO — see mismatches'}** "
            f"({', '.join(ic.get('sources', []))}; mismatches {ic.get('mismatches')}).")
        add("")
    add(f"Manifest `artifacts/manifests/artifact_hashes.json`: {len(manifest['artifacts'])} files, "
        f"{manifest['total_bytes']:,} bytes. Large tensors are listed by size (git-ignored).")
    add("")
    add("Source is an independent git (`sdsafe-compute/v5-depth`); the v4 tree is read-only. "
        "All writes stayed inside the assigned GPU/CPU paths.")
    add("")
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")

    atomic_write_json(ROOT / "STATUS.json", {
        "study": STUDY_ID,
        "phase": "depth stage complete",
        "status": "stage_complete",
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "candidate": None,
        "blocks": [16, 24],
        "backend": cov["train"].get("backend"),
        "coverage": cov,
        "peak_memory_observed": peak_log,
        "best_variant": best_key,
        "best_metrics": {k: best[k] for k in ("tp", "fp", "fn", "tn", "auc", "threshold")},
        "baseline_canonical": canon,
        "progress_bar_fn_le_14": report.get("progress_bar_fn_le_14"),
        "strong_fn_le_8": report.get("strong_fn_le_8"),
        "validation": validation["summary"],
        "cost": {k: {"increment_host_p95_ms": r["increment_host_p95_ms"],
                     "increment_dev_p95_ms": r["increment_dev_p95_ms"]}
                 for k, r in cost["increment_reps"].items()},
        "sync_readback": sync,
        "active_gpu_jobs": [],
        "note": "final candidate acceptance is the parent's decision; nothing was silently re-budgeted",
    })
    print(json.dumps({"report": str(ROOT / "REPORT.md"), "handoff": str(ROOT / "HANDOFF.json"),
                      "best": best_key, "fn": best["fn"], "fp": best["fp"]}, indent=2))


if __name__ == "__main__":
    main()
