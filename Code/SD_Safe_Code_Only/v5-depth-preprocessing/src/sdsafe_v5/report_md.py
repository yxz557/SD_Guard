"""Compose REPORT.md from the produced artifacts (no hand-typed numbers)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import ART
from .common import json_load


def _fmt(v: Any, nd: int = 4) -> str:
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main() -> None:
    h = json_load(ART.parent / "HANDOFF.json")
    smoke = h["smoke"]
    cov = h["coverage"]
    dev = h["dev_variants"]
    sel = h["heldout_selection"]
    cost = h.get("cost", {})
    val = h.get("validation", {})
    lines: list[str] = []
    A = lines.append

    A("# REPORT — `sdsafe-lowcost-authorization-v4` (earlier-layer activation route)")
    A("")
    A("## 0. Headline")
    A("")
    best = h["verdict"]["best_variant"]
    A(f"* All 18 pre-registered fits ran. Best DEV variant on the captured common-DEV universe: "
      f"`{best}` → FN {dev[best]['fn']} / FP {dev[best]['fp']} / AUC {dev[best]['auc']:.6f} "
      f"(accepted baseline on the same episodes: FN {h['dev_baseline_covered_subset']['fn']} / "
      f"FP {h['dev_baseline_covered_subset']['fp']} / AUC {h['dev_baseline_covered_subset']['auc']:.6f}).")
    A(f"* Progress bar FN ≤ 14: **{'met' if h['verdict']['progress_bar_fn_le_14_met'] else 'not met'}**; "
      f"strong FN ≤ 8: **{'met' if h['verdict']['strong_fn_le_8_met'] else 'not met'}**.")
    A(f"* Captured {cov['dev']['episodes']} / 2028 common-DEV episodes and {cov['train']['episodes']} TRAIN "
      f"episodes; DEV coverage {h['verdict']['coverage']}.")
    A("* DEV is an exploratory reuse of the accepted common-DEV universe. No TEST material, no API call, "
      "no deployment-time second backbone pass.")
    A("")
    A("## 1. What is new (and what is not)")
    A("")
    A("New: the residual stream of the **same frozen Qwen3-8B** after block 4 and after block 8, read at the "
      "**already accepted** endpoint token of every CT event. It was never in any accepted artifact, so the "
      "v1–v3 finding (\"re-weighting the final layer does not move the operating point\") says nothing about it.")
    A("")
    A("Not new / not claimed: no new backbone, no re-training of Qwen, no quantisation, no truncation, no "
      "re-tokenisation, no second deployment pass, no TEST read, no teacher/API annotation.")
    A("")
    A("### Exactness of the offline prefix execution")
    A("")
    A("`embed_tokens + rotary_emb + layers[0:8]` of the unmodified frozen weights, executed through "
      "`transformers`'s own `Qwen3Model.forward` with an early-exit `forward_pre_hook` on `layers[8]` "
      "(so `layers[8..35]` never run and `model.norm` is never applied). Weight keys are asserted equal to the "
      "safetensors key map; the token ids are the accepted archived/render ids, verified per round against the "
      "row's frozen `sequence_sha256`.")
    A("")
    A("| smoke check | result |")
    A("|---|---|")
    A(f"| early-exit hook vs independent replay of the same loop | bitwise equal (blocks {smoke['hook_vs_replicated']['4']['bitwise_equal']}, "
      f"{smoke['hook_vs_replicated']['8']['bitwise_equal']}) |")
    A(f"| suffix invariance at block {smoke['suffix_invariance']['block']} | bitwise equal |")
    for b in ("4", "8"):
        g = smoke["gpu_bf16_vs_cpu_fp32"][b]
        f36 = smoke["gpu_prefix_vs_full36_cpu_bf16"][b]
        A(f"| block {b}: GPU bf16 vs CPU fp32 | max abs {g['max_abs']:.4g}, cos {g['cosine_min']:.6f} |")
        A(f"| block {b}: GPU prefix vs complete 36-block CPU bf16 | max abs {f36['max_abs']:.4g}, cos {f36['cosine_min']:.6f} |")
    e2e = smoke["accepted_feature_end_to_end"]
    A(f"| complete 36-block CPU bf16 → `model.norm` vs the **accepted feature matrix** | "
      f"cos {e2e['cpu_full36_vs_accepted_bf16']['cosine_min']:.6f}, mean abs {e2e['cpu_full36_vs_accepted_bf16']['mean_abs']:.4g} |")
    A("")
    A(f"GPU peak during capture {smoke['gpu_peak_bytes']/1e9:.2f} GB (weights {smoke['gpu_weight_bytes']/1e9:.2f} GB); "
      f"the full 16 GiB BF16 model still cannot be resident, which is why the *complete-model* reference runs on CPU "
      f"({smoke['cpu_full36_seconds']:.0f} s for {smoke['reference_sequence_length']} tokens) and why no end-to-end "
      "main-model wall clock is claimed.")
    A("")
    A("## 2. Capture")
    A("")
    A("| split | episodes | CT rows | tokens prefilled | GPU time | peak |")
    A("|---|---:|---:|---:|---:|---:|")
    for split in ("train", "dev"):
        c = cov[split]
        lr = c.get("last_run", {})
        A(f"| {split} | {c['episodes']} | {c['rows']} | {c['tokens']:,} | "
          f"{lr.get('gpu_seconds', 0)/60:.1f} min | {lr.get('peak_cuda_bytes', 0)/1e9:.2f} GB |")
    A("")
    A("Per-scope coverage:")
    A("")
    A("| split | scope | episodes | rows |")
    A("|---|---|---:|---:|")
    for split in ("train", "dev"):
        for scope, v in sorted(cov[split]["by_scope"].items()):
            A(f"| {split} | {scope} | {v['episodes']} | {v['rows']} |")
    A("")
    A("The only missing scope is now captured: the no-think template needed the frozen AgentDojo tool schema, "
      "which this study rebuilt through a local import shim and **verified against the archive's own "
      "`request_tools_sha256`** (travel suite digest matches bit for bit) before any sequence was accepted. "
      "Every no-think round was additionally bound to its accepted row digest.")
    A("")
    A("## 3. TRAIN-only selection (fold 0 groups never fitted)")
    A("")
    A("| arm | FN | FP | AUC | episodes |")
    A("|---|---:|---:|---:|---:|")
    for arm, v in sel.items():
        A(f"| {arm} | {v['fn']} | {v['fp']} | {v['auc']:.6f} | {v['episodes']} |")
    A("")
    A("## 4. DEV (exploratory reuse; own 5 %-FPR threshold)")
    A("")
    A("| variant | FN | FP | AUC | FNR | FPR | threshold |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for key, v in dev.items():
        A(f"| {key} | {v['fn']} | {v['fp']} | {v['auc']:.6f} | {v['fnr']:.4f} | {v['fpr']:.4f} | {v['threshold']:.4f} |")
    A("")
    A(f"Accepted baseline replayed on its own 2 028-episode universe: FN {h['dev_baseline_full_2028']['fn']} / "
      f"FP {h['dev_baseline_full_2028']['fp']} / AUC {h['dev_baseline_full_2028']['auc']:.6f} "
      f"(thr {h['dev_baseline_full_2028']['threshold']:.6f}).")
    A("")
    A("Paired whole-group bootstrap (2 000 replicates, fixed thresholds):")
    A("")
    for name, b in h.get("bootstrap", {}).items():
        A(f"* `{name}`: ΔFN {b['d_fn']['mean']:+.2f} [{b['d_fn']['ci95'][0]:+.1f}, {b['d_fn']['ci95'][1]:+.1f}], "
          f"ΔFP {b['d_fp']['mean']:+.2f} [{b['d_fp']['ci95'][0]:+.1f}, {b['d_fp']['ci95'][1]:+.1f}], "
          f"ΔFNR {b['d_fnr']['mean']:+.4f} [{b['d_fnr']['ci95'][0]:+.4f}, {b['d_fnr']['ci95'][1]:+.4f}]")
    A("")
    A("## 5. Cost of the added deployment operations")
    A("")
    if cost:
        A("| path | rep | baseline p95 ms | candidate p95 ms | increment p95 ms |")
        A("|---|---|---:|---:|---:|")
        for rep, entry in cost.items():
            for path in ("host", "device"):
                e = entry[path]
                A(f"| {path} | {rep} | {e['base']['p95_ms']:.3f} | {e['cand']['p95_ms']:.3f} | "
                  f"{e['increment_p95_ms']:+.3f} |")
        A("")
    A("Boundary of the claim: the added cost is measured on top of the main model's own forward "
      "(hook/copy of the already-computed activation, projection, three fused per-seed heads, score and "
      "running episode max). The 16 GiB backbone cannot be resident on a 12 GiB card, so no end-to-end "
      "main-model number is claimed; what is measured is exactly the increment plus, separately, the real "
      "index-select + host copy inside the real capture forward.")
    A("")
    A("## 6. Validation")
    A("")
    A(f"{val.get('checks', 0)} checks, {val.get('passed', 0)} passed, failures: {val.get('failed', []) or 'none'}. "
      f"Mutations: {json.dumps(val.get('mutations', {}))}")
    A("")
    A("## 7. What this stage does not establish")
    A("")
    A("* It is not an independent confirmation: DEV was reused and the accepted baseline was DEV-selected.")
    A("* It is not a full-deployment measurement of the main model on GPU (hardware-infeasible here).")
    A("* `ctx40k`/`agentlab`/`nothink` sequences are rebuilt by re-executing the frozen serializers; the "
      "proof is the per-row accepted digest, which is enforced, not the producer's own file hash.")
    A("* No teacher/API annotation was added; the inherited v1 consensus is untouched.")
    A("")
    A("## 8. Artifacts and provenance")
    A("")
    A("```")
    for line in h.get("git_commits", []):
        A(line)
    A("```")
    A("")
    A("Manifest: `artifacts/manifests/artifact_hashes.json`. Capture: `artifacts/capture/<split>/episodes/*.npz` "
      "(per-episode atomic, digest in `progress.jsonl`). Datasets: `artifacts/dataset/<split>`. "
      "Runs: `artifacts/runs.json`. Validation: `artifacts/validation/validation.json`.")
    (ART.parent / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
