"""Aggregate the 9 sequential cost passes into the runtime-final summary.

Same aggregation as summarize_r4.py; the directory is parameterised so the final round writes
artifacts/corrected-r3/runtime-final/ without overwriting the r4 evidence.
"""
from __future__ import annotations
import json, statistics
from pathlib import Path
import numpy as np

import sys
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else
           '/root/sdsafe-compute/v6-param/artifacts/corrected-r3/runtime-final')
NAME = sys.argv[2] if len(sys.argv) > 2 else 'final'
rows = []
for p in (0, 1, 2):
    d = json.loads((OUT / f'cost_r4_p{p}.json').read_text())
    for r in d['repeats']:
        r['process'] = p
        rows.append(r)

def med(xs):
    xs = [x for x in xs if x is not None]
    return float(statistics.median(xs)) if xs else None

def worst(xs):
    xs = [x for x in xs if x is not None]
    return float(max(xs)) if xs else None

summary = {'passes': len(rows), 'processes': 3, 'repeats_per_process': 3,
           'device': 'cuda:0 RTX 4090', 'threads': 2, 'modes': {}}
KEYS = ('detector_p50_ms', 'detector_p95_ms', 'detector_p99_ms',
        'detector_int_p50_ms', 'detector_int_p95_ms', 'detector_int_p99_ms',
        'full_p50_ms', 'full_p95_ms', 'full_p99_ms', 'view_feat_p95_ms',
        'model_p95_ms', 'shared_p95_ms', 'adapter_p95_ms', 'segment_decode_p95_ms',
        'episode_final_p95_ms', 'throughput_per_s', 'wall_throughput_per_s')
for mode in ('A', 'R', 'T'):
    for variant in ('gpu', 'sync'):
        rs = [r for r in rows if r['mode'] == mode and r['variant'] == variant]
        if not rs:
            continue
        key = f'{mode}_{variant}'
        entry = {'n_passes': len(rs)}
        for k in KEYS:
            vals = [r.get(k) for r in rs if r.get(k) is not None]
            if vals:
                entry[k + '_median'] = med(vals)
                entry[k + '_worst'] = worst(vals)
        entry['s2_fn'] = sorted(set(int(r['s2_fn']) for r in rs))
        entry['s2_fp'] = sorted(set(int(r['s2_fp']) for r in rs))
        entry['fusion_fn'] = sorted(set(int(r['fusion_fn']) for r in rs))
        entry['fusion_fp'] = sorted(set(int(r['fusion_fp']) for r in rs))
        entry['stud_maxdiff_vs_saved_max'] = worst([r['stud_maxdiff_vs_saved'] for r in rs])
        entry['fused_maxdiff_vs_saved_max'] = worst([r['fused_maxdiff_vs_saved'] for r in rs])
        # slices: median over passes of each slice p95
        slices = {}
        names = set()
        for r in rs:
            names.update(r.get('slices', {}))
        for name in sorted(names):
            p95 = [r['slices'][name]['p95_ms'] for r in rs if name in r.get('slices', {})]
            p50 = [r['slices'][name]['p50_ms'] for r in rs if name in r.get('slices', {})]
            n = [r['slices'][name]['n'] for r in rs if name in r.get('slices', {})]
            slices[name] = {'n': n[0] if n else 0, 'p50_ms_median': med(p50),
                            'p95_ms_median': med(p95), 'p95_ms_worst': worst(p95)}
        entry['slices'] = slices
        summary['modes'][key] = entry

# gate evaluation on the honest integrable paths: R (rawtext, parse charged) and T integrated
for key in ('A_gpu', 'A_sync', 'R_gpu', 'R_sync', 'T_gpu', 'T_sync'):
    if key not in summary['modes']:
        continue
    e = summary['modes'][key]
    e['p95_median'] = e.get('detector_p95_ms_median')
    e['p95_worst'] = e.get('detector_p95_ms_worst')
    e['integrable_p95_median'] = e.get('detector_int_p95_ms_median')
    e['integrable_p95_worst'] = e.get('detector_int_p95_ms_worst')
    e['cost_passed_strict_worst_le_1ms'] = bool(e['integrable_p95_worst'] <= 1.0)
    e['cost_passed_median_le_1ms'] = bool(e['integrable_p95_median'] <= 1.0)
summary['gate'] = {
    'definition': 'full detector-added p95 on the honest integrable path: R detector '
                  '(parse/JSON/role/source all charged) or T detector + measured source-adapter '
                  'boundary recovery; strict = worst of 9 passes <= 1.0 ms',
    'A_tokens_only': 'reported, not required to meet 1 ms',
    'T_pure_with_adapter_excluded': 'shown only as an upper bound; the prepared source has no '
                                    'separate controller message list, so pure T is not claimed '
                                    'as an actual integration',
    'borderline': 'final round: R detector p95 median 0.9956 / worst 1.0260 ms and T integrated '
                  'median 1.0025 / worst 1.0136 ms over 9 passes; the 1 ms strict worst-of-9 bar '
                  'is still missed by 1.4-2.6% (no definition was relaxed)',
}
(OUT / 'cost_final_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
print(json.dumps(summary['gate'], indent=1))
for k, e in summary['modes'].items():
    print(k, 'det p95 med/worst', round(e.get('detector_p95_ms_median') or -1, 4),
          round(e.get('detector_p95_ms_worst') or -1, 4),
          'int p95 med/worst', round(e.get('detector_int_p95_ms_median') or -1, 4),
          round(e.get('detector_int_p95_ms_worst') or -1, 4),
          'fn/fp', e['fusion_fn'], e['fusion_fp'])
