"""Print the one-line FINAL_RUNTIME_JSON gate record from the final cost summary.

The numbers are read from artifacts/corrected-r3/runtime-final/cost_final_summary.json and
accuracy_final.json; nothing is recomputed or adjusted here.
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path('/root/sdsafe-compute/v6-param/artifacts/corrected-r3/runtime-final')
s = json.loads((OUT / 'cost_final_summary.json').read_text())
R = s['modes']['R_gpu']
T = s['modes']['T_gpu']
acc = json.loads((OUT / 'accuracy_final.json').read_text())
par = json.loads((OUT / 'parity_r4.json').read_text())
# scope is valid iff the frozen scope/parity/accuracy contract held in every measured pass:
# full 25003-event parity within the float32 numeric bound, no dropped view/bag/row, 0 fallbacks,
# the frozen tau giving 12/79 in all 9 passes, and graph scores equal to the saved r4 scores
# within float32/fusion noise (the known backend drift, not a semantic change).
scope_valid = bool(
    par['events'] == 25003 and par['episodes'] == 2028 and
    all(par['modes'][m]['bad_rows'] == 0 and par['modes'][m]['bag_bad'] == 0 and
        par['modes'][m]['dense_maxdiff'] == 0.0 and
        par['modes'][m]['logit_maxdiff'] <= 1e-4 and
        par['modes'][m]['fused_maxdiff'] <= 1e-6 and
        par['modes'][m]['graph_fallback'] == 0 for m in ('R', 'T')) and
    R['fusion_fn'] == [12] and R['fusion_fp'] == [79] and
    T['fusion_fn'] == [12] and T['fusion_fp'] == [79] and
    R['stud_maxdiff_vs_saved_max'] <= 1e-4 and R['fused_maxdiff_vs_saved_max'] <= 1e-6 and
    T['stud_maxdiff_vs_saved_max'] <= 1e-4 and T['fused_maxdiff_vs_saved_max'] <= 1e-6
)
rec = {
    'fn': int(R['fusion_fn'][0]),
    'fp': int(R['fusion_fp'][0]),
    'p95_ms': round(float(R['integrable_p95_median']), 4),
    'worst_p95_ms': round(float(R['integrable_p95_worst']), 4),
    'strict_cost_passed': bool(float(R['integrable_p95_worst']) <= 1.0 and
                               float(T['integrable_p95_worst']) <= 1.0),
    'scope_valid': scope_valid,
    'annotations': 0,
}
print('FINAL_RUNTIME_JSON: ' + json.dumps(rec))
