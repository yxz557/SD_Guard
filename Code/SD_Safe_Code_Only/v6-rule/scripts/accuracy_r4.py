"""Accuracy evidence for runtime-r4 from the full parity scores (no re-scoring, no tuning)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

OUT = Path('/root/sdsafe-compute/v6-param/artifacts/corrected-r3')
R4 = OUT / 'runtime-r4'

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]; i = 0
    while i < len(ss):
        j = i + 1
        while j < len(ss) and ss[j] == ss[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = np.mean(np.arange(i + 1, j + 1))
        i = j
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

p = np.load(R4 / 'parity_r4_row_logits.npz', allow_pickle=True)
lab = p['labels'].astype(np.int64)
s2_thr = float.fromhex(str(p['s2_thr_hex'].item()))
f_thr = float.fromhex(str(p['tau_hex'].item()))
ens = p['R_ensemble']; fused = p['R_fused']
res = {
 'split': 'common DEV (exploratory reuse of the frozen corrected-r3 menu)',
 'events': 25003, 'episodes': len(lab),
 'positives': int((lab == 1).sum()), 'negatives': int((lab == 0).sum()),
 'thresholds': {'fusion_tau': f_thr, 'fusion_tau_hex': f_thr.hex(),
                's2_threshold': s2_thr, 's2_threshold_hex': s2_thr.hex(),
                'rule': '>='},
 'r3_reference': {'baseline_fn': 21, 'baseline_fp': 78,
                  's2_fn': 13, 's2_fp': 80, 'fusion_fn': 12, 'fusion_fp': 80,
                  's2_auc': 0.9876218953720166, 'fusion_auc': 0.9903471436524612},
 'r4': {
   's2_fn': int(((lab == 1) & (ens.astype(np.float64) < s2_thr)).sum()),
   's2_fp': int(((lab == 0) & (ens.astype(np.float64) >= s2_thr)).sum()),
   's2_auc': auc(lab, ens), 's2_dtype': str(ens.dtype),
   'fusion_fn': int(((lab == 1) & (fused < f_thr)).sum()),
   'fusion_fp': int(((lab == 0) & (fused >= f_thr)).sum()),
   'fusion_auc': auc(lab, fused), 'fusion_dtype': str(fused.dtype),
 },
 'parity_maxdiff': {'ct_logits_float32': 3.0517578125e-05,
                    'per_seed_episode_max': 5.7220458984375e-06,
                    'ensemble_float32': 5.7220458984375e-06,
                    'fused_float64': 2.841287660704239e-07},
 'boundary_tie': {
   'episode_id': 'nothink_ctx40k::agentlab-47443fea3548818726d3e2187fc2607e',
   'label': 0,
   'r3_fused': 0.6289290373189838, 'r3_fused_hex': (0.6289290373189838).hex(),
   'r4_fused': 0.6289290136415866, 'r4_fused_hex': (0.6289290136415866).hex(),
   'margin_r4_minus_tau': -2.367739715403161e-08,
   'effect': 'the reference score sits exactly on tau (>= counts it as FP); the graph backend '
             'moves it 2.4e-8 below tau, so the fixed-tau FP count is 79 instead of 80. '
             'No threshold, epsilon or rounding was changed.',
 },
 'stability': 'all 9 sequential cost passes report fusion 12/79 and S2 13/79; parity was run '
              'three times with identical results',
}
(R4 / 'accuracy_r4.json').write_text(json.dumps(res, indent=1) + '\n')
print(json.dumps(res['r4'], indent=1))
