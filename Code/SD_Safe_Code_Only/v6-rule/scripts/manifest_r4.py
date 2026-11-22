"""SHA manifest + frozen-input verification for runtime-r4 (no writes to frozen files)."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path('/root/sdsafe-compute/v6-param')
OUT = ROOT / 'artifacts' / 'corrected-r3'
R4 = OUT / 'runtime-r4'

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

files = []
for base in [ROOT / 'src' / 'sdsafe_v6', ROOT / 'scripts', ROOT / 'tests']:
    for p in sorted(base.rglob('*.py')):
        if '__pycache__' in str(p):
            continue
        files.append((p, sha(p)))
for p in sorted(R4.rglob('*')):
    if p.is_file():
        files.append((p, sha(p)))
for p in sorted((ROOT / 'logs').glob('r4_*.log')):
    files.append((p, sha(p)))
manifest = {'files': {str(p.relative_to(ROOT)): h for p, h in files}}
(R4 / 'MANIFEST_R4.json').write_text(json.dumps(manifest, indent=1) + '\n')

frozen = json.loads((ROOT / 'artifacts' / 'teacher' / 'FROZEN_MANIFEST.json').read_text())
frozen_caches = frozen.get('frozen_teacher_caches', frozen)
checks = {}
for name, meta in frozen_caches.items():
    p = ROOT / 'artifacts' / 'teacher' / name
    if p.is_file() and isinstance(meta, dict) and 'sha256' in meta:
        checks[f'artifacts/teacher/{name}'] = {'expected': meta['sha256'], 'actual': sha(p),
                                               'ok': sha(p) == meta['sha256']}
for name in ('scaler_corrected_r3.npz', 'fusion_frozen_corrected_r3_S2.json',
             'replay_S2.npz', 'results_corrected_r3.json', 'validation_corrected_r3.json'):
    p = OUT / name
    checks[f'artifacts/corrected-r3/{name}'] = {'actual': sha(p)}
for s in (0, 1, 2):
    for arm in ('S2', 'S0', 'S3'):
        p = OUT / f'ckpt_{arm}_seed{s}.pt'
        checks[f'artifacts/corrected-r3/{p.name}'] = {'actual': sha(p)}
# compare with the frozen r3 handoff values
h = json.loads((ROOT / 'HANDOFF_R3.json').read_text())['sha256']
mismatch = []
for k, v in h.items():
    p = OUT / k
    if p.is_file():
        got = sha(p)
        checks[f'artifacts/corrected-r3/{k}'] = {'expected': v, 'actual': got, 'ok': got == v}
        if got != v:
            mismatch.append(k)
out = {'frozen_check_mismatches': mismatch, 'checks': checks}
(R4 / 'frozen_sha_r4.json').write_text(json.dumps(out, indent=1) + '\n')
print('files', len(manifest['files']), 'frozen mismatches', mismatch)
for k, v in checks.items():
    if 'ok' in v:
        print(k, v['ok'])
