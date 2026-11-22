"""Driver: deterministic picks + one fresh subprocess per (blocks, shape) smoke."""
import json, os, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path('/root/sdsafe-compute/v5-depth')
PREP = ROOT / 'prepared'
OUT = ROOT / 'artifacts' / 'smoke_depth'

def scan(args):
    split, name = args
    z = np.load(PREP / split / name, allow_pickle=True)
    sl = np.asarray(z['seq_len']); rr = np.asarray(z['round_id'])
    j = int(np.argmax(sl))
    return {'split': split, 'episode_id': str(z['episode_id']), 'scope': str(z['scope']),
            'max_len': int(sl[j]), 'best_round': int(rr[j]), 'rounds': int(sl.size),
            'multi': bool(sl.size >= 2)}

def pick(rows):
    rows = sorted(rows, key=lambda r: r['max_len'])
    boundary = rows[-1]
    cand = [r for r in rows if r['multi'] and 4500 <= r['max_len'] <= 9000] or [r for r in rows if r['multi']]
    medium = min(cand, key=lambda r: abs(r['max_len'] - 7000))
    cshort = [r for r in rows if r['multi'] and 120 <= r['max_len'] <= 800] or [r for r in rows if r['multi']]
    short = min(cshort, key=lambda r: abs(r['max_len'] - 260))
    out = {}
    for role, r in (('short', short), ('medium', medium), ('boundary', boundary)):
        out[role] = {'episode_id': r['episode_id'], 'split': r['split'], 'round_id': r['best_round'],
                     'max_len': r['max_len'], 'scope': r['scope']}
    return out

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idx = json.loads((PREP / 'train' / 'index.json').read_text())
    tasks = [('train', n) for n in idx.values()]
    with Pool(16) as p:
        rows = p.map(scan, tasks, chunksize=20)
    picks = pick(rows)
    picks_path = OUT / 'picks.json'
    picks_path.write_text(json.dumps(picks, indent=2))
    print('PICKS', json.dumps(picks), flush=True)
    blocks_list = [int(x) for x in os.environ.get('SMOKE_BLOCKS', '16,24').split(',')]
    shapes = os.environ.get('SMOKE_SHAPES', 'short,medium,boundary,equiv,equiv_boundary').split(',')
    jobs = []
    for b in blocks_list:
        for shape in shapes:
            if shape.startswith('equiv'):
                chunk = os.environ.get('SMOKE_CHUNK', '2048')
                tag = f'_c{chunk}' if chunk else '_mono'
                out = OUT / f'b{b}_{shape}{tag}.json'
                log = OUT / f'b{b}_{shape}{tag}.log'
                cmd = [sys.executable, '-m', 'sdsafe_v5.smoke_depth', '--blocks', str(b), '--shape', shape,
                       '--picks', str(picks_path), '--out', str(out)]
                if chunk:
                    cmd += ['--chunk', chunk]
            else:
                out = OUT / f'b{b}_{shape}.json'
                log = OUT / f'b{b}_{shape}.log'
                cmd = [sys.executable, '-m', 'sdsafe_v5.smoke_depth', '--blocks', str(b), '--shape', shape,
                       '--picks', str(picks_path), '--out', str(out)]
            jobs.append((b, shape, cmd, out, log))
    env = dict(os.environ)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = str(ROOT / 'src')
    env['SDSAFE_UPSTREAM_MIRROR'] = '/dev/shm/sdsafe-gpu-v4/upstream'
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    summary = {}
    for b, shape, cmd, out, log in jobs:
        t0 = time.time()
        with open(log, 'w') as fh:
            rc = subprocess.call(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
        dt = round(time.time() - t0, 1)
        rec = {'blocks': b, 'shape': shape, 'rc': rc, 'seconds': dt, 'out': str(out)}
        if out.is_file():
            d = json.loads(out.read_text())
            rec['sequence_length'] = d.get('sequence_length')
            rec['monolithic'] = d.get('monolithic')
            rec['chunked'] = d.get('chunked')
            rec['equivalence'] = d.get('equivalence')
            rec['equivalence_pass'] = d.get('equivalence_pass')
        summary[f'b{b}_{shape}'] = rec
        print(json.dumps(rec), flush=True)
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))

if __name__ == '__main__':
    main()
