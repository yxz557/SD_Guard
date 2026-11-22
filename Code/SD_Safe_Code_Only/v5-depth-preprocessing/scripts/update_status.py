"""Continuously refresh STATUS.json from real progress files (no invented numbers)."""
import json, os, subprocess, time
from pathlib import Path

ROOT = Path('/root/sdsafe-compute/v5-depth')
ART = ROOT / 'artifacts'

def read_progress(split):
    p = ART / 'capture' / split / 'progress.jsonl'
    rows = []
    if p.is_file():
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows

def pending_sync(split):
    state_p = ART / '_sync' / 'state.json'
    state = json.loads(state_p.read_text()) if state_p.is_file() else {}
    d = ART / 'capture' / split / 'episodes'
    n = 0
    for f in d.glob('*.npz'):
        rel = str(f.relative_to(ROOT))
        if rel not in state:
            n += 1
    return n

def gpu():
    try:
        out = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True).strip()
        return out
    except Exception:
        return '?'

def main():
    while True:
        st = json.loads((ROOT/'STATUS.json').read_text()) if (ROOT/'STATUS.json').is_file() else {}
        st['updated'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        st['gpu_state'] = gpu()
        cov = {}
        for split, universe, target in (('train', 8229, 122566048), ('dev', 2028, 26841965)):
            rows = read_progress(split)
            done = len([r for r in rows if r.get('npz_sha256')])
            tok = sum(int(r.get('tokens',0)) for r in rows)
            el = 0.0
            if rows:
                t0 = rows[0].get('wall_seconds'); el = sum(float(r.get('wall_seconds',0)) for r in rows)
            rate = tok / el if el > 0 else 0.0
            cov[split] = {'done': done, 'universe': universe, 'remaining': universe-done,
                          'tokens_done': tok, 'tokens_target': target,
                          'tokens_per_second': round(rate,1),
                          'eta_seconds': int((target-tok)/rate) if rate>0 and done>0 and done<universe else 0,
                          'pending_sync_files': pending_sync(split)}
        st['capture'] = cov
        pidf = ROOT/'logs'/'capture_train.pid'
        st['active_pid'] = int(pidf.read_text().strip()) if pidf.is_file() else None
        st['stop_command'] = "pkill -TERM -f 'sdsafe_v5[.]capture_depth' ; # resume: bash logs/start_capture_train.sh"
        (ROOT/'STATUS.json').write_text(json.dumps(st, indent=2) + '\n')
        time.sleep(60)

if __name__ == '__main__':
    main()
