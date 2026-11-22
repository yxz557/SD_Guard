import json, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
D=Path('/root/sdsafe-compute/v5-depth/prepared')
def one(args):
    split,name=args
    z=np.load(D/split/name,allow_pickle=True)
    sl=np.asarray(z['seq_len']); ep=str(z['episode_id']); sc=str(z['scope'])
    return int(sl.max()), int(sl.sum()), ep, sc, name
if __name__=='__main__':
    for split in ('train','dev'):
        idx=json.loads((D/split/'index.json').read_text())
        tasks=[(split,n) for n in idx.values()]
        with Pool(16) as p:
            rows=p.map(one,tasks,chunksize=20)
        rows.sort(reverse=True)
        print(split, 'max:', rows[:5])
        tot=sum(r[1] for r in rows); print(split,'episodes',len(rows),'tokens',tot)
