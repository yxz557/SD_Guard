from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np, torch
from sdsafe_v6 import argmodel, argfeats, feats, model, train

ART=Path('/root/sdsafe-compute/v6-param/artifacts')
def main():
    dev={k:v for k,v in np.load(ART/'cache_dev_v3.npz',allow_pickle=True).items()}
    sc=np.load(ART/'scaler_v6_v3.npz',allow_pickle=True)
    tr=train.Tensors(dev,sc['mean'],sc['std'],'cuda')
    ar=argmodel.ArgTensors(argfeats.load(ART/'arg_cache_dev_v3.npz'),'cuda')
    z=np.load(ART/'dev_episode_S4_v3.npz',allow_pickle=True)
    ids=[str(x) for x in z['episode_id'].tolist()]
    per=[]
    rows=np.arange(tr.n)
    for seed in (0,1,2):
        p=torch.load(ART/f'ckpt_S4_v3_seed{seed}.pt',map_location='cpu',weights_only=False)
        net=argmodel.ArgStudent(model.ModelConfig(**{k:v for k,v in p['config'].items() if k in model.ModelConfig.__dataclass_fields__}))
        net.load_state_dict(p['state_dict']); net.eval().cuda()
        out=np.zeros(tr.n,np.float32)
        with torch.no_grad():
            for st in range(0,tr.n,512):
                ch=rows[st:st+512]
                out[st:st+ch.size]=net(tr.batch(ch),ar.batch(ch))['main'].float().cpu().numpy()
        ep=train.episode_scores_for(tr,rows,out,ids)
        per.append(ep)
    per=np.stack(per); ens=per.mean(axis=0,dtype=np.float32)
    diff=float(np.abs(ens-np.asarray(z['ensemble'],np.float32)).max())
    print(json.dumps({'sha_ckpt_seed0':hashlib.sha256((ART/'ckpt_S4_v3_seed0.pt').read_bytes()).hexdigest(),
                      'saved_ensemble_max_abs_diff':diff,'per_seed_fn':[int((per[s][np.asarray(z['episode_id'])==e]==e).sum()) if False else None for s in range(3)]},default=float))
    np.savez(ART/'dev_episode_S4_v3_rescored.npz',episode_id=z['episode_id'],per_seed=per,ensemble=ens)
    print('diff',diff)
if __name__=='__main__': main()
