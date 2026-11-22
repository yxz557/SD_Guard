"""Small stratified pilot: component cost + CPU/CUDA/CUDA-graph student forward."""
from __future__ import annotations
import json, sys, time, statistics
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sdsafe_v6 import baseline, data, evaluate as ev, runtime_fast, view
from transformers import AutoTokenizer

ART = runtime_fast.ART
OUT = ART / "corrected-r3"

def pct(a,q): return float(np.percentile(a,q)) if len(a) else float('nan')

def main():
    tok = AutoTokenizer.from_pretrained(str(data.MODEL_DIR), trust_remote_code=False)
    views = [json.loads(l) for l in (OUT/"views_dev.jsonl").read_text().splitlines() if l.strip()]
    by_ep={}
    for v in views: by_ep.setdefault(v["episode_id"],[]).append(v)
    for vs in by_ep.values(): vs.sort(key=lambda r:(int(r["round_id"]),int(r["token_end"]),int(r["row_index"])))
    base = ev.load_dev_base(); ep_ids=[str(x) for x in base["episode_id"].tolist()]
    # stratified: 2 eps per scope (cot/ctx40k/nothink), + 2 longest episodes
    scope_of={}
    for v in views: scope_of[v["episode_id"]]=v["scope"]
    picks=[]
    for sc in ("cot","ctx40k","nothink"):
        es=[e for e in ep_ids if scope_of.get(e)==sc]
        picks+=es[:2]
    picks += sorted(ep_ids, key=lambda e:-sum(int(v["prefix_chars"]) for v in by_ep[e]))[:2]
    rows=data.load_rows("dev"); by_rows={}
    for r in rows.values():
        if r["episode_id"] in picks: by_rows.setdefault(r["episode_id"],[]).append(r)
    for vs in by_rows.values(): vs.sort(key=lambda r:(int(r["round_id"]),int(r["token_end"]),int(r["row_index"])))
    tc=runtime_fast.TextCache(); bp=runtime_fast.BlockProcessor(tc)
    feats=[]; comp={"decode":[],"parse":[],"event":[],"model":[]}
    for ep in picks:
        rec0=data.load_prepared("dev",ep)
        ids=rec0["ids_flat"]; offsets=[int(x) for x in rec0["ids_offsets"].tolist()]
        rids=[int(x) for x in rec0["round_id"].tolist()]; plen=[int(x) for x in rec0["prompt_length"].tolist()]
        seq=[int(x) for x in rec0["seq_len"].tolist()]
        st=runtime_fast.EpisodeStream(episode_id=ep,ids=ids,offsets=offsets,round_ids=rids,
                                      prompt_len=plen,seq_len=seq,tok=tok,tc=tc,bp=bp)
        last=None
        for r in by_rows[ep]:
            rid=int(r["round_id"]); first=rid!=last; last=rid
            override=body=delta=blocks=None
            if first:
                ri=rids.index(rid); off=offsets[ri]; pl=plen[ri]
                t=time.perf_counter(); override=runtime_fast.decode_ids(tok,ids[off:off+pl]); comp["decode"].append((time.perf_counter()-t)*1e3)
                t=time.perf_counter(); body=runtime_fast.decode_ids(tok,ids[off+pl:off+int(r["token_end"])]); comp["decode"].append((time.perf_counter()-t)*1e3)
                t=time.perf_counter(); blocks=view.parse_blocks(override); comp["parse"].append((time.perf_counter()-t)*1e3)
            else:
                cur=int(st.cursor); end=int(r["token_end"])
                delta=runtime_fast.decode_ids(tok,ids[off+cur:off+end]) if end>cur else ""
            meta={"row_index":int(r["row_index"]),"event_id":r["event_id"],"scope":r["scope"],
                  "group_id":r["group_id"],"row_label":bool(r["label_value"]),"episode_label":bool(r["strict_final"])}
            t=time.perf_counter()
            got=st.event(rec_meta=meta,t_start=int(r["token_start"]),t_end=int(r["token_end"]),
                         event_type=r["event_type"],round_id=rid,token_start=int(r["token_start"]),
                         token_end=int(r["token_end"]),fast=True,timed=True,
                         call_index=(None if r["call_index"] is None else int(r["call_index"])),
                         prompt_text_override=override,body_text_override=body,
                         text_delta_override=delta,blocks_override=blocks)
            comp["event"].append((time.perf_counter()-t)*1e3)
            if not got["rec"]["no_new_emission"] and not got["rec"]["unknown_adapter"]:
                feats.append(got["feats"])
    print("episodes",len(picks),"scored feats",len(feats))
    for k,a in comp.items(): print(k,"n",len(a),"p50",round(pct(a,50),3),"p95",round(pct(a,95),3))
    # models
    scorer=runtime_fast.FrozenScorer(arm="S2",device="cuda",artdir=OUT,scaler_path=OUT/"scaler_corrected_r3.npz",fusion_path=OUT/"fusion_frozen_corrected_r3_S2.json")
    # warm
    for f in feats[:10]: scorer.row_logits(f)
    ts=[]; outs=[]
    for f in feats:
        t=time.perf_counter(); o=scorer.row_logits(f); ts.append((time.perf_counter()-t)*1e3); outs.append(o)
    print("GPU merged forward p50",round(pct(ts,50),3),"p95",round(pct(ts,95),3))
    # CPU load
    scorer_cpu=runtime_fast.FrozenScorer(arm="S2",device="cpu",artdir=OUT,scaler_path=OUT/"scaler_corrected_r3.npz",fusion_path=OUT/"fusion_frozen_corrected_r3_S2.json")
    for f in feats[:20]: scorer_cpu.row_logits(f)
    import torch; torch.set_num_threads(1)
    ts2=[]; outs2=[]
    for f in feats:
        t=time.perf_counter(); o=scorer_cpu.row_logits(f); ts2.append((time.perf_counter()-t)*1e3); outs2.append(o)
    print("CPU 1thr forward p50",round(pct(ts2,50),3),"p95",round(pct(ts2,95),3))
    d=np.abs(np.stack(outs)-np.stack(outs2)).max(); print("cpu-vs-gpu maxdiff",d)
    torch.set_num_threads(2)
    for f in feats[:20]: scorer_cpu.row_logits(f)
    ts3=[]
    for f in feats:
        t=time.perf_counter(); scorer_cpu.row_logits(f); ts3.append((time.perf_counter()-t)*1e3)
    print("CPU 2thr forward p50",round(pct(ts3,50),3),"p95",round(pct(ts3,95),3))
    out={"episodes":len(picks),"scored_feats":len(feats),
         "components_ms":{k:{"n":len(a),"p50":pct(a,50),"p95":pct(a,95)} for k,a in comp.items()},
         "gpu_merged_forward_ms":{"p50":pct(ts,50),"p95":pct(ts,95)},
         "cpu_1thread_forward_ms":{"p50":pct(ts2,50),"p95":pct(ts2,95)},
         "cpu_2thread_forward_ms":{"p50":pct(ts3,50),"p95":pct(ts3,95)},
         "cpu_vs_gpu_maxdiff":float(d),
         "note":"stratified whole-episode pilot (cot/ctx40k/nothink + 2 longest); "
                "used only to pick the backend, no threshold/model change"}
    (ART/'corrected-r3'/'runtime-r4').mkdir(parents=True,exist_ok=True)
    (ART/'corrected-r3'/'runtime-r4'/'pilot_r4.json').write_text(json.dumps(out,indent=1)+"\n")

if __name__=="__main__": main()
