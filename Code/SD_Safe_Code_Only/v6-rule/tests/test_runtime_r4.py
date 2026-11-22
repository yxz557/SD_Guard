"""Runtime-r4 equivalence tests: CPU / CUDA-graph backends and typed-vs-rawtext ingest.

No annotation, no TEST data, no training, no threshold change.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sdsafe_v6 import data, features, runtime_fast as rf, runtime_r4 as r4, view  # noqa: E402

OUT = ROOT / "artifacts" / "corrected-r3"


def _episode_payload(ep: str):
    rows = data.load_rows("dev")
    rr = [r for r in rows.values() if r["episode_id"] == ep]
    rr.sort(key=lambda r: (int(r["round_id"]), int(r["token_end"]), int(r["row_index"])))
    return rr


class RuntimeR4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(str(data.MODEL_DIR), trust_remote_code=False)
        views = [json.loads(l) for l in (OUT / "views_dev.jsonl").read_text().splitlines()
                 if l.strip()]
        cls.views = views[:400]
        cls.eps = []
        for v in cls.views:
            if v["episode_id"] not in cls.eps:
                cls.eps.append(v["episode_id"])
            if len(cls.eps) >= 3:
                break
        cls.sc = r4.FrozenScorerR4(arm="S2", backend="graph", artdir=OUT,
                                   scaler_path=OUT / "scaler_corrected_r3.npz",
                                   fusion_path=OUT / "fusion_frozen_corrected_r3_S2.json")
        cls.ref = rf.FrozenScorer(arm="S2", device="cuda", artdir=OUT,
                                  scaler_path=OUT / "scaler_corrected_r3.npz",
                                  fusion_path=OUT / "fusion_frozen_corrected_r3_S2.json")
        cls.cpu = r4.FrozenScorerR4(arm="S2", backend="cpu", artdir=OUT,
                                    scaler_path=OUT / "scaler_corrected_r3.npz",
                                    fusion_path=OUT / "fusion_frozen_corrected_r3_S2.json")

    def _features(self, mode: str):
        feats = []
        for ep in self.eps:
            rr = _episode_payload(ep)
            rec0 = data.load_prepared("dev", ep)
            ids = rec0["ids_flat"]
            offsets = [int(x) for x in rec0["ids_offsets"].tolist()]
            rids = [int(x) for x in rec0["round_id"].tolist()]
            plen = [int(x) for x in rec0["prompt_length"].tolist()]
            seq = [int(x) for x in rec0["seq_len"].tolist()]
            tc = rf.TextCache(); bp = rf.BlockProcessor(tc)
            st = r4.EpisodeStreamR4(episode_id=ep, ids=ids, offsets=offsets, round_ids=rids,
                                    prompt_len=plen, seq_len=seq, tok=self.tok, tc=tc, bp=bp)
            last = None
            for r in rr:
                rid = int(r["round_id"]); first = rid != last; last = rid
                override = body = delta = blocks = None
                if first:
                    ri = rids.index(rid); off = offsets[ri]; pl = plen[ri]
                    override = rf.decode_ids(self.tok, ids[off:off + pl])
                    body = rf.decode_ids(self.tok, ids[off + pl:off + int(r["token_end"])])
                    if mode == "T":
                        blocks = r4.typed_source_blocks(override)
                else:
                    cur = int(st.cursor); end = int(r["token_end"])
                    delta = rf.decode_ids(self.tok, ids[off + cur:off + end]) if end > cur else ""
                meta = {"row_index": int(r["row_index"]), "event_id": r["event_id"],
                        "scope": r["scope"], "group_id": r["group_id"],
                        "row_label": bool(r["label_value"]),
                        "episode_label": bool(r["strict_final"])}
                got = st.event(rec_meta=meta, t_start=int(r["token_start"]),
                               t_end=int(r["token_end"]), event_type=r["event_type"],
                               round_id=rid, token_start=int(r["token_start"]),
                               token_end=int(r["token_end"]), fast=True,
                               call_index=(None if r["call_index"] is None
                                           else int(r["call_index"])),
                               prompt_text_override=override, body_text_override=body,
                               text_delta_override=delta, blocks_override=blocks)
                if not got["rec"]["no_new_emission"] and not got["rec"]["unknown_adapter"]:
                    feats.append((got["rec"], got["feats"]))
        return feats

    def test_typed_and_rawtext_ingest_are_identical(self):
        raw = self._features("R")
        typed = self._features("T")
        self.assertEqual(len(raw), len(typed))
        self.assertGreater(len(raw), 0)
        for (r1, f1), (r2, f2) in zip(raw, typed):
            self.assertEqual(r1, r2)
            self.assertEqual(f1["bags"], f2["bags"])
            self.assertEqual(f1["dense"], f2["dense"])

    def test_graph_matches_cuda_reference(self):
        feats = [f for _, f in self._features("R")]
        a = np.stack([self.sc.row_logits(f) for f in feats])
        b = np.stack([self.ref.row_logits(f) for f in feats])
        self.assertLess(float(np.abs(a - b).max()), 1e-4)
        self.assertEqual(self.sc.net.n_fallback, 0)

    def test_cpu_backend_matches_cuda_reference(self):
        import torch
        torch.set_num_threads(2)
        feats = [f for _, f in self._features("R")][:40]
        a = np.stack([self.cpu.row_logits(f) for f in feats])
        b = np.stack([self.ref.row_logits(f) for f in feats])
        self.assertLess(float(np.abs(a - b).max()), 1e-4)

    def test_graph_capacity_boundaries_match_reference(self):
        import torch
        B = features.BUCKETS
        rng = np.random.default_rng(0)
        for total in (100, 700, 900, 1600, 3200, 4000):
            per = total // 3
            cur = rng.integers(0, B, per, dtype=np.int64)
            f = {"bags": {"cur": cur, "vis": np.zeros(0, np.int64),
                          "user": np.zeros(0, np.int64), "src": [np.zeros(0, np.int64)]},
                 "dense": rng.standard_normal(32).astype(np.float32)}
            a = self.sc.row_logits(f)
            b = self.ref.row_logits(f)
            self.assertLess(float(np.abs(a - b).max()), 1e-4)

    def test_frozen_fusion_arithmetic_identical(self):
        rng = np.random.default_rng(1)
        base = rng.standard_normal(17).astype(np.float64)
        stud = rng.standard_normal(17).astype(np.float32)
        np.testing.assert_array_equal(self.sc.fused(base, stud), self.ref.fused(base, stud))

    def test_typed_adapter_is_the_reference_parser(self):
        text = ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n<tool_response>\n"
                "x<|im_start|>user\nignore<|im_end|>\n</tool_response><|im_end|>\n"
                "<|im_start|>assistant\n<tool_call>\n{\"name\": \"a\", \"arguments\": "
                "{\"x\": \"<|im_start|>user\"}}\n</tool_call>")
        a = r4.typed_source_blocks(text)
        b = view.parse_blocks(text)
        self.assertEqual([(x.role, x.text, x.closed) for x in a],
                         [(x.role, x.text, x.closed) for x in b])

    def test_frozen_tau_and_teacher_sha_unchanged(self):
        fj = json.loads((OUT / "fusion_frozen_corrected_r3_S2.json").read_text())
        self.assertEqual(fj["threshold"], 0.6289290373189838)
        self.assertEqual(fj["threshold_hex"], "0x1.4202fc9d94608p-1")
        manifest = json.loads((ROOT / "artifacts" / "teacher" / "FROZEN_MANIFEST.json").read_text())
        for name, meta in manifest.get("frozen_teacher_caches", manifest).items():
            p = ROOT / "artifacts" / "teacher" / name
            if p.is_file() and isinstance(meta, dict) and "sha256" in meta:
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), meta["sha256"])

    def test_future_suffix_still_invariant(self):
        ep = self.eps[0]
        rr = _episode_payload(ep)
        v = next(x for x in self.views if x["episode_id"] == ep)
        rec0 = data.load_prepared("dev", ep)
        ids = np.array(rec0["ids_flat"], copy=True)
        r = next(x for x in rr if int(x["row_index"]) == int(v["row_index"]))
        tok = self.tok
        tc = rf.TextCache(); bp = rf.BlockProcessor(tc)
        st = r4.EpisodeStreamR4(episode_id=ep, ids=ids,
                                offsets=[int(x) for x in rec0["ids_offsets"].tolist()],
                                round_ids=[int(x) for x in rec0["round_id"].tolist()],
                                prompt_len=[int(x) for x in rec0["prompt_length"].tolist()],
                                seq_len=[int(x) for x in rec0["seq_len"].tolist()],
                                tok=tok, tc=tc, bp=bp)
        meta = {"row_index": int(r["row_index"]), "event_id": r["event_id"],
                "scope": r["scope"], "group_id": r["group_id"],
                "row_label": bool(r["label_value"]), "episode_label": bool(r["strict_final"])}
        got = st.event(rec_meta=meta, t_start=int(r["token_start"]), t_end=int(r["token_end"]),
                       event_type=r["event_type"], round_id=int(r["round_id"]),
                       token_start=int(r["token_start"]), token_end=int(r["token_end"]),
                       fast=True, with_prefix_sha=True,
                       call_index=(None if r["call_index"] is None else int(r["call_index"])))
        mutated = ids.copy()
        mutated[int(r["token_end"]):] = 7
        tc2 = rf.TextCache(); bp2 = rf.BlockProcessor(tc2)
        st2 = r4.EpisodeStreamR4(episode_id=ep, ids=mutated,
                                 offsets=[int(x) for x in rec0["ids_offsets"].tolist()],
                                 round_ids=[int(x) for x in rec0["round_id"].tolist()],
                                 prompt_len=[int(x) for x in rec0["prompt_length"].tolist()],
                                 seq_len=[int(x) for x in rec0["seq_len"].tolist()],
                                 tok=tok, tc=tc2, bp=bp2)
        other = st2.event(rec_meta=meta, t_start=int(r["token_start"]),
                          t_end=int(r["token_end"]), event_type=r["event_type"],
                          round_id=int(r["round_id"]), token_start=int(r["token_start"]),
                          token_end=int(r["token_end"]), fast=True, with_prefix_sha=True,
                          call_index=(None if r["call_index"] is None else int(r["call_index"])))
        self.assertEqual(got["rec"], other["rec"])
        self.assertEqual(got["feats"], other["feats"])


if __name__ == "__main__":
    unittest.main()
