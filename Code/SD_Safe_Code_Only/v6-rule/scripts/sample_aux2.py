import json, sys
sys.path.insert(0, "/root/sdsafe-compute/v6-param/src")
from sdsafe_v6 import data, sample
rows = data.load_rows("train")
used=set()
for v in json.load(open("/root/sdsafe-compute/v6-param/artifacts/sample_aux.json")).values(): used |= set(v)
quotas = {"pos_call": 500, "pos_text": 500, "pre_call": 500, "neg_call": 500, "neg_text": 500}
s = sample.sample_events(rows, quotas=quotas, salt="v6param-aux2", max_per_group=8, max_per_episode=1,
                         exclude_row_index=used)
picked = s["picked"]
json.dump(picked, open("/root/sdsafe-compute/v6-param/artifacts/sample_aux2.json", "w"), indent=1)
json.dump({"counts": {k: len(v) for k, v in picked.items()}, "total": sum(len(v) for v in picked.values()),
           "salt": "v6param-aux2", "excluded_previous": len(used), "available": s["available"]},
          open("/root/sdsafe-compute/v6-param/artifacts/sample_aux2.meta.json", "w"), indent=1)
print({k: len(v) for k, v in picked.items()}, sum(len(v) for v in picked.values()))
