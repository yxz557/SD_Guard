import json, sys
sys.path.insert(0, "/root/sdsafe-compute/v6-param/src")
from sdsafe_v6 import data, sample
rows = data.load_rows("train")
quotas = {"pos_call": 600, "pos_text": 600, "pre_call": 500, "neg_call": 350, "neg_text": 350}
s = sample.sample_events(rows, quotas=quotas, salt="v6param-formal", max_per_group=6, max_per_episode=1)
pilot = json.load(open("/root/sdsafe-compute/v6-param/artifacts/sample_pilot400.json"))
merged = {k: sorted(set(v) | set(pilot.get(k, []))) for k, v in s["picked"].items()}
json.dump(merged, open("/root/sdsafe-compute/v6-param/artifacts/sample_aux.json", "w"), indent=1)
json.dump({"counts": {k: len(v) for k, v in merged.items()}, "total": sum(len(v) for v in merged.values()),
           "quotas": quotas, "salt": "v6param-formal+pilot400", "max_per_group": 6,
           "available": s["available"]},
          open("/root/sdsafe-compute/v6-param/artifacts/sample_aux.meta.json", "w"), indent=1)
print(json.dumps({"counts": {k: len(v) for k, v in merged.items()}, "total": sum(len(v) for v in merged.values())}))
