import json, sys
sys.path.insert(0, "/root/sdsafe-compute/v6-param/src")
from sdsafe_v6 import data, sample
quotas = json.loads(sys.argv[1]); out = sys.argv[2]; salt = sys.argv[3]
rows = data.load_rows("train")
s = sample.sample_events(rows, quotas=quotas, salt=salt)
json.dump(s["picked"], open(out, "w"), indent=1)
json.dump({k: v for k, v in s.items() if k != "rows"}, open(out + ".meta.json", "w"), indent=1)
print(json.dumps({"counts": s["counts"], "total": s["total"], "available": s["available"]}))
