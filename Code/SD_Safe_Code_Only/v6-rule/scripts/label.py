import json, sys
from pathlib import Path
sys.path.insert(0, "/root/sdsafe-compute/v6-param/src")
from sdsafe_v6 import annotation_guard, teacher
annotation_guard.require_no_annotations('scripts/label.py')
views=Path(sys.argv[1]); cache=Path(sys.argv[2]); workers=int(sys.argv[3]) if len(sys.argv)>3 else 24
recs=[json.loads(l) for l in views.read_text().splitlines()]
led=teacher.label_events(recs, cache_path=cache, workers=workers)
print(json.dumps(led))
