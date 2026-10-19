"""The crane, unchanged (#180): fly the crane lift at r3c2 and the placement
r3c2 -> r3c3 through tools/place_sweep/fly.py and hash the outcome, every event
and the final object pose.  Run once against an export of origin/main
(git archive origin/main src tools tests/unit native_mujoco scenes) and once
against this branch; the hashes must match.
Run: python3 -B crane_identity.py <repo root>"""
import hashlib
import json
import sys

root = sys.argv[1]
sys.path[:0] = [root, root + "/src", root + "/native_mujoco", root + "/tests/unit"]
from tools.place_sweep import fly as F
out = {}
for name, kw in (("lift_r3c2", dict(dest=None)), ("place_r3c2_r3c3", dict())):
    r = F.fly(**kw)
    o = r["outcome"]
    ev = [(k, e) for k, e in r["events"]]
    blob = json.dumps({"outcome": o, "events": ev, "final": r["final_object"]}, sort_keys=True, default=str)
    out[name] = {"sha": hashlib.sha256(blob.encode()).hexdigest()[:16], "criteria": r["criteria"],
                 "error_mm": round(r["error_mm"], 6), "n_events": len(ev)}
print(json.dumps(out))
