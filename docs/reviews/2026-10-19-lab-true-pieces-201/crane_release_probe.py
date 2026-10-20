"""The crane's release under each pad contact (#201): fly the crane lift at
r3c2 (red_cube) and print what the release measured.
Run: python3 crane_release_probe.py CONDIM [TORSION [PRIORITY]]"""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO, REPO / "src", REPO / "native_mujoco", REPO / "tests" / "unit", HERE):
    sys.path.insert(0, str(p))
from fly_batch import patch_contact
c = {"condim": int(sys.argv[1])}
if len(sys.argv) > 2:
    c["torsion"] = float(sys.argv[2])
if len(sys.argv) > 3:
    c["priority"] = int(sys.argv[3])
patch_contact(c)
from tools.place_sweep import fly as F
r = F.fly(dest=None)
rep = r["outcome"].get("replaced") or {}
print(json.dumps(dict(contact=c, criteria=r["criteria"], replaced=rep,
                      pad_contact_after_release_mm=r["pad_contact_after_release_mm"]), default=str))
