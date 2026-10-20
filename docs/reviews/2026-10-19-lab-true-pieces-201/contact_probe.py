"""What the pads touch while holding (#201): fly one side-on lift and, at every
grip-force read while the hand holds the object (> 1 N), count the contacts
between each pad and the object and record their `dim` and the torque the
pad contacts transmit about the pinch axis.  One row per candidate.

Run: python3 contact_probe.py OID ORIGIN [CANDIDATE ...]  (names in CANDIDATES)
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO, REPO / "src", REPO / "native_mujoco", REPO / "tests" / "unit", HERE):
    sys.path.insert(0, str(p))

from fly_batch import apply_contact  # noqa: E402

CANDIDATES = {"base": {}, "c4": {"condim": 4}, "c6": {"condim": 6},
              "no_mccd": {"multiccd": False}, "c4_no_mccd": {"condim": 4, "multiccd": False}}
PADS = ("r_thumb_col", "r_finger_col")


def probe(oid, origin, cand):
    import mujoco
    from tools.place_sweep import fly as F
    from tools.withdrawal_replay import rebuild

    init, grip = rebuild.World.__init__, rebuild.World.grip_force
    seen = {"n": 0, "per_pad": {p: [] for p in PADS}, "dims": set(), "torsion_nm": []}

    def __init__(self):
        init(self)
        apply_contact(self.m, CANDIDATES[cand])

    def grip_force(self):
        f = grip(self)
        if f > 1.0:
            m, d = self.m, self.d
            body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, oid)
            pads = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, p): p for p in PADS}
            count = {p: 0 for p in PADS}
            tors = 0.0
            for i in range(d.ncon):
                c = d.contact[i]
                g = {int(c.geom1), int(c.geom2)}
                pad = next((pads[x] for x in g if x in pads), None)
                if pad is None or not any(m.geom_bodyid[x] == body for x in g):
                    continue
                count[pad] += 1
                seen["dims"].add(int(c.dim))
                fr = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, fr)
                tors = max(tors, abs(fr[3]))      # torsional: about the contact normal
            seen["n"] += 1
            for p in PADS:
                seen["per_pad"][p].append(count[p])
            seen["torsion_nm"].append(tors)
        return f

    rebuild.World.__init__, rebuild.World.grip_force = __init__, grip_force
    try:
        r = F.fly(oid=oid, origin=origin, dest=None, grasp="best")
    finally:
        rebuild.World.__init__, rebuild.World.grip_force = init, grip
    pp = {p: (min(v), max(v)) if v else None for p, v in seen["per_pad"].items()}
    return dict(oid=oid, origin=origin, candidate=cand, samples=seen["n"],
                contacts_per_pad_min_max=pp, dims=sorted(seen["dims"]),
                torsion_force_max=round(max(seen["torsion_nm"], default=0.0), 4),
                criteria=r.get("criteria"), refused=r.get("refused"))


if __name__ == "__main__":
    oid, origin, *cands = sys.argv[1:]
    for c in cands or list(CANDIDATES):
        print(json.dumps(probe(oid, origin, c)), flush=True)
