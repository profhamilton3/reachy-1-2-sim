"""Search for a grasp attitude for the PLACE ability (#174, step 2).

Written when the lift and the place had different attitudes; since #195 they
share `CraneSpec()` (docs/reviews/2026-10-19-crane-retune-195/ holds the
October search, which drives `one` below).  For a placement, the attitude is
held from the pick to the set-down (turned only about the vertical, so the
object stays upright), so it must suit both cells.  Every candidate on a grid
of (face, tilt, lean) is run through the whole placement preflight,
`crane_place_live.plan_crane_place` -- the pick at the origin with every crane
check, the destination ladder, the carry with the held object, and the exit
check -- for one object and one pair of cells.  No motion.

Run: ``python -m tools.place_sweep.attitude --out ROWS.json [--from r2c2 --to r2c1]``
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from dataclasses import replace

from . import _REPO  # noqa: F401  (paths)

TILTS = [10.0 + 2.5 * i for i in range(17)]          # 10 .. 50 deg
LEANS = [-180.0 + 10.0 * i for i in range(36)]       # -180 .. 170 deg
FACES = ("x", "y")


def one(args):
    oid, origin, dest, face, tilt, lean = args
    from host_ik_arm import HostIKArm
    from reachy_ai.motion.kinematics import CartesianPlanner
    from reachy_ai.scene.awareness import SceneModel
    from reachy_ai.tasks import crane_pick_live as CP
    from reachy_ai.tasks import crane_place_live as PL
    from .__main__ import SCENE

    scene = SceneModel.from_yaml(SCENE)
    x, y, z = scene.cell_center(f"cell_{origin}")
    pos = (x, y, z + scene.get(oid).size[2] / 2.0)
    scene.update_poses({oid: pos})
    planner = CartesianPlanner(HostIKArm(), scene=scene)
    spec = replace(CP.CraneSpec(), face_axis=face, tilt_deg=tilt, lean_bearing_deg=lean)
    seen = CP.Observation(pos, (1.0, 0.0, 0.0, 0.0), 0.0, False)
    events = []
    t0 = time.monotonic()
    row = {"face": face, "tilt": tilt, "lean": lean}
    try:
        pp = PL.plan_crane_place(planner, scene, oid, dest, seen, spec=spec,
                                 on_event=lambda k, **kw: events.append(k))
        held = pp.checks.get("held", {})
        row.update(feasible=True, carry_rung=pp.carry_rung, psi_deg=pp.psi_deg,
                   pad_route_in_mm=pp.pick.checks.get("pad_route", {}).get("route_min_mm")
                   if isinstance(pp.pick.checks.get("pad_route"), dict) else None,
                   pick_route=pp.pick.checks.get("pad_route"),
                   held_min_others_mm=min((v["others_mm"] for v in held.values()), default=None),
                   held_min_arm_mm=min((v["arm_mm"] for v in held.values()), default=None),
                   exits={k: round(1000 * v["clearance_m"], 1) for k, v in pp.checks["exits"].items()},
                   pick_q0=[round(v, 2) for v in pp.pick.rungs[0]],
                   dest_q0=[round(v, 2) for v in pp.dest.rungs[0]])
    except CP.CraneRefused as exc:
        row.update(feasible=False, at="destination" if "plan_checked" in events else "pick",
                   stage=exc.stage, reason=exc.reason[:200])
    except Exception as exc:          # a candidate that breaks the planner is reported
        row.update(feasible=False, at="error", stage=type(exc).__name__, reason=str(exc)[:200])
    row["seconds"] = round(time.monotonic() - t0, 1)
    return row


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--object", default="red_cube")
    p.add_argument("--from", dest="origin", default="r2c2")
    p.add_argument("--to", dest="dest", default="r2c1")
    p.add_argument("--procs", type=int, default=max(1, mp.cpu_count() - 2))
    a = p.parse_args(argv)
    grid = [(a.object, a.origin, a.dest, f, t, l) for f in FACES for t in TILTS for l in LEANS]
    rows = []
    with mp.Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(one, grid, chunksize=4)):
            rows.append(r)
            if r["feasible"]:
                print("FEASIBLE", r["face"], r["tilt"], r["lean"], flush=True)
            if i % 100 == 0:
                print(f"{i}/{len(grid)}", flush=True)
    with open(a.out, "w") as fh:
        json.dump(rows, fh, indent=1, default=str)
    print("feasible:", sum(r["feasible"] for r in rows), "of", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
