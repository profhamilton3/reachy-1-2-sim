"""python -m tools.place_sweep --out REPORT.md [--json ROWS.json] [--objects a,b]"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import pathlib
import sys
import time

from . import _REPO

SCENE = str(_REPO / "scenes" / "FWDCenterLabSivaPool.yaml")
#: The box-shaped objects of the scene: the 60 mm cube, the 70x70x50 mm foam
#: block, and a 40 mm pool box (all three pool boxes are the same shape).
OBJECTS = ("red_cube", "foam_block", "pool_box_1")


def reachable_cells(scene):
    return [c for c in scene.grid_cells() if "no-right-arm" not in scene.get(c).tags]


def sweep_one(oid: str, origin: str, dest: str):
    from host_ik_arm import HostIKArm
    from reachy_ai.motion.kinematics import CartesianPlanner
    from reachy_ai.scene.awareness import SceneModel
    from reachy_ai.tasks import crane_pick_live as CP
    from reachy_ai.tasks import crane_place_live as PL

    scene = SceneModel.from_yaml(SCENE)
    x, y, z = scene.cell_center(origin)
    pos = (x, y, z + scene.get(oid).size[2] / 2.0)
    scene.update_poses({oid: pos})
    planner = CartesianPlanner(HostIKArm(), scene=scene)
    seen = CP.Observation(pos, (1.0, 0.0, 0.0, 0.0), 0.0, False)
    t0 = time.monotonic()
    row = {"object": oid, "from": origin.replace("cell_", ""), "to": dest.replace("cell_", "")}
    events = []
    try:
        pp = PL.plan_crane_place(planner, scene, oid, dest, seen,
                                 on_event=lambda k, **kw: events.append((k, kw)))
        row.update(feasible=True, carry_rung=pp.carry_rung, psi_deg=round(pp.psi_deg, 2),
                   stage="", reason="")
    except CP.CraneRefused as exc:
        picked = any(k == "plan_checked" for k, _ in events)
        row.update(feasible=False, stage=exc.stage, reason=exc.reason,
                   at="destination" if picked else "pick")
        tried = [kw for k, kw in events if k == "place_carry_refused"]
        if tried:
            row["carry_refusals"] = tried
    row["seconds"] = round(time.monotonic() - t0, 1)
    return row


def report(rows, sha):
    today = datetime.date.today().isoformat()
    out = [f"# Issue #174 offline feasibility sweep — placement ({today})", "",
           f"reachy-1-2-sim at `{sha or 'worktree'}`.  Preflight only "
           "(`crane_place_live.plan_crane_place`): no motion, no simulator.  IK: the host "
           "mirror of the simulator's service.  Scene FWDCenterLabSivaPool, the object upright "
           "at the origin cell's centre, nothing else on the board.  Reachable cells: the seven "
           "without the `no-right-arm` tag.", ""]
    for oid in sorted({r["object"] for r in rows}, key=lambda o: OBJECTS.index(o)):
        mine = [r for r in rows if r["object"] == oid]
        ok = [r for r in mine if r["feasible"]]
        out += [f"## {oid}", "", f"Feasible: **{len(ok)} of {len(mine)}** ordered pairs."
                + ("  " + ", ".join(f"{r['from']}→{r['to']}" for r in ok) if ok else ""), "",
                "| from → to | feasible | refused at | stage | reason |", "|---|---|---|---|---|"]
        for r in mine:
            out.append(f"| {r['from']} → {r['to']} | {'**yes**' if r['feasible'] else 'no'} | "
                       f"{r.get('at', '')} | {r['stage']} | {r['reason']} |")
        why = collections.Counter((r.get("at"), r["stage"], r["reason"]) for r in mine
                                  if not r["feasible"])
        out += ["", "Refusals by reason:", ""]
        for (at, stage, reason), n in why.most_common():
            out.append(f"- {n} × {at} / {stage}: {reason}")
        out.append("")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    from reachy_ai.scene.awareness import SceneModel

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--json")
    p.add_argument("--objects", default=",".join(OBJECTS))
    p.add_argument("--sha", default="")
    a = p.parse_args(argv)
    cells = reachable_cells(SceneModel.from_yaml(SCENE))
    rows = []
    for oid in a.objects.split(","):
        for o in cells:
            for d in cells:
                if o == d:
                    continue
                r = sweep_one(oid, o, d)
                rows.append(r)
                print(f"{oid} {r['from']}->{r['to']}: "
                      f"{'FEASIBLE' if r['feasible'] else r['at'] + ' ' + r['stage'] + ': ' + r['reason']}"
                      f" ({r['seconds']} s)", flush=True)
    pathlib.Path(a.out).write_text(report(rows, a.sha))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(rows, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
