"""python -m tools.place_sweep --out REPORT.md [--json ROWS.json] [--objects a,b]
   [--grasp crane|side|best] [--lift] [--procs N]

``--grasp``: the crane (default, as before #180), the side-on grasp
(`side_grasp_live`), or the planner choice (both planned, the larger margin
wins; each grasp's margin and refusal are recorded).  ``--lift`` sweeps the
lift (pick, hold, put back) at every origin instead of the placements."""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import math
import multiprocessing as mp
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


def sweep_one(oid: str, origin: str, dest: str, grasp: str = "crane"):
    from host_ik_arm import HostIKArm
    from reachy_ai.motion.kinematics import CartesianPlanner
    from reachy_ai.scene.awareness import SceneModel
    from reachy_ai.tasks import crane_pick_live as CP
    from reachy_ai.tasks import crane_place_live as PL
    from reachy_ai.tasks import side_grasp_live as SG

    scene = SceneModel.from_yaml(SCENE)
    x, y, z = scene.cell_center(origin)
    pos = (x, y, z + scene.get(oid).size[2] / 2.0)
    scene.update_poses({oid: pos})
    planner = CartesianPlanner(HostIKArm(), scene=scene)
    seen = CP.Observation(pos, (1.0, 0.0, 0.0, 0.0), 0.0, False)
    t0 = time.monotonic()
    row = {"object": oid, "from": origin.replace("cell_", ""),
           "to": dest.replace("cell_", "") if dest else "", "grasp": grasp}
    events = []
    on_event = lambda k, **kw: events.append((k, kw))  # noqa: E731
    try:
        if grasp == "crane" and dest:
            pp = PL.plan_crane_place(planner, scene, oid, dest, seen, on_event=on_event)
            row.update(used="crane", margin=SG.plan_margin(pp.pick, pp))
        elif grasp == "crane":
            pk = CP.plan_crane_pick(planner, scene, oid, seen, on_event=on_event)
            row.update(used="crane", margin=SG.plan_margin(pk))
            pp = None
        elif grasp == "side" and dest:
            pp = SG.plan_side_place(planner, scene, oid, dest, seen, on_event=on_event)
            row.update(used=SG.GRASP_SIDE, margin=pp.checks["margin"])
        elif grasp == "side":
            pk = SG.plan_side_pick(planner, scene, oid, seen, on_event=on_event)
            row.update(used=SG.GRASP_SIDE, margin=pk.checks["margin"])
            pp = None
        else:
            ch = (SG.plan_place_choice(planner, scene, oid, dest, seen, on_event=on_event)
                  if dest else SG.plan_lift_choice(planner, scene, oid, seen, on_event=on_event))
            row.update(used=ch.used, margins={k: (v or {}).get("margin_mm")
                                              for k, v in ch.margins.items()},
                       limits={k: (v or {}).get("limit") for k, v in ch.margins.items()},
                       refusals=ch.refusals)
            pp = ch.plan if dest else None
        row.update(feasible=True, stage="", reason="")
        if pp is not None:
            row.update(carry_rung=pp.carry_rung, psi_deg=round(pp.psi_deg, 2))
            if pp.pick.kind == "side":
                row["azimuth_deg"] = round(math.degrees(math.atan2(pp.pick.approach[1],
                                                                   pp.pick.approach[0])), 1)
    except CP.CraneRefused as exc:
        picked = any(k == "plan_checked" or (k == "side_pick_candidate" and not kw.get("stage"))
                     for k, kw in events)
        row.update(feasible=False, stage=exc.stage, reason=exc.reason,
                   at="destination" if picked else "pick")
        if exc.detail.get("refusals"):
            row["refusals"] = exc.detail["refusals"]
        tried = [kw for k, kw in events if k == "place_carry_refused"]
        if tried:
            row["carry_refusals"] = tried
    row["seconds"] = round(time.monotonic() - t0, 1)
    return row


def _job(args):
    return sweep_one(*args)


def report(rows, sha, n_cells=None):
    today = datetime.date.today().isoformat()
    out = [f"# Offline feasibility sweep — placement ({today}; #174, side-on #180)", "",
           f"reachy-1-2-sim at `{sha or 'worktree'}`.  Preflight only "
           "(`crane_place_live.plan_crane_place`): no motion, no simulator.  IK: the host "
           "mirror of the simulator's service.  Scene FWDCenterLabSivaPool, the object upright "
           "at the origin cell's centre, nothing else on the board.  Reachable cells: the "
           f"{n_cells if n_cells is not None else 'ones'} without the `no-right-arm` tag.", ""]
    order = lambda o: (OBJECTS.index(o) if o in OBJECTS else len(OBJECTS), o)  # noqa: E731
    for oid in sorted({r["object"] for r in rows}, key=order):
        mine = [r for r in rows if r["object"] == oid]
        ok = [r for r in mine if r["feasible"]]
        out += [f"## {oid}", "", f"Feasible: **{len(ok)} of {len(mine)}** ordered pairs."
                + ("  " + ", ".join(f"{r['from']}→{r['to']}" for r in ok) if ok else ""), "",
                "| from → to | feasible | grasp used | margin mm (side-on / crane) | refused at "
                "| stage | reason |", "|---|---|---|---|---|---|---|"]
        for r in mine:
            if "margins" in r:
                m = " / ".join("—" if r["margins"].get(k) is None else f"{r['margins'][k]:.2f}"
                               for k in ("side-on", "crane"))
            elif r.get("margin"):
                m = f"{r['margin']['margin_mm']:.2f} ({r['margin']['limit']})"
            else:
                m = ""
            reason = r["reason"] or "; ".join(f"{k}: {v}" for k, v in
                                              (r.get("refusals") or {}).items() if v)
            out.append(f"| {r['from']} → {r['to'] or '(lift)'} | "
                       f"{'**yes**' if r['feasible'] else 'no'} | {r.get('used', '')} | {m} | "
                       f"{r.get('at', '')} | {r['stage']} | {reason} |")
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
    p.add_argument("--grasp", default="crane", choices=("crane", "side", "best"))
    p.add_argument("--lift", action="store_true")
    p.add_argument("--procs", type=int, default=1)
    a = p.parse_args(argv)
    cells = reachable_cells(SceneModel.from_yaml(SCENE))
    jobs = []
    for oid in a.objects.split(","):
        for o in cells:
            if a.lift:
                jobs.append((oid, o, "", a.grasp))
                continue
            for d in cells:
                if o != d:
                    jobs.append((oid, o, d, a.grasp))
    rows = []
    # Under the spawn start method the workers cannot see this module as
    # __main__; they import it by its package path.
    from tools.place_sweep.__main__ import _job as job
    with mp.Pool(max(1, a.procs)) as pool:
        for r in pool.imap(job, jobs):
            rows.append(r)
            print(f"{r['object']} {r['from']}->{r['to']}: "
                  f"{('FEASIBLE ' + str(r.get('used'))) if r['feasible'] else r['at'] + ' ' + r['stage'] + ': ' + r['reason']}"
                  f" ({r['seconds']} s)", flush=True)
    pathlib.Path(a.out).write_text(report(rows, a.sha, len(cells)))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(rows, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
