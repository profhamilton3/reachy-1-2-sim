"""Planner-independent reach check for the place-only grasp search (#174).

`attitude.py` asks the crane planner, whose grasp solve is a bounded set of
IK starts (`crane_pick_live.GRASP_STARTS`).  A "no" from it can be the start
set rather than the arm.  This asks the arm: for each attitude, a bounded
least-squares solve inside the joint travel from many random starts, for

  * the PICK -- the jaw-gap centre on the object at the origin cell, the hand
    at the attitude the crane would use there; and, where that exists,
  * the SET-DOWN -- the held object (as the pick would hold it, moved
    `GRASP_SHIFT_M` toward the thumb) standing upright on the destination
    centre, the hand at the same attitude turned by ANY angle about the
    vertical (the object's own heading does not matter).

Each reports the least position error found (0 = reachable).  No motion.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys

import numpy as np

from . import _REPO  # noqa: F401

STARTS = 24
TOL_M = 0.0005

#: The arm's FORWARD branch: shoulder pitch at or below 0 (the arm in front of
#: the torso) and shoulder roll above -90 deg.  The crane's start set, PRESENT
#: and the rig corridor all live in it.  The joint travel also admits the arm
#: thrown back over the top (pitch ~ +90, roll ~ -180), which reaches poses
#: here that are no use to this rig: unconstrained, every "reachable at both
#: cells" answer was that branch.
FORWARD = {0: (-150.0, 0.0), 1: (-90.0, 10.0)}


def _bounds():
    from reachy_ai.motion.kinematics import joint_limits

    lims = joint_limits("right")
    lo = np.array([a for a, _ in lims])
    hi = np.array([b for _, b in lims])
    for i, (a, b) in FORWARD.items():
        lo[i], hi[i] = max(lo[i], a), min(hi[i], b)
    return lo, hi


def _solve(target_gap, rot, grip, lo, hi, rng, starts=STARTS):
    from scipy.optimize import least_squares
    from reachy_ai.motion import grasp_alignment as G
    from reachy_ai.motion.kinematics import _rotvec

    def f(q):
        h = G.hand_frames(q, grip)
        return np.concatenate([target_gap - h.gap_mid, 0.5 * _rotvec(rot @ h.thumb_rot.T)])
    from reachy_ai.motion.kinematics import link_frames

    best, good = None, []
    for _ in range(starts):
        q0 = lo + rng.random(7) * (hi - lo)
        r = least_squares(f, q0, bounds=(lo, hi), xtol=1e-9, ftol=1e-9, max_nfev=400)
        pe = float(np.linalg.norm(r.fun[:3]))
        ae = math.degrees(float(np.linalg.norm(r.fun[3:])) / 0.5)
        score = pe + 0.001 * ae
        if best is None or score < best[0]:
            best = (score, pe, ae, r.x)
        if pe <= TOL_M and ae <= 1.0:
            good.append((pe, ae, r.x))
    if good:
        # Of the poses that reach it, the highest elbow -- the crane's own
        # preference (`plan_crane_pick`), and the family its checks expect.
        pe, ae, q = max(good, key=lambda g: float(link_frames(list(g[2]))[1][2]))
        return pe, ae, q
    return best[1], best[2], best[3]


def one(args):
    oid, origin, dest, face, tilt, lean, seed = args
    from dataclasses import replace
    from reachy_ai.motion import grasp_alignment as G
    from reachy_ai.scene.awareness import SceneModel
    from reachy_ai.tasks import crane_pick_live as CP
    from reachy_ai.tasks import crane_place_live as PL
    from .__main__ import SCENE

    rng = np.random.default_rng(seed)
    lo, hi = _bounds()
    scene = SceneModel.from_yaml(SCENE)
    x, y, z = scene.cell_center(f"cell_{origin}")
    obj = G.object_box(scene.get(oid), (x, y, z + scene.get(oid).size[2] / 2.0),
                       (1.0, 0.0, 0.0, 0.0))
    spec = replace(CP.CraneSpec(), face_axis=face, tilt_deg=tilt, lean_bearing_deg=lean)
    rot = CP.grasp_rotation(obj, spec)
    pe, ae, q = _solve(obj.center, rot, spec.opening_deg, lo, hi, rng)
    row = {"face": face, "tilt": tilt, "lean": lean, "pick_err_mm": round(1000 * pe, 2),
           "pick_att_deg": round(ae, 3)}
    if pe > TOL_M or ae > 1.0:
        row["pick"] = False
        return row
    row["pick"] = True
    h = G.hand_frames(q, spec.opening_deg)
    rot_rel, t_rel = G.grasp_transform(h, obj)
    t_rel = t_rel + PL.GRASP_SHIFT_M * (h.thumb_rot.T @ -h.jaw)
    v = h.thumb_rot.T @ (h.gap_mid - h.thumb.center)
    tx, ty, _ = scene.cell_center(f"cell_{dest}")
    best = None
    for psi in range(-180, 180, 15):
        rd = PL._rotz(psi) @ h.thumb_rot
        axes = rd @ rot_rel
        target = np.array([tx, ty, scene.table_surface_z + float(np.abs(axes[2]) @ obj.half)])
        g = target - rd @ t_rel + rd @ v
        dpe, dae, _dq = _solve(g, rd, spec.opening_deg, lo, hi, rng, starts=8)
        if best is None or dpe + 0.001 * dae < best[0] + 0.001 * best[1]:
            best = (dpe, dae, psi)
        if dpe < TOL_M * 0.2 and dae < 0.05:
            break
    row.update(dest_err_mm=round(1000 * best[0], 2), dest_att_deg=round(best[1], 3),
               dest_psi=best[2], dest=best[0] <= TOL_M and best[1] <= 1.0)
    return row


def main(argv=None) -> int:
    from .attitude import FACES, LEANS, TILTS

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--object", default="red_cube")
    p.add_argument("--from", dest="origin", default="r2c2")
    p.add_argument("--to", dest="dest", default="r2c1")
    p.add_argument("--procs", type=int, default=6)
    a = p.parse_args(argv)
    grid = [(a.object, a.origin, a.dest, f, t, l, i)
            for i, (f, t, l) in enumerate((f, t, l) for f in FACES for t in TILTS for l in LEANS)]
    rows = []
    with mp.Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(one, grid, chunksize=4)):
            rows.append(r)
            if r.get("dest"):
                print("BOTH", r, flush=True)
            if i % 100 == 0:
                print(f"{i}/{len(grid)}", flush=True)
    with open(a.out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print("pick reachable:", sum(r["pick"] for r in rows), "both:",
          sum(bool(r.get("dest")) for r in rows), "of", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())


def full_check(args):
    """An attitude the arm can reach at both cells, run through the whole
    placement preflight, with its own reach solutions as IK seeds (the pick
    start set and the destination seeds) and its best heading tried first."""
    oid, origin, dest, face, tilt, lean, psi, seed = args
    from dataclasses import replace
    from host_ik_arm import HostIKArm
    from reachy_ai.motion import grasp_alignment as G
    from reachy_ai.motion.kinematics import CartesianPlanner
    from reachy_ai.scene.awareness import SceneModel
    from reachy_ai.tasks import crane_pick_live as CP
    from reachy_ai.tasks import crane_place_live as PL
    from .__main__ import SCENE

    rng = np.random.default_rng(seed)
    lo, hi = _bounds()
    scene = SceneModel.from_yaml(SCENE)
    x, y, z = scene.cell_center(f"cell_{origin}")
    pos = (x, y, z + scene.get(oid).size[2] / 2.0)
    scene.update_poses({oid: pos})
    obj = G.object_box(scene.get(oid), pos, (1.0, 0.0, 0.0, 0.0))
    spec = replace(CP.CraneSpec(), face_axis=face, tilt_deg=tilt, lean_bearing_deg=lean)
    rot = CP.grasp_rotation(obj, spec)
    _pe, _ae, qp = _solve(obj.center, rot, spec.opening_deg, lo, hi, rng)
    h = G.hand_frames(qp, spec.opening_deg)
    rot_rel, t_rel = G.grasp_transform(h, obj)
    t_rel = t_rel + PL.GRASP_SHIFT_M * (h.thumb_rot.T @ -h.jaw)
    v = h.thumb_rot.T @ (h.gap_mid - h.thumb.center)
    tx, ty, _ = scene.cell_center(f"cell_{dest}")
    rd = PL._rotz(psi) @ h.thumb_rot
    axes = rd @ rot_rel
    target = np.array([tx, ty, scene.table_surface_z + float(np.abs(axes[2]) @ obj.half)])
    _dpe, _dae, qd = _solve(target - rd @ t_rel + rd @ v, rd, spec.opening_deg, lo, hi, rng)
    planner = CartesianPlanner(HostIKArm(), scene=scene)
    seen = CP.Observation(pos, (1.0, 0.0, 0.0, 0.0), 0.0, False)
    events = []
    row = {"face": face, "tilt": tilt, "lean": lean, "psi_reach": psi,
           "pick_seed": [round(float(a), 3) for a in qp],
           "dest_seed": [round(float(a), 3) for a in qd]}
    try:
        psi0 = PL.bearing_deg((tx, ty)) - PL.bearing_deg(pos[:2])
        pp = PL.plan_crane_place(planner, scene, oid, dest, seen, spec=spec,
                                 pick_starts=list(CP.GRASP_STARTS) + [list(qp)],
                                 dest_seeds=[list(qd)],
                                 headings=[psi0 + d for d in range(-180, 180, 15)
                                           if abs(((psi0 + d) - psi + 180) % 360 - 180) < 31],
                                 on_event=lambda k, **kw: events.append((k, kw)))
        held = pp.checks.get("held", {})
        row.update(feasible=True, psi_deg=round(pp.psi_deg, 2), carry_rung=pp.carry_rung,
                   pick_route=pp.pick.checks.get("pad_route"),
                   pick_q0=[round(a, 2) for a in pp.pick.rungs[0]],
                   dest_q0=[round(a, 2) for a in pp.dest.rungs[0]],
                   held={k: {kk: vv for kk, vv in val.items() if kk in ("table_mm", "others_mm", "arm_mm")}
                         for k, val in held.items()},
                   exits={k: round(1000 * e["clearance_m"], 1) for k, e in pp.checks["exits"].items()},
                   pad_route_out=pp.checks.get("pad_route_out"))
    except CP.CraneRefused as exc:
        picked = any(k == "plan_checked" for k, _ in events)
        row.update(feasible=False, at="destination" if picked else "pick",
                   stage=exc.stage, reason=exc.reason[:220],
                   tried=[kw for k, kw in events if k == "place_carry_refused"][-6:])
    return row


def full_main(argv=None) -> int:
    """``python -c 'from tools.place_sweep.reach import full_main; full_main()' REACH.json OUT.json``:
    the whole preflight for every attitude REACH.json found reachable at both cells."""
    src, out = (argv or sys.argv[1:])[:2]
    rows = [r for r in json.load(open(src)) if r.get("dest")]
    grid = [("red_cube", "r2c2", "r2c1", r["face"], r["tilt"], r["lean"], r["dest_psi"], i)
            for i, r in enumerate(rows)]
    done = []
    with mp.Pool(7) as pool:
        for r in pool.imap_unordered(full_check, grid):
            done.append(r)
            print(("FEASIBLE " if r["feasible"] else "refused  ") + f"{r['face']} {r['tilt']} {r['lean']}"
                  + ("" if r["feasible"] else f" {r['at']} {r['stage']}: {r['reason'][:120]}"),
                  flush=True)
    with open(out, "w") as fh:
        json.dump(done, fh, indent=1, default=str)
    print("feasible:", sum(r["feasible"] for r in done), "of", len(done))
    return 0
