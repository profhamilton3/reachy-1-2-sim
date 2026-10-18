"""Which exits from PRESENT a placed object leaves open (#174, exit check).

A placement ends at PRESENT with the object standing on its destination
cell.  From PRESENT the arm has one measured exit: LOWER_TO_REST (to REST,
the hand on the board's right margin; HOME IS REST since the October layout,
#192, and the rail-pocket exit STOW_FROM_SIDE went with the pocket).  An
object put where the exit passes would make the arm's next request refuse --
or, worse, hit it.  The exit is judged three ways, with the object standing
on the cell (upright, turned as the placement would leave it):

  * GATE -- the executor's own live check (`footprint.route_footprint_clearances`
    against `rig_routes.FOOTPRINT_MARGIN`, the rule `_footprint_refusal`
    applies before it flies the exit);
  * ROUTE -- the whole route, every leg, joint-space lines sampled at <= 2 deg,
    tube hand at the wider aperture, against the object (reported, with 2 cm
    given for comparison: the planned margin the rig corridor was held to);
  * FLIGHT -- the route flown in the offline world (#172: table, rails),
    from PRESENT, waypoint by waypoint at its measured duration: any contact
    between the robot and the object, and how far the object moved.

An exit is VALIDATED when the gate passes and the flight neither touches nor
moves the object (> 1 mm, D6's own bound for other objects).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from . import _REPO

MODEL = str(_REPO / "native_mujoco" / "model" / "reachy_1_2.xml")
SCENE = str(_REPO / "scenes" / "FWDCenterLabSivaPool.yaml")

#: The exits from PRESENT, by the posture they reach.
EXITS = (("rest", "LOWER_TO_REST"),)
#: The object may move this much in the flight and the exit still count.
MOVED_MAX_M = 0.001
_STEP_HZ = 25.0


def _scene_with(oid: str, cell: str, yaw_deg: float):
    from reachy_ai.scene.awareness import SceneModel

    scene = SceneModel.from_yaml(SCENE)
    x, y, z = scene.cell_center(cell)
    pos = (x, y, z + scene.get(oid).size[2] / 2.0)
    q = (math.cos(math.radians(yaw_deg) / 2), 0.0, 0.0, math.sin(math.radians(yaw_deg) / 2))
    scene.update_poses({oid: pos})
    obj = scene.get(oid)
    import dataclasses
    scene._objects[oid] = dataclasses.replace(obj, quat=q)
    return scene, pos, q


def route_waypoints(route: str) -> List[Tuple[str, Dict[str, float], float]]:
    from reachy_ai.motion import rig_routes as R
    return [(w.name, dict(w.pose), float(w.seconds)) for w in R.route_named(route)]


def geometry(oid: str, cell: str, yaw_deg: float, route: str) -> Dict[str, Any]:
    """GATE and ROUTE for one exit, the object standing on ``cell``."""
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.motion.footprint import route_footprint_clearances
    from reachy_ai.motion.kinematics import CartesianPlanner, hand_radius

    scene, _pos, _q = _scene_with(oid, cell, yaw_deg)
    gate = route_footprint_clearances(scene, [route]).get(oid)
    planner = CartesianPlanner(arm=None, scene=scene, side="right")
    poses = [dict(R.PRESENT)] + [p for _n, p, _s in route_waypoints(route)]
    worst, where = math.inf, ""
    for (a, b) in zip(poses, poses[1:]):
        qa = [a[j] for j in R.ARM7]
        qb = [b[j] for j in R.ARM7]
        g = max(a.get("r_gripper", R.OPEN), b.get("r_gripper", R.OPEN), key=hand_radius)
        steps = max(2, math.ceil(max(abs(u - v) for u, v in zip(qa, qb)) / 2.0) + 1)
        c = planner.path_clearances(qa, qb, steps=steps, gripper_deg=g).get(oid)
        if c is not None and c.distance < worst:
            worst, where = c.distance, f"{c.link}"
    return {"gate_m": None if gate is None else gate.distance,
            "gate_ok": gate is None or gate.distance >= R.FOOTPRINT_MARGIN,
            "route_m": worst, "route_link": where}


def fly(oid: str, cell: str, yaw_deg: float, route: str) -> Dict[str, Any]:
    """FLIGHT: the exit flown offline from PRESENT, the object on ``cell``."""
    import mujoco
    from simulation_core import SimulationCore
    from reachy_ai.motion import rig_routes as R

    _scene, pos, quat = _scene_with(oid, cell, yaw_deg)
    core = SimulationCore.from_paths(MODEL, SCENE)
    m, d = core.model, core.data
    mujoco.mj_resetData(m, d)
    jq = {}
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        jq[name] = int(m.jnt_qposadr[j])
    for n, v in R.PRESENT.items():
        if n in jq:
            d.qpos[jq[n]] = math.radians(v)
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, oid)
    fj = int(m.body_jntadr[body])
    qa = int(m.jnt_qposadr[fj])
    d.qpos[qa:qa + 3] = pos
    d.qpos[qa + 3:qa + 7] = quat
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    core.controller.sync_targets_to_current(d)
    for n in R.PRESENT:
        try:
            core.controller.set_compliant("act_" + n, False)
        except KeyError:
            pass
    robot = {b for b in range(m.nbody)
             if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(("r_", "l_"))}
    start = d.qpos[qa:qa + 3].copy()
    contact = {"n": 0, "deepest_m": 0.0, "bodies": set(), "closest_m": math.inf, "closest": ""}
    mine = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == body]
    # Box-box mj_geomDistance reads 0 for boxes that are apart (a known
    # artefact here), so the pads -- the hand's only collision geoms, both
    # boxes -- are measured by box separation from the measured hand instead.
    arm = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in robot
           and int(m.geom_contype[g]) & 2
           and int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX)]
    from reachy_ai.motion import grasp_alignment as G
    shape = _scene.get(oid)
    fromto = np.zeros(6)

    def run(seconds: float) -> None:
        for i in range(int(round(seconds / m.opt.timestep))):
            core.advance()
            if i % 5:
                continue
            if i % 25 == 0:
                deg = [math.degrees(d.qpos[jq[n]]) for n in R.ARM7]
                h = G.hand_frames(deg, math.degrees(d.qpos[jq["r_gripper"]]))
                box = G.object_box(shape, d.qpos[qa:qa + 3], d.qpos[qa + 3:qa + 7])
                for pad, name in ((h.thumb, "r_gripper_thumb"), (h.finger, "r_gripper_finger")):
                    sep = G.obb_separation(pad, box)
                    if sep < contact["closest_m"]:
                        contact["closest_m"], contact["closest"] = float(sep), name
                for ga in arm:
                    for gb in mine:
                        dist = mujoco.mj_geomDistance(m, d, ga, gb, 0.25, fromto)
                        if dist < contact["closest_m"]:
                            contact["closest_m"] = float(dist)
                            contact["closest"] = mujoco.mj_id2name(
                                m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[ga]))
            for k in range(d.ncon):
                c = d.contact[k]
                b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                if body in (b1, b2) and ({b1, b2} & robot):
                    other = b2 if b1 == body else b1
                    contact["n"] += 1
                    contact["deepest_m"] = max(contact["deepest_m"], -float(c.dist))
                    contact["bodies"].add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, other))

    run(0.5)
    cur = {n: v for n, v in R.PRESENT.items()}
    for _name, pose, secs in route_waypoints(route):
        steps = max(1, int(secs * _STEP_HZ))
        a = dict(cur)
        for i in range(1, steps + 1):
            for n, v in pose.items():
                if n in jq:
                    core.controller.set_goal_position("act_" + n, math.radians(
                        a.get(n, v) + i / steps * (v - a.get(n, v))))
            run(1.0 / _STEP_HZ)
        cur.update(pose)
        run(0.5)
    moved = float(np.linalg.norm(d.qpos[qa:qa + 3] - start))
    return {"contacts": contact["n"], "deepest_m": contact["deepest_m"],
            "touched_by": sorted(contact["bodies"]), "moved_m": moved,
            "closest_m": contact["closest_m"], "closest_body": contact["closest"],
            "ok": contact["n"] == 0 and moved <= MOVED_MAX_M}


def exit_check(oid: str, cell: str, yaw_deg: float = 0.0,
               flight: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {"object": oid, "cell": cell.replace("cell_", ""),
                           "yaw_deg": round(yaw_deg, 2), "exits": {}}
    for posture, route in EXITS:
        g = geometry(oid, cell, yaw_deg, route)
        f = fly(oid, cell, yaw_deg, route) if flight else None
        ok = g["gate_ok"] and (f is None or f["ok"])
        out["exits"][posture] = {"route": route, **g, "flight": f, "validated": ok}
    out["allowed"] = [p for p, e in out["exits"].items() if e["validated"]]
    return out
