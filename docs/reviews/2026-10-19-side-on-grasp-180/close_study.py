"""The side-on CLOSE, isolated (#180).  Offline physics, no planner in the loop.

The first side-on flights (flights/first.md) slid in, straddled and closed
on nothing: the 6 cm cube was carried 48 mm forward along the approach, a
pool cylinder tipped off the table.  This study closes on each object from a
level hand with the conditions varied one at a time, to find a close that
holds:

  * the arm is put at a side-on grasp pose (`side_grasp_live` plan, rung 0)
    and settled with the gripper at the opening, nothing in the hand;
  * the object is then stood in the REALISED jaw: its centre on the closing
    line ``thumb_mm`` from the thumb pad's inner face, ``along_mm`` ahead of
    the realised jaw-gap centre along the approach;
  * the gripper closes to ``close_deg`` over ``close_s`` (`primitives.smooth_move`,
    as the executor does), holds 1 s, and the arm climbs the plan's lift
    ladder 5 rungs (5 cm) at the executor's rate.

Reported: the object's displacement during the close (along the approach,
across the jaw, up), the gripper's final angle and grip force, whether both
pads hold it (`grasping`), its tilt, its rise after the lift, and how far it
creeps in the hand over a 2 s hold there (the executor's slip criterion,
`crane_pick_live.SLIP_RESOLUTION_M` = 0.1 mm).

Run: python3 close_study.py --out search/close_study.jsonl   (multiprocess)
"""
import argparse
import itertools
import json
import math
import multiprocessing as mp
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO, REPO / "src", REPO / "native_mujoco", REPO / "tests" / "unit"):
    sys.path.insert(0, str(p))

CELL = "r2c2"


def _plan(oid, cell, opening, height=None, pitch=None):
    from dataclasses import replace
    from host_ik_arm import HostIKArm
    from reachy_ai.motion.kinematics import CartesianPlanner
    from reachy_ai.scene.awareness import SceneModel
    from reachy_ai.tasks import crane_pick_live as CP
    from reachy_ai.tasks import side_grasp_live as SG
    from tools.withdrawal_replay.rebuild import SCENE
    scene = SceneModel.from_yaml(SCENE)
    x, y, z = scene.cell_center(f"cell_{cell}")
    pos = (x, y, z + scene.get(oid).size[2] / 2.0)
    scene.update_poses({oid: pos})
    spec = replace(SG.SideSpec(), opening_deg=opening)
    if height is not None:
        spec = replace(spec, grasp_height_m=height)
    if pitch is not None:
        spec = replace(spec, pitches_deg=(pitch,))
    return SG.plan_side_pick(CartesianPlanner(HostIKArm(), scene=scene), scene, oid,
                             CP.Observation(pos, (1.0, 0, 0, 0), 0.0, False), spec=spec)


def run(case):
    import mujoco
    from reachy_ai.motion import grasp_alignment as G
    from reachy_ai.motion import primitives as P
    from reachy_ai.motion.kinematics import R_ARM_JOINTS
    from reachy_ai.tasks import side_grasp_live as SG
    from tools.place_sweep import fly as F

    oid, opening, thumb_mm, along_mm, close_deg, close_s = (
        case["oid"], case["opening"], case["thumb_mm"], case["along_mm"], case["close_deg"],
        case["close_s"])
    try:
        plan = _plan(oid, case.get("cell", CELL), opening, case.get("height"), case.get("pitch"))
    except Exception as exc:              # a refused plan is a row
        return dict(case, error=f"plan: {exc}")
    w, scene = F.start_world(origin=CELL, oid=oid)
    clock = F._Clock(w)
    saved = P.time.sleep
    P.time.sleep = clock.sleep
    try:
        arm = F.OfflineArm(w)
        d = w.d
        qa, _ = w.free[oid]
        d.qpos[qa:qa + 3] = (0.15, 0.95, 0.2)          # out of the way
        for n, v in zip(R_ARM_JOINTS, plan.rungs[0]):
            d.qpos[w.jadr[n][0]] = math.radians(v)
        d.qpos[w.jadr["r_gripper"][0]] = math.radians(opening)
        d.qvel[:] = 0.0
        mujoco.mj_forward(w.m, d)
        w.core.controller.sync_targets_to_current(d)
        for n, v in zip(R_ARM_JOINTS, plan.rungs[0]):
            arm.__dict__[n].goal_position = v
        arm.r_gripper.goal_position = opening
        w.run(1.5)
        qm = w.arm()
        h = G.hand_frames(qm, w.gripper())
        app = SG.approach_of(h.thumb_rot)
        n = h.closing.copy()
        n[2] = 0.0
        n /= np.linalg.norm(n)
        half = 0.5 * scene.get(oid).size[0]
        c = h.thumb_face + (thumb_mm / 1000.0 + half) * n
        c = c + (along_mm / 1000.0) * app + float((h.gap_mid - c) @ app) * app
        c[2] = scene.table_surface_z + 0.5 * scene.get(oid).size[2]
        yaw = math.atan2(n[1], n[0]) + math.radians(case.get("yaw_deg", 0.0))
        d.qpos[qa:qa + 3] = c
        d.qpos[qa + 3:qa + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        d.qvel[:] = 0.0
        mujoco.mj_forward(w.m, d)
        w.run(0.3)
        p0 = d.qpos[qa:qa + 3].copy()
        P.smooth_move(arm, {"r_gripper": close_deg}, close_s)
        w.run(1.0)
        p1 = d.qpos[qa:qa + 3].copy()
        dp = p1 - p0
        g = w.grippers if hasattr(w, "grippers") else None
        grips, _ = w.core.snapshot_grippers()
        gr = next(x for x in grips if x["side"] == "right")
        for k in range(1, plan.spec.lift_rungs + 1):
            P.execute_trajectory(arm, [plan.up(k)], R_ARM_JOINTS, rate_hz=50)
            P.time.sleep(0.4)
        p2 = d.qpos[qa:qa + 3].copy()

        def in_hand():
            hh = G.hand_frames(w.arm(), w.gripper())
            return hh.thumb_rot.T @ (d.qpos[qa:qa + 3] - hh.thumb.center)
        a0 = in_hand()
        P.time.sleep(2.0)
        hold_slip = float(np.linalg.norm(in_hand() - a0))
        qo = d.qpos[qa + 3:qa + 7]
        tilt = math.degrees(math.acos(max(-1, min(1, 1 - 2 * (qo[1] ** 2 + qo[2] ** 2)))))
        grips2, _ = w.core.snapshot_grippers()
        gr2 = next(x for x in grips2 if x["side"] == "right")
        return dict(case,
                    moved_along_mm=round(1000 * float(dp @ app), 2),
                    moved_across_mm=round(1000 * float(dp @ n), 2),
                    moved_up_mm=round(1000 * float(dp[2]), 2),
                    gripper_after_close_deg=round(w.gripper(), 2),
                    grip_force_n=round(gr["grip_force_n"], 2), grasping=bool(gr["grasping"]),
                    rise_mm=round(1000 * float(p2[2] - p1[2]), 2), tilt_deg=round(tilt, 2),
                    grasping_after_lift=bool(gr2["grasping"]),
                    hold_slip_mm=round(1000 * hold_slip, 3), grip_force_lift_n=round(gr2["grip_force_n"], 1),
                    azimuth_deg=plan.checks["side"]["azimuth_deg"],
                    gap_above_table_mm=round(1000 * (float(h.gap_mid[2]) - scene.table_surface_z), 1))
    finally:
        P.time.sleep = saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--grid", default="base")
    a = p.parse_args()
    cases = []
    if a.grid == "base":
        for oid, openings in (("red_cube", (-65.0,)), ("pool_box_1", (-65.0, -35.0, -30.0)),
                              ("pool_cyl_1", (-65.0, -35.0, -30.0))):
            for opening, thumb, along, close, cs in itertools.product(
                    openings, (0.5, 2.0), (-5.0, 0.0, 5.0), (-5.0,), (0.8, 3.0)):
                cases.append(dict(oid=oid, opening=opening, thumb_mm=thumb, along_mm=along,
                                  close_deg=close, close_s=cs))
    else:
        cases = [json.loads(line) for line in open(a.grid)]
    with mp.Pool(max(1, mp.cpu_count() - 2)) as pool:
        rows = pool.map(run, cases)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
