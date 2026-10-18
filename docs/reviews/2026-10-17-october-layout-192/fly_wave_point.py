"""Offline physics: WAVE and POINT at the October mount (#192), flown from REST.

The panel's own code paths, unchanged -- `rig_motion.travel` (LIFT_TO_PRESENT
and LOWER_TO_REST with their guards), `rig_motion.wave`, and the motion
worker's `_point_at_target` (hover height, `point_at`'s lift search and escort,
the return to PRESENT) -- on the simulator's MuJoCo model and contact
settings, in FWDCenterLabSivaPool.  Only what they talk to is replaced, as in
`tools/place_sweep/fly.py`: joint goals go to the world's actuators through a
minimum-jerk streamer standing in for the SDK's `goto` (100 Hz), positions come
back from its joints, IK is the host mirror of the simulator's, and
`time.sleep` advances the physics.  No server, SDK or bridge: the method of
`fly_rest.py`, which validated LIFT_TO_PRESENT and LOWER_TO_REST.

Measured through every flight at 100 Hz: the worst MJCF distance from any
robot collision geom to any object on the board (box-box pairs capped at
3 cm, where MuJoCo's distance is exact), every robot contact, and how far
each object moved.

    PYTHONPATH=src python3 docs/reviews/2026-10-17-october-layout-192/fly_wave_point.py . wave
    PYTHONPATH=src python3 docs/reviews/2026-10-17-october-layout-192/fly_wave_point.py . point none
    PYTHONPATH=src python3 docs/reviews/2026-10-17-october-layout-192/fly_wave_point.py . point soda_can
"""
import json
import math
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(sys.argv[1]).resolve()
sys.path[:0] = [str(REPO), str(REPO / "web"), str(REPO / "src"),
                str(REPO / "native_mujoco"), str(REPO / "tests" / "unit")]
import mujoco  # noqa: E402

from tools.place_sweep.fly import OfflineArm, OfflineRobot  # noqa: E402
from tools.withdrawal_replay.rebuild import SCENE, World  # noqa: E402
from reachy_ai.motion import primitives as P  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.scene.awareness import SceneModel  # noqa: E402
from reachy_ai.tasks import rig_motion as M  # noqa: E402

CELLS = [f"cell_r{r}c{c}" for r in (1, 2, 3) for c in (1, 2, 3)]
TICK = 0.01


class Flight:
    def __init__(self, layout):
        self.w = w = World()
        self.m, self.d = w.m, w.d
        self.scene = SceneModel.from_yaml(SCENE)
        w.core.reset()
        for oid, cell in layout.items():
            x, y, z = self.scene.cell_center(cell)
            qa, _ = w.free[oid]
            self.d.qpos[qa:qa + 3] = [x, y, z + self.scene.get(oid).size[2] / 2 + 0.001]
            self.d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
        for n in R.R_JOINTS:
            self.d.qpos[w.jadr[n][0]] = math.radians(R.REST[n])
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)
        w.core.controller.sync_targets_to_current(self.d)
        for n in R.R_JOINTS:
            w.core.controller.set_compliant("act_" + n, False)
        w.run(1.0)
        self.layout = dict(layout)
        self.robot_geoms = [g for g in range(self.m.ngeom) if self.m.geom_contype[g] == 2]
        self.obj_geoms = {oid: [g for g in range(self.m.ngeom)
                                if self.m.geom_bodyid[g] == mujoco.mj_name2id(
                                    self.m, mujoco.mjtObj.mjOBJ_BODY, oid)]
                          for oid in layout}
        self.t = 0.0
        self.reset_watch()

    # -- what is watched ---------------------------------------------------
    def positions(self):
        return {oid: self.d.qpos[self.w.free[oid][0]:self.w.free[oid][0] + 3].copy()
                for oid in self.layout}

    def reset_watch(self):
        self.start = self.positions()
        self.worst = {oid: 9.0 for oid in self.layout}
        self.contacts = set()

    def watch(self):
        box = mujoco.mjtGeom.mjGEOM_BOX
        for oid, gs in self.obj_geoms.items():
            for g in gs:
                for r in self.robot_geoms:
                    # Box-box mj_geomDistance returns a spurious 0 for a large
                    # distmax; capped at 3 cm for those pairs, where it is exact.
                    cap = 0.03 if self.m.geom_type[r] == box == self.m.geom_type[g] else 0.3
                    dd = mujoco.mj_geomDistance(self.m, self.d, r, g, cap, None)
                    if dd < self.worst[oid]:
                        self.worst[oid] = dd
        for c in self.d.contact[:self.d.ncon]:
            names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or
                     mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[g])
                     for g in (c.geom1, c.geom2)]
            if any(n.startswith("r_") for n in names):
                self.contacts.add(tuple(sorted(names)))

    def report(self):
        now = self.positions()
        return {"worst_mjcf_cm": {k: round(100 * v, 1) for k, v in self.worst.items()},
                "moved_mm": {k: round(1000 * float(np.linalg.norm(now[k] - self.start[k])), 2)
                             for k in self.layout},
                "robot_contacts": sorted(self.contacts)}

    # -- the stand-ins ------------------------------------------------------
    def sleep(self, dt):
        n = max(1, int(round(float(dt) / TICK)))
        for _ in range(n):
            self.w.run(TICK)
            self.t += TICK
            self.watch()

    def monotonic(self):
        return self.t

    def move(self, arm, pose, seconds):
        """The SDK's goto: minimum jerk from where each joint is now."""
        names = [n for n in pose if hasattr(arm, n)]
        start = {n: getattr(arm, n).present_position for n in names}
        steps = max(1, int(round(seconds / TICK)))
        for k in range(1, steps + 1):
            s = k / steps
            f = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
            for n in names:
                getattr(arm, n).goal_position = start[n] + f * (pose[n] - start[n])
            self.sleep(TICK)


def run(layout, body):
    f = Flight(layout)
    saved = (time.sleep, time.monotonic, M.sdk_move, P._stream, P.look_at)
    time.sleep, time.monotonic = f.sleep, f.monotonic
    M.sdk_move = f.move
    P._stream = lambda arm, pose, duration: f.move(arm, pose, duration)
    P.look_at = lambda *a, **k: None
    try:
        arm = OfflineArm(f.w)
        robot = OfflineRobot(arm)
        return body(f, arm, robot)
    finally:
        time.sleep, time.monotonic, M.sdk_move, P._stream, P.look_at = saved


def gross_error(arm, target):
    return {n: round(getattr(arm, n).present_position - target[n], 1) for n in R.GROSS_JOINTS}


def leg(f, label, fn):
    f.reset_watch()
    row = {"leg": label}
    try:
        row["result"] = fn()
    except (M.RecoveryNeeded, M.RouteError) as exc:
        row["error"] = str(exc)
    row.update(f.report())
    print(json.dumps(row, default=str), flush=True)
    return row


def wave_body(f, arm, robot):
    leg(f, "LIFT_TO_PRESENT", lambda: M.travel(arm, R.POSTURE_PRESENT, robot=robot))
    leg(f, "WAVE", lambda: (M.wave(arm), gross_error(arm, R.PRESENT),
                            R.posture_of(M.present_pose(arm))))
    leg(f, "LOWER_TO_REST", lambda: (M.travel(arm, R.POSTURE_REST, robot=robot),
                                      gross_error(arm, R.REST)))


def point_body(targets):
    from motion_worker import _point_at_target

    def body(f, arm, robot):
        leg(f, "LIFT_TO_PRESENT", lambda: M.travel(arm, R.POSTURE_PRESENT, robot=robot))
        for kind, what in targets:
            def one():
                live = {oid: tuple(float(v) for v in p) for oid, p in f.positions().items()}
                job = {"scene_file": SCENE, "live": live}
                job["object_id" if kind == "object" else "cell"] = what
                out = _point_at_target(job, robot, arm, lambda *_: None)
                return out, R.posture_of(M.present_pose(arm))
            leg(f, f"POINT {kind} {what}", one)
        leg(f, "LOWER_TO_REST", lambda: (M.travel(arm, R.POSTURE_REST, robot=robot),
                                          gross_error(arm, R.REST)))
    return body


WAVE_LAYOUTS = (
    {"soda_can": "cell_r1c3", "blue_cylinder": "cell_r2c3", "red_cube": "cell_r3c3"},
    {"blue_cylinder": "cell_r1c3", "soda_can": "cell_r2c3", "foam_block": "cell_r3c3"},
    {"soda_can": "cell_r2c2", "blue_cylinder": "cell_r1c1", "pool_cyl_1": "cell_r3c1"},
)

if __name__ == "__main__":
    what = sys.argv[2]
    if what == "wave":
        for layout in WAVE_LAYOUTS:
            print(json.dumps({"layout": layout}), flush=True)
            run(layout, wave_body)
    else:
        oid = sys.argv[3]
        layout = {} if oid == "none" else {oid: "cell_r2c2"}
        targets = [("cell", c) for c in CELLS]
        if oid != "none":
            targets.append(("object", oid))
        print(json.dumps({"layout": layout}), flush=True)
        run(layout, point_body(targets))
