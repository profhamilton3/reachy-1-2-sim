"""Fly the placement offline: `execute_crane_pick(..., place=...)` against the
offline world, through an SDK-shaped arm (#174).

Nothing here is a second implementation of the placement: the crane module
flies, unchanged.  What is replaced is what it talks to --

  * the arm: joint goals go to the offline world's actuators, positions come
    back from its joints, and IK is the host mirror of the simulator's service
    (`tests/unit/host_ik_arm.py`);
  * time: every `time.sleep` in `primitives` advances the physics by that
    long, and the crane's clock reads simulated time;
  * the observation: the object's pose and the right gripper's grip force,
    read from the world as the server streams them, never stale.

So the run is the placement's own sequence on the live model's physics,
without the server, the SDK, the bridge or their latencies -- which is the
difference to expect between this and a live run.  It is the offline
prediction the #174 live check is compared with.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import _REPO  # noqa: F401  (paths)

OID = "red_cube"


class _Clock:
    def __init__(self, world, on_tick=None):
        self.world, self.t, self.on_tick = world, 0.0, on_tick

    def sleep(self, dt: float) -> None:
        n = max(1, int(round(float(dt) / self.world.m.opt.timestep)))
        for i in range(n):
            self.world.core.advance()
            if self.on_tick is not None and i % 10 == 0:
                self.on_tick()
        self.t += n * self.world.m.opt.timestep

    def monotonic(self) -> float:
        return self.t


class _Joint:
    def __init__(self, world, name: str):
        self._w, self._n, self._goal = world, name, world.deg(name)
        self.compliant = False

    @property
    def goal_position(self) -> float:
        return self._goal

    @goal_position.setter
    def goal_position(self, v: float) -> None:
        self._goal = float(v)
        self._w.target(self._n, float(v))

    @property
    def present_position(self) -> float:
        return self._w.deg(self._n)


class OfflineArm:
    def __init__(self, world):
        from host_ik_arm import HostIKArm
        from reachy_ai.motion.kinematics import R_ARM_JOINTS

        for n in list(R_ARM_JOINTS) + ["r_gripper"]:
            setattr(self, n, _Joint(world, n))
        self._ik = HostIKArm()

    def inverse_kinematics(self, M, q0=None):
        return self._ik.inverse_kinematics(M, q0=q0)

    def forward_kinematics(self, q):
        return self._ik.forward_kinematics(q)


class OfflineRobot:
    def __init__(self, arm):
        self.r_arm = arm

    def turn_on(self, *_a):
        pass


#: Where the crane picks the 6 cm cube up at the October mount, and where it
#: sets it down (#195: the one origin, and its clean destination).
ORIGIN = "cell_r3c2"
DEST = "r3c3"


def start_world(cube_xy_offset=(0.0, 0.0), yaw_deg: float = 0.0, origin: str = ORIGIN):
    """The offline world with the arm stiff at PRESENT, the cube upright on
    ``origin`` (moved by ``cube_xy_offset``, turned ``yaw_deg``)."""
    import mujoco
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.scene.awareness import SceneModel
    from tools.withdrawal_replay.rebuild import SCENE, World

    w = World()
    w.core.reset()
    scene = SceneModel.from_yaml(SCENE)
    x, y, z = scene.cell_center(origin)
    pos = (x + cube_xy_offset[0], y + cube_xy_offset[1], z + scene.get(OID).size[2] / 2.0)
    d = w.d
    for n, v in R.PRESENT.items():
        if n in w.jadr:
            d.qpos[w.jadr[n][0]] = math.radians(v)
    qa, _ = w.free[OID]
    d.qpos[qa:qa + 3] = pos
    d.qpos[qa + 3:qa + 7] = (math.cos(math.radians(yaw_deg) / 2), 0.0, 0.0,
                             math.sin(math.radians(yaw_deg) / 2))
    d.qvel[:] = 0.0
    mujoco.mj_forward(w.m, d)
    w.core.controller.sync_targets_to_current(d)
    for n in R.PRESENT:
        try:
            w.core.controller.set_compliant("act_" + n, False)
        except KeyError:
            pass
    w.run(1.0)
    return w, scene


#: The recorded states' and the events' clock origin (an arbitrary wall time).
T0 = 1_000_000.0
#: States are recorded at about this rate, as the live recorder does.
RECORD_HZ = 30.0


def _state(w, step: int, t: float) -> Dict[str, Any]:
    """One state in the live recorder's shape (`states.jsonl`)."""
    d = w.d
    joints = []
    for name, (qa, da) in w.jadr.items():
        joints.append({"name": name, "position_rad": float(d.qpos[qa]),
                       "velocity_rad_s": float(d.qvel[da]), "compliant": False})
    objects = []
    for oid, (qa, _da) in w.free.items():
        objects.append({"object_id": oid, "pos_xyz": [float(v) for v in d.qpos[qa:qa + 3]],
                        "quat_wxyz": [float(v) for v in d.qpos[qa + 3:qa + 7]]})
    return {"type": "state", "sim_step": step, "sim_time_s": t, "host_time": T0 + t,
            "joints": joints, "objects": objects}


def as_task(run: Dict[str, Any], dest: Optional[str]) -> Dict[str, Any]:
    """A flown run as the panel's task record (`panel-task-final.json`):
    what `evaluation.lift_replay.replay_place` / `replay_lift` read."""
    out = run["outcome"]
    t_end = run["states"][-1]["host_time"]
    route = "CRANE_PLACE" if dest else "CRANE_LIFT"
    return {
        "task_id": "offline", "state": "completed",
        "events": [{"role": "user", "text": "offline", "at": T0 - 1.0},
                   {"role": "reachy", "text": "Executing in simulation.", "at": T0},
                   {"role": "reachy", "text": "done", "at": t_end}],
        "proposal": {"object_id": OID, "route": route, "route_version": 1,
                     "expected_start_posture": "present"},
        "execution_evidence": {
            "start_posture": "present", "final_posture": "present",
            "crane": dict(out, events=run["kept_events"],
                          plan={"preflight_passed": True, "plan_sha256": "0" * 64})}}


def fly(cube_xy_offset=(0.0, 0.0), yaw_deg: float = 0.0, dest: Optional[str] = DEST,
        place_spec=None, crane_spec=None, abort_when=None,
        record: bool = False, origin: str = ORIGIN, limits=None) -> Dict[str, Any]:
    """One offline placement from ``origin`` -- or, with ``dest=None``, the
    lift (pick, hold, put back), flown the same way.  ``abort_when(events)``
    (optional) makes the caller's should_abort say yes -- to exercise the
    abort path.  ``limits`` replaces the default `CraneLimits`."""
    from reachy_ai.motion import primitives as P
    from reachy_ai.motion.kinematics import CartesianPlanner
    from reachy_ai.tasks import crane_pick_live as CP
    from reachy_ai.tasks import crane_place_live as PL

    w, scene = start_world(cube_xy_offset, yaw_deg, origin=origin)
    events: List[Tuple[str, Dict[str, Any]]] = []
    contact = {"after_release_mm": 0.0, "released": False, "samples": 0}

    states: List[Dict[str, Any]] = []
    kept: List[Dict[str, Any]] = []
    rec = {"next": 0.0}

    def tick():
        if record and clock.t >= rec["next"]:
            states.append(_state(w, w.core.step, clock.t))
            rec["next"] = clock.t + 1.0 / RECORD_HZ
        if contact["released"]:
            contact["after_release_mm"] = max(contact["after_release_mm"],
                                              1000 * w.pad_contact_m())
            contact["samples"] += 1

    clock = _Clock(w, tick)

    def observe():
        p, q = w.cube()
        f = w.grip_force()
        return CP.Observation(tuple(float(v) for v in p), tuple(float(v) for v in q),
                              float(f), f > 0.05, 0.0)

    def on_event(kind, **kw):
        events.append((kind, kw))
        if record:
            import json as _json
            kept.append(_json.loads(_json.dumps(dict(kind=kind, t=T0 + clock.t, **kw),
                                                default=str)))
        if kind == "RELEASED":
            contact["released"] = True

    saved = (P.time.sleep, CP.time.monotonic, P.look_at, P.stow_from_side)
    P.time.sleep = clock.sleep            # the same `time` module object both use
    CP.time.monotonic = clock.monotonic
    P.look_at = lambda *a, **k: None
    P.stow_from_side = lambda *a, **k: None
    try:
        arm = OfflineArm(w)
        p0, q0 = w.cube()
        scene.update_poses({OID: tuple(float(v) for v in p0)})
        planner = CartesianPlanner(arm, scene=scene)
        kw = {}
        if place_spec is not None:
            kw["place"] = place_spec
        if crane_spec is not None:
            kw["spec"] = crane_spec
        if limits is not None:
            kw["limits"] = limits
        try:
            if dest is None:
                lift = CP.plan_crane_pick(planner, scene, OID, observe(),
                                          **{k: v for k, v in kw.items() if k != "place"})
                plan = None
            else:
                plan = PL.plan_crane_place(planner, scene, OID, dest, observe(), **kw)
                lift = plan.pick
        except CP.CraneRefused as exc:
            return {"refused": f"{exc.stage}: {exc.reason}"}
        start = {oid: w.d.qpos[w.free[oid][0]:w.free[oid][0] + 3].copy() for oid in w.free}
        out = CP.execute_crane_pick(
            OfflineRobot(arm), planner, scene, lift, observe, on_event=on_event,
            should_abort=(lambda: abort_when(events)) if abort_when else None,
            stow=False, place=plan)
    finally:
        P.time.sleep, CP.time.monotonic, P.look_at, P.stow_from_side = saved
    p, q = w.cube()
    wq, xq, yq, zq = q
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (xq * xq + yq * yq)))))
    others = {oid: float(np.linalg.norm(w.d.qpos[w.free[oid][0]:w.free[oid][0] + 3] - v))
              for oid, v in start.items() if oid != OID}
    rep = out.get("replaced") or {}
    return {
        "outcome": out, "events": events,
        "final_object": [float(v) for v in p],
        "error_mm": (1000 * PL.placement_error(p, plan.centre) if plan is not None
                     else 1000 * PL.placement_error(p, p0)),
        "tilt_deg": tilt,
        "others_moved_max_mm": 1000 * max(others.values(), default=0.0),
        "drag_mm": rep.get("release_drag_mm"),
        "pad_contact_after_release_mm": contact["after_release_mm"],
        "withdrawal": (out.get("withdrawal") or {}).get("clean"),
        "way_out": (out.get("withdrawal") or {}).get("way_out"),
        "aborted": (out.get("place") or {}).get("aborted"),
        "setdown_margin": (out.get("place") or {}).get("setdown_margin_deg"),
        "criteria": out.get("criteria"), "final": out.get("final"),
        "sim_s": clock.t,
        "states": states, "kept_events": kept,
    }
