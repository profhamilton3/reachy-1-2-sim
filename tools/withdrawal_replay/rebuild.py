"""Rebuild a recorded lift's release state in the offline world (#172).

Three things are rebuilt, each checked against the recording:

  * THE PLAN.  `plan_crane_pick` re-run on the preflight's object pose with
    the host mirror of the simulator's IK (`tests/unit/host_ik_arm.py`), then
    every recorded correction re-applied with `replan`.  The grasp-depth
    MEASURE's commanded pose must come out within 0.01 deg (it comes out
    within ~0.001).
  * THE COMMANDED POSE AT RELEASE.  The lowering is a dense line from the
    lift's top rung down the ladder; the point it stopped at is not recorded,
    so it is the point on that line which, held offline, settles nearest the
    recorded joints.  `cmd_fit_deg` says how near.
  * THE PHYSICS.  Every joint, its velocity and every object pose written
    into the offline `SimulationCore`, the arm commanded to that pose, the
    gripper to its close angle, and the per-joint compliance as recorded.

Nothing is stepped until an option asks for it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping

import numpy as np

from . import _REPO
from reachy_ai.motion import grasp_alignment as G  # noqa: E402
from reachy_ai.tasks import crane_pick_live as CP  # noqa: E402

from .record import OID, corrections  # noqa: E402

MODEL = str(_REPO / "native_mujoco" / "model" / "reachy_1_2.xml")
SCENE = str(_REPO / "scenes" / "FWDCenterLabSivaPool.yaml")
ARM = list(CP.R_ARM_JOINTS)
SPEC, LIM = CP.CraneSpec(), CP.CraneLimits()


class World:
    """The offline world, posed from a recorded state and stepped on demand."""

    def __init__(self) -> None:
        import mujoco
        from simulation_core import SimulationCore

        self._mj = mujoco
        self.core = SimulationCore.from_paths(MODEL, SCENE)
        m = self.m = self.core.model
        self.d = self.core.data
        self.jadr: Dict[str, tuple] = {}
        self.free: Dict[str, tuple] = {}
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                self.jadr[name] = (int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]))
            elif m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j]))
                self.free[body] = (int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]))
        self._pads = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
                      for b in ("r_gripper_thumb", "r_gripper_finger")}
        self._cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OID)

    def pose(self, state: Mapping[str, Any], velocities: bool = True) -> None:
        d = self.d
        d.qvel[:] = 0.0
        for j in state["joints"]:
            if j["name"] in self.jadr:
                qa, da = self.jadr[j["name"]]
                d.qpos[qa] = j["position_rad"]
                if velocities:
                    d.qvel[da] = j.get("velocity_rad_s", 0.0)
        for o in state["objects"]:
            if o["object_id"] in self.free:
                qa, _ = self.free[o["object_id"]]
                d.qpos[qa:qa + 3] = o["pos_xyz"]
                d.qpos[qa + 3:qa + 7] = o["quat_wxyz"]
        self._mj.mj_forward(self.m, d)
        self.core.controller.sync_targets_to_current(d)
        for j in state["joints"]:
            try:
                self.core.controller.set_compliant("act_" + j["name"], bool(j["compliant"]))
            except KeyError:
                pass

    def deg(self, joint: str) -> float:
        return math.degrees(float(self.d.qpos[self.jadr[joint][0]]))

    def arm(self) -> List[float]:
        return [self.deg(n) for n in ARM]

    def gripper(self) -> float:
        return self.deg("r_gripper")

    def cube(self):
        qa, _ = self.free[OID]
        return self.d.qpos[qa:qa + 3].copy(), self.d.qpos[qa + 3:qa + 7].copy()

    def target(self, joint: str, deg: float) -> None:
        self.core.controller.set_goal_position("act_" + joint, math.radians(deg))

    def command_arm(self, q) -> None:
        for n, v in zip(ARM, q):
            self.target(n, v)

    def pad_contact_m(self) -> float:
        """Deepest pad-on-cube penetration now (m); 0.0 when they do not touch."""
        deep = 0.0
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1 = int(self.m.geom_bodyid[c.geom1])
            b2 = int(self.m.geom_bodyid[c.geom2])
            if ({b1, b2} & self._pads) and self._cube in (b1, b2) and c.dist <= 0.0:
                deep = max(deep, -float(c.dist))
        return deep

    def grip_force(self) -> float:
        grippers, _ = self.core.snapshot_grippers()
        return next(g["grip_force_n"] for g in grippers if g["side"] == "right")

    def run(self, seconds: float, watch=None) -> None:
        for i in range(int(round(seconds / self.m.opt.timestep))):
            self.core.advance()
            if watch is not None and i % 10 == 0:
                watch()


def _deg(state: Mapping[str, Any], joint: str) -> float:
    return math.degrees(next(j for j in state["joints"]
                             if j["name"] == joint)["position_rad"])


def rebuild_plan(record: Mapping[str, Any]):
    """The plan that flew, its planner (host IK) and the scene model."""
    from host_ik_arm import HostIKArm
    from reachy_ai.motion.kinematics import CartesianPlanner
    from reachy_ai.scene.awareness import SceneModel

    obj = record["preflight_object"]
    scene = SceneModel.from_yaml(SCENE)
    scene.update_poses({k: tuple(v) for k, v in obj["others"].items()})
    scene.update_poses({OID: tuple(obj["position"])})
    planner = CartesianPlanner(HostIKArm(), scene=scene)
    seen = CP.Observation(tuple(obj["position"]), tuple(obj["quat"]), 0.0, False)
    plan = CP.plan_crane_pick(planner, scene, OID, seen)
    for e in corrections(record):
        label = e["label"]
        rung = SPEC.hover_rungs if label == "hover" else int(label.split("+")[1])
        plan = CP.replan(planner, scene, plan, np.asarray(e["shift_mm"]) / 1000.0, rung)
    grasp = next(e for e in record["events"]
                 if e["kind"] == "MEASURE" and e["label"] == "grasp depth")
    err = float(np.abs(np.asarray(plan.rungs[0]) - np.asarray(grasp["q_commanded"])).max())
    if err > 0.01:
        raise AssertionError(f"rebuilt plan differs from the recorded one by {err:.4f} deg")
    return plan, planner, scene, err


class Rebuilt:
    """One lift, rebuilt: plan, commanded pose at release, offline world."""

    def __init__(self, record: Mapping[str, Any]) -> None:
        self.record = record
        self.name = record["name"]
        self.plan, self.planner, self.scene, self.plan_err_deg = rebuild_plan(record)
        self.state = record["release_start"]
        self.g_held = _deg(self.state, "r_gripper")
        self.world = World()
        self.cmd, self.cmd_fit_deg = self._fit_cmd()
        z_cmd = G.hand_frames(self.cmd, SPEC.close_deg).gap_mid[2]
        self.above = max([j for j in range(SPEC.hover_rungs + 1)
                          if G.hand_frames(self.plan.rungs[j], SPEC.close_deg).gap_mid[2]
                          <= z_cmd + 1e-4] or [0])

    def _fit_cmd(self):
        k = SPEC.lift_rungs
        lowering = G.densify([self.plan.rungs[k]] + [self.plan.rungs[j]
                                                     for j in range(k - 1, -1, -1)], 0.05)
        rec = np.asarray([_deg(self.state, n) for n in ARM])
        grasp = next(e for e in self.record["events"]
                     if e["kind"] == "MEASURE" and e["label"] == "grasp depth")
        guess = rec - np.asarray(grasp["joint_offset_deg"])
        near = int(np.argmin([np.abs(np.asarray(q) - guess).max() for q in lowering]))
        best = None
        for i in range(max(0, near - 40), min(len(lowering), near + 41), 4):
            self.hold_at(lowering[i], velocities=False)
            self.world.run(0.4)
            err = float(np.abs(np.asarray(self.world.arm()) - rec).max())
            if best is None or err < best[0]:
                best = (err, i)
        return list(lowering[best[1]]), best[0]

    def hold_at(self, cmd, velocities: bool = True, state=None) -> None:
        """Pose the world at the recorded state (or ``state``), the arm
        commanded to ``cmd`` and the gripper to its close angle."""
        w = self.world
        w.pose(state or self.state, velocities=velocities)
        w.command_arm(cmd)
        w.target("r_gripper", SPEC.close_deg)

    def measured(self, cmd):
        """(measured arm, gripper, object box, joint offset from ``cmd``, object position)."""
        w = self.world
        q = w.arm()
        pos, quat = w.cube()
        live = G.object_box(self.scene.get(OID), pos, quat)
        return q, w.gripper(), live, G.joint_offset(q, cmd), pos
