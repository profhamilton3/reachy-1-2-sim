"""#74: the distal servos must not limit-cycle when their force saturates.

Found in the #74 corridor flights (2026-10-09): r_forearm_yaw drifted 24-52 deg
off a commanded 0 while the server had received 0.  Recorded states showed
|velocity| ~18-26 rad/s with the position barely moving, and the force at
+-15 N m -- a period-2 oscillation (the force flips sign every 2 ms step).

Why: these joints carry almost no inertia (forearm yaw ~6e-4 kg m^2).  Under
`implicitfast` the actuator's kv term is integrated implicitly only while the
force is unclamped; once it hits `forcerange` the clamp has no velocity
derivative, kv acts explicitly, and explicit damping is stable only for
dt < 2 I / kv (0.24 ms here, against a 2 ms step).  The joint bang-bangs
between the force limits and its mean drifts.  The wave saturates forearm yaw
on every swing, which is how it got in.

The fix is joint `armature` (a servo's reflected rotor inertia, which the
MJCF had at 0), at least kv * timestep on every distal joint.
"""
from __future__ import annotations

import math
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (_ROOT / "native_mujoco", _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

mujoco = pytest.importorskip("mujoco", reason="native MuJoCo not available")

MODEL = str(_ROOT / "native_mujoco" / "model" / "reachy_1_2.xml")
SCENE = str(_ROOT / "scenes" / "FWDCenterLabSivaPool.yaml")
DISTAL = [f"{s}_{j}" for s in "rl" for j in ("forearm_yaw", "wrist_pitch")]


def _actuator_for(m, joint):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
    return next(a for a in range(m.nu) if m.actuator_trnid[a][0] == jid), jid


@pytest.mark.parametrize("joint", DISTAL)
def test_armature_keeps_saturated_damping_stable(joint):
    m = mujoco.MjModel.from_xml_path(MODEL)
    a, jid = _actuator_for(m, joint)
    kv = -m.actuator_biasprm[a, 2]
    arm = m.dof_armature[m.jnt_dofadr[jid]]
    assert arm >= kv * m.opt.timestep, (joint, arm, kv)


def _world():
    from joint_map import JOINT_TABLE
    from objects import build_scene_model_xml
    from scene_compiler import tracked_object_ids
    from scene_io import load_scene
    from simulation_core import SimulationCore
    doc = load_scene(SCENE)
    core = SimulationCore(mujoco.MjModel.from_xml_string(build_scene_model_xml(doc, MODEL)),
                          tracked_ids=tracked_object_ids(doc))
    for e in JOINT_TABLE:
        if e.sdk_name.startswith("r_"):
            core.controller.set_compliant(e.mjcf_index, False)
    return core, {e.sdk_name: e.mjcf_index for e in JOINT_TABLE}


@pytest.mark.parametrize("joint,start_deg,start_vel", [
    ("r_forearm_yaw", -40.0, -18.0),     # the recorded excursion
    ("r_wrist_pitch", 30.0, 18.0),
])
def test_a_saturated_excursion_returns_to_target_without_chattering(joint, start_deg, start_vel):
    core, idx = _world()
    m, d = core.model, core.data
    i = idx[joint]
    core.controller.sync_targets_to_current(d)
    targets = [d.qpos[k] for k in range(len(idx))]
    core.set_targets(targets)                       # hold the pose; target 0 for `joint`
    d.qpos[i] = math.radians(start_deg)
    d.qvel[m.jnt_dofadr[i]] = start_vel
    mujoco.mj_forward(m, d)
    forces = []
    for _ in range(int(1.0 / m.opt.timestep)):
        core.advance()
        forces.append(d.actuator_force[i])
    assert abs(math.degrees(d.qpos[i] - targets[i])) < 1.0
    tail = forces[-100:]
    flips = sum(1 for f0, f1 in zip(tail, tail[1:]) if f0 * f1 < 0 and abs(f0) > 1 and abs(f1) > 1)
    assert flips == 0, f"{joint} force still flipping sign: {flips} times in the last 100 steps"
