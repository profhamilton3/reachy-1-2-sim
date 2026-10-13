"""#2: motion driven against the scene, not beside it.

Two integration checks the scene technology had unit coverage for but no
motion-level test of:

(a) PLANNING REFUSES A PATH THROUGH THE TABLE.  `CartesianPlanner.plan_segment`
    against the panel scene (FWDCenterLabSivaPool) raises `CollisionError` for
    a segment whose pad dips under the tabletop -- before any IK is solved, so
    nothing could be commanded from it.

(b) A GRASP MOVES A TRACKED OBJECT UNDER PHYSICS.  Read through
    `ObjectTracker`, the same tracker the native server streams object poses
    from.  The world is compiled the way `native_mujoco/server.py` compiles it
    (`scene_io.load_scene` resolves `extends:`, then `build_scene_model_xml`),
    with the model's default contact settings (no-slip 10, ADR-0005).

    SCOPE, stated so the test is not read as more than it is: it starts from
    the recorded 2026-10-06 crane hold (red_cube already in the right gripper,
    `tests/fixtures/crane/hold_state_2026_10_06.json`) -- the approach and the
    pinch are not simulated here, and #55 owns grasp reliability.  It drives
    `SimulationCore`'s actuator controller directly, not the SDK bridge.  What
    it does establish: the held object rides the hand, and opening the hand
    leaves it resting somewhere else on the board, with nothing else moved.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (_ROOT / "native_mujoco", _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reachy_ai.motion.kinematics import (  # noqa: E402
    CartesianPlanner, CollisionError, IKPolicy,
)
from reachy_ai.motion.rig_routes import OBJECT_DRIFT_TOL  # noqa: E402
from reachy_ai.scene.awareness import SceneModel  # noqa: E402

_SCENE = _ROOT / "scenes" / "FWDCenterLabSivaPool.yaml"
_MODEL = _ROOT / "native_mujoco" / "model" / "reachy_1_2.xml"
_HOLD = _ROOT / "tests" / "fixtures" / "crane" / "hold_state_2026_10_06.json"


# ── (a) planning a below-table path ──────────────────────────────────────────

@pytest.fixture(scope="module")
def planner():
    return CartesianPlanner(arm=None, scene=SceneModel.from_yaml(str(_SCENE)))


def _never_solve(*_a, **_k):
    raise AssertionError("IK was solved for a path the pad check should refuse")


def test_a_segment_through_the_tabletop_is_refused_before_any_ik(planner, monkeypatch):
    surface = planner.scene.table_surface_z
    above = (0.42, -0.10, surface + 0.10)
    below = (0.42, -0.10, surface - 0.06)          # inside the table's footprint
    monkeypatch.setattr(planner, "solve", _never_solve)
    with pytest.raises(CollisionError) as info:
        planner.plan_segment(above, below, 8, [0.0] * 7, ik=IKPolicy.FAST,
                             segment="descend through the board")
    text = str(info.value)
    assert "below_table" in text and "table_top" in text


def test_the_same_descent_stopping_above_the_surface_passes_the_pad_check(planner):
    surface = planner.scene.table_surface_z
    cart = planner.interpolate((0.42, -0.10, surface + 0.10),
                               (0.42, -0.10, surface + 0.03), 8)
    planner.check_collisions(cart)                 # raises on a violation


# ── (b) a physics carry relocates a tracked object ───────────────────────────

def _server_world():
    """The compiled world and tracker, built the way the native server does."""
    import mujoco
    from joint_map import JOINT_TABLE
    from objects import build_scene_model_xml
    from scene_compiler import tracked_object_ids
    from scene_io import load_scene
    from simulation_core import SimulationCore

    doc = load_scene(str(_SCENE))
    model = mujoco.MjModel.from_xml_string(build_scene_model_xml(doc, str(_MODEL)))
    core = SimulationCore(model, tracked_ids=tracked_object_ids(doc))
    return core, JOINT_TABLE


def _load_hold(core, joint_table):
    import mujoco
    fx = json.loads(_HOLD.read_text())
    m, d = core.model, core.data
    for j in fx["joints"]:
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j["name"])
        if i >= 0:
            d.qpos[m.jnt_qposadr[i]] = j["position_rad"]
            d.qvel[m.jnt_dofadr[i]] = j["velocity_rad_s"]
    for o in fx["objects"]:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["object_id"])
        a = m.jnt_qposadr[m.body_jntadr[b]]
        d.qpos[a:a + 3] = o["pos_xyz"]
        d.qpos[a + 3:a + 7] = o["quat_wxyz"]
    mujoco.mj_forward(m, d)
    recorded = {j["name"]: j["position_rad"] for j in fx["joints"]}
    targets = [fx["commanded_targets_rad"].get(e.sdk_name, recorded.get(e.sdk_name, 0.0))
               for e in joint_table]
    for e in joint_table:                          # the arm is holding: stiff, as live
        if e.sdk_name.startswith("r_"):
            core.controller.set_compliant(e.mjcf_index, False)
    core.set_targets(targets)
    return targets


@pytest.fixture(scope="module")
def carried():
    pytest.importorskip("mujoco", reason="native MuJoCo not available")
    core, joint_table = _server_world()
    names = [e.sdk_name for e in joint_table]
    targets = _load_hold(core, joint_table)

    def poses():
        return {p.object_id: p.pos_xyz for p in core.objects.poses(core.data)}

    def run(seconds):
        for _ in range(int(seconds / core.model.opt.timestep)):
            core.advance()

    run(0.5)
    start = poses()
    gripping_at_start = core.snapshot_grippers()[0][0]["grasping"]

    # Carry: sweep the shoulder roll outward and the pitch up a little over 2 s.
    roll, pitch = names.index("r_shoulder_roll"), names.index("r_shoulder_pitch")
    peak_z, steps = start["red_cube"][2], 40
    for k in range(1, steps + 1):
        t = list(targets)
        t[roll] -= math.radians(12) * k / steps
        t[pitch] -= math.radians(4) * k / steps
        core.set_targets(t)
        run(0.05)
        peak_z = max(peak_z, poses()["red_cube"][2])
    run(0.5)
    in_hand = poses()

    t[names.index("r_gripper")] = math.radians(-45)    # negative opens
    core.set_targets(t)
    run(1.5)
    return dict(start=start, in_hand=in_hand, end=poses(), peak_z=peak_z,
                gripping_at_start=gripping_at_start,
                gripping_at_end=core.snapshot_grippers()[0][0]["grasping"],
                surface=SceneModel.from_yaml(str(_SCENE)).table_surface_z,
                table=SceneModel.from_yaml(str(_SCENE)).table)


def test_the_tracker_sees_the_held_object(carried):
    assert "red_cube" in carried["start"]
    assert carried["gripping_at_start"] is True
    fx = json.loads(_HOLD.read_text())["objects"][0]["pos_xyz"]
    assert math.dist(carried["start"]["red_cube"], fx) < 0.002   # held, not dropped


def test_the_held_object_rides_the_hand(carried):
    assert math.dist(carried["start"]["red_cube"], carried["in_hand"]["red_cube"]) > 0.05
    assert carried["peak_z"] - carried["start"]["red_cube"][2] > 0.02


def test_releasing_leaves_it_somewhere_else_on_the_board(carried):
    end, start = carried["end"]["red_cube"], carried["start"]["red_cube"]
    assert carried["gripping_at_end"] is False
    assert math.hypot(end[0] - start[0], end[1] - start[1]) > 0.05
    assert end[2] == pytest.approx(carried["surface"] + 0.03, abs=0.005)  # resting, 6 cm cube
    t = carried["table"]
    assert abs(end[0] - t.center[0]) < t.size[0] / 2
    assert abs(end[1] - t.center[1]) < t.size[1] / 2


def test_nothing_else_on_the_board_moved(carried):
    for oid, p in carried["start"].items():
        if oid != "red_cube":
            assert math.dist(p, carried["end"][oid]) < OBJECT_DRIFT_TOL / 20, oid
