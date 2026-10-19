"""#180: the side-on grasp -- geometry, the opening, the shape rule, the
box-hand preflight and the planner choice.

Preflight only: IK is the host mirror of the simulator's service
(`host_ik_arm.py`), nothing moves.  The planning tests take 15-30 s each (a
side-on plan solves up to 4 candidates' ladders and every check).
"""

import math
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco", "."):
    _abs = os.path.join(_HERE, _p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

pytest.importorskip("scipy")

import numpy as np  # noqa: E402

from host_ik_arm import HostIKArm  # noqa: E402
from reachy_ai.motion import grasp_alignment as G  # noqa: E402
from reachy_ai.motion.kinematics import ArmClearanceError, CartesianPlanner  # noqa: E402
from reachy_ai.scene.awareness import SceneModel, SceneObject  # noqa: E402
from reachy_ai.tasks import crane_pick_live as CP  # noqa: E402
from reachy_ai.tasks import side_grasp_live as SG  # noqa: E402

SCENE = os.path.join(_HERE, "../../scenes/FWDCenterLabSivaPool.yaml")
SPEC = SG.SideSpec()


def _world(oid="red_cube", cell="r2c2", others=None):
    """``oid`` upright on a cell centre (as tools/place_sweep sweep_one does);
    ``others`` {id: (x, y, z)} stand where given."""
    scene = SceneModel.from_yaml(SCENE)
    x, y, z = scene.cell_center(f"cell_{cell}")
    pos = (x, y, z + scene.get(oid).size[2] / 2.0)
    scene.update_poses({oid: pos, **(others or {})})
    seen = CP.Observation(pos, (1.0, 0.0, 0.0, 0.0), 0.0, False)
    return scene, CartesianPlanner(HostIKArm(), scene=scene), seen


def _on(scene, cell, oid):
    x, y, z = scene.cell_center(f"cell_{cell}")
    return (x, y, z + scene.get(oid).size[2] / 2.0)


def _az(plan):
    return math.degrees(math.atan2(plan.approach[1], plan.approach[0]))


# ── Pure geometry ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("az", [0.0, 37.0, 90.0, -120.0])
def test_side_rotation_has_the_thumb_frame_x_up_and_the_approach_at_the_azimuth(az):
    rot = SG.side_rotation(az)
    assert rot[2, 0] > 0.99                                  # thumb-frame x is up
    a = SG.approach_of(rot)
    assert np.degrees(np.arctan2(a[1], a[0])) == pytest.approx(az, abs=1e-6) or \
        abs(abs(az) - 180.0) < 1e-6
    assert a[2] == 0.0 and np.linalg.norm(a) == pytest.approx(1.0)
    assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-9)


def test_pitch_tilts_the_approach_down():
    rot = SG.side_rotation(20.0, 5.0)
    app = -rot[:, 2]
    assert app[2] == pytest.approx(-math.sin(math.radians(5.0)), abs=1e-9)
    assert math.degrees(math.asin(-app[2])) == pytest.approx(5.0, abs=1e-9)
    # still horizontal-azimuth 20 deg
    assert math.degrees(math.atan2(app[1], app[0])) == pytest.approx(20.0, abs=1e-6)
    h = SG.approach_of(rot)
    assert math.degrees(math.atan2(h[1], h[0])) == pytest.approx(20.0, abs=1e-6)


def test_free_gap_at_the_two_flown_openings():
    assert 1000 * SG.free_gap_m(-65.0) == pytest.approx(67.8, abs=0.2)
    assert 1000 * SG.free_gap_m(-68.0) == pytest.approx(69.0, abs=0.2)


def test_opening_and_close_follow_the_object_width():
    assert SG.side_opening_deg(0.060, SPEC) == -65.0
    assert -31.0 <= SG.side_opening_deg(0.040, SPEC) <= -30.0
    assert SG.side_close_deg(0.060, SPEC) == -35.0
    assert SG.side_close_deg(0.040, SPEC) == -5.0


@pytest.mark.parametrize("oid,shape", [("red_cube", "cube"), ("pool_box_1", "cube"),
                                       ("blue_cylinder", "cylinder"),
                                       ("pool_cyl_1", "cylinder")])
def test_side_shape_accepts_cubes_and_cylinders(oid, shape):
    scene = SceneModel.from_yaml(SCENE)
    assert SG.side_shape(scene.get(oid)) == shape


@pytest.mark.parametrize("oid", ["soda_can", "foam_block"])
def test_side_shape_refuses_the_can_and_the_foam_block(oid):
    scene = SceneModel.from_yaml(SCENE)
    with pytest.raises(CP.CraneRefused) as e:
        SG.side_shape(scene.get(oid))
    assert e.value.stage == "shape"


def test_the_70_mm_cylinder_does_not_fit_the_open_jaw():
    scene, planner, seen = _world("blue_cylinder", "r2c2")
    with pytest.raises(CP.CraneRefused) as e:
        SG.side_candidates(planner, scene, "blue_cylinder", seen)
    assert e.value.stage == "opening"


def test_object_box_align_turns_a_cylinder_box_to_the_jaw():
    scene = SceneModel.from_yaml(SCENE)
    cyl = scene.get("pool_cyl_1")
    yaw = math.radians(30.0)
    quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    pos = (0.3, 0.0, 0.05)
    plain = G.object_box(cyl, pos, quat)
    assert np.allclose(plain.axes[:, 0], [math.cos(yaw), math.sin(yaw), 0.0], atol=1e-9)
    want = np.array([math.cos(math.radians(70)), math.sin(math.radians(70)), 0.0])
    aligned = G.object_box(cyl, pos, quat, align=want)
    assert np.allclose(aligned.axes[:, 0], want, atol=1e-9)
    assert np.allclose(aligned.axes[:, 2], plain.axes[:, 2], atol=1e-9)    # same axis
    assert np.allclose(aligned.axes.T @ aligned.axes, np.eye(3), atol=1e-9)
    assert np.allclose(aligned.half, plain.half) and np.allclose(aligned.center, plain.center)
    # align=None is the old behaviour exactly
    none = G.object_box(cyl, pos, quat, align=None)
    assert np.array_equal(none.axes, plain.axes) and np.array_equal(none.center, plain.center)


def test_object_box_ignores_align_for_a_box():
    scene = SceneModel.from_yaml(SCENE)
    cube = scene.get("red_cube")
    yaw = math.radians(30.0)
    quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    plain = G.object_box(cube, (0.3, 0.0, 0.03), quat)
    turned = G.object_box(cube, (0.3, 0.0, 0.03), quat, align=(0.0, 1.0, 0.0))
    assert np.array_equal(plain.axes, turned.axes)
    assert np.array_equal(plain.half, turned.half)


def _route(thumb, finger):
    return G.RouteReport(thumb, finger, 0, 0, None, 1)


def _end():
    return G.PadClearance(0.01, 0.01, 0.01, 0.01, 0.0, 0.003, 0.02, 0.01, 0.01, 0.0, 0.0,
                          (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def test_centring_shift_without_a_target_equalises_and_with_one_sets_the_thumb():
    end, route = _end(), _route(0.004, 0.0015)
    n = np.asarray(end.face_normal)
    old = G.centring_shift(end, route, 0.02)
    assert np.allclose(old @ n, 0.5 * (0.004 - 0.0015))
    same = G.centring_shift(end, route, 0.02, target_thumb_m=None)
    assert np.array_equal(old, same)
    tgt = G.centring_shift(end, route, 0.02, target_thumb_m=0.0015)
    assert tgt @ n == pytest.approx(0.004 - 0.0015)
    # the other components are untouched
    assert np.allclose(tgt - (tgt @ n) * n, old - (old @ n) * n)


def test_hand_closing_is_orthogonal_to_the_hand_axis():
    for q, g in (([0.0] * 7, -65.0), ([0.0, 10.0, 40.0, -50.0, 5.0, -20.0, -5.0], -35.0)):
        h = G.hand_frames(q, g)
        assert abs(float(h.closing @ h.thumb_rot[:, 2])) < 1e-9
        assert np.linalg.norm(h.closing) == pytest.approx(1.0)


# ── The crane plan defaults ───────────────────────────────────────────────────

def test_a_crane_plan_keeps_its_one_ladder():
    scene, planner, seen = _world("red_cube", "r3c2")
    plan = CP.plan_crane_pick(planner, scene, "red_cube", seen)
    assert plan.kind == "crane" and plan.lift is None and plan.approach is None
    assert plan.up(3) is plan.rungs[3]
    assert plan.up_count == len(plan.rungs)
    q = plan.rungs[2]
    assert plan.progress(q, plan.spec.opening_deg) == pytest.approx(
        float(G.hand_frames(q, plan.spec.opening_deg).gap_mid[2]))
    assert plan.face_half_m == pytest.approx(0.014)


# ── Planning (15-30 s each) ───────────────────────────────────────────────────

def test_the_box_hand_clears_the_rail_where_the_tube_does_not():
    scene, planner, seen = _world("red_cube", "r1c1")
    plan = SG.plan_side_pick(planner, scene, "red_cube", seen)
    assert plan.kind == "side"
    name = "descent hover->grasp"
    seq, grip, tgt, _rate = CP._segments(plan)[name]
    with pytest.raises(ArmClearanceError) as e:
        planner.check_arm_path(seq, segment=name, gripper_deg=grip, target=tgt)
    assert e.value.obstacle == "rig_rail_inner_left"
    boxed = SG.box_hand(planner)
    assert SG.box_hand(boxed) is boxed                        # idempotent
    boxed.check_arm_path(seq, segment=name, gripper_deg=grip, target=tgt)    # passes


def test_the_box_hand_still_refuses_a_hand_through_an_obstacle():
    scene, planner, seen = _world("red_cube", "r2c2")
    plan = SG.plan_side_pick(planner, scene, "red_cube", seen)
    # a pool box standing on the grasp, in the middle of the slide-in
    g = G.hand_frames(plan.rungs[0], plan.spec.opening_deg).gap_mid
    z = scene.table_surface_z + 0.02
    scene.update_poses({"pool_box_2": (float(g[0]), float(g[1]) + 0.0, z)})
    name = "descent hover->grasp"
    seq, grip, tgt, _rate = CP._segments(plan)[name]
    with pytest.raises(ArmClearanceError) as e:
        SG.box_hand(planner).check_arm_path(seq, segment=name, gripper_deg=grip, target=tgt)
    assert e.value.obstacle == "pool_box_2"
    with pytest.raises(CP.CraneRefused) as r:
        SG.check_side_extras(scene, plan, plan.spec)
    assert r.value.stage == "neighbours"


def test_a_red_cube_side_pick_at_r2c2_is_planned_with_a_lift_ladder_and_a_margin():
    scene, planner, seen = _world("red_cube", "r2c2")
    plan = SG.plan_side_pick(planner, scene, "red_cube", seen)
    assert plan.kind == "side" and plan.lift and plan.up_count == len(plan.lift)
    assert plan.up(2) is plan.lift[2]
    assert plan.checks["margin"]["margin_mm"] > 0
    assert plan.checks["side"]["neighbours_mm"] >= 10
    assert plan.spec.opening_deg == -65.0 and plan.spec.close_deg == -35.0
    # progress is the distance back along the (horizontal) approach
    q = plan.rungs[3]
    g = G.hand_frames(q, plan.spec.opening_deg).gap_mid
    assert plan.progress(q, plan.spec.opening_deg) == pytest.approx(
        -float(g[:2] @ plan.approach[:2]))
    assert plan.face_half_m == pytest.approx(float(G.THUMB_PAD[2][2]))


def test_a_red_cube_side_pick_at_r3c1_is_refused():
    scene, planner, seen = _world("red_cube", "r3c1")
    with pytest.raises(CP.CraneRefused):
        SG.plan_side_pick(planner, scene, "red_cube", seen)


# r1c2 is the cell whose box moves the red_cube r2c2 approach from 0 to 90 deg
# (a box on r2c3 or r2c1 leaves it at 0, 64-85 mm clear)
NEIGHBOUR_CELL = "r1c2"


def test_a_neighbour_changes_or_blocks_the_approach_and_keeps_10_mm():
    scene, planner, seen = _world("red_cube", "r2c2")
    free = SG.plan_side_pick(planner, scene, "red_cube", seen)
    pos = _on(scene, NEIGHBOUR_CELL, "pool_box_2")
    scene, planner, seen = _world("red_cube", "r2c2", {"pool_box_2": pos})
    try:
        plan = SG.plan_side_pick(planner, scene, "red_cube", seen)
    except CP.CraneRefused as exc:
        assert "pool_box_2" in exc.reason
        return
    assert abs((_az(plan) - _az(free) + 180.0) % 360.0 - 180.0) > 1.0
    assert plan.checks["side"]["neighbours_mm"] >= 10


def test_place_choice_r2c2_to_r1c1_is_side_on_because_the_crane_is_refused():
    scene, planner, seen = _world("red_cube", "r2c2")
    ch = SG.plan_place_choice(planner, scene, "red_cube", "r1c1", seen)
    assert ch.used == "side-on"
    assert ch.refusals["crane"] is not None and ch.refusals["side-on"] is None
    assert ch.plan.pick.kind == "side"
