"""The robot's mount, stated once per language, agrees everywhere (#192).

The October layout moved the torso to (0.206, 0, 1.15) in the scene frame.  The
MJCF places it, and the Python side restates it in a handful of places that
cannot import the MJCF (the motion layer, the native server's placement module,
RViz's URDF).  A mount move that reaches some of them and not others is how a
planner ends up 0.2 m away from the arm it plans for -- so each restatement is
checked against the compiled model here, along with the other things #192 moved
together: REST as the start pose, the right shoulder-roll stop, the phantom
left arm, the pedestal stub and the board view.

Requires the `mujoco` package (native macOS arm64 side); skipped without it.
"""

import ast
import math
import os
import sys

import pytest

mujoco = pytest.importorskip("mujoco")
import numpy as np  # noqa: E402

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "native_mujoco"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from reachy_ai.motion import gaze  # noqa: E402
from reachy_ai.motion import kinematics as K  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.scene.awareness import ARM_BASE_IN_WORLD, SceneModel  # noqa: E402

_MODEL = os.path.join(_ROOT, "native_mujoco", "model", "reachy_1_2.xml")
_SCENE = os.path.join(_ROOT, "scenes", "FWDCenterLabSivaPool.yaml")

ARM8 = R.R_JOINTS


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(_MODEL)


@pytest.fixture(scope="module")
def zero(model):
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    mujoco.mj_kinematics(model, d)
    return d


def _body(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()


class TestOneMount:
    def test_mjcf_torso_is_arm_base(self, model, zero):
        pos, rot = _body(model, zero, "torso")
        assert pos == pytest.approx(ARM_BASE_IN_WORLD, abs=1e-9)
        assert rot == pytest.approx(np.eye(3), abs=1e-12)

    def test_the_mount_is_the_october_layout(self):
        assert ARM_BASE_IN_WORLD == (0.206, 0.0, 1.15)
        assert R.CURRENT_MOUNT == R.MOUNT_OCTOBER

    def test_pedestal_body_is_the_scene_origin(self, model, zero):
        # The scene's frame_id and RViz's fixed frame are both `pedestal`.
        pos, _ = _body(model, zero, "pedestal")
        assert pos == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)

    def test_gaze_neck_pivot_is_head_x(self, model, zero):
        pos, _ = _body(model, zero, "head_x")
        assert gaze.NECK_PIVOT_IN_WORLD == pytest.approx(tuple(pos), abs=1e-9)

    def test_kinematics_base_and_shoulder(self, model, zero):
        assert tuple(K._BASE) == pytest.approx(ARM_BASE_IN_WORLD)
        shoulder, _, _, _ = K.link_frames([0.0] * 7)
        pos, _ = _body(model, zero, "r_shoulder")
        assert shoulder == pytest.approx(pos, abs=1e-9)

    def test_crane_shoulder(self, model, zero):
        from reachy_ai.tasks import crane_pick_live as CP
        pos, _ = _body(model, zero, "r_shoulder")
        assert CP._SHOULDER_XY == pytest.approx((pos[0], pos[1]), abs=1e-9)

    def test_placement_shoulder(self, model, zero):
        import placement
        pos, _ = _body(model, zero, "r_shoulder")
        assert placement.SHOULDER_XYZ == pytest.approx(tuple(pos), abs=1e-9)

    def test_rviz_urdf_gets_the_mount(self):
        import urdf_with_mount as U
        with open(os.path.join(_ROOT, "native_mujoco", "model",
                               "reachy.urdf"), encoding="utf-8") as f:
            stock = f.read()
        moved = U.with_mount(stock, ARM_BASE_IN_WORLD)
        assert '<origin xyz="0.206 0 1.15" rpy="0 0 0" />' in moved
        assert moved.count('xyz="0.206 0 1.15"') == 1
        with pytest.raises(ValueError):
            U.with_mount("<robot/>", ARM_BASE_IN_WORLD)


class TestStartPoseIsRest:
    def test_home_keyframe_right_arm_is_rest(self, model):
        from joint_map import JOINT_TABLE
        key = model.key("home")
        by_name = {e.sdk_name: e.mjcf_index for e in JOINT_TABLE}
        for name in ARM8:
            want = math.radians(R.REST[name])
            assert key.qpos[by_name[name]] == pytest.approx(want, abs=1e-4)
            assert key.ctrl[by_name[name]] == pytest.approx(want, abs=1e-4)

    def test_rest_is_inside_the_travel(self):
        assert K.within_limits([R.REST[j] for j in R.ARM7])

    def test_home_is_rest(self):
        assert R.POSTURE_HOME == R.POSTURE_REST
        assert R.posture_of(R.REST) == R.POSTURE_REST
        assert set(R.POSTURES) == {R.POSTURE_REST, R.POSTURE_PRESENT}

    def test_rest_stays_put_with_the_motors_off(self):
        """The sim starts here with the motors off: REST must be a pose the
        arm stays in.  3 s from the start keyframe, as the server resets."""
        from simulation_core import SimulationCore
        core = SimulationCore.from_paths(_MODEL, _SCENE)
        core.reset()
        m, d = core.model, core.data
        for _ in range(int(3.0 / m.opt.timestep)):
            core.advance()
        now = {}
        for name in ARM8:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            now[name] = math.degrees(d.qpos[m.jnt_qposadr[jid]])
        assert max(abs(now[n] - R.REST[n]) for n in R.GROSS_JOINTS) < 2.0
        assert now["r_gripper"] > 15.0                       # still shut
        assert R.posture_of(now) == R.POSTURE_REST
        on_board = set()
        for c in d.contact[:d.ncon]:
            names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                     for g in (c.geom1, c.geom2)}
            bodies = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
                      for g in (c.geom1, c.geom2)}
            if "table_top" in bodies:
                on_board |= {n for n in names if n.startswith("r_")}
        # resting on the hand's palm and back, not the pads
        assert on_board & {"r_hand_back_col", "r_palm_col"}
        assert not on_board & {"r_thumb_col", "r_finger_col"}


class TestShoulderRollStop:
    """+0.354 rad (+20.3 deg), a documented SDK-visible deviation (ADR-0010)."""

    def test_mjcf_joint_and_actuator(self, model):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_shoulder_roll")
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_r_shoulder_roll")
        assert tuple(model.jnt_range[jid]) == pytest.approx((-3.14, 0.354))
        assert tuple(model.actuator_ctrlrange[aid]) == pytest.approx((-3.14, 0.354))

    def test_python_tables(self):
        from joint_map import JOINT_TABLE
        entry = next(e for e in JOINT_TABLE if e.sdk_name == "r_shoulder_roll")
        assert entry.limits_rad == pytest.approx((-3.14, 0.354))
        assert K.JOINT_LIMITS_DEG["r_shoulder_roll"][1] == pytest.approx(
            math.degrees(0.354))

    def test_fake_server_ik_bounds(self):
        with open(os.path.join(_ROOT, "fake_reachy_server.py"), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", "") == "_R_ARM_LIMITS" for t in node.targets):
                limits = ast.literal_eval(node.value)
                assert limits[1][1] == pytest.approx(0.354)
                return
        pytest.fail("_R_ARM_LIMITS not found in fake_reachy_server.py")

    def test_left_arm_unchanged(self, model):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "l_shoulder_roll")
        assert tuple(model.jnt_range[jid]) == pytest.approx((-0.174, 3.14))


class TestPhantomLeftArm:
    def test_left_arm_neither_collides_nor_renders(self, model):
        left = [g for g in range(model.ngeom)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                      model.geom_bodyid[g]) or "").startswith("l_")
                and not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                           model.geom_bodyid[g]) or "").startswith("l_antenna")]
        assert left, "no left-arm geoms found"
        for g in left:
            assert model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0
            assert model.geom_group[g] >= 3      # never drawn by a camera (0-2)

    def test_left_arm_keeps_its_mass(self, model):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_upper_arm")
        rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_upper_arm")
        assert model.body_mass[bid] == pytest.approx(model.body_mass[rid])


class TestHandCollision:
    """The hand's simplified collision (#192): two boxes filling the visual
    shell, in the body that turns with wrist roll, clear of the R12-502 pads."""

    def test_the_boxes_are_robot_collision_geoms_with_no_mass(self, model):
        thumb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_gripper_thumb")
        for name in ("r_hand_back_col", "r_palm_col"):
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert g >= 0, name
            assert model.geom_bodyid[g] == thumb
            assert (model.geom_contype[g], model.geom_conaffinity[g]) == (2, 5)
            assert model.geom_group[g] == 3

    def test_the_palm_stays_clear_of_the_pads(self, model):
        def z_range(name):
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            z, h = model.geom_pos[g][2], model.geom_size[g][2]
            return z - h, z + h
        palm_bottom = z_range("r_palm_col")[0]
        pad_top = z_range("r_thumb_col")[1]
        assert palm_bottom - pad_top >= 0.004                 # 5 mm, as built

    def test_the_boxes_stay_inside_the_visual_shell(self, model):
        shell = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_thumb_body")
        lo = model.geom_pos[shell] - model.geom_size[shell]
        hi = model.geom_pos[shell] + model.geom_size[shell]
        for name in ("r_hand_back_col", "r_palm_col"):
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert np.all(model.geom_pos[g] - model.geom_size[g] >= lo - 1e-9), name
            assert np.all(model.geom_pos[g] + model.geom_size[g] <= hi + 1e-9), name


class TestPedestalStub:
    def test_the_stub_does_not_collide(self, model):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pedestal_geom")
        assert (model.geom_contype[g], model.geom_conaffinity[g]) == (0, 0)

    def test_stub_clears_the_board_and_the_rig(self):
        from simulation_core import load_world
        m, *_ = load_world(_MODEL, _SCENE)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        ped = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pedestal_geom")
        checked = 0
        for g in range(m.ngeom):
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            if body == "table_top" or body.startswith("rig_rail"):
                assert mujoco.mj_geomDistance(m, d, ped, g, 1.0, None) > 0.01, body
                checked += 1
        assert checked == 6


def _neck_pose(model, pitch_deg, yaw_deg, roll_deg=0.0):
    d = mujoco.MjData(model)
    for joint, deg in (("neck_roll", roll_deg), ("neck_pitch", pitch_deg),
                       ("neck_yaw", yaw_deg)):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        d.qpos[model.jnt_qposadr[jid]] = math.radians(deg)
    mujoco.mj_forward(model, d)
    return d


def _eye(model, data, name):
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"{name}_camera")
    return data.cam_xpos[cid], data.cam_xmat[cid].reshape(3, 3)


def _mj_pixel(model, data, cam, point):
    """``point`` through MuJoCo's own camera pose, with gaze's focal lengths."""
    pos, rot = _eye(model, data, cam[0])
    v = rot.T @ (np.asarray(point) - pos)
    return 320 + cam[2] * v[0] / -v[2], 240 - cam[3] * v[1] / -v[2]


#: The torso-frame gripper tip at REST (#208's example), in the world frame.
REST_TIP = tuple(b + t for b, t in zip(ARM_BASE_IN_WORLD, (0.141, -0.397, -0.397)))


class TestGazeAwayFromYawZero:
    """The neck chain is roll, pitch, yaw, with yaw applied LAST (#208)."""

    @pytest.mark.parametrize("yaw", (0.0, 15.0, -15.0, 38.0, -38.0, 45.0, -45.0,
                                     120.0, -120.0))
    @pytest.mark.parametrize("pitch", (-30.0, 0.0, 30.0, 64.0))
    def test_project_matches_the_mjcf_cameras(self, model, pitch, yaw):
        d = _neck_pose(model, pitch, yaw)
        # Points spread over MuJoCo's own left frame, near and far.
        pos, rot = _eye(model, d, "left")
        _, _, fx, fy = gaze.CAMERAS[0]
        for depth in (0.4, 0.9):
            for u in (80, 320, 560):
                for v in (60, 240, 420):
                    local = ((u - 320) / fx * depth, -(v - 240) / fy * depth, -depth)
                    point = tuple(pos + rot @ np.array(local))
                    for cam in gaze.CAMERAS:
                        got = gaze.project(point, pitch, yaw, cam)
                        want = _mj_pixel(model, d, cam, point)
                        assert got == pytest.approx(want, abs=0.5)

    @pytest.mark.parametrize("target", [
        REST_TIP,                    # the hand at REST: yaw about -38
        (0.55, 0.40, 0.74),          # on the table, to the left
        (0.60, -0.55, 1.25),         # level with the neck, to the right
        (-0.40, 0.70, 0.90),         # behind-left: yaw past 90
        (-0.40, -0.70, 1.05),        # behind-right
    ])
    def test_unbiased_aim_points_the_head_at_the_target(self, model, monkeypatch,
                                                        target):
        monkeypatch.setattr(gaze, "PITCH_BIAS_DEG", 0.0)
        d = _neck_pose(model, *gaze.neck_angles_for(target, clamp=False))
        pivot, rot = _body(model, d, "head")
        want = np.asarray(target) - pivot
        cos = float(rot[:, 0] @ want / np.linalg.norm(want))
        # BUILT_IN_PITCH_DEG is 10, the MJCF's 0.174 rad is 9.97.
        assert math.degrees(math.acos(min(1.0, cos))) < 0.05

    @pytest.mark.parametrize("target", [
        REST_TIP,                    # pitch clamps
        (0.55, 0.40, 0.74),
        (-0.40, 0.70, 0.90),         # behind-left
        (0.30, -0.30, 0.45),         # pitch clamps, well off centre
    ])
    def test_the_aim_faces_the_target_even_when_pitch_clamps(self, model, target):
        # Facing it: midway between the two eyes' columns is the frame centre.
        d = _neck_pose(model, *gaze.neck_angles_for(target))
        columns = [_mj_pixel(model, d, cam, target)[0] for cam in gaze.CAMERAS]
        assert sum(columns) / 2 == pytest.approx(320.0, abs=1.5)

    def test_the_rest_tip_is_in_both_frames(self, model):
        pitch, yaw = gaze.neck_angles_for(REST_TIP)
        assert pitch == gaze.PITCH_LIMITS[1]
        assert yaw == pytest.approx(-38.0, abs=0.5)
        d = _neck_pose(model, pitch, yaw)
        for cam in gaze.CAMERAS:
            u, v = _mj_pixel(model, d, cam, REST_TIP)
            assert 0 < u < 640 and 0 < v < 480
            assert gaze.project(REST_TIP, pitch, yaw, cam) == pytest.approx(
                (u, v), abs=0.5)

    def test_board_view_pose_faces_an_off_centre_group(self, model):
        points = [(x, y - 0.35, z) for x, y, z in gaze.BOARD_CELL_CORNERS]
        pose = gaze.board_view_pose(points)
        centroid = tuple(np.mean(points, axis=0))
        d = _neck_pose(model, pose["neck_pitch"], pose["neck_yaw"])
        columns = [_mj_pixel(model, d, cam, centroid)[0] for cam in gaze.CAMERAS]
        assert sum(columns) / 2 == pytest.approx(320.0, abs=1.5)


class TestBoardView:
    def test_corners_are_the_scene_cells(self):
        scene = SceneModel.from_yaml(_SCENE)
        xs, ys = [], []
        for cell in scene.grid_cells():
            obj = scene.get(cell)
            xs += [obj.center[0] - obj.size[0] / 2, obj.center[0] + obj.size[0] / 2]
            ys += [obj.center[1] - obj.size[1] / 2, obj.center[1] + obj.size[1] / 2]
        want = {(round(x, 4), round(y, 4)) for x in (min(xs), max(xs))
                for y in (max(ys), min(ys))}
        got = {(round(c[0], 4), round(c[1], 4)) for c in gaze.BOARD_CELL_CORNERS}
        assert got == want

    def test_named_pose_is_the_computed_one(self):
        computed = gaze.board_view_pose(gaze.BOARD_CELL_CORNERS)
        for joint in ("neck_roll", "neck_pitch", "neck_yaw"):
            assert gaze.BOARD_VIEW[joint] == pytest.approx(computed[joint], abs=0.05)
        assert computed["margin_px"] > 50.0      # all nine cells, with room

    def test_pinhole_matches_the_mjcf_cameras(self, model):
        # The board-view model places each eye from the MJCF offsets; check it
        # against MuJoCo's own camera pose at the board view.
        d = mujoco.MjData(model)
        for joint, deg in gaze.BOARD_VIEW.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            d.qpos[model.jnt_qposadr[jid]] = math.radians(deg)
        mujoco.mj_forward(model, d)
        for cam in gaze.CAMERAS:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA,
                                    f"{cam[0]}_camera")
            pos, rot = d.cam_xpos[cid], d.cam_xmat[cid].reshape(3, 3)
            for corner in gaze.BOARD_CELL_CORNERS:
                v = rot.T @ (np.asarray(corner) - pos)
                u_mj = 320 + cam[2] * v[0] / -v[2]
                v_mj = 240 - cam[3] * v[1] / -v[2]
                u, vv = gaze.project(corner, gaze.BOARD_VIEW["neck_pitch"],
                                     gaze.BOARD_VIEW["neck_yaw"], cam)
                assert (u, vv) == pytest.approx((u_mj, v_mj), abs=0.5)
