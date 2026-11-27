"""primitives.look_at aims through gaze.neck_angles_for (#220).

It had its own formula, which turned the neck yaw-then-pitch (the MJCF applies
yaw last, #208) and left out the head's built-in 10 deg of downward pitch.  The
last class needs the `mujoco` package and is skipped without it.
"""

import math
import os
import sys

import pytest

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))

from reachy_ai.motion import gaze  # noqa: E402
from reachy_ai.motion import primitives as P  # noqa: E402

NECK = ("neck_roll", "neck_pitch", "neck_yaw")


class _Joint:
    def __init__(self):
        self.goal_position = 0.0
        self.present_position = 0.0
        self.compliant = True


class _Head:
    def __init__(self):
        for name in NECK:
            setattr(self, name, _Joint())


class _Robot:
    def __init__(self):
        self.head = _Head()
        self.turned_on = []

    def turn_on(self, part):
        self.turned_on.append(part)


@pytest.fixture
def moves(monkeypatch):
    """Every smooth_move look_at asks for, as (device, pose, duration)."""
    calls = []
    monkeypatch.setattr(P, "smooth_move",
                        lambda dev, pose, duration=2.0: calls.append((dev, dict(pose), duration)))
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    return calls


def _commanded(moves, target):
    P.look_at(_Robot(), target)
    return moves[-1][1]


def _from_pivot(dx, dy, dz):
    px, py, pz = gaze.NECK_PIVOT_IN_WORLD
    return (px + dx, py + dy, pz + dz)


def _old_formula(target):
    """look_at before #220, kept here to state what changed and what did not."""
    dx, dy, dz = (target[i] - gaze.NECK_PIVOT_IN_WORLD[i] for i in range(3))
    pitch = math.degrees(math.atan2(-dz, math.hypot(dx, dy)))
    yaw = math.degrees(math.atan2(dy, dx))
    return (min(max(pitch, gaze.PITCH_LIMITS[0]), gaze.PITCH_LIMITS[1]),
            min(max(yaw, gaze.YAW_LIMITS[0]), gaze.YAW_LIMITS[1]))


class TestOneAimingImplementation:
    @pytest.mark.parametrize("target", [
        (0.43, 0.0, 0.766), (0.43, -0.152, 0.766), (0.28, 0.152, 0.766),
        (0.50, 0.0, 0.90), (0.347, -0.397, 0.753),
    ])
    def test_commands_exactly_what_gaze_computes(self, moves, target):
        pitch, yaw = gaze.neck_angles_for(target)
        pose = _commanded(moves, target)
        assert pose == {"neck_roll": 0.0, "neck_pitch": pytest.approx(pitch),
                        "neck_yaw": pytest.approx(yaw)}

    def test_stiffens_the_neck_and_moves_the_head_over_duration(self, moves):
        robot = _Robot()
        P.look_at(robot, (0.43, 0.0, 0.766), duration=0.7)
        assert robot.turned_on == ["head"]
        assert [getattr(robot.head, n).compliant for n in NECK] == [False] * 3
        assert [(dev is robot.head, secs) for dev, _, secs in moves] == [(True, 0.7)]

    def test_default_duration_is_one_second(self, moves):
        P.look_at(_Robot(), (0.43, 0.0, 0.766))
        assert moves[-1][2] == 1.0

    def test_a_robot_with_no_head_is_left_alone(self, moves):
        P.look_at(object(), (0.43, 0.0, 0.766))
        assert moves == []


class TestYawZero:
    @pytest.mark.parametrize("offset", [(0.5, 0.0, -0.3), (0.4, 0.0, -0.4), (0.6, 0.0, 0.1)])
    def test_differs_from_the_old_aim_by_the_tilt_and_bias_only(self, moves, offset):
        target = _from_pivot(*offset)
        old_pitch, _ = _old_formula(target)
        pose = _commanded(moves, target)
        assert pose["neck_yaw"] == pytest.approx(0.0)
        assert pose["neck_pitch"] == pytest.approx(
            old_pitch - gaze.BUILT_IN_PITCH_DEG + gaze.PITCH_BIAS_DEG)


class TestLimits:
    @pytest.mark.parametrize("offset, joint, limit", [
        ((0.1, 0.0, -0.6), "neck_pitch", 64.0),
        ((0.1, 0.0, 0.6), "neck_pitch", -45.0),
        ((-1.0, 0.05, 0.0), "neck_yaw", 159.0),
        ((-1.0, -0.05, 0.0), "neck_yaw", -159.0),
    ])
    def test_out_of_reach_pins_at_the_same_limit_as_before(self, moves, offset,
                                                          joint, limit):
        target = _from_pivot(*offset)
        old = dict(zip(("neck_pitch", "neck_yaw"), _old_formula(target)))
        assert old[joint] == limit
        assert _commanded(moves, target)[joint] == limit

    def test_every_direction_stays_inside_the_travel(self, moves):
        for az in range(-180, 181, 15):
            for el in range(-90, 91, 15):
                a, e = math.radians(az), math.radians(el)
                pose = _commanded(moves, _from_pivot(
                    0.6 * math.cos(e) * math.cos(a), 0.6 * math.cos(e) * math.sin(a),
                    0.6 * math.sin(e)))
                assert -45.0 <= pose["neck_pitch"] <= 64.0
                assert -159.0 <= pose["neck_yaw"] <= 159.0

    def test_below_and_behind_looks_down_forward(self, moves):
        # BEHIND the pivot, not every target the old aim pinned at a stop is
        # pinned there now (#220, PR #221).  This is the clearest case: the old
        # aim turned the head round to the yaw stop.  Yaw is applied after
        # pitch, so that pointed the head up and back; with pitch at its stop
        # the nearer gaze to a point under the chest faces forward.
        target = _from_pivot(-0.2, 0.0, -1.2)
        assert _old_formula(target) == (64.0, 159.0)
        pose = _commanded(moves, target)
        assert (pose["neck_pitch"], pose["neck_yaw"]) == (64.0, 0.0)


# ── Against the compiled model's own cameras ─────────────────────────────────

#: Distance from the cameras of the targets below.  The board's cells are about
#: this far off, and gaze.PITCH_BIAS_DEG was fitted on them.  The bias is one
#: constant, so the vertical tolerance holds at this distance, not at any.
AIM_DISTANCE_M = 0.55
#: Pixels off the frame centre (640 x 480), midway between the two eyes.
TOL_HORIZONTAL_PX = 1.0
TOL_VERTICAL_YAW_ZERO_PX = 2.0
#: Off centre the bias, added to pitch, moves the picture by only
#: bias x cos(yaw), so the eyes aim a little high.
TOL_VERTICAL_PX = 15.0


@pytest.fixture(scope="module")
def model():
    mujoco = pytest.importorskip("mujoco")
    return mujoco.MjModel.from_xml_path(
        os.path.join(_ROOT, "native_mujoco", "model", "reachy_1_2.xml"))


def _at(model, pitch_deg, yaw_deg):
    import mujoco
    data = mujoco.MjData(model)
    for joint, deg in (("neck_pitch", pitch_deg), ("neck_yaw", yaw_deg)):
        data.qpos[model.joint(joint).qposadr[0]] = math.radians(deg)
    mujoco.mj_forward(model, data)
    return data


class TestAimInTheModelsCameras:
    @pytest.mark.parametrize("yaw", (0.0, 38.0, -38.0, 45.0, -45.0))
    @pytest.mark.parametrize("pitch", (0.0, 30.0, 64.0))
    def test_the_target_lands_at_the_image_centre(self, model, moves, pitch, yaw):
        import numpy as np

        # A target on the eyes' own line of sight with the neck at (pitch, yaw).
        data = _at(model, pitch, yaw)
        cams = [model.camera(f"{c[0]}_camera").id for c in gaze.CAMERAS]
        forward = -data.cam_xmat[cams[0]].reshape(3, 3)[:, 2]
        target = tuple(np.mean([data.cam_xpos[c] for c in cams], axis=0)
                       + AIM_DISTANCE_M * forward)

        pose = _commanded(moves, target)
        assert -45.0 < pose["neck_pitch"] < 64.0            # not at a limit
        assert pose["neck_yaw"] == pytest.approx(yaw, abs=1.0)

        data = _at(model, pose["neck_pitch"], pose["neck_yaw"])
        pixels = []
        for (_, _, fx, fy), cam in zip(gaze.CAMERAS, cams):
            v = data.cam_xmat[cam].reshape(3, 3).T @ (np.asarray(target) - data.cam_xpos[cam])
            pixels.append((320 + fx * v[0] / -v[2], 240 - fy * v[1] / -v[2]))
        for u, v in pixels:
            assert 0 < u < 640 and 0 < v < 480
        off_u, off_v = np.mean(pixels, axis=0) - (320.0, 240.0)
        assert abs(off_u) <= TOL_HORIZONTAL_PX
        assert abs(off_v) <= (TOL_VERTICAL_YAW_ZERO_PX if yaw == 0.0 else TOL_VERTICAL_PX)
