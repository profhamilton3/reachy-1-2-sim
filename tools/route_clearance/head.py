"""The head angles for each waypoint (design section 6): the neck angles that aim the eyes at the gripper tip, inside
the tabletop's neck limits, and whether the tip is in each eye's frame there.

The model's own FK and cameras, posed by mj_forward with the arm at the waypoint: the tip is the r_tip site (the SDK's
forward_kinematics point) and the eyes are left_camera / right_camera, pinhole at their compiled fovy over a 640x480
frame (the measured profile's resolution; the render does not apply lens distortion). Not gaze.project, whose yaw
order is wrong away from yaw 0 (#208). Roll stays 0.
"""
import math

import numpy as np

from .limits import NECK_LIMITS_DEG

FRAME_PX = (640, 480)


def _project(world, cam, point):
    m, d = world.model, world.data
    rot = d.cam_xmat[cam].reshape(3, 3)
    local = rot.T @ (point - d.cam_xpos[cam])
    depth = -local[2]
    if depth <= 1e-6:
        return None
    w, h = FRAME_PX
    f = (h / 2.0) / math.tan(math.radians(float(m.cam_fovy[cam])) / 2.0)
    return (w / 2.0 + f * local[0] / depth, h / 2.0 - f * local[1] / depth)


def _off_axis(world, point):
    """(horizontal, vertical) angles of the point off the eyes' mean optical axis, radians."""
    d = world.data
    cams = list(world.cams.values())
    centre = np.mean([d.cam_xpos[c] for c in cams], axis=0)
    rot = d.cam_xmat[cams[0]].reshape(3, 3)
    local = rot.T @ (point - centre)
    depth = -local[2]
    return np.array([math.atan2(local[0], depth), math.atan2(local[1], depth)])


def aim(world, q):
    """Neck angles (deg) that aim at the tip with the arm at q, projected onto the limits each step, and what the
    eyes see there."""
    lo = np.array([NECK_LIMITS_DEG["neck_pitch"][0], NECK_LIMITS_DEG["neck_yaw"][0]])
    hi = np.array([NECK_LIMITS_DEG["neck_pitch"][1], NECK_LIMITS_DEG["neck_yaw"][1]])

    def error(angles):
        world.set_arm(q, {"neck_roll": 0.0, "neck_pitch": angles[0], "neck_yaw": angles[1]})
        return _off_axis(world, world.tip_world())

    x = np.array([30.0, 0.0])
    for _ in range(40):
        e = error(x)
        jac = np.zeros((2, 2))
        for k in range(2):
            dx = np.zeros(2)
            dx[k] = 0.01
            jac[:, k] = (error(x + dx) - e) / 0.01
        step = np.linalg.lstsq(jac, -e, rcond=None)[0]
        x_new = np.clip(x + step, lo, hi)
        if np.max(np.abs(x_new - x)) < 1e-4:
            x = x_new
            break
        x = x_new
    e = error(x)
    tip = world.tip_world()
    view = {}
    for side, cam in sorted(world.cams.items()):
        px = _project(world, cam, tip)
        inside = px is not None and 0.0 <= px[0] < FRAME_PX[0] and 0.0 <= px[1] < FRAME_PX[1]
        view[side] = (inside, px)
    return {
        "neck_deg": {"neck_roll": 0.0, "neck_pitch": float(x[0]), "neck_yaw": float(x[1])},
        "at_limit": bool(np.any(np.isclose(x, lo, atol=1e-3)) or np.any(np.isclose(x, hi, atol=1e-3))),
        "off_axis_deg": float(math.degrees(float(np.hypot(*e)))),
        "tip_in_view_left": view["left"][0],
        "tip_in_view_right": view["right"][0],
        "tip_px_left": view["left"][1],
        "tip_px_right": view["right"][1],
    }
