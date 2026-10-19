"""Which crane attitudes can straddle the 60 mm cube with the palm clear?  (#195)

Hand geometry only (no arm, no reach): the pads and the palm in the thumb frame (the IK
frame; `hand_rotation` and the thumb frame coincide), placed at a (tilt, lean azimuth)
attitude over red_cube with `plan_crane_pick`'s own grasp rules -- pads equalised on the
faces, height equalising "pad centres below the top" and "lowest pad above the table",
then raised until the palm clears the top by `CraneSpec.palm_clearance_m`.  Reported: the
pads' depth below the top, the lowest pad above the table, the side clearances and the
jaw tilt.  The lean azimuth is measured from the JAW in the horizontal: 90 tilts the hand
about the jaw (both pads stay level), 0/180 tilts it along the jaw (one pad rises).

PYTHONPATH=src:native_mujoco python docs/reviews/2026-10-19-crane-retune-195/palm_geometry.py
"""
import math
import numpy as np
from reachy_ai.motion import grasp_alignment as G

PALM_M = 0.002
HALF = np.array([0.03, 0.03, 0.03])        # red_cube
TABLE_Z = 0.741


def local_boxes(grip=-65.0):
    h = G.hand_frames([0, 0, 0, -90, 0, 0, 0], grip)
    T, o = h.thumb_rot, h.gap_mid
    out = {b.name: (T.T @ (b.center - o), T.T @ b.axes, b.half) for b in (h.thumb, h.finger, h.palm)}
    out['faces'] = (T.T @ (h.thumb_face - o), T.T @ (h.finger_face - o))
    return out


def place(rot, gap, boxes):
    out = {n: G.Box(n, gap + rot @ v[0], rot @ v[1], v[2]) for n, v in boxes.items() if n != 'faces'}
    out['hand'] = G.HandFrames(None, rot, rot, out['thumb_pad'], out['finger_pad'],
                               gap + rot @ boxes['faces'][0], gap + rot @ boxes['faces'][1],
                               out['palm'])
    return out


def rotation(tilt, az_from_jaw):
    """Thumb-frame rotation: approach (-z) tilted `tilt` from straight down toward the
    horizontal direction `az_from_jaw` (deg) from the jaw (world x)."""
    t, a = math.radians(tilt), math.radians(az_from_jaw)
    z = -np.array([math.sin(t) * math.cos(a), math.sin(t) * math.sin(a), -math.cos(t)])
    j = np.array([1.0, 0.0, 0.0]); j = j - (j @ z) * z; j /= np.linalg.norm(j)
    y = -j                                   # grasp_rotation: y = -jaw
    return np.column_stack([np.cross(y, z), y, z])


def evaluate(tilt, az, boxes):
    obj = G.Box('cube', np.array([0.0, 0.0, TABLE_Z + HALF[2]]), np.eye(3), HALF)
    R = rotation(tilt, az)
    gap = obj.center.copy()
    top = obj.highest_z
    for _ in range(3):                       # the plan's rules, iterated to settle
        b = place(R, gap, boxes)
        below = top - max(b['thumb_pad'].center[2], b['finger_pad'].center[2])
        above = min(b['thumb_pad'].lowest_z, b['finger_pad'].lowest_z) - TABLE_Z
        gap = gap - np.array([0, 0, 0.5 * (above - below)])
    for _ in range(3):                       # pads equalised on the faces (x is the normal)
        b = place(R, gap, boxes)
        pc = G.pad_clearance(b['hand'], obj)
        gap = gap + 0.5 * (pc.thumb_side_m - pc.finger_side_m) * np.asarray(pc.face_normal)
    for _ in range(40):                      # then up until the palm clears the top
        b = place(R, gap, boxes)
        palm = G.obb_separation(b['palm'], obj)
        if palm >= PALM_M - 1e-5:
            break
        gap = gap + np.array([0, 0, max(PALM_M - palm, 0.0005)])
    th, fi = b['thumb_pad'], b['finger_pad']
    return dict(tilt=tilt, az=az, palm_mm=round(1000 * palm, 1),
                thumb_below_top_mm=round(1000 * (top - th.center[2]), 1),
                finger_below_top_mm=round(1000 * (top - fi.center[2]), 1),
                lowest_pad_above_table_mm=round(1000 * (min(th.lowest_z, fi.lowest_z) - TABLE_Z), 1),
                thumb_mm=round(1000 * G.obb_separation(th, obj), 1),
                finger_mm=round(1000 * G.obb_separation(fi, obj), 1))


if __name__ == '__main__':
    boxes = local_boxes()
    print('tilt az | palm thumb_below finger_below lowest_pad thumb_clear finger_clear')
    for tilt in (0, 10, 20, 30, 40, 50, 60):
        for az in (0, 45, 90, 135, 180, 225, 270, 315):
            r = evaluate(tilt, az, boxes)
            ok = r['thumb_below_top_mm'] > 0 and r['finger_below_top_mm'] > 0 and r['lowest_pad_above_table_mm'] >= 0
            print(f"{tilt:4} {az:4} | {r['palm_mm']:5} {r['thumb_below_top_mm']:6} {r['finger_below_top_mm']:6} "
                  f"{r['lowest_pad_above_table_mm']:6} {r['thumb_mm']:5} {r['finger_mm']:5} {'OK' if ok else ''}")
