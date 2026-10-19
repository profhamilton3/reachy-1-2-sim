"""The level (side-on) hand, from the MJCF boxes (#180).  No IK, no physics.

A level hand: the thumb frame's x axis vertical (up), its z axis horizontal
(the approach is -z).  For each gripper angle: the free gap between the two
pads (the box-box separation, which is what an object must pass through), the
finger pad's face angle against the thumb pad's (the wedge the two make when
closed on a round object), and how far below the pad centres the palm, the
hand back and the forearm capsule reach (what the table sees).

Run: PYTHONPATH=../../../src python3 hand_geometry.py > hand_geometry.txt
"""
import math
import sys

import numpy as np

sys.path.insert(0, "../../../src")
from reachy_ai.motion import grasp_alignment as G  # noqa: E402
from reachy_ai.motion import kinematics as K  # noqa: E402


def level_rot(azimuth_deg):
    a = math.radians(azimuth_deg)
    app = np.array([math.cos(a), math.sin(a), 0.0])
    z, x = -app, np.array([0.0, 0.0, 1.0])
    return np.column_stack([x, np.cross(z, x), z])


def boxes(R, wrist, g):
    """Pads, palm and back for a hand at rotation R (thumb frame) and wrist."""
    thumb_origin = wrist + R @ K._THUMB_ORIGIN_OFFSET     # roll already in R
    rf = R @ K._rotx(math.radians(g))
    finger_origin = thumb_origin + R @ K._FINGER_HINGE_OFFSET
    mk = lambda n, o, Rb, c, h: G.Box(n, o + Rb @ np.asarray(c), Rb, np.asarray(h))
    return {
        "thumb": mk("thumb", thumb_origin, R, *G.THUMB_PAD[1:]),
        "finger": mk("finger", finger_origin, rf, *G.FINGER_PAD[1:]),
        "palm": mk("palm", thumb_origin, R, *G.PALM[1:]),
        "back": mk("back", thumb_origin, R, (0.0, -0.018, -0.003), (0.025, 0.028, 0.019)),
    }


R = level_rot(0.0)
wrist = np.zeros(3)
print("level hand, approach +x; thumb-frame x = up")
print("gripper_deg  free_gap_mm  finger_face_vs_thumb_deg  gap_mid_from_wrist_mm(x,y,z)")
for g in (-68.0, -65.0, -60.0, -55.0, -50.0, -45.0, -40.0, -35.0, -30.0, -25.0, -20.0,
          -15.0, -10.0, -5.0, 0.0):
    b = boxes(R, wrist, g)
    gap = G.obb_separation(b["thumb"], b["finger"])
    # angle between the pads' inner-face normals in the horizontal plane
    nt = -b["thumb"].axes[:, 1]
    nf = b["finger"].axes[:, 1]
    ang = math.degrees(math.acos(max(-1, min(1, float(-nt @ nf)))))
    tf = b["thumb"].center - b["thumb"].axes[:, 1] * b["thumb"].half[1]
    ff = b["finger"].center + b["finger"].axes[:, 1] * b["finger"].half[1]
    mid = 0.5 * (tf + ff)
    print(f"{g:8.1f}  {1000 * gap:9.1f}  {ang:10.1f}   {np.round(1000 * mid, 1)}")

b = boxes(R, wrist, -65.0)
pad_c = 0.5 * (b["thumb"].center[2] + b["finger"].center[2])
print()
print("heights relative to the pad centres (level hand, any opening):")
for n in ("thumb", "finger", "palm", "back"):
    print(f"  {n:6s} lowest {1000 * (b[n].lowest_z - pad_c):6.1f} mm, highest "
          f"{1000 * (b[n].highest_z - pad_c):6.1f} mm")
print(f"  forearm capsule (radius {1000 * K._FOREARM_RADIUS:.0f} mm, ends at the wrist): "
      f"lowest {1000 * (wrist[2] - K._FOREARM_RADIUS - pad_c):6.1f} mm")
print()
print("along the approach (+x), relative to the jaw-gap centre at -65 deg:")
tf = b["thumb"].center - b["thumb"].axes[:, 1] * b["thumb"].half[1]
ff = b["finger"].center + b["finger"].axes[:, 1] * b["finger"].half[1]
mid = 0.5 * (tf + ff)
for n in ("thumb", "finger", "palm", "back"):
    xs = b[n].corners()[:, 0] - mid[0]
    print(f"  {n:6s} x from {1000 * xs.min():6.1f} to {1000 * xs.max():6.1f} mm")
print(f"  wrist at {1000 * (wrist[0] - mid[0]):6.1f} mm")
