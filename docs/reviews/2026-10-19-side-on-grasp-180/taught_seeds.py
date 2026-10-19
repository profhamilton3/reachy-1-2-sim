"""Siva's taught tip orientations as IK seeds for the side-on grasp (#180).

Input: ../2026-10-14-taught-board-poses-185/taught_poses.json (torso frame,
right_tip 4x4, read back on the real robot).  These are used as ATTITUDES and
joint-space SEEDS only, never as positions: the taught spacing is still open
(#185, 0.84x / 0.92x).

For each taught square: the hand's approach azimuth (the tip's -z, flattened)
and its tilt from level, and one joint solution of the full taught 6-D tip
pose inside the sim's joint travel (MJCF ranges, r_shoulder_roll to +20.3 deg
since #192), from the host mirror of the simulator's IK
(`tests/unit/host_ik_arm.py`), multi-start over `crane_pick_live.GRASP_STARTS`.
The square -> cell map is #185's: A = r3 (far), C = r1 (near), columns agree.

Run: python3 taught_seeds.py > taught_seeds.txt   (writes taught_seeds.json)
"""
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO / "src", REPO / "tests" / "unit"):
    sys.path.insert(0, str(p))

from host_ik_arm import HostIKArm  # noqa: E402
from reachy_ai.motion.kinematics import tip_pose, within_limits  # noqa: E402
from reachy_ai.tasks.crane_pick_live import GRASP_STARTS  # noqa: E402

taught = json.loads((REPO / "docs/reviews/2026-10-14-taught-board-poses-185/taught_poses.json")
                    .read_text())
CELL = {"A": "r3", "B": "r2", "C": "r1"}
arm = HostIKArm()
out = {"source": "docs/reviews/2026-10-14-taught-board-poses-185/taught_poses.json",
       "squares": {}}
print("square cell   azimuth  tilt_down  pos_err_mm rot_err_deg  joints_deg")
for sq, M in taught["board_poses"].items():
    M = np.asarray(M, dtype=float)
    app = -M[:3, 2]
    az = math.degrees(math.atan2(app[1], app[0]))
    tilt = math.degrees(math.asin(max(-1.0, min(1.0, -app[2]))))
    best = None
    for s in [[0.0] * 7] + GRASP_STARTS:
        q = arm.inverse_kinematics(M, q0=s)
        if not within_limits(q):
            continue
        F = tip_pose(q)
        F[:3, 3] -= 0.0   # tip_pose is already in the torso frame
        pe = float(np.linalg.norm(F[:3, 3] - M[:3, 3]))
        re = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(F[:3, :3].T @ M[:3, :3]) - 1) / 2))))
        if best is None or pe + 0.001 * re < best[0] + 0.001 * best[1]:
            best = (pe, re, [round(float(v), 3) for v in q])
    cell = CELL[sq[0]] + "c" + sq[1]
    out["squares"][sq] = {"cell": cell, "approach_azimuth_deg": round(az, 2),
                          "tilt_down_deg": round(tilt, 2), "pos_err_mm": round(1000 * best[0], 3),
                          "rot_err_deg": round(best[1], 3), "joints_deg": best[2]}
    print(f"{sq:6s} {cell:5s} {az:8.1f} {tilt:9.1f} {1000 * best[0]:10.3f} {best[1]:10.3f}  {best[2]}")
(HERE / "taught_seeds.json").write_text(json.dumps(out, indent=1) + "\n")
