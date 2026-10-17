"""#185 analysis: taught board poses vs the sim's arm model (numpy only).

Usage: python3 analyze.py taught_poses.json > analysis.txt
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from urdf_fk import fk  # noqa: E402
from ik import solve    # noqa: E402

np.set_printoptions(precision=4, suppress=True)
d = json.load(open(sys.argv[1]))
bp = {k: np.array(v) for k, v in d["board_poses"].items()}
keys = sorted(bp)
# native_mujoco/model/reachy_1_2.xml right-arm joint ranges (rad), in degrees
SIM = np.degrees(np.array([(-2.618, 1.57), (-3.14, 0.174), (-1.57, 1.57), (-2.182, 0.0),
                           (-1.745, 1.745), (-0.785, 0.785), (-0.785, 0.785)]))


def rot_deg(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2, -1, 1))))


print("== 1. Frame and tool point")
q = np.array(d["fk_pair"]["joints_deg"])
T = bp["B2"].copy(); T[2, 3] += d["fk_pair"]["safe_height_m"]
F = fk(q)
print("URDF torso->right_tip FK of B2_SAFE_JOINTS vs BOARD_POSES[B2]+SAFE_HEIGHT: "
      "pos %.3f mm, rot %.4f deg" % (1000 * np.linalg.norm(F[:3, 3] - T[:3, 3]), rot_deg(F, T)))
W = fk(q, "wrist")
print("same joints, wrist (what fake_reachy_server returns) is %.4f m from the tip"
      % np.linalg.norm(W[:3, 3] - F[:3, 3]))
print("r_shoulder_roll at B2_SAFE: %.1f deg (sim stop %.1f)" % (q[1], SIM[1, 1]))

print("\n== 2. Reach of the 9 taught 6-D tip poses inside sim joint travel")
print("order: sh_pitch sh_roll arm_yaw elbow forearm_yaw wr_pitch wr_roll")
seeds = [q, np.array([10, 0, 20, -95, 25, -5, 40.]), np.array([0, -10, 40, -100, 0, 0, 45.]),
         np.array([30, 10, 10, -120, 40, 10, 30.]), np.array([-10, -20, 60, -80, -20, 0, 0.])]
for k in keys:
    qs, pe, ae = min((solve(bp[k], np.clip(s, SIM[:, 0], SIM[:, 1]), SIM) for s in seeds),
                     key=lambda r: r[1] + r[2] / 2)
    stops = [i for i in range(7) if min(qs[i] - SIM[i, 0], SIM[i, 1] - qs[i]) < 0.5]
    print(f"{k} q {np.round(qs, 1)} err {pe:.1f} mm / {ae:.1f} deg  on stop: {stops or '-'}")

print("\n== 3. Taught grid (affine fit; rows near->far C,B,A; columns 1..3 = +y..-y)")
P = np.array([bp[k][:3, 3] for k in keys])
hand = np.array([-bp[k][:3, 2] for k in keys])
print("hand axis vertical component, max |z|: %.3f (gripper horizontal)" % np.abs(hand[:, 2]).max())
I = {"C": -1, "B": 0, "A": 1}
A = np.c_[np.ones(9), [[I[k[0]], int(k[1]) - 2] for k in keys]]
C, *_ = np.linalg.lstsq(A, P, rcond=None); O, u, v = C; r = P - A @ C
print("centre", O, " row step", u, " column step", v)
print("|row| %.4f m  |col| %.4f m  angle %.1f deg  rows %.1f deg off +x  z range %.4f..%.4f"
      % (np.linalg.norm(u), np.linalg.norm(v),
         np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v))),
         np.degrees(np.arctan2(u[1], u[0])), P[:, 2].min(), P[:, 2].max()))
for k, rr in zip(keys, r):
    print(f"  {k} residual mm {np.round(1000 * rr, 1)}  |{1000 * np.linalg.norm(rr):.1f}|")
print("rms %.1f mm, max %.1f mm" % (1000 * np.sqrt((r ** 2).sum(1).mean()), 1000 * np.linalg.norm(r, axis=1).max()))

print("\n== 4. Context only: MCC/SivaPool grid (a different lab) in the torso frame")
print("sim cell_r2c2 (0.4318, 0.0, 0.74 - 1.0), pitch 0.1524; taught minus sim centre:",
      np.round(O - np.array([0.4318, 0.0, -0.26]), 4))
