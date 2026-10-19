"""A stub arm whose FK/IK mirror the simulator's fake server, for host tests.

`fake_reachy_server._arm_fk` / `_arm_ik` are what the simulator's arm
kinematics service evaluates: the gripper tip (URDF right_tip, #187), and an
IK that solves the wrist with scipy L-BFGS-B over the same joint bounds and the
same cost, then re-solves the joints above the roll for the tip.  The server module cannot be imported on the host (it needs grpc),
so this file re-states the maths and `SOURCE_SHA256` pins it to the server's
source: `source_digest()` extracts the relevant definitions with `ast`,
whitespace-normalises them and hashes them, and a test compares the two.  If
the server's kinematics change, that test fails and this mirror must be
re-derived.

Degrees in, degrees out, exactly like the SDK arm the planner is given.
Requires scipy (callers use `pytest.importorskip("scipy")`).
"""

import ast
import hashlib
import os
import re

import numpy as np
from scipy.optimize import minimize

#: sha256 of the whitespace-stripped source of the server's arm-kinematics
#: definitions (see `source_digest`), recorded 2026-10-06 at origin/main
#: 7094ab0; re-recorded 2026-10-17 for #192, whose only change to them is
#: r_shoulder_roll's upper bound, 0.17 -> 0.354 (docs/adr/0010); re-recorded
#: 2026-10-19 for #187, FK/IK in the tip frame.
SOURCE_SHA256 = "fa63d6775e811b69b4ed7a5a79723727a6404eb2bf0821bbf5cad7efae6abeb6"

_R_SHOULDER_Y = -0.19
_UPPER_ARM = 0.28
_FOREARM = 0.25
_WRIST_ROLL_ORIGIN = (0.0, 0.0, -0.0325)
_R_TIP = (0.0, -0.012, -0.085)
_R_ARM_LIMITS = [
    (-3.14, 1.57), (-1.57, 0.354), (-2.09, 2.09),
    (-2.35, 0.0), (-2.62, 2.62), (-0.79, 0.79), (-0.79, 0.79),
]


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=float)


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=float)


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)


def _tx(x, y, z):
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    return T


def wrist_fk(q):
    """Radians in, 4x4 wrist pose (arm frame) out -- `_wrist_fk(q, "right")`."""
    T = _tx(0.0, _R_SHOULDER_Y, 0.0)
    T = T @ _ry(q[0]) @ _rx(q[1]) @ _rz(q[2])
    T = T @ _tx(0, 0, -_UPPER_ARM)
    T = T @ _ry(q[3]) @ _rz(q[4])
    T = T @ _tx(0, 0, -_FOREARM)
    T = T @ _ry(q[5]) @ _rx(q[6])
    return T


def tip_offset(R, roll):
    """`_tip_offset(R, roll, "right")`."""
    return R @ _rx(-roll)[:3, :3] @ np.array(_WRIST_ROLL_ORIGIN) + R @ np.array(_R_TIP)


def arm_fk(q):
    """Radians in, 4x4 tip pose (arm frame) out -- `_arm_fk(q, "right")`."""
    T = wrist_fk(q)
    T[:3, 3] += tip_offset(T[:3, :3], q[6])
    return T


def wrist_ik(target, q0=None):
    """`_wrist_ik(target, "right", q0)`: radians in and out."""
    q_init = np.zeros(7) if q0 is None else np.asarray(q0, dtype=float)[:7]

    def cost(q):
        T = wrist_fk(q)
        return (np.sum((T[:3, 3] - target[:3, 3]) ** 2)
                + 0.1 * np.sum((T[:3, :3] - target[:3, :3]) ** 2))

    res = minimize(cost, q_init, method="L-BFGS-B", bounds=_R_ARM_LIMITS,
                   options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-6})
    return res.x, bool(res.fun < 0.01)


_IK_LANDED = 1e-10


def wrist_residual(q, target):
    T = wrist_fk(q)
    return np.concatenate([T[:3, 3] - target[:3, 3],
                           np.sqrt(0.1) * (T[:3, :3] - target[:3, :3]).ravel()])


def resolve_above_roll(target, q, iters=100, damping=1e-6):
    """`_resolve_above_roll(target, "right", q)`."""
    lims = np.array(_R_ARM_LIMITS)[:6]
    lo, hi = lims[:, 0], lims[:, 1]
    q = np.array(q, dtype=float)
    q[:6] = np.clip(q[:6], lo, hi)
    r = wrist_residual(q, target)
    for _ in range(iters):
        J = np.zeros((len(r), 6))
        for i in range(6):
            dq = q.copy()
            dq[i] += 1e-7
            J[:, i] = (wrist_residual(dq, target) - r) / 1e-7
        free = np.ones(6, dtype=bool)
        for _k in range(3):
            Jf = J[:, free]
            step = np.zeros(6)
            step[free] = np.linalg.solve(Jf.T @ Jf + damping * np.eye(int(free.sum())), -Jf.T @ r)
            push = ((q[:6] <= lo + 1e-12) & (step < 0)) | ((q[:6] >= hi - 1e-12) & (step > 0))
            if not push.any():
                break
            free &= ~push
        alpha, moved = 1.0, False
        while alpha > 1e-4:
            qn = q.copy()
            qn[:6] = np.clip(q[:6] + alpha * step, lo, hi)
            rn = wrist_residual(qn, target)
            if rn @ rn < r @ r - 1e-14:
                q, r, moved = qn, rn, True
                break
            alpha *= 0.5
        if not moved:
            break
    return q, float(r @ r)


def arm_ik(target, q0=None):
    """`_arm_ik(target, "right", q0)`: radians in and out."""
    R = target[:3, :3]
    roll0 = 0.0 if q0 is None else float(np.asarray(q0, dtype=float)[6])
    wrist = np.array(target, dtype=float)
    wrist[:3, 3] = target[:3, 3] - tip_offset(R, roll0)
    q, success = wrist_ik(wrist, q0)
    wrist[:3, 3] = target[:3, 3] - tip_offset(R, q[6])
    r = wrist_residual(q, wrist)
    if float(r @ r) < _IK_LANDED:
        return q, True
    q2, cost = resolve_above_roll(wrist, q)
    if cost < _IK_LANDED:
        return q2, True
    return q, success


class HostIKArm:
    """The slice of the SDK arm `CartesianPlanner` uses: degrees in and out."""

    def __init__(self):
        self.ik_calls = 0

    def inverse_kinematics(self, M, q0=None):
        self.ik_calls += 1
        q0r = None if q0 is None else np.radians(np.asarray(q0, dtype=float))
        q, _ok = arm_ik(np.asarray(M, dtype=float), q0r)
        return list(np.degrees(q))

    def forward_kinematics(self, q):
        return arm_fk(np.radians(np.asarray(list(q)[:7], dtype=float)))


_SERVER_NAMES = {"_R_SHOULDER_Y", "_UPPER_ARM", "_FOREARM", "_R_ARM_LIMITS",
                 "_WRIST_ROLL_ORIGIN", "_R_TIP", "_IK_LANDED"}
_SERVER_FUNCS = ("_rx", "_ry", "_rz", "_tx", "_wrist_fk", "_tip_offset", "_arm_fk",
                 "_wrist_ik", "_wrist_residual", "_resolve_above_roll", "_arm_ik")


def source_digest(server_path=None):
    """sha256 of the server's arm-kinematics source, whitespace-stripped.

    Extracted with `ast` (the module cannot be imported: it needs grpc) and
    stripped of whitespace so a Python-version difference in formatting cannot
    change it.  Comments inside the extracted segments are part of the digest.
    """
    if server_path is None:
        server_path = os.path.join(os.path.dirname(__file__), "..", "..",
                                   "fake_reachy_server.py")
    with open(server_path, encoding="utf-8") as f:
        text = f.read()
    tree = ast.parse(text)
    wanted = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _SERVER_FUNCS:
            wanted[node.name] = ast.get_source_segment(text, node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in _SERVER_NAMES:
                    wanted[t.id] = ast.get_source_segment(text, node)
    missing = (_SERVER_NAMES | set(_SERVER_FUNCS)) - set(wanted)
    if missing:
        raise AssertionError(f"server definitions not found: {sorted(missing)}")
    blob = "\n".join(re.sub(r"\s+", "", wanted[k]) for k in sorted(wanted))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
