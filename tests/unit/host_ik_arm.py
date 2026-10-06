"""A stub arm whose FK/IK mirror the simulator's fake server, for host tests.

`fake_reachy_server._arm_fk` / `_arm_ik` are what the simulator's arm
kinematics service evaluates: scipy L-BFGS-B over the same joint bounds and the
same cost.  The server module cannot be imported on the host (it needs grpc),
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
#: 7094ab0.
SOURCE_SHA256 = "640d1d1ade3f0be30f9de92c6864b486604aae76f3804495a3466b2d16b33732"

_R_SHOULDER_Y = -0.19
_UPPER_ARM = 0.28
_FOREARM = 0.25
_R_ARM_LIMITS = [
    (-3.14, 1.57), (-1.57, 0.17), (-2.09, 2.09),
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


def arm_fk(q):
    """Radians in, 4x4 wrist pose (arm frame) out -- `_arm_fk(q, "right")`."""
    T = _tx(0.0, _R_SHOULDER_Y, 0.0)
    T = T @ _ry(q[0]) @ _rx(q[1]) @ _rz(q[2])
    T = T @ _tx(0, 0, -_UPPER_ARM)
    T = T @ _ry(q[3]) @ _rz(q[4])
    T = T @ _tx(0, 0, -_FOREARM)
    T = T @ _ry(q[5]) @ _rx(q[6])
    return T


def arm_ik(target, q0=None):
    """`_arm_ik(target, "right", q0)`: radians in and out."""
    q_init = np.zeros(7) if q0 is None else np.asarray(q0, dtype=float)[:7]

    def cost(q):
        T = arm_fk(q)
        return (np.sum((T[:3, 3] - target[:3, 3]) ** 2)
                + 0.1 * np.sum((T[:3, :3] - target[:3, :3]) ** 2))

    res = minimize(cost, q_init, method="L-BFGS-B", bounds=_R_ARM_LIMITS,
                   options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-6})
    return res.x, bool(res.fun < 0.01)


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


_SERVER_NAMES = {"_R_SHOULDER_Y", "_UPPER_ARM", "_FOREARM", "_R_ARM_LIMITS"}
_SERVER_FUNCS = ("_rx", "_ry", "_rz", "_tx", "_arm_fk", "_arm_ik")


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
