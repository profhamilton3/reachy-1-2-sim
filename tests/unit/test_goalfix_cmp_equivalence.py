"""Equivalence tests: ported/reused functions vs their vendored originals.

Each original is vendored verbatim under tests/fixtures/goalfix_cmp/reference/
and checked against a pinned sha256 before any of its code is used, so a
silent edit to the vendored copy (or the original going stale) fails loudly
here rather than quietly invalidating the equivalence claim.

The vendored ``verify_goal_feedback.py`` is never imported as a module --
its top-level script body opens real Stage 2 evidence paths on import, which
this test suite must never touch outside the narrow, explicit D8 read in
test_goalfix_cmp_historical.py. Only the ``classify`` function (and the
``LOOKBACK_NS`` constant it closes over) is extracted via ``ast`` and
``exec``'d in an isolated namespace.
"""
import ast
import hashlib
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp import echo  # noqa: E402
from tools.goalfix_cmp import _io as gio  # noqa: E402
from tools.goalfix_cmp import clearance as cl  # noqa: E402
from tools.goalfix_cmp import _minjerk

_REFERENCE_DIR = os.path.join(_HERE, "../fixtures/goalfix_cmp/reference")
_VERIFY_GOAL_FEEDBACK_PATH = os.path.join(_REFERENCE_DIR, "verify_goal_feedback.py")
_VERIFY_GOAL_FEEDBACK_SHA256 = (
    "9862d8f15244e423ffe55b79e73418719c2af90c460c5c199ffed8cd2b2cdd5b")
_HOVER_WRISTBALL_PATH = os.path.join(_REFERENCE_DIR, "hover_wristball.py")
_HOVER_WRISTBALL_SHA256 = (
    "134205897253f7fcfff9c346b0543c82a3d3494c4e5fcfad436203c815c6ecb3")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_from_vendored(*names: str):
    """Extract only the named top-level defs/assigns from the vendored
    verify_goal_feedback.py -- never executes the file's module-level
    script body, which opens real Stage 2 evidence on import."""
    actual = _sha256(_VERIFY_GOAL_FEEDBACK_PATH)
    assert actual == _VERIFY_GOAL_FEEDBACK_SHA256, (
        "vendored verify_goal_feedback.py has drifted from the pinned "
        f"original (expected {_VERIFY_GOAL_FEEDBACK_SHA256}, got {actual})")

    source = open(_VERIFY_GOAL_FEEDBACK_PATH).read()
    tree = ast.parse(source)
    wanted = set(names)
    nodes = [n for n in tree.body
             if (isinstance(n, ast.FunctionDef) and n.name in wanted)
             or (isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in wanted for t in n.targets))]
    module = ast.Module(body=nodes, type_ignores=[])
    import hashlib as _hashlib
    import os as _os
    ns = {"np": np, "hashlib": _hashlib, "os": _os}
    exec(compile(module, _VERIFY_GOAL_FEEDBACK_PATH, "exec"), ns)
    for name in names:
        assert name in ns, f"{name!r} not found in vendored verify_goal_feedback.py"
    return ns


def _load_original_classify():
    return _extract_from_vendored("classify", "LOOKBACK_NS")["classify"]


class TestEchoClassifyEquivalence:
    def test_genuine_echo_count_matches_original_on_single_epoch_flight(self, tmp_path):
        original_classify = _load_original_classify()

        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS}, seed=11)
        route = [mf.Waypoint(
            "A", {"r_shoulder_pitch": -0.6, "r_elbow_pitch": -0.4,
                  "r_arm_yaw": 0.15, "r_wrist_pitch": -0.25}, 3.0)]
        sim.fly(route, echo_rate=0.3)
        result = sim.result()
        mf.write_evidence(tmp_path, result.state_rows, result.command_rows)
        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")

        mine = echo.classify_commands(evd)
        mine_counts = echo.count_labels(mine)
        # A clean single-epoch flight with no goto_context: the only exact
        # matches this scenario can produce are the forced echoes
        # themselves -- confirm no subclass/ambiguity muddies the compare.
        assert mine_counts.start_coincidence == 0
        assert mine_counts.path_coincidence == 0
        assert mine_counts.timing_ambiguous == 0

        states = evd.states
        commands = evd.commands
        jc_idx = np.nonzero(commands.joint_command_mask())[0]
        cmds_orig = [(int(commands.seq[i]), list(commands.target_rad[i, :8]), None)
                     for i in jc_idx]
        st_orig = states.wall_time_ns.astype(np.int64)
        spos32_orig = states.position_rad[:, :8].astype(np.float32).astype(np.float64)
        t_hi_orig = np.array(
            [states.wall_time_ns[evd.brackets[i].hi_state_index] for i in jc_idx],
            dtype=np.int64)

        cls, _age = original_classify(cmds_orig, st_orig, spos32_orig, t_hi_orig)
        original_echo_count = int((cls == 1).sum())

        assert original_echo_count == mine_counts.genuine_echo
        assert original_echo_count == len(result.forced_echoes)


class TestIndepWristBallEquivalence:
    def test_matches_vendored_hover_wristball(self):
        actual = _sha256(_HOVER_WRISTBALL_PATH)
        assert actual == _HOVER_WRISTBALL_SHA256, (
            "vendored hover_wristball.py has drifted from the pinned original "
            f"(expected {_HOVER_WRISTBALL_SHA256}, got {actual})")
        source = open(_HOVER_WRISTBALL_PATH).read()
        tree = ast.parse(source)
        wanted = {"indep_wrist_ball", "_Rx", "_Ry", "_Rz"}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
        assert len(nodes) == 4, "indep_wrist_ball (or its _Rx/_Ry/_Rz helpers) not found"
        module = ast.Module(body=nodes, type_ignores=[])

        import math as _math
        from reachy_ai.motion import rig_routes as R
        ns = {"np": np, "math": _math, "R": R}
        exec(compile(module, _HOVER_WRISTBALL_PATH, "exec"), ns)
        original_indep_wrist_ball = ns["indep_wrist_ball"]

        rng = np.random.default_rng(3)
        center, size = (0.5, -0.3, 1.1), (0.1, 0.1, 0.1)
        for _ in range(20):
            J = {j: float(v) for j, v in zip(R.ARM7, rng.uniform(-60, 60, size=7))}
            J["r_gripper"] = float(rng.uniform(-45, 20))
            want = original_indep_wrist_ball(J, center, size)
            got = cl.indep_wrist_ball(J, center, size)
            assert got == pytest.approx(want, abs=1e-9)


class TestIntegrityCheckEquivalence:
    def test_sha_sums_of_verified_agree_with_io_module(self, tmp_path):
        ns = _extract_from_vendored("sha", "sums_of", "verified")
        original_sha, original_sums_of, original_verified = (
            ns["sha"], ns["sums_of"], ns["verified"])

        d = tmp_path
        (d / "a.txt").write_bytes(b"hello world")
        (d / "b.txt").write_bytes(b"goodbye world")
        sums = {"a.txt": original_sha(str(d / "a.txt")),
                "b.txt": original_sha(str(d / "b.txt"))}
        (d / "SHA256SUMS").write_text(
            "\n".join(f"{h}  {name}" for name, h in sums.items()) + "\n")

        assert gio.sha256_of(d / "a.txt") == original_sha(str(d / "a.txt"))
        assert gio.parse_sha256sums(d / "SHA256SUMS") == original_sums_of(str(d))

        original_path = original_verified(str(d), "a.txt", original_sums_of(str(d)))
        mine_path = gio.verified(d, "a.txt", gio.parse_sha256sums(d / "SHA256SUMS"))
        assert str(mine_path) == original_path

        (d / "a.txt").write_bytes(b"tampered")
        with pytest.raises(AssertionError):
            original_verified(str(d), "a.txt", original_sums_of(str(d)))
        with pytest.raises(gio.IntegrityError):
            gio.verified(d, "a.txt", gio.parse_sha256sums(d / "SHA256SUMS"))


class TestMinJerkEquivalence:
    def test_matches_reachy_sdk_trajectory(self):
        """Optional pin (assignment §2.4): our local minimum-jerk formula
        against reachy_sdk.trajectory.interpolation.minimum_jerk with zero
        boundary velocity/acceleration (its default, and the only case a
        goto ever uses). Skipped wherever reachy_sdk isn't installed --
        the system interpreter never has it; ~/goalfix-venv does."""
        pytest.importorskip("reachy_sdk")
        from reachy_sdk.trajectory.interpolation import minimum_jerk as sdk_minimum_jerk

        rng = np.random.default_rng(0)
        for _ in range(20):
            a, b, duration = rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0.1, 5.0)
            f = sdk_minimum_jerk(np.array([a]), np.array([b]), duration)
            for frac in (0.0, 0.1, 0.37, 0.5, 0.8, 1.0):
                t = frac * duration
                want = float(f(t)[0])
                got = _minjerk.pose_at(a, b, frac)
                assert got == pytest.approx(want, abs=1e-9)
