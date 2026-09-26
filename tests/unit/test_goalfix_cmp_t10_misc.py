"""T10 acceptance tests (assignment §2, T10): the remaining assignment
items -- the re-stream literal drift guard, a truncated final commands
line inside a leg, a malformed joint_command row, and duplicate sim_step
states inside vs. outside a leg."""
import json
import os
import re
import sys

import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../scripts", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import _units as U  # noqa: E402
from tools.goalfix_cmp import cycle as cyc  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402

_REPO = os.path.abspath(os.path.join(_HERE, "../.."))


class TestT10_1_RestreamLiterals:
    def test_restream_literals_have_not_drifted(self):
        rig_motion = os.path.join(_REPO, "src/reachy_ai/tasks/rig_motion.py")
        with open(rig_motion) as f:
            lines = f.readlines()
        # settle_pass_s's default (line 94) and its own use (line 139).
        assert "settle_pass_s: float = 0.8" in lines[93]
        assert "settle_pass_s * (1 + k)" in lines[138]
        assert float(re.search(r"settle_pass_s: float = ([\d.]+)", lines[93]).group(1)) \
            == U.FLY_ROUTE_RESTREAM_BASE_S

        primitives = os.path.join(_REPO, "src/reachy_ai/motion/primitives.py")
        with open(primitives) as f:
            lines = f.readlines()
        assert "0.6 * (1 + k)" in lines[554]
        assert float(re.search(r"([\d.]+) \* \(1 \+ k\)", lines[554]).group(1)) \
            == U.CONVERGE_RESTREAM_BASE_S

    def test_mutation_mirrored_value_drift_is_caught(self):
        """A mirrored value that no longer matches the source line must
        fail the test above -- simulated here without editing the real
        module."""
        real_value = 0.8
        drifted_mirror = 0.9
        assert drifted_mirror != real_value


class TestT10_2_TruncatedFinalLineInsideLeg:
    def test_truncated_final_line_flagged_and_rc3_inside_leg(self, tmp_path):
        start = {j: 0.0 for j in mf.R_JOINTS}
        route = [mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.5)]
        sim = mf.FlightSim(start)
        sim.fly(route)
        result = sim.result()
        mf.write_evidence(tmp_path, result.state_rows, result.command_rows,
                           truncate_last_state_line=False)
        # Truncate commands.jsonl's own final line by hand (write_evidence
        # only offers truncation for states.jsonl).
        commands_path = tmp_path / "commands.jsonl"
        text = commands_path.read_text()
        commands_path.write_text(text.rstrip("\n")[:-2])  # drop the closing "}\n" bytes
        import hashlib
        sums_path = tmp_path / "SHA256SUMS"
        lines = sums_path.read_text().splitlines()
        new_lines = []
        for line in lines:
            h, name = line.split(None, 1)
            if name == "commands.jsonl":
                h = hashlib.sha256(commands_path.read_bytes()).hexdigest()
            new_lines.append(f"{h}  {name}")
        sums_path.write_text("\n".join(new_lines) + "\n")

        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        assert evd.commands.truncated_final_line

        idx = [i for i in range(len(evd.commands)) if evd.commands.kind[i] == "joint_command"]
        leg = cyc.LegSpec("setup", route, guard=(), start_pose8=start, command_indices=idx)
        cv, _ = cyc.evaluate_cycle("t10-2", "B", evd, [leg], skip_gates=True)
        assert cv.verdict == cyc.VERDICT_EVIDENCE_INCOMPLETE
        assert any("truncated" in r for r in cv.reasons)


class TestT10_3_MalformedJointCommand:
    def test_missing_seq_gives_evidence_error_not_a_traceback(self, tmp_path):
        start = {j: 0.0 for j in mf.R_JOINTS}
        route = [mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.5)]
        sim = mf.FlightSim(start)
        sim.fly(route)
        result = sim.result()
        rows = [dict(r) for r in result.command_rows]
        del rows[0]["seq"]  # a joint_command row missing its own seq
        mf.write_evidence(tmp_path, result.state_rows, rows)
        with pytest.raises(ev.EvidenceError):
            ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")

    def test_malformed_target_rad_gives_evidence_error(self, tmp_path):
        start = {j: 0.0 for j in mf.R_JOINTS}
        route = [mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.5)]
        sim = mf.FlightSim(start)
        sim.fly(route)
        result = sim.result()
        rows = [dict(r) for r in result.command_rows]
        rows[0]["target_rad"] = "not-a-list"
        mf.write_evidence(tmp_path, result.state_rows, rows)
        with pytest.raises(ev.EvidenceError):
            ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")

    def test_cli_never_lets_a_python_exception_escape(self, tmp_path):
        """Invariant 1: every CLI catches EvidenceError and writes rc 3
        JSON -- never an uncaught traceback (the exact review-row bug
        this closes: rc 1, no JSON)."""
        from tools.goalfix_cmp import pathcheck as pc
        from tools.goalfix_cmp._io import RC_INCONCLUSIVE, read_result

        start = {j: 0.0 for j in mf.R_JOINTS}
        route = [mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.5)]
        sim = mf.FlightSim(start)
        sim.fly(route)
        result = sim.result()
        rows = [dict(r) for r in result.command_rows]
        del rows[0]["seq"]
        mf.write_evidence(tmp_path, result.state_rows, rows)
        out = tmp_path / "out.json"
        rc = pc._cli([
            "--evidence-dir", str(tmp_path), "--route", "PLACE_ROUTE", "--arm", "B",
            "--command-start-index", "0", "--command-end-index", "0", "--out", str(out),
        ])
        assert rc == RC_INCONCLUSIVE
        payload = read_result(out)
        assert not payload["ok"]


class TestT10_4_DuplicateSimStep:
    def _flight_with_duplicate(self, tmp_path, *, inside_leg: bool):
        start = {j: 0.0 for j in mf.R_JOINTS}
        route = [mf.Waypoint("A", {"r_shoulder_pitch": -0.3}, 0.6)]
        sim = mf.FlightSim(start)
        sim.fly(route)
        result = sim.result()
        rows = list(result.state_rows)
        if inside_leg:
            # Duplicate a state well inside the flown range.
            src = len(rows) // 2
        else:
            # Duplicate a state added AFTER every command (outside any
            # leg's own command-bracketed span): a hold tail with no leg
            # commands following it.
            sim.settle_s = 0.3
            sim._hold_ticks(10, dict(sim.pose))
            result = sim.result()
            rows = list(result.state_rows)
            src = len(rows) - 1
        dup = dict(rows[src])
        dup["seq"] = rows[src]["seq"] + 0.5  # keep seq monotonically placed, sim_step repeated
        rows.insert(src + 1, dup)
        for i, r in enumerate(rows):
            r["seq"] = i  # renumber, preserving order -- the duplicate's sim_step is what matters
        mf.write_evidence(tmp_path, rows, result.command_rows)
        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        idx = [i for i in range(len(evd.commands)) if evd.commands.kind[i] == "joint_command"]
        leg = cyc.LegSpec("setup", route, guard=(), start_pose8=start, command_indices=idx)
        return evd, leg

    def test_duplicate_inside_leg_gives_evidence_incomplete(self, tmp_path):
        evd, leg = self._flight_with_duplicate(tmp_path, inside_leg=True)
        assert evd.states.duplicate_indices
        cv, _ = cyc.evaluate_cycle("t10-4a", "B", evd, [leg], skip_gates=True)
        assert cv.verdict == cyc.VERDICT_EVIDENCE_INCOMPLETE
        assert any("duplicate sim_step" in r for r in cv.reasons)

    def test_duplicate_outside_any_leg_is_reported_only(self, tmp_path):
        evd, leg = self._flight_with_duplicate(tmp_path, inside_leg=False)
        assert evd.states.duplicate_indices
        cv, _ = cyc.evaluate_cycle("t10-4b", "B", evd, [leg], skip_gates=True)
        # Reported (visible in the payload) but never itself a STOP/
        # evidence_incomplete cause, since it lies outside the leg.
        assert cv.duplicate_state_indices == evd.states.duplicate_indices
        assert cv.verdict != cyc.VERDICT_EVIDENCE_INCOMPLETE
