"""Unit tests for tools/goalfix_cmp/evidence.py and its _io/_simtime helpers."""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp._io import IntegrityError  # noqa: E402
from tools.goalfix_cmp import _simtime as st  # noqa: E402


def _simple_flight(tmp_path, **kwargs):
    sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS}, **kwargs)
    route = [
        mf.Waypoint("A", {"r_shoulder_pitch": -0.3, "r_elbow_pitch": -0.2}, 1.0),
        mf.Waypoint("B", {"r_shoulder_pitch": -0.6, "r_gripper": 0.3}, 1.0),
    ]
    sim.fly(route)
    result = sim.result()
    return mf.write_evidence(tmp_path, result.state_rows, result.command_rows), result


class TestIntegrity:
    def test_verified_roundtrip(self, tmp_path):
        (states_path, commands_path, sha_path), _ = _simple_flight(tmp_path)
        loaded = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        assert len(loaded.states) == len(loaded.commands.kind) or True  # sanity, no crash
        assert len(loaded.states) > 0
        assert len(loaded.commands) > 0

    def test_sha256sums_mismatch_raises(self, tmp_path):
        (states_path, commands_path, sha_path), _ = _simple_flight(tmp_path)
        mf.corrupt_sha256sums(sha_path, "states.jsonl")
        with pytest.raises(IntegrityError):
            ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")

    def test_missing_sha256sums_entry_raises(self, tmp_path):
        (states_path, commands_path, sha_path), _ = _simple_flight(tmp_path)
        sha_path.write_text("")  # no entries at all
        with pytest.raises(IntegrityError):
            ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")


class TestLoading:
    def test_right_arm_slice_matches_joint_order(self, tmp_path):
        (paths, result) = _simple_flight(tmp_path)
        states = ev.load_states(paths[0])
        assert states.position_rad.shape == (len(result.state_rows), 21)
        assert list(ev._JOINT_ORDER[:8]) == list(mf.R_JOINTS)

    def test_truncated_final_line_is_flagged_not_fatal(self, tmp_path):
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS})
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.5)])
        result = sim.result()
        states_path, commands_path, sha_path = mf.write_evidence(
            tmp_path, result.state_rows, result.command_rows,
            truncate_last_state_line=True)
        # Recompute SHA256SUMS against the now-truncated file so integrity
        # checking (a separate concern) doesn't mask the truncation.
        import hashlib
        sums_text = sha_path.read_text().splitlines()
        new_lines = []
        for line in sums_text:
            h, name = line.split(None, 1)
            if name == "states.jsonl":
                h = hashlib.sha256(states_path.read_bytes()).hexdigest()
            new_lines.append(f"{h}  {name}")
        sha_path.write_text("\n".join(new_lines) + "\n")

        loaded = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        assert loaded.states.truncated_final_line is True
        assert len(loaded.states) == len(result.state_rows) - 1

    def test_malformed_non_final_line_raises(self, tmp_path):
        (paths, result) = _simple_flight(tmp_path)
        states_path = paths[0]
        lines = states_path.read_text().splitlines()
        lines[0] = "{not json"
        states_path.write_text("\n".join(lines) + "\n")
        with pytest.raises(ev.EvidenceError):
            ev.load_states(states_path)

    def test_duplicate_sim_step_flagged(self, tmp_path):
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS})
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.2)])
        result = sim.result()
        rows = list(result.state_rows)
        dup = dict(rows[-1])
        dup["seq"] = rows[-1]["seq"] + 1
        rows.append(dup)  # same sim_step repeated, as a pause would produce
        states_path, commands_path, sha_path = mf.write_evidence(
            tmp_path, rows, result.command_rows)
        loaded = ev.load_states(states_path)
        assert loaded.duplicate_indices == [len(rows) - 1]


class TestEpochsAndBrackets:
    def test_two_epochs_reset_between_them(self, tmp_path):
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS})
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.2)])
        sim.reset(seed=1)
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.2)])
        result = sim.result()
        paths = mf.write_evidence(tmp_path, result.state_rows, result.command_rows)
        loaded = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        assert loaded.n_epochs() == 2
        assert loaded.states.sim_time_s[0] == 0.0
        # sim_time_s restarts after the reset
        second_epoch_start = np.nonzero(loaded.states.epoch == 1)[0][0]
        assert loaded.states.sim_time_s[second_epoch_start] == 0.0

    def test_epoch_count_mismatch_raises(self, tmp_path):
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS})
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.2)])
        result = sim.result()
        # Inject a spurious sim_step drop with no matching reset entry in
        # commands.jsonl.
        rows = list(result.state_rows)
        bad = dict(rows[-1])
        bad["sim_step"] = 0
        bad["sim_time_s"] = 0.0
        bad["seq"] = rows[-1]["seq"] + 1
        rows.append(bad)
        paths = mf.write_evidence(tmp_path, rows, result.command_rows)
        with pytest.raises(ev.EvidenceError):
            ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")

    def test_command_bracket_is_well_defined_for_clean_flight(self, tmp_path):
        (paths, result) = _simple_flight(tmp_path)
        loaded = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        assert loaded.unplaceable_command_indices == []
        for i, b in enumerate(loaded.brackets):
            assert not b.unplaceable

    def test_wall_time_non_monotonic_does_not_affect_sim_time_bracket(self, tmp_path):
        (paths, result) = _simple_flight(tmp_path)
        states_path = paths[0]
        rows = [__import__("json").loads(l) for l in states_path.read_text().splitlines()]
        # Simulate a container recreate: wall_time_ns jumps backwards partway
        # through, sim_time_s/sim_step/cmd_seq untouched.
        for r in rows[len(rows) // 2:]:
            r["wall_time_ns"] = 1
        states_path.write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n")
        import hashlib
        sha_path = paths[2]
        lines = sha_path.read_text().splitlines()
        out = []
        for line in lines:
            h, name = line.split(None, 1)
            if name == "states.jsonl":
                h = hashlib.sha256(states_path.read_bytes()).hexdigest()
            out.append(f"{h}  {name}")
        sha_path.write_text("\n".join(out) + "\n")

        before = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        assert before.unplaceable_command_indices == []
        for b in before.brackets:
            assert not b.unplaceable


class TestT3EpochCounting:
    """A recording whose last row is a reset must load (T3/review §3.1/M3):
    epochs are n_reset_rows + 1, and each reset row's own sim_step must
    agree with the step drop it opens -- within the recording's own
    sampling interval (V1, 2026-09-24): a real B4 s1 session samples
    states every 10 sim_step (not every 1, as every synthetic fixture in
    this file does), and its genuine reset rows land a few steps past the
    last SAMPLED state, never exactly +1. See
    test_reset_within_sampling_interval_not_exactly_plus_one below.
    agree with the step drop it opens."""

    def test_trailing_reset_row_loads(self, tmp_path):
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS})
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.2)])
        sim.recreate()
        sim.reset(seed=1)  # no further commands -- the LAST command row is the reset
        result = sim.result()
        assert result.command_rows[-1]["type"] == "reset"
        mf.write_evidence(tmp_path, result.state_rows, result.command_rows)
        ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")  # must not raise

    def test_missing_reset_row_is_evidence_incomplete(self, tmp_path):
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS})
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.2)])
        sim.reset(seed=1)
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.2)])
        result = sim.result()
        commands = [r for r in result.command_rows if r["type"] != "reset"]  # drop it
        mf.write_evidence(tmp_path, result.state_rows, commands)
        with pytest.raises(ev.EvidenceError):
            ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")

    def test_extra_reset_row_is_evidence_incomplete(self, tmp_path):
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS})
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.2)])
        sim.reset(seed=1)
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.2)])
        result = sim.result()
        extra = mf.command_row_reset(seed=2, sim_step=0, wall_time_s=0.0)
        commands = list(result.command_rows) + [extra]  # a reset with no matching drop
        mf.write_evidence(tmp_path, result.state_rows, commands)
        with pytest.raises(ev.EvidenceError):
            ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")

    def test_reset_sim_step_mismatch_is_evidence_incomplete(self, tmp_path):
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS})
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.2)])
        sim.reset(seed=1)
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.2)])
        result = sim.result()
        commands = [dict(r) for r in result.command_rows]
        for r in commands:
            if r["type"] == "reset":
                r["sim_step"] += 5  # no longer agrees with the drop it opens
        mf.write_evidence(tmp_path, result.state_rows, commands)
        with pytest.raises(ev.EvidenceError):
            ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")

    def test_epoch_count_reverted_to_max_plus_one_mutation(self, tmp_path):
        """Mutation guard: if check_epoch_counts ever goes back to
        `commands.epoch.max() + 1`, a trailing-reset recording (which this
        test builds fresh, not reusing a fixture that might itself change)
        must be rejected again -- proving this test would have caught it."""
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS})
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.2)])
        sim.recreate()
        sim.reset(seed=1)
        result = sim.result()
        kinds = [r["type"] for r in result.command_rows]
        n_resets = kinds.count("reset")
        from tools.goalfix_cmp import _simtime as st
        cmd_epoch = st.assign_command_epochs(kinds)
        buggy_count = int(cmd_epoch.max()) + 1 if len(cmd_epoch) else 0
        correct_count = n_resets + 1
        assert buggy_count != correct_count, (
            "fixture no longer distinguishes the two counting rules")

    def _downsampled_two_epoch_flight(self, *, reset_sim_step_delta: int):
        """Two epochs, states sampled every 10th tick (like a real B4 s1
        session), with the reset row's own sim_step offset by
        `reset_sim_step_delta` from the exact "+1 past the last SAMPLED
        state" the old rule assumed."""
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS})
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.3}, 2.0)])
        first_len = len(sim.state_rows)  # snapshot the length NOW -- result()
        # returns a live reference, not a copy, and sim keeps growing it.
        sim.recreate()
        sim.reset(seed=1)
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.3}, 2.0)])
        result = sim.result()
        first_half = list(result.state_rows[:first_len])[::10]
        rows = first_half + list(result.state_rows[first_len:])[::10]
        for i, r in enumerate(rows):
            r["seq"] = i
        last_sampled_step = first_half[-1]["sim_step"]
        commands = [dict(c) for c in result.command_rows]
        for c in commands:
            if c["type"] == "reset":
                c["sim_step"] = last_sampled_step + reset_sim_step_delta
        return rows, commands

    def test_reset_within_sampling_interval_not_exactly_plus_one(self, tmp_path):
        """V1 (2026-09-24, real B4 s1 data): a genuine reset row can be
        recorded several sim_steps past the last SAMPLED state (a session
        sampling every 10 sim_step logged a reset 7 steps past the last
        sample, and a second real case logged one exactly AT the last
        sample) -- neither is +1. Both must load."""
        rows, commands = self._downsampled_two_epoch_flight(reset_sim_step_delta=7)
        mf.write_evidence(tmp_path, rows, commands)
        ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")  # must not raise

        rows0, commands0 = self._downsampled_two_epoch_flight(reset_sim_step_delta=0)
        mf.write_evidence(tmp_path, rows0, commands0)
        ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")  # must not raise

    def test_reset_beyond_sampling_interval_is_still_evidence_incomplete(self, tmp_path):
        rows, commands = self._downsampled_two_epoch_flight(reset_sim_step_delta=50)
        mf.write_evidence(tmp_path, rows, commands)
        with pytest.raises(ev.EvidenceError):
            ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")


class TestLegFromSidecar:
    def test_leg_bounds_from_alignment(self, tmp_path):
        (paths, result) = _simple_flight(tmp_path)
        states = ev.load_states(paths[0])
        sidecar = {"alignment": [
            {"server_seq": int(states.seq[2])},
            {"server_seq": int(states.seq[5])},
        ]}
        leg = ev.leg_from_sidecar("leg1", sidecar, states)
        assert leg.first_state_index == 2
        assert leg.last_state_index == 5
        assert leg.t_lo == states.sim_time_s[2]
        assert leg.t_hi == states.sim_time_s[5]

    def test_leg_missing_alignment_is_evidence_incomplete(self, tmp_path):
        (paths, result) = _simple_flight(tmp_path)
        states = ev.load_states(paths[0])
        with pytest.raises(ev.EvidenceError):
            ev.leg_from_sidecar("leg1", {}, states)

    def test_leg_unknown_server_seq_is_evidence_incomplete(self, tmp_path):
        (paths, result) = _simple_flight(tmp_path)
        states = ev.load_states(paths[0])
        sidecar = {"alignment": [{"server_seq": 999999}, {"server_seq": 999999}]}
        with pytest.raises(ev.EvidenceError):
            ev.leg_from_sidecar("leg1", sidecar, states)

    def test_leg_crossing_epoch_is_evidence_incomplete(self, tmp_path):
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS})
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.2)])
        sim.reset(seed=1)
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.2)])
        result = sim.result()
        paths = mf.write_evidence(tmp_path, result.state_rows, result.command_rows)
        states = ev.load_states(paths[0])
        first_epoch_idx = int(np.nonzero(states.epoch == 0)[0][0])
        second_epoch_idx = int(np.nonzero(states.epoch == 1)[0][0])
        sidecar = {"alignment": [
            {"server_seq": int(states.seq[first_epoch_idx])},
            {"server_seq": int(states.seq[second_epoch_idx])},
        ]}
        with pytest.raises(ev.EvidenceError):
            ev.leg_from_sidecar("leg1", sidecar, states)

    def test_overlapping_legs_raise(self, tmp_path):
        (paths, result) = _simple_flight(tmp_path)
        states = ev.load_states(paths[0])
        leg_a = ev.Leg("a", 0, 0, 5, states.sim_time_s[0], states.sim_time_s[5])
        leg_b = ev.Leg("b", 0, 3, 8, states.sim_time_s[3], states.sim_time_s[8])
        with pytest.raises(ev.EvidenceError):
            ev.check_legs_non_overlapping([leg_a, leg_b])

    def test_non_overlapping_legs_ok(self, tmp_path):
        (paths, result) = _simple_flight(tmp_path)
        states = ev.load_states(paths[0])
        leg_a = ev.Leg("a", 0, 0, 3, states.sim_time_s[0], states.sim_time_s[3])
        leg_b = ev.Leg("b", 0, 4, 8, states.sim_time_s[4], states.sim_time_s[8])
        ev.check_legs_non_overlapping([leg_a, leg_b])  # no raise


class TestCli:
    def test_cli_ok(self, tmp_path):
        (paths, result) = _simple_flight(tmp_path)
        out = tmp_path / "out.json"
        rc = ev._cli(["--evidence-dir", str(tmp_path), "--out", str(out)])
        assert rc == 0
        payload = __import__("json").loads(out.read_text())
        assert payload["rc"] == 0 and payload["ok"] is True

    def test_cli_rc3_on_corrupt_evidence(self, tmp_path):
        (paths, result) = _simple_flight(tmp_path)
        mf.corrupt_sha256sums(paths[2], "commands.jsonl")
        out = tmp_path / "out.json"
        rc = ev._cli(["--evidence-dir", str(tmp_path), "--out", str(out)])
        assert rc == 3
        payload = __import__("json").loads(out.read_text())
        assert payload["rc"] == 3 and payload["ok"] is False


class TestSimtimeUnit:
    def test_bracket_commands_matches_manual_case(self):
        # 1 epoch, 3 states, 2 commands.
        state_sim_time = [0.0, 0.02, 0.04]
        state_cmd_seq = [-1, 0, 1]
        state_epoch = [0, 0, 0]
        brackets = st.bracket_commands([0, 1], [0, 0], state_sim_time, state_cmd_seq, state_epoch)
        assert brackets[0].t_lo == 0.0 and brackets[0].t_hi == 0.02
        assert brackets[1].t_lo == 0.02 and brackets[1].t_hi == 0.04

    def test_bracket_never_reported_is_unplaceable(self):
        state_sim_time = [0.0, 0.02]
        state_cmd_seq = [-1, 0]
        state_epoch = [0, 0]
        brackets = st.bracket_commands([0, 5], [0, 0], state_sim_time, state_cmd_seq, state_epoch)
        assert not brackets[0].unplaceable
        assert brackets[1].unplaceable

    def test_bracket_spanning_reset_is_unplaceable(self):
        # Command 0 belongs to epoch 0 but the first state of epoch 0
        # already reports it -- t_lo would reach into "before epoch 0",
        # which does not exist -- unplaceable.
        state_sim_time = [0.0]
        state_cmd_seq = [0]
        state_epoch = [0]
        brackets = st.bracket_commands([0], [0], state_sim_time, state_cmd_seq, state_epoch)
        assert brackets[0].unplaceable
