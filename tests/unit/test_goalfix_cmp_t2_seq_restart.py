"""T2 acceptance tests (assignment §2, T2): command bracketing across a
container recreate. Before the fix (review §3.1/M2, E2/E11), a per-cycle
recreate left native's carried cmd_seq outranking the new bridge's own
seq numbering, so np.maximum.accumulate never came back down and every
new (low-seq) command in the epoch became unplaceable -- 373 of 374 in
the reviewed reproduction. echo.py's rc-0-on-a-mostly-unplaceable-B-file
and inconclusive_baseline-on-an-echo-laden-A-file bugs (E2/E11) are the
same root cause; the gate this commit adds (`check_no_unplaceable_in_range`,
wired into every CLI) is what turns "wrong answer" into "evidence
incomplete" until T4 gives the fix somewhere real metrics can be computed."""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp import _simtime as st  # noqa: E402
from tools.goalfix_cmp import echo  # noqa: E402
from tools.goalfix_cmp._io import RC_INCONCLUSIVE, read_result  # noqa: E402

ROUTE = [
    mf.Waypoint("A", {"r_shoulder_pitch": -0.3, "r_elbow_pitch": -0.2}, 0.4),
    mf.Waypoint("B", {"r_shoulder_pitch": -0.6, "r_gripper": 0.3}, 0.4),
]


def _multi_cycle_session(n_cycles: int, **fly_kwargs):
    sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS})
    for c in range(n_cycles):
        if c > 0:
            sim.recreate()
            sim.reset(seed=c)
        sim.fly(ROUTE, **(fly_kwargs if c > 0 else {}))
    return sim.result()


class TestRecreateResetLegsCarriedCmdSeq:
    def test_every_leg_command_is_placeable_across_recreates(self, tmp_path):
        result = _multi_cycle_session(4)
        mf.write_evidence(tmp_path, result.state_rows, result.command_rows)
        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        assert evd.n_epochs() == 4
        assert evd.unplaceable_command_indices == []
        # At least one epoch must actually have exercised the restart path,
        # or this test would pass vacuously (the pre-fix bug too, on a
        # session with no recreate).
        assert any(info.restart for info in evd.bridge_sessions.values())

    def test_b_file_with_injected_echoes_gives_echo_stop(self, tmp_path):
        result = _multi_cycle_session(2, echo_rate=0.5)
        mf.write_evidence(tmp_path, result.state_rows, result.command_rows)
        out = tmp_path / "out.json"
        # Classify epoch 1 (the post-recreate one) in full.
        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        jc_idx = [i for i in np.nonzero(evd.commands.joint_command_mask())[0]
                  if evd.commands.epoch[i] == 1]
        rc = echo._cli([
            "--evidence-dir", str(tmp_path), "--arm", "B",
            "--segment-start-index", str(min(jc_idx)),
            "--segment-end-index", str(max(jc_idx)),
            "--out", str(out),
        ])
        payload = read_result(out)
        assert payload["counts"]["genuine_echo"] > 0, payload
        from tools.goalfix_cmp._io import RC_STOP
        assert rc == RC_STOP, payload

    def test_a_file_with_injected_echoes_is_manipulated_not_inconclusive(self, tmp_path):
        result = _multi_cycle_session(2, echo_rate=0.5)
        mf.write_evidence(tmp_path, result.state_rows, result.command_rows)
        out = tmp_path / "out.json"
        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        jc_idx = [i for i in np.nonzero(evd.commands.joint_command_mask())[0]
                  if evd.commands.epoch[i] == 1]
        rc = echo._cli([
            "--evidence-dir", str(tmp_path), "--arm", "A",
            "--segment-start-index", str(min(jc_idx)),
            "--segment-end-index", str(max(jc_idx)),
            "--out", str(out),
        ])
        payload = read_result(out)
        assert payload["counts"]["genuine_echo"] > 0, payload
        from tools.goalfix_cmp._io import RC_OK
        assert rc == RC_OK, payload  # "manipulated" -- not inconclusive_baseline

    def test_one_removed_state_report_gives_rc3(self, tmp_path):
        # No settle hold and no re-stream passes, so the very last state
        # row is exactly the tick that reports the final command applied
        # -- dropping it leaves that command genuinely unreported (`fg >=
        # len(st)`), not merely bracketed more widely (removing an
        # interior state, or a hold-tail state that just repeats the same
        # cmd_seq, only widens a bracket or has no effect at all).
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS}, settle_s=0.0)
        sim.fly(ROUTE)
        sim.recreate()
        sim.reset(seed=1)
        sim.fly(ROUTE, echo_rate=0.0)
        result = sim.result()
        rows = list(result.state_rows)[:-1]
        mf.write_evidence(tmp_path, rows, result.command_rows)
        out = tmp_path / "out.json"
        rc = echo._cli([
            "--evidence-dir", str(tmp_path), "--arm", "B", "--out", str(out),
        ])
        assert rc == RC_INCONCLUSIVE


class TestMidEpochSeqDrop:
    def test_mid_epoch_bridge_restart_without_reset_is_ambiguous(self, tmp_path):
        """A bridge reconnect with no native reset: the command seq stream
        drops mid-epoch, with no accompanying `reset` row. Per the
        assignment ("do not guess"), this is reported ambiguous, never
        resolved by assumption."""
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS})
        sim.fly(ROUTE)
        # Simulate a bridge-only restart (no native reset): the bridge's own
        # seq numbering restarts, but there is no `reset` command row and no
        # sim_step drop.
        sim._cmd_seq = 0
        sim.fly(ROUTE)
        result = sim.result()
        mf.write_evidence(tmp_path, result.state_rows, result.command_rows)
        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        assert evd.n_epochs() == 1
        assert evd.bridge_sessions[0].ambiguous
        assert len(evd.unplaceable_command_indices) > 0
        for i in evd.unplaceable_command_indices:
            assert evd.brackets[i].seq_ambiguous
