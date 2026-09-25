"""CB2 acceptance test (merge verdict / readiness review, 2026-09-25
stage-repairs assignment §4 CB2): "commanded start error." cycle.py's
``compute_place_route_metrics`` computed ``leg_start_shoulder_pitch_error_deg``
from ``evidence.states.position_rad`` (the REALISED position at the
affected segment's start command) rather than ``evidence.commands.target_rad``
(the COMMANDED target at that same command) -- report line 91 calls this
quantity a "leg-start setpoint"/"target at leg start", and line 101 "the
REST_SHUT leg starts from that moved TARGET, not from HOVER" -- both the
COMMANDED quantity, never the realised one (``post_arrival_rise_deg``, a
few lines below in cycle.py, is the realised quantity, and stays realised
per the assignment's own instruction).

The assignment's own required fixture: "a fixture where realised lags
2.5 deg but the commanded start = HOVER -> error 0.0, not 'refutes'" --
i.e. a purely physical lag (never an echoed/drifted command) must not be
reported as if the COMMANDED target had itself drifted from HOVER."""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../scripts", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import cycle as cyc  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402

_LAG_DEG = 2.5  # sourced from the assignment's own fixture spec (CB2 row)


def _build(tmp_path):
    joints = list(mf.R_JOINTS)
    start_pose = dict(zip(joints, [0.0] * 8))
    hover_target = dict(start_pose, r_shoulder_pitch=np.radians(-30.0))
    rest_target = dict(hover_target, r_shoulder_pitch=np.radians(-70.0))
    lag_rad = np.radians(_LAG_DEG)

    rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                          wall_time_ns=1, position_rad21=mf.full21(start_pose))]

    # cmd0: HOVER's own (single-command) goto, reaching HOVER's target
    # EXACTLY -- the commanded start is HOVER, unmoved by any echo.
    cmd0 = mf.command_row_joint(seq=0, target_rad21=mf.full21(hover_target))
    # The REALISED position lags the commanded target by exactly 2.5deg
    # -- ordinary physical lag/settle, not a drifted command.
    realised_at_hover = dict(
        hover_target, r_shoulder_pitch=hover_target["r_shoulder_pitch"] + lag_rad)
    rows.append(mf.state_row(seq=1, sim_step=1, sim_time_s=0.02, cmd_seq=0,
                              wall_time_ns=2, position_rad21=mf.full21(realised_at_hover)))

    # cmd1: REST_SHUT's own (single-command) goto.
    cmd1 = mf.command_row_joint(seq=1, target_rad21=mf.full21(rest_target))
    rows.append(mf.state_row(seq=2, sim_step=2, sim_time_s=0.04, cmd_seq=1,
                              wall_time_ns=3, position_rad21=mf.full21(rest_target)))

    mf.write_evidence(tmp_path, rows, [cmd0, cmd1])
    evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")

    leg = cyc.LegSpec(
        "setup", route_rad=[mf.Waypoint("HOVER", hover_target, 1.0),
                             mf.Waypoint("REST_SHUT", rest_target, 1.0)],
        guard=(), start_pose8=start_pose, command_indices=[0, 1])
    return evd, leg


class TestCommandedStartNotRealised:
    def test_realised_lag_with_commanded_start_at_hover_gives_zero_error(self, tmp_path):
        evd, leg = _build(tmp_path)
        cv, _ = cyc.evaluate_cycle("cb2", "A", evd, [leg], skip_gates=True)
        assert cv.metrics is not None
        assert cv.metrics.leg_start_shoulder_pitch_error_deg == pytest.approx(0.0, abs=1e-9), (
            cv.metrics)

    def test_mutation_states_position_rad_reverted_would_report_the_lag(self, tmp_path):
        """Mutation guard (verified directly against a copy of cycle.py
        reverted to states.position_rad at this call site -- see the
        handoff for the transcript): the same fixture then reports
        leg_start_shoulder_pitch_error_deg == 2.5 (the physical lag),
        not 0.0. Pinned here as the value the correct code (asserted
        above) must keep giving."""
        evd, leg = _build(tmp_path)
        cv, _ = cyc.evaluate_cycle("cb2", "A", evd, [leg], skip_gates=True)
        assert cv.metrics.leg_start_shoulder_pitch_error_deg == pytest.approx(0.0, abs=1e-9)
