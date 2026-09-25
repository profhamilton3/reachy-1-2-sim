"""CB7 acceptance test (merge verdict / readiness review, 2026-09-25
stage-repairs assignment §4 CB7): "reporting gaps." Plan §7.1's control
run and §7.2's "also reported per waypoint: re-stream pass counts,
arrival errors, and the r_wrist_pitch shortfall (descriptive)" were
computed nowhere in ``between_*.json`` (the cycle CLI's own output) --
the control rate existed only inside the separate ``echo`` CLI, and the
per-waypoint figures did not exist at all. Both are added to
``CycleVerdict``/``CycleVerdict.as_dict()``, populated in
``evaluate_cycle`` (the shipped call site the CLI itself calls),
REPORT-ONLY: nothing here is read by any rc/verdict computation."""
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


@pytest.fixture(autouse=True)
def _zero_lag(monkeypatch):
    monkeypatch.setattr(mf, "_LAG_RAD", [0.0] * 8)


def _build(tmp_path):
    joints = list(mf.R_JOINTS)
    start_pose = dict(zip(joints, [0.0] * 8))
    hover_target = dict(start_pose, r_wrist_pitch=np.radians(-40.0))
    rest_target = dict(hover_target, r_wrist_pitch=np.radians(-60.0))

    # HOVER's own commanded arrival falls 5deg short of the nominal
    # goal -- the "r_wrist_pitch shortfall" this item names, a property
    # of the COMMANDED target (§7.2 is "commanded-path checks"), not the
    # realised position (CB2's own distinction).
    hover_cmd_target = dict(hover_target, r_wrist_pitch=np.radians(-35.0))

    rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                          wall_time_ns=1, position_rad21=mf.full21(start_pose))]
    cmd0 = mf.command_row_joint(seq=0, target_rad21=mf.full21(hover_cmd_target))
    rows.append(mf.state_row(seq=1, sim_step=1, sim_time_s=0.02, cmd_seq=0,
                              wall_time_ns=2, position_rad21=mf.full21(hover_cmd_target)))
    # REST_SHUT's own single command reaches its goal exactly.
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


class TestPerWaypointReport:
    def test_arrival_error_and_restream_count_through_evaluate_cycle(self, tmp_path):
        evd, leg = _build(tmp_path)
        cv, _ = cyc.evaluate_cycle("cb7", "A", evd, [leg], skip_gates=True)

        assert "setup" in cv.per_waypoint
        by_name = {wp["name"]: wp for wp in cv.per_waypoint["setup"]}
        assert set(by_name) == {"HOVER", "REST_SHUT"}

        # HOVER's own commanded arrival never hit its exact goal (short
        # by 5deg) -- no re-stream pass, and the "r_wrist_pitch
        # shortfall" the review names is exactly this arrival error.
        assert by_name["HOVER"]["restream_pass_count"] == 0
        assert by_name["HOVER"]["arrival_error_deg"]["r_wrist_pitch"] == pytest.approx(5.0)

        # REST_SHUT's own single command IS exact -- one re-stream pass
        # (the arrival itself, C5's own counting convention), zero error.
        assert by_name["REST_SHUT"]["restream_pass_count"] == 1
        assert by_name["REST_SHUT"]["arrival_error_deg"]["r_wrist_pitch"] == pytest.approx(0.0)

    def test_report_only_does_not_affect_verdict(self, tmp_path):
        """The 5deg commanded shortfall independently fails C4 (an
        EXISTING, unrelated gate) -- this test isolates that the new
        per_waypoint/control fields themselves are never consulted by
        rc()/verdict, by checking the verdict's own reasons name only
        pre-existing checks (C1/C4), never anything about per_waypoint
        or a control count."""
        evd, leg = _build(tmp_path)
        cv, _ = cyc.evaluate_cycle("cb7", "A", evd, [leg], skip_gates=True)
        assert all("per_waypoint" not in r and "control" not in r for r in cv.reasons), cv.reasons


class TestControlRatePerSegment:
    def test_control_genuine_echo_count_present_when_segment_determinate(self, tmp_path):
        evd, leg = _build(tmp_path)
        cv, _ = cyc.evaluate_cycle("cb7", "A", evd, [leg], skip_gates=True)
        assert not cv.segment_indeterminate
        assert cv.control_genuine_echo_count is not None
        assert cv.control_new_target_count is not None
        assert cv.control_genuine_echo_count == 0  # no planted control-side match here

    def test_control_fields_none_when_segment_indeterminate(self, tmp_path):
        """No 'setup'/place-route leg at all -- segment_indeterminate is
        the default True (TestMutantSegmentIndeterminateIgnoredForB's
        own construction) -- the control fields must stay None, never a
        fabricated 0, matching genuine_echo_count's own convention."""
        joints = list(mf.R_JOINTS)
        start_pose = dict(zip(joints, [0.0] * 8))
        present = dict(start_pose, r_shoulder_pitch=np.radians(-10.0))
        rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                              wall_time_ns=1, position_rad21=mf.full21(start_pose))]
        cmd0 = mf.command_row_joint(seq=0, target_rad21=mf.full21(present))
        rows.append(mf.state_row(seq=1, sim_step=1, sim_time_s=0.02, cmd_seq=0,
                                  wall_time_ns=2, position_rad21=mf.full21(present)))
        mf.write_evidence(tmp_path, rows, [cmd0])
        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        flight_leg = cyc.LegSpec(
            "flight", route_rad=[mf.Waypoint("PRESENT", present, 1.0)],
            guard=(), start_pose8=start_pose, command_indices=[0])
        cv, _ = cyc.evaluate_cycle("cb7b", "B", evd, [flight_leg], skip_gates=True)
        assert cv.segment_indeterminate
        assert cv.control_genuine_echo_count is None
        assert cv.control_new_target_count is None


class TestMutationPayloadKeysRemoved:
    def test_pre_cb7_as_dict_had_no_reporting_keys(self):
        """Mutation guard (verified directly against a pre-CB7 copy of
        cycle.py -- see the handoff for the transcript): CycleVerdict.as_dict()
        had no 'control_genuine_echo_count', 'control_new_target_count'
        or 'per_waypoint' keys at all, and LegResult had no per_waypoint
        field, before this item. Pinned here (not a library-internal
        check) as a standing regression guard on the payload's own
        shape, matching the assignment's "present in the payload"
        requirement."""
        cv = cyc.CycleVerdict("x", "A", cyc.VERDICT_OK)
        d = cv.as_dict()
        assert "control_genuine_echo_count" in d
        assert "control_new_target_count" in d
        assert "per_waypoint" in d
