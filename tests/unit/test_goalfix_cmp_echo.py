"""Unit tests for tools/goalfix_cmp/echo.py."""
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
from tools.goalfix_cmp._minjerk import pose_at  # noqa: E402


def _load(tmp_path, state_rows, command_rows):
    paths = mf.write_evidence(tmp_path, state_rows, command_rows)
    return ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")


class TestCleanFlight:
    def test_zero_genuine_echoes(self, tmp_path):
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS})
        route = [
            mf.Waypoint("A", {"r_shoulder_pitch": -0.5, "r_elbow_pitch": -0.3}, 1.5),
            mf.Waypoint("B", {"r_shoulder_pitch": -0.8, "r_gripper": 0.35}, 1.2),
        ]
        sim.fly(route)
        result = sim.result()
        evd = _load(tmp_path, result.state_rows, result.command_rows)
        results = echo.classify_commands(evd)
        counts = echo.count_labels(results)
        assert counts.genuine_echo == 0
        assert counts.fresh > 0


class TestForcedEchoes:
    def test_injected_echoes_are_all_detected_as_genuine(self, tmp_path):
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS}, seed=7)
        route = [
            mf.Waypoint("A", {"r_shoulder_pitch": -0.5, "r_elbow_pitch": -0.3,
                               "r_arm_yaw": 0.1, "r_wrist_pitch": -0.2}, 2.0),
        ]
        sim.fly(route, echo_rate=0.44)
        result = sim.result()
        evd = _load(tmp_path, result.state_rows, result.command_rows)
        results = echo.classify_commands(evd)

        forced_by_cmd_joint = {(fe.command_index, fe.joint_index) for fe in result.forced_echoes}
        assert len(forced_by_cmd_joint) > 0

        detected_genuine = set()
        for i, r in enumerate(results):
            if r is None:
                continue
            for jidx, jname in enumerate(mf.R_JOINTS):
                if r.joints[jname].label == echo.GENUINE_ECHO:
                    detected_genuine.add((i, jidx))

        assert forced_by_cmd_joint <= detected_genuine
        # No spurious genuine echoes beyond what was injected.
        assert detected_genuine == forced_by_cmd_joint


class TestBoundaries:
    def _bracket_case(self, tmp_path, *, source_age_s, joint_index=0):
        """One epoch: an epoch-start state, a source state at `source_age_s`
        before the command's t_hi, and the command itself, whose target for
        `joint_index` bit-exactly copies the source state's position."""
        joints = list(mf.R_JOINTS)
        base_pos = [0.1 + 0.01 * k for k in range(8)]
        rows = []
        # epoch-start state (t=0, distinct baseline)
        rows.append(mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                                  wall_time_ns=1, position_rad21=mf.full21(
                                      dict(zip(joints, base_pos)))))
        t_hi = 1.0
        source_time = t_hi - source_age_s
        source_pos = list(base_pos)
        source_pos[joint_index] = 0.777
        rows.append(mf.state_row(seq=1, sim_step=1, sim_time_s=source_time, cmd_seq=-1,
                                  wall_time_ns=2, position_rad21=mf.full21(
                                      dict(zip(joints, source_pos)))))
        # A filler state strictly between the source and t_hi so t_lo != source.
        filler_time = (source_time + t_hi) / 2
        rows.append(mf.state_row(seq=2, sim_step=2, sim_time_s=filler_time, cmd_seq=0,
                                  wall_time_ns=3, position_rad21=mf.full21(
                                      dict(zip(joints, base_pos)))))
        target = list(base_pos)
        target[joint_index] = float(np.float32(0.777))
        cmd = mf.command_row_joint(seq=1, target_rad21=mf.full21(dict(zip(joints, target))))
        commands = [cmd]
        # The t_hi state: reports cmd_seq=1, at sim_time_s=t_hi.
        rows.append(mf.state_row(seq=3, sim_step=3, sim_time_s=t_hi, cmd_seq=1,
                                  wall_time_ns=4, position_rad21=mf.full21(
                                      dict(zip(joints, base_pos)))))
        return rows, commands

    def test_echo_at_0_49s_counts(self, tmp_path):
        rows, commands = self._bracket_case(tmp_path, source_age_s=0.49)
        evd = _load(tmp_path, rows, commands)
        results = echo.classify_commands(evd)
        r = results[0]
        assert r is not None
        assert r.joints[mf.R_JOINTS[0]].label == echo.GENUINE_ECHO
        assert abs(r.joints[mf.R_JOINTS[0]].age_s - 0.49) < 1e-9

    def test_echo_at_0_51s_does_not_count(self, tmp_path):
        rows, commands = self._bracket_case(tmp_path, source_age_s=0.51)
        evd = _load(tmp_path, rows, commands)
        results = echo.classify_commands(evd)
        r = results[0]
        assert r.joints[mf.R_JOINTS[0]].label == echo.FRESH


class TestCoincidences:
    def test_start_coincidence(self, tmp_path):
        joints = list(mf.R_JOINTS)
        pose = dict(zip(joints, [0.2] * 8))
        rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                              wall_time_ns=1, position_rad21=mf.full21(pose))]
        target = dict(pose)
        target["r_shoulder_pitch"] = float(np.float32(pose["r_shoulder_pitch"]))
        cmd = mf.command_row_joint(seq=0, target_rad21=mf.full21(target))
        rows.append(mf.state_row(seq=1, sim_step=1, sim_time_s=0.02, cmd_seq=0,
                                  wall_time_ns=2, position_rad21=mf.full21(pose)))
        evd = _load(tmp_path, rows, [cmd])
        results = echo.classify_commands(evd)
        assert results[0].joints["r_shoulder_pitch"].label == echo.START_COINCIDENCE

    def test_path_coincidence(self, tmp_path):
        joints = list(mf.R_JOINTS)
        start8 = {"r_shoulder_pitch": -0.4, "r_elbow_pitch": -0.2}
        goal8 = {"r_shoulder_pitch": -0.7, "r_elbow_pitch": -0.5}
        seconds = 1.0
        base_pose = dict(zip(joints, [0.0] * 8))
        base_pose.update(start8)
        rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                              wall_time_ns=1, position_rad21=mf.full21(base_pose))]
        # An unrelated earlier state whose shoulder_pitch happens to equal
        # the min-jerk value at tau=0.5 -- this is what a genuine echo
        # detector would flag, unless the path-coincidence carve-out fires.
        tau = 0.5
        sp_value = pose_at(start8["r_shoulder_pitch"], goal8["r_shoulder_pitch"], tau)
        ep_value = pose_at(start8["r_elbow_pitch"], goal8["r_elbow_pitch"], tau)
        decoy_pose = dict(base_pose)
        decoy_pose["r_shoulder_pitch"] = float(np.float32(sp_value)) + 0.05  # not a match itself
        rows.append(mf.state_row(seq=1, sim_step=1, sim_time_s=0.3, cmd_seq=-1,
                                  wall_time_ns=2, position_rad21=mf.full21(decoy_pose)))
        # The "source" state that bit-exactly holds the min-jerk value (as
        # if this were a genuine echo) -- present so the exact-match search
        # finds something to subclassify.
        source_pose = dict(base_pose)
        source_pose["r_shoulder_pitch"] = float(np.float32(sp_value))
        rows.append(mf.state_row(seq=2, sim_step=2, sim_time_s=0.35, cmd_seq=-1,
                                  wall_time_ns=3, position_rad21=mf.full21(source_pose)))

        target = dict(base_pose)
        target["r_shoulder_pitch"] = float(np.float32(sp_value))
        target["r_elbow_pitch"] = ep_value  # the other moving joint, at the same tau
        cmd = mf.command_row_joint(seq=0, target_rad21=mf.full21(target))
        rows.append(mf.state_row(seq=3, sim_step=3, sim_time_s=0.5, cmd_seq=0,
                                  wall_time_ns=4, position_rad21=mf.full21(target)))

        evd = _load(tmp_path, rows, [cmd])
        ctx = echo.GotoContext(start8=start8, goal8=goal8, seconds=seconds)
        results = echo.classify_commands(evd, goto_context=[ctx])
        assert results[0].joints["r_shoulder_pitch"].label == echo.PATH_COINCIDENCE

    def test_timing_ambiguous_match_only_at_hi_state(self, tmp_path):
        joints = list(mf.R_JOINTS)
        base = dict(zip(joints, [0.1] * 8))
        rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                              wall_time_ns=1, position_rad21=mf.full21(base))]
        exact = float(np.float32(0.555))
        target = dict(base)
        target["r_shoulder_pitch"] = exact
        cmd = mf.command_row_joint(seq=0, target_rad21=mf.full21(target))
        # The only state reporting this command (t_hi) happens to already
        # sit at the target value -- produced by/after application, so it
        # must not count as an echo SOURCE.
        hi_pose = dict(base)
        hi_pose["r_shoulder_pitch"] = exact
        rows.append(mf.state_row(seq=1, sim_step=1, sim_time_s=0.02, cmd_seq=0,
                                  wall_time_ns=2, position_rad21=mf.full21(hi_pose)))
        evd = _load(tmp_path, rows, [cmd])
        results = echo.classify_commands(evd)
        jr = results[0].joints["r_shoulder_pitch"]
        assert jr.label == echo.TIMING_AMBIGUOUS


class TestControl:
    def test_control_window_before_epoch_start_is_unavailable(self, tmp_path):
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS})
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.3}, 0.5)])
        result = sim.result()
        evd = _load(tmp_path, result.state_rows, result.command_rows)
        control = echo.classify_commands(evd, shift_s=-echo.CONTROL_SHIFT_S)
        counts = echo.count_labels(control)
        assert counts.unavailable > 0
        assert counts.genuine_echo == 0

    def test_control_rate_is_low_on_clean_flight(self, tmp_path):
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS})
        route = [mf.Waypoint("A", {"r_shoulder_pitch": -0.5, "r_elbow_pitch": -0.3}, 3.0)]
        sim.fly(route)
        result = sim.result()
        evd = _load(tmp_path, result.state_rows, result.command_rows)
        # Not enough sim time (<20s) for the shift to land inside the
        # window at all -- the whole control run is unavailable, which is
        # itself the correct, honest answer (never a fabricated "0%").
        control = echo.classify_commands(evd, shift_s=-echo.CONTROL_SHIFT_S)
        counts = echo.count_labels(control)
        assert counts.genuine_echo == 0


class TestCli:
    def test_cli_b_stop_on_genuine_echo(self, tmp_path):
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS}, seed=3)
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.5, "r_elbow_pitch": -0.3,
                                    "r_arm_yaw": 0.2, "r_wrist_pitch": -0.1}, 2.0)],
                echo_rate=0.3)
        result = sim.result()
        mf.write_evidence(tmp_path, result.state_rows, result.command_rows)
        out = tmp_path / "out.json"
        rc = echo._cli(["--evidence-dir", str(tmp_path), "--arm", "B", "--out", str(out)])
        assert rc == 2

    def test_cli_a_inconclusive_on_no_echoes(self, tmp_path):
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS})
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.5}, 1.0)])
        result = sim.result()
        mf.write_evidence(tmp_path, result.state_rows, result.command_rows)
        out = tmp_path / "out.json"
        rc = echo._cli(["--evidence-dir", str(tmp_path), "--arm", "A", "--out", str(out)])
        assert rc == 3
