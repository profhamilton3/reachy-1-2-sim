"""T7 acceptance tests (assignment §2, T7): start coincidence at each leg's
own turn_on (plan §7.1/review §3.3, E8). The old rule required the match's
source to be the EPOCH's absolute first state, so it could never fire on a
real leg (which starts after a settle, not at the epoch's very first
sample) and never on a cycle's second leg at all."""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import echo  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402


def _load(tmp_path, rows, commands):
    mf.write_evidence(tmp_path, rows, commands)
    return ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")


class TestE8AfterSettle:
    def test_turn_on_after_15s_settle_is_start_coincidence(self, tmp_path):
        """E8: a turn_on after a 15 s settle, whose first setpoint equals
        the present, must be start_coincidence, not genuine_echo."""
        joints = list(mf.R_JOINTS)
        pose = dict(zip(joints, [0.2] * 8))
        rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                              wall_time_ns=1, position_rad21=mf.full21(pose))]
        # 15 s of settle: continuous present-position readback, no commands.
        for k in range(1, 751):  # 750 * 20ms = 15s
            rows.append(mf.state_row(seq=k, sim_step=k, sim_time_s=k * 0.02, cmd_seq=-1,
                                      wall_time_ns=1 + k, position_rad21=mf.full21(pose)))
        turn_on_state_index = len(rows) - 1  # the reading in force at turn_on

        target = dict(pose)
        target["r_shoulder_pitch"] = float(np.float32(pose["r_shoulder_pitch"]))
        cmd = mf.command_row_joint(seq=0, target_rad21=mf.full21(target))
        rows.append(mf.state_row(seq=len(rows), sim_step=len(rows), sim_time_s=len(rows) * 0.02,
                                  cmd_seq=0, wall_time_ns=1 + len(rows),
                                  position_rad21=mf.full21(pose)))

        evd = _load(tmp_path, rows, [cmd])
        results = echo.classify_commands(
            evd, leg_turn_on_state_index={0: turn_on_state_index})
        assert results[0].joints["r_shoulder_pitch"].label == echo.START_COINCIDENCE

        # Without the leg_turn_on_state_index wiring, the OLD rule would
        # have missed this too (source isn't the epoch's absolute first
        # state) -- confirms the fixture actually exercises the fix, not a
        # vacuous pass.
        results_old_rule = echo.classify_commands(evd)
        assert results_old_rule[0].joints["r_shoulder_pitch"].label != echo.START_COINCIDENCE


class TestSecondLegTurnOn:
    def test_flight_leg_turn_on_is_also_start_coincidence(self, tmp_path):
        """The same pattern on the flight leg (a cycle's SECOND turn_on,
        same epoch): must also give start_coincidence."""
        joints = list(mf.R_JOINTS)
        setup_pose = dict(zip(joints, [0.0] * 8))
        setup_pose["r_shoulder_pitch"] = -0.3

        rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                              wall_time_ns=1, position_rad21=mf.full21(setup_pose))]
        setup_target = dict(setup_pose)
        setup_target["r_elbow_pitch"] = -0.2
        cmd0 = mf.command_row_joint(seq=0, target_rad21=mf.full21(setup_target))
        rows.append(mf.state_row(seq=1, sim_step=1, sim_time_s=0.02, cmd_seq=0,
                                  wall_time_ns=2, position_rad21=mf.full21(setup_target)))

        # Settle before the flight leg's own turn_on -- present position
        # held constant, no commands (a fresh SDK client + turn_on, per
        # plan §5 P8, before ANY post-reset motion for this second leg).
        present_at_flight_turn_on = dict(setup_target)
        for k in range(2, 52):  # 1 s settle
            rows.append(mf.state_row(seq=k, sim_step=k, sim_time_s=k * 0.02, cmd_seq=0,
                                      wall_time_ns=1 + k,
                                      position_rad21=mf.full21(present_at_flight_turn_on)))
        flight_turn_on_state_index = len(rows) - 1

        flight_target = dict(present_at_flight_turn_on)
        flight_target["r_shoulder_pitch"] = float(
            np.float32(present_at_flight_turn_on["r_shoulder_pitch"]))
        cmd1 = mf.command_row_joint(seq=1, target_rad21=mf.full21(flight_target))
        rows.append(mf.state_row(seq=len(rows), sim_step=len(rows), sim_time_s=len(rows) * 0.02,
                                  cmd_seq=1, wall_time_ns=1 + len(rows),
                                  position_rad21=mf.full21(flight_target)))

        evd = _load(tmp_path, rows, [cmd0, cmd1])
        results = echo.classify_commands(
            evd, leg_turn_on_state_index={
                0: 0,  # setup leg's own turn_on: the epoch-start state
                1: flight_turn_on_state_index,
            })
        # Setup leg's first command: r_elbow_pitch moves, r_shoulder_pitch
        # is a carry (same as start) so it is never subclassified there.
        assert results[1].joints["r_shoulder_pitch"].label == echo.START_COINCIDENCE


class TestNotEveryExactMatchIsStartCoincidence:
    def test_exact_match_mid_goto_is_still_genuine_echo(self, tmp_path):
        """An exact match to the present position in the MIDDLE of a goto
        (not a leg's own first command) still gives genuine_echo."""
        joints = list(mf.R_JOINTS)
        pose = dict(zip(joints, [0.1] * 8))
        rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                              wall_time_ns=1, position_rad21=mf.full21(pose))]
        cmd0 = mf.command_row_joint(seq=0, target_rad21=mf.full21(
            dict(pose, r_elbow_pitch=-0.4)))
        rows.append(mf.state_row(seq=1, sim_step=1, sim_time_s=0.02, cmd_seq=0,
                                  wall_time_ns=2, position_rad21=mf.full21(pose)))
        # A later "echoed" command bit-exactly repeats the present position
        # read back at sim_time 0.0 -- not this leg's first command.
        target1 = dict(pose, r_elbow_pitch=-0.4)
        target1["r_shoulder_pitch"] = float(np.float32(pose["r_shoulder_pitch"]))
        cmd1 = mf.command_row_joint(seq=1, target_rad21=mf.full21(target1))
        rows.append(mf.state_row(seq=2, sim_step=2, sim_time_s=0.04, cmd_seq=1,
                                  wall_time_ns=3, position_rad21=mf.full21(target1)))

        evd = _load(tmp_path, rows, [cmd0, cmd1])
        results = echo.classify_commands(
            evd, leg_turn_on_state_index={0: 0})  # only command 0 is a leg's own first
        assert results[1].joints["r_shoulder_pitch"].label == echo.GENUINE_ECHO


class TestMutationOldEpochFirstStateRuleRestored:
    def test_old_rule_would_miss_the_flight_leg(self, tmp_path):
        """Mutation guard: the old epoch-first-state rule (checked against
        `prev is None` and the EPOCH's own first state) can never fire on a
        second leg -- restoring it must be caught by
        test_flight_leg_turn_on_is_also_start_coincidence above."""
        joints = list(mf.R_JOINTS)
        pose = dict(zip(joints, [0.0] * 8))
        rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                              wall_time_ns=1, position_rad21=mf.full21(pose))]
        cmd0 = mf.command_row_joint(seq=0, target_rad21=mf.full21(
            dict(pose, r_elbow_pitch=-0.2)))
        rows.append(mf.state_row(seq=1, sim_step=1, sim_time_s=0.02, cmd_seq=0,
                                  wall_time_ns=2, position_rad21=mf.full21(
                                      dict(pose, r_elbow_pitch=-0.2))))
        second_leg_target = dict(pose, r_elbow_pitch=-0.2)
        second_leg_target["r_shoulder_pitch"] = float(np.float32(pose["r_shoulder_pitch"]))
        cmd1 = mf.command_row_joint(seq=1, target_rad21=mf.full21(second_leg_target))
        rows.append(mf.state_row(seq=2, sim_step=2, sim_time_s=0.04, cmd_seq=1,
                                  wall_time_ns=3, position_rad21=mf.full21(second_leg_target)))
        evd = _load(tmp_path, rows, [cmd0, cmd1])
        # prev is not None for command 1 (it's the epoch's 2nd command), so
        # the OLD default-fallback rule cannot classify it as start
        # coincidence even by accident -- proving the fallback alone is
        # insufficient and the explicit leg_turn_on_state_index mapping is
        # what T7 actually needs.
        results = echo.classify_commands(evd)
        assert results[1].joints["r_shoulder_pitch"].label != echo.START_COINCIDENCE
