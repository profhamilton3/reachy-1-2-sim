"""Unit tests for tools/goalfix_cmp/_units.py (T1) and the realistic-fixture
infrastructure it enables in make_fixtures.py: a real rig_routes flight
through FlightSim's degrees mode, and container-recreate/seq-restart +
carried-cmd_seq modelling (T2's fixtures)."""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import _units as U  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402


class TestRouteRad:
    def test_matches_the_wire_formula(self):
        rad = U.route_rad(R.PLACE_ROUTE)
        assert len(rad) == len(R.PLACE_ROUTE)
        for wp_deg, wp_rad in zip(R.PLACE_ROUTE, rad):
            assert wp_rad.name == wp_deg.name
            assert wp_rad.seconds == wp_deg.seconds
            for j in R.R_JOINTS:
                expect = float(np.float32(np.deg2rad(wp_deg.pose.get(j, 0.0))))
                assert wp_rad.pose_rad8[j] == expect
                assert wp_rad.pose[j] == expect  # .pose aliases pose_rad8

    def test_never_returns_a_plausible_degree_value_as_radians(self):
        # BACK's r_shoulder_pitch is 40.0 degrees == 0.698 rad, nowhere near
        # 40 rad -- route_rad must have actually converted, not passed the
        # degree value through.
        rad = U.route_rad(R.PLACE_ROUTE)
        back = next(wp for wp in rad if wp.name == "BACK")
        assert abs(back.pose_rad8["r_shoulder_pitch"] - 0.698) < 1e-3

    def test_guard_and_tol_pass_through(self):
        rad = U.route_rad(R.LIFT_TO_PRESENT)
        assert rad[0].guard == R._PRESENT_GUARD
        assert rad[0].tol_deg == R.LIFT_TO_PRESENT[0].tol


class TestFlightSimDegreesMode:
    def test_real_place_route_flies_in_degrees_mode(self):
        sim = mf.FlightSim(dict(R.HOME), pose_units="deg")
        sim.fly(R.PLACE_ROUTE)
        result = sim.result()
        assert result.goto_goal_names == [wp.name for wp in R.PLACE_ROUTE]
        # Every command target is a plausible radian value, never a raw
        # degree one left unconverted (e.g. BACK's 40.0 would show up
        # verbatim as an out-of-range "radian").
        for row in result.command_rows:
            if row["type"] != "joint_command":
                continue
            assert all(abs(v) < 2 * np.pi for v in row["target_rad"])

    def test_restream_pass_is_bit_exact_with_route_rad(self):
        sim = mf.FlightSim(dict(R.HOME), pose_units="deg", restream_passes=1)
        route = (R.PLACE_ROUTE[0],)  # GRIP_SHUT: gripper-only, quick to reach
        sim.fly(route)
        result = sim.result()
        rad = U.route_rad(route)[0]
        # The re-stream pass's commands hold the goal exactly -- bit-exact
        # against route_rad's own conversion of the same waypoint (T1's C5
        # acceptance test: "re-stream setpoints equal route_rad(...) bit for
        # bit, and a radians() value without float32 does not").
        last_cmd = result.command_rows[-1]
        for i, name in enumerate(mf.R_JOINTS):
            assert last_cmd["target_rad"][i] == rad.pose_rad8[name]
            unquantised = float(np.radians(route[0].pose.get(name, 0.0)))
            if unquantised != rad.pose_rad8[name]:
                assert last_cmd["target_rad"][i] != unquantised


class TestContainerRecreateAndSeqRestart:
    def test_recreate_restarts_bridge_seq_but_native_cmd_seq_carries(self, tmp_path):
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS})
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.2)])
        pre_native_seq = sim._native_cmd_seq
        assert pre_native_seq >= 0
        sim.recreate()
        assert sim._cmd_seq == 0  # bridge-local numbering restarted
        sim.reset(seed=1)
        # The reset epoch's very first state reports the CARRIED (stale)
        # native cmd_seq, not the restarted bridge counter.
        epoch1_start = sim.state_rows[-1]
        assert epoch1_start["cmd_seq"] == pre_native_seq
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.2)])
        result = sim.result()
        # Once the new bridge's own first command is applied, cmd_seq drops
        # to the new (low, restarted) numbering.
        first_new_cmd = next(r for r in result.command_rows
                              if r["type"] == "joint_command" and r["seq"] == 0)
        assert first_new_cmd is not None

    def test_trailing_reset_row_is_the_last_command(self, tmp_path):
        sim = mf.FlightSim({j: 0.0 for j in mf.R_JOINTS})
        sim.fly([mf.Waypoint("A", {"r_shoulder_pitch": -0.2}, 0.2)])
        sim.recreate()
        sim.reset(seed=1)
        result = sim.result()
        assert result.command_rows[-1]["type"] == "reset"
