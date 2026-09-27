"""T1 acceptance tests (assignment §2, T1): the shipped `pathcheck` CLI on a
clean, real-route radian flight -- must pass, not false-STOP on units
(review §3.2/M1, E1). Exercises the CLI function directly (in-process),
which is exactly what `python -m tools.goalfix_cmp.pathcheck` runs."""
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import pathcheck as pc  # noqa: E402
from tools.goalfix_cmp._io import RC_OK, RC_STOP, read_result  # noqa: E402
from tools.goalfix_cmp._units import route_rad  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402


def _fly_real_route(route, start_pose_deg, **flightsim_kwargs):
    sim = mf.FlightSim(dict(start_pose_deg), pose_units="deg", **flightsim_kwargs)
    sim.fly(route)
    return sim.result()


class TestPathcheckCliOnRealRoutes:
    def test_clean_place_route_flight_passes_as_b(self, tmp_path):
        result = _fly_real_route(R.PLACE_ROUTE, R.HOME, restream_passes=1)
        mf.write_evidence(tmp_path, result.state_rows, result.command_rows)
        out = tmp_path / "out.json"
        rc = pc._cli([
            "--evidence-dir", str(tmp_path), "--route", "PLACE_ROUTE", "--arm", "B",
            "--command-start-index", "0",
            "--command-end-index", str(len(result.command_rows) - 1),
            "--out", str(out),
        ])
        payload = read_result(out)
        assert rc == RC_OK, payload
        for cid in pc.ALL_CHECKS:
            assert payload[cid]["passed"], (cid, payload[cid])

    def test_clean_lift_to_present_flight_passes_as_b(self, tmp_path):
        result = _fly_real_route(R.LIFT_TO_PRESENT, R.REST, restream_passes=1)
        mf.write_evidence(tmp_path, result.state_rows, result.command_rows)
        out = tmp_path / "out.json"
        rc = pc._cli([
            "--evidence-dir", str(tmp_path), "--route", "LIFT_TO_PRESENT", "--arm", "B",
            "--command-start-index", "0",
            "--command-end-index", str(len(result.command_rows) - 1),
            "--out", str(out),
        ])
        payload = read_result(out)
        assert rc == RC_OK, payload

    def test_c7_gripper_range_holds_on_real_open_shut(self, tmp_path):
        # GRIP_SHUT (open -> shut) then a bare gripper-open move (shut ->
        # open, REST's own value) -- the real OPEN/SHUT extremes, in
        # radians, must fall inside pathcheck's GRIPPER_LO/HI_RAD.
        route = (R.PLACE_ROUTE[0],)  # GRIP_SHUT
        result = _fly_real_route(route, R.HOME)
        for row in result.command_rows:
            if row["type"] != "joint_command":
                continue
            g = row["target_rad"][mf.R_JOINTS.index("r_gripper")]
            assert pc.GRIPPER_LO_RAD - 1e-6 <= g <= pc.GRIPPER_HI_RAD + 1e-6

    def test_c5_exactness_bit_for_bit_vs_route_rad_not_bare_radians(self, tmp_path):
        route = (R.PLACE_ROUTE[0],)
        result = _fly_real_route(route, R.HOME, restream_passes=1)
        last = result.command_rows[-1]
        rad = route_rad(route)[0]
        for i, name in enumerate(mf.R_JOINTS):
            assert last["target_rad"][i] == rad.pose_rad8[name]
        bare = float(np.radians(route[0].pose["r_gripper"]))
        if bare != rad.pose_rad8["r_gripper"]:
            assert last["target_rad"][mf.R_JOINTS.index("r_gripper")] != bare
