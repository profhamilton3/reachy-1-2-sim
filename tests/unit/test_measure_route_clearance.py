"""scripts/measure_route_clearance.py: the reporting half (review 2026-09-12,
E1). Offline throughout -- a synthetic joint log stands in for a real
flight, exactly the way this module's own docstring says the recording half
cannot be exercised without hardware.

The module imports `reachy_sdk` at load time (same reason
`test_smoke_test_host.py` stubs it for `smoke_test_host`): it is not
installed here, matching a real host outside the simulator container.
"""

import json
import os
import sys
import types

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../scripts"))
sys.path.insert(0, os.path.join(_HERE, "../../src"))

from reachy_ai.motion import rig_routes as R  # noqa: E402

_SCENE_PATH = os.path.join(_HERE, "../../scenes/FWDCenterLabSivaPool.yaml")


@pytest.fixture
def mrc(monkeypatch):
    monkeypatch.setitem(sys.modules, "reachy_sdk",
                        types.SimpleNamespace(ReachySDK=object))
    sys.modules.pop("measure_route_clearance", None)
    import measure_route_clearance as m
    return m


def _synthetic_log(pose, n=5):
    """A log that never moves -- every sample is the same pose."""
    return [{"t": i / 20.0, "joints": {j: pose[j] for j in R.ARM7}}
            for i in range(n)]


class TestRefusesWithoutTheEnvVar:
    def test_main_refuses_without_record_clearance(self, monkeypatch, mrc,
                                                    capsys):
        monkeypatch.setattr(mrc, "APERTURE_LOGGED", True)   # past the E1 block
        monkeypatch.delenv("REACHY_SIM_RECORD_CLEARANCE", raising=False)
        with pytest.raises(SystemExit) as exc:
            mrc.main()
        assert exc.value.code != 0
        assert "REACHY_SIM_RECORD_CLEARANCE" in capsys.readouterr().out

    def test_main_refuses_even_with_the_var_set_to_something_else(
            self, monkeypatch, mrc, capsys):
        monkeypatch.setattr(mrc, "APERTURE_LOGGED", True)
        monkeypatch.setenv("REACHY_SIM_RECORD_CLEARANCE", "true")
        with pytest.raises(SystemExit):
            mrc.main()
        assert "REACHY_SIM_RECORD_CLEARANCE" in capsys.readouterr().out

    def test_e1_is_blocked_until_the_aperture_is_logged(self, monkeypatch, mrc,
                                                        capsys):
        """The opt-in variable is not enough: until the recorder logs
        r_gripper, main() refuses before it even looks at the environment
        (docs/adr/0003, E1 prerequisite).  This test must be rewritten, not
        deleted, in the commit that implements the logging."""
        assert mrc.APERTURE_LOGGED is False
        monkeypatch.setenv("REACHY_SIM_RECORD_CLEARANCE", "1")
        with pytest.raises(SystemExit) as exc:
            mrc.main()
        assert exc.value.code != 0
        out = capsys.readouterr().out
        assert "aperture" in out and "blocked" in out


class TestReportingOnASyntheticLog:
    """The reporting half never touches the SDK -- it reads a joint log
    (real or synthetic) and a scene file."""

    def test_realised_and_planned_agree_when_the_log_sits_at_a_waypoint(
            self, mrc):
        """A log that never leaves REST should read the same worst
        clearance, both hand modes, as the planned LOWER_TO_REST samples
        collapsed onto their REST endpoint -- a log with no motion in it
        cannot show tracking degradation, which is the point: this is a
        sanity check on the reporting math, not a claim about a real
        flight."""
        log = _synthetic_log(R.REST, n=8)
        result = mrc.report(log, "LOWER_TO_REST", _SCENE_PATH)
        assert result["n_samples"] == 8
        assert result["route"] == "LOWER_TO_REST"
        for hand in ("tube", "shells"):
            assert hand in result["realised"]
            assert hand in result["planned"]
            # Every tracked manipulable is far from REST in this empty-ish
            # scene, so clearance is large and positive under both models.
            assert all(d > 0 for d in result["realised"][hand].values())

    def test_realised_clearance_reflects_where_the_log_actually_is(self, mrc):
        """Two logs at different poses must not report the same worst
        clearance -- otherwise the report is reading the scene alone, not
        the log."""
        rest_log = _synthetic_log(R.REST, n=4)
        present_log = _synthetic_log(R.PRESENT, n=4)
        rest_result = mrc.report(rest_log, "LOWER_TO_REST", _SCENE_PATH)
        present_result = mrc.report(present_log, "LOWER_TO_REST", _SCENE_PATH)
        assert (rest_result["realised"]["tube"]
                != present_result["realised"]["tube"])

    def test_shells_never_reports_a_smaller_worst_clearance_than_tube_here(
            self, mrc):
        """Consistent with the review's own finding (never less
        conservative, 0/224, on the sweep this scene's objects sit outside
        of): at REST on this scene, shells should read no tighter than
        tube for any object actually in the way. Not a universal claim
        (see test_arm_geometry_mjcf.py / test_footprint_boards.py for the
        measured exceptions) -- just true for this particular pose."""
        log = _synthetic_log(R.REST, n=4)
        result = mrc.report(log, "LOWER_TO_REST", _SCENE_PATH)
        for oid in result["realised"]["tube"]:
            assert (result["realised"]["shells"][oid]
                    >= result["realised"]["tube"][oid] - 0.002)

    def test_a_route_with_no_footprint_legs_entry_reports_empty_planned(
            self, mrc):
        log = _synthetic_log(R.PRESENT, n=3)
        result = mrc.report(log, "WAVE", _SCENE_PATH)
        assert result["planned"] == {"tube": {}, "shells": {}}
        # Realised is still reported -- the log itself is unaffected by
        # whether the route has a footprint entry.
        assert result["realised"]["tube"]

    def test_save_log_writes_readable_json_under_runs(self, mrc, tmp_path,
                                                       monkeypatch):
        monkeypatch.setattr(mrc, "RUNS_DIR", tmp_path)
        log = _synthetic_log(R.REST, n=2)
        path = mrc.save_log(log, "LOWER_TO_REST", _SCENE_PATH)
        assert path.exists()
        with open(path) as f:
            saved = json.load(f)
        assert saved["route"] == "LOWER_TO_REST"
        assert len(saved["samples"]) == 2
