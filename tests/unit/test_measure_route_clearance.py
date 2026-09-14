"""scripts/measure_route_clearance.py: the reporting half (review 2026-09-12,
E1) and, since 2026-09-14, the recording half's aperture logging that E1 was
blocked on. Offline throughout -- a synthetic joint log stands in for a real
flight, exactly the way this module's own docstring says the recording half
cannot be exercised without hardware.

The module imports `reachy_sdk` at load time (same reason
`test_smoke_test_host.py` stubs it for `smoke_test_host`): it is not
installed here, matching a real host outside the simulator container.
"""

import json
import math
import os
import sys
import types

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../scripts"))
sys.path.insert(0, os.path.join(_HERE, "../../src"))

from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.motion.kinematics import (  # noqa: E402
    _GRIPPER_OPEN_LIMIT_DEG,
    _GRIPPER_SHUT_LIMIT_DEG,
    hand_radius,
    link_capsules,
)
from reachy_ai.scene.awareness import SceneModel  # noqa: E402

_SCENE_PATH = os.path.join(_HERE, "../../scenes/FWDCenterLabSivaPool.yaml")


@pytest.fixture
def mrc(monkeypatch):
    monkeypatch.setitem(sys.modules, "reachy_sdk",
                        types.SimpleNamespace(ReachySDK=object))
    sys.modules.pop("measure_route_clearance", None)
    import measure_route_clearance as m
    return m


#: Sentinel for `_synthetic_log`'s `gripper_deg` default, distinct from
#: `None` -- which is itself one of the invalid values these tests inject
#: on purpose (`test_non_numeric_or_non_finite_always_raises`), so the
#: "use the posture's own aperture" default cannot be spelled `None`.
_USE_POSTURE_APERTURE = object()


def _synthetic_log(pose, n=5, gripper_deg=_USE_POSTURE_APERTURE, omit_gripper=False):
    """A log that never moves -- every sample is the same pose. Includes a
    valid `r_gripper` reading by default (`pose["r_gripper"]`, the same
    posture the rest of the sample comes from). Pass `gripper_deg` to use a
    specific aperture instead -- including an intentionally invalid one
    (`None`, a string, NaN...) for the validation tests -- or
    `omit_gripper=True` to leave the key out entirely, simulating a
    schema_version 1 log for the missing-data tests below."""
    joints_base = {j: pose[j] for j in R.ARM7}
    samples = []
    for i in range(n):
        joints = dict(joints_base)
        if not omit_gripper:
            joints["r_gripper"] = (pose["r_gripper"]
                                   if gripper_deg is _USE_POSTURE_APERTURE
                                   else gripper_deg)
        samples.append({"t": i / 20.0, "joints": joints})
    return samples


class _StubJoint:
    def __init__(self, value):
        self.present_position = value


class _StubArm:
    """A read-only stand-in for `reachy.r_arm`: every `rig_routes.R_JOINTS`
    name (ARM7 plus `r_gripper`) has a `.present_position`, nothing else.
    `record_joint_log` only ever reads that attribute, so this is enough to
    prove it reads all eight joints the same way, gripper included."""

    def __init__(self, pose):
        for name in R.R_JOINTS:
            setattr(self, name, _StubJoint(pose[name]))


class TestRefusesWithoutTheEnvVar:
    def test_main_refuses_without_record_clearance(self, monkeypatch, mrc,
                                                    capsys):
        monkeypatch.delenv("REACHY_SIM_RECORD_CLEARANCE", raising=False)
        with pytest.raises(SystemExit) as exc:
            mrc.main()
        assert exc.value.code != 0
        assert "REACHY_SIM_RECORD_CLEARANCE" in capsys.readouterr().out

    def test_main_refuses_even_with_the_var_set_to_something_else(
            self, monkeypatch, mrc, capsys):
        monkeypatch.setenv("REACHY_SIM_RECORD_CLEARANCE", "true")
        with pytest.raises(SystemExit):
            mrc.main()
        assert "REACHY_SIM_RECORD_CLEARANCE" in capsys.readouterr().out


class TestRecordingIncludesTheAperture:
    """Replaces the old hard block (`APERTURE_LOGGED`, deleted this commit)
    with checks on what the recorder actually does now: every sample
    carries a real, validated `r_gripper` reading, in degrees, read the
    same way and at the same instant as the other seven joints -- proof the
    recorder meets E1's prerequisite, not a flag saying so."""

    def test_every_sample_includes_r_gripper(self, mrc):
        arm = _StubArm(R.REST)
        samples = mrc.record_joint_log(arm, duration_s=0.1, hz=50.0)
        assert samples
        for s in samples:
            assert "r_gripper" in s["joints"]

    def test_gripper_is_read_the_same_way_as_the_other_seven_joints(self, mrc):
        """record_joint_log must not special-case r_gripper: same
        getattr(...).present_position access, same sample, same instant."""
        arm = _StubArm(R.REST)
        [sample] = mrc.record_joint_log(arm, duration_s=0.05, hz=20.0)[:1]
        assert set(sample["joints"]) == set(R.R_JOINTS)
        for name in R.R_JOINTS:
            assert sample["joints"][name] == pytest.approx(R.REST[name])

    def test_recorded_values_are_degrees_not_radians(self, mrc):
        """R.REST's own r_gripper (-45.0) is a degree value in the MJCF's
        commanded range; a radian reading of the same physical angle would
        be off by a factor of ~57 and fail the aperture range check below.
        Confirms record_joint_log applies no unit conversion."""
        arm = _StubArm(R.REST)
        [sample] = mrc.record_joint_log(arm, duration_s=0.05, hz=20.0)[:1]
        assert sample["joints"]["r_gripper"] == pytest.approx(-45.0)
        # and it passes the same validation a real recorded log must pass
        mrc._validated_gripper_deg(sample["joints"]["r_gripper"], 0, 0.0)

    def test_sample_timing_is_elapsed_seconds_from_a_monotonic_clock(self, mrc):
        """`t` is `time.monotonic()`-based elapsed time since the call
        started -- non-decreasing, starting near zero, and roughly tracking
        the requested duration at the requested rate (best-effort: the
        module docstring says spacing is not guaranteed uniform, so this
        checks the shape, not exact spacing). Runs against the real clock
        rather than a mocked one, at a short duration, so it exercises the
        actual `time.monotonic()`/`time.sleep()` loop end to end."""
        arm = _StubArm(R.REST)
        hz, duration = 50.0, 0.2
        samples = mrc.record_joint_log(arm, duration_s=duration, hz=hz)
        ts = [s["t"] for s in samples]
        assert ts == sorted(ts)
        assert all(t >= 0 for t in ts)
        assert ts[0] == pytest.approx(0.0, abs=1.0 / hz)
        assert ts[-1] < duration
        # Roughly the expected count at this rate -- generous tolerance for
        # scheduling jitter under a test runner, not a precision claim.
        expected = duration * hz
        assert abs(len(samples) - expected) <= max(2, 0.5 * expected)

    def test_main_validates_the_recording_before_saving_or_reporting(
            self, monkeypatch, mrc, tmp_path, capsys):
        """A bad recording (here: r_gripper reads NaN, as a disconnected
        joint might) must stop main() before anything is written to
        `runs/` or printed as a report -- not just be caught later by
        someone reading the saved file."""
        monkeypatch.setenv("REACHY_SIM_RECORD_CLEARANCE", "1")
        monkeypatch.setattr(mrc, "RUNS_DIR", tmp_path)

        class _BadArm(_StubArm):
            def __init__(self, pose):
                super().__init__(pose)
                self.r_gripper = _StubJoint(float("nan"))

        monkeypatch.setattr(mrc, "ReachySDK",
                            lambda host, sdk_port: types.SimpleNamespace(
                                r_arm=_BadArm(R.REST)))
        monkeypatch.setattr(sys, "argv",
                            ["measure_route_clearance.py", "--route",
                             "LOWER_TO_REST", "--duration", "0.05"])
        with pytest.raises(SystemExit) as exc:
            mrc.main()
        assert exc.value.code != 0
        out = capsys.readouterr().out
        assert "not finite" in out or "FAIL" in out
        assert list(tmp_path.iterdir()) == []  # nothing saved


class TestApertureValidation:
    """`validated_samples` / `_validated_gripper_deg` in isolation: the
    rules from the module docstring's "Missing or invalid aperture"
    section, each checked on its own without going through `report()`."""

    def test_missing_aperture_raises_by_default(self, mrc):
        log = _synthetic_log(R.REST, n=3, omit_gripper=True)
        with pytest.raises(mrc.ApertureDataError, match=r"sample 0.*missing"):
            mrc.validated_samples(log)

    def test_missing_aperture_names_every_offending_sample_position(self, mrc):
        log = _synthetic_log(R.REST, n=3, omit_gripper=True)
        with pytest.raises(mrc.ApertureDataError) as exc:
            mrc.validated_samples(log)
        assert "sample 0" in str(exc.value)
        assert "t=0.0" in str(exc.value) or "t=0" in str(exc.value)

    def test_missing_aperture_can_be_explicitly_allowed(self, mrc):
        log = _synthetic_log(R.REST, n=3, omit_gripper=True)
        q7_and_gripper, assumed = mrc.validated_samples(
            log, allow_missing_aperture=True)
        assert assumed == [0, 1, 2]
        assert all(g is None for _q7, g in q7_and_gripper)

    @pytest.mark.parametrize("bad_value", [
        None, "wide open", float("nan"), float("inf"), float("-inf"),
    ])
    def test_non_numeric_or_non_finite_always_raises(self, mrc, bad_value):
        log = _synthetic_log(R.REST, n=1, gripper_deg=bad_value)
        with pytest.raises(mrc.ApertureDataError):
            mrc.validated_samples(log)
        # allow_missing_aperture does NOT rescue a present-but-bad value --
        # only a genuinely MISSING key.
        with pytest.raises(mrc.ApertureDataError):
            mrc.validated_samples(log, allow_missing_aperture=True)

    @pytest.mark.parametrize("out_of_range", [
        _GRIPPER_OPEN_LIMIT_DEG - 5.0,   # past the open limit
        _GRIPPER_SHUT_LIMIT_DEG + 5.0,   # past the shut limit
        999.0,
    ])
    def test_out_of_range_value_raises(self, mrc, out_of_range):
        log = _synthetic_log(R.REST, n=1, gripper_deg=out_of_range)
        with pytest.raises(mrc.ApertureDataError, match="outside"):
            mrc.validated_samples(log)

    @pytest.mark.parametrize("in_range", [
        _GRIPPER_OPEN_LIMIT_DEG, _GRIPPER_SHUT_LIMIT_DEG, 0.0, -45.0,
        _GRIPPER_OPEN_LIMIT_DEG - 0.4,   # inside the +/-0.5 deg tolerance
    ])
    def test_in_range_value_is_accepted(self, mrc, in_range):
        log = _synthetic_log(R.REST, n=1, gripper_deg=in_range)
        q7_and_gripper, assumed = mrc.validated_samples(log)
        assert assumed == []
        assert q7_and_gripper[0][1] == pytest.approx(in_range)


class TestAperturePropagatesIntoClearance:
    """The measured aperture must actually drive the geometry per sample --
    not a single default applied to the whole log."""

    def test_two_apertures_at_the_same_pose_give_different_realised_clearance(
            self, mrc):
        scene = SceneModel.from_yaml(_SCENE_PATH)
        wide_log = _synthetic_log(R.REST, n=4, gripper_deg=_GRIPPER_OPEN_LIMIT_DEG)
        shut_log = _synthetic_log(R.REST, n=4, gripper_deg=_GRIPPER_SHUT_LIMIT_DEG)
        wide = mrc.realised_clearance(wide_log, scene)
        shut = mrc.realised_clearance(shut_log, scene)
        assert wide["tube"] != shut["tube"]
        assert hand_radius(_GRIPPER_OPEN_LIMIT_DEG, R.REST["r_wrist_roll"]) > \
            hand_radius(_GRIPPER_SHUT_LIMIT_DEG, R.REST["r_wrist_roll"])

    def test_realised_clearance_matches_a_direct_link_capsules_call(self, mrc):
        """Cross-check against the geometry module directly, not just
        against itself: report()'s realised clearance for a single-sample
        log must equal `scene.clearances(link_capsules(...))` computed by
        hand at that pose and aperture."""
        scene = SceneModel.from_yaml(_SCENE_PATH)
        gripper_deg = -30.0
        log = _synthetic_log(R.REST, n=1, gripper_deg=gripper_deg)
        result = mrc.realised_clearance(log, scene)
        q7 = [R.REST[j] for j in R.ARM7]
        expected = scene.clearances(link_capsules(q7, "right", gripper_deg))
        for oid, c in expected.items():
            assert result["tube"][oid] == pytest.approx(c.distance)

    def test_per_sample_aperture_within_one_log(self, mrc):
        """A single log whose aperture changes sample to sample (as a real
        flight's would) must use each sample's own reading, not the
        first's or an aggregate."""
        pose = R.REST
        log = [
            {"t": 0.0, "joints": {**{j: pose[j] for j in R.ARM7},
                                  "r_gripper": _GRIPPER_SHUT_LIMIT_DEG}},
            {"t": 0.05, "joints": {**{j: pose[j] for j in R.ARM7},
                                   "r_gripper": _GRIPPER_OPEN_LIMIT_DEG}},
        ]
        scene = SceneModel.from_yaml(_SCENE_PATH)
        result = mrc.realised_clearance(log, scene)
        q7 = [pose[j] for j in R.ARM7]
        shut_only = scene.clearances(link_capsules(q7, "right", _GRIPPER_SHUT_LIMIT_DEG))
        open_only = scene.clearances(link_capsules(q7, "right", _GRIPPER_OPEN_LIMIT_DEG))
        for oid in result["tube"]:
            worst = min(shut_only[oid].distance, open_only[oid].distance)
            assert result["tube"][oid] == pytest.approx(worst)
            # and it is strictly the open sample driving it here, since the
            # open hand is never tighter than the shut one at REST
            assert result["tube"][oid] == pytest.approx(open_only[oid].distance)


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
        assert result["realised_aperture_assumed_samples"] == []
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
        """Consistent with the review's own finding: at REST on this
        scene, with REST's own commanded aperture, shells should read no
        tighter than tube for any object actually in the way. Not a
        universal claim (see test_arm_geometry_mjcf.py /
        test_footprint_boards.py for the measured exceptions) -- just true
        for this particular pose and aperture."""
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

    def test_report_raises_on_missing_aperture_by_default(self, mrc):
        """report() must not quietly fall back for a caller that did not
        ask for the fallback -- this is the behaviour that replaces the
        old hard block: it is not a flag check, it is that the reporting
        math itself refuses an unusable log."""
        log = _synthetic_log(R.REST, n=3, omit_gripper=True)
        with pytest.raises(mrc.ApertureDataError):
            mrc.report(log, "LOWER_TO_REST", _SCENE_PATH)

    def test_report_with_missing_aperture_allowed_names_the_assumed_samples(
            self, mrc):
        pose = R.REST
        log = _synthetic_log(pose, n=2)  # both valid, posture's own aperture
        log.append({"t": 2 / 20.0,
                   "joints": {j: pose[j] for j in R.ARM7}})  # missing r_gripper
        result = mrc.report(log, "LOWER_TO_REST", _SCENE_PATH,
                           allow_missing_aperture=True)
        assert result["realised_aperture_assumed_samples"] == [2]
        assert result["n_samples"] == 3


class TestBackwardCompatibilityWithSchemaVersion1Logs:
    """A log recorded before this commit -- no `r_gripper` key on any
    sample, no `schema_version` field -- must still be readable, but only
    when the caller explicitly says so, and only ever as the old
    assumed-open behaviour, visibly reported rather than silently revived."""

    def test_old_schema_log_reproduces_the_old_assumed_open_report(self, mrc):
        pose = R.REST
        old_style_log = [{"t": i / 20.0, "joints": {j: pose[j] for j in R.ARM7}}
                        for i in range(4)]  # no "r_gripper" anywhere, no
                                            # "schema_version" on the log dict
        scene = SceneModel.from_yaml(_SCENE_PATH)
        q7 = [pose[j] for j in R.ARM7]
        expected_open = scene.clearances(link_capsules(q7, "right", None))

        result = mrc.report(old_style_log, "LOWER_TO_REST", _SCENE_PATH,
                           allow_missing_aperture=True)
        assert result["realised_aperture_assumed_samples"] == [0, 1, 2, 3]
        for oid, c in expected_open.items():
            assert result["realised"]["tube"][oid] == pytest.approx(c.distance)

    def test_old_schema_log_refused_without_the_explicit_flag(self, mrc):
        pose = R.REST
        old_style_log = [{"t": 0.0, "joints": {j: pose[j] for j in R.ARM7}}]
        with pytest.raises(mrc.ApertureDataError, match="schema_version"):
            mrc.report(old_style_log, "LOWER_TO_REST", _SCENE_PATH)


class TestSaveLog:
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

    def test_save_log_writes_schema_version_and_the_aperture(self, mrc, tmp_path,
                                                              monkeypatch):
        monkeypatch.setattr(mrc, "RUNS_DIR", tmp_path)
        log = _synthetic_log(R.REST, n=2)
        path = mrc.save_log(log, "LOWER_TO_REST", _SCENE_PATH)
        with open(path) as f:
            saved = json.load(f)
        assert saved["schema_version"] == mrc.LOG_SCHEMA_VERSION == 2
        for sample in saved["samples"]:
            assert "r_gripper" in sample["joints"]
            assert isinstance(sample["joints"]["r_gripper"], (int, float))
