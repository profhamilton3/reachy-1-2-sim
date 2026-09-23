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
    joint_path,
    link_capsules,
)
from reachy_ai.scene.awareness import SceneModel, SceneObject  # noqa: E402

_SCENE_PATH = os.path.join(_HERE, "../../scenes/FWDCenterLabSivaPool.yaml")


@pytest.fixture
def mrc(monkeypatch):
    monkeypatch.setitem(sys.modules, "reachy_sdk",
                        types.SimpleNamespace(ReachySDK=object))
    sys.modules.pop("measure_route_clearance", None)
    import measure_route_clearance as m
    return m


def _stub_identity_ok(monkeypatch, mrc):
    """Bypass e1_identity.verify_simulator_identity with an ok=True stub --
    for tests exercising main()'s APERTURE path (E1 readiness work item 2
    added an identity check ahead of it; see test_e1_identity.py for that
    check's own tests)."""
    monkeypatch.setattr(
        mrc.e1_identity, "verify_simulator_identity",
        lambda **kw: mrc.e1_identity.SimulatorIdentityCheck(
            ok=True, reasons=(), run_dir="", manifest={},
            scene_chain_sha256={}, joint_agreement_deg={}, status={},
            checked_at_wall_ns=0))


#: Sentinel for `_synthetic_log`'s `gripper_deg` default, distinct from
#: `None` -- which is itself one of the invalid values these tests inject
#: on purpose (`test_non_numeric_or_non_finite_always_raises`), so the
#: "use the posture's own aperture" default cannot be spelled `None`.
_USE_POSTURE_APERTURE = object()


def _synthetic_log(pose, n=5, gripper_deg=_USE_POSTURE_APERTURE, omit_gripper=False,
                   omit_wall_time_ns=False):
    """A log that never moves -- every sample is the same pose. Includes a
    valid `r_gripper` reading by default (`pose["r_gripper"]`, the same
    posture the rest of the sample comes from). Pass `gripper_deg` to use a
    specific aperture instead -- including an intentionally invalid one
    (`None`, a string, NaN...) for the validation tests -- or
    `omit_gripper=True` to leave the key out entirely, simulating a
    schema_version 1 log for the missing-data tests below.

    Includes a synthetic (but real-shaped) `wall_time_ns` per sample by
    default, 20 ms apart -- the current schema (3) requires it; pass
    `omit_wall_time_ns=True` to build a schema-3-shaped log missing it, for
    TestSchemaAwareLoading's own test of that requirement."""
    joints_base = {j: pose[j] for j in R.ARM7}
    samples = []
    for i in range(n):
        joints = dict(joints_base)
        if not omit_gripper:
            joints["r_gripper"] = (pose["r_gripper"]
                                   if gripper_deg is _USE_POSTURE_APERTURE
                                   else gripper_deg)
        sample = {"t": i / 20.0, "joints": joints}
        if not omit_wall_time_ns:
            sample["wall_time_ns"] = 1_000_000_000 + i * 20_000_000
        samples.append(sample)
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


class TestLazySdkConstruction:
    """N1 (2026-09-15 matrix readiness): `ReachySDK` is constructed lazily,
    inside `_read_sdk_joints`, which `verify_simulator_identity` only calls
    AFTER the loopback and run-dir checks pass. A mistyped `--host` (or a
    stray `REACHY_IP`, the `--host` default) must never open a gRPC
    channel to whatever answers there before the refusal fires."""

    def test_reachysdk_not_constructed_before_loopback_refusal(
            self, monkeypatch, mrc, tmp_path):
        """Uses the REAL e1_identity.verify_simulator_identity (not
        stubbed), so the loopback check that fires first actually runs."""
        monkeypatch.setenv("REACHY_SIM_RECORD_CLEARANCE", "1")
        monkeypatch.setattr(mrc, "RUNS_DIR", tmp_path)
        calls = []
        monkeypatch.setattr(
            mrc, "ReachySDK",
            lambda host, sdk_port: (
                calls.append((host, sdk_port)),
                types.SimpleNamespace(r_arm=_StubArm(R.REST)))[1])
        monkeypatch.setattr(
            sys, "argv",
            ["measure_route_clearance.py", "--host", "192.168.1.50",
             "--route", "LOWER_TO_REST", "--duration", "0.05",
             "--record-root", str(tmp_path)])
        with pytest.raises(SystemExit) as exc:
            mrc.main()
        assert exc.value.code == 2
        assert calls == [], (
            "ReachySDK must not be constructed before the loopback refusal")

    def test_reachysdk_is_constructed_once_identity_passes(
            self, monkeypatch, mrc, tmp_path):
        monkeypatch.setenv("REACHY_SIM_RECORD_CLEARANCE", "1")
        monkeypatch.setattr(mrc, "RUNS_DIR", tmp_path)
        calls = []
        monkeypatch.setattr(
            mrc, "ReachySDK",
            lambda host, sdk_port: (
                calls.append((host, sdk_port)),
                types.SimpleNamespace(r_arm=_StubArm(R.REST)))[1])
        _stub_identity_ok(monkeypatch, mrc)
        # This test is only about WHEN ReachySDK is constructed, not the
        # sidecar-linking pipeline (covered by test_link_e1_flight.py and
        # test_e1_identity.py) -- stub it out so a bare identity.run_dir=""
        # from _stub_identity_ok doesn't fail this test for an unrelated
        # reason.
        monkeypatch.setattr(
            mrc.link_e1_flight, "build_base_sidecar",
            lambda **kw: {"settled_pose_check": {}})
        monkeypatch.setattr(
            sys, "argv",
            ["measure_route_clearance.py", "--route", "LOWER_TO_REST",
             "--duration", "0.05", "--record-root", str(tmp_path)])
        mrc.main()
        assert len(calls) == 1


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

    def test_main_validates_the_recording_before_saving_a_normal_log(
            self, monkeypatch, mrc, tmp_path, capsys):
        """A bad recording (here: r_gripper reads NaN, as a disconnected
        joint might) must stop main() before anything is written under a
        NORMAL (`save_log`) filename or printed as a report -- not just be
        caught later by someone reading the saved file. It is not thrown
        away, though -- see test_main_preserves_a_failed_recording below."""
        monkeypatch.setenv("REACHY_SIM_RECORD_CLEARANCE", "1")
        monkeypatch.setattr(mrc, "RUNS_DIR", tmp_path)

        class _BadArm(_StubArm):
            def __init__(self, pose):
                super().__init__(pose)
                self.r_gripper = _StubJoint(float("nan"))

        monkeypatch.setattr(mrc, "ReachySDK",
                            lambda host, sdk_port: types.SimpleNamespace(
                                r_arm=_BadArm(R.REST)))
        _stub_identity_ok(monkeypatch, mrc)
        monkeypatch.setattr(sys, "argv",
                            ["measure_route_clearance.py", "--route",
                             "LOWER_TO_REST", "--duration", "0.05",
                             "--record-root", str(tmp_path)])
        with pytest.raises(SystemExit) as exc:
            mrc.main()
        assert exc.value.code != 0
        out = capsys.readouterr().out
        assert "not finite" in out or "FAIL" in out
        saved = list(tmp_path.iterdir())
        assert all("_INVALID" in p.name for p in saved), (
            "only the marked-invalid diagnostic file may exist -- never a "
            "file save_log's own naming scheme would produce")

    def test_main_preserves_a_failed_recording_as_invalid_diagnostic(
            self, monkeypatch, mrc, tmp_path, capsys):
        """The samples that WERE recorded before the bad reading are not
        lost: main() saves them, marked unambiguously invalid, so the
        flight can be diagnosed instead of re-flown blind (finding 2)."""
        monkeypatch.setenv("REACHY_SIM_RECORD_CLEARANCE", "1")
        monkeypatch.setattr(mrc, "RUNS_DIR", tmp_path)

        class _BadArm(_StubArm):
            def __init__(self, pose):
                super().__init__(pose)
                self.r_gripper = _StubJoint(float("nan"))

        monkeypatch.setattr(mrc, "ReachySDK",
                            lambda host, sdk_port: types.SimpleNamespace(
                                r_arm=_BadArm(R.REST)))
        _stub_identity_ok(monkeypatch, mrc)
        monkeypatch.setattr(sys, "argv",
                            ["measure_route_clearance.py", "--route",
                             "LOWER_TO_REST", "--duration", "0.05",
                             "--record-root", str(tmp_path)])
        with pytest.raises(SystemExit):
            mrc.main()

        saved = list(tmp_path.iterdir())
        assert len(saved) == 1
        path = saved[0]
        assert path.name.endswith("_INVALID.json")
        with open(path) as f:
            doc = json.load(f)
        assert doc["valid"] is False
        assert "not finite" in doc["error"]
        assert doc["route"] == "LOWER_TO_REST"
        assert len(doc["samples"]) >= 1
        # every sample the (short) recording actually produced is kept,
        # NaN aperture included -- nothing here silently drops it
        assert any(
            isinstance(s["joints"]["r_gripper"], float)
            and math.isnan(s["joints"]["r_gripper"])
            for s in doc["samples"])
        # and it can never pass for valid E1 data if read back the normal
        # (schema-aware) way -- refused outright, not merely re-raising
        with pytest.raises(mrc.UnsupportedSchemaVersionError):
            mrc.validated_samples(
                doc["samples"], schema_version=mrc.schema_version_of(doc),
                allow_missing_aperture=True)


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

    def test_missing_aperture_can_be_explicitly_allowed_for_a_legacy_schema(
            self, mrc):
        """The rescue applies only to a log DECLARED as a supported legacy
        schema (schema_version=1 here) -- see TestSchemaAwareLoading for
        what happens when the same missing key is declared current."""
        log = _synthetic_log(R.REST, n=3, omit_gripper=True)
        q7_and_gripper, assumed = mrc.validated_samples(
            log, schema_version=1, allow_missing_aperture=True)
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


class TestSchemaAwareLoading:
    """`schema_version` -- not just per-sample key presence -- gates
    whether `allow_missing_aperture` may rescue a missing reading.
    Confirms the two review requirements directly: a CURRENT-schema log
    with a missing aperture always fails, and an UNKNOWN schema_version is
    refused outright rather than guessed at."""

    def test_current_schema_missing_aperture_always_raises(self, mrc):
        """schema_version == LOG_SCHEMA_VERSION (3) with a missing
        r_gripper key must raise even with the flag -- a gap in the
        CURRENT schema is a dropped field or a bad recording, never an
        old log format."""
        log = _synthetic_log(R.REST, n=2, omit_gripper=True)
        with pytest.raises(mrc.ApertureDataError):
            mrc.validated_samples(
                log, schema_version=mrc.LOG_SCHEMA_VERSION,
                allow_missing_aperture=False)
        with pytest.raises(mrc.ApertureDataError):
            mrc.validated_samples(
                log, schema_version=mrc.LOG_SCHEMA_VERSION,
                allow_missing_aperture=True)

    def test_schema_2_is_recognised_but_still_raises_on_missing_aperture(
            self, mrc):
        """E1 readiness (assignment 2026-09-14, work item 3) contract
        detail resolved before implementation: schema 2 became a
        RECOGNISED legacy schema the day schema 3 shipped (no longer
        UnsupportedSchemaVersionError) -- but it was never eligible for the
        missing-aperture rescue, because schema 2 logs always had
        r_gripper. A missing key in a declared schema-2 log must still
        raise ApertureDataError, allow_missing_aperture notwithstanding."""
        assert 2 in mrc._SUPPORTED_LEGACY_SCHEMA_VERSIONS
        assert 2 not in mrc._MISSING_APERTURE_ELIGIBLE_SCHEMA_VERSIONS
        log = _synthetic_log(R.REST, n=2, omit_gripper=True)
        with pytest.raises(mrc.ApertureDataError):
            mrc.validated_samples(log, schema_version=2,
                                  allow_missing_aperture=False)
        with pytest.raises(mrc.ApertureDataError):
            mrc.validated_samples(log, schema_version=2,
                                  allow_missing_aperture=True)

    def test_current_schema_missing_wall_time_ns_raises_sample_timestamp_error(
            self, mrc):
        """The other half of the same work item: a schema-3 (current) log
        missing wall_time_ns is rejected by validated_samples -- a distinct
        exception from ApertureDataError, since the aperture itself is
        fine here."""
        log = _synthetic_log(R.REST, n=2, omit_wall_time_ns=True)
        with pytest.raises(mrc.SampleTimestampError):
            mrc.validated_samples(log, schema_version=mrc.LOG_SCHEMA_VERSION)
        # schema 1 and 2 predate the field -- never checked for it
        mrc.validated_samples(log, schema_version=1, allow_missing_aperture=True)
        mrc.validated_samples(log, schema_version=2)

    def test_missing_both_aperture_and_wall_time_ns_raises_aperture_error_first(
            self, mrc):
        """A sample missing both fields is diagnosed as an aperture
        problem, not a timestamp one -- aperture is checked first per
        sample (see validated_samples). This is the exact shape of several
        pre-existing tests below (old_style_log has neither field), so the
        check order is pinned here rather than left implicit."""
        log = _synthetic_log(R.REST, n=1, omit_gripper=True,
                             omit_wall_time_ns=True)
        with pytest.raises(mrc.ApertureDataError):
            mrc.validated_samples(log, schema_version=mrc.LOG_SCHEMA_VERSION)

    def test_default_schema_version_is_current_so_the_flag_alone_never_rescues(
            self, mrc):
        """Calling validated_samples/report without specifying
        schema_version (the ordinary case for a fresh recording) must
        behave as schema_version==LOG_SCHEMA_VERSION -- allow_missing_aperture
        by itself must not be enough."""
        log = _synthetic_log(R.REST, n=2, omit_gripper=True)
        with pytest.raises(mrc.ApertureDataError):
            mrc.validated_samples(log, allow_missing_aperture=True)
        with pytest.raises(mrc.ApertureDataError):
            mrc.report(log, "LOWER_TO_REST", _SCENE_PATH,
                       allow_missing_aperture=True)

    def test_legacy_schema_missing_aperture_still_requires_the_flag(self, mrc):
        """schema_version==1 alone is not enough either -- the flag is
        still required, only now it is actually able to work."""
        log = _synthetic_log(R.REST, n=2, omit_gripper=True)
        with pytest.raises(mrc.ApertureDataError):
            mrc.validated_samples(log, schema_version=1,
                                  allow_missing_aperture=False)

    @pytest.mark.parametrize("bad_version", [0, 4, -1, 1.5, "2"])
    def test_unrecognised_schema_version_is_rejected_outright(
            self, mrc, bad_version):
        """Neither the current schema nor a version this module has
        explicit legacy fallback logic for -- refused before any
        per-sample check, regardless of the flag, and with a distinct
        exception type from a data problem."""
        log = _synthetic_log(R.REST, n=2)  # samples are otherwise fine
        for allow in (False, True):
            with pytest.raises(mrc.UnsupportedSchemaVersionError):
                mrc.validated_samples(
                    log, schema_version=bad_version,
                    allow_missing_aperture=allow)
        assert not issubclass(
            mrc.UnsupportedSchemaVersionError, mrc.ApertureDataError)
        assert not issubclass(
            mrc.ApertureDataError, mrc.UnsupportedSchemaVersionError)

    def test_invalid_log_schema_version_sentinel_is_itself_unrecognised(
            self, mrc):
        """save_invalid_log's own sentinel must be a value this rejection
        path actually rejects -- otherwise a diagnostic file could be
        mistaken for a real schema."""
        assert mrc._INVALID_LOG_SCHEMA_VERSION not in (
            (mrc.LOG_SCHEMA_VERSION,) + mrc._SUPPORTED_LEGACY_SCHEMA_VERSIONS)

    def test_load_log_round_trips_a_saved_log_and_its_schema_version(
            self, mrc, tmp_path, monkeypatch):
        monkeypatch.setattr(mrc, "RUNS_DIR", tmp_path)
        log = _synthetic_log(R.REST, n=3)
        path = mrc.save_log(log, "LOWER_TO_REST", _SCENE_PATH)
        loaded = mrc.load_log(path)
        assert mrc.schema_version_of(loaded) == mrc.LOG_SCHEMA_VERSION == 3
        assert loaded["samples"] == log
        # and reading it the schema-aware way works end to end
        result = mrc.report(loaded["samples"], loaded["route"],
                           loaded["scene"],
                           schema_version=mrc.schema_version_of(loaded))
        assert result["realised_aperture_assumed_samples"] == []

    def test_schema_version_of_treats_an_absent_field_as_legacy_1(self, mrc):
        legacy_shaped = {"route": "LOWER_TO_REST", "samples": []}
        assert mrc.schema_version_of(legacy_shaped) == 1
        assert 1 in mrc._SUPPORTED_LEGACY_SCHEMA_VERSIONS


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
            {"t": 0.0, "wall_time_ns": 1_000_000_000,
             "joints": {**{j: pose[j] for j in R.ARM7},
                       "r_gripper": _GRIPPER_SHUT_LIMIT_DEG}},
            {"t": 0.05, "wall_time_ns": 1_050_000_000,
             "joints": {**{j: pose[j] for j in R.ARM7},
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
        """A log DECLARED schema_version=1 (a mixed/partially-migrated
        legacy log, one sample missing the key) is rescued sample-by
        sample; see TestSchemaAwareLoading for the schema_version==2
        case, which never rescues."""
        pose = R.REST
        log = _synthetic_log(pose, n=2)  # both valid, posture's own aperture
        log.append({"t": 2 / 20.0,
                   "joints": {j: pose[j] for j in R.ARM7}})  # missing r_gripper
        result = mrc.report(log, "LOWER_TO_REST", _SCENE_PATH,
                           schema_version=1, allow_missing_aperture=True)
        assert result["realised_aperture_assumed_samples"] == [2]
        assert result["n_samples"] == 3

    def test_report_includes_aperture_provenance_strings(self, mrc):
        """Finding 3: report() must say, in the result itself, where each
        half's aperture came from -- not only in a docstring."""
        log = _synthetic_log(R.REST, n=3)
        result = mrc.report(log, "LOWER_TO_REST", _SCENE_PATH)
        assert result["realised_aperture_source"] == mrc.REALISED_APERTURE_SOURCE
        assert result["planned_aperture_policy"] == mrc.PLANNED_APERTURE_POLICY
        assert "measured" in result["realised_aperture_source"]
        assert "per-sample" in result["realised_aperture_source"]
        assert "ENDPOINT" in result["planned_aperture_policy"]
        assert "worst-case" in result["planned_aperture_policy"]

    def test_tube_planned_clearance_matches_the_actual_endpoint_selection(
            self, mrc):
        """Ties `planned_aperture_policy`'s TUBE claim -- worst-case of the
        two commanded LEG-ENDPOINT apertures, applied to every interpolated
        sample on that leg -- to real numbers, computed independently of
        `planned_clearance` itself (a link_capsules call per leg-endpoint
        choice, not a call into the function under test). shells does NOT
        follow this rule any more (see the class below); this test covers
        tube only."""
        route = "LOWER_TO_REST"
        scene = SceneModel.from_yaml(_SCENE_PATH)
        waypoints = R.FOOTPRINT_LEGS[route]
        assert len(waypoints) >= 2  # otherwise this route can't show the policy

        out = {}
        for a, b in zip(waypoints, waypoints[1:]):
            qa = [a[j] for j in R.ARM7]
            qb = [b[j] for j in R.ARM7]
            # independently reproduce "worst-case of the two ENDPOINTS":
            # whichever endpoint's own aperture yields the larger
            # hand_radius, used for every point on this leg.
            a_r = hand_radius(a["r_gripper"], a["r_wrist_roll"])
            b_r = hand_radius(b["r_gripper"], b["r_wrist_roll"])
            worst_gripper = a["r_gripper"] if a_r >= b_r else b["r_gripper"]
            for q in joint_path(qa, qb, steps=13):
                caps = link_capsules(q, "right", worst_gripper, hand="tube")
                for oid, c in scene.clearances(caps).items():
                    worst = out.get(oid)
                    if worst is None or c.distance < worst:
                        out[oid] = c.distance

        actual = mrc.planned_clearance(route, 13, scene)
        for oid in out:
            assert actual["tube"][oid] == pytest.approx(out[oid]), (
                f"planned_aperture_policy's tube claim doesn't match "
                f"planned_clearance's real behaviour for {oid}")

    def test_shells_planned_clearance_interpolates_aperture_per_sample(
            self, mrc):
        """Ties `planned_aperture_policy`'s SHELLS claim -- the aperture is
        interpolated per sample in lockstep with the arm joints, not held at
        a leg-endpoint -- to real numbers, computed independently of
        `planned_clearance_by_link` (a link_capsules call per interpolated
        sample, not a call into the function under test).

        Uses the B2 scene and `link="finger"`, not the generic pool scene
        this test used before (issue #137 review): on the pool scene
        `finger` never binds, so this test passed under the endpoint-rule
        bug AND under the hold-at-start mutation alike -- it never actually
        exercised the interpolation policy. B2 is the scene where `finger`
        does bind (see TestB2S2FingerClearanceRootCause below)."""
        route = "LOWER_TO_REST"
        scene = SceneModel.from_yaml(_B2_SCENE_PATH)
        waypoints = R.FOOTPRINT_LEGS[route]
        steps = 13

        out = {}
        for a, b in zip(waypoints, waypoints[1:]):
            qa = [a[j] for j in R.ARM7]
            qb = [b[j] for j in R.ARM7]
            path = joint_path(qa, qb, steps=steps)
            for i, q in enumerate(path):
                t = i / (steps - 1)
                gripper_deg = a["r_gripper"] + t * (
                    b["r_gripper"] - a["r_gripper"])
                caps = [c for c in
                       link_capsules(q, "right", gripper_deg, hand="shells")
                       if c[0] == "finger"]
                for oid, c in scene.clearances(caps).items():
                    worst = out.get(oid)
                    if worst is None or c.distance < worst:
                        out[oid] = c.distance

        actual = mrc.planned_clearance_by_link(
            route, steps, scene, "finger", hand="shells")
        for oid in out:
            assert actual[oid] == pytest.approx(out[oid]), (
                f"planned_aperture_policy's shells claim doesn't match "
                f"planned_clearance_by_link's real behaviour for {oid}")


class TestBackwardCompatibilityWithSchemaVersion1Logs:
    """A log recorded before this commit -- no `r_gripper` key on any
    sample, no `schema_version` field on disk (schema_version_of treats
    that as 1) -- must still be readable, but only when the caller
    explicitly declares `schema_version=1` AND passes
    `allow_missing_aperture=True`, and only ever as the old assumed-open
    behaviour, visibly reported rather than silently revived."""

    def test_old_schema_log_reproduces_the_old_assumed_open_report(self, mrc):
        pose = R.REST
        old_style_log = [{"t": i / 20.0, "joints": {j: pose[j] for j in R.ARM7}}
                        for i in range(4)]  # no "r_gripper" anywhere, no
                                            # "schema_version" on the log dict
        scene = SceneModel.from_yaml(_SCENE_PATH)
        q7 = [pose[j] for j in R.ARM7]
        expected_open = scene.clearances(link_capsules(q7, "right", None))

        result = mrc.report(old_style_log, "LOWER_TO_REST", _SCENE_PATH,
                           schema_version=1, allow_missing_aperture=True)
        assert result["realised_aperture_assumed_samples"] == [0, 1, 2, 3]
        for oid, c in expected_open.items():
            assert result["realised"]["tube"][oid] == pytest.approx(c.distance)

    def test_old_schema_log_refused_without_the_explicit_flag(self, mrc):
        """schema_version=1 alone is not enough -- allow_missing_aperture
        is still required even for a genuinely legacy-declared log."""
        pose = R.REST
        old_style_log = [{"t": 0.0, "joints": {j: pose[j] for j in R.ARM7}}]
        with pytest.raises(mrc.ApertureDataError, match="schema_version"):
            mrc.report(old_style_log, "LOWER_TO_REST", _SCENE_PATH,
                       schema_version=1)

    def test_old_schema_log_refused_by_default_schema_version_even_with_the_flag(
            self, mrc):
        """The exact scenario finding 1 is about: calling report() the
        ordinary way (no schema_version override) on a log missing
        r_gripper must raise even WITH allow_missing_aperture=True,
        because the default schema_version is the CURRENT one, and a
        current-schema log has no excuse for a missing aperture."""
        pose = R.REST
        old_style_log = [{"t": 0.0, "joints": {j: pose[j] for j in R.ARM7}}]
        with pytest.raises(mrc.ApertureDataError):
            mrc.report(old_style_log, "LOWER_TO_REST", _SCENE_PATH,
                       allow_missing_aperture=True)

    def test_old_schema_log_from_disk_round_trips_through_load_log(
            self, mrc, tmp_path):
        """The realistic path: a genuine schema-1 file on disk (as written
        before this module existed), loaded and read via
        schema_version_of -- not a hand-typed schema_version kwarg."""
        pose = R.REST
        on_disk = {"route": "LOWER_TO_REST", "scene": _SCENE_PATH,
                  "sample_hz": 20.0,
                  "samples": [{"t": 0.0, "joints": {j: pose[j] for j in R.ARM7}}]}
                  # no "schema_version" key at all -- a real pre-existing file
        path = tmp_path / "old.json"
        with open(path, "w") as f:
            json.dump(on_disk, f)

        loaded = mrc.load_log(path)
        version = mrc.schema_version_of(loaded)
        assert version == 1
        result = mrc.report(loaded["samples"], loaded["route"], loaded["scene"],
                           schema_version=version, allow_missing_aperture=True)
        assert result["realised_aperture_assumed_samples"] == [0]


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
        assert saved["schema_version"] == mrc.LOG_SCHEMA_VERSION == 3
        for sample in saved["samples"]:
            assert "r_gripper" in sample["joints"]
            assert isinstance(sample["joints"]["r_gripper"], (int, float))


class TestSaveInvalidLog:
    """`save_invalid_log` in isolation (finding 2) -- exercised directly,
    not only through `main()`'s one failure path, so its contract is
    pinned regardless of what triggers it."""

    def test_writes_a_distinctly_named_file_with_reason_and_samples(
            self, mrc, tmp_path, monkeypatch):
        monkeypatch.setattr(mrc, "RUNS_DIR", tmp_path)
        log = _synthetic_log(R.REST, n=3, omit_gripper=True)
        path = mrc.save_invalid_log(log, "LOWER_TO_REST", _SCENE_PATH,
                                    "sample 1: something went wrong")
        assert path.name.endswith("_INVALID.json")
        with open(path) as f:
            doc = json.load(f)
        assert doc["valid"] is False
        assert doc["error"] == "sample 1: something went wrong"
        assert doc["route"] == "LOWER_TO_REST"
        assert len(doc["samples"]) == 3  # every recorded sample kept

    def test_never_collides_with_a_normal_save_log_filename(
            self, mrc, tmp_path, monkeypatch):
        """save_log and save_invalid_log must never be able to produce the
        same filename for the same route/timestamp -- the suffix is what
        keeps a normal analysis path from ever opening an invalid log by
        accident."""
        monkeypatch.setattr(mrc, "RUNS_DIR", tmp_path)
        log = _synthetic_log(R.REST, n=2)
        good = mrc.save_log(log, "LOWER_TO_REST", _SCENE_PATH)
        bad = mrc.save_invalid_log(log, "LOWER_TO_REST", _SCENE_PATH, "boom")
        assert good != bad
        assert "_INVALID" not in good.name
        assert "_INVALID" in bad.name

    def test_schema_version_is_never_a_recognised_one(self, mrc, tmp_path,
                                                       monkeypatch):
        monkeypatch.setattr(mrc, "RUNS_DIR", tmp_path)
        log = _synthetic_log(R.REST, n=1)
        path = mrc.save_invalid_log(log, "LOWER_TO_REST", _SCENE_PATH, "boom")
        doc = mrc.load_log(path)
        version = mrc.schema_version_of(doc)
        assert version not in (
            (mrc.LOG_SCHEMA_VERSION,) + mrc._SUPPORTED_LEGACY_SCHEMA_VERSIONS)
        with pytest.raises(mrc.UnsupportedSchemaVersionError):
            mrc.validated_samples(doc["samples"], schema_version=version,
                                  allow_missing_aperture=True)

    def test_preserves_the_offending_sample_itself(self, mrc, tmp_path,
                                                    monkeypatch):
        """The sample that failed validation is not stripped out before
        saving -- it is exactly what a diagnosis needs to see."""
        monkeypatch.setattr(mrc, "RUNS_DIR", tmp_path)
        log = _synthetic_log(R.REST, n=2, gripper_deg=float("nan"))
        path = mrc.save_invalid_log(log, "LOWER_TO_REST", _SCENE_PATH,
                                    "sample 0: r_gripper=nan is not finite")
        doc = mrc.load_log(path)
        assert all(math.isnan(s["joints"]["r_gripper"]) for s in doc["samples"])


# ── Regression: the 2026-09-22 B2 s2 finger-clearance root cause ───────────
# outputs/analysis-2026-09-22-b2s2-finger-clearance-root-cause.md. Two
# independent, previously-conflated bugs:
#
#   (a) shells held aperture at the tube rule's worst-case ENDPOINT, which
#       for LOWER_TO_REST is OPEN -- but `finger` is closest to an object
#       when SHUT, so the endpoint rule never evaluated the pose the arm
#       actually passes through. Fixed by `_shells_leg_samples` (per-sample
#       interpolation) in `planned_clearance`/`planned_clearance_by_link`.
#   (b) a "setup" recording flies a route's FULL sequence (from HOME), but
#       was compared against `FOOTPRINT_LEGS`'s guard-scoped subset -- a
#       different, shorter arc. Fixed by `full_route_poses` /
#       `report(..., full_route=True)`.
#
# Both boards are real scene YAMLs already in the repo
# (scenes/e1_boards/*.yaml) -- the exact scenes the flown sessions used.
_B2_SCENE_PATH = os.path.join(
    _HERE, "../../scenes/e1_boards/B2_foam_r3c3.yaml")
_B1_SCENE_PATH = os.path.join(
    _HERE, "../../scenes/e1_boards/B1_evidence.yaml")


class TestB2S2FingerClearanceRootCause:
    """Pins the corrected numbers from the root-cause analysis, and
    independently reproduces the old (wrong) numbers they replace, so a
    future change to either policy has to look at both."""

    def test_lower_to_rest_finger_old_endpoint_policy_was_9_41cm(self, mrc):
        """Reproduces the BUG: holding shells' aperture at LOWER_TO_REST's
        worst-hand_radius ENDPOINT (OPEN, -45deg) on every leg never
        evaluates the SHUT pose the arm actually reaches at REST_SHUT,
        understating how close `finger` gets to `foam_block`. Computed
        independently of `planned_clearance_by_link` (not a call into
        current, fixed code) -- this pins the historical wrong answer,
        not today's behaviour."""
        route = "LOWER_TO_REST"
        scene = SceneModel.from_yaml(_B2_SCENE_PATH)
        waypoints = R.FOOTPRINT_LEGS[route]
        best = None
        for a, b in zip(waypoints, waypoints[1:]):
            qa = [a[j] for j in R.ARM7]
            qb = [b[j] for j in R.ARM7]
            worst_gripper = max(a["r_gripper"], b["r_gripper"], key=hand_radius)
            for q in joint_path(qa, qb, steps=200):
                for c in link_capsules(q, "right", worst_gripper, hand="shells"):
                    if c[0] != "finger":
                        continue
                    d = scene.clearances([c])["foam_block"].distance
                    if best is None or d < best:
                        best = d
        assert best * 100 == pytest.approx(9.41, abs=0.01)

    def test_lower_to_rest_finger_corrected_planned_is_3_37cm(self, mrc):
        """The fix: `planned_clearance_by_link` interpolates shells'
        aperture per sample, so it does evaluate the SHUT pose at
        REST_SHUT -- the actual worst point, 3.37 cm, not 9.41."""
        scene = SceneModel.from_yaml(_B2_SCENE_PATH)
        result = mrc.planned_clearance_by_link(
            "LOWER_TO_REST", 200, scene, "finger", hand="shells")
        assert result["foam_block"] * 100 == pytest.approx(3.37, abs=0.01)

    def test_lower_to_rest_finger_corrected_is_close_to_the_realised_worst(
            self, mrc):
        """The whole point of the fix: the corrected planned value should
        sit close to (and, per the analysis, just under) the realised
        worst sample of 3.543/3.545 cm actually flown -- not 5.87 cm away
        from it, which is what the old, wrong 9.41 cm produced."""
        scene = SceneModel.from_yaml(_B2_SCENE_PATH)
        result = mrc.planned_clearance_by_link(
            "LOWER_TO_REST", 200, scene, "finger", hand="shells")
        realised_worst_cm = 3.543
        delta = result["foam_block"] * 100 - realised_worst_cm
        assert delta == pytest.approx(-0.17, abs=0.02)

    def test_place_route_footprint_scope_understates_setup_role_by_route_truncation(
            self, mrc):
        """Reproduces the SECOND bug: FOOTPRINT_LEGS["PLACE_ROUTE"] is only
        (HOVER, REST_SHUT, REST) -- a "setup" recording that flies the
        WHOLE named route from HOME passes through SWING_2, far closer to
        `soda_can` than anything in that scoped-down subset, so comparing
        it against the footprint-only planned value is comparing two
        different arcs. This pins the historical, route-truncated wrong
        answer (43.28), not today's full-route behaviour."""
        scene = SceneModel.from_yaml(_B1_SCENE_PATH)
        footprint_only = mrc.planned_clearance_by_link(
            "PLACE_ROUTE", 200, scene, "finger", hand="shells")
        assert footprint_only["soda_can"] * 100 == pytest.approx(43.28, abs=0.01)

    def test_place_route_full_route_finger_corrected_is_31_66cm(self, mrc):
        """The fix: comparing against `full_route_poses("PLACE_ROUTE")`
        (the whole flown route, HOME included) rather than
        `FOOTPRINT_LEGS`'s scoped-down subset gives 31.66 cm, matching the
        realised worst sample (31.38 cm, near SWING_2) to within 0.3 cm --
        not the footprint-only 43.28 cm, an 11.89 cm route-scope
        artefact."""
        scene = SceneModel.from_yaml(_B1_SCENE_PATH)
        full = mrc.full_route_poses("PLACE_ROUTE")
        result = mrc.planned_clearance_by_link(
            "PLACE_ROUTE", 200, scene, "finger", hand="shells", waypoints=full)
        assert result["soda_can"] * 100 == pytest.approx(31.66, abs=0.01)

    def test_full_route_poses_starts_with_the_departed_posture(self, mrc):
        """PLACE_ROUTE and RAISE_TO_SIDE leave the rail pocket (HOME);
        LOWER_TO_REST and STOW_FROM_SIDE start already at the side pose
        (PRESENT); STOW_ROUTE and LIFT_TO_PRESENT start at the tabletop
        rest pose (REST) -- the posture each route's own first named
        waypoint is actually flown from, per rig_routes.py's own
        documentation of what `fly_route` commands."""
        assert mrc.full_route_poses("PLACE_ROUTE")[0] == R.HOME
        assert mrc.full_route_poses("RAISE_TO_SIDE")[0] == R.HOME
        assert mrc.full_route_poses("STOW_ROUTE")[0] == R.REST
        assert mrc.full_route_poses("STOW_FROM_SIDE")[0] == R.PRESENT
        assert mrc.full_route_poses("LOWER_TO_REST")[0] == R.PRESENT
        assert mrc.full_route_poses("LIFT_TO_PRESENT")[0] == R.REST

    def test_full_route_poses_for_lower_to_rest_matches_footprint_legs(
            self, mrc):
        """LOWER_TO_REST and LIFT_TO_PRESENT are the two routes where the
        full flown sequence and FOOTPRINT_LEGS's subset are the SAME arc
        (the analysis's check-order item 4) -- so report(full_route=True)
        must reproduce report(full_route=False) exactly for them, and the
        B2 LOWER_TO_REST bug above is the aperture bug alone, not a route-
        scope truncation too."""
        scene = SceneModel.from_yaml(_B2_SCENE_PATH)
        via_footprint = mrc.planned_clearance("LOWER_TO_REST", 50, scene)
        via_full_route = mrc.planned_clearance(
            "LOWER_TO_REST", 50, scene,
            waypoints=mrc.full_route_poses("LOWER_TO_REST"))
        assert via_full_route == via_footprint

    def test_report_full_route_flag_selects_full_route_poses(self, mrc):
        """report(full_route=True) must actually thread through to
        `planned_clearance`'s `waypoints` argument, not just accept and
        ignore the flag -- and must say so in `planned_route_scope`."""
        log = _synthetic_log(R.HOME, n=2)
        footprint_result = mrc.report(log, "PLACE_ROUTE", _B1_SCENE_PATH)
        full_route_result = mrc.report(
            log, "PLACE_ROUTE", _B1_SCENE_PATH, full_route=True)
        assert footprint_result["planned"] != full_route_result["planned"]
        assert "FOOTPRINT_LEGS" in footprint_result["planned_route_scope"]
        assert "full_route_poses" in full_route_result["planned_route_scope"]

    def test_planned_aperture_policy_documents_the_split(self, mrc):
        """The policy string itself must say the two hand models disagree,
        so a reader of a `report()` result -- not just this test suite --
        can see why shells and tube apertures are computed differently."""
        assert "tube" in mrc.PLANNED_APERTURE_POLICY
        assert "shells" in mrc.PLANNED_APERTURE_POLICY
        assert "ENDPOINT" in mrc.PLANNED_APERTURE_POLICY
        assert "interpolat" in mrc.PLANNED_APERTURE_POLICY


# ── Issue #137 follow-up (mutation gaps found reviewing PR #136, 59f471a) ──
#
# Review found two of the existing regression tests above pass under
# mutations that would reintroduce the original bugs:
#
#   (a) holding shells' aperture at each leg's START value (instead of
#       interpolating) still passes every test, because the real B2
#       LOWER_TO_REST worst finger sample (3.37 cm) sits at REST_SHUT,
#       which is itself a LEG BOUNDARY -- the leg that starts there already
#       evaluates the SHUT aperture at its own first sample. A mutation
#       catcher needs the worst sample strictly INSIDE a leg, where
#       neither a start-held nor an endpoint-held aperture ever looks.
#   (b) switching tube to the same per-sample interpolation shells uses
#       also passes every test, because on the real routes/scenes tube's
#       own worst sample already sits at the leg's worst-hand_radius
#       endpoint (mathematically guaranteed there, by hand_radius's own
#       monotonicity -- see `planned_clearance`'s docstring) -- so nothing
#       here happens to probe an INTERIOR arm position where the two
#       policies would actually disagree.
#
# Both classes below build a synthetic two-pose "leg" and a small tracked
# probe object placed at that leg's own true-worst sample -- computed once,
# offline, from the real geometry (`link_capsules`), independent of
# `planned_clearance`/`planned_clearance_by_link` -- then pin three
# independently-computed numbers against each other (true interpolation,
# hold-at-start, worst-endpoint) before checking which one the function
# under test actually reproduces. Verified 2026-09-22 by applying
# mutations of `_shells_leg_samples`/`_tube_leg_samples` matching (a)/(b)
# above IN PLACE to scripts/measure_route_clearance.py (the fixed file
# backed up to a scratch copy and restored from it before commit): both
# mutations move the reported clearance by centimeters, not
# floating-point noise (outputs/handoff-2026-09-22-issue-137-tests.md).


class TestShellsAperturePolicyCatchesAnInteriorMinimum:
    """Priority 1, finding (a): a synthetic leg whose worst `finger` sample
    is strictly inside it, not at either endpoint."""

    #: A fixed REST arm pose; the "leg" is degenerate in ARM position (both
    #: ends are the same q7) and spans the gripper's FULL commanded range --
    #: isolating the aperture policy from any arm-position effect, so the
    #: only thing that can move the worst sample around is the aperture
    #: itself.
    _POSE_A = {**{j: R.REST[j] for j in R.ARM7},
              "r_gripper": _GRIPPER_SHUT_LIMIT_DEG}
    _POSE_B = {**{j: R.REST[j] for j in R.ARM7},
              "r_gripper": _GRIPPER_OPEN_LIMIT_DEG}
    _STEPS = 200

    def _probe_scene(self):
        """B2's own objects, plus one small synthetic tracked sphere placed
        at the `finger` capsule's own aperture-sweep position roughly
        opposite the sweep's centroid -- the interior sample the sweep
        actually passes closest to, found once by direct inspection of
        `link_capsules`, not by calling anything under test."""
        base = SceneModel.from_yaml(_B2_SCENE_PATH)
        q7 = [R.REST[j] for j in R.ARM7]
        n = self._STEPS
        p0s = []
        for i in range(n):
            t = i / (n - 1)
            g = (self._POSE_A["r_gripper"]
                + t * (self._POSE_B["r_gripper"] - self._POSE_A["r_gripper"]))
            (_name, p0, _p1, _rad) = [
                c for c in link_capsules(q7, "right", g, hand="shells")
                if c[0] == "finger"][0]
            p0s.append(p0)
        cx = sum(p[0] for p in p0s) / n
        cy = sum(p[1] for p in p0s) / n
        cz = sum(p[2] for p in p0s) / n
        mx, my, mz = p0s[n // 2]
        dx, dy, dz = mx - cx, my - cy, mz - cz
        dn = math.sqrt(dx * dx + dy * dy + dz * dz)
        offset = 0.02
        center = (mx + offset * dx / dn, my + offset * dy / dn,
                  mz + offset * dz / dn)
        probe = SceneObject(id="probe_finger", kind="sphere", center=center,
                            size=(0.006, 0.006, 0.006), dynamic=True,
                            tracked=True)
        return SceneModel(base.frame_id, list(base.objects.values()) + [probe],
                          table_id=None)

    def _worst_at_constant_gripper(self, scene, path, g_const):
        best = None
        for q in path:
            caps = [c for c in link_capsules(q, "right", g_const, hand="shells")
                    if c[0] == "finger"]
            d = scene.clearances(caps)["probe_finger"].distance
            if best is None or d < best:
                best = d
        return best

    def _worst_interpolated(self, scene, path):
        steps = len(path)
        best = None
        for i, q in enumerate(path):
            t = i / (steps - 1)
            g = (self._POSE_A["r_gripper"]
                + t * (self._POSE_B["r_gripper"] - self._POSE_A["r_gripper"]))
            caps = [c for c in link_capsules(q, "right", g, hand="shells")
                    if c[0] == "finger"]
            d = scene.clearances(caps)["probe_finger"].distance
            if best is None or d < best:
                best = d
        return best

    def test_worst_finger_sample_is_strictly_interior_to_the_leg(self, mrc):
        """Ground truth, independent of the function under test: a dense
        per-sample scan finds its minimum strictly between the two
        endpoints -- the shape neither a start-held nor an endpoint-held
        aperture can ever reproduce."""
        scene = self._probe_scene()
        qa = [self._POSE_A[j] for j in R.ARM7]
        qb = [self._POSE_B[j] for j in R.ARM7]
        path = joint_path(qa, qb, steps=self._STEPS)
        best, best_i = None, None
        for i, q in enumerate(path):
            t = i / (self._STEPS - 1)
            g = (self._POSE_A["r_gripper"]
                + t * (self._POSE_B["r_gripper"] - self._POSE_A["r_gripper"]))
            caps = [c for c in link_capsules(q, "right", g, hand="shells")
                    if c[0] == "finger"]
            d = scene.clearances(caps)["probe_finger"].distance
            if best is None or d < best:
                best, best_i = d, i
        assert 0 < best_i < self._STEPS - 1, (
            "the probe must sit where the aperture sweep's true worst "
            "sample is strictly inside the leg, not at either endpoint -- "
            f"got index {best_i} of {self._STEPS - 1}")

    def test_planned_clearance_by_link_matches_interpolation_not_the_wrong_rules(
            self, mrc):
        """The actual assertion issue #137 asked for: `planned_clearance_by_link`
        must match the true (dense, independently-computed) interpolated
        worst case -- and that true worst must be strictly TIGHTER (a
        smaller, more conservative number) than what either the
        hold-at-start mutation or the old tube-style endpoint rule would
        have reported here. Either wrong rule reporting a much LOOSER
        clearance is exactly how a re-introduced version of the original
        bug would look."""
        scene = self._probe_scene()
        qa = [self._POSE_A[j] for j in R.ARM7]
        qb = [self._POSE_B[j] for j in R.ARM7]
        path = joint_path(qa, qb, steps=self._STEPS)

        true_worst = self._worst_interpolated(scene, path)
        hold_at_start = self._worst_at_constant_gripper(
            scene, path, self._POSE_A["r_gripper"])
        old_endpoint_rule = self._worst_at_constant_gripper(
            scene, path,
            max(self._POSE_A["r_gripper"], self._POSE_B["r_gripper"],
               key=hand_radius))

        actual = mrc.planned_clearance_by_link(
            "LOWER_TO_REST", self._STEPS, scene, "finger", hand="shells",
            waypoints=(self._POSE_A, self._POSE_B))

        assert actual["probe_finger"] == pytest.approx(true_worst)
        assert true_worst < hold_at_start - 0.01, (
            "holding aperture at the leg's start must miss this leg's "
            "true worst sample by more than a centimetre")
        assert true_worst < old_endpoint_rule - 0.01, (
            "the old tube-style worst-endpoint rule must miss this leg's "
            "true worst sample by more than a centimetre")


class TestTubeApertureStaysPinnedToTheEndpointRule:
    """Priority 1, finding (b): a synthetic leg where the arm actually
    moves, with a probe placed at the interior arm position the sweep
    passes closest to -- independent of aperture, since tube's capsule
    POSITION never depends on gripper_deg (only its radius does; see
    `link_capsules`'s tube branch). Pins tube to the worst-endpoint rule
    and shows per-sample interpolation would report a much looser number
    here."""

    _POSE_A = {**{j: R.PRESENT[j] for j in R.ARM7},
              "r_gripper": _GRIPPER_OPEN_LIMIT_DEG}
    _POSE_B = {**{j: R.REST[j] for j in R.ARM7},
              "r_gripper": _GRIPPER_SHUT_LIMIT_DEG}
    _STEPS = 41

    def _probe_scene(self):
        qa = [self._POSE_A[j] for j in R.ARM7]
        qb = [self._POSE_B[j] for j in R.ARM7]
        path = joint_path(qa, qb, steps=self._STEPS)
        mids = []
        for q in path:
            (_name, p0, p1, _rad) = [
                c for c in link_capsules(q, "right", 0.0, hand="tube")
                if c[0] == "hand"][0]
            mids.append(tuple((a + b) / 2 for a, b in zip(p0, p1)))
        n = len(mids)
        cx = sum(m[0] for m in mids) / n
        cy = sum(m[1] for m in mids) / n
        cz = sum(m[2] for m in mids) / n
        mx, my, mz = mids[n // 2]
        dx, dy, dz = mx - cx, my - cy, mz - cz
        dn = math.sqrt(dx * dx + dy * dy + dz * dz)
        offset = 0.20
        center = (mx + offset * dx / dn, my + offset * dy / dn,
                  mz + offset * dz / dn)
        probe = SceneObject(id="probe_tube", kind="sphere", center=center,
                            size=(0.006, 0.006, 0.006), dynamic=True,
                            tracked=True)
        return SceneModel("pedestal", [probe], table_id=None)

    def _worst_at_constant_gripper(self, scene, path, g_const):
        best = None
        for q in path:
            caps = [c for c in link_capsules(q, "right", g_const, hand="tube")
                    if c[0] == "hand"]
            d = scene.clearances(caps)["probe_tube"].distance
            if best is None or d < best:
                best = d
        return best

    def _worst_interpolated(self, scene, path):
        steps = len(path)
        best = None
        for i, q in enumerate(path):
            t = i / (steps - 1)
            g = (self._POSE_A["r_gripper"]
                + t * (self._POSE_B["r_gripper"] - self._POSE_A["r_gripper"]))
            caps = [c for c in link_capsules(q, "right", g, hand="tube")
                    if c[0] == "hand"]
            d = scene.clearances(caps)["probe_tube"].distance
            if best is None or d < best:
                best = d
        return best

    def test_geometric_closest_sample_is_strictly_interior(self, mrc):
        """Ground truth, independent of the function under test: the
        segment's own closest approach to the probe (aperture aside, since
        tube's position never depends on it) sits strictly inside the leg
        -- so a rule that only ever looks at the two endpoints is checking
        the wrong place unless it happens to also be conservative there."""
        scene = self._probe_scene()
        qa = [self._POSE_A[j] for j in R.ARM7]
        qb = [self._POSE_B[j] for j in R.ARM7]
        path = joint_path(qa, qb, steps=self._STEPS)
        best, best_i = None, None
        for i, q in enumerate(path):
            caps = [c for c in link_capsules(q, "right", 0.0, hand="tube")
                    if c[0] == "hand"]
            d = scene.clearances(caps)["probe_tube"].distance
            if best is None or d < best:
                best, best_i = d, i
        assert 0 < best_i < self._STEPS - 1

    def test_planned_clearance_matches_the_endpoint_rule_not_interpolation(
            self, mrc):
        scene = self._probe_scene()
        qa = [self._POSE_A[j] for j in R.ARM7]
        qb = [self._POSE_B[j] for j in R.ARM7]
        path = joint_path(qa, qb, steps=self._STEPS)

        a_r = hand_radius(self._POSE_A["r_gripper"], self._POSE_A["r_wrist_roll"])
        b_r = hand_radius(self._POSE_B["r_gripper"], self._POSE_B["r_wrist_roll"])
        worst_gripper = (self._POSE_A["r_gripper"] if a_r >= b_r
                         else self._POSE_B["r_gripper"])

        endpoint_worst = self._worst_at_constant_gripper(scene, path, worst_gripper)
        interpolated_worst = self._worst_interpolated(scene, path)

        actual = mrc.planned_clearance(
            "LOWER_TO_REST", self._STEPS, scene,
            waypoints=(self._POSE_A, self._POSE_B))

        assert actual["tube"]["probe_tube"] == pytest.approx(endpoint_worst)
        # Per-sample interpolation would report a much LOOSER (larger)
        # clearance here -- proof a switch to it would be a real
        # regression against this leg, not a no-op the way it is on every
        # route/scene this repo already tests against.
        assert interpolated_worst > endpoint_worst + 0.02, (
            "interpolation must diverge from the endpoint rule by more "
            "than 2 cm on this leg, or the probe isn't actually interior")


class TestPlannedClearanceByLinkRejectsUnknownLinks:
    """Priority 2, finding 1: `planned_clearance_by_link` used to return
    `{}` silently for a link name no capsule of `hand` ever produces (e.g.
    a typo like `"fingers"`) -- indistinguishable from a route with no
    `FOOTPRINT_LEGS` entry. It must raise instead."""

    def test_unknown_link_name_raises_valueerror(self, mrc):
        scene = SceneModel.from_yaml(_B2_SCENE_PATH)
        with pytest.raises(ValueError, match="fingers"):
            mrc.planned_clearance_by_link(
                "LOWER_TO_REST", 13, scene, "fingers", hand="shells")

    def test_shells_only_link_raises_under_hand_tube(self, mrc):
        """`"finger"` is a real capsule name -- just never one
        `link_capsules(hand="tube")` produces (tube has exactly one
        capsule, `"hand"`). Must raise the same way a nonsense name does,
        not silently return `{}`."""
        scene = SceneModel.from_yaml(_B2_SCENE_PATH)
        with pytest.raises(ValueError, match="finger"):
            mrc.planned_clearance_by_link(
                "LOWER_TO_REST", 13, scene, "finger", hand="tube")

    def test_invalid_hand_raises_valueerror(self, mrc):
        scene = SceneModel.from_yaml(_B2_SCENE_PATH)
        with pytest.raises(ValueError, match="hand"):
            mrc.planned_clearance_by_link(
                "LOWER_TO_REST", 13, scene, "hand", hand="tubes")

    def test_unknown_link_raises_even_when_the_route_has_no_footprint_entry(
            self, mrc):
        """The check is about the caller's `link` argument, not about
        whether there happen to be any samples to check it against -- it
        must fire before the empty-waypoints early return, not be masked
        by it."""
        scene = SceneModel.from_yaml(_B2_SCENE_PATH)
        with pytest.raises(ValueError):
            mrc.planned_clearance_by_link("WAVE", 13, scene, "fingers")

    def test_every_real_capsule_name_is_accepted_for_its_own_hand(self, mrc):
        """The other direction of the same fix: a real capsule name must
        never be rejected -- this is a targeted refusal, not a stricter
        allowlist that happens to break real callers."""
        scene = SceneModel.from_yaml(_B2_SCENE_PATH)
        for hand, names in mrc._CAPSULE_NAMES_BY_HAND.items():
            for link in names:
                mrc.planned_clearance_by_link(
                    "LOWER_TO_REST", 5, scene, link, hand=hand)  # must not raise

    def test_capsule_names_by_hand_matches_link_capsules_reality(self, mrc):
        """The allowlist itself must not drift from what `link_capsules`
        actually produces -- checked directly against a real pose on both
        hand models, rather than trusted as a hand-maintained constant."""
        q7 = [R.REST[j] for j in R.ARM7]
        for hand in mrc.HAND_MODES:
            names = {c[0] for c in link_capsules(q7, "right", -30.0, hand=hand)}
            assert names == mrc._CAPSULE_NAMES_BY_HAND[hand]
