"""E1 readiness (assignment 2026-09-14, work item 2): scripts/e1_identity.py.

Offline throughout: every fake run directory is written with the REAL
`native_mujoco.recorder.Recorder` (so the on-disk layout this module reads
can't drift from what the native server actually writes), `read_sdk_joints`
/ `http_get` / `now_ns` / `wall_clock_ns` / `sleep` are all injected, and no
socket, server, or SDK connection is ever opened.
"""

import json
import math
import os
import signal
import sys
import types

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../scripts"))
sys.path.insert(0, os.path.join(_HERE, "../../src"))
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))

import e1_identity as ei  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402
from recorder import Recorder  # noqa: E402

_BOARD_SCENE = os.path.join(_HERE, "../../scenes/e1_boards/B4_pool_box_1_r2c3.yaml")

#: A parked pose (any ARM7 pose plus a gripper reading) to build synthetic
#: joint states from -- values are irrelevant to the checks except that SDK
#: and server agree (or deliberately do not).
_POSE_DEG = dict(R.REST)


def _rad_joints(pose_deg, offsets_deg=None, compliant_by_name=None, effort_by_name=None):
    """`compliant_by_name`/`effort_by_name` are None by default, which keeps
    every existing caller's state dict byte-identical (issue #116's
    TestRequireCompliance is the only thing that passes them)."""
    offsets_deg = offsets_deg or {}
    out = []
    for name in R.R_JOINTS:
        j = {"name": name,
             "position_rad": math.radians(pose_deg[name] + offsets_deg.get(name, 0.0))}
        if compliant_by_name is not None:
            j["compliant"] = compliant_by_name.get(name, False)
        if effort_by_name is not None:
            j["effort"] = effort_by_name.get(name, 0.0)
        out.append(j)
    return out


def _sdk_joints(pose_deg, offsets_deg=None):
    offsets_deg = offsets_deg or {}
    return {name: pose_deg[name] + offsets_deg.get(name, 0.0) for name in R.R_JOINTS}


def _write_run_dir(tmp_path, *, manifest_meta, wall_time_ns, sim_step,
                   pose_deg=None, server_offsets_deg=None,
                   compliant_by_name=None, effort_by_name=None, commands=None):
    """One live run directory, written with the real Recorder.

    `commands` (issue #116) are recorded verbatim to commands.jsonl before
    the state, matching the real server's order (submit_command ->
    recorder.record_command happens before the tick that later pushes a
    state) -- for TestRequireCompliance's "sent but not applied" reporting.
    """
    pose_deg = pose_deg or _POSE_DEG
    rec = Recorder.new(tmp_path, manifest_meta)
    for c in (commands or []):
        rec.record_command(c)
    rec.record_state({
        "type": "state", "seq": 1, "sim_step": sim_step, "sim_time_s": 0.1,
        "wall_time_ns": wall_time_ns, "scene_revision": "r1",
        "joints": _rad_joints(pose_deg, server_offsets_deg,
                               compliant_by_name, effort_by_name),
        "objects": [], "grippers": [],
    })
    rec.finalize(total_steps=sim_step, duration_s=0.1)
    return rec.run_dir


def _manifest_for(scene_path, *, contacts_tracked=True):
    scene_path = os.path.abspath(scene_path)
    manifest = {
        "model_path": "model/reachy_1_2.xml",
        "model_sha256": "deadbeef",
        "scene_path": scene_path,
        "scene_sha256": ei._sha256_file(__import__("pathlib").Path(scene_path)),
        "scene_revision": "r1",
    }
    # Priority 1 (2026-09-15 matrix readiness): every "good" manifest fixture
    # in this file defaults to tracking on, matching a real server started
    # with --record; contacts_tracked=False (or omitted entirely) is
    # exercised explicitly by TestContactsTrackedManifestGate below.
    if contacts_tracked is not None:
        manifest["contacts_tracked"] = contacts_tracked
    return manifest


def _good_http_get(url):
    return {"backend": "mujoco-remote", "frames_stale": False}


def _never_call(*_a, **_kw):
    raise AssertionError("should not have been called")


class TestLoopbackOnly:
    def test_non_loopback_host_fails_and_reads_nothing_else(self, tmp_path):
        result = ei.verify_simulator_identity(
            host="192.168.1.50", port=50051,
            scene_path=_BOARD_SCENE, record_root=str(tmp_path),
            read_sdk_joints=_never_call, http_get=_never_call,
            now_ns=_never_call, wall_clock_ns=lambda: 123,
        )
        assert result.ok is False
        assert len(result.reasons) == 1
        assert "physical robot host is refused" in result.reasons[0]
        assert result.scene_chain_sha256 == {}

    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1"])
    def test_loopback_hosts_pass_the_host_check(self, tmp_path, host):
        wall_time_ns = 10_000_000_000
        run_dir = _write_run_dir(
            tmp_path, manifest_meta=_manifest_for(_BOARD_SCENE),
            wall_time_ns=wall_time_ns, sim_step=100)
        result = ei.verify_simulator_identity(
            host=host, port=50051, scene_path=_BOARD_SCENE,
            record_root=str(tmp_path),
            read_sdk_joints=lambda: _sdk_joints(_POSE_DEG),
            http_get=_good_http_get,
            now_ns=lambda: wall_time_ns, wall_clock_ns=lambda: 0,
            sleep=lambda s: None, min_reads=1,
        )
        assert result.ok is True, result.reasons
        assert result.run_dir == str(run_dir)


class TestLiveRunDirectory:
    def test_no_run_directory_gives_a_distinct_reason(self, tmp_path):
        result = ei.verify_simulator_identity(
            host="localhost", port=50051, scene_path=_BOARD_SCENE,
            record_root=str(tmp_path / "does_not_exist"),
            read_sdk_joints=_never_call, http_get=_good_http_get,
            now_ns=lambda: 0, wall_clock_ns=lambda: 0, min_reads=1,
        )
        assert result.ok is False
        assert any("no live physics run directory" in r for r in result.reasons)

    def test_missing_manifest_gives_a_distinct_reason(self, tmp_path):
        run_dir = tmp_path / "run_20260914_000000"
        run_dir.mkdir()
        (run_dir / "states.jsonl").write_text(
            json.dumps({"sim_step": 1, "wall_time_ns": 5_000,
                       "joints": _rad_joints(_POSE_DEG)}) + "\n")
        result = ei.verify_simulator_identity(
            host="localhost", port=50051, scene_path=_BOARD_SCENE,
            record_root=str(tmp_path),
            read_sdk_joints=lambda: _sdk_joints(_POSE_DEG),
            http_get=_good_http_get,
            now_ns=lambda: 5_000, wall_clock_ns=lambda: 0, min_reads=1,
        )
        assert result.ok is False
        assert any("manifest.json is missing" in r for r in result.reasons)

    def test_stale_wall_time_ns_gives_a_distinct_reason(self, tmp_path):
        wall_time_ns = 10_000_000_000
        _write_run_dir(
            tmp_path, manifest_meta=_manifest_for(_BOARD_SCENE),
            wall_time_ns=wall_time_ns, sim_step=100)
        # "now" is 2 seconds after the sample -- older than max_round_age_s.
        result = ei.verify_simulator_identity(
            host="localhost", port=50051, scene_path=_BOARD_SCENE,
            record_root=str(tmp_path),
            read_sdk_joints=lambda: _sdk_joints(_POSE_DEG),
            http_get=_good_http_get,
            now_ns=lambda: wall_time_ns + 2_000_000_000,
            wall_clock_ns=lambda: 0, min_reads=1,
        )
        assert result.ok is False
        assert any("is stale" in r or "old" in r for r in result.reasons)

    def test_sim_step_not_advancing_gives_a_distinct_reason(self, tmp_path):
        wall_time_ns = 10_000_000_000
        _write_run_dir(
            tmp_path, manifest_meta=_manifest_for(_BOARD_SCENE),
            wall_time_ns=wall_time_ns, sim_step=100)
        result = ei.verify_simulator_identity(
            host="localhost", port=50051, scene_path=_BOARD_SCENE,
            record_root=str(tmp_path),
            read_sdk_joints=lambda: _sdk_joints(_POSE_DEG),
            http_get=_good_http_get,
            now_ns=lambda: wall_time_ns, wall_clock_ns=lambda: 0,
            sleep=lambda s: None, min_reads=2, min_read_interval_s=0.0,
        )
        assert result.ok is False
        assert any("sim_step did not advance" in r for r in result.reasons)

    def test_two_live_run_directories_are_ambiguous(self, tmp_path):
        wall_time_ns = 10_000_000_000
        # Explicit distinct run_dir paths -- Recorder.new()'s own naming is
        # second-resolution (run_<YYYYMMDD_HHMMSS>), so two calls within the
        # same wall-clock second would silently collide on one directory.
        for name in ("run_A", "run_B"):
            rec = Recorder(tmp_path / name, _manifest_for(_BOARD_SCENE))
            rec.record_state({
                "type": "state", "seq": 1, "sim_step": 100,
                "sim_time_s": 0.1, "wall_time_ns": wall_time_ns,
                "scene_revision": "r1", "joints": _rad_joints(_POSE_DEG),
                "objects": [], "grippers": [],
            })
            rec.finalize(total_steps=100, duration_s=0.1)
        result = ei.verify_simulator_identity(
            host="localhost", port=50051, scene_path=_BOARD_SCENE,
            record_root=str(tmp_path),
            read_sdk_joints=lambda: _sdk_joints(_POSE_DEG),
            http_get=_good_http_get,
            now_ns=lambda: wall_time_ns, wall_clock_ns=lambda: 0, min_reads=1,
        )
        assert result.ok is False
        assert any("all look live at once" in r for r in result.reasons)


class TestSceneIdentity:
    def test_scene_sha_mismatch_names_both_shas(self, tmp_path):
        board_copy = tmp_path / "board.yaml"
        board_copy.write_text(open(_BOARD_SCENE).read() + "\n# tampered\n")
        wall_time_ns = 10_000_000_000
        manifest_meta = _manifest_for(_BOARD_SCENE)  # sha of the ORIGINAL file
        _write_run_dir(tmp_path, manifest_meta=manifest_meta,
                       wall_time_ns=wall_time_ns, sim_step=100)
        result = ei.verify_simulator_identity(
            host="localhost", port=50051, scene_path=str(board_copy),
            record_root=str(tmp_path),
            read_sdk_joints=lambda: _sdk_joints(_POSE_DEG),
            http_get=_good_http_get,
            now_ns=lambda: wall_time_ns, wall_clock_ns=lambda: 0,
            sleep=lambda s: None, min_reads=1,
        )
        assert result.ok is False
        mismatch = [r for r in result.reasons if "scene sha mismatch" in r]
        assert len(mismatch) == 1
        assert manifest_meta["scene_sha256"] in mismatch[0]
        assert result.scene_chain_sha256[str(board_copy.resolve())] in mismatch[0]


class TestBridgeBackend:
    def test_status_unreachable_is_a_distinct_reason(self, tmp_path):
        wall_time_ns = 10_000_000_000
        _write_run_dir(tmp_path, manifest_meta=_manifest_for(_BOARD_SCENE),
                       wall_time_ns=wall_time_ns, sim_step=100)

        def _boom(url):
            raise OSError("connection refused")

        result = ei.verify_simulator_identity(
            host="localhost", port=50051, scene_path=_BOARD_SCENE,
            record_root=str(tmp_path),
            read_sdk_joints=lambda: _sdk_joints(_POSE_DEG), http_get=_boom,
            now_ns=lambda: wall_time_ns, wall_clock_ns=lambda: 0,
            sleep=lambda s: None, min_reads=1,
        )
        assert result.ok is False
        assert any("unreachable" in r for r in result.reasons)

    def test_wrong_backend_label_is_a_distinct_reason(self, tmp_path):
        wall_time_ns = 10_000_000_000
        _write_run_dir(tmp_path, manifest_meta=_manifest_for(_BOARD_SCENE),
                       wall_time_ns=wall_time_ns, sim_step=100)
        result = ei.verify_simulator_identity(
            host="localhost", port=50051, scene_path=_BOARD_SCENE,
            record_root=str(tmp_path),
            read_sdk_joints=lambda: _sdk_joints(_POSE_DEG),
            http_get=lambda url: {"backend": "fixture", "frames_stale": False},
            now_ns=lambda: wall_time_ns, wall_clock_ns=lambda: 0,
            sleep=lambda s: None, min_reads=1,
        )
        assert result.ok is False
        assert any("backend='fixture'" in r for r in result.reasons)

    def test_stale_frames_is_a_distinct_reason(self, tmp_path):
        wall_time_ns = 10_000_000_000
        _write_run_dir(tmp_path, manifest_meta=_manifest_for(_BOARD_SCENE),
                       wall_time_ns=wall_time_ns, sim_step=100)
        result = ei.verify_simulator_identity(
            host="localhost", port=50051, scene_path=_BOARD_SCENE,
            record_root=str(tmp_path),
            read_sdk_joints=lambda: _sdk_joints(_POSE_DEG),
            http_get=lambda url: {"backend": "mujoco-remote", "frames_stale": True},
            now_ns=lambda: wall_time_ns, wall_clock_ns=lambda: 0,
            sleep=lambda s: None, min_reads=1,
        )
        assert result.ok is False
        assert any("frames_stale=True" in r for r in result.reasons)


class TestJointAgreement:
    def test_one_joint_1p5_deg_off_names_the_joint(self, tmp_path):
        wall_time_ns = 10_000_000_000
        _write_run_dir(tmp_path, manifest_meta=_manifest_for(_BOARD_SCENE),
                       wall_time_ns=wall_time_ns, sim_step=100,
                       pose_deg=_POSE_DEG)
        result = ei.verify_simulator_identity(
            host="localhost", port=50051, scene_path=_BOARD_SCENE,
            record_root=str(tmp_path),
            read_sdk_joints=lambda: _sdk_joints(
                _POSE_DEG, {"r_wrist_roll": 1.5}),
            http_get=_good_http_get,
            now_ns=lambda: wall_time_ns, wall_clock_ns=lambda: 0,
            sleep=lambda s: None, min_reads=1,
        )
        assert result.ok is False
        disagreement = [r for r in result.reasons if "joint disagreement" in r]
        assert len(disagreement) == 1
        assert "r_wrist_roll" in disagreement[0]

    def test_half_a_degree_off_passes(self, tmp_path):
        wall_time_ns = 10_000_000_000
        _write_run_dir(tmp_path, manifest_meta=_manifest_for(_BOARD_SCENE),
                       wall_time_ns=wall_time_ns, sim_step=100,
                       pose_deg=_POSE_DEG)
        result = ei.verify_simulator_identity(
            host="localhost", port=50051, scene_path=_BOARD_SCENE,
            record_root=str(tmp_path),
            read_sdk_joints=lambda: _sdk_joints(
                _POSE_DEG, {"r_wrist_roll": 0.5}),
            http_get=_good_http_get,
            now_ns=lambda: wall_time_ns, wall_clock_ns=lambda: 0,
            sleep=lambda s: None, min_reads=1,
        )
        assert result.ok is True, result.reasons


class TestContactsTrackedManifestGate:
    """Priority 1 (2026-09-15 matrix readiness, the matrix gate): a manifest
    whose `contacts_tracked` is missing, `False`, or any non-`True` value
    refuses -- `native_mujoco/server.py` sets this from `_record_contacts`,
    and a matrix is many server restarts; a manifest that doesn't say
    tracking was on must not let a flight through."""

    def test_missing_contacts_tracked_refuses(self, tmp_path):
        manifest = _manifest_for(_BOARD_SCENE, contacts_tracked=None)
        assert "contacts_tracked" not in manifest
        wall_time_ns = 10_000_000_000
        _write_run_dir(
            tmp_path, manifest_meta=manifest,
            wall_time_ns=wall_time_ns, sim_step=100)
        result = ei.verify_simulator_identity(
            host="localhost", port=50051, scene_path=_BOARD_SCENE,
            record_root=str(tmp_path),
            read_sdk_joints=lambda: _sdk_joints(_POSE_DEG),
            http_get=_good_http_get,
            now_ns=lambda: wall_time_ns, wall_clock_ns=lambda: 0,
            sleep=lambda s: None, min_reads=1,
        )
        assert result.ok is False
        assert any("manifest.contacts_tracked is None" in r
                  for r in result.reasons)

    def test_false_contacts_tracked_refuses(self, tmp_path):
        manifest = _manifest_for(_BOARD_SCENE, contacts_tracked=False)
        wall_time_ns = 10_000_000_000
        _write_run_dir(
            tmp_path, manifest_meta=manifest,
            wall_time_ns=wall_time_ns, sim_step=100)
        result = ei.verify_simulator_identity(
            host="localhost", port=50051, scene_path=_BOARD_SCENE,
            record_root=str(tmp_path),
            read_sdk_joints=lambda: _sdk_joints(_POSE_DEG),
            http_get=_good_http_get,
            now_ns=lambda: wall_time_ns, wall_clock_ns=lambda: 0,
            sleep=lambda s: None, min_reads=1,
        )
        assert result.ok is False
        assert any("manifest.contacts_tracked is False" in r
                  for r in result.reasons)

    def test_true_contacts_tracked_is_unchanged_pass(self, tmp_path):
        manifest = _manifest_for(_BOARD_SCENE, contacts_tracked=True)
        wall_time_ns = 10_000_000_000
        _write_run_dir(
            tmp_path, manifest_meta=manifest,
            wall_time_ns=wall_time_ns, sim_step=100)
        result = ei.verify_simulator_identity(
            host="localhost", port=50051, scene_path=_BOARD_SCENE,
            record_root=str(tmp_path),
            read_sdk_joints=lambda: _sdk_joints(_POSE_DEG),
            http_get=_good_http_get,
            now_ns=lambda: wall_time_ns, wall_clock_ns=lambda: 0,
            sleep=lambda s: None, min_reads=1,
        )
        assert result.ok is True, result.reasons


class TestContainerCheck:
    """N8/A2 (2026-09-15 matrix readiness): run via `docker compose exec`,
    the recorder shares no clock with the host native server it's
    checking. Refuse rather than let the operator "fix" an otherwise
    fail-closed refusal by pointing --record-root somewhere wrong."""

    def test_dockerenv_present_refuses_even_with_everything_else_good(
            self, tmp_path):
        wall_time_ns = 10_000_000_000
        _write_run_dir(
            tmp_path, manifest_meta=_manifest_for(_BOARD_SCENE),
            wall_time_ns=wall_time_ns, sim_step=100)
        result = ei.verify_simulator_identity(
            host="localhost", port=50051, scene_path=_BOARD_SCENE,
            record_root=str(tmp_path),
            read_sdk_joints=lambda: _sdk_joints(_POSE_DEG),
            http_get=_good_http_get,
            now_ns=lambda: wall_time_ns, wall_clock_ns=lambda: 0,
            sleep=lambda s: None, min_reads=1,
            in_container=lambda: True,
        )
        assert result.ok is False
        assert any("running inside a container" in r for r in result.reasons)

    def test_dockerenv_absent_does_not_add_a_reason(self, tmp_path):
        wall_time_ns = 10_000_000_000
        _write_run_dir(
            tmp_path, manifest_meta=_manifest_for(_BOARD_SCENE),
            wall_time_ns=wall_time_ns, sim_step=100)
        result = ei.verify_simulator_identity(
            host="localhost", port=50051, scene_path=_BOARD_SCENE,
            record_root=str(tmp_path),
            read_sdk_joints=lambda: _sdk_joints(_POSE_DEG),
            http_get=_good_http_get,
            now_ns=lambda: wall_time_ns, wall_clock_ns=lambda: 0,
            sleep=lambda s: None, min_reads=1,
            in_container=lambda: False,
        )
        assert result.ok is True, result.reasons


class TestAllGood:
    def test_ok_true_and_scene_chain_has_board_and_parents(self, tmp_path):
        """3 rounds of GENUINE progress: the injected `sleep` doubles as a
        fake background writer, appending a fresher state (advanced
        sim_step and wall_time_ns) each time it is called between rounds --
        so this proves the interleaved, round-by-round design actually
        tracks a live-advancing stream, not just a single static
        snapshot re-read three times."""
        rec = Recorder.new(tmp_path, _manifest_for(_BOARD_SCENE))
        clock = {"wall_ns": 10_000_000_000, "sim_step": 100}

        def _write_state():
            clock["sim_step"] += 1
            clock["wall_ns"] += 20_000_000  # one 50 Hz state-push tick
            rec.record_state({
                "type": "state", "seq": clock["sim_step"],
                "sim_step": clock["sim_step"], "sim_time_s": 0.1,
                "wall_time_ns": clock["wall_ns"], "scene_revision": "r1",
                "joints": _rad_joints(_POSE_DEG), "objects": [], "grippers": [],
            })

        _write_state()  # seed the sample round 0 will read
        result = ei.verify_simulator_identity(
            host="localhost", port=50051, scene_path=_BOARD_SCENE,
            record_root=str(tmp_path),
            read_sdk_joints=lambda: _sdk_joints(_POSE_DEG),
            http_get=_good_http_get,
            now_ns=lambda: clock["wall_ns"], wall_clock_ns=lambda: 999,
            sleep=lambda s: _write_state(), min_reads=3, min_read_interval_s=0.0,
        )
        assert result.ok is True, result.reasons
        assert result.run_dir == str(rec.run_dir)
        assert result.checked_at_wall_ns == 999
        names = set(result.scene_chain_sha256)
        assert any("e1_boards" in n and "B4_pool_box_1_r2c3.yaml" in n
                  for n in names)
        assert any("FWDCenterLabSivaPool.yaml" in n for n in names)
        assert any("FWDCenterLabSiva.yaml" in n for n in names)
        assert any("FWDCenterLabMCC.yaml" in n for n in names)
        assert len(result.joint_agreement_deg) == len(R.R_JOINTS)


class TestMainRefusesBeforeRecording:
    def test_main_exits_2_before_record_joint_log_when_identity_fails(
            self, monkeypatch, tmp_path, capsys):
        import measure_route_clearance as mrc

        monkeypatch.setenv("REACHY_SIM_RECORD_CLEARANCE", "1")
        monkeypatch.setattr(mrc, "RUNS_DIR", tmp_path)
        monkeypatch.setattr(
            mrc, "ReachySDK",
            lambda host, sdk_port: types.SimpleNamespace(r_arm=object()))

        def _boom(*_a, **_kw):
            raise AssertionError(
                "record_joint_log must not run when identity fails")

        monkeypatch.setattr(mrc, "record_joint_log", _boom)
        monkeypatch.setattr(
            mrc.e1_identity, "verify_simulator_identity",
            lambda **kw: ei.SimulatorIdentityCheck(
                ok=False, reasons=("synthetic refusal",), run_dir="",
                manifest={}, scene_chain_sha256={}, joint_agreement_deg={},
                status={}, checked_at_wall_ns=0))
        monkeypatch.setattr(
            sys, "argv",
            ["measure_route_clearance.py", "--route", "LOWER_TO_REST",
             "--duration", "0.05", "--record-root", str(tmp_path)])

        with pytest.raises(SystemExit) as exc:
            mrc.main()
        assert exc.value.code == 2
        out = capsys.readouterr().out
        assert "synthetic refusal" in out


class TestRequireCompliance:
    """Issue #116: the bounded pre-motion check. Reads the state stream in
    the run dir -- never the SDK client's cached `joint.compliant`, which is
    written by the client and read back only at connect (see
    `require_compliance`'s docstring) -- so it is judged on what the physics
    actually did (or, per the Stage 1 shape, did not do)."""

    @staticmethod
    def _ticking_clock(start_ns=10_000_000_000):
        now = {"ns": start_ns}

        def now_ns():
            return now["ns"]

        def sleep(seconds):
            now["ns"] += int(seconds * 1e9)

        return now, now_ns, sleep

    def test_all_stiff_fresh_state_passes_immediately(self):
        now, now_ns, _ = self._ticking_clock()
        state = {
            "seq": 5, "sim_step": 200, "wall_time_ns": now["ns"],
            "joints": [{"name": n, "compliant": False, "effort": 1.0}
                       for n in R.R_JOINTS],
        }
        result = ei.require_compliance(
            run_dir="/unused", joints=R.R_JOINTS, compliant=False,
            timeout_s=3.0, max_state_age_s=0.5,
            read_last_state=lambda p: state, now_ns=now_ns,
            sleep=lambda s: (_ for _ in ()).throw(
                AssertionError("must not sleep when already satisfied")),
        )
        assert result.ok is True, result.reasons
        assert result.waited_s == pytest.approx(0.0, abs=1e-9)
        assert result.per_joint["r_gripper"]["compliant"] is False

    def test_waits_and_polls_until_stiff_on_the_third_read(self):
        now, now_ns, sleep = self._ticking_clock()
        calls = {"n": 0}

        def fake_read(_path):
            calls["n"] += 1
            still_compliant = calls["n"] < 3
            return {
                "seq": calls["n"], "sim_step": 100 + calls["n"],
                "wall_time_ns": now["ns"],
                "joints": [{"name": n, "compliant": still_compliant, "effort": 0.0}
                           for n in R.R_JOINTS],
            }

        result = ei.require_compliance(
            run_dir="/unused", joints=R.R_JOINTS, compliant=False,
            timeout_s=3.0, max_state_age_s=0.5,
            read_last_state=fake_read, now_ns=now_ns, sleep=sleep,
        )
        assert result.ok is True, result.reasons
        assert calls["n"] == 3
        assert result.waited_s > 0

    def test_stage1_shape_elbow_stuck_compliant_past_timeout(self, tmp_path):
        """The reported #116 shape: seq 2 (compliant=False for the other
        seven right-arm joints) was dropped at reset, so r_elbow_pitch never
        went stiff -- effort 0.0 because the motor really is off."""
        compliant_by_name = {n: False for n in R.R_JOINTS}
        compliant_by_name["r_elbow_pitch"] = True
        effort_by_name = {n: 2.5 for n in R.R_JOINTS}
        effort_by_name["r_elbow_pitch"] = 0.0
        commands = [
            {"type": "joint_command", "seq": 1, "target_rad": [0.0] * 21,
             "compliant": [False if i == 0 else None for i in range(21)]},
            {"type": "joint_command", "seq": 2, "target_rad": [0.0] * 21,
             "compliant": [False if 1 <= i <= 7 else None for i in range(21)]},
        ]
        run_dir = _write_run_dir(
            tmp_path, manifest_meta=_manifest_for(_BOARD_SCENE),
            wall_time_ns=10_000_000_000, sim_step=100,
            compliant_by_name=compliant_by_name, effort_by_name=effort_by_name,
            commands=commands)
        _, now_ns, sleep = self._ticking_clock(10_000_000_000)
        result = ei.require_compliance(
            run_dir=str(run_dir), joints=R.R_JOINTS, compliant=False,
            timeout_s=0.01, max_state_age_s=1.0,
            now_ns=now_ns, sleep=sleep,
        )
        assert result.ok is False
        assert any("r_elbow_pitch" in r and "effort=0.0" in r
                  for r in result.reasons)
        assert any("seq 2" in r and "r_elbow_pitch=False" in r
                  for r in result.reasons)

    def test_stale_stream_never_passes_even_if_it_says_stiff(self, tmp_path):
        run_dir = _write_run_dir(
            tmp_path, manifest_meta=_manifest_for(_BOARD_SCENE),
            wall_time_ns=0, sim_step=100,
            compliant_by_name={n: False for n in R.R_JOINTS})
        _, now_ns, sleep = self._ticking_clock(10_000_000_000)  # 10s later
        result = ei.require_compliance(
            run_dir=str(run_dir), joints=R.R_JOINTS, compliant=False,
            timeout_s=0.01, max_state_age_s=0.5,
            now_ns=now_ns, sleep=sleep,
        )
        assert result.ok is False
        assert any("stale" in r for r in result.reasons)

    def test_expects_compliant_true_names_the_still_stiff_joint(self, tmp_path):
        compliant_by_name = {n: True for n in R.R_JOINTS}
        compliant_by_name["r_wrist_roll"] = False
        run_dir = _write_run_dir(
            tmp_path, manifest_meta=_manifest_for(_BOARD_SCENE),
            wall_time_ns=10_000_000_000, sim_step=100,
            compliant_by_name=compliant_by_name)
        _, now_ns, sleep = self._ticking_clock(10_000_000_000)
        result = ei.require_compliance(
            run_dir=str(run_dir), joints=R.R_JOINTS, compliant=True,
            timeout_s=0.01, max_state_age_s=1.0,
            now_ns=now_ns, sleep=sleep,
        )
        assert result.ok is False
        assert any("r_wrist_roll" in r for r in result.reasons)

    def test_injected_sleep_bounds_the_poll_count_no_busy_loop(self):
        sleeps = []
        now, now_ns, real_sleep = self._ticking_clock(0)

        def counted_sleep(s):
            sleeps.append(s)
            real_sleep(s)

        result = ei.require_compliance(
            run_dir="/nonexistent/run_dir", joints=["r_gripper"],
            compliant=False, timeout_s=1.0, max_state_age_s=0.5,
            read_last_state=lambda p: None,
            now_ns=now_ns, sleep=counted_sleep,
        )
        assert result.ok is False
        assert any("no readable" in r for r in result.reasons)
        # >= 20 Hz poll rate -> at most timeout_s / period polls, plus one.
        assert len(sleeps) <= round(1.0 / (1.0 / 20.0)) + 1

    # -- F2 (PR #118 review): min_cmd_seq ties evidence to the current call --

    def test_min_cmd_seq_rejects_a_stale_but_fresh_looking_sample(self):
        """A sample whose cmd_seq has not moved past the pre-call baseline
        must not pass, even though it is fresh and every joint already
        reports the wanted compliance -- it may be leftover from before the
        call this check is gating (e.g. compliance persisted through a
        reset)."""
        now, now_ns, sleep = self._ticking_clock()
        state = {
            "seq": 5, "sim_step": 200, "wall_time_ns": now["ns"], "cmd_seq": 3,
            "joints": [{"name": n, "compliant": False, "effort": 1.0}
                       for n in R.R_JOINTS],
        }
        result = ei.require_compliance(
            run_dir="/unused", joints=R.R_JOINTS, compliant=False,
            timeout_s=0.01, max_state_age_s=0.5, min_cmd_seq=3,
            read_last_state=lambda p: state, now_ns=now_ns,
            sleep=sleep,
        )
        assert result.ok is False
        assert any("has not advanced past the pre-call baseline" in r
                    for r in result.reasons)

    def test_min_cmd_seq_passes_once_cmd_seq_advances_past_the_baseline(self):
        now, now_ns, _ = self._ticking_clock()
        state = {
            "seq": 5, "sim_step": 200, "wall_time_ns": now["ns"], "cmd_seq": 4,
            "joints": [{"name": n, "compliant": False, "effort": 1.0}
                       for n in R.R_JOINTS],
        }
        result = ei.require_compliance(
            run_dir="/unused", joints=R.R_JOINTS, compliant=False,
            timeout_s=0.01, max_state_age_s=0.5, min_cmd_seq=3,
            read_last_state=lambda p: state, now_ns=now_ns,
            sleep=lambda s: None,
        )
        assert result.ok is True, result.reasons

    def test_min_cmd_seq_fails_closed_when_state_has_no_cmd_seq(self):
        now, now_ns, sleep = self._ticking_clock()
        state = {
            "seq": 5, "sim_step": 200, "wall_time_ns": now["ns"],
            "joints": [{"name": n, "compliant": False, "effort": 1.0}
                       for n in R.R_JOINTS],
        }
        result = ei.require_compliance(
            run_dir="/unused", joints=R.R_JOINTS, compliant=False,
            timeout_s=0.01, max_state_age_s=0.5, min_cmd_seq=3,
            read_last_state=lambda p: state, now_ns=now_ns,
            sleep=sleep,
        )
        assert result.ok is False
        assert any("no numeric cmd_seq" in r for r in result.reasons)

    def test_min_cmd_seq_rejects_a_bool_cmd_seq(self):
        """H3 (issue #119): `isinstance(True, (int, float))` is True, so a
        `cmd_seq` of `True`/`False` must not be accepted as a numeric
        sequence value -- match F3's `not isinstance(x, bool)` strictness."""
        now, now_ns, sleep = self._ticking_clock()
        state = {
            "seq": 5, "sim_step": 200, "wall_time_ns": now["ns"],
            "cmd_seq": True,
            "joints": [{"name": n, "compliant": False, "effort": 1.0}
                       for n in R.R_JOINTS],
        }
        result = ei.require_compliance(
            run_dir="/unused", joints=R.R_JOINTS, compliant=False,
            timeout_s=0.01, max_state_age_s=0.5, min_cmd_seq=0,
            read_last_state=lambda p: state, now_ns=now_ns,
            sleep=sleep,
        )
        assert result.ok is False
        assert any("no numeric cmd_seq" in r for r in result.reasons)

    def test_min_cmd_seq_rejects_nan(self):
        """H3 (issue #119): `NaN <= min_cmd_seq` is always False, so before
        the `math.isfinite` guard a `cmd_seq` of `NaN` looked "advanced past
        the baseline" no matter what `min_cmd_seq` was -- a silent bypass of
        the F2 sequence check."""
        now, now_ns, sleep = self._ticking_clock()
        state = {
            "seq": 5, "sim_step": 200, "wall_time_ns": now["ns"],
            "cmd_seq": float("nan"),
            "joints": [{"name": n, "compliant": False, "effort": 1.0}
                       for n in R.R_JOINTS],
        }
        result = ei.require_compliance(
            run_dir="/unused", joints=R.R_JOINTS, compliant=False,
            timeout_s=0.01, max_state_age_s=0.5, min_cmd_seq=3,
            read_last_state=lambda p: state, now_ns=now_ns,
            sleep=sleep,
        )
        assert result.ok is False
        assert any("no numeric cmd_seq" in r for r in result.reasons)

    def test_min_cmd_seq_rejects_infinity(self):
        """H3 (issue #119): `math.isfinite` also rejects `Infinity`, which
        -- like `NaN` -- would otherwise always satisfy `> min_cmd_seq`
        regardless of the actual baseline."""
        now, now_ns, sleep = self._ticking_clock()
        state = {
            "seq": 5, "sim_step": 200, "wall_time_ns": now["ns"],
            "cmd_seq": float("inf"),
            "joints": [{"name": n, "compliant": False, "effort": 1.0}
                       for n in R.R_JOINTS],
        }
        result = ei.require_compliance(
            run_dir="/unused", joints=R.R_JOINTS, compliant=False,
            timeout_s=0.01, max_state_age_s=0.5, min_cmd_seq=3,
            read_last_state=lambda p: state, now_ns=now_ns,
            sleep=sleep,
        )
        assert result.ok is False
        assert any("no numeric cmd_seq" in r for r in result.reasons)

    def test_min_cmd_seq_default_none_keeps_old_behaviour(self):
        """Not passing min_cmd_seq at all must behave exactly as before F2
        -- freshness plus per-joint compliance is still sufficient."""
        now, now_ns, _ = self._ticking_clock()
        state = {
            "seq": 5, "sim_step": 200, "wall_time_ns": now["ns"],
            "joints": [{"name": n, "compliant": False, "effort": 1.0}
                       for n in R.R_JOINTS],
        }
        result = ei.require_compliance(
            run_dir="/unused", joints=R.R_JOINTS, compliant=False,
            timeout_s=0.01, max_state_age_s=0.5,
            read_last_state=lambda p: state, now_ns=now_ns,
            sleep=lambda s: None,
        )
        assert result.ok is True, result.reasons

    # -- F3 (PR #118 review): missing/malformed compliant fails closed --

    def test_missing_compliant_key_fails_closed(self):
        """Before F3, bool(None) == False would have silently satisfied a
        compliant=False gate -- a state stream that does not report
        compliance at all must never look stiff by omission."""
        now, now_ns, sleep = self._ticking_clock()
        joints = [{"name": n, "effort": 1.0} for n in R.R_JOINTS]
        state = {"seq": 1, "sim_step": 10, "wall_time_ns": now["ns"],
                  "joints": joints}
        result = ei.require_compliance(
            run_dir="/unused", joints=R.R_JOINTS, compliant=False,
            timeout_s=0.01, max_state_age_s=0.5,
            read_last_state=lambda p: state, now_ns=now_ns,
            sleep=sleep,
        )
        assert result.ok is False
        assert any("compliant field is missing or not a bool" in r
                    for r in result.reasons)
        assert all(result.per_joint[n]["compliant"] is None for n in R.R_JOINTS)

    def test_missing_compliant_key_fails_closed_for_the_compliant_true_gate_too(self):
        """Fail-closed must hold for BOTH gate directions, not just
        compliant=False -- an unknown state is never treated as satisfying
        either expectation."""
        now, now_ns, sleep = self._ticking_clock()
        joints = [{"name": n, "effort": 1.0} for n in R.R_JOINTS]
        state = {"seq": 1, "sim_step": 10, "wall_time_ns": now["ns"],
                  "joints": joints}
        result = ei.require_compliance(
            run_dir="/unused", joints=R.R_JOINTS, compliant=True,
            timeout_s=0.01, max_state_age_s=0.5,
            read_last_state=lambda p: state, now_ns=now_ns,
            sleep=sleep,
        )
        assert result.ok is False

    def test_explicit_null_compliant_fails_closed(self):
        now, now_ns, sleep = self._ticking_clock()
        joints = [{"name": n, "compliant": None, "effort": 1.0}
                  for n in R.R_JOINTS]
        state = {"seq": 1, "sim_step": 10, "wall_time_ns": now["ns"],
                  "joints": joints}
        result = ei.require_compliance(
            run_dir="/unused", joints=R.R_JOINTS, compliant=False,
            timeout_s=0.01, max_state_age_s=0.5,
            read_last_state=lambda p: state, now_ns=now_ns,
            sleep=sleep,
        )
        assert result.ok is False
        assert any("compliant field is missing or not a bool" in r
                    for r in result.reasons)

    def test_non_bool_compliant_fails_closed(self):
        now, now_ns, sleep = self._ticking_clock()
        joints = [{"name": n, "compliant": "false", "effort": 1.0}
                  for n in R.R_JOINTS]
        state = {"seq": 1, "sim_step": 10, "wall_time_ns": now["ns"],
                  "joints": joints}
        result = ei.require_compliance(
            run_dir="/unused", joints=R.R_JOINTS, compliant=False,
            timeout_s=0.01, max_state_age_s=0.5,
            read_last_state=lambda p: state, now_ns=now_ns,
            sleep=sleep,
        )
        assert result.ok is False
        assert any("compliant field is missing or not a bool" in r
                    for r in result.reasons)


class TestReadLastStateTailReading:
    """F1 (PR #118 review): `_read_last_state` must read a bounded TAIL of
    states.jsonl, not the whole file, and still tolerate a torn last line
    and a single record wider than the initial window."""

    def test_small_tail_window_still_finds_the_last_line(self, tmp_path):
        p = tmp_path / "states.jsonl"
        lines = [json.dumps({"seq": i}) for i in range(1, 101)]
        p.write_text("\n".join(lines) + "\n")
        result = ei._read_last_state(p, tail_bytes=8)
        assert result == {"seq": 100}

    def test_default_window_reads_a_normal_state_unchanged(self, tmp_path):
        p = tmp_path / "states.jsonl"
        state = {"seq": 3, "sim_step": 10, "wall_time_ns": 1,
                 "joints": [{"name": "r_gripper", "position_rad": 0.0}]}
        p.write_text(json.dumps(state) + "\n")
        assert ei._read_last_state(p) == state

    def test_a_record_wider_than_the_initial_window_is_still_found(self, tmp_path):
        p = tmp_path / "states.jsonl"
        big_state = {"seq": 1, "padding": "x" * 200}
        p.write_text(json.dumps(big_state) + "\n")
        result = ei._read_last_state(p, tail_bytes=16)
        assert result == big_state

    def test_torn_last_line_falls_back_to_the_previous_complete_line(self, tmp_path):
        p = tmp_path / "states.jsonl"
        good = json.dumps({"seq": 7, "ok": True})
        with open(p, "wb") as f:
            f.write((good + "\n").encode())
            f.write(b'{"seq": 8, "torn": tr')  # partial write, no closing brace
        result = ei._read_last_state(p)
        assert result == {"seq": 7, "ok": True}

    def test_torn_last_line_survives_a_tiny_tail_window_too(self, tmp_path):
        """The fragment nearest the seek point is dropped on every retry,
        not just the first -- a torn write must not be mistaken for the
        deliberately-clipped fragment at a small window's boundary."""
        p = tmp_path / "states.jsonl"
        good = json.dumps({"seq": 7, "ok": True})
        with open(p, "wb") as f:
            f.write((good + "\n").encode())
            f.write(b'{"seq": 8, "torn": tr')
        result = ei._read_last_state(p, tail_bytes=4)
        assert result == {"seq": 7, "ok": True}

    def test_empty_file_returns_none(self, tmp_path):
        p = tmp_path / "states.jsonl"
        p.write_text("")
        assert ei._read_last_state(p) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert ei._read_last_state(tmp_path / "does_not_exist.jsonl") is None

    def test_bounded_read_does_not_pull_in_the_whole_large_file(
            self, tmp_path, monkeypatch):
        """The point of F1: a 20 Hz poll against a large states.jsonl must
        cost a bounded read, not a whole-file read that grows with the run.
        Patches `open` only inside the `e1_identity` module's own namespace
        (module-global lookup shadows the builtin there, nowhere else), so
        pytest's own I/O is untouched."""
        p = tmp_path / "states.jsonl"
        filler = "\n".join(json.dumps({"seq": i}) for i in range(20_000))
        p.write_text(filler + "\n" + json.dumps({"seq": 999999}) + "\n")
        file_size = p.stat().st_size
        assert file_size > 200_000  # sanity: this is the large file F1 is about

        class _CountingFile:
            def __init__(self, f):
                self._f = f
                self.total_read = 0

            def seek(self, *a, **kw):
                return self._f.seek(*a, **kw)

            def read(self, n=-1):
                data = self._f.read(n)
                self.total_read += len(data)
                return data

            def __enter__(self):
                return self

            def __exit__(self, *a):
                self._f.close()
                return False

        import builtins
        real_builtin_open = builtins.open
        opened = []

        def spy_open(path, mode="r", *a, **kw):
            f = real_builtin_open(path, mode, *a, **kw)
            if mode == "rb":
                wrapped = _CountingFile(f)
                opened.append(wrapped)
                return wrapped
            return f

        monkeypatch.setattr(ei, "open", spy_open, raising=False)

        tail_bytes = 1024
        result = ei._read_last_state(p, tail_bytes=tail_bytes)

        assert result == {"seq": 999999}
        assert len(opened) == 1
        # H2 (issue #119): pin what the docstring actually claims -- a
        # bounded read, not merely "less than half the file". One
        # doubling is allowed (the initial window may miss a line
        # boundary), but no more.
        assert opened[0].total_read <= 2 * tail_bytes

    def test_tail_bytes_zero_does_not_spin_forever(self, tmp_path):
        """H1 (issue #119): before the clamp, `tail_bytes=0` pinned
        `window` at 0 forever -- `min(size, 0*2) == 0` and `0 >= size` is
        false for any non-empty file, so the loop never terminated. Use a
        custom exception for the hang detector, not `TimeoutError`: it is
        an `OSError` subclass, and `_read_last_state`'s own `except
        OSError` would silently swallow it."""
        p = tmp_path / "states.jsonl"
        p.write_text(json.dumps({"seq": 1}) + "\n")

        class _Hung(Exception):
            pass

        def _on_alarm(signum, frame):
            raise _Hung("tail_bytes=0 did not terminate")

        old_handler = signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(2)
        try:
            result = ei._read_last_state(p, tail_bytes=0)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        assert result == {"seq": 1}

    def test_negative_tail_bytes_does_not_raise(self, tmp_path):
        """H1 (issue #119): before the clamp, `tail_bytes<0` reached
        `f.read()` with a negative count, which raises `ValueError` (not
        the `OSError` this function fails closed on) and propagates."""
        p = tmp_path / "states.jsonl"
        p.write_text(json.dumps({"seq": 1}) + "\n")
        result = ei._read_last_state(p, tail_bytes=-1)
        assert result == {"seq": 1}
