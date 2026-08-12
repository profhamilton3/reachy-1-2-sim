"""Unit tests for R12-603: recording and replay.

All tests run offline — no MuJoCo or network required.
"""

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "native_mujoco"))

from recorder import Recorder
from replayer import Replayer


# ── Recorder basics ───────────────────────────────────────────────────────────

class TestRecorderBasics:

    def test_creates_run_directory(self, tmp_path):
        rec = Recorder.new(tmp_path)
        assert rec.run_dir.exists()
        assert rec.run_dir.is_dir()

    def test_run_dir_has_timestamp_prefix(self, tmp_path):
        rec = Recorder.new(tmp_path)
        assert rec.run_dir.name.startswith("run_")

    def test_manifest_written_on_create(self, tmp_path):
        rec = Recorder.new(tmp_path, {"model_path": "/opt/reachy.xml"})
        manifest = json.loads((rec.run_dir / "manifest.json").read_text())
        assert manifest["format_version"] == 1
        assert "started_at" in manifest
        assert manifest.get("model_path") == "/opt/reachy.xml"

    def test_state_file_created(self, tmp_path):
        rec = Recorder.new(tmp_path)
        assert (rec.run_dir / "states.jsonl").exists()

    def test_command_file_created(self, tmp_path):
        rec = Recorder.new(tmp_path)
        assert (rec.run_dir / "commands.jsonl").exists()


# ── Recording state ───────────────────────────────────────────────────────────

class TestRecordState:

    def test_records_state_to_jsonl(self, tmp_path):
        rec = Recorder.new(tmp_path)
        rec.record_state({"seq": 1, "sim_step": 10, "sim_time_s": 0.02})
        rec.record_state({"seq": 2, "sim_step": 20, "sim_time_s": 0.04})
        lines = (rec.run_dir / "states.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        d = json.loads(lines[0])
        assert d["seq"] == 1
        assert d["sim_time_s"] == pytest.approx(0.02)

    def test_multiple_states_all_valid_json(self, tmp_path):
        rec = Recorder.new(tmp_path)
        for i in range(50):
            rec.record_state({"seq": i, "joints": [{"name": "r_shoulder_pitch", "position_rad": 0.1 * i}]})
        lines = (rec.run_dir / "states.jsonl").read_text().strip().split("\n")
        assert len(lines) == 50
        for line in lines:
            json.loads(line)   # must not raise


# ── Recording commands ────────────────────────────────────────────────────────

class TestRecordCommands:

    def test_records_joint_command(self, tmp_path):
        rec = Recorder.new(tmp_path)
        rec.record_command({"type": "joint_command", "seq": 1,
                            "target_rad": [0.0] * 21})
        lines = (rec.run_dir / "commands.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["type"] == "joint_command"
        assert len(d["target_rad"]) == 21

    def test_record_reset_adds_seed(self, tmp_path):
        rec = Recorder.new(tmp_path)
        rec.record_reset(seed=42, sim_step=500)
        lines = (rec.run_dir / "commands.jsonl").read_text().strip().split("\n")
        d = json.loads(lines[0])
        assert d["type"] == "reset"
        assert d["seed"] == 42
        assert d["sim_step"] == 500

    def test_record_reset_null_seed(self, tmp_path):
        rec = Recorder.new(tmp_path)
        rec.record_reset(seed=None, sim_step=0)
        d = json.loads((rec.run_dir / "commands.jsonl").read_text().strip())
        assert d["seed"] is None


# ── Finalize ──────────────────────────────────────────────────────────────────

class TestFinalize:

    def test_finalize_updates_manifest(self, tmp_path):
        rec = Recorder.new(tmp_path)
        rec.record_state({"seq": 1})
        rec.record_reset(seed=7, sim_step=0)
        rec.finalize(total_steps=12500, duration_s=25.0)
        manifest = json.loads((rec.run_dir / "manifest.json").read_text())
        assert manifest["total_steps"] == 12500
        assert manifest["total_duration_s"] == pytest.approx(25.0)
        assert 7 in manifest["seeds"]
        assert manifest["state_count"] == 1
        assert manifest["command_count"] == 1

    def test_files_readable_after_finalize(self, tmp_path):
        rec = Recorder.new(tmp_path)
        for i in range(5):
            rec.record_state({"seq": i})
        rec.finalize(total_steps=5, duration_s=0.1)
        # Should still be parseable after close
        lines = (rec.run_dir / "states.jsonl").read_text().strip().split("\n")
        assert len(lines) == 5


# ── Replayer ──────────────────────────────────────────────────────────────────

class TestReplayer:

    def _make_recording(self, tmp_path):
        rec = Recorder.new(tmp_path, {"model_path": "/opt/model.xml"})
        for i in range(3):
            rec.record_state({"seq": i, "sim_time_s": i * 0.02})
        rec.record_command({"type": "joint_command", "seq": 1, "target_rad": [0.5]*21})
        rec.record_command({"type": "joint_command", "seq": 2, "target_rad": [0.0]*21})
        rec.record_reset(seed=99, sim_step=250)
        rec.finalize(total_steps=3, duration_s=0.06)
        return rec.run_dir

    def test_replay_manifest(self, tmp_path):
        run_dir = self._make_recording(tmp_path)
        rp = Replayer(run_dir)
        m = rp.manifest()
        assert m["format_version"] == 1
        assert m["model_path"] == "/opt/model.xml"
        assert m["total_steps"] == 3

    def test_replay_states(self, tmp_path):
        run_dir = self._make_recording(tmp_path)
        states = list(Replayer(run_dir).states())
        assert len(states) == 3
        assert states[0]["seq"] == 0
        assert states[2]["sim_time_s"] == pytest.approx(0.04)

    def test_replay_commands(self, tmp_path):
        run_dir = self._make_recording(tmp_path)
        cmds = list(Replayer(run_dir).commands())
        assert len(cmds) == 3   # 2 joint_commands + 1 reset

    def test_replay_joint_commands_only(self, tmp_path):
        run_dir = self._make_recording(tmp_path)
        jcmds = list(Replayer(run_dir).joint_commands())
        assert len(jcmds) == 2
        assert all(c["type"] == "joint_command" for c in jcmds)

    def test_replay_resets_only(self, tmp_path):
        run_dir = self._make_recording(tmp_path)
        resets = list(Replayer(run_dir).resets())
        assert len(resets) == 1
        assert resets[0]["seed"] == 99

    def test_summary_string(self, tmp_path):
        run_dir = self._make_recording(tmp_path)
        summary = Replayer(run_dir).summary()
        assert "total_steps" in summary
        assert "3" in summary

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Replayer(tmp_path / "no_such_run")

    def test_missing_manifest_raises(self, tmp_path):
        run_dir = tmp_path / "run_bad"
        run_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="manifest"):
            Replayer(run_dir)

    def test_empty_command_file_yields_nothing(self, tmp_path):
        run_dir = tmp_path / "run_empty"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text('{"format_version": 1}')
        (run_dir / "states.jsonl").write_text("")
        (run_dir / "commands.jsonl").write_text("")
        rp = Replayer(run_dir)
        assert list(rp.commands()) == []
        assert list(rp.states()) == []
