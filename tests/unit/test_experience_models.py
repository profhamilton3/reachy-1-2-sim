"""R12-801: Unit tests for experience data contracts (models.py + recipe.py)."""

from __future__ import annotations

import json
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from reachy_ai.experience.models import (
    ControlPanelTaskSpec,
    EpisodeConfig,
    EpisodeResult,
    EpisodeStatus,
    PickPlaceTaskSpec,
    SimulatorIdentity,
    TaskSpec,
    TrialRecord,
)
from reachy_ai.motion.recipe import PrimitiveStep, TrajectoryRecipe


# ---------------------------------------------------------------------------
# EpisodeStatus
# ---------------------------------------------------------------------------

class TestEpisodeStatus:
    def test_all_values_present(self):
        values = {s.value for s in EpisodeStatus}
        assert values == {"PENDING", "RUNNING", "SUCCEEDED", "FAILED",
                          "INVALID", "ABORTED", "CANCELLED"}

    def test_is_terminal(self):
        assert EpisodeStatus.PENDING.is_terminal() is False
        assert EpisodeStatus.RUNNING.is_terminal() is False
        assert EpisodeStatus.SUCCEEDED.is_terminal() is True
        assert EpisodeStatus.FAILED.is_terminal() is True
        assert EpisodeStatus.INVALID.is_terminal() is True
        assert EpisodeStatus.ABORTED.is_terminal() is True
        assert EpisodeStatus.CANCELLED.is_terminal() is True

    def test_str_value(self):
        assert EpisodeStatus("SUCCEEDED") is EpisodeStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# SimulatorIdentity
# ---------------------------------------------------------------------------

class TestSimulatorIdentity:
    def _make(self, **kw) -> SimulatorIdentity:
        defaults = dict(
            model_sha256="abc123",
            scene_sha256="def456",
            mujoco_version="3.1.0",
            host_os="Darwin",
            host_arch="arm64",
        )
        defaults.update(kw)
        return SimulatorIdentity(**defaults)

    def test_all_required_fields_exist(self):
        fields = {f.name for f in SimulatorIdentity.__dataclass_fields__.values()}
        required = {
            "identity_version", "repository_git_sha", "working_tree_dirty",
            "model_source_path", "model_sha256", "compiled_model_sha256",
            "scene_source_path", "scene_sha256", "scene_revision",
            "scene_schema_version", "scene_compiler_version", "physics_profile_id",
            "protocol_version", "mujoco_version", "python_version", "host_os",
            "host_arch", "backend_name", "calibration_profile_id",
            "sensor_effect_profile_id",
        }
        assert required <= fields, f"Missing fields: {required - fields}"

    def test_frozen(self):
        ident = self._make()
        with pytest.raises((TypeError, AttributeError)):
            ident.model_sha256 = "newvalue"  # type: ignore[misc]

    def test_json_round_trip(self):
        ident = self._make(repository_git_sha="cafebabe", working_tree_dirty=True)
        recovered = SimulatorIdentity.from_json(ident.to_json())
        assert recovered == ident

    def test_from_dict_ignores_unknown_keys(self):
        d = self._make().to_dict()
        d["future_field"] = "ignored"
        SimulatorIdentity.from_dict(d)  # should not raise

    def test_matches_identical(self):
        a = self._make()
        b = self._make(repository_git_sha="different_sha")
        assert a.matches(b)

    def test_matches_different_model(self):
        a = self._make(model_sha256="aaa")
        b = self._make(model_sha256="bbb")
        assert not a.matches(b)

    def test_matches_different_scene(self):
        a = self._make(scene_sha256="xxx")
        b = self._make(scene_sha256="yyy")
        assert not a.matches(b)

    def test_matches_different_schema_version(self):
        a = self._make(scene_schema_version="1.0")
        b = self._make(scene_schema_version="2.0")
        assert not a.matches(b)

    def test_default_identity_version(self):
        assert SimulatorIdentity().identity_version == 1


# ---------------------------------------------------------------------------
# TaskSpec
# ---------------------------------------------------------------------------

class TestTaskSpec:
    def test_json_round_trip(self):
        spec = TaskSpec(
            task_id="t1",
            task_type="operate_control",
            timeout_steps=3000,
            success_definition="control reaches ON state",
        )
        recovered = TaskSpec.from_json(spec.to_json())
        assert recovered.task_id == "t1"
        assert recovered.timeout_steps == 3000

    def test_control_panel_spec(self):
        spec = ControlPanelTaskSpec(
            task_id="cp1",
            task_type="operate_control",
            control_id="btn_red",
            requested_final_state=True,
            neighbor_control_ids=["btn_green"],
        )
        d = spec.to_dict()
        assert d["control_id"] == "btn_red"
        assert d["neighbor_control_ids"] == ["btn_green"]

    def test_pick_place_spec(self):
        spec = PickPlaceTaskSpec(
            task_id="pp1",
            task_type="pick_and_place",
            object_id="red_cube",
            target_pose=[0.5, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
        )
        assert spec.object_id == "red_cube"
        assert len(spec.target_pose) == 7

    def test_from_dict_ignores_unknown_keys(self):
        d = {"task_id": "x", "task_type": "op", "future_field": 99}
        spec = TaskSpec.from_dict(d)
        assert spec.task_id == "x"


# ---------------------------------------------------------------------------
# EpisodeConfig
# ---------------------------------------------------------------------------

class TestEpisodeConfig:
    def _identity(self) -> SimulatorIdentity:
        return SimulatorIdentity(model_sha256="m1", scene_sha256="s1")

    def test_json_round_trip(self):
        cfg = EpisodeConfig(
            simulator_identity=self._identity(),
            seed=42,
            max_steps=5000,
            render_mode="off",
        )
        recovered = EpisodeConfig.from_json(cfg.to_json())
        assert recovered.seed == 42
        assert recovered.max_steps == 5000
        assert recovered.simulator_identity.model_sha256 == "m1"

    def test_nested_identity_preserved(self):
        cfg = EpisodeConfig(
            simulator_identity=SimulatorIdentity(
                model_sha256="abc",
                scene_sha256="def",
                mujoco_version="3.1.0",
            ),
            seed=7,
        )
        d = cfg.to_dict()
        assert d["simulator_identity"]["model_sha256"] == "abc"
        assert d["simulator_identity"]["mujoco_version"] == "3.1.0"


# ---------------------------------------------------------------------------
# EpisodeResult
# ---------------------------------------------------------------------------

class TestEpisodeResult:
    def test_default_status_is_pending(self):
        r = EpisodeResult(episode_id="e1", trial_id="t1")
        assert r.status == EpisodeStatus.PENDING

    def test_json_round_trip(self):
        r = EpisodeResult(
            episode_id="e1",
            trial_id="t1",
            status=EpisodeStatus.SUCCEEDED,
            success=True,
            end_sim_step=2500,
            metrics={"clearance_m": 0.032},
            hard_violations=[],
        )
        recovered = EpisodeResult.from_json(r.to_json())
        assert recovered.status == EpisodeStatus.SUCCEEDED
        assert recovered.success is True
        assert recovered.end_sim_step == 2500
        assert recovered.metrics["clearance_m"] == pytest.approx(0.032)

    def test_status_serialised_as_string(self):
        r = EpisodeResult(episode_id="e", trial_id="t", status=EpisodeStatus.ABORTED)
        d = r.to_dict()
        assert d["status"] == "ABORTED"
        assert isinstance(d["status"], str)

    def test_all_status_values_round_trip(self):
        for status in EpisodeStatus:
            r = EpisodeResult(episode_id="e", trial_id="t", status=status)
            recovered = EpisodeResult.from_json(r.to_json())
            assert recovered.status == status


# ---------------------------------------------------------------------------
# TrajectoryRecipe
# ---------------------------------------------------------------------------

class TestTrajectoryRecipe:
    def _make(self) -> TrajectoryRecipe:
        return TrajectoryRecipe(
            recipe_id="baseline_v1",
            task_type="operate_control",
            arm="right",
            primitive_sequence=[
                PrimitiveStep("guard", {"duration_s": 1.3}),
                PrimitiveStep("approach_standoff", {"standoff_m": 0.10}),
                PrimitiveStep("press_or_sweep", {"depth_m": 0.012}),
                PrimitiveStep("retract", {"standoff_m": 0.10}),
            ],
            bounded_parameters={
                "standoff_m": {"value": 0.10, "min": 0.05, "max": 0.20, "units": "m"},
                "depth_m": {"value": 0.012, "min": 0.005, "max": 0.025, "units": "m"},
            },
            source="baseline",
        )

    def test_json_round_trip(self):
        r = self._make()
        recovered = TrajectoryRecipe.from_json(r.to_json())
        assert recovered.recipe_id == "baseline_v1"
        assert len(recovered.primitive_sequence) == 4
        assert recovered.primitive_sequence[1].primitive == "approach_standoff"
        assert recovered.primitive_sequence[1].parameters["standoff_m"] == pytest.approx(0.10)

    def test_yaml_round_trip(self):
        yaml = pytest.importorskip("yaml")
        r = self._make()
        recovered = TrajectoryRecipe.from_yaml(r.to_yaml())
        assert recovered.recipe_id == r.recipe_id
        assert len(recovered.primitive_sequence) == len(r.primitive_sequence)

    def test_empty_recipe(self):
        r = TrajectoryRecipe(recipe_id="empty")
        assert r.primitive_sequence == []
        recovered = TrajectoryRecipe.from_json(r.to_json())
        assert recovered.primitive_sequence == []

    def test_primitive_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            PrimitiveStep("")

    def test_primitive_callable_parameter_raises(self):
        with pytest.raises(ValueError, match="callable"):
            PrimitiveStep("guard", {"fn": lambda x: x})

    def test_from_dict_ignores_unknown_keys(self):
        d = {
            "recipe_id": "r1",
            "primitive_sequence": [],
            "future_field": "ignored",
        }
        r = TrajectoryRecipe.from_dict(d)
        assert r.recipe_id == "r1"

    def test_schema_version_field(self):
        r = self._make()
        assert isinstance(r.recipe_version, int)
        assert r.recipe_version >= 1
