"""sim #212, owner decision D5 (2026-11-27): the colour variants of the Pool scene
share its route validations because they ARE its geometry.

`rig_routes.SCENE_GEOMETRY_PARENTS` records that, explicitly.  This file is what
makes the record true: the compiled models are identical except for geom_rgba,
and the resolved scene documents (which the planner reads) differ only in
colour and naming.  If a variant ever moves, resizes, re-masses or re-tags
anything, these tests fail and the record has to go.

The experience reuse gate (`compatibility.check_reuse`) is deliberately NOT
given this equivalence (owner decision, the same day): it stays exact.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "native_mujoco", _ROOT / "src", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reachy_ai.motion import rig_routes as R  # noqa: E402
from scene_io import load_scene  # noqa: E402

_SCENES = _ROOT / "scenes"
PARENT = "FWDCenterLabSivaPool"
VARIANTS = ("FWDCenterLabSivaPoolBlack", "FWDCenterLabSivaPoolRandom")
ROUTES = sorted({r.route for r in R.ROUTE_COMPATIBILITY})
_MODEL = str(_ROOT / "native_mujoco" / "model" / "reachy_1_2.xml")


def _variant_docs(tmp_path):
    """(label, scene name, resolved document) of every variant, including
    per-run random scenes at several seeds."""
    from make_part_colour_scene import write_scene
    for name in VARIANTS:
        yield name, name, load_scene(_SCENES / f"{name}.yaml")
    for seed in (0, 1, 20261127, 2 ** 32 - 1):
        doc = load_scene(write_scene(seed, tmp_path))
        yield f"seed {seed}", doc["name"], doc


def _geometry_document(doc):
    """Everything in a resolved scene except colour and naming."""
    out = json.loads(json.dumps(doc))
    for key in ("name", "description", "color_randomization"):
        out.pop(key, None)
    for obj in out["objects"]:
        (obj.get("material") or {}).pop("rgba", None)
    return out


def _model_bytes_without_colour(doc):
    """The whole compiled MjModel as MuJoCo serialises it, with geom_rgba
    zeroed: arrays, sizes, names and the solver options alike."""
    mujoco = pytest.importorskip("mujoco")
    import numpy as np
    from contact_model import apply_contact_model
    from objects import build_scene_model_xml
    model = mujoco.MjModel.from_xml_string(build_scene_model_xml(doc, _MODEL))
    apply_contact_model(model, None)
    model.geom_rgba[:] = 0
    buffer = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return buffer.tobytes()


# ── The equivalence itself ───────────────────────────────────────────────────

class TestTheVariantsAreTheParentsGeometry:
    def test_the_record_names_exactly_these_variants(self):
        assert {(g.scene, g.parent) for g in R.SCENE_GEOMETRY_PARENTS} == {
            (v, PARENT) for v in VARIANTS}
        assert all(g.evidence for g in R.SCENE_GEOMETRY_PARENTS)
        # No chains: a parent is never itself somebody's variant.
        assert not {g.parent for g in R.SCENE_GEOMETRY_PARENTS} & set(VARIANTS)

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_the_scene_file_extends_the_recorded_parent(self, variant):
        import yaml
        raw = yaml.safe_load((_SCENES / f"{variant}.yaml").read_text())
        assert raw["extends"] == f"{R.geometry_scene(variant)}.yaml"
        assert raw["name"] == variant

    def test_compiled_models_are_identical_except_for_rgba(self, tmp_path):
        parent = _model_bytes_without_colour(load_scene(_SCENES / f"{PARENT}.yaml"))
        # The comparison is only meaningful if a recompile reproduces itself.
        assert parent == _model_bytes_without_colour(load_scene(_SCENES / f"{PARENT}.yaml"))
        for label, _name, doc in _variant_docs(tmp_path):
            assert _model_bytes_without_colour(doc) == parent, label

    def test_the_variants_really_do_differ_in_colour(self, tmp_path):
        parent = load_scene(_SCENES / f"{PARENT}.yaml")
        for label, _name, doc in _variant_docs(tmp_path):
            assert [o["material"] for o in doc["objects"] if "material" in o] != \
                   [o["material"] for o in parent["objects"] if "material" in o], label

    def test_the_scene_documents_differ_only_in_colour_and_naming(self, tmp_path):
        """The planner reads the document, not the model: tags and classes
        count too."""
        parent = _geometry_document(load_scene(_SCENES / f"{PARENT}.yaml"))
        for label, _name, doc in _variant_docs(tmp_path):
            assert _geometry_document(doc) == parent, label

    @pytest.mark.parametrize("change", [
        lambda o: o["pose"]["position"].__setitem__(0, o["pose"]["position"][0] + 0.001),
        lambda o: o["physics"].__setitem__("mass", o["physics"]["mass"] + 0.001),
        lambda o: o["physics"].__setitem__("friction", [0.7, 0.01, 0.001]),
    ], ids=["moved 1 mm", "1 g heavier", "other friction"])
    def test_both_comparisons_would_catch_a_real_change(self, change):
        parent = load_scene(_SCENES / f"{PARENT}.yaml")
        doc = copy.deepcopy(parent)
        change(next(o for o in doc["objects"] if o["id"] == "part110_a"))
        assert _geometry_document(doc) != _geometry_document(parent)
        assert _model_bytes_without_colour(doc) != _model_bytes_without_colour(parent)


# ── What the record changes ──────────────────────────────────────────────────

class TestRouteValidations:
    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("route", ROUTES)
    def test_a_variant_gets_the_parents_answer(self, route, variant):
        assert R.check_route(route, variant) == R.check_route(route, PARENT)
        assert R.check_route(route, variant)[0] is True

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_the_record_still_says_where_the_route_was_flown(self, variant):
        """Not a silent alias: the row handed back is the parent's own, and the
        reason for using it can be asked for."""
        row = R.validation_for("WAVE", variant)
        assert row is R.validation_for("WAVE", PARENT)
        assert row.scene == PARENT
        assert R.geometry_parent(variant).parent == PARENT
        assert R.geometry_parent(PARENT) is None

    @pytest.mark.parametrize("scene", [
        "FWDCenterLabSiva", "FWDCenterLabSivaPoolBlackish", "tabletop_demo",
        # Children of the Pool scene too, but they MOVE objects: extending a
        # scene is not what earns its validations.
        "B1_evidence", "B4_pool_box_1_r2c3",
    ])
    def test_any_other_scene_is_still_refused(self, scene):
        ok, why = R.check_route("WAVE", scene)
        assert ok is False and scene in why

    def test_a_rejection_recorded_for_the_parent_reaches_the_variants(self, monkeypatch):
        rejected = R.ValidationAttempt("SWING", PARENT, "2026-01-01", "rejected",
                                       "2 mm inside the rail")
        monkeypatch.setattr(R, "VALIDATION_ATTEMPTS", (rejected,))
        for variant in VARIANTS:
            ok, why = R.check_route("SWING", variant)
            assert ok is False and "2 mm inside the rail" in why and variant in why

    def test_the_mount_rule_is_unchanged_for_a_variant(self):
        for variant in VARIANTS:
            assert R.validation_for("WAVE", variant, mount="august") is None


class TestTheReuseGateStaysExact:
    def test_a_promoted_parent_trial_is_still_refused_in_a_variant(self):
        """Owner decision 2026-11-27: route validations only.  A different
        scene file is a different world to the reuse gate."""
        from reachy_ai.experience.compatibility import (
            KIND_ABILITY_ROUTE, ReuseCandidate, ReuseRequest, check_reuse)
        from reachy_ai.experience.identity import build_simulator_identity
        from objects import build_scene_model_xml
        pytest.importorskip("mujoco")

        def identity(name):
            path = _SCENES / f"{name}.yaml"
            return dataclasses.replace(
                build_simulator_identity(
                    model_path=_MODEL, scene_path=str(path),
                    compiled_xml=build_scene_model_xml(load_scene(path), _MODEL)),
                working_tree_dirty=False)

        candidate = ReuseCandidate(
            trial_id="t", identity=identity(PARENT), task_type="wave", arm="right",
            kind=KIND_ABILITY_ROUTE, route="WAVE", route_version=1, recipe_id="",
            recipe_version=0, start_posture="present", promotion_state="promoted",
            status="SUCCEEDED", success=True, live_interactive=False,
            obstacles=frozenset())
        request = ReuseRequest(task_type="wave", arm="right", route="WAVE",
                               route_version=1, start_posture="present")
        assert check_reuse(candidate, identity(PARENT), request).allowed
        for variant in VARIANTS:
            decision = check_reuse(candidate, identity(variant), request)
            assert not decision.allowed
            assert {m.field for m in decision.mismatches} == {
                "scene_sha256", "compiled_model_sha256"}
