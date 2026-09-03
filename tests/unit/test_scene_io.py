"""Unit tests for scene inheritance (native_mujoco/scene_io.py).

The point of `extends` is that a variant scene does not fork the measured
geometry it varies.  Every test here is really the same question: does the
child get the parent's measurements without restating them, and does it get
them RIGHT.
"""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "native_mujoco"))

from scene_io import SceneLoadError, load_scene

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def write(tmp_path, name, doc):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return p


PARENT = {
    "schema_version": "1.0",
    "name": "parent",
    "world": {"gravity": [0, 0, -9.81], "background_rgba": [0.1, 0.2, 0.3, 1.0]},
    "objects": [
        {"id": "a", "geometry": {"kind": "box", "size": [1, 1, 1]},
         "material": {"rgba": [1.0, 0, 0, 1.0]}, "tags": ["x"]},
        {"id": "b", "geometry": {"kind": "cylinder", "radius": 0.1},
         "material": {"rgba": [0, 0, 1.0, 1.0]}},
    ],
}


class TestInheritance:
    def test_a_child_that_says_nothing_is_its_parent(self, tmp_path):
        write(tmp_path, "p.yaml", PARENT)
        c = write(tmp_path, "c.yaml", {"extends": "p.yaml"})
        doc = load_scene(c)
        assert doc["objects"] == PARENT["objects"]
        assert doc["world"] == PARENT["world"]

    def test_patching_one_field_keeps_the_rest_of_that_object(self, tmp_path):
        """The whole reason this exists: recolouring must not cost the
        geometry, which is the measured part."""
        write(tmp_path, "p.yaml", PARENT)
        c = write(tmp_path, "c.yaml", {
            "extends": "p.yaml",
            "objects": [{"id": "a", "material": {"rgba": [0.5, 0.5, 0.5, 1.0]}}],
        })
        a = {o["id"]: o for o in load_scene(c)["objects"]}["a"]
        assert a["material"]["rgba"] == [0.5, 0.5, 0.5, 1.0]
        assert a["geometry"] == {"kind": "box", "size": [1, 1, 1]}
        assert a["tags"] == ["x"]

    def test_an_unknown_id_is_added_not_merged(self, tmp_path):
        write(tmp_path, "p.yaml", PARENT)
        c = write(tmp_path, "c.yaml", {
            "extends": "p.yaml", "objects": [{"id": "c", "geometry": {}}]})
        ids = [o["id"] for o in load_scene(c)["objects"]]
        assert ids == ["a", "b", "c"]

    def test_scalars_and_mappings_merge_key_by_key(self, tmp_path):
        write(tmp_path, "p.yaml", PARENT)
        c = write(tmp_path, "c.yaml", {
            "extends": "p.yaml", "name": "child",
            "world": {"background_rgba": [0.9, 0.9, 0.9, 1.0]}})
        doc = load_scene(c)
        assert doc["name"] == "child"
        assert doc["world"]["background_rgba"] == [0.9, 0.9, 0.9, 1.0]
        assert doc["world"]["gravity"] == [0, 0, -9.81]   # untouched

    def test_a_plain_list_is_replaced_not_merged(self, tmp_path):
        """Anonymous list elements have no identity to merge on, so guessing
        would be worse than making the child restate the list."""
        write(tmp_path, "p.yaml", PARENT)
        c = write(tmp_path, "c.yaml", {
            "extends": "p.yaml",
            "objects": [{"id": "a", "tags": ["y", "z"]}]})
        a = {o["id"]: o for o in load_scene(c)["objects"]}["a"]
        assert a["tags"] == ["y", "z"]

    def test_extends_key_is_not_left_in_the_document(self, tmp_path):
        write(tmp_path, "p.yaml", PARENT)
        c = write(tmp_path, "c.yaml", {"extends": "p.yaml"})
        assert "extends" not in load_scene(c)


class TestDropObjects:
    def test_it_removes_the_named_objects(self, tmp_path):
        write(tmp_path, "p.yaml", PARENT)
        c = write(tmp_path, "c.yaml", {"extends": "p.yaml", "drop_objects": ["b"]})
        assert [o["id"] for o in load_scene(c)["objects"]] == ["a"]

    def test_a_typo_raises_rather_than_silently_keeping_the_object(self, tmp_path):
        """Dropping is used to build single-class training runs.  A silent
        no-op here puts an unlabelled object in every frame of a set whose
        whole premise is that only one class is present."""
        write(tmp_path, "p.yaml", PARENT)
        c = write(tmp_path, "c.yaml", {"extends": "p.yaml", "drop_objects": ["bb"]})
        with pytest.raises(SceneLoadError, match="bb"):
            load_scene(c)


class TestBadInput:
    def test_a_self_reference_raises(self, tmp_path):
        c = write(tmp_path, "c.yaml", {"extends": "c.yaml", "name": "c"})
        with pytest.raises(SceneLoadError, match="itself"):
            load_scene(c)

    def test_a_cycle_raises_rather_than_hanging(self, tmp_path):
        write(tmp_path, "a.yaml", {"extends": "b.yaml", "name": "a"})
        b = write(tmp_path, "b.yaml", {"extends": "a.yaml", "name": "b"})
        with pytest.raises(SceneLoadError, match="cycle"):
            load_scene(b)

    def test_a_missing_parent_names_the_file(self, tmp_path):
        c = write(tmp_path, "c.yaml", {"extends": "nope.yaml"})
        with pytest.raises(SceneLoadError, match="nope.yaml"):
            load_scene(c)


class TestTheRealVariant:
    """FWDCenterLabSiva exists to be handed to someone else, so its inherited
    content is checked against the parent rather than against a fixture."""

    @pytest.fixture(scope="class")
    def scenes(self):
        base = os.path.join(_REPO, "scenes")
        parent = yaml.safe_load(
            open(os.path.join(base, "FWDCenterLabMCC.yaml")).read())
        child = load_scene(os.path.join(base, "FWDCenterLabSiva.yaml"))
        return parent, child

    def test_it_inherits_the_measured_geometry_untouched(self, scenes):
        parent, child = scenes
        p = {o["id"]: o for o in parent["objects"]}
        c = {o["id"]: o for o in child["objects"]}
        for oid in ("red_cube", "blue_cylinder", "soda_can", "foam_block"):
            assert c[oid]["geometry"] == p[oid]["geometry"]
            assert c[oid]["pose"] == p[oid]["pose"]
            assert c[oid]["physics"] == p[oid]["physics"]

    def test_every_object_is_repainted_achromatic(self, scenes):
        """The real cube and cylinder measure R=G=B to within a few counts.
        A saturated sim object teaches a detector a feature the real feed does
        not have, which was the defect this scene was made to fix."""
        _, child = scenes
        c = {o["id"]: o for o in child["objects"]}
        for oid in ("red_cube", "blue_cylinder"):
            r, g, b = c[oid]["material"]["rgba"][:3]
            assert max(r, g, b) - min(r, g, b) <= 0.02, oid

    def test_no_class_is_still_named_after_a_colour(self, scenes):
        _, child = scenes
        classes = {o["semantic_class"] for o in child["objects"]
                   if "detector-target" in (o.get("tags") or [])}
        assert classes == {"cube", "cylinder", "can", "foam"}

    def test_the_trays_are_gone(self, scenes):
        """Flat coloured zones with no real counterpart, in frame every time."""
        _, child = scenes
        ids = {o["id"] for o in child["objects"]}
        assert "left_tray" not in ids and "right_tray" not in ids

    def test_the_grid_and_rig_survive(self, scenes):
        parent, child = scenes
        p = {o["id"] for o in parent["objects"]}
        c = {o["id"] for o in child["objects"]}
        assert {i for i in p if i.startswith(("cell_", "rig_", "grid_"))} <= c
