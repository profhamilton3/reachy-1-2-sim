"""sim #212: `geometry.kind: compound` -- several box/cylinder parts on one body.

A through-hole cannot be one primitive, and MuJoCo collides a mesh as its convex
hull, which fills the hole.  So a holed part is built from primitives.  This
file covers every layer that reads geometry: schema and custom validation,
compiler, runtime placement, gripper pinch, planner scene model, truth export
and RViz markers.
"""

from __future__ import annotations

import copy
import math
import os
import pathlib
import sys

import pytest
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "native_mujoco", _ROOT / "src", _ROOT / "ros"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scene_compiler import SceneCompilerError, compile_scene  # noqa: E402
from scene_io import SceneLoadError, compound_bounds  # noqa: E402
from scene_loader import SceneValidationError, load_scene  # noqa: E402

try:
    import jsonschema  # noqa: F401
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False

IDENT = [1.0, 0.0, 0.0, 0.0]


def _box_part(size, pos, quat=None):
    return {"kind": "box", "size": list(size), "pos": list(pos),
            "quat_wxyz": list(quat or IDENT)}


def holed_block_geometry(h=0.0254):
    """Outer 49.21 mm square, 25.4 mm square bore: two long and two short walls."""
    long_wall = [0.04921, 0.011905, h]
    short_wall = [0.011905, 0.0254, h]
    off = 0.0186525
    return {"kind": "compound", "parts": [
        _box_part(long_wall, [0.0, off, 0.0]),
        _box_part(long_wall, [0.0, -off, 0.0]),
        _box_part(short_wall, [off, 0.0, 0.0]),
        _box_part(short_wall, [-off, 0.0, 0.0]),
    ]}


def _obj(geometry, oid="holed", mass=0.16, dynamic=True, **extra):
    obj = {"id": oid, "geometry": geometry,
           "pose": {"position": [0.5, 0.0, 0.5], "orientation_wxyz": list(IDENT)},
           "material": {"rgba": [0.8, 0.6, 0.1, 1.0]},
           "physics": {"dynamic": dynamic, "collision": True, "mass": mass,
                       "friction": [0.8, 0.01, 0.001], "restitution": 0.05}}
    obj.update(extra)
    return obj


def _doc(*objects):
    return {"schema_version": "1.0", "name": "compound-test",
            "world": {"gravity": [0.0, 0.0, -9.81]}, "objects": list(objects)}


def _load(tmp_path, doc):
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(doc))
    return load_scene(path)


# ── Schema and custom validation ─────────────────────────────────────────────

class TestValidation:
    def test_a_valid_compound_loads_with_its_parts(self, tmp_path):
        scene = _load(tmp_path, _doc(_obj(holed_block_geometry())))
        geo = scene.objects[0].geometry
        assert geo.kind == "compound"
        assert len(geo.parts) == 4
        assert geo.parts[0].kind == "box"
        assert geo.parts[0].size == pytest.approx((0.04921, 0.011905, 0.0254))
        assert geo.parts[0].pos == pytest.approx((0.0, 0.0186525, 0.0))
        assert geo.parts[0].quat_wxyz == pytest.approx(tuple(IDENT))
        # The envelope is reported as `size`, so a reader that only knows
        # "full extents" is right about the outside of the object.
        assert geo.size == pytest.approx((0.04921, 0.04921, 0.0254))

    @pytest.mark.parametrize("mutate, needle", [
        (lambda g: g.update(size=[1, 1, 1]), "size"),
        (lambda g: g.update(colour="red"), "colour"),
        (lambda g: g["parts"][0].update(rgba=[1, 0, 0, 1]), "rgba"),
        (lambda g: g["parts"][0].update(radius=0.01), "radius"),
        (lambda g: g["parts"][0].pop("size"), "size"),
        (lambda g: g["parts"][0].pop("pos"), "pos"),
        (lambda g: g["parts"][0].pop("quat_wxyz"), "quat_wxyz"),
        (lambda g: g["parts"][0].update(kind="sphere"), "sphere"),
        (lambda g: g.update(parts=[]), "parts"),
        (lambda g: g.pop("parts"), "parts"),
    ])
    def test_bad_shapes_fail(self, tmp_path, mutate, needle):
        geo = holed_block_geometry()
        mutate(geo)
        with pytest.raises(SceneValidationError):
            _load(tmp_path, _doc(_obj(geo)))
        # The custom check alone must refuse it too: the server skips JSON
        # Schema when jsonschema is not installed.
        with pytest.raises(SceneCompilerError, match=needle):
            compile_scene(_doc(_obj(geo)))

    def test_a_cylinder_part_refuses_size(self):
        geo = {"kind": "compound", "parts": [
            {"kind": "cylinder", "radius": 0.02, "length": 0.03, "size": [1, 1, 1],
             "pos": [0, 0, 0], "quat_wxyz": IDENT}]}
        with pytest.raises(SceneCompilerError, match="size"):
            compile_scene(_doc(_obj(geo)))

    @pytest.mark.parametrize("field, value, needle", [
        ("quat_wxyz", [1.0, 1.0, 0.0, 0.0], "unit"),
        ("pos", [float("nan"), 0.0, 0.0], "finite"),
        ("size", [float("inf"), 0.01, 0.01], "finite"),
        ("size", [0.0, 0.01, 0.01], "positive"),
        ("pos", [0.0, 0.0], "pos"),
    ])
    def test_bad_values_fail(self, tmp_path, field, value, needle):
        geo = holed_block_geometry()
        geo["parts"][0][field] = value
        with pytest.raises(SceneCompilerError, match=needle):
            compile_scene(_doc(_obj(geo)))
        with pytest.raises(SceneValidationError):
            _load(tmp_path, _doc(_obj(geo)))

    def test_the_envelope_must_be_centred_on_the_object_origin(self, tmp_path):
        """Placement, the planner and truth export all read an object as
        centre +/- size/2, so an off-centre compound would be silently wrong."""
        geo = holed_block_geometry()
        for part in geo["parts"]:
            part["pos"][2] += 0.01
        with pytest.raises(SceneValidationError, match="centred"):
            _load(tmp_path, _doc(_obj(geo)))
        with pytest.raises(SceneCompilerError, match="centred"):
            compile_scene(_doc(_obj(geo)))

    def test_a_dynamic_compound_needs_a_mass(self, tmp_path):
        obj = _obj(holed_block_geometry())
        del obj["physics"]["mass"]
        with pytest.raises(SceneCompilerError, match="mass"):
            compile_scene(_doc(obj))
        with pytest.raises(SceneValidationError, match="mass"):
            _load(tmp_path, _doc(obj))

    def test_compound_with_articulation_is_refused(self, tmp_path):
        obj = _obj(holed_block_geometry(),
                   articulation={"joint": "hinge", "axis": [0, 0, 1]})
        with pytest.raises(SceneCompilerError, match="articulation"):
            compile_scene(_doc(obj))
        with pytest.raises(SceneValidationError, match="articulation"):
            _load(tmp_path, _doc(obj))

    @pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
    def test_the_schema_itself_rejects_an_unknown_part_field(self):
        import json
        import jsonschema as js
        schema = json.loads((_ROOT / "scenes" / "scene.schema.json").read_text())
        geo = holed_block_geometry()
        js.validate(_doc(_obj(geo)), schema)
        geo["parts"][0]["extra"] = 1
        with pytest.raises(js.ValidationError):
            js.validate(_doc(_obj(geo)), schema)

    def test_bounds_of_a_rotated_cylinder_part(self):
        # A cylinder tipped 90 deg about y lies along x.
        s = math.sqrt(0.5)
        geo = {"kind": "compound", "parts": [
            {"kind": "cylinder", "radius": 0.01, "length": 0.08,
             "pos": [0, 0, 0], "quat_wxyz": [s, 0.0, s, 0.0]}]}
        lo, hi = compound_bounds(geo)
        assert lo == pytest.approx((-0.04, -0.01, -0.01), abs=1e-9)
        assert hi == pytest.approx((0.04, 0.01, 0.01), abs=1e-9)

    def test_bounds_of_something_not_compound_is_an_error(self):
        with pytest.raises(SceneLoadError):
            compound_bounds({"kind": "box", "size": [1, 1, 1]})


# ── Compiler ─────────────────────────────────────────────────────────────────

class TestCompiler:
    def test_one_geom_per_part_and_one_inertial(self):
        xml = compile_scene(_doc(_obj(holed_block_geometry())))
        assert xml.count("<geom ") == 4
        assert xml.count("<inertial ") == 1
        assert 'mass="0.160000"' in xml
        # Mass lives on the inertial alone; a per-geom mass would be a second,
        # conflicting statement of it.
        assert all("mass=" not in line for line in xml.splitlines()
                   if "<geom " in line)
        assert xml.count('friction="0.8000 0.0100 0.0010"') == 4
        assert xml.count('contype="4" conaffinity="7"') == 4
        assert xml.count('rgba="0.800 0.600 0.100 1.000"') == 4

    def test_the_compiled_body_has_the_total_mass_and_exact_inertia(self):
        mujoco = pytest.importorskip("mujoco")
        h, mass = 0.0254, 0.16
        model = mujoco.MjModel.from_xml_string(
            compile_scene(_doc(_obj(holed_block_geometry(h), mass=mass))))
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "holed")
        assert int(model.body_geomnum[bid]) == 4
        assert float(model.body_mass[bid]) == pytest.approx(mass)
        assert list(model.body_ipos[bid]) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
        # Four non-overlapping walls == outer prism minus the square bore, so
        # the inertia has a closed form.
        a, b = 0.04921, 0.0254
        rho = mass / ((a * a - b * b) * h)
        izz = rho * h * (a ** 4 - b ** 4) / 6.0
        ixx = rho * h * ((a ** 4 - b ** 4) / 12.0 + h * h * (a * a - b * b) / 12.0)
        assert sorted(model.body_inertia[bid]) == pytest.approx(
            sorted([ixx, ixx, izz]), rel=1e-4)

    def test_rounding_residue_is_written_as_zero(self):
        """The products of inertia of a symmetric compound cancel to about
        1e-22, and Python 3.12 changed how sum() rounds, so the same scene
        compiled to different text on the host (3.14) and in the container
        (3.8): different content hashes for one world."""
        s = math.sqrt(0.5)
        ring = {"kind": "compound", "parts": [
            _box_part([0.01, 0.005, 0.02],
                      [0.02 * math.cos(k * math.pi / 8), 0.02 * math.sin(k * math.pi / 8), 0.0],
                      [math.cos(k * math.pi / 16), 0.0, 0.0, math.sin(k * math.pi / 16)])
            for k in range(16)]}
        for geometry in (ring, holed_block_geometry()):
            xml = compile_scene(_doc(_obj(geometry)))
            inertial = next(l for l in xml.splitlines() if "<inertial" in l)
            assert 'pos="0.000000000 0.000000000 0.000000000"' in inertial
            full = inertial.split('fullinertia="')[1].split('"')[0].split()
            assert full[3:] == ["0.000000000000e+00"] * 3, full

    def test_a_real_product_of_inertia_is_kept(self):
        """An L shape (centred bounding box) has a non-zero Ixy; only residue
        is dropped."""
        geometry = {"kind": "compound", "parts": [
            _box_part([0.04, 0.01, 0.01], [0.0, -0.015, 0.0]),
            _box_part([0.01, 0.03, 0.01], [-0.015, 0.005, 0.0])]}
        xml = compile_scene(_doc(_obj(geometry)))
        inertial = next(l for l in xml.splitlines() if "<inertial" in l)
        full = [float(v) for v in inertial.split('fullinertia="')[1].split('"')[0].split()]
        assert abs(full[3]) > 0.05 * max(full[:3])

    def test_a_static_compound_has_no_joint_and_no_mass_requirement(self):
        obj = _obj(holed_block_geometry(), dynamic=False)
        del obj["physics"]["mass"]
        xml = compile_scene(_doc(obj))
        assert "<freejoint" not in xml and "<inertial" not in xml
        assert xml.count("<geom ") == 4

    def test_collision_false_applies_to_every_part(self):
        obj = _obj(holed_block_geometry())
        obj["physics"]["collision"] = False
        assert compile_scene(_doc(obj)).count('contype="0" conaffinity="0"') == 4

    def test_compiling_does_not_change_the_document(self):
        doc = _doc(_obj(holed_block_geometry()))
        before = copy.deepcopy(doc)
        compile_scene(doc)
        assert doc == before


# ── Runtime placement ────────────────────────────────────────────────────────

def _placer(*objects):
    mujoco = pytest.importorskip("mujoco")
    from placement import ObjectPlacer
    doc = _doc(*objects)
    model = mujoco.MjModel.from_xml_string(compile_scene(doc))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return ObjectPlacer(model, data, doc), model, data


class TestPlacement:
    @pytest.mark.parametrize("kwargs", [
        {"size": [0.1, 0.1, 0.1]}, {"radius": 0.03}, {"length": 0.05},
        {"rgba": [1, 0, 0, 1]}, {"mass": 0.2}, {},
    ])
    def test_reshape_refuses_a_compound_object(self, kwargs):
        from placement import PlacementError
        placer, model, _ = _placer(_obj(holed_block_geometry()))
        before = (model.geom_size.copy(), model.geom_rgba.copy(), model.body_mass.copy())
        with pytest.raises(PlacementError, match="compound"):
            placer.reshape("holed", **kwargs)
        assert (model.geom_size == before[0]).all()
        assert (model.geom_rgba == before[1]).all()
        assert (model.body_mass == before[2]).all()

    def test_reshape_refuses_a_one_part_compound_too(self):
        from placement import PlacementError
        geo = {"kind": "compound", "parts": [_box_part([0.04, 0.04, 0.04], [0, 0, 0])]}
        placer, _, _ = _placer(_obj(geo))
        with pytest.raises(PlacementError, match="compound"):
            placer.reshape("holed", rgba=[1, 0, 0, 1])

    def test_rest_height_uses_every_part_not_the_first(self):
        """A T-shaped compound whose first part is a thin top plate: the
        underside is the stem's, 3 cm below the origin."""
        geo = {"kind": "compound", "parts": [
            _box_part([0.06, 0.06, 0.01], [0.0, 0.0, 0.025]),
            _box_part([0.02, 0.02, 0.05], [0.0, 0.0, -0.005]),
        ]}
        cell = {"id": "cell_r1c1", "geometry": {"kind": "box", "size": [0.12, 0.12, 0.002]},
                "pose": {"position": [0.4, 0.0, 0.699]}, "physics": {"collision": False},
                "tags": ["grid-cell"]}
        placer, _, _ = _placer(_obj(geo), cell)
        placed = placer.place("holed", "r1c1")
        assert placed.position[2] == pytest.approx(0.700 + 0.030 + 0.001, abs=1e-6)


# ── Gripper pinch across two parts ───────────────────────────────────────────

class TestGripperPinch:
    """A holed part gripped across its bore touches the finger on one wall and
    the thumb on the opposite wall: two geoms, one object."""

    _XML = """
    <mujoco>
      <worldbody>
        <geom name="r_finger_col" type="box" size="0.002 0.02 0.02" pos="-0.0265 0 1"
              contype="2" conaffinity="5"/>
        <geom name="r_thumb_col" type="box" size="0.002 0.02 0.02" pos="0.0265 0 1"
              contype="2" conaffinity="5"/>
        {bodies}
      </worldbody>
    </mujoco>
    """

    def _states(self, bodies):
        mujoco = pytest.importorskip("mujoco")
        from gripper import GripperModel
        model = mujoco.MjModel.from_xml_string(self._XML.format(bodies=bodies))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return GripperModel(model).update(data)

    def test_finger_and_thumb_on_different_parts_of_one_body_is_a_grasp(self):
        wall = ('<geom type="box" size="0.006 0.0246 0.0127" pos="{x} 0 0" '
                'contype="4" conaffinity="7"/>')
        body = ('<body name="part" pos="0 0 1"><freejoint/>'
                + wall.format(x=-0.0187) + wall.format(x=0.0187) + '</body>')
        state = self._states(body)["right"]
        assert state.finger_contacts and state.thumb_contacts
        assert state.grasping is True
        assert len(state.grasped_geoms) == 1

    def test_two_separate_bodies_are_still_not_a_grasp(self):
        cube = ('<body name="{n}" pos="{x} 0 1"><freejoint/>'
                '<geom type="box" size="0.006 0.02 0.02" contype="4" conaffinity="7"/>'
                '</body>')
        state = self._states(cube.format(n="a", x=-0.0187)
                             + cube.format(n="b", x=0.0187))["right"]
        assert state.finger_contacts and state.thumb_contacts
        assert state.grasping is False

    def test_two_geoms_of_one_static_body_are_still_not_a_grasp(self):
        """Unchanged for furniture: only a free (movable) body is one object."""
        wall = ('<geom type="box" size="0.006 0.02 0.02" pos="{x} 0 0" '
                'contype="4" conaffinity="7"/>')
        body = ('<body name="rig" pos="0 0 1">' + wall.format(x=-0.0187)
                + wall.format(x=0.0187) + '</body>')
        assert self._states(body)["right"].grasping is False


# ── Planner scene model ──────────────────────────────────────────────────────

class TestSceneModel:
    def test_extents_are_the_envelope_and_clearance_treats_it_as_a_box(self):
        from reachy_ai.scene.awareness import SceneModel, object_sdf
        scene = SceneModel.from_doc(_doc(_obj(holed_block_geometry())))
        obj = scene.get("holed")
        assert obj.size == pytest.approx((0.04921, 0.04921, 0.0254))
        # Off a corner of the SQUARE block: a cylinder of the same width would
        # call this 7 mm clear and let a link touch the corner.
        corner = (0.5 + 0.0246, 0.0 + 0.0246, 0.5)
        assert object_sdf(obj, corner) <= 1e-4
        # The bore is treated as solid: refusing more, never less.
        assert object_sdf(obj, (0.5, 0.0, 0.5)) < 0


# ── Truth export ─────────────────────────────────────────────────────────────

class TestTruthExport:
    def _truth(self, semantic_class, geometry=None):
        from truth_export import build_truth
        doc = _doc(_obj(geometry or holed_block_geometry(),
                        semantic_class=semantic_class))
        state = {"type": "state", "scene_revision": "r1", "sim_step": 1,
                 "sim_time_s": 0.002, "paused": False,
                 "objects": [{"object_id": "holed", "pos_xyz": [0.5, 0, 0.5],
                              "quat_wxyz": IDENT}]}
        return build_truth(state, doc)["objects"][0]

    def test_a_compound_block_reports_its_outer_box(self):
        o = self._truth("block")
        assert o["shape"] == "box" and o["radius_m"] is None
        assert o["size_m"] == pytest.approx([0.04921, 0.04921, 0.0254])

    def test_a_compound_cylinder_reports_its_outer_cylinder(self):
        o = self._truth("cylinder")
        assert o["shape"] == "cylinder"
        assert o["radius_m"] == pytest.approx(0.04921 / 2)

    def test_a_compound_with_no_outline_class_is_refused(self):
        """Fail closed: a compound's outline cannot be read from its parts (a
        ring of boxes is a cylinder)."""
        from truth_export import TruthError
        with pytest.raises(TruthError, match="outline"):
            self._truth("widget")

    def test_a_cylinder_class_with_a_non_square_footprint_is_refused(self):
        from truth_export import TruthError
        geo = {"kind": "compound", "parts": [_box_part([0.05, 0.03, 0.02], [0, 0, 0])]}
        with pytest.raises(TruthError, match="outline"):
            self._truth("cylinder", geo)


# ── RViz markers (a generated view of the scene) ─────────────────────────────

class TestRvizViews:
    def test_a_compound_becomes_one_primitive_view_per_part(self, tmp_path):
        from scene_marker_publisher import primitive_views
        obj = _obj(holed_block_geometry())
        # 90 deg yaw: the part at +y in the object frame lands at -x in the scene.
        s = math.sqrt(0.5)
        obj["pose"] = {"position": [0.5, 0.2, 0.7], "orientation_wxyz": [s, 0, 0, s]}
        views = primitive_views(_load(tmp_path, _doc(obj)).objects[0])
        assert len(views) == 4
        assert {v.geometry.kind for v in views} == {"box"}
        assert views[0].pose.position == pytest.approx((0.5 - 0.0186525, 0.2, 0.7))
        assert views[0].pose.orientation_wxyz == pytest.approx((s, 0, 0, s))
        assert views[0].material.rgba == pytest.approx((0.8, 0.6, 0.1, 1.0))

    def test_a_live_position_override_moves_the_whole_compound(self, tmp_path):
        from scene_marker_publisher import primitive_views
        obj = _load(tmp_path, _doc(_obj(holed_block_geometry()))).objects[0]
        views = primitive_views(obj, position=(1.0, 2.0, 3.0))
        assert views[0].pose.position == pytest.approx((1.0, 2.0 + 0.0186525, 3.0))

    def test_a_plain_object_is_its_own_single_view(self, tmp_path):
        from scene_marker_publisher import primitive_views
        obj = _load(tmp_path, _doc(_obj({"kind": "box", "size": [0.1, 0.1, 0.1]}))).objects[0]
        assert primitive_views(obj) == [obj]
