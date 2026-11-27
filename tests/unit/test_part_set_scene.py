"""sim #212: Willis's sixteen printed parts in FWDCenterLabSivaPool.

The table below is copied from the issue, not from the YAML, so the scene is
checked against the specification rather than against itself.  Nothing here
reads the STEP or STL files: they are not committed, and the scene is built
from primitives.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "native_mujoco", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from placement import (  # noqa: E402
    PlacementError,
    cells_from_scene,
    pool_ids,
    shoulder_distance,
)
from scene_io import SceneLoadError, compound_bounds, load_scene  # noqa: E402
from scene_loader import SceneValidationError  # noqa: E402
from scene_loader import load_scene as validate_scene  # noqa: E402

_SCENES = _ROOT / "scenes"
FAMILY = _SCENES / "FWDCenterLabSivaPool.yaml"
BLACK = _SCENES / "FWDCenterLabSivaPoolBlack.yaml"
RANDOM = _SCENES / "FWDCenterLabSivaPoolRandom.yaml"

BLOCK, ROUND = "block", "cylinder"
SIDE, RADIUS, BORE_R = 0.04921, 0.024606, 0.0127
SHORT, TALL = 0.02540, 0.06985

YELLOW = (0.812, 0.627, 0.137, 1.0)
ORANGE = (0.784, 0.353, 0.090, 1.0)
GREEN = (0.306, 0.604, 0.204, 1.0)
DUSTY_BLUE = (0.420, 0.510, 0.600, 1.0)
PALETTE = {YELLOW, ORANGE, GREEN, DUSTY_BLUE}

#: design: (shape, height m, hole, real volume cm3, mass kg, colour)
DESIGNS = {
    "110": (BLOCK, SHORT, False, 61.5, 0.175, YELLOW),
    "111": (BLOCK, SHORT, True, 48.7, 0.160, YELLOW),
    "120": (BLOCK, TALL, False, 169.2, 0.270, ORANGE),
    "121": (BLOCK, TALL, True, 133.8, 0.240, ORANGE),
    "210": (ROUND, SHORT, False, 48.3, 0.160, DUSTY_BLUE),
    "211": (ROUND, SHORT, True, 35.4, 0.150, DUSTY_BLUE),
    "220": (ROUND, TALL, False, 132.8, 0.235, GREEN),
    "221": (ROUND, TALL, True, 97.4, 0.205, GREEN),
}
PART_IDS = [f"part{d}_{c}" for d in DESIGNS for c in "ab"]


def _design(oid):
    return DESIGNS[oid[4:7]]


def _by_id(doc):
    return {o["id"]: o for o in doc["objects"]}


@pytest.fixture(scope="module")
def doc():
    return load_scene(FAMILY)


@pytest.fixture(scope="module")
def parts(doc):
    objects = _by_id(doc)
    return {oid: objects[oid] for oid in PART_IDS}


def _envelope(obj):
    geo = obj["geometry"]
    if geo["kind"] == "compound":
        lo, hi = compound_bounds(geo)
        return tuple(h - l for l, h in zip(lo, hi))
    if geo["kind"] == "box":
        return tuple(geo["size"])
    return (2 * geo["radius"], 2 * geo["radius"], geo["length"])


class TestTheSixteenParts:
    def test_exactly_these_sixteen_are_declared(self, doc):
        found = sorted(o["id"] for o in doc["objects"] if o["id"].startswith("part"))
        assert found == sorted(PART_IDS)

    @pytest.mark.parametrize("oid", PART_IDS)
    def test_shape_size_and_hole(self, parts, oid):
        shape, height, hole, _, _, _ = _design(oid)
        obj = parts[oid]
        geo = obj["geometry"]
        assert obj["semantic_class"] == shape
        assert _envelope(obj) == pytest.approx((SIDE, SIDE, height), abs=2e-5)
        if hole:
            assert geo["kind"] == "compound"
            assert len(geo["parts"]) == (4 if shape == BLOCK else 16)
            assert all(p["kind"] == "box" for p in geo["parts"])
        else:
            assert geo["kind"] == ("box" if shape == BLOCK else "cylinder")

    @pytest.mark.parametrize("oid", PART_IDS)
    def test_mass_friction_and_colour(self, parts, oid):
        shape, _, _, _, mass, colour = _design(oid)
        obj = parts[oid]
        phys = obj["physics"]
        assert phys["dynamic"] is True and phys["collision"] is True
        assert phys["mass"] == pytest.approx(mass)
        assert phys["friction"] == pytest.approx(
            [0.8 if shape == BLOCK else 0.75, 0.01, 0.001])
        assert tuple(obj["material"]["rgba"]) == pytest.approx(colour)
        assert obj["tracked"] is True
        assert {"pool", "part-set", "manipulable", "pickable"} <= set(obj["tags"])

    def test_masses_follow_the_owner_rule(self, doc):
        """1.25x to 2.25x the heaviest object the scene already had (red_cube),
        linear on volume, to the nearest 5 g."""
        old = [o for o in doc["objects"]
               if "part-set" not in (o.get("tags") or [])
               and (o.get("physics") or {}).get("dynamic")]
        heaviest = max(old, key=lambda o: o["physics"]["mass"])
        assert (heaviest["id"], heaviest["physics"]["mass"]) == ("red_cube", 0.12)
        lo_m, hi_m = 1.25 * 0.12, 2.25 * 0.12
        vols = [d[3] for d in DESIGNS.values()]
        for name, (_, _, _, vol, mass, _) in DESIGNS.items():
            linear = lo_m + (vol - min(vols)) / (max(vols) - min(vols)) * (hi_m - lo_m)
            assert abs(mass - linear) <= 0.0025 + 1e-9, name

    def test_colour_follows_the_family_never_the_hole(self, parts):
        for design in ("11", "12", "21", "22"):
            colours = {tuple(parts[f"part{design}{f}_{c}"]["material"]["rgba"])
                       for f in "01" for c in "ab"}
            assert len(colours) == 1, design

    def test_no_object_in_the_scene_refers_to_a_mesh(self, doc):
        """The printer files are not committed and may not be published."""
        assert not [o["id"] for o in doc["objects"]
                    if o["geometry"]["kind"] == "mesh"]
        for path in (FAMILY, BLACK, RANDOM):
            assert "input_3D_printer_files/digital" not in path.read_text()

    @pytest.mark.parametrize("oid", [i for i in PART_IDS if _design(i)[2]])
    def test_the_hole_is_open(self, parts, oid):
        """No part reaches inside the 25.4 mm bore, so a ray down the axis
        passes through."""
        mujoco = pytest.importorskip("mujoco")
        import numpy as np
        from scene_compiler import compile_scene
        obj = dict(parts[oid], pose={"position": [0, 0, 0.5]})
        model = mujoco.MjModel.from_xml_string(compile_scene({"objects": [obj]}))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        geomid = np.zeros(1, dtype=np.int32)

        def hit(dx, dy):
            return mujoco.mj_ray(model, data, np.array([dx, dy, 1.0]),
                                 np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)

        assert hit(0.0, 0.0) < 0
        for k in range(32):
            a = 2 * math.pi * k / 32
            c, s = math.cos(a), math.sin(a)
            # Clear out to r = 12.4 mm: the 25.4 mm bore of the real part.
            assert hit(0.0124 * c, 0.0124 * s) < 0, (oid, k)
            if _design(oid)[0] == ROUND:
                assert hit(0.0140 * c, 0.0140 * s) > 0, (oid, k)
        if _design(oid)[0] == BLOCK:
            # The issue's four walls make a SQUARE bore, open to its corners.
            for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                assert hit(sx * 0.0124, sy * 0.0124) < 0, oid
            for dx, dy in ((0.0131, 0.0), (-0.0131, 0.0), (0.0, 0.0131), (0.0, -0.0131)):
                assert hit(dx, dy) > 0, (oid, dx, dy)

    def test_ring_inertia_is_close_to_a_true_annulus(self, parts):
        """The sixteen segments overlap near the bore, so the compiled inertia
        is an approximation.  Hold it to 5 % of the ideal hollow cylinder."""
        mujoco = pytest.importorskip("mujoco")
        from scene_compiler import compile_scene
        for oid in ("part211_a", "part221_a"):
            obj = parts[oid]
            m, h = obj["physics"]["mass"], _design(oid)[1]
            model = mujoco.MjModel.from_xml_string(compile_scene({"objects": [obj]}))
            got = sorted(model.body_inertia[1])
            izz = m * (RADIUS ** 2 + BORE_R ** 2) / 2.0
            ixx = m * (3 * (RADIUS ** 2 + BORE_R ** 2) + h * h) / 12.0
            assert got == pytest.approx(sorted([ixx, ixx, izz]), rel=0.05), oid


def _xy_hull_area(geo_parts):
    """Area of the convex hull, seen from above, of the given box parts."""
    from scene_io import rotation_matrix
    pts = []
    for p in geo_parts:
        rot = rotation_matrix(p["quat_wxyz"])
        for sx in (-0.5, 0.5):
            for sy in (-0.5, 0.5):
                v = (sx * p["size"][0], sy * p["size"][1])
                pts.append(tuple(round(p["pos"][a] + rot[a][0] * v[0] + rot[a][1] * v[1], 5)
                                 for a in (0, 1)))
    pts = sorted(set(pts))

    def half(seq):
        out = []
        for p in seq:
            while len(out) > 1 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                    <= (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])):
                out.pop()
            out.append(p)
        return out[:-1]

    h = half(pts) + half(pts[::-1])
    return abs(sum(h[i - 1][0] * h[i][1] - h[i][0] * h[i - 1][1] for i in range(len(h)))) / 2


class TestEndFaces:
    """Only some parts of a holed object reach its end faces; the rest stop
    0.1 mm short.  With all sixteen ring segments on the floor each parked ring
    carried 64 contacts, and the no-slip pass (ADR-0005), whose cost grows with
    the square of the contacts on one body, took the physics step from 0.14 ms
    to 1.7 ms of the 2 ms budget."""

    HOLED = [i for i in PART_IDS if i[6] == "1"]

    @pytest.mark.parametrize("oid", HOLED)
    def test_the_standing_area_is_that_of_a_flat_face(self, parts, oid):
        all_parts = parts[oid]["geometry"]["parts"]
        height = _design(oid)[1]
        full = [p for p in all_parts if p["size"][2] == pytest.approx(height, abs=1e-9)]
        short = [p for p in all_parts if p not in full]
        assert len(full) == (2 if _design(oid)[0] == BLOCK else 8)
        assert all(p["size"][2] == pytest.approx(height - 0.0002, abs=1e-9) for p in short)
        assert all(p["pos"][2] == 0 for p in all_parts)      # same at both ends
        assert _xy_hull_area(full) == pytest.approx(_xy_hull_area(all_parts), rel=1e-3)

    @pytest.mark.parametrize("flipped", [False, True])
    @pytest.mark.parametrize("oid", ["part111_a", "part121_a", "part211_a", "part221_a"])
    def test_only_the_full_height_parts_touch_the_ground(self, parts, oid, flipped):
        mujoco = pytest.importorskip("mujoco")
        from scene_compiler import compile_scene
        half = _design(oid)[1] / 2
        obj = dict(parts[oid], pose={
            "position": [0, 0, half + 0.0005],
            "orientation_wxyz": [0, 1, 0, 0] if flipped else [1, 0, 0, 0]})
        xml = compile_scene({"objects": [obj]}).replace(
            "<worldbody>", '<worldbody><geom type="plane" size="1 1 0.1"/>')
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        for _ in range(1000):
            mujoco.mj_step(model, data)
        touching = {int(data.contact[i].geom2) for i in range(data.ncon)}
        assert len(touching) == (2 if _design(oid)[0] == BLOCK else 8)
        assert data.xpos[1][2] == pytest.approx(half, abs=0.0005)
        assert abs(data.xmat[1][8]) > math.cos(math.radians(0.5))


class TestParking:
    @pytest.mark.parametrize("oid", PART_IDS)
    def test_parked_flat_on_the_floor_out_of_reach(self, parts, oid):
        pose = parts[oid]["pose"]
        x, y, z = pose["position"]
        assert z == pytest.approx(_design(oid)[1] / 2, abs=1e-5)
        assert pose["orientation_wxyz"] == [1.0, 0.0, 0.0, 0.0]
        assert shoulder_distance(x, y, z) > 0.9

    def test_nothing_parked_overlaps_anything_else_parked(self, doc):
        parked = [o for o in doc["objects"] if "pool" in (o.get("tags") or [])]
        for i, a in enumerate(parked):
            for b in parked[i + 1:]:
                dx = abs(a["pose"]["position"][0] - b["pose"]["position"][0])
                dy = abs(a["pose"]["position"][1] - b["pose"]["position"][1])
                ea, eb = _envelope(a), _envelope(b)
                assert (dx > (ea[0] + eb[0]) / 2 + 0.02
                        or dy > (ea[1] + eb[1]) / 2 + 0.02), (a["id"], b["id"])

    def test_the_parts_are_pool_slots(self, doc):
        assert set(PART_IDS) <= set(pool_ids(doc))

    def test_the_scene_validates(self, doc):
        assert len(validate_scene(FAMILY, document=doc).objects) == len(doc["objects"])


# ── Colour variants ──────────────────────────────────────────────────────────

def _without_part_colours(doc):
    out = json.loads(json.dumps(doc))
    for key in ("name", "description", "color_randomization"):
        out.pop(key, None)
    for o in out["objects"]:
        if o["id"] in PART_IDS:
            o["material"].pop("rgba")
    return out


def _part_colours(doc):
    return {oid: tuple(_by_id(doc)[oid]["material"]["rgba"]) for oid in PART_IDS}


class TestBlackVariant:
    def test_every_part_is_matte_black(self):
        assert set(_part_colours(load_scene(BLACK)).values()) == {(0.05, 0.05, 0.05, 1.0)}

    def test_nothing_else_differs_from_the_family_scene(self, doc):
        black = load_scene(BLACK)
        assert black["name"] == "FWDCenterLabSivaPoolBlack"
        assert _without_part_colours(black) == _without_part_colours(doc)
        validate_scene(BLACK)


class TestRandomVariant:
    def _child(self, tmp_path, seed):
        path = tmp_path / f"seed{seed}.yaml"
        path.write_text(f"extends: {RANDOM}\nname: R{seed}\n"
                        f"color_randomization: {{seed: {seed}}}\n")
        return load_scene(path)

    def test_nothing_but_part_colour_differs(self, doc):
        rnd = load_scene(RANDOM)
        assert rnd["name"] == "FWDCenterLabSivaPoolRandom"
        assert _without_part_colours(rnd) == _without_part_colours(doc)
        validate_scene(RANDOM)

    def test_colours_come_from_the_palette(self):
        assert set(_part_colours(load_scene(RANDOM)).values()) <= PALETTE

    def test_one_file_always_means_one_world(self):
        """The physics server, RViz and the planner each load the file for
        themselves; they must agree on every colour."""
        assert load_scene(RANDOM) == load_scene(RANDOM)
        direct = _part_colours(load_scene(RANDOM))
        validated = validate_scene(RANDOM)
        assert direct == {o.id: o.material.rgba for o in validated.objects
                          if o.id in PART_IDS}

    def test_the_seed_changes_the_colours_and_is_kept_in_the_document(self, tmp_path):
        a, b = self._child(tmp_path, 1), self._child(tmp_path, 2)
        assert _part_colours(a) != _part_colours(b)
        assert a["color_randomization"]["seed"] == 1
        assert a["color_randomization"]["tag"] == "part-set"

    def test_the_assignment_is_pinned_across_machines(self, tmp_path):
        """A recorded seed must give the same colours on any Python."""
        import hashlib
        doc = self._child(tmp_path, 12345)
        palette = doc["color_randomization"]["palette"]
        for oid, got in _part_colours(doc).items():
            digest = hashlib.sha256(f"12345:{oid}".encode()).digest()
            assert got == pytest.approx(
                tuple(palette[int.from_bytes(digest[:8], "big") % len(palette)])), oid

    def test_colour_does_not_follow_shape_height_or_hole(self, tmp_path):
        """Over many seeds every design takes every colour, so perception
        cannot use colour as a shortcut for geometry."""
        seen = {d: set() for d in DESIGNS}
        for seed in range(60):
            for oid, colour in _part_colours(self._child(tmp_path, seed)).items():
                seen[oid[4:7]].add(colour)
        assert all(v == PALETTE for v in seen.values())

    def test_only_tagged_objects_are_recoloured(self, doc):
        rnd = _by_id(load_scene(RANDOM))
        for oid, obj in _by_id(doc).items():
            if oid not in PART_IDS:
                assert rnd[oid] == obj, oid

    @pytest.mark.parametrize("block, needle", [
        ("{tag: part-set, palette: [[1,0,0,1],[0,1,0,1]]}", "seed"),
        ("{tag: part-set, seed: -1, palette: [[1,0,0,1],[0,1,0,1]]}", "seed"),
        ("{tag: part-set, seed: 1.5, palette: [[1,0,0,1],[0,1,0,1]]}", "seed"),
        ("{tag: part-set, seed: 1, palette: [[1,0,0,1]]}", "palette"),
        ("{tag: part-set, seed: 1, palette: [[1,0,0],[0,1,0,1]]}", "palette"),
        ("{tag: nothing-has-this, seed: 1, palette: [[1,0,0,1],[0,1,0,1]]}", "tag"),
        ("{tag: part-set, seed: 1, palette: [[1,0,0,1],[0,1,0,1]], mode: x}", "mode"),
    ])
    def test_a_bad_randomisation_block_fails(self, tmp_path, block, needle):
        path = tmp_path / "bad.yaml"
        path.write_text(f"extends: {FAMILY}\ncolor_randomization: {block}\n")
        with pytest.raises((SceneLoadError, SceneValidationError), match=needle):
            load_scene(path)


class TestPerRunSceneGenerator:
    def test_a_given_seed_writes_a_child_that_loads_with_that_seed(self, tmp_path):
        from make_part_colour_scene import write_scene
        path = write_scene(seed=777, out_dir=tmp_path)
        doc = load_scene(path)
        assert doc["color_randomization"]["seed"] == 777
        assert doc["name"] == "FWDCenterLabSivaPoolRandom"
        assert set(_part_colours(doc).values()) <= PALETTE
        assert write_scene(seed=777, out_dir=tmp_path).read_text() == path.read_text()

    def test_with_no_seed_each_run_draws_its_own(self, tmp_path):
        from make_part_colour_scene import write_scene
        seeds = {load_scene(write_scene(out_dir=tmp_path))["color_randomization"]["seed"]
                 for _ in range(4)}
        assert len(seeds) == 4

    def test_the_default_directory_is_ignored_by_git_and_visible_to_docker(self):
        import make_part_colour_scene as gen
        assert gen.DEFAULT_OUT_DIR == _SCENES / "generated"
        assert "scenes/generated/" in (_ROOT / ".gitignore").read_text()


class TestRunManifest:
    def test_the_server_records_the_seed_and_the_colours(self):
        pytest.importorskip("mujoco")
        pytest.importorskip("websockets")
        from server import ReachyMujocoServer
        srv = ReachyMujocoServer(
            str(_ROOT / "native_mujoco" / "model" / "reachy_1_2.xml"),
            "127.0.0.1", 0, scene_path=str(RANDOM))
        manifest = srv._build_recorder_manifest()
        record = manifest["color_randomization"]
        doc = load_scene(RANDOM)
        assert record["seed"] == doc["color_randomization"]["seed"]
        assert record["colors"] == {k: list(v) for k, v in _part_colours(doc).items()}
        json.dumps(manifest)

    def test_a_scene_without_randomisation_records_null(self):
        pytest.importorskip("mujoco")
        pytest.importorskip("websockets")
        from server import ReachyMujocoServer
        srv = ReachyMujocoServer(
            str(_ROOT / "native_mujoco" / "model" / "reachy_1_2.xml"),
            "127.0.0.1", 0, scene_path=str(FAMILY))
        assert srv._build_recorder_manifest()["color_randomization"] is None


# ── The exit gate: every part rests stably on a board cell ──────────────────

_SIM_SECONDS = 2.0


@pytest.fixture(scope="module")
def server_world(doc):
    """The server's own world: compiled by its constructor, stepped by its
    SimState exactly as `_sim_thread` steps it (without sockets or rendering)."""
    pytest.importorskip("mujoco")
    pytest.importorskip("websockets")
    from server import ReachyMujocoServer
    return ReachyMujocoServer(
        str(_ROOT / "native_mujoco" / "model" / "reachy_1_2.xml"),
        "127.0.0.1", 0, scene_path=str(FAMILY))


@pytest.mark.parametrize("oid", PART_IDS)
def test_each_part_rests_stably_on_a_board_cell(server_world, doc, oid):
    import mujoco
    from server import SimState
    from scene_compiler import interactive_specs, tracked_object_ids
    model = server_world._model
    sim = SimState(model, tracked_ids=tracked_object_ids(doc),
                   interactive_specs=interactive_specs(doc), scene_doc=doc)
    cell = cells_from_scene(doc)["r2c2"]
    sim.submit_place({"request_id": "t", "object_id": oid, "cell": "r2c2"})
    sim.apply_pending()
    result = sim.drain_place_results()[0]
    assert result["accepted"], result
    dt = model.opt.timestep
    for _ in range(int(round(_SIM_SECONDS / dt))):
        sim.apply_pending()
        sim.control_step(dt)
        mujoco.mj_step(model, sim.data)

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, oid)
    pos = sim.data.xpos[bid]
    half = _design(oid)[1] / 2
    # The face that carries it.  The grid cell's top_z is a 1 mm decal above
    # this that does not collide.
    board = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table_top")
    g = int(model.body_geomadr[board])
    board_top = float(sim.data.geom_xpos[g][2] + model.geom_size[g][2])
    assert board_top == pytest.approx(0.740, abs=1e-6)
    # Not slid, not sunk, not hovering.
    assert math.hypot(pos[0] - cell.x, pos[1] - cell.y) < 0.002
    assert pos[2] == pytest.approx(board_top + half, abs=0.0005)
    # Upright: the object's z axis within one degree of the world's.
    assert sim.data.xmat[bid][8] > math.cos(math.radians(1.0))
    # Still.
    adr = model.jnt_dofadr[model.body_jntadr[bid]]
    assert max(abs(float(v)) for v in sim.data.qvel[adr:adr + 3]) < 0.002
    assert max(abs(float(v)) for v in sim.data.qvel[adr + 3:adr + 6]) < 0.05
    # Actually carried by the board, not caught on the arm.
    others = set()
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        pair = [int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])]
        if bid in pair:
            other = pair[1 - pair.index(bid)]
            others.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other))
    assert others == {"table_top"}, others


def test_reshape_refuses_a_holed_part_through_the_server_path(server_world, doc):
    from server import SimState
    sim = SimState(server_world._model, scene_doc=doc)
    sim.submit_place({"request_id": "r", "object_id": "part221_a", "cell": "r2c2",
                      "reshape": {"rgba": [1, 0, 0, 1]}})
    sim.apply_pending()
    result = sim.drain_place_results()[0]
    assert result["accepted"] is False and "compound" in result["error"]
    with pytest.raises(PlacementError, match="compound"):
        sim.placer.reshape("part221_a", mass=0.5)
