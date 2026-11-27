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


def _ray_down(obj):
    """hit(x, y): does a vertical ray through the object's own (x, y) touch it?"""
    mujoco = pytest.importorskip("mujoco")
    import numpy as np
    from scene_compiler import compile_scene
    model = mujoco.MjModel.from_xml_string(
        compile_scene({"objects": [dict(obj, pose={"position": [0, 0, 0.5]})]}))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geomid = np.zeros(1, dtype=np.int32)
    return lambda x, y: mujoco.mj_ray(model, data, np.array([x, y, 1.0]),
                                      np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid) >= 0


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
            # A block is four walls plus the ring that rounds its bore (D2).
            assert len(geo["parts"]) == (20 if shape == BLOCK else 16)
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
    def test_the_hole_is_a_round_16_sided_bore(self, parts, oid):
        """Every holed part, block or cylinder, has the same opening: a regular
        16-gon of inradius 12.7 mm, clear right through, and nothing else
        (owner decision D2, 2026-11-27: round, like the printed parts)."""
        hit = _ray_down(parts[oid])
        assert not hit(0.0, 0.0)
        step = 2 * math.pi / 16
        vertex = BORE_R / math.cos(step / 2)             # 12.949 mm
        for k in range(16):
            for angle, edge in ((k * step, BORE_R), ((k + 0.5) * step, vertex)):
                c, s_ = math.cos(angle), math.sin(angle)
                # 10 um either side of the polygon: on a flat, and at a corner.
                assert not hit((edge - 1e-5) * c, (edge - 1e-5) * s_), (oid, k, angle)
                assert hit((edge + 1e-5) * c, (edge + 1e-5) * s_), (oid, k, angle)

    @pytest.mark.parametrize("oid", ["part111_a", "part121_a"])
    def test_a_holed_block_is_solid_everywhere_outside_its_bore(self, parts, oid):
        """No slit between walls and ring, and no square corner left open."""
        hit = _ray_down(parts[oid])
        half, step = SIDE / 2, 2 * math.pi / 16

        def inside_bore(x, y, grow):
            return all(x * math.cos(k * step) + y * math.sin(k * step) <= BORE_R + grow
                       for k in range(16))

        n = 121
        for i in range(n):
            for j in range(n):
                x = -half + 1e-5 + i * (SIDE - 2e-5) / (n - 1)
                y = -half + 1e-5 + j * (SIDE - 2e-5) / (n - 1)
                if inside_bore(x, y, -1e-5):
                    assert not hit(x, y), (oid, x, y)
                elif not inside_bore(x, y, 1e-5):
                    assert hit(x, y), (oid, x, y)
        # The old square opening's corners, explicitly.
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            assert hit(sx * 0.01265, sy * 0.01265), oid

    @pytest.mark.parametrize("oid", ["part111_a", "part121_a"])
    def test_the_bore_ring_of_a_block(self, parts, oid):
        """Sixteen boxes from r = 12.7 to 18.0 mm (past the square opening's
        corners at 17.96), all of them short of both end faces."""
        height = _design(oid)[1]
        ring = parts[oid]["geometry"]["parts"][4:]
        assert len(ring) == 16
        for k, part in enumerate(ring):
            radius = math.hypot(part["pos"][0], part["pos"][1])
            assert radius - part["size"][0] / 2 == pytest.approx(0.0127, abs=2e-7), k
            assert radius + part["size"][0] / 2 == pytest.approx(0.0180, abs=2e-7), k
            assert math.atan2(part["pos"][1], part["pos"][0]) % (2 * math.pi) == pytest.approx(
                k * 2 * math.pi / 16, abs=1e-5), k
            assert part["size"][2] == pytest.approx(height - 0.0002, abs=1e-9), k
            assert part["pos"][2] == 0
        assert 0.0180 > BORE_R * math.sqrt(2)

    def test_block_inertia_against_a_round_bored_block(self, parts):
        """Inertia comes from the parts at one density (as before D2).  The ring
        lies largely inside the walls and counts twice there, which draws mass
        inward: the compiled inertia is LOW, by 3.7 to 9.5 % (held to 11)."""
        mujoco = pytest.importorskip("mujoco")
        from scene_compiler import compile_scene
        for oid in ("part111_a", "part121_a"):
            obj = parts[oid]
            m, h = obj["physics"]["mass"], _design(oid)[1]
            model = mujoco.MjModel.from_xml_string(compile_scene({"objects": [obj]}))
            got = sorted(model.body_inertia[1])
            # A square prism with a round 25.4 mm bore, exactly.
            area = SIDE ** 2 - math.pi * BORE_R ** 2
            polar = SIDE ** 4 / 6 - math.pi * BORE_R ** 4 / 2
            izz = m * polar / area
            ixx = m * (polar / area / 2 + h * h / 12)
            ratios = [g / w for g, w in zip(got, sorted([ixx, ixx, izz]))]
            assert all(0.89 < r < 1.0 for r in ratios), (oid, ratios)

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
        """Under scenes/, the one scene directory compose mounts into the
        container.  git itself is asked: outputs/ is NOT ignored."""
        import shutil
        import subprocess
        import make_part_colour_scene as gen
        assert gen.DEFAULT_OUT_DIR == _SCENES / "generated"
        assert "./scenes:/opt/scenes" in (_ROOT / "docker-compose.yml").read_text()
        if shutil.which("git") is None or not (_ROOT / ".git").exists():
            pytest.skip("no git checkout to ask")
        probe = gen.DEFAULT_OUT_DIR / "FWDCenterLabSivaPoolRandom_seed1.yaml"
        assert subprocess.run(["git", "-C", str(_ROOT), "check-ignore", "-q",
                               str(probe)]).returncode == 0

    def test_the_seed_is_written_in_the_scene_file(self, tmp_path):
        from make_part_colour_scene import write_scene
        assert "color_randomization: {seed: 424242}" in write_scene(424242, tmp_path).read_text()

    def test_a_dataset_runs_manifest_records_the_seed_and_colours(self, tmp_path):
        mujoco = pytest.importorskip("mujoco")
        from calibration import synthetic_defaults
        from dataset import DatasetGenerator
        from make_part_colour_scene import write_scene
        from objects import build_scene_model_xml
        for path, seed in ((write_scene(31337, tmp_path), 31337), (FAMILY, None)):
            doc = load_scene(path)
            model = mujoco.MjModel.from_xml_string(build_scene_model_xml(doc))
            gen = DatasetGenerator(model, doc, synthetic_defaults(64, 48),
                                   width=64, height=48, seed=5)
            try:
                manifest = gen.generate(tmp_path / f"set-{seed}", 1)
            finally:
                gen.close()
            record = manifest["color_randomization"]
            assert manifest["seed"] == 5                   # placement seed, unchanged
            if seed is None:
                assert record is None
            else:
                assert record["seed"] == seed
                assert record["colors"] == {k: list(v) for k, v in _part_colours(doc).items()}
            on_disk = json.loads((tmp_path / f"set-{seed}" / "manifest.json").read_text())
            assert on_disk["color_randomization"] == record


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


# ── Step-cost guard (owner decision D3, 2026-11-27) ─────────────────────────
#
# WHY COUNT CONTACTS INSTEAD OF TIMING A STEP.  A timing test fails on a busy
# machine and passes on a fast one.  The cost that appeared was a contact count:
# the no-slip pass (ADR-0005) grows with the square of the contacts on ONE body,
# and a holed cylinder with all sixteen faces down (64 contacts) took the step
# from 0.14 to 1.7 ms of its 2.0 ms budget.  The count is deterministic.
#
# THE NUMBERS are the accepted design (D1), measured on the board with MuJoCo
# 3.11.0 and the same in 36 of 36 placements (9 cells x 4 yaws):
#   solid block     4   the four corners of its one face
#   holed block     8   two full-height walls x 4 corners; the other two walls
#                       and all sixteen bore-ring boxes stop 0.1 mm short
#   solid cylinder  5   MuJoCo's cylinder-on-box contact set
#   holed cylinder 32   eight full-height ring segments x 4 corners
# Raising one of these is a step-cost change: re-run
# docs/reviews/2026-11-27-part-set-212/step_cost.py and say what it did.
MAX_CONTACTS_AT_REST = {(BLOCK, False): 4, (BLOCK, True): 8,
                        (ROUND, False): 5, (ROUND, True): 32}


@pytest.mark.parametrize("oid", PART_IDS)
def test_resting_on_a_cell_a_part_makes_no_more_contacts_than_the_design(server_world, doc, oid):
    import mujoco
    from server import SimState
    from scene_compiler import interactive_specs, tracked_object_ids
    model = server_world._model
    sim = SimState(model, tracked_ids=tracked_object_ids(doc),
                   interactive_specs=interactive_specs(doc), scene_doc=doc)
    sim.submit_place({"object_id": oid, "cell": "r2c2"})
    sim.apply_pending()
    assert sim.drain_place_results()[0]["accepted"]
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, oid)
    steps = int(round(_SIM_SECONDS / model.opt.timestep))
    most = 0
    for step in range(steps):
        sim.apply_pending()
        sim.control_step(model.opt.timestep)
        mujoco.mj_step(model, sim.data)
        if step >= steps - 250:                      # the last half second
            active = sum(
                1 for i in range(sim.data.ncon)
                if sim.data.contact[i].efc_address >= 0
                and bid in (model.geom_bodyid[sim.data.contact[i].geom1],
                            model.geom_bodyid[sim.data.contact[i].geom2]))
            most = max(most, active)
    assert sim.data.xmat[bid][8] > math.cos(math.radians(1.0)), "not upright"
    shape, _, hole, *_ = _design(oid)
    assert 0 < most <= MAX_CONTACTS_AT_REST[(shape, hole)], most


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
