"""P1b (#182): the grader accepts PerceivedScene documents (perceived_scene.v1) next to viewer bundles.

The documents are built here from the pair_ok fixture's own measurements, the way the tabletop builder would make them.
"""
import copy
import hashlib
import json
import math
import os
import pathlib
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))
sys.path.insert(0, os.path.join(_HERE, "../../scripts"))

import perception_grade as pg  # noqa: E402
import perceived_scene as ps  # noqa: E402

_FIX = pathlib.Path(_HERE).resolve().parents[1] / "tests" / "fixtures" / "perception_grade"
PROFILE = "771392183ddc6c62"
SHA = "a" * 64
RIG = ps.RigIdentity(1, "sim", "FWDCenterLabSivaPool", "october-2026", PROFILE, "ZERO", PROFILE, 0,
                     "FWDCenterLabSivaPool", "0" * 64, "rev-1")
PIPE = ps.PipelineIdentity("b" * 64, ("cube", "cylinder", "empty"), "sgbm-bidir-1px/inner60-median", "none", "none",
                           "7786a65", False)
SRC = ps.SourcePair("fixture-pair-1", SHA, SHA, SHA, "2026-10-21T03:00:19Z", (0.0, 64.0, 0.0))
INF = (math.inf,) * 3

# the pair_ok measurements: blue_cylinder on r3c3, red_cube on r1c2, soda_can (no class) on r2c2
CYLINDER = (0.3497, -0.143, -0.3069)
CUBE = (0.0775, -0.0054, -0.3449)
SODA = (0.2084, -0.0081, -0.2813)
CELLS = ps.CELL_NAMES


@pytest.fixture()
def truth():
    return json.loads((_FIX / "truth_before.json").read_text())


@pytest.fixture()
def pair():
    return {k: json.loads((_FIX / "pair_ok" / f"{k}.json").read_text()) for k in ("pair", "analysis", "depth")}


def obj(kind, surface, centre=None, bounds=INF, provisional=True, source=SRC):
    return ps.PerceivedObject(kind, 0.9, tuple(centre or surface), tuple(surface), bounds, 0.95, provisional, None, None,
                              None, "unknown", None, 500, 1.0, 0.004, source, PROFILE)


def scene(objects, states=None, off_board=(), board=True, rig=RIG):
    """A scene; `states` maps cell name -> (state, object_index) and every other cell is unknown."""
    cells = ()
    if board:
        states = states or {}
        cells = tuple(ps.CellReport(n, *states.get(n, ("unknown", None))) for n in CELLS)
    objects = list(objects)
    for name, (state, index) in (states or {}).items():
        if index is not None:     # the object names its cell, as the builder's output does
            objects[index] = ps.PerceivedObject(**dict(vars(objects[index]), cell=name))
    bd = ps.BoardPose((0.2258, 0.0, -0.4095), (1.0, 0.0, 0.0, 0.0), (0.003,) * 3, "twin", 0, 0.0, 0.0) if board else None
    return ps.PerceivedScene(ps.SCHEMA_VERSION, "0123456789abcdef0123456789abcdef", "2026-10-21T03:05:00Z", rig, PIPE,
                             bd, cells, tuple(objects), tuple(off_board), ())


def base_scene(**kw):
    objects = [obj("cylinder", CYLINDER), obj("cube", CUBE), obj("cylinder", SODA)]
    return scene(objects, kw.pop("states", {"r3c3": ("cylinder", 0), "r1c2": ("cube", 1), "r2c2": ("cylinder", 2)}), **kw)


# --- the profile table ----------------------------------------------------------------------

def test_the_known_profile_id_is_the_hash_of_its_identity_like_the_viewers():
    for pid, profile in pg.KNOWN_SIM_PROFILES.items():
        assert hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()[:16] == pid


def test_the_known_profile_is_the_fixture_pairs_lens(pair):
    known = pg.KNOWN_SIM_PROFILES[PROFILE]
    for key in pg._PROFILE_KEYS:
        assert known[key] == pair["depth"]["profile"][key], key


# --- a scene is graded like the pair it came from -------------------------------------------

def test_an_uncorrected_scene_gives_the_same_matches_and_errors_as_the_pair(truth, pair):
    zipped = pg.grade_pair(truth, pair)
    scened = pg.grade_scene(truth, base_scene())
    assert {m["object_id"]: m["correct"] for m in scened["matches"]} == {"red_cube": True, "blue_cylinder": True}
    assert [h["object_id"] for h in scened["oov_hits"]] == ["soda_can"] and not scened["false_positives"]
    by_id = {m["object_id"]: m["err"] for m in zipped["matches"]}
    for m in scened["matches"]:
        for key, value in by_id[m["object_id"]].items():
            assert m["err"][key] == pytest.approx(value), (m["object_id"], key)
    assert scened["in_view"] == zipped["in_view"] and scened["neck_deg"] == zipped["neck_deg"]
    assert scened["perceived"]["surface_correction_id"] == "none" and scened["profile_id"] == PROFILE


def test_the_gate_is_on_the_surface_point_and_the_errors_are_on_the_centre(truth):
    plain = pg.grade_scene(truth, base_scene())
    shifted = [obj("cylinder", CYLINDER, centre=(CYLINDER[0], CYLINDER[1], CYLINDER[2] - 0.04)),
               obj("cube", CUBE, centre=(CUBE[0] + 0.01, CUBE[1], CUBE[2])), obj("cylinder", SODA)]
    corrected = pg.grade_scene(truth, scene(shifted, {"r3c3": ("cylinder", 0), "r1c2": ("cube", 1),
                                                      "r2c2": ("cylinder", 2)}))
    assert len(corrected["matches"]) == 2 and not corrected["false_positives"]
    assert not [m for m in corrected["misses"] if m["on_board"]]
    old = {m["object_id"]: m["err"] for m in plain["matches"]}
    new = {m["object_id"]: m["err"] for m in corrected["matches"]}
    for oid in old:
        assert new[oid]["gate_mm"] == pytest.approx(old[oid]["gate_mm"])          # the gate does not move
    assert new["blue_cylinder"]["z_mm"] == pytest.approx(old["blue_cylinder"]["z_mm"] - 40.0)
    assert new["red_cube"]["x_mm"] == pytest.approx(old["red_cube"]["x_mm"] + 10.0)
    assert new["blue_cylinder"]["centre_mm"] != pytest.approx(old["blue_cylinder"]["centre_mm"])


def test_a_provisional_bounds_object_is_graded_and_flagged(truth):
    bounded = obj("cube", CUBE, bounds=(0.0233, 0.0233, 0.0313))
    s = pg.grade_scene(truth, scene([obj("cylinder", CYLINDER, bounds=(0.01,) * 3, provisional=False), bounded],
                                    {"r3c3": ("cylinder", 0), "r1c2": ("cube", 1)}))
    flags = {m["object_id"]: m["bounds_provisional"] for m in s["matches"]}
    assert flags == {"blue_cylinder": False, "red_cube": True}    # infinite or finite, a provisional object still counts
    inf = pg.grade_scene(truth, base_scene())
    assert all(m["bounds_provisional"] for m in inf["matches"]) and len(inf["matches"]) == 2


def test_a_wrong_class_a_miss_and_a_false_positive(truth):
    wrong = pg.grade_scene(truth, scene([obj("cylinder", CUBE), obj("cylinder", CYLINDER)]))
    assert {m["object_id"]: m["correct"] for m in wrong["matches"]} == {"red_cube": False, "blue_cylinder": True}
    missing = pg.grade_scene(truth, scene([obj("cylinder", CYLINDER)]))
    miss = next(m for m in missing["misses"] if m["object_id"] == "red_cube" and m["on_board"])
    assert miss["no_boxes"] and "no object near it" in pg._miss_line(miss)
    stray = pg.grade_scene(truth, scene([obj("cylinder", CYLINDER), obj("cube", CUBE), obj("cube", (0.45, 0.2, -0.3))]))
    assert [f["label"] for f in stray["false_positives"]] == ["cube"] and stray["false_positives"][0]["nearest_mm"] > 40


def test_an_off_board_object_can_match_a_floor_piece_but_stays_out_of_the_headline(truth):
    s = pg.grade_scene(truth, scene([obj("cylinder", CYLINDER)], off_board=[obj("cube", CUBE)]))
    assert s["n_detections"] == 2
    assert {m["object_id"] for m in s["matches"]} == {"blue_cylinder", "red_cube"}


# --- cells ----------------------------------------------------------------------------------

def test_cell_outcomes_come_from_the_scenes_cell_reports_including_unknown(truth):
    states = {"r3c3": ("cylinder", 0), "r1c2": ("unknown", None), "r1c1": ("empty", None), "r2c3": ("cube", 1)}
    s = pg.grade_scene(truth, scene([obj("cylinder", CYLINDER), obj("cube", SODA)], states))
    cells = s["cells"]
    assert cells["r3c3"] == {"truth": "cylinder", "reported": "cylinder", "outcome": "correct"}
    assert cells["r1c2"] == {"truth": "cube", "reported": "unknown", "outcome": "unknown"}     # unknown is not wrong
    assert cells["r1c1"] == {"truth": "empty", "reported": "empty", "outcome": "correct"}
    assert cells["r2c3"] == {"truth": "empty", "reported": "cube", "outcome": "wrong"}         # an object on an empty cell
    assert cells["r2c1"]["outcome"] == "unknown" and cells["r2c2"]["outcome"] == "oov"
    assert [e["cell"] for e in s["empty_dets"]] == ["r1c1"] and s["empty_dets"][0]["correct"] is True


def test_an_empty_report_over_an_occupied_cell_is_wrong_and_a_scene_without_a_board_is_all_unknown(truth):
    s = pg.grade_scene(truth, scene([obj("cylinder", CYLINDER)], {"r1c2": ("empty", None)}))
    assert s["cells"]["r1c2"]["outcome"] == "wrong" and s["empty_dets"][0]["correct"] is False
    none = pg.grade_scene(truth, scene([obj("cylinder", CYLINDER), obj("cube", CUBE)], board=False))
    assert {c["outcome"] for c in none["cells"].values()} <= {"unknown", "oov"} and not none["empty_dets"]
    summary = pg.summarize([s])
    assert summary["empty"]["false_positive"] == 1 and summary["cell_outcomes"]["r1c2"] == {"wrong": 1}


# --- refusals -------------------------------------------------------------------------------

def test_a_scene_for_another_scene_or_profile_or_the_robot_is_refused(truth):
    other = ps.RigIdentity(**dict(vars(RIG), scene_sha256="1" * 64))
    with pytest.raises(pg.GradeError, match="scene_sha256"):
        pg.grade_scene(truth, base_scene(rig=other))
    other = ps.RigIdentity(**dict(vars(RIG), scene_revision="rev-2"))
    with pytest.raises(pg.GradeError, match="scene_revision"):
        pg.grade_scene(truth, base_scene(rig=other))
    unknown = ps.RigIdentity(**dict(vars(RIG), intrinsics_sha256="deadbeef"))
    with pytest.raises(pg.GradeError, match="unknown sim stereo profile"):
        pg.grade_scene(truth, base_scene(rig=unknown))
    robot = ps.RigIdentity(1, "robot", "FWDCenterLabMCC", "october-2026", "abc@zoom=OUT", "OUT", "c" * 64, 0)
    with pytest.raises(pg.GradeError, match="only a sim"):
        pg.grade_scene(truth, scene([], board=False, rig=robot))


def test_an_empty_scene_needs_the_neck_from_the_manifest_and_the_two_must_agree(truth):
    empty = scene([])
    with pytest.raises(pg.GradeError, match="neck_deg"):
        pg.grade_scene(truth, empty)
    graded = pg.grade_scene(truth, empty, neck_deg=[0.0, 64.0, 0.0])
    assert not graded["matches"]
    assert {m["object_id"] for m in graded["misses"] if m["on_board"]} == {"blue_cylinder", "red_cube"}
    with pytest.raises(pg.GradeError, match="differs"):
        pg.grade_scene(truth, base_scene(), neck_deg=[0.0, 50.0, 0.0])
    with pytest.raises(pg.GradeError, match="three finite"):
        pg.grade_scene(truth, empty, neck_deg=[0.0, math.nan, 0.0])


def test_objects_from_different_head_poses_are_refused(truth):
    other = ps.SourcePair("p2", SHA, SHA, SHA, "2026-10-21T03:00:19Z", (0.0, 50.0, 0.0))
    with pytest.raises(pg.GradeError, match="different head poses"):
        pg.grade_scene(truth, scene([obj("cylinder", CYLINDER), obj("cube", CUBE, source=other)]))


# --- the manifest ---------------------------------------------------------------------------

def _run(tmp_path, entries, name="L"):
    (tmp_path / name).mkdir(exist_ok=True)
    for f in ("truth_before.json", "truth_after.json"):
        (tmp_path / name / f).write_text((_FIX / "truth_before.json").read_text())
    manifest = tmp_path / "layouts.json"
    manifest.write_text(json.dumps({"layouts": [{"name": name, "truth_before": f"{name}/truth_before.json",
                                                 "truth_after": f"{name}/truth_after.json", "pairs": entries}]}))
    return manifest


def test_a_manifest_of_scene_documents_is_graded_end_to_end(tmp_path):
    (tmp_path / "L").mkdir()
    (tmp_path / "L" / "one.json").write_text(ps.dumps(base_scene()))
    (tmp_path / "L" / "empty.json").write_text(ps.dumps(scene([])))
    manifest = _run(tmp_path, ["L/one.json", {"scene": "L/empty.json", "neck_deg": [0.0, 64.0, 0.0]}])
    assert pg.main(["--manifest", str(manifest), "--out-dir", str(tmp_path / "out"), "--bias-loo"]) == 0
    result = json.loads((tmp_path / "out" / "result.json").read_text())
    assert result["overall"]["pairs"] == 2 and not result["rejected_pairs"]
    assert result["config"]["input_form"] == "perceived_scene.v1" and result["config"]["surface_corrections"] == ["none"]
    assert result["inputs"][0]["pairs"][0]["sha256"] == hashlib.sha256((tmp_path / "L" / "one.json").read_bytes()).hexdigest()
    assert result["overall"]["error_3d_mm"]["x_mm"]["n"] == 2 and "measured_bias_v1" in result
    report = (tmp_path / "out" / "report.md").read_text()
    for needle in ("PerceivedScene documents", "Signed error", "measured-bias.v1"):
        assert needle in report


def test_a_layout_may_not_mix_bundles_and_scenes_and_a_bad_document_is_rejected_not_graded(tmp_path):
    (tmp_path / "L").mkdir()
    (tmp_path / "L" / "one.json").write_text(ps.dumps(base_scene()))
    (tmp_path / "L" / "bad.json").write_text(json.dumps({"schema_version": "perceived_scene.v1"}))
    (tmp_path / "L" / "p.zip").write_bytes(b"x")
    mixed = _run(tmp_path, ["L/one.json", "L/p.zip"])
    assert pg.main(["--manifest", str(mixed), "--out-dir", str(tmp_path / "o1")]) == 1
    manifest = _run(tmp_path, ["L/one.json", "L/bad.json"])
    assert pg.main(["--manifest", str(manifest), "--out-dir", str(tmp_path / "o2")]) == 0
    result = json.loads((tmp_path / "o2" / "result.json").read_text())
    assert result["overall"]["pairs"] == 1 and "missing field" in result["rejected_pairs"][0]["reason"]
    bad_entry = _run(tmp_path, [{"scene": "L/one.json", "extra": 1}])
    assert pg.main(["--manifest", str(bad_entry), "--out-dir", str(tmp_path / "o3")]) == 1


# --- signed errors and measured-bias.v1 -----------------------------------------------------

def test_signed_errors_per_axis_are_in_every_row_and_in_the_blocks(truth, pair):
    for m in pg.grade_pair(truth, pair)["matches"]:
        e = m["err"]
        assert {"x_mm", "y_mm", "z_mm", "toward_camera"} <= set(e)
        assert math.sqrt(e["x_mm"] ** 2 + e["y_mm"] ** 2) == pytest.approx(e["horizontal_mm"])
        assert math.sqrt(e["x_mm"] ** 2 + e["y_mm"] ** 2 + e["z_mm"] ** 2) == pytest.approx(e["centre_mm"])
    # +10 mm in x on the measurement is +10 mm in x_mm, not an absolute value
    pair["depth"]["measurements"][1]["xyz_torso_m"][0] += 0.010
    cube = next(m for m in pg.grade_pair(truth, pair)["matches"] if m["object_id"] == "red_cube")
    before = next(m for m in pg.grade_pair(truth, {**pair, "depth": json.loads((_FIX / "pair_ok" / "depth.json").read_text())}
                                           )["matches"] if m["object_id"] == "red_cube")
    assert cube["err"]["x_mm"] - before["err"]["x_mm"] == pytest.approx(10.0)
    s = pg.summarize([pg.grade_pair(truth, pair)])
    block = s["error_3d_mm"]
    assert block["x_mm"]["n"] == 2 and block["z_mm"]["mean"] is not None and block["along_ray_signed_mm"]["n"] == 2
    assert set(s["error_3d_by_distance_mm"]) and all("z_mm" in b for b in s["error_3d_by_distance_mm"].values())


def _layout(name, rows):
    """A layout with matched rows: (kind, object id, signed delta (x, y, z) in mm); the ray points straight down."""
    matches = [{"kind": k, "object_id": oid, "on_board": True, "distance_m": 0.5, "cell": "r1c1", "label": k,
                "correct": True, "via": "3d",
                "err": {"x_mm": d[0], "y_mm": d[1], "z_mm": d[2], "toward_camera": [0.0, 0.0, 1.0],
                        "centre_mm": math.sqrt(sum(v * v for v in d)), "horizontal_mm": math.hypot(d[0], d[1]),
                        "along_ray_mm": -d[2], "gate_mm": 0.0}} for k, oid, d in rows]
    return {"name": name, "accepted": True, "pairs": [{"matches": matches}]}


def test_measured_bias_is_fitted_leave_one_layout_out_and_removes_a_constant_bias():
    layouts = [_layout("L1", [("cube", "a", (-10.0, -4.0, 20.0)), ("cylinder", "b", (-5.0, -2.0, 18.0))]),
               _layout("L2", [("cube", "c", (-10.0, -4.0, 20.0)), ("cylinder", "d", (-5.0, -2.0, 18.0))]),
               _layout("L3", [("cube", "e", (-10.0, -4.0, 20.0)), ("cylinder", "f", (-5.0, -2.0, 18.0))])]
    b = pg.bias_loo(layouts)
    assert b["scheme"] == "measured-bias.v1" and b["n_rows"] == 6 and b["n_graded"] == 6 and b["n_unfitted"] == 0
    assert b["n_distinct"] == 6
    assert b["after"]["centre_mm"]["max"] == pytest.approx(0.0, abs=0.06)       # rounded to 0.1 mm
    assert b["before"]["z_mm"]["mean"] == pytest.approx(19.0)
    assert all(f["n_fit"] == {"cube": 2, "cylinder": 2} for f in b["folds"])


def test_the_fold_never_fits_on_the_layout_it_grades_and_a_kind_without_a_fit_is_counted():
    layouts = [_layout("L1", [("cube", "a", (0.0, 0.0, 30.0))]),
               _layout("L2", [("cylinder", "b", (0.0, 0.0, 10.0))])]
    b = pg.bias_loo(layouts)
    assert b["n_graded"] == 0 and b["n_unfitted"] == 2       # each kind exists in one layout only: nothing to fit on
    layouts.append(_layout("L3", [("cube", "c", (0.0, 0.0, 10.0))]))
    b = pg.bias_loo(layouts)
    fold = {f["layout"]: f for f in b["folds"]}
    assert fold["L1"]["bias_mm"]["cube"] == [0.0, 0.0, 10.0] and fold["L3"]["bias_mm"]["cube"] == [0.0, 0.0, 30.0]
    assert b["n_graded"] == 2 and b["n_unfitted"] == 1
    assert b["after"]["z_mm"]["min"] == pytest.approx(-20.0) and b["after"]["z_mm"]["max"] == pytest.approx(20.0)


def test_rejected_layouts_and_off_board_rows_are_left_out_of_the_fit():
    bad = _layout("X", [("cube", "z", (0.0, 0.0, 999.0))])
    bad["accepted"] = False
    floor = _layout("F", [("cube", "f", (0.0, 0.0, 500.0))])
    floor["pairs"][0]["matches"][0]["on_board"] = False
    good = [_layout("L1", [("cube", "a", (0.0, 0.0, 20.0))]), _layout("L2", [("cube", "b", (0.0, 0.0, 20.0))])]
    b = pg.bias_loo([bad, floor] + good)
    assert b["n_rows"] == 2 and b["bias_mm_all_rows"]["cube"][2] == 20.0
    assert pg.bias_loo([bad, floor] + good, score_off_board=True)["n_rows"] == 3
