"""#186: grading the viewer's detections and 3D positions against simulator truth.

Fixtures (tests/fixtures/perception_grade): a truth export of FWDCenterLabSivaPool with
red_cube on r1c2, blue_cylinder on r3c3 and soda_can (a class the detector lacks) on r2c2;
one viewer pair at board_view (neck 0, 64, 0) whose detections are consistent with it.
"""

import copy
import hashlib
import io
import json
import math
import os
import pathlib
import sys
import zipfile

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))
sys.path.insert(0, os.path.join(_HERE, "../../scripts"))

import perception_grade as pg  # noqa: E402

_FIX = pathlib.Path(_HERE).resolve().parents[1] / "tests" / "fixtures" / "perception_grade"


def _load(name):
    return json.loads((_FIX / name).read_text())


@pytest.fixture()
def truth():
    return _load("truth_before.json")


@pytest.fixture()
def pair():
    return {k: json.loads((_FIX / "pair_ok" / f"{k}.json").read_text())
            for k in ("pair", "analysis", "depth")}


def _det(pair, label):
    return next(i for i, d in enumerate(pair["analysis"]["detections"]) if d["label"] == label)


def _drop_detection(pair, index):
    """Remove detection `index` and renumber the measurements."""
    pair["analysis"]["detections"].pop(index)
    pair["depth"]["measurements"] = [
        dict(m, detection_index=m["detection_index"] - (m["detection_index"] > index))
        for m in pair["depth"]["measurements"] if m["detection_index"] != index]


# --- geometry ------------------------------------------------------------------------

def _viewer_camera_to_torso(points, neck_deg):
    """Port of the viewer's camera_to_torso (reachy-tabletop-ai stereo_depth.py @ fec44b6)."""
    def ax(axis, deg):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
                "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
                "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]
    rot = ax("y", math.degrees(0.174))
    for axis, deg in zip("xyz", neck_deg):
        rot = rot @ ax(axis, deg)
    o2h = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    head = np.asarray(points) @ o2h.T + [0.0033, -0.03625, 0.061]
    return head @ rot.T + [0.015, 0, 0.095]


def test_rig_inverts_the_viewers_camera_to_torso(pair):
    rig = pg.Rig(pair["depth"]["profile"], {"neck_roll": 3.0, "neck_pitch": 55.0, "neck_yaw": -7.0})
    for p in ([0.0, 0.0, 0.5], [0.1, -0.05, 0.62], [-0.2, 0.1, 0.9]):
        torso = _viewer_camera_to_torso(p, (3.0, 55.0, -7.0))
        assert np.allclose(rig.to_torso(p), torso, atol=1e-9)
        assert np.allclose(rig.to_camera(torso, "right"), p, atol=1e-9)


def test_left_camera_is_one_baseline_to_the_left(pair):
    rig = pg.Rig(pair["depth"]["profile"], {"neck_roll": 0.0, "neck_pitch": 0.0, "neck_yaw": 0.0})
    gap = rig.origin("left") - rig.origin("right")
    assert np.allclose(gap, [0, 0.0725, 0], atol=1e-9)               # +y is the robot's left


def test_projection_centre_and_behind(pair):
    rig = pg.Rig(pair["depth"]["profile"], {"neck_roll": 0.0, "neck_pitch": 0.0, "neck_yaw": 0.0})
    axis = rig.R @ pg._OPTICAL_TO_HEAD @ [0, 0, 1.0]                 # optical +z in the torso frame
    assert rig.project(rig.origin("right") + axis, "right") == pytest.approx((320, 240))
    assert rig.project(rig.origin("right") - axis, "right") is None


def test_support_function_of_box_and_cylinder():
    box = {"shape": "box", "size_m": [0.06, 0.06, 0.06], "quat_wxyz": [1, 0, 0, 0], "radius_m": None}
    cyl = {"shape": "cylinder", "size_m": [0.07, 0.07, 0.1], "quat_wxyz": [1, 0, 0, 0], "radius_m": 0.035}
    up, side = np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
    assert pg.half_extent_along(box, up) == pytest.approx(0.03)
    assert pg.half_extent_along(box, np.ones(3) / math.sqrt(3)) == pytest.approx(0.03 * math.sqrt(3))
    assert pg.half_extent_along(cyl, up) == pytest.approx(0.05)
    assert pg.half_extent_along(cyl, side) == pytest.approx(0.035)


# --- one pair --------------------------------------------------------------------------

def test_consistent_pair_is_graded_correct(truth, pair):
    r = pg.grade_pair(truth, pair)
    assert {m["object_id"]: m["correct"] for m in r["matches"]} == {"red_cube": True, "blue_cylinder": True}
    assert [h["object_id"] for h in r["oov_hits"]] == ["soda_can"]         # not scored correct, not an FP
    assert not r["false_positives"]
    assert not [m for m in r["misses"] if m["on_board"]]
    assert all(e["correct"] for e in r["empty_dets"]) and len(r["empty_dets"]) == 6
    for m in r["matches"]:
        assert m["via"] == "3d" and m["err"]["gate_mm"] < 10          # injected error is ~5 mm
        # the gate point is nearer than the centre, so the centre error exceeds it
        assert m["err"]["centre_mm"] > m["err"]["gate_mm"] and m["err"]["along_ray_mm"] < 0
    assert r["cells"]["r2c2"]["outcome"] == "oov"
    assert r["cells"]["r1c2"] == {"truth": "cube", "reported": "cube", "outcome": "correct"}


def test_headline_leaves_the_floor_pool_out_and_reports_it_separately(truth, pair):
    s = pg.summarize([pg.grade_pair(truth, pair)])
    assert s["per_kind"]["cube"]["truth_in_view"] == 1 and s["per_kind"]["cylinder"]["truth_in_view"] == 1
    assert s["off_board"]["missed"] >= 10 and s["off_board"]["scored_in_headline"] is False
    folded = pg.summarize([pg.grade_pair(truth, pair)], score_off_board=True)
    assert folded["per_kind"]["cube"]["truth_in_view"] > 1


def test_horizontal_error_is_against_the_true_centre(truth, pair):
    m = next(m for m in pg.grade_pair(truth, pair)["matches"] if m["object_id"] == "red_cube")
    assert m["err"]["horizontal_mm"] < m["err"]["centre_mm"]
    s = pg.summarize([pg.grade_pair(truth, pair)])
    assert s["error_3d_mm"]["horizontal_mm"]["n"] == 2
    assert set(s["error_3d_by_cell_mm"]) == {"r1c2", "r3c3"}


def test_a_miss(truth, pair):
    _drop_detection(pair, _det(pair, "cube"))
    r = pg.grade_pair(truth, pair)
    assert [m["object_id"] for m in r["misses"] if m["on_board"]] == ["red_cube"]
    assert r["cells"]["r1c2"]["reported"] == "unknown" and r["cells"]["r1c2"]["outcome"] == "unknown"
    s = pg.summarize([r])
    assert s["per_kind"]["cube"]["recall"] == 0.0 and s["per_object"]["red_cube"]["missed"] == 1
    assert s["confusion"]["cube"] == {"missed": 1}


def test_a_false_positive(truth, pair):
    """A cube call 150 mm from anything real."""
    pair["analysis"]["detections"].append({"label": "cube", "class_id": 0, "confidence": 0.6,
                                           "box_xywh": [10, 10, 40, 40]})
    pair["depth"]["measurements"].append({
        "detection_index": len(pair["analysis"]["detections"]) - 1, "label": "cube", "status": "measured",
        "xyz_torso_m": [0.20, 0.30, -0.40]})
    r = pg.grade_pair(truth, pair)
    assert len(r["false_positives"]) == 1 and r["false_positives"][0]["measured"]
    assert r["false_positives"][0]["nearest_mm"] > 100
    s = pg.summarize([r])
    assert s["false_positives"] == 1 and s["per_kind"]["cube"]["precision"] == 0.5


def test_a_wrong_class(truth, pair):
    pair["analysis"]["detections"][_det(pair, "cube")]["label"] = "cylinder"
    r = pg.grade_pair(truth, pair)
    wrong = next(m for m in r["matches"] if m["object_id"] == "red_cube")
    assert wrong["correct"] is False and wrong["label"] == "cylinder"
    assert r["cells"]["r1c2"]["outcome"] == "wrong"
    s = pg.summarize([r])
    assert s["confusion"]["cube"] == {"cylinder": 1}
    assert s["class_accuracy"] == 0.5
    # three cylinder calls: the real cylinder, the mislabelled cube, and the can (an OOV hit)
    assert s["per_kind"]["cube"]["recall"] == 0.0 and s["per_kind"]["cylinder"]["precision"] == pytest.approx(1 / 3, abs=1e-3)


def test_a_detection_without_3d_matches_in_2d_only(truth, pair):
    i = _det(pair, "cube")
    pair["depth"]["measurements"][i] = {"detection_index": i, "label": "cube", "status": "unavailable",
                                       "reason": "Too few consistent stereo matches inside this detection."}
    r = pg.grade_pair(truth, pair)
    m = next(m for m in r["matches"] if m["object_id"] == "red_cube")
    assert m["via"] == "2d" and m["err"] is None and m["correct"]
    s = pg.summarize([r])
    assert s["matched_2d_only"] == 1 and s["error_3d_mm"]["gate_mm"]["n"] == 1


def test_empty_is_graded_against_the_cell(truth, pair):
    """An `empty` box over the cell that holds the cube is a false positive; over an empty cell, correct."""
    r0 = pg.grade_pair(truth, pair)
    cube_box = pair["analysis"]["detections"][_det(pair, "cube")]["box_xywh"]
    pair["analysis"]["detections"].append({"label": "empty", "class_id": 2, "confidence": 0.5,
                                           "box_xywh": cube_box})
    r = pg.grade_pair(truth, pair)
    new = r["empty_dets"][-1]
    assert new["cell"] == "r1c2" and new["correct"] is False
    assert len(r0["empty_dets"]) == 6
    e = pg.summarize([r])["empty"]
    assert (e["correct"], e["false_positive"], e["unassigned"]) == (6, 1, 0)


def test_an_empty_box_on_no_cell_is_unassigned(truth, pair):
    pair["analysis"]["detections"].append({"label": "empty", "class_id": 2, "confidence": 0.4,
                                           "box_xywh": [0, 0, 8, 8]})
    e = pg.summarize([pg.grade_pair(truth, pair)])["empty"]
    assert e["unassigned"] == 1 and e["false_positive"] == 0


def test_tighter_radius_turns_matches_into_a_miss_and_a_false_positive(truth, pair):
    r = pg.grade_pair(truth, pair, radius_m=0.002)
    assert not r["matches"] and r["false_positives"]
    # the miss records that a detection box still covers the object
    assert any(m["box_covers_label"] for m in r["misses"] if m["on_board"])


def test_pair_without_depth_or_neck_is_ungradable(truth, pair):
    with pytest.raises(pg.GradeError, match="Measure in 3D"):
        pg.grade_pair(truth, {"analysis": pair["analysis"], "pair": pair["pair"]})
    del pair["depth"]["head_pose_used"]
    del pair["pair"]["head_context"]
    with pytest.raises(pg.GradeError, match="neck"):
        pg.grade_pair(truth, pair)


# --- the static-scene bracket --------------------------------------------------------------

def test_bracket_accepts_an_unchanged_scene():
    ok, reasons, worst = pg.check_bracket(_load("truth_before.json"), _load("truth_after.json"))
    assert ok and not reasons and worst == 0.0


def test_bracket_rejects_a_moved_object():
    ok, reasons, worst = pg.check_bracket(_load("truth_before.json"), _load("truth_after_moved.json"))
    assert not ok and worst == pytest.approx(0.005)
    assert reasons == ["red_cube moved 5.0 mm"]


def test_bracket_tolerance_is_one_millimetre():
    before, after = _load("truth_before.json"), _load("truth_after.json")
    after["objects"][0]["pos_xyz"][0] += 0.0009
    assert pg.check_bracket(before, after)[0]
    after["objects"][0]["pos_xyz"][0] += 0.0003
    assert not pg.check_bracket(before, after)[0]


def test_bracket_rejects_a_revision_change_and_a_changed_object_set():
    before, after = _load("truth_before.json"), _load("truth_after.json")
    after["scene_revision"] = "rev-2"
    after["objects"].pop()
    ok, reasons, _ = pg.check_bracket(before, after)
    assert not ok and any("scene_revision" in r for r in reasons) and any("object set" in r for r in reasons)


# --- the whole run, from files ----------------------------------------------------------------

def _zip_pair(dest, src=_FIX / "pair_ok", tamper=False):
    files = {n: (src / n).read_bytes() for n in ("pair.json", "analysis.json", "depth.json")}
    manifest = {"schema_version": "1", "files": [
        {"name": n, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in files.items()]}
    if tamper:
        files["analysis.json"] += b" "
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n, b in {**files, "manifest.json": json.dumps(manifest).encode()}.items():
            zf.writestr(n, b)
    dest.write_bytes(buf.getvalue())


def _run_dir(tmp_path, after="truth_after.json"):
    (tmp_path / "A").mkdir()
    for name, src in (("truth_before.json", "truth_before.json"), ("truth_after.json", after)):
        (tmp_path / "A" / name).write_text((_FIX / src).read_text())
    _zip_pair(tmp_path / "A" / "pair1.zip")
    _zip_pair(tmp_path / "A" / "pair2.zip")
    manifest = {"layouts": [{"name": "A", "truth_before": "A/truth_before.json",
                             "truth_after": "A/truth_after.json",
                             "pairs": ["A/pair1.zip", "A/pair2.zip"]}]}
    (tmp_path / "layouts.json").write_text(json.dumps(manifest))
    return tmp_path / "layouts.json"


def test_miss_line_says_when_no_box_covers_the_object():
    base = {"object_id": "red_cube", "kind": "cube", "cell": "r1c2"}
    assert pg._miss_line(dict(base, box_covers_label=None)) == \
        "- miss: red_cube (cube, r1c2); no detection box covers it"
    assert pg._miss_line(dict(base, box_covers_label="cylinder")) == \
        "- miss: red_cube (cube, r1c2); a box covers it labelled cylinder"


def test_end_to_end_writes_result_and_report(tmp_path):
    m = _run_dir(tmp_path)
    assert pg.main(["--manifest", str(m), "--out-dir", str(tmp_path / "out")]) == 0
    result = json.loads((tmp_path / "out" / "result.json").read_text())
    assert result["schema_version"] == "perception_grade.v1"
    assert result["config"]["match_radius_m"] == 0.04 and result["config"]["score_off_board"] is False
    assert result["overall"]["pairs"] == 2 and not result["rejected_pairs"]
    assert result["inputs"][0]["pairs"][0]["sha256"]
    report = (tmp_path / "out" / "report.md").read_text()
    for needle in ("40 mm", "horizontal", "OOV", "By camera distance", "Size error: not available"):
        assert needle in report


def test_end_to_end_rejects_a_moved_layout(tmp_path, capsys):
    m = _run_dir(tmp_path, after="truth_after_moved.json")
    assert pg.main(["--manifest", str(m), "--out-dir", str(tmp_path / "out")]) == 0
    result = json.loads((tmp_path / "out" / "result.json").read_text())
    assert result["layouts"][0]["accepted"] is False and result["overall"] is None
    assert "red_cube moved 5.0 mm" in result["layouts"][0]["bracket"]["reasons"]
    assert "WARNING" in capsys.readouterr().err
    assert "REJECTED" in (tmp_path / "out" / "report.md").read_text()


def test_a_tampered_bundle_is_rejected_not_graded(tmp_path):
    m = _run_dir(tmp_path)
    _zip_pair(tmp_path / "A" / "pair2.zip", tamper=True)
    pg.main(["--manifest", str(m), "--out-dir", str(tmp_path / "out")])
    result = json.loads((tmp_path / "out" / "result.json").read_text())
    assert result["overall"]["pairs"] == 1
    assert "SHA-256" in result["rejected_pairs"][0]["reason"]


@pytest.mark.parametrize("rel", ["../escape.json", "/etc/hosts", "http://example.com/x.json"])
def test_manifest_paths_must_stay_below_the_manifest(tmp_path, rel):
    (tmp_path / "sub").mkdir()
    (tmp_path.parent / "escape.json").write_text("{}")
    (tmp_path / "sub" / "layouts.json").write_text(json.dumps({"layouts": [{
        "name": "A", "truth_before": rel, "truth_after": rel, "pairs": [rel]}]}))
    assert pg.main(["--manifest", str(tmp_path / "sub" / "layouts.json"),
                    "--out-dir", str(tmp_path / "out")]) == 1


def test_manifest_with_unknown_fields_or_repeated_names_fails(tmp_path):
    for doc in ({"layouts": [], "extra": 1},
                {"layouts": [{"name": "A", "truth_before": "x", "truth_after": "y", "pairs": ["z"], "oops": 1}]}):
        (tmp_path / "m.json").write_text(json.dumps(doc))
        with pytest.raises(pg.GradeError):
            pg.load_manifest(tmp_path / "m.json")


def test_unpacked_folder_is_accepted(truth):
    parsed, digest = pg.load_pair(_FIX / "pair_ok")
    assert digest == "" and pg.grade_pair(truth, parsed)["matches"]
