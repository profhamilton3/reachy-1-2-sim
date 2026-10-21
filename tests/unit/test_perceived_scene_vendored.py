"""P0b (#182): the vendored perceived_scene.v1 contract (OD-1) accepts and refuses the upstream golden fixtures.

native_mujoco/perceived_scene.py is a version-pinned copy of reachy-tabletop-ai's scene_contract.py, and the sim's
panel container is Python 3.8: this file and the vendored one must stay 3.8-clean.
"""
import ast
import hashlib
import os
import pathlib
import re
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))

import perceived_scene as ps  # noqa: E402

_ROOT = pathlib.Path(_HERE).resolve().parents[1]
_VENDORED = _ROOT / "native_mujoco" / "perceived_scene.py"
_FIX = _ROOT / "tests" / "fixtures" / "perceived_scene"
_MARKER = "# ---- end vendor header; the upstream file follows byte for byte ----\n"

VALID = sorted((_FIX / "valid").glob("*.json"))
INVALID = sorted((_FIX / "invalid").glob("*.json"))

# invalid fixture -> a phrase its refusal must contain: the same reasons as upstream's tests/test_scene_contract.py.
REASONS = {
    "unknown_field": "unknown field(s): extra",
    "unknown_nested_field": "RigIdentity: unknown field(s)",
    "missing_field": "missing field(s): off_board",
    "wrong_schema_version": "schema_version",
    "object_quaternion_not_unit": "PerceivedObject.quat_wxyz: quaternion",
    "board_quaternion_not_unit": "BoardPose.quat_wxyz: quaternion",
    "eight_cells": "exactly the nine cells",
    "ten_cells": "exactly the nine cells",
    "cells_out_of_order": "exactly the nine cells",
    "cells_without_board": "has no cells",
    "bad_zoom_level": "zoom_level",
    "robot_calibration_id_zoom_mismatch": "does not carry the zoom_level",
    "sim_field_on_robot_rig": "sim-only",
    "nan_value": "NaN",
    "infinity_value": "Infinity",
    "overflow_value": "must be finite",
    "null_bound_not_provisional": "must be finite",
    "negative_bound": "negative",
    "boolean_as_number": "expected a number",
    "confidence_above_one": "outside [0, 1]",
    "bad_kind": "PerceivedObject.kind",
    "size_m_with_unknown_size_source": "size_source unknown cannot have a size_m",
    "cell_points_at_wrong_kind": "but the object is a",
    "cell_object_index_out_of_range": "out of range",
    "unknown_cell_with_object_index": "only for cube or cylinder",
    "object_disagrees_with_its_cell": "says cell",
    "off_board_object_with_cell": "has no cell",
    "object_calibration_differs_from_rig": "differs from the rig",
    "bad_sha256": "64 lowercase hex",
    "empty_code_sha_not_dirty": "needs dirty=true",
    "bad_utc_time": "UTC time",
    "single_pair_fit_with_spread": "single_pair_fit",
    "duplicate_key": "duplicate key",
    "not_an_object": "expected an object",
    "not_json": "not valid JSON",
}


def _split_header():
    text = _VENDORED.read_text()
    assert text.count(_MARKER) == 1, "the vendor header marker must appear exactly once"
    header, body = text.split(_MARKER)
    return header, body


def test_the_vendored_file_matches_the_sha256_in_its_header():
    header, body = _split_header()
    recorded = re.search(r"source SHA-256 \(of everything below the marker line\): ([0-9a-f]{64})", header)
    assert recorded, "the header must record the file's SHA-256"
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == recorded.group(1)


def test_the_header_names_the_source_repo_path_and_commit():
    header, _ = _split_header()
    assert "profhamilton3/reachy-tabletop-ai" in header
    assert "src/reachy_ai/perception/scene_contract.py" in header
    assert re.search(r"source commit: [0-9a-f]{40}", header)


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.stem)
def test_a_valid_fixture_is_accepted_and_round_trips_byte_for_byte(path):
    text = path.read_text()
    scene = ps.loads(text)
    assert ps.dumps(scene) == text


@pytest.mark.parametrize("path", INVALID, ids=lambda p: p.stem)
def test_an_invalid_fixture_is_refused_for_the_upstream_reason(path):
    with pytest.raises(ps.SceneContractError) as refused:
        ps.loads(path.read_text())
    assert REASONS[path.stem] in str(refused.value)


def test_every_invalid_fixture_has_a_stated_reason():
    assert {p.stem for p in INVALID} == set(REASONS)
    assert len(VALID) == 3


def test_the_copied_fixtures_are_the_pinned_bytes():
    sums = {}
    for line in (_FIX / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        sums[name] = digest
    on_disk = {str(p.relative_to(_FIX)) for p in _FIX.rglob("*") if p.is_file()
               and p.name not in ("SHA256SUMS", "VENDORED.md")}
    assert set(sums) == on_disk
    for name, digest in sums.items():
        assert hashlib.sha256((_FIX / name).read_bytes()).hexdigest() == digest, name


def test_the_fixtures_equal_upstream_when_a_tabletop_checkout_is_given():
    upstream = os.environ.get("TABLETOP_AI_REPO")
    if not upstream:
        pytest.skip("TABLETOP_AI_REPO is not set: only the pinned SHA256SUMS are compared (tabletop is not a "
                    "dependency of this repo's tests)")
    root = pathlib.Path(upstream) / "tests" / "fixtures" / "perceived_scene"
    for path in list(VALID) + list(INVALID):
        assert (root / path.parent.name / path.name).read_bytes() == path.read_bytes(), path.name
    header, body = _split_header()
    source = pathlib.Path(upstream) / "src" / "reachy_ai" / "perception" / "scene_contract.py"
    assert source.read_text() == body


@pytest.mark.parametrize("path", [_VENDORED, pathlib.Path(__file__)], ids=["vendored", "this test"])
def test_python_3_8_syntax(path):
    tree = ast.parse(path.read_text(), feature_version=(3, 8))
    unions = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr)]
    generics = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                and n.value.id in {"list", "dict", "tuple", "set", "frozenset", "type"}]
    assert not unions and not generics
