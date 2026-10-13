"""#2: `scripts/smoke_test_host.py`'s scene check, offline.

The rest of the smoke test needs a running simulator and the SDK; check [7]
does not, so it is pinned here.  It asserts what the script promises and no
more: the scene's manipulable objects load, and each grasp point passes
`SceneModel.check_point` (the pad-point rule -- not reachability, not
whole-arm clearance).
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (_ROOT / "scripts", _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import smoke_test_host as T  # noqa: E402

_SCENES = _ROOT / "scenes"


@pytest.mark.parametrize("name", [
    "FWDCenterLabSivaPool", "FWDCenterLabSiva", "FWDCenterLabMCC", "tabletop_demo",
])
def test_every_shipped_board_passes(name, capsys):
    assert T.check_scene(str(_SCENES / f"{name}.yaml")) == 0
    assert "[FAIL]" not in capsys.readouterr().out


def test_the_panel_scene_loads_all_ten_objects_through_extends(capsys):
    # FWDCenterLabSivaPool inherits four objects from FWDCenterLabSiva; a
    # loader that ignored `extends:` would report six.
    assert T.check_scene(str(_SCENES / "FWDCenterLabSivaPool.yaml")) == 0
    assert "10 manipulable object(s)" in capsys.readouterr().out


def test_a_grasp_point_below_the_tabletop_fails(tmp_path, capsys):
    doc = yaml.safe_load((_SCENES / "tabletop_demo.yaml").read_text())
    obj = next(o for o in doc["objects"] if o["id"] == "red_cube")
    obj["pose"]["position"][2] = 0.60          # inside the table's footprint, under its top
    path = tmp_path / "sunk.yaml"
    path.write_text(yaml.safe_dump(doc))
    assert T.check_scene(str(path)) == 1
    out = capsys.readouterr().out
    assert "[FAIL]  red_cube grasp point" in out and "below_table" in out


def test_a_scene_that_does_not_load_is_a_failure(tmp_path, capsys):
    assert T.check_scene(str(tmp_path / "missing.yaml")) == 1
    assert "scene did not load" in capsys.readouterr().out


def test_a_scene_with_no_manipulable_objects_skips_and_says_why(capsys):
    assert T.check_scene(str(_SCENES / "control_panel.yaml")) == 0
    assert "declares no manipulable objects" in capsys.readouterr().out


def test_the_default_scene_follows_the_container_setting(monkeypatch):
    monkeypatch.setenv("REACHY_SIM_SCENE_FILE", "/opt/scenes/FWDCenterLabMCC.yaml")
    assert T.default_scene_path() == str(_SCENES / "FWDCenterLabMCC.yaml")
    monkeypatch.delenv("REACHY_SIM_SCENE_FILE")
    assert T.default_scene_path() == str(_SCENES / "FWDCenterLabSivaPool.yaml")
