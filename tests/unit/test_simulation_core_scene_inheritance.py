"""SimulationCore loads a scene the way the native server does (`extends:`).

`_load_world` used to read the scene YAML raw, so a child scene lost every
object it inherits: FWDCenterLabSivaPool compiled offline with its six pool
objects only.  The server resolves `extends:` (native_mujoco/server.py), so the
offline world and the live one differed.  These pin them to the same thing.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (_ROOT / "native_mujoco", _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytest.importorskip("mujoco", reason="native MuJoCo not available")

from objects import build_scene_model_xml  # noqa: E402
from scene_compiler import tracked_object_ids  # noqa: E402
from scene_io import load_scene  # noqa: E402
from simulation_core import SimulationCore, load_world  # noqa: E402

MODEL = str(_ROOT / "native_mujoco" / "model" / "reachy_1_2.xml")
POOL = str(_ROOT / "scenes" / "FWDCenterLabSivaPool.yaml")


def test_an_inheriting_scene_keeps_its_parents_objects():
    ids = SimulationCore.from_paths(MODEL, POOL).objects.object_ids
    for inherited in ("red_cube", "blue_cylinder", "soda_can", "foam_block"):
        assert inherited in ids
    assert len(ids) == 10


@pytest.mark.parametrize("scene", ["FWDCenterLabSivaPool", "FWDCenterLabSiva",
                                   "FWDCenterLabMCC", "tabletop_demo"])
def test_the_offline_world_is_the_servers_world(scene):
    path = str(_ROOT / "scenes" / f"{scene}.yaml")
    _model, tracked, _specs, xml = load_world(MODEL, path)
    doc = load_scene(path)
    assert xml == build_scene_model_xml(doc, MODEL)
    assert tracked == tracked_object_ids(doc)
