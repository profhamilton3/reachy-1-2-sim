"""#55 / ADR-0005: the MuJoCo no-slip contact model (simulator-only).

Offline.  Covers the validator, the model default and override, the server's
argparse / handshake / recorder-manifest reporting, the panel SimLink status,
and a recorded-hold creep reproduction (default vs ``--noslip-iterations 0``).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import types

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (_ROOT / "native_mujoco", _ROOT / "src", _ROOT / "web"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mujoco  # noqa: E402

import contact_model as cm  # noqa: E402
import server as native_server  # noqa: E402
from simulation_core import SimulationCore, load_world  # noqa: E402

MODEL = str(_ROOT / "native_mujoco" / "model" / "reachy_1_2.xml")
SCENE = str(_ROOT / "scenes" / "FWDCenterLabSivaPool.yaml")


# -- validator ----------------------------------------------------------------

@pytest.mark.parametrize("v,expected", [(0, 0), (10, 10), (50, 50), ("7", 7), (3.0, 3)])
def test_validator_accepts(v, expected):
    assert cm.validate_noslip_iterations(v) == expected


@pytest.mark.parametrize("v", [-1, 51, True, False, 2.5, "x", "", None, [1]])
def test_validator_rejects(v):
    with pytest.raises(ValueError, match="noslip_iterations"):
        cm.validate_noslip_iterations(v)


# -- model default and override ------------------------------------------------

def test_model_file_default_is_ten():
    m = mujoco.MjModel.from_xml_path(MODEL)
    assert m.opt.noslip_iterations == 10
    assert m.opt.noslip_tolerance == pytest.approx(1e-6)   # unchanged default
    d = cm.apply_contact_model(m)
    assert d["noslip_iterations"] == 10 and d["source"] == "model"
    assert d["model_default_noslip_iterations"] == 10
    assert d["cone"] == "pyramidal" and d["solver"] == "Newton"
    json.dumps(d)                                           # JSON-serialisable


def test_apply_override_zero_and_bad_value():
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = cm.apply_contact_model(m, 0)
    assert m.opt.noslip_iterations == 0
    assert (d["noslip_iterations"], d["source"], d["model_default_noslip_iterations"]) == (0, "override", 10)
    with pytest.raises(ValueError):
        cm.apply_contact_model(m, 99)


def test_load_world_and_from_paths():
    model, *_ = load_world(MODEL, SCENE)
    assert model.opt.noslip_iterations == 10
    model0, *_ = load_world(MODEL, SCENE, noslip_iterations=0)
    assert model0.opt.noslip_iterations == 0
    core = SimulationCore.from_paths(MODEL, SCENE)
    assert core.contact_model["source"] == "model" and core.contact_model["noslip_iterations"] == 10
    core0 = SimulationCore.from_paths(MODEL, SCENE, noslip_iterations=0)
    assert core0.contact_model["source"] == "override"
    assert core0.contact_model["noslip_iterations"] == 0
    assert core0.model.opt.noslip_iterations == 0


def test_physics_profile_id():
    m = mujoco.MjModel.from_xml_path(MODEL)
    assert cm.physics_profile_id(cm.apply_contact_model(m)) == "default"
    assert cm.physics_profile_id(cm.apply_contact_model(m, 10)) == "default"
    assert cm.physics_profile_id(cm.apply_contact_model(m, 0)) == "noslip_iterations=0"
    assert cm.physics_profile_id(None) == "default"


# -- server: argparse, handshake, manifest -------------------------------------

def _parse(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["server.py", *argv])
    captured = {}

    class _Stop(Exception):
        pass

    def fake_server(*a, **kw):
        captured.update(kw)
        raise _Stop

    monkeypatch.setattr(native_server, "ReachyMujocoServer", fake_server)
    with pytest.raises(_Stop):
        native_server.main()
    return captured


def test_server_argparse_default_and_override(monkeypatch):
    assert _parse([], monkeypatch)["noslip_iterations"] is None
    assert _parse(["--noslip-iterations", "0"], monkeypatch)["noslip_iterations"] == 0
    assert _parse(["--noslip-iterations", "25"], monkeypatch)["noslip_iterations"] == 25


@pytest.mark.parametrize("bad", ["-1", "51", "2.5", "x"])
def test_server_argparse_rejects_bad_values(bad, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["server.py", "--noslip-iterations", bad])
    with pytest.raises(SystemExit) as e:
        native_server.main()
    assert e.value.code == 2


def _stub(override):
    m = mujoco.MjModel.from_xml_path(MODEL)
    return types.SimpleNamespace(
        _contact_model=cm.apply_contact_model(m, override),
        _model_path="/tmp/none.xml", _scene_path=None,
        _sim=types.SimpleNamespace(scene_revision="r1"), _calibration=None,
        _enable_depth=False, _enable_seg=False, _record_contacts=False,
        _effects=types.SimpleNamespace(blur_sigma=0.0, noise_std=0.0,
                                       drop_probability=0.0, latency_ms=0.0))


def test_hello_capabilities_advertise_contact_model():
    caps = native_server.ReachyMujocoServer._hello_capabilities(_stub(None))
    assert caps["execution_lease"] is True
    assert caps["contact_model"]["noslip_iterations"] == 10
    caps0 = native_server.ReachyMujocoServer._hello_capabilities(_stub(0))
    assert caps0["contact_model"]["noslip_iterations"] == 0
    assert caps0["contact_model"]["source"] == "override"


def test_manifest_records_contact_model_and_profile():
    build = native_server.ReachyMujocoServer._build_recorder_manifest
    default, off = build(_stub(None)), build(_stub(0))
    assert default["contact_model"]["noslip_iterations"] == 10
    assert default["physics_profile_id"] == "default"
    assert off["contact_model"]["noslip_iterations"] == 0
    assert off["physics_profile_id"] == "noslip_iterations=0"
    assert off["physics_profile_id"] != default["physics_profile_id"]


# -- panel SimLink -------------------------------------------------------------

def test_simlink_status_contact_model():
    from panel_sim_link import SimLink
    link = SimLink()
    assert link.status["contact_model"] is None        # before / without a handshake
    link._server_capabilities = {"execution_lease": True}
    assert link.status["contact_model"] is None        # older server: never invented
    advertised = {"noslip_iterations": 10, "source": "model"}
    link._server_capabilities = {"execution_lease": True, "contact_model": advertised}
    assert link.status["contact_model"] == advertised
