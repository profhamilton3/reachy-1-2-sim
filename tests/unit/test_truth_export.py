"""#186: the read-only truth export (native_mujoco/truth_export.py, scripts/export_truth.py)."""

import asyncio
import copy
import json
import os
import pathlib
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))
sys.path.insert(0, os.path.join(_HERE, "../../scripts"))

import truth_export as te  # noqa: E402
from scene_io import load_scene  # noqa: E402

_REPO = pathlib.Path(_HERE).resolve().parents[1]
_FIX = _REPO / "tests" / "fixtures" / "perception_grade"


@pytest.fixture(scope="module")
def scene():
    return load_scene(_REPO / "scenes" / "FWDCenterLabSivaPool.yaml")


def _state(scene, **over):
    cells = {o["id"]: o for o in scene["objects"] if "grid-cell" in (o.get("tags") or [])}
    cube_pos = list(cells["cell_r1c2"]["pose"]["position"])
    cube_pos[2] += 0.03 + 0.0005
    st = {"type": "state", "sim_step": 7, "sim_time_s": 0.014, "scene_revision": "rev-1",
          "objects": [
              {"object_id": "red_cube", "pos_xyz": cube_pos, "quat_wxyz": [1, 0, 0, 0]},
              {"object_id": "blue_cylinder", "pos_xyz": [0.35, 0.95, 0.05], "quat_wxyz": [1, 0, 0, 0]}]}
    st.update(over)
    return st


def test_build_joins_state_with_kind_size_and_cell(scene):
    doc = te.build_truth(_state(scene), scene)
    assert doc["schema_version"] == "truth.v1" and doc["quat_order"] == "wxyz"
    assert doc["arm_base_in_world"] == [0.206, 0.0, 1.15]
    assert (doc["scene_revision"], doc["sim_step"]) == ("rev-1", 7)
    by = {o["id"]: o for o in doc["objects"]}
    assert by["red_cube"]["kind"] == "cube" and by["red_cube"]["shape"] == "box"
    assert by["red_cube"]["size_m"] == [0.06, 0.06, 0.06] and by["red_cube"]["cell"] == "r1c2"
    cyl = by["blue_cylinder"]
    assert cyl["shape"] == "cylinder" and cyl["radius_m"] == 0.035
    assert cyl["size_m"] == [0.07, 0.07, 0.1] and cyl["cell"] is None      # parked on the floor
    assert len(doc["cells"]) == 9
    assert [o["id"] for o in doc["objects"]] == sorted(by)


def test_a_state_object_the_scene_does_not_know_is_an_error(scene):
    st = _state(scene)
    st["objects"].append({"object_id": "ghost", "pos_xyz": [0, 0, 0], "quat_wxyz": [1, 0, 0, 0]})
    with pytest.raises(te.TruthError, match="not in the scene file"):
        te.build_truth(st, scene)


@pytest.mark.parametrize("mutate,match", [
    (lambda s: s["objects"][0].update(pos_xyz=[0, float("nan"), 0]), "finite"),
    (lambda s: s["objects"][0].update(pos_xyz=[0, 0]), "3 finite"),
    (lambda s: s["objects"][0].update(quat_wxyz=[2, 0, 0, 0]), "unit quaternion"),
    (lambda s: s.update(scene_revision=""), "scene_revision"),
    (lambda s: s.update(sim_step=-1), "sim_step"),
    (lambda s: s.update(type="camera_frame"), "not a state"),
    (lambda s: s["objects"].append(copy.deepcopy(s["objects"][0])), "duplicate"),
    (lambda s: s.update(objects=[{"object_id": "x" * 65, "pos_xyz": [0] * 3, "quat_wxyz": [1, 0, 0, 0]}]), "bad object id"),
])
def test_bad_states_are_refused(scene, mutate, match):
    st = _state(scene)
    mutate(st)
    with pytest.raises(te.TruthError, match=match):
        te.build_truth(st, scene)


def test_object_count_is_bounded(scene):
    st = _state(scene)
    st["objects"] = [dict(st["objects"][0], object_id=f"o{i}") for i in range(te.MAX_OBJECTS + 1)]
    with pytest.raises(te.TruthError, match="exceeds the limit"):
        te.build_truth(st, scene)


def test_validate_rejects_unknown_fields_and_wrong_schema(scene):
    doc = te.build_truth(_state(scene), scene)
    te.validate_truth(doc)
    for bad in (dict(doc, extra=1), dict(doc, schema_version="truth.v2")):
        with pytest.raises(te.TruthError):
            te.validate_truth(bad)
    bad = copy.deepcopy(doc)
    bad["objects"][0]["surprise"] = 1
    with pytest.raises(te.TruthError, match="object fields"):
        te.validate_truth(bad)


def test_checked_in_fixtures_validate():
    for name in ("truth_before.json", "truth_after.json", "truth_after_moved.json"):
        te.validate_truth(json.loads((_FIX / name).read_text()))


# --- the WebSocket client is read-only -------------------------------------------------

class FakeWS:
    def __init__(self, incoming):
        self.incoming, self.sent = list(incoming), []

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    async def recv(self):
        if not self.incoming:
            await asyncio.sleep(3600)
        return json.dumps(self.incoming.pop(0))


ACK = {"type": "hello_ack", "protocol_version": 1}


def test_client_sends_only_hello_and_heartbeat_acks_then_returns_first_state():
    state = {"type": "state", "sim_step": 3}
    ws = FakeWS([ACK, {"type": "heartbeat", "sent_ns": 99}, {"type": "camera_frame"},
                 {"type": "scene_ack"}, state, {"type": "state", "sim_step": 4}])
    got = asyncio.run(te.read_one_state(ws, timeout_s=2))
    assert got == state
    assert [m["type"] for m in ws.sent] == ["hello", "heartbeat_ack"]
    assert ws.sent[1]["echo_ns"] == 99
    assert {m["type"] for m in ws.sent} <= te.ALLOWED_SEND_TYPES


def test_client_refuses_to_send_anything_else():
    ws = FakeWS([])
    for kind in ("joint_command", "reset", "pause", "place_object", "scene_load", "acquire_control"):
        with pytest.raises(te.TruthError, match="read-only"):
            asyncio.run(te._send(ws, {"type": kind}))
    assert ws.sent == []


def test_client_times_out_instead_of_blocking():
    ws = FakeWS([ACK])
    with pytest.raises(te.TruthError, match="no state"):
        asyncio.run(te.read_one_state(ws, timeout_s=0.2))


@pytest.mark.parametrize("incoming,match", [
    ([{"type": "error"}], "expected hello_ack"),
    ([{"type": "hello_ack", "protocol_version": 2}], "protocol"),
    ([ACK, {"type": "shutdown"}], "shutting down"),
])
def test_client_handshake_failures_are_explicit(incoming, match):
    with pytest.raises(te.TruthError, match=match):
        asyncio.run(te.read_one_state(FakeWS(incoming), timeout_s=1))


def test_fetch_state_reports_a_refused_connection():
    with pytest.raises(te.TruthError, match="127.0.0.1"):
        asyncio.run(te.fetch_state("ws://127.0.0.1:9", timeout_s=1))


def test_cli_refuses_a_non_local_url(capsys):
    import export_truth
    assert export_truth.main(["--url", "ws://10.0.0.5:8765"]) == 2
    assert "non-local" in capsys.readouterr().err
