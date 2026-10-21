"""#186: read-only, schema-versioned object truth for perception grading.

`build_truth` joins ONE `state` message (poses, sim step, scene revision) with the
scene document (kind, size, grid cells).  `fetch_state` gets that message over the
native server's WebSocket as a short-lived read-only client: it sends `hello` and
`heartbeat_ack`, nothing else, and returns after the first state it sees.  Nothing
here can advance or change the simulation, and nothing is written to the sim.

Frames and order: poses are in the SCENE frame (`pedestal`), SI units, quaternion
`[w, x, y, z]`.  The URDF torso frame is the scene frame shifted by
`ARM_BASE_IN_WORLD` with no rotation (ADR-0010).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import pathlib
from typing import Any, Mapping, Optional

from placement import cells_from_scene

SCHEMA_VERSION = "truth.v1"
PROTOCOL_VERSION = 1
CLIENT_ID = "truth-export"

#: Torso origin in the scene frame (ADR-0010, `scene/awareness.py`).
ARM_BASE_IN_WORLD = (0.206, 0.0, 1.15)

MAX_OBJECTS = 256
MAX_ID_LEN = 64
MAX_DOC_BYTES = 256 * 1024            # same bound as the protocol's scene payload
MAX_WS_BYTES = 8 * 1024 * 1024        # camera frames also arrive on the socket

#: Everything this client is allowed to put on the wire.
ALLOWED_SEND_TYPES = frozenset({"hello", "heartbeat_ack"})

_TOP_KEYS = {"schema_version", "scene_name", "scene_file", "scene_sha256", "scene_revision",
             "sim_step", "sim_time_s", "paused", "frame", "quat_order", "arm_base_in_world",
             "exported_utc", "objects", "cells"}
_OBJECT_KEYS = {"id", "kind", "semantic_class", "shape", "size_m", "radius_m",
                "pos_xyz", "quat_wxyz", "cell"}
_CELL_KEYS = {"name", "x", "y", "top_z", "half_extent"}


class TruthError(ValueError):
    """The state, scene or document is not usable; the message says why."""


def _finite_vec(value: Any, n: int, what: str) -> list:
    if (not isinstance(value, (list, tuple)) or len(value) != n
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                       and math.isfinite(v) for v in value)):
        raise TruthError(f"{what} must be {n} finite numbers")
    return [float(v) for v in value]


def _geometry(obj: Mapping[str, Any]) -> tuple:
    """(shape, bounding size [x, y, z] full extents, radius or None) of a scene object."""
    geo = obj.get("geometry") or {}
    shape = geo.get("kind")
    if shape == "box":
        size = _finite_vec(geo.get("size"), 3, f"{obj.get('id')}: box size")
        radius = None
    elif shape == "cylinder":
        r, length = geo.get("radius"), geo.get("length")
        _finite_vec([r, length], 2, f"{obj.get('id')}: cylinder radius/length")
        size, radius = [2.0 * float(r), 2.0 * float(r), float(length)], float(r)
    else:
        raise TruthError(f"{obj.get('id')}: unsupported geometry kind {shape!r}")
    if min(size) <= 0:
        raise TruthError(f"{obj.get('id')}: non-positive size {size}")
    return shape, size, radius


def _cell_of(pos: list, cells: Mapping[str, Any]) -> Optional[str]:
    """The grid cell an object stands on, with the panel's own containment rule."""
    for name, c in cells.items():
        if (abs(pos[0] - c.x) <= c.half_extent and abs(pos[1] - c.y) <= c.half_extent
                and pos[2] > c.top_z - 0.01):
            return name
    return None


def build_truth(state: Mapping[str, Any], scene_doc: Mapping[str, Any], *,
                scene_file: str = "", scene_sha256: str = "",
                exported_utc: str = "") -> dict:
    """One `state` message + the scene document -> a `truth.v1` document."""
    if state.get("type") not in (None, "state"):
        raise TruthError(f"not a state message: {state.get('type')!r}")
    revision = state.get("scene_revision")
    if not isinstance(revision, str) or not revision:
        raise TruthError("state has no scene_revision")
    step, sim_time = state.get("sim_step"), state.get("sim_time_s")
    if (not isinstance(step, int) or isinstance(step, bool) or step < 0
            or not isinstance(sim_time, (int, float)) or not math.isfinite(sim_time)):
        raise TruthError("state has no usable sim_step / sim_time_s")
    entries = state.get("objects")
    if not isinstance(entries, list):
        raise TruthError("state has no objects list")
    if len(entries) > MAX_OBJECTS:
        raise TruthError(f"{len(entries)} objects exceeds the limit of {MAX_OBJECTS}")

    by_id = {}
    for o in scene_doc.get("objects") or []:
        if isinstance(o, Mapping) and isinstance(o.get("id"), str):
            by_id[o["id"]] = o
    cells = cells_from_scene(scene_doc)

    objects, seen = [], set()
    for e in entries:
        oid = e.get("object_id") if isinstance(e, Mapping) else None
        if not isinstance(oid, str) or not oid or len(oid) > MAX_ID_LEN:
            raise TruthError(f"bad object id {oid!r}")
        if oid in seen:
            raise TruthError(f"duplicate object id {oid!r}")
        seen.add(oid)
        if oid not in by_id:
            raise TruthError(f"{oid!r} is in the sim's state but not in the scene file; "
                             "is --scene the scene the server is running?")
        pos = _finite_vec(e.get("pos_xyz"), 3, f"{oid}: pos_xyz")
        quat = _finite_vec(e.get("quat_wxyz"), 4, f"{oid}: quat_wxyz")
        if abs(math.sqrt(sum(q * q for q in quat)) - 1.0) > 1e-3:
            raise TruthError(f"{oid}: quat_wxyz is not a unit quaternion")
        shape, size, radius = _geometry(by_id[oid])
        sem = by_id[oid].get("semantic_class")
        objects.append({"id": oid, "kind": sem if isinstance(sem, str) else "",
                        "semantic_class": sem if isinstance(sem, str) else "",
                        "shape": shape, "size_m": size, "radius_m": radius,
                        "pos_xyz": pos, "quat_wxyz": quat, "cell": _cell_of(pos, cells)})
    objects.sort(key=lambda o: o["id"])

    doc = {
        "schema_version": SCHEMA_VERSION,
        "scene_name": str(scene_doc.get("name", "")),
        "scene_file": scene_file, "scene_sha256": scene_sha256,
        "scene_revision": revision, "sim_step": step, "sim_time_s": float(sim_time),
        "paused": bool(state.get("paused", False)),
        "frame": "scene (pedestal): x forward, y left, z up; metres",
        "quat_order": "wxyz",
        "arm_base_in_world": list(ARM_BASE_IN_WORLD),
        "exported_utc": exported_utc,
        "objects": objects,
        "cells": [{"name": n, "x": c.x, "y": c.y, "top_z": c.top_z, "half_extent": c.half_extent}
                  for n, c in sorted(cells.items())],
    }
    validate_truth(doc)
    return doc


def validate_truth(doc: Any) -> dict:
    """Strictly check a `truth.v1` document (unknown fields fail); returns it."""
    if not isinstance(doc, dict):
        raise TruthError("truth document must be an object")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise TruthError(f"schema_version {doc.get('schema_version')!r}, expected {SCHEMA_VERSION!r}")
    if set(doc) != _TOP_KEYS:
        raise TruthError(f"fields differ from {SCHEMA_VERSION}: "
                         + ", ".join(sorted(set(doc) ^ _TOP_KEYS)))
    if doc["quat_order"] != "wxyz":
        raise TruthError("quat_order must be 'wxyz'")
    _finite_vec(doc["arm_base_in_world"], 3, "arm_base_in_world")
    if not isinstance(doc["scene_revision"], str) or not doc["scene_revision"]:
        raise TruthError("scene_revision missing")
    if not isinstance(doc["sim_step"], int) or doc["sim_step"] < 0:
        raise TruthError("sim_step must be a non-negative integer")
    if not isinstance(doc["objects"], list) or len(doc["objects"]) > MAX_OBJECTS:
        raise TruthError("objects must be a list of at most %d" % MAX_OBJECTS)
    if not isinstance(doc["cells"], list) or len(doc["cells"]) > MAX_OBJECTS:
        raise TruthError("cells must be a list of at most %d" % MAX_OBJECTS)
    ids = set()
    for o in doc["objects"]:
        if not isinstance(o, dict) or set(o) != _OBJECT_KEYS:
            raise TruthError("object fields differ from the schema")
        if not isinstance(o["id"], str) or not o["id"] or len(o["id"]) > MAX_ID_LEN or o["id"] in ids:
            raise TruthError(f"bad or duplicate object id {o['id']!r}")
        ids.add(o["id"])
        if o["shape"] not in ("box", "cylinder"):
            raise TruthError(f"{o['id']}: shape {o['shape']!r}")
        _finite_vec(o["pos_xyz"], 3, f"{o['id']}: pos_xyz")
        _finite_vec(o["quat_wxyz"], 4, f"{o['id']}: quat_wxyz")
        size = _finite_vec(o["size_m"], 3, f"{o['id']}: size_m")
        if min(size) <= 0:
            raise TruthError(f"{o['id']}: non-positive size")
        if o["shape"] == "cylinder":
            _finite_vec([o["radius_m"]], 1, f"{o['id']}: radius_m")
        if o["cell"] is not None and not isinstance(o["cell"], str):
            raise TruthError(f"{o['id']}: cell must be a name or null")
    for c in doc["cells"]:
        if not isinstance(c, dict) or set(c) != _CELL_KEYS:
            raise TruthError("cell fields differ from the schema")
        _finite_vec([c["x"], c["y"], c["top_z"], c["half_extent"]], 4, f"cell {c.get('name')}")
    if len(json.dumps(doc)) > MAX_DOC_BYTES:
        raise TruthError("truth document exceeds its size limit")
    return doc


def file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# The read-only WebSocket client
# ---------------------------------------------------------------------------

async def _send(ws: Any, message: Mapping[str, Any]) -> None:
    """The only way this module writes to the socket, so the allow-list is enforced."""
    if message.get("type") not in ALLOWED_SEND_TYPES:
        raise TruthError(f"refusing to send {message.get('type')!r}: this client is read-only")
    await ws.send(json.dumps(message))


async def read_one_state(ws: Any, *, timeout_s: float) -> dict:
    """hello -> hello_ack -> answer heartbeats -> return the first `state`."""
    await _send(ws, {"type": "hello", "protocol_version": PROTOCOL_VERSION,
                     "client_id": CLIENT_ID})

    async def run() -> dict:
        ack = json.loads(await ws.recv())
        if ack.get("type") != "hello_ack":
            raise TruthError(f"expected hello_ack, got {ack.get('type')!r}")
        if ack.get("protocol_version") != PROTOCOL_VERSION:
            raise TruthError(f"server speaks protocol {ack.get('protocol_version')!r}, "
                             f"this client {PROTOCOL_VERSION}")
        while True:
            msg = json.loads(await ws.recv())
            kind = msg.get("type")
            if kind == "state":
                return msg
            if kind == "heartbeat":
                await _send(ws, {"type": "heartbeat_ack", "echo_ns": msg.get("sent_ns", 0)})
            elif kind == "shutdown":
                raise TruthError("the server is shutting down")
            # camera_frame, scene_ack, place_ack, error: not this client's job.

    try:
        return await asyncio.wait_for(run(), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise TruthError(f"no state from the simulator within {timeout_s:g} s") from exc


async def fetch_state(url: str, *, timeout_s: float = 10.0) -> dict:
    """Connect, read one fresh state, disconnect.  Raises TruthError, never blocks forever."""
    import websockets
    try:
        async with websockets.connect(url, ping_interval=None, max_size=MAX_WS_BYTES,
                                      open_timeout=timeout_s) as ws:
            return await read_one_state(ws, timeout_s=timeout_s)
    except TruthError:
        raise
    except Exception as exc:                       # refused, closed, bad JSON, timeout
        raise TruthError(f"{url}: {exc.__class__.__name__}: {exc}") from exc
