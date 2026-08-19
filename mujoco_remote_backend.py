"""
R12-403: MuJoCo remote backend — Docker-side WebSocket client.

Connects to the native macOS MuJoCo server at
ws://host.docker.internal:8765 (or REACHY_SIM_MUJOCO_URL).

Responsibilities
----------------
* Maintain the WebSocket connection with heartbeat and auto-reconnect.
* Translate kinematic_backend JointCommand objects → protocol.JointCommand.
* Consume state messages and expose the latest SimulationSnapshot.
* Consume camera_frame messages and write /tmp/reachy_{left,right}.jpg so the
  existing gRPC CameraService and ROS publisher keep working unchanged.
* Expose a `latest_snapshot()` compatible with the existing backend interface.
* On disconnect: emit DEGRADED status, optionally fall back to KinematicBackend
  if REACHY_SIM_ALLOW_FIXTURE_FALLBACK=true.

This file is imported inside the Docker container (Python 3.8, no mujoco).
It must not import mujoco or any native-server module.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import pathlib
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Tuple

log = logging.getLogger(__name__)

_DEFAULT_URL = "ws://host.docker.internal:8765"
_RECONNECT_DELAY = 2.0     # seconds between reconnect attempts
_DEADLINE_MS = int(os.environ.get("REACHY_SIM_REMOTE_DEADLINE_MS", "1000"))
_ALLOW_FALLBACK = os.environ.get("REACHY_SIM_ALLOW_FIXTURE_FALLBACK", "false").lower() == "true"
_MUJOCO_URL = os.environ.get("REACHY_SIM_MUJOCO_URL", _DEFAULT_URL)

_LEFT_JPG  = pathlib.Path("/tmp/reachy_left.jpg")
_RIGHT_JPG = pathlib.Path("/tmp/reachy_right.jpg")

from kinematic_backend import JOINT_DEFS, JointCommand, JointSample, SimulationSnapshot

# Must match native_mujoco/protocol.py — inlined here so this module
# runs inside Docker without access to the native server tree.
PROTOCOL_VERSION = 1


class ConnectionState(str, Enum):
    CONNECTING = "connecting"
    READY      = "ready"
    DEGRADED   = "degraded"
    STOPPED    = "stopped"


@dataclass
class RemoteJointSample:
    name: str
    uid: int
    position_rad: float = 0.0
    velocity_rad_s: float = 0.0
    effort: float = 0.0
    compliant: bool = False


@dataclass
class RemoteSnapshot:
    seq: int = 0
    sim_step: int = 0
    sim_time_s: float = 0.0
    wall_time_ns: int = 0
    scene_revision: str = ""
    state: ConnectionState = ConnectionState.CONNECTING
    joints: Dict[str, RemoteJointSample] = field(default_factory=dict)
    # R12-502: SDK force-sensor uid -> grip force (N); per-side grasp flags.
    force_sensors: Dict[int, float] = field(default_factory=dict)
    grasping: Dict[str, bool] = field(default_factory=dict)
    # R12-503: tracked object poses keyed by object id (for RViz/browser).
    objects: Dict[str, dict] = field(default_factory=dict)
    # R12-504: interactive control on/off states keyed by control id.
    interactive: Dict[str, dict] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()

    def age_ms(self) -> float:
        if self.wall_time_ns == 0:
            return float("inf")
        return (time.monotonic_ns() - self.wall_time_ns) / 1_000_000


class MujocoRemoteBackend:
    """Docker-side backend that proxies commands to the native MuJoCo server."""

    def __init__(self, url: str = _MUJOCO_URL) -> None:
        self._url = url
        self._lock = threading.Lock()
        self._snapshot = RemoteSnapshot()
        self._conn_state = ConnectionState.CONNECTING
        self._cmd_seq = 0
        self._pending_cmds: List[JointCommand] = []
        self._shutdown = threading.Event()
        self._reconnect_count = 0

        # build uid→index map from kinematic_backend JOINT_DEFS
        self._uid_to_idx: Dict[int, int] = {}
        self._idx_to_uid: Dict[int, int] = {}
        for i, (name, uid) in enumerate(JOINT_DEFS):
            self._uid_to_idx[uid] = i
            self._idx_to_uid[i] = uid

        self._thread: Optional[threading.Thread] = None

    # ── Public API (same shape as KinematicBackend) ───────────────────────

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run_forever, daemon=True, name="mujoco-remote"
        )
        self._thread.start()
        log.info("MujocoRemoteBackend started, connecting to %s", self._url)

    def stop(self) -> None:
        self._shutdown.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        log.info("MujocoRemoteBackend stopped")

    def submit_command(self, cmd: JointCommand) -> None:
        with self._lock:
            self._pending_cmds.append(cmd)

    def submit_commands(self, cmds: List[JointCommand]) -> None:
        with self._lock:
            self._pending_cmds.extend(cmds)

    def latest_snapshot(self) -> RemoteSnapshot:
        with self._lock:
            snap = self._snapshot
        if snap.age_ms() > _DEADLINE_MS:
            with self._lock:
                self._conn_state = ConnectionState.DEGRADED
        return snap

    @property
    def connection_state(self) -> ConnectionState:
        with self._lock:
            return self._conn_state

    def uid_by_name(self, name: str) -> Optional[int]:
        for n, uid in JOINT_DEFS:
            if n == name:
                return uid
        return None

    # ── Internal event loop in background thread ──────────────────────────

    def _run_forever(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._connect_loop())
        finally:
            loop.close()

    async def _connect_loop(self) -> None:
        import websockets

        while not self._shutdown.is_set():
            try:
                log.info("Connecting to %s (attempt %d)",
                         self._url, self._reconnect_count + 1)
                async with websockets.connect(
                    self._url,
                    ping_interval=None,   # we handle heartbeat ourselves
                    close_timeout=3,
                ) as ws:
                    self._reconnect_count += 1
                    await self._session(ws)
            except Exception as exc:
                log.warning("Connection error: %s — retry in %.1fs",
                            exc, _RECONNECT_DELAY)
                with self._lock:
                    self._conn_state = ConnectionState.DEGRADED
            if not self._shutdown.is_set():
                await asyncio.sleep(_RECONNECT_DELAY)

    async def _session(self, ws) -> None:
        # Handshake
        hello = {"type": "hello", "protocol_version": PROTOCOL_VERSION,
                 "client_id": "docker-core"}
        await ws.send(json.dumps(hello))
        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        ack = json.loads(raw)
        if ack.get("type") != "hello_ack":
            raise RuntimeError(f"Unexpected handshake reply: {ack.get('type')}")
        log.info("Connected to MuJoCo server (sim_fps=%.0f camera_fps=%.0f)",
                 ack.get("sim_fps", 0), ack.get("camera_fps", 0))
        with self._lock:
            self._conn_state = ConnectionState.READY

        last_hb = time.monotonic()

        async def _recv() -> None:
            nonlocal last_hb
            async for raw in ws:
                if self._shutdown.is_set():
                    break
                msg = json.loads(raw)
                mtype = msg.get("type")

                if mtype == "state":
                    self._ingest_state(msg)

                elif mtype == "camera_frame":
                    self._ingest_camera_frame(msg)

                elif mtype == "heartbeat":
                    last_hb = time.monotonic()
                    await ws.send(json.dumps({
                        "type": "heartbeat_ack",
                        "echo_ns": msg.get("sent_ns", 0),
                    }))

                elif mtype == "heartbeat_ack":
                    last_hb = time.monotonic()

                elif mtype == "shutdown":
                    log.info("Server sent shutdown: %s", msg.get("reason", ""))
                    return

                elif mtype == "error":
                    log.warning("Server error: [%s] %s",
                                msg.get("code"), msg.get("message"))

        async def _send() -> None:
            while not self._shutdown.is_set():
                with self._lock:
                    cmds = list(self._pending_cmds)
                    self._pending_cmds.clear()

                if cmds:
                    target, compliant, speed, torque = self._build_command(cmds)
                    self._cmd_seq += 1
                    cmd_msg = {
                        "type": "joint_command",
                        "seq": self._cmd_seq,
                        "target_rad": target,
                        "compliant": compliant,
                        "speed_limit_rad_s": speed,
                        "torque_limit_percent": torque,
                    }
                    await ws.send(json.dumps(cmd_msg))

                # periodic heartbeat from our side
                if time.monotonic() - last_hb > 2.0:
                    await ws.send(json.dumps(
                        {"type": "heartbeat",
                         "sent_ns": time.monotonic_ns()}
                    ))

                await asyncio.sleep(0.02)

        await asyncio.gather(_recv(), _send())

    # ── Ingest helpers ───────────────────────────────────────────────────

    def _ingest_state(self, msg: dict) -> None:
        joints: Dict[str, RemoteJointSample] = {}
        for j in msg.get("joints") or []:
            name = j.get("name", "")
            uid = j.get("uid", 0)
            sample = RemoteJointSample(
                name=name,
                uid=uid,
                position_rad=float(j.get("position_rad", 0.0)),
                velocity_rad_s=float(j.get("velocity_rad_s", 0.0)),
                effort=float(j.get("effort", 0.0)),
            )
            joints[name] = sample

        force_sensors = {
            int(fs.get("uid")): float(fs.get("force", 0.0))
            for fs in (msg.get("force_sensors") or [])
            if fs.get("uid") is not None
        }
        grasping = {
            str(g.get("side")): bool(g.get("grasping", False))
            for g in (msg.get("grippers") or [])
        }
        objects = {
            str(o.get("object_id")): {
                "pos_xyz": o.get("pos_xyz", [0.0, 0.0, 0.0]),
                "quat_wxyz": o.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
            }
            for o in (msg.get("objects") or [])
            if o.get("object_id") is not None
        }
        interactive = {
            str(c.get("id")): {
                "type": c.get("type", ""),
                "on": bool(c.get("on", False)),
                "value": float(c.get("value", 0.0)),
            }
            for c in (msg.get("interactive") or [])
            if c.get("id") is not None
        }
        snap = RemoteSnapshot(
            seq=int(msg.get("seq", 0)),
            sim_step=int(msg.get("sim_step", 0)),
            sim_time_s=float(msg.get("sim_time_s", 0.0)),
            wall_time_ns=time.monotonic_ns(),
            scene_revision=str(msg.get("scene_revision", "")),
            state=ConnectionState.READY,
            joints=joints,
            force_sensors=force_sensors,
            grasping=grasping,
            objects=objects,
            interactive=interactive,
        )
        with self._lock:
            self._snapshot = snap
            self._conn_state = ConnectionState.READY

    def _ingest_camera_frame(self, msg: dict) -> None:
        cam = msg.get("camera", "")
        b64 = msg.get("jpeg_b64", "")
        if not b64:
            return
        try:
            jpeg = base64.b64decode(b64)
        except Exception:
            return

        if cam == "left_camera":
            _atomic_write(_LEFT_JPG, jpeg)
        elif cam == "right_camera":
            _atomic_write(_RIGHT_JPG, jpeg)

    def _build_command(self, cmds: List[JointCommand]):
        """Merge per-UID SDK commands into 21-element protocol lists.

        Returns (target_rad, compliant, speed_limit_rad_s, torque_limit_percent).
        The three optional lists hold None for joints left unchanged (R12-501).
        """
        with self._lock:
            snap = self._snapshot

        # Start targets from current positions so unspecified joints hold.
        target = [0.0] * 21
        for name, sample in snap.joints.items():
            idx = self._uid_to_idx.get(sample.uid)
            if idx is not None:
                target[idx] = sample.position_rad

        compliant: List[Optional[bool]] = [None] * 21
        speed: List[Optional[float]] = [None] * 21
        torque: List[Optional[float]] = [None] * 21

        for cmd in cmds:
            idx = self._uid_to_idx.get(cmd.uid)
            if idx is None:
                continue
            if cmd.goal_position is not None:
                target[idx] = float(cmd.goal_position)
            if cmd.compliant is not None:
                compliant[idx] = bool(cmd.compliant)
            if cmd.speed_limit is not None:
                speed[idx] = float(cmd.speed_limit)
            if cmd.torque_limit is not None:
                torque[idx] = float(cmd.torque_limit)

        return target, compliant, speed, torque

    # ── SimulationSnapshot bridge (for fake_reachy_server compatibility) ─

    def to_kinematic_snapshot(self) -> SimulationSnapshot:
        """Convert the latest RemoteSnapshot to a kinematic_backend.SimulationSnapshot.

        The returned snapshot is keyed by joint name and uses JointSample objects,
        matching exactly what KinematicBackend.latest_snapshot() returns — so
        FakeJointService and state_file_writer work with either backend.
        """
        remote = self.latest_snapshot()
        joints: Dict[str, JointSample] = {}
        seen_names: set = set()

        for name, s in remote.joints.items():
            seen_names.add(name)
            joints[name] = JointSample(
                name=name,
                uid=s.uid,
                present_position=s.position_rad,
                goal_position=s.position_rad,
                present_speed=s.velocity_rad_s,
                present_load=s.effort,
                temperature=35.0,
                compliant=s.compliant,
                speed_limit=0.0,
                torque_limit=100.0,
            )

        # Fill any joints the server hasn't reported yet with safe defaults.
        for def_name, uid in JOINT_DEFS:
            if def_name not in seen_names:
                joints[def_name] = JointSample(
                    name=def_name,
                    uid=uid,
                    present_position=0.0,
                    goal_position=0.0,
                    present_speed=0.0,
                    present_load=0.0,
                    temperature=35.0,
                    compliant=uid not in (33, 34),  # antennae start stiff
                    speed_limit=0.0,
                    torque_limit=100.0,
                )

        return SimulationSnapshot(
            sequence=remote.seq,
            sim_step=remote.sim_step,
            sim_time_s=remote.sim_time_s,
            wall_time_ns=remote.wall_time_ns if remote.wall_time_ns else time.monotonic_ns(),
            joints=joints,
        )


class KinematicBridge:
    """Presents the KinematicBackend interface over a MujocoRemoteBackend.

    Used by fake_reachy_server.py so FakeJointService and state_file_writer
    work identically whether the active backend is kinematic or mujoco-remote.
    """

    def __init__(self, remote: MujocoRemoteBackend) -> None:
        self._r = remote

    def latest_snapshot(self) -> SimulationSnapshot:
        return self._r.to_kinematic_snapshot()

    def submit_command(self, cmd: JointCommand) -> None:
        self._r.submit_command(cmd)

    def uid_by_name(self, name: str) -> Optional[int]:
        return self._r.uid_by_name(name)


def _atomic_write(path: pathlib.Path, data: bytes) -> None:
    """Write bytes atomically via a temp file."""
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    except OSError as exc:
        log.debug("Failed to write %s: %s", path, exc)
