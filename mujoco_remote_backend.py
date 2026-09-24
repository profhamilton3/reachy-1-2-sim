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
* On disconnect: emit DEGRADED status. There is no live fallback to
  KinematicBackend — fake_reachy_server.py picks one writer per process at
  startup (REACHY_SIM_BACKEND), and nothing here switches that mid-session.

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
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Tuple

log = logging.getLogger(__name__)

_DEFAULT_URL = "ws://host.docker.internal:8765"
_RECONNECT_DELAY = 2.0     # seconds between reconnect attempts
_DEADLINE_MS = int(os.environ.get("REACHY_SIM_REMOTE_DEADLINE_MS", "1000"))
_MUJOCO_URL = os.environ.get("REACHY_SIM_MUJOCO_URL", _DEFAULT_URL)

_LEFT_JPG  = pathlib.Path("/tmp/reachy_left.jpg")
_RIGHT_JPG = pathlib.Path("/tmp/reachy_right.jpg")
# Named by whichever writer is actually producing the frames above right now
# — see web/camera_server.py's /status (issue #40) and camera_fixture.py's
# matching writer for the fixture backend.
_FRAME_META = pathlib.Path("/tmp/reachy_frame_meta.json")

from kinematic_backend import JOINT_DEFS, JointCommand, JointSample, SimulationSnapshot

# Must match native_mujoco/protocol.py — inlined here so this module
# runs inside Docker without access to the native server tree.
PROTOCOL_VERSION = 1


class ConnectionState(str, Enum):
    CONNECTING = "connecting"
    READY      = "ready"
    RESETTING  = "resetting"
    DEGRADED   = "degraded"
    ABORTED    = "aborted"
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
        # Re-entrant: _send drains the pending queue and builds the command
        # under one hold, and _build_command/latest_snapshot take it too.
        self._lock = threading.RLock()
        self._snapshot = RemoteSnapshot()
        self._conn_state = ConnectionState.CONNECTING
        self._cmd_seq = 0
        self._pending_cmds: List[JointCommand] = []
        self._pending_reset = False
        # R12-605: zoom level awaiting forwarding to the native server.  Holds
        # only the LAST requested level — the lens has no queue, and a client
        # that flips in→out→in before the next send tick wants the final state.
        self._pending_zoom: Optional[str] = None
        self._last_target: Optional[List[float]] = None
        # The goal each joint REPORTS (per protocol index): the latest goal the
        # bridge accepted, whether or not it has been sent yet.  None = no goal
        # known, which reports the present position.  Reporting the present
        # position unconditionally was the bug: reachy_sdk caches the reported
        # goal, starts every goto from it, and could echo it back as a setpoint.
        self._reported_goal: List[Optional[float]] = [None] * len(JOINT_DEFS)
        # The last compliance value this bridge SENT per joint (None = none
        # since the baseline).  A state saying "compliant" can arrive after a
        # stiffening command was sent; this keeps that stale flag from moving
        # the joint's target.
        self._cmd_compliant: List[Optional[bool]] = [None] * len(JOINT_DEFS)
        # Seeding the command baseline.  The native server starts, and restarts
        # after every reset, with each goal equal to the current pose
        # (sync_targets_to_current), so the bridge's baseline is the pose of the
        # first state of a session or the first post-reset state.
        self._seed_pending = True
        self._awaiting_reset = False
        self._reset_ack_step: Optional[int] = None
        self._last_sim_step: Optional[int] = None
        self._shutdown = threading.Event()
        self._reconnect_count = 0

        # Build uid→index map from kinematic_backend JOINT_DEFS once at construction
        # so _build_command() works before the first reset.
        self._uid_to_idx: Dict[int, int] = {}
        self._idx_to_uid: Dict[int, int] = {}
        for i, (name, uid) in enumerate(JOINT_DEFS):
            self._uid_to_idx[uid] = i
            self._idx_to_uid[i] = uid

        self._thread: Optional[threading.Thread] = None

        # Reset acknowledgement (Gate 8-D).
        self._pending_reset_id: Optional[str] = None
        self._reset_ack_event: threading.Event = threading.Event()
        # True once a real reset_ack matching _pending_reset_id has arrived,
        # False once the timeout aborts it, None while a reset is still in
        # flight (or none was ever requested).  _reset_ack_event alone can't
        # carry this: _on_reset_timeout also sets it, so a caller that only
        # checks "is the event set" cannot tell a genuine ack from a refused
        # or unacknowledged reset (B1) -- wait_for_reset() checks this too.
        self._reset_ok: Optional[bool] = None
        # Timeout for waiting on a reset ack (seconds).
        self._reset_timeout: float = float(
            os.environ.get("REACHY_SIM_RESET_TIMEOUT_S", "5.0")
        )

    def request_reset(self) -> threading.Event:
        """Ask the native server to reset the scene.

        Generates a unique request_id, enters RESETTING state (commands are
        held until the matching reset_ack arrives), and returns a threading.Event
        that is set when the ack arrives (or the timeout fires).  Callers that
        need to block until reset is confirmed can call event.wait(timeout).
        """
        rid = uuid.uuid4().hex
        with self._lock:
            self._pending_reset = True
            self._pending_reset_id = rid
            # A command still queued (not yet built/sent) belongs to the
            # pre-reset baseline -- the native server drops it too (#116).
            # Only commands submitted AFTER this point are held and sent
            # onto the post-reset pose (B2: these used to survive and be
            # replayed, and reported, against the wrong baseline).
            self._pending_cmds = []
            # Re-seed goals from the post-reset pose.  Commands wait for it.
            self._unseed_locked()
            self._awaiting_reset = True
            self._reset_ack_step = None
            self._conn_state = ConnectionState.RESETTING
            self._reset_ok = None
        self._reset_ack_event.clear()
        return self._reset_ack_event

    def wait_for_reset(self, timeout: Optional[float] = None) -> bool:
        """Block until a reset_ack arrives or the reset is aborted.

        Returns True only when a real reset_ack matching the request was
        received -- never on a timeout/abort, even though that also sets
        the event (so waiters don't hang; see _on_reset_timeout).
        """
        t = timeout if timeout is not None else self._reset_timeout
        self._reset_ack_event.wait(timeout=t)
        with self._lock:
            return self._reset_ok is True

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
            self._accept_goal_locked(cmd)

    def submit_commands(self, cmds: List[JointCommand]) -> None:
        with self._lock:
            self._pending_cmds.extend(cmds)
            for cmd in cmds:
                self._accept_goal_locked(cmd)

    def _accept_goal_locked(self, cmd: JointCommand) -> None:
        idx = self._uid_to_idx.get(cmd.uid)
        if idx is not None and cmd.goal_position is not None:
            self._reported_goal[idx] = float(cmd.goal_position)

    def _unseed_locked(self) -> None:
        """Forget every commanded and reported goal; the next baseline state re-seeds them."""
        self._last_target = None
        self._reported_goal = [None] * len(JOINT_DEFS)
        self._cmd_compliant = [None] * len(JOINT_DEFS)
        self._seed_pending = True

    def _seed_locked(self, positions: Dict[int, float]) -> None:
        """Baseline = this state's pose, with any goal still waiting to be sent on top."""
        base = [0.0] * len(JOINT_DEFS)
        for idx, pos in positions.items():
            base[idx] = pos
        self._last_target = list(base)
        self._reported_goal = list(base)
        for cmd in self._pending_cmds:
            self._accept_goal_locked(cmd)
        self._seed_pending = False
        self._awaiting_reset = False
        self._reset_ack_step = None

    def _joint_is_compliant_locked(self, idx: int) -> bool:
        """Compliant by the native state, and not since stiffened by this bridge."""
        if self._cmd_compliant[idx] is False:
            return False
        uid = self._idx_to_uid.get(idx)
        for s in self._snapshot.joints.values():
            if s.uid == uid:
                return s.compliant
        return False

    def request_zoom(self, level: str) -> None:
        """Ask the native server to change the camera zoom level (R12-605).

        `level` is an SDK ZoomLevelPossibilities name, lower-cased:
        in / out / inter / zero.  Fire-and-forget — the native server applies it
        on its render thread and logs a warning if it refuses; there is no ack,
        and a level sent while disconnected is simply dropped along with every
        other pending command.  Validation lives on the native side (zoom.py)
        so there is one source of truth for what the lens can do.
        """
        with self._lock:
            self._pending_zoom = str(level).strip().lower()

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

    def _on_reset_timeout(self, rid: Optional[str] = None) -> None:
        """Called from the asyncio event loop when a reset_ack does not arrive in time.

        Gated on `_pending_reset_id` (and, when known, the specific request id
        this timer was scheduled for) -- NOT on `_conn_state`, and (N1, PR #141
        re-review) NOT on `_awaiting_reset` either.  `_awaiting_reset` is
        cleared by `_seed_locked` as soon as the first post-reset state
        arrives, which under real traffic can happen well before its own
        genuine reset_ack (the native server queues states and acks
        separately).  Gating the timeout on it made this a no-op whenever
        that state won the race, so a reset whose ack was then lost outright
        would stay unresolved (`_reset_ok=None`) forever instead of settling
        to a bounded failure.  `_pending_reset_id` alone is still correct: it
        is cleared ONLY by a matching ack or a firing timeout, so a
        stale/mismatched timer (an earlier reset a reconnect or a later reset
        already resolved) still can't abort a different, still-pending reset.
        `rid=None` (manual/legacy call) matches whatever reset is pending.
        """
        with self._lock:
            if (self._pending_reset_id is not None
                    and (rid is None or self._pending_reset_id == rid)):
                log.error(
                    "Reset ack timed out after %.1f s — entering ABORTED state",
                    self._reset_timeout,
                )
                self._conn_state = ConnectionState.ABORTED
                self._pending_reset_id = None
                self._reset_ok = False
                # Whether the reset happened is unknown: take the very next
                # state as the baseline rather than waiting on a reset that
                # may never show (_track_goals_locked's post_reset check
                # includes `not self._awaiting_reset`).  Goals stay unseeded
                # (present) until that state arrives.  Idempotent: the state
                # may already have cleared this (see above), and the timeout
                # must still settle `_reset_ok` to False either way.
                self._awaiting_reset = False
                # Signal waiters so they don't hang; they can check
                # connection_state and wait_for_reset()'s return value (not
                # just that the event fired).  Scoped to the abort branch: a
                # stale/mismatched timer (an earlier reset a reconnect or a
                # later reset already resolved) must not wake a waiter on a
                # DIFFERENT, still-genuinely-pending reset early.
                self._reset_ack_event.set()

    async def _connect_loop(self) -> None:
        import websockets
        import websockets.exceptions as _ws_exc

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
            except (OSError, ConnectionError, _ws_exc.WebSocketException) as exc:
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
            # A new session may be a new native process whose goals are its
            # current pose.  Never replay the previous session's targets to it:
            # re-seed from this session's first state.
            self._unseed_locked()
            self._awaiting_reset = False
            self._reset_ack_step = None
            self._last_sim_step = None

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

                elif mtype == "reset_ack":
                    req_id = msg.get("request_id", "")
                    matched = False
                    with self._lock:
                        # Matched on _pending_reset_id alone, not
                        # _conn_state and (N1, PR #141 re-review) not
                        # _awaiting_reset either.
                        #
                        # Not _conn_state: _ingest_state flips _conn_state
                        # back to READY on any state message (including a
                        # pre-reset one that keeps arriving while a reset is
                        # refused/delayed), so a real ack can land well after
                        # _conn_state has already left RESETTING.
                        #
                        # Not _awaiting_reset: the native server queues
                        # states and acks separately, so the first post-reset
                        # state can arrive and seed the baseline -- which
                        # clears _awaiting_reset via _seed_locked -- before
                        # its OWN genuine ack does.  Gating the match on
                        # _awaiting_reset made that ordering fall into the
                        # "unexpected" branch below and leave a successful
                        # reset's _reset_ok stuck at None forever (N1).
                        # _pending_reset_id is cleared ONLY here or by a
                        # firing timeout, so it alone is still sufficient to
                        # refuse a stale/mismatched/duplicate/late
                        # (post-timeout) ack: those never carry the current
                        # pending id.
                        if (self._pending_reset_id is not None
                                and req_id == self._pending_reset_id):
                            if self._awaiting_reset:
                                self._reset_ack_step = int(msg.get("sim_step", 0))
                            self._conn_state = ConnectionState.READY
                            self._pending_reset_id = None
                            self._reset_ok = True
                            matched = True
                            # _last_target is already None; it is re-seeded from
                            # the first post-reset state (see _ingest_state).
                            log.info(
                                "Reset ack received (id=%s sim_step=%s)",
                                req_id, msg.get("sim_step"),
                            )
                        else:
                            log.warning(
                                "Unexpected reset_ack id=%s (expected %s)",
                                req_id, self._pending_reset_id,
                            )
                    # Only a matched ack wakes a waiter.  An unmatched/late
                    # one must not set the event early for a DIFFERENT,
                    # still-genuinely-pending reset -- that waiter would then
                    # read _reset_ok=None and report failure for its OWN,
                    # still-in-flight reset (PR #141 re-review, Non-blocking
                    # #1).  A matched-but-late (post-timeout) ack can't reach
                    # here at all: the timeout already cleared
                    # _pending_reset_id, so it falls into the else branch too.
                    if matched:
                        self._reset_ack_event.set()

                elif mtype == "shutdown":
                    log.info("Server sent shutdown: %s", msg.get("reason", ""))
                    return

                elif mtype == "error":
                    log.warning("Server error: [%s] %s",
                                msg.get("code"), msg.get("message"))

        async def _send() -> None:
            while not self._shutdown.is_set():
                with self._lock:
                    resetting = self._conn_state == ConnectionState.RESETTING
                    do_reset = self._pending_reset
                    reset_id = self._pending_reset_id if do_reset else None
                    if do_reset:
                        self._pending_reset = False
                    # Hold commands during reset — keep them in _pending_cmds —
                    # and until a baseline state has seeded the goals, so a
                    # command is never built on a pre-reset or previous-session
                    # pose.  Drained and built under this one hold, so a goal
                    # accepted meanwhile is never overwritten by the build.
                    built = None
                    if not resetting and self._last_target is not None and self._pending_cmds:
                        cmds = list(self._pending_cmds)
                        self._pending_cmds.clear()
                        built = self._build_command(cmds)

                if do_reset and reset_id:
                    await ws.send(json.dumps(
                        {"type": "reset", "request_id": reset_id}
                    ))
                    # Schedule a timeout: if no ack in time, abort.  Bound to
                    # this specific request id so a stale timer from an
                    # earlier reset (e.g. one a reconnect already resolved)
                    # can never abort a later, unrelated one.
                    asyncio.get_event_loop().call_later(
                        self._reset_timeout, self._on_reset_timeout, reset_id
                    )

                # R12-605: forward a requested zoom level.  Sent outside the
                # `cmds` block because zoom is independent of joint motion — a
                # client may zoom without ever commanding a joint.
                with self._lock:
                    zoom, self._pending_zoom = self._pending_zoom, None
                if zoom is not None:
                    await ws.send(json.dumps(
                        {"type": "zoom_command", "level": zoom, "camera": ""}
                    ))
                    log.info("Zoom command forwarded: %s", zoom)

                if built is not None:
                    target, compliant, speed, torque = built
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
                compliant=bool(j.get("compliant", False)),
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
        positions = {self._uid_to_idx[s.uid]: s.position_rad
                     for s in joints.values() if s.uid in self._uid_to_idx}
        with self._lock:
            self._snapshot = snap
            self._conn_state = ConnectionState.READY
            self._track_goals_locked(snap.sim_step, positions, joints)

    def _track_goals_locked(self, sim_step: int, positions: Dict[int, float],
                            joints: Dict[str, RemoteJointSample]) -> None:
        # sim_step only rises within an episode and restarts at 0 on a native
        # reset, so a step that does not rise marks the first post-reset state
        # -- whoever asked for the reset.  The reset_ack alone cannot: the
        # native server queues states and acks separately, so pre-reset states
        # can still arrive after the ack.
        restarted = self._last_sim_step is not None and sim_step <= self._last_sim_step
        self._last_sim_step = sim_step
        if restarted and not self._seed_pending:
            self._unseed_locked()
        if self._seed_pending:
            post_reset = (not self._awaiting_reset or restarted
                          or (self._reset_ack_step is not None
                              and sim_step <= self._reset_ack_step))
            if positions and post_reset:
                self._seed_locked(positions)
            return
        # A compliant joint's native goal tracks its pose, so its target does
        # too: stiffening it later holds it where it is instead of snapping it
        # back to a goal from before it went compliant.
        for s in joints.values():
            idx = self._uid_to_idx.get(s.uid)
            if idx is not None and s.compliant and self._cmd_compliant[idx] is not False:
                self._last_target[idx] = s.position_rad
                self._reported_goal[idx] = s.position_rad

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
        else:
            return
        _atomic_write(_FRAME_META, json.dumps(
            {"backend": "mujoco-remote", "wall_time_ns": time.time_ns()}
        ).encode())

    def _build_command(self, cmds: List[JointCommand]):
        """Merge per-UID SDK commands into 21-element protocol lists.

        Returns (target_rad, compliant, speed_limit_rad_s, torque_limit_percent).
        The three optional lists hold None for joints left unchanged (R12-501).
        """
        # Start from the LAST COMMANDED targets so joints not in this batch keep
        # their commanded goal.  (Seeding from current positions instead let
        # unspecified joints — e.g. the neck — ratchet toward wherever gravity
        # had dragged them, since every arm command re-sent their sagging
        # position as the goal.)  _send only builds once a baseline state has
        # seeded _last_target; a direct call before that seeds from the
        # current snapshot, as it always has.
        with self._lock:
            if self._last_target is None:
                positions = {self._uid_to_idx[s.uid]: s.position_rad
                             for s in self._snapshot.joints.values()
                             if s.uid in self._uid_to_idx}
                self._seed_locked(positions)
            present = {self._uid_to_idx[s.uid]: s.position_rad
                       for s in self._snapshot.joints.values()
                       if s.uid in self._uid_to_idx}
            target = list(self._last_target)

            compliant: List[Optional[bool]] = [None] * 21
            speed: List[Optional[float]] = [None] * 21
            torque: List[Optional[float]] = [None] * 21
            goal_later = {self._uid_to_idx.get(c.uid) for c in self._pending_cmds
                          if c.goal_position is not None}

            for k, cmd in enumerate(cmds):
                idx = self._uid_to_idx.get(cmd.uid)
                if idx is None:
                    continue
                if cmd.goal_position is not None:
                    target[idx] = float(cmd.goal_position)
                if cmd.compliant is not None:
                    compliant[idx] = bool(cmd.compliant)
                    # Stiffening a compliant joint with no goal holds it where
                    # it is -- what the SDK's own turn_on assumes when it sets
                    # its local goal to the present position -- rather than
                    # driving it back to a target from before it went
                    # compliant.  An already-stiff joint keeps its target.
                    if (cmd.compliant is False and cmd.goal_position is None
                            and self._joint_is_compliant_locked(idx) and idx in present):
                        target[idx] = present[idx]
                        later = any(c.goal_position is not None
                                    and self._uid_to_idx.get(c.uid) == idx
                                    for c in cmds[k + 1:])
                        if not later and idx not in goal_later:
                            self._reported_goal[idx] = present[idx]
                    self._cmd_compliant[idx] = bool(cmd.compliant)
                if cmd.speed_limit is not None:
                    speed[idx] = float(cmd.speed_limit)
                if cmd.torque_limit is not None:
                    torque[idx] = float(cmd.torque_limit)

            self._last_target = list(target)
            return target, compliant, speed, torque

    def reported_goals(self) -> List[float]:
        """Per protocol index, the goal each joint reports: the latest accepted
        goal, else the present position."""
        with self._lock:
            present = {self._uid_to_idx[s.uid]: s.position_rad
                       for s in self._snapshot.joints.values()
                       if s.uid in self._uid_to_idx}
            return [g if g is not None else present.get(i, 0.0)
                    for i, g in enumerate(self._reported_goal)]

    # ── SimulationSnapshot bridge (for fake_reachy_server compatibility) ─

    def to_kinematic_snapshot(self) -> SimulationSnapshot:
        """Convert the latest RemoteSnapshot to a kinematic_backend.SimulationSnapshot.

        The returned snapshot is keyed by joint name and uses JointSample objects,
        matching exactly what KinematicBackend.latest_snapshot() returns — so
        FakeJointService and state_file_writer work with either backend.
        """
        with self._lock:
            remote = self.latest_snapshot()
            goals = self.reported_goals()
        joints: Dict[str, JointSample] = {}
        seen_names: set = set()

        for name, s in remote.joints.items():
            seen_names.add(name)
            idx = self._uid_to_idx.get(s.uid)
            joints[name] = JointSample(
                name=name,
                uid=s.uid,
                present_position=s.position_rad,
                # The goal the joint was last given (see _reported_goal), not
                # its present position.
                goal_position=goals[idx] if idx is not None else s.position_rad,
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
