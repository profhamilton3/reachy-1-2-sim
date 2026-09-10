"""Read-only WebSocket link from the command panel to the native simulator (#50).

WHY THE PANEL NEEDS ITS OWN CLIENT
----------------------------------
`/scene` serves the scene *file*: grid cells and placeable ids, no tags and no
live poses.  A planner reading only that cannot tell a can sitting on r2c2 from
one parked on the floor in the pool, which is exactly the distinction "put the
recycle item in the bin" turns on.  The authoritative answer is in the `state`
messages the simulator pushes, so the coordinator subscribes to them.

WHY A SECOND CLIENT IS SAFE
---------------------------
`native_mujoco/server.py` keeps one state, frame, and place_ack queue PER
CONNECTION, after a measured bug where a shared queue split the stream between
clients and delivered one client's placement ack to another.  So this link
takes nothing away from the browser panel's own socket.

WHAT IT IS ALLOWED TO SEND
--------------------------
`hello`, `heartbeat_ack`, and the two execution-lease messages — nothing else.

Never `place_object`: that teleports scene state, it is scene setup rather than
a grasp, and using it to report task success is the specific false claim the
design brief forbids.  No `joint_command`, no `reset`, no `pause`, no
`scene_load` either.  Nothing here moves anything or edits the scene.

`acquire_control` / `release_control` (issue #51) are the exception, and they
are not motion: they ask the SERVER to refuse everyone else's scene edits while
a task runs.  The lease names a separate `motion_client_id` — the SDK bridge —
precisely because this link does not do the moving.

HEARTBEATS ARE NOT OPTIONAL
---------------------------
The server drops any client it has not heard from within 6 s (`_HB_DEADLINE`).
A link that only listened would go silent, be dropped, and leave the planner
reading a snapshot that stopped updating — with nothing saying so.  That exact
bug already cost the browser panel its socket once.  This answers every
heartbeat, and treats a snapshot older than `stale_after_s` as not live.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import queue
import uuid
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

#: Where the simulator is.  In the compatibility container the sim runs on the
#: HOST, so this is `host.docker.internal` and never `localhost` — the same
#: variable docker-compose already hands the mujoco-remote backend.
DEFAULT_URL = os.environ.get("REACHY_SIM_MUJOCO_URL", "ws://127.0.0.1:8765")

PROTOCOL_VERSION = 1
_CLIENT_ID = "command-panel"
_RECONNECT_DELAY_S = 2.0
_HELLO_TIMEOUT_S = 5.0
#: A snapshot older than this is not "live" any more.  The server pushes state
#: at roughly 30 Hz, so two seconds is many missed frames, not a slow tick.
STALE_AFTER_S = 2.0

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class SimSnapshot:
    """One coherent view of the simulated world.

    Immutable, and replaced wholesale rather than mutated, so a planner that
    grabbed a snapshot reasons over one instant even while the next arrives.
    """

    sim_step: int = 0
    sim_time_s: float = 0.0
    scene_revision: str = ""
    seq: int = 0
    objects: Dict[str, Vec3] = field(default_factory=dict)
    received_at: float = 0.0            # time.monotonic() when ingested
    dropped_objects: Tuple[str, ...] = ()   # ids whose pose was not finite

    def age_s(self) -> float:
        return max(0.0, time.monotonic() - self.received_at)


class SimLink:
    """Background asyncio client that keeps the latest `state` message.

    Follows the pattern already established by `mujoco_remote_backend.py`: one
    thread owning one event loop, manual heartbeats, and reconnect on loss.
    It is deliberately much smaller — no commands, no camera frames, no file
    writes, no reset handling.
    """

    def __init__(self, url: str = "", *, stale_after_s: float = STALE_AFTER_S) -> None:
        self.url = url or DEFAULT_URL
        self._stale_after = stale_after_s
        self._lock = threading.Lock()
        self._snapshot: Optional[SimSnapshot] = None
        # never_started|connecting|connected|unavailable|stopped.  The first
        # value matters: the request that lazily builds the panel also starts
        # this link, and asking for the status in that same request must not
        # report "stopped" for a link that is about to come up.
        self._state = "never_started"
        self._detail = ""
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Live socket and outstanding lease requests, both owned by the link
        # thread's loop.  Requests are keyed by request_id so a control_ack
        # that arrives interleaved with state messages is matched to the caller
        # that asked for it rather than to whoever happens to be waiting.
        self._ws = None
        self._pending: Dict[str, "queue.Queue"] = {}
        self._server_capabilities: Dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        # Set synchronously, before the thread exists, so a status read in the
        # same request that started the link tells the truth about it.
        self._set_state("connecting", "starting up")
        self._thread = threading.Thread(
            target=self._run, name="panel-sim-link", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(lambda: None)   # wake the loop
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None
        with self._lock:
            self._state = "stopped"

    # -- reads -------------------------------------------------------------

    def snapshot(self) -> Optional[SimSnapshot]:
        """The latest snapshot, or None when there is no live one.

        A stale snapshot is reported as no snapshot rather than as data: the
        whole point of this link is to stop the planner asserting things about
        the board it cannot currently see.
        """
        with self._lock:
            snap = self._snapshot
        if snap is None or snap.age_s() > self._stale_after:
            return None
        return snap

    @property
    def status(self) -> dict:
        with self._lock:
            snap = self._snapshot
            state, detail = self._state, self._detail
        live = snap is not None and snap.age_s() <= self._stale_after
        return {
            "state": state,
            "live": live,
            "detail": detail,
            "url": self.url,
            "age_s": round(snap.age_s(), 2) if snap else None,
            "scene_revision": snap.scene_revision if snap else "",
            "sim_step": snap.sim_step if snap else 0,
        }

    # -- internals ---------------------------------------------------------

    def _set_state(self, state: str, detail: str = "") -> None:
        with self._lock:
            self._state = state
            self._detail = detail

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._connect_loop())
        except Exception as exc:                      # never kill the process
            self._set_state("unavailable", f"{exc.__class__.__name__}: {exc}")
        finally:
            try:
                loop.close()
            finally:
                self._loop = None

    async def _connect_loop(self) -> None:
        try:
            import websockets
        except ImportError as exc:
            # The camera page must keep serving frames without this.  Missing
            # the library degrades the panel to scene-file semantics, which is
            # stage 1's behaviour and already says what it cannot see.
            self._set_state("unavailable", f"websockets not installed: {exc}")
            return

        while not self._stop.is_set():
            self._set_state("connecting")
            try:
                async with websockets.connect(
                    self.url,
                    ping_interval=None,      # the server's own heartbeat is used
                    max_size=8 * 1024 * 1024,
                    open_timeout=_HELLO_TIMEOUT_S,
                ) as ws:
                    await self._session(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._set_state("connecting", f"{exc.__class__.__name__}: {exc}")
            with self._lock:
                self._snapshot = None    # never present a stale world as live
            if self._stop.is_set():
                break
            await asyncio.sleep(_RECONNECT_DELAY_S)
        self._set_state("stopped")

    async def _session(self, ws) -> None:
        await ws.send(json.dumps({
            "type": "hello",
            "protocol_version": PROTOCOL_VERSION,
            "client_id": _CLIENT_ID,
        }))
        raw = await asyncio.wait_for(ws.recv(), timeout=_HELLO_TIMEOUT_S)
        ack = json.loads(raw)
        if ack.get("type") != "hello_ack":
            raise RuntimeError(f"expected hello_ack, got {ack.get('type')!r}")
        self._set_state("connected", f"server {ack.get('server_version', '?')}")
        self._server_capabilities = dict(ack.get("capabilities") or {})
        self._ws = ws
        try:
            await self._pump(ws)
        finally:
            self._ws = None
            # A dropped socket drops the lease with it, server-side.  Fail any
            # caller still waiting rather than leaving it blocked for its full
            # timeout on an answer that can no longer come.
            self._fail_pending("the link to the simulator dropped")

    async def _pump(self, ws) -> None:
        while not self._stop.is_set():
            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == "control_ack":
                waiter = self._pending.pop(str(msg.get("request_id") or ""), None)
                if waiter is not None:
                    waiter.put(msg)
                continue
            if mtype == "state":
                self._ingest_state(msg)
            elif mtype == "heartbeat":
                await ws.send(json.dumps({
                    "type": "heartbeat_ack",
                    "echo_ns": msg.get("sent_ns", 0),
                }))
            elif mtype == "shutdown":
                self._set_state("connecting", "server is shutting down")
                return
            # camera_frame, place_ack, scene_ack, error: not this link's job.

    # -- execution lease ---------------------------------------------------

    @property
    def supports_lease(self) -> bool:
        """Whether the connected server advertises arbitration.

        Asked rather than assumed: against an older server `acquire_control`
        is an unknown message that is quietly ignored, and a caller that took
        silence for a grant would execute with the scene wide open.
        """
        return bool(self._server_capabilities.get("execution_lease"))

    def acquire_control(self, motion_client_id: str, *, reason: str = "",
                        ttl_s: float = 120.0,
                        timeout: float = 5.0) -> Tuple[bool, str]:
        """Ask the server to refuse everyone else's scene edits.  (granted, why_not)"""
        if not self.supports_lease:
            return False, ("this simulator does not support the execution "
                           "lease, so scene edits cannot be arbitrated")
        ok, reply = self._request({
            "type": "acquire_control",
            "client_id": _CLIENT_ID,
            "motion_client_id": motion_client_id,
            "reason": reason,
            "ttl_s": ttl_s,
        }, timeout)
        if not ok:
            return False, reply
        if reply.get("granted"):
            return True, ""
        return False, (reply.get("error")
                       or f"the lease is held by {reply.get('holder') or 'another client'}")

    def release_control(self, timeout: float = 5.0) -> None:
        """Give the lease back.  Best effort: the server also releases it when
        this socket closes and when its TTL expires, so a failure here delays
        the unfreeze rather than losing it."""
        self._request({"type": "release_control"}, timeout)

    def _request(self, message: dict, timeout: float) -> Tuple[bool, Any]:
        loop, ws = self._loop, self._ws
        if loop is None or ws is None:
            return False, "not connected to the simulator"
        request_id = uuid.uuid4().hex
        message = dict(message, request_id=request_id)
        waiter: "queue.Queue" = queue.Queue(maxsize=1)
        self._pending[request_id] = waiter
        try:
            future = asyncio.run_coroutine_threadsafe(
                ws.send(json.dumps(message)), loop
            )
            future.result(timeout=timeout)
            return True, waiter.get(timeout=timeout)
        except queue.Empty:
            return False, "the simulator did not answer in time"
        except Exception as exc:
            return False, f"{exc.__class__.__name__}: {exc}"
        finally:
            self._pending.pop(request_id, None)

    def _fail_pending(self, why: str) -> None:
        for request_id in list(self._pending):
            waiter = self._pending.pop(request_id, None)
            if waiter is not None:
                try:
                    waiter.put_nowait({"granted": False, "error": why})
                except queue.Full:      # pragma: no cover - maxsize 1, one put
                    pass

    def _ingest_state(self, msg: dict) -> None:
        objects: Dict[str, Vec3] = {}
        dropped = []
        for entry in msg.get("objects") or []:
            oid = entry.get("object_id")
            pos = entry.get("pos_xyz")
            if not isinstance(oid, str) or not oid:
                continue
            # A NaN or inf pose must never reach the planner: it would compare
            # false against every tolerance and quietly make a proposal that
            # can never be revalidated.  Drop it and say which.
            if (not isinstance(pos, (list, tuple)) or len(pos) != 3
                    or not all(isinstance(v, (int, float)) and math.isfinite(v)
                               for v in pos)):
                dropped.append(oid)
                continue
            objects[oid] = (float(pos[0]), float(pos[1]), float(pos[2]))

        snap = SimSnapshot(
            sim_step=int(msg.get("sim_step") or 0),
            sim_time_s=float(msg.get("sim_time_s") or 0.0),
            scene_revision=str(msg.get("scene_revision") or ""),
            seq=int(msg.get("seq") or 0),
            objects=objects,
            received_at=time.monotonic(),
            dropped_objects=tuple(dropped),
        )
        with self._lock:
            self._snapshot = snap
