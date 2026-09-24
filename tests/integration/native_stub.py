"""A loopback stand-in for the native MuJoCo server, for offline bridge tests.

Speaks the same websocket protocol mujoco_remote_backend.py talks to
(hello/hello_ack, 50 Hz `state`, `joint_command`, `reset`/`reset_ack`,
heartbeats), so the bridge's real `_connect_loop`/`_session`/`_send`/`_recv`
run unmodified.  Behind it is a toy plant, not physics: each stiff joint moves
first-order toward `target + bias` (the bias stands in for gravity sag), a
compliant joint stays where it is unless a test moves it "by hand".

Every command and every state sent is logged, which is what the tests assert
on -- the equivalent of the native recorder's commands.jsonl / states.jsonl.
No mujoco, no Docker, no native server tree.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Dict, List, Optional

from kinematic_backend import JOINT_DEFS

N = len(JOINT_DEFS)
IDX = {name: i for i, (name, _) in enumerate(JOINT_DEFS)}


class NativeStub:
    STATE_HZ = 50
    SUBSTEPS = 10            # sim steps per state, as the native server's 500/50
    TAU_S = 0.05             # first-order time constant of a stiff joint

    def __init__(self, pose: Optional[Dict[str, float]] = None,
                 bias: Optional[Dict[str, float]] = None,
                 compliant: bool = True) -> None:
        self.lock = threading.Lock()
        self.pos = [0.0] * N
        for name, v in (pose or {}).items():
            self.pos[IDX[name]] = v
        self.bias = [0.0] * N
        for name, v in (bias or {}).items():
            self.bias[IDX[name]] = v
        self.compliant = [compliant] * N
        self.initial = list(self.pos)
        self.target = list(self.pos)      # sync_targets_to_current at startup
        self.step = 0
        self.seq = 0
        self.cmd_seq = 0
        self.commands: List[dict] = []    # {t, seq, target, compliant}
        self.states: List[dict] = []      # {t, sim_step, pos}
        self.resets = 0
        self._conns: set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop: Optional[asyncio.Event] = None
        self._thread: Optional[threading.Thread] = None
        self.port: Optional[int] = None
        self._ready = threading.Event()

    # ── test-side controls ────────────────────────────────────────────────
    def move_by_hand(self, name: str, value: float) -> None:
        with self.lock:
            self.pos[IDX[name]] = value

    def restart(self, pose: Dict[str, float]) -> None:
        """Pretend the native process restarted: new pose, fresh goals and step
        counter, everything compliant again -- and drop every connection."""
        with self.lock:
            for name, v in pose.items():
                self.pos[IDX[name]] = v
            self.target = list(self.pos)
            self.compliant = [True] * N
            self.step = 0
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._close_all(), self._loop).result(2)

    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start(self) -> "NativeStub":
        self._thread = threading.Thread(target=self._run, daemon=True, name="native-stub")
        self._thread.start()
        assert self._ready.wait(5), "native stub did not start"
        return self

    def stop(self) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(5)

    def _run(self) -> None:
        import websockets
        loop = asyncio.new_event_loop()
        self._loop = loop

        async def main():
            self._stop = asyncio.Event()
            async with websockets.serve(self._handler, "127.0.0.1", 0) as srv:
                self.port = next(iter(srv.sockets)).getsockname()[1]
                ticker = asyncio.ensure_future(self._tick())
                self._ready.set()
                await self._stop.wait()
                ticker.cancel()

        loop.run_until_complete(main())
        loop.close()

    async def _close_all(self) -> None:
        for ws in list(self._conns):
            await ws.close()

    # ── plant + state broadcast ───────────────────────────────────────────
    async def _tick(self) -> None:
        dt_state = 1.0 / self.STATE_HZ
        dt = dt_state / self.SUBSTEPS
        a = min(1.0, dt / self.TAU_S)
        while True:
            await asyncio.sleep(dt_state)
            with self.lock:
                # State first, then physics: like the native server, the state
                # published right after a reset (sim_step 0) is the reset pose.
                self.seq += 1
                msg = {
                    "type": "state", "seq": self.seq, "sim_step": self.step,
                    "sim_time_s": self.step * dt, "cmd_seq": self.cmd_seq,
                    "joints": [{"name": n, "uid": u, "position_rad": self.pos[i],
                                "velocity_rad_s": 0.0, "effort": 0.0,
                                "compliant": self.compliant[i]}
                               for i, (n, u) in enumerate(JOINT_DEFS)],
                }
                self.states.append({"t": time.monotonic(), "sim_step": self.step,
                                    "pos": list(self.pos)})
                for _ in range(self.SUBSTEPS):
                    for i in range(N):
                        if not self.compliant[i]:
                            self.pos[i] += (self.target[i] + self.bias[i] - self.pos[i]) * a
                self.step += self.SUBSTEPS
            raw = json.dumps(msg)
            for ws in list(self._conns):
                try:
                    await ws.send(raw)
                except Exception:
                    pass

    async def _handler(self, ws) -> None:
        hello = json.loads(await ws.recv())
        assert hello.get("type") == "hello"
        await ws.send(json.dumps({"type": "hello_ack", "sim_fps": 500,
                                  "camera_fps": 0, "num_joints": N}))
        self._conns.add(ws)
        try:
            async for raw in ws:
                msg = json.loads(raw)
                t = msg.get("type")
                if t == "joint_command":
                    with self.lock:
                        tgt = msg.get("target_rad") or []
                        cmp_ = msg.get("compliant") or [None] * N
                        for i in range(N):
                            if cmp_[i] is not None:
                                self.compliant[i] = bool(cmp_[i])
                            if i < len(tgt):
                                self.target[i] = float(tgt[i])
                        self.cmd_seq = msg.get("seq", self.cmd_seq)
                        self.commands.append({"t": time.monotonic(), "seq": msg.get("seq"),
                                              "target": list(self.target),
                                              "compliant": list(cmp_)})
                elif t == "reset":
                    with self.lock:          # like _reset_physics: back to the initial pose
                        self.step = 0
                        self.pos = list(self.initial)
                        self.target = list(self.pos)
                        self.resets += 1
                    await ws.send(json.dumps({"type": "reset_ack",
                                              "request_id": msg.get("request_id", ""),
                                              "sim_step": 0, "scene_revision": "stub"}))
                elif t == "heartbeat":
                    await ws.send(json.dumps({"type": "heartbeat_ack",
                                              "echo_ns": msg.get("sent_ns", 0)}))
        except Exception:
            pass
        finally:
            self._conns.discard(ws)
