"""
R12-400: Native macOS arm64 MuJoCo WebSocket server for Reachy 1.2.

Lifecycle
---------
1. Load reachy_1_2.xml (or model path from --model).
2. Accept one WebSocket connection at a time on --host:--port (default 127.0.0.1:8765).
3. Run a tight asyncio + MuJoCo step loop (background thread → main loop via queue).
4. Push state and camera frames to connected clients.
5. Apply joint_command, scene_load, reset, pause from clients.
6. Enforce heartbeat deadlines; reconnect transparently.

Launch via mjpython (required for MuJoCo viewer on macOS):
    cd native_mujoco
    mjpython server.py [--model model/reachy_1_2.xml] [--host 127.0.0.1] [--port 8765]
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import logging
import os
import pathlib
import threading
import time
from typing import Any, Dict, Optional

import mujoco
import numpy as np
import websockets
import websockets.exceptions

from joint_map import JOINT_TABLE, NUM_JOINTS
from protocol import (
    PROTOCOL_VERSION,
    CameraFrame,
    Error,
    Hello,
    HelloAck,
    HeartbeatAck,
    JointCommand,
    Pause,
    Reset,
    ResetAck,
    SceneAck,
    SceneLoad,
    Shutdown,
    State,
    decode,
    encode,
    jpeg_to_b64,
    message_type,
    validate_joint_command,
)
from renderer import StereoRenderer, jpeg_to_b64

log = logging.getLogger("reachy12.mujoco.server")

_MODEL_DIR = pathlib.Path(__file__).parent / "model"
_DEFAULT_MODEL = _MODEL_DIR / "reachy_1_2.xml"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8765
_SIM_STEP_HZ  = 500        # physics steps per second
_CAMERA_HZ    = 15         # camera render target
_STATE_HZ     = 50         # state message rate to client
_HB_INTERVAL  = 2.0        # heartbeat period (s)
_HB_DEADLINE  = 6.0        # max time without heartbeat before disconnect (s)
_CAM_WIDTH    = int(os.environ.get("REACHY_SIM_CAMERA_WIDTH", "640"))
_CAM_HEIGHT   = int(os.environ.get("REACHY_SIM_CAMERA_HEIGHT", "480"))


class SimState:
    """Mutable simulation state — owned by the sim thread."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.step = 0
        self.paused = False
        self.scene_revision = "initial"
        self._lock = threading.Lock()
        self._cmd_seq = 0
        self._pending_cmd: Optional[Dict[str, Any]] = None
        self._pending_reset: Optional[Dict[str, Any]] = None
        self._pending_pause: Optional[bool] = None

    # --- Thread-safe command submission (from asyncio handlers) ---

    def submit_command(self, msg: Dict[str, Any]) -> None:
        with self._lock:
            self._pending_cmd = msg

    def submit_reset(self, msg: Dict[str, Any]) -> None:
        with self._lock:
            self._pending_reset = msg

    def submit_pause(self, paused: bool) -> None:
        with self._lock:
            self._pending_pause = paused

    # --- Called by sim thread each step ---

    def apply_pending(self) -> Optional[str]:
        """Apply queued commands; return reset request_id if reset occurred."""
        with self._lock:
            pause = self._pending_pause
            cmd = self._pending_cmd
            reset_req = self._pending_reset
            self._pending_pause = None
            self._pending_cmd = None
            self._pending_reset = None

        if pause is not None:
            self.paused = pause

        reset_id = None
        if reset_req is not None:
            mujoco.mj_resetData(self.model, self.data)
            self.step = 0
            reset_id = reset_req.get("request_id", "")

        if cmd is not None:
            tgt = cmd.get("target_rad", [])
            mask = cmd.get("mask")
            for entry in JOINT_TABLE:
                idx = entry.mjcf_index
                if mask is None or (mask and mask[idx]):
                    lo, hi = entry.limits_rad
                    self.data.ctrl[idx] = float(np.clip(tgt[idx], lo, hi))
            self._cmd_seq = cmd.get("seq", self._cmd_seq)

        return reset_id

    def snapshot_joints(self) -> list:
        joints = []
        for entry in JOINT_TABLE:
            i = entry.mjcf_index
            joints.append({
                "name": entry.sdk_name,
                "uid": entry.uid,
                "position_rad": float(self.data.qpos[i]),
                "velocity_rad_s": float(self.data.qvel[i]),
                "effort": float(self.data.qfrc_actuator[i]) if i < len(self.data.qfrc_actuator) else 0.0,
            })
        return joints

    def copy_data(self) -> mujoco.MjData:
        """Deep copy of MjData for use in the render thread."""
        d = mujoco.MjData(self.model)
        mujoco.mj_copyData(d, self.model, self.data)
        return d


class ReachyMujocoServer:

    def __init__(self, model_path: str, host: str, port: int) -> None:
        log.info("Loading model: %s", model_path)
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._sim = SimState(self._model)
        self._host = host
        self._port = port

        self._seq = 0
        self._cam_seq = {"left_camera": 0, "right_camera": 0}
        self._shutdown = threading.Event()
        self._connected_ws: Optional[Any] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # queues for thread→asyncio communication
        self._state_q: asyncio.Queue = None   # type: ignore[assignment]
        self._frame_q: asyncio.Queue = None   # type: ignore[assignment]
        self._reset_ack_q: asyncio.Queue = None  # type: ignore[assignment]

        self._renderer: Optional[StereoRenderer] = None

    # ── Sim thread (physics + periodic render) ───────────────────────────────

    def _sim_thread(self) -> None:
        log.info("Sim thread started (%.0f Hz)", _SIM_STEP_HZ)
        dt = self._model.opt.timestep
        cam_every = max(1, int(_SIM_STEP_HZ / _CAMERA_HZ))
        state_every = max(1, int(_SIM_STEP_HZ / _STATE_HZ))

        renderer = StereoRenderer(self._model, width=_CAM_WIDTH, height=_CAM_HEIGHT)
        self._renderer = renderer

        step_period = 1.0 / _SIM_STEP_HZ
        next_step = time.monotonic()

        while not self._shutdown.is_set():
            now = time.monotonic()
            if now < next_step:
                time.sleep(max(0.0, next_step - now - 0.0001))
                continue
            next_step += step_period

            reset_id = self._sim.apply_pending()

            if not self._sim.paused:
                mujoco.mj_step(self._model, self._sim.data)
                self._sim.step += 1

            # State push
            if self._sim.step % state_every == 0 and self._loop:
                state = self._build_state()
                asyncio.run_coroutine_threadsafe(
                    self._state_q.put(state), self._loop
                )

            # Camera render
            if self._sim.step % cam_every == 0 and self._loop:
                data_copy = self._sim.copy_data()
                frames = renderer.render_stereo(data_copy)
                for cam_name, fr in frames.items():
                    self._cam_seq[cam_name] += 1
                    cam_msg = CameraFrame(
                        camera=cam_name,
                        seq=self._cam_seq[cam_name],
                        sim_step=self._sim.step,
                        sim_time_s=float(self._sim.data.time),
                        scene_revision=self._sim.scene_revision,
                        width=fr.width,
                        height=fr.height,
                        jpeg_b64=jpeg_to_b64(fr.jpeg_bytes),
                        render_us=fr.render_us,
                    )
                    asyncio.run_coroutine_threadsafe(
                        self._frame_q.put(cam_msg), self._loop
                    )

            # Reset ack
            if reset_id is not None and self._loop:
                ack = ResetAck(
                    request_id=reset_id,
                    sim_step=self._sim.step,
                    scene_revision=self._sim.scene_revision,
                )
                asyncio.run_coroutine_threadsafe(
                    self._reset_ack_q.put(ack), self._loop
                )

        renderer.close()
        log.info("Sim thread stopped")

    def _build_state(self) -> State:
        self._seq += 1
        return State(
            seq=self._seq,
            sim_step=self._sim.step,
            sim_time_s=float(self._sim.data.time),
            scene_revision=self._sim.scene_revision,
            paused=self._sim.paused,
            joints=self._sim.snapshot_joints(),
        )

    # ── WebSocket handler ────────────────────────────────────────────────────

    async def _handle_connection(self, ws) -> None:
        addr = ws.remote_address
        log.info("Client connected from %s", addr)

        # Handshake
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            msg = decode(raw)
            if msg.get("type") != "hello":
                await ws.send(Error(code="handshake_error",
                                    message="Expected hello").encode())
                return
            client_ver = msg.get("protocol_version", 0)
            if client_ver != PROTOCOL_VERSION:
                await ws.send(Error(
                    code="version_mismatch",
                    message=f"Server requires protocol {PROTOCOL_VERSION}, "
                            f"client sent {client_ver}",
                ).encode())
                return
        except asyncio.TimeoutError:
            log.warning("Handshake timeout from %s", addr)
            return

        ack = HelloAck(
            sim_fps=_SIM_STEP_HZ,
            camera_fps=_CAMERA_HZ,
            num_joints=NUM_JOINTS,
        )
        await ws.send(ack.encode())
        self._connected_ws = ws
        log.info("Handshake complete with %s", addr)

        last_hb_recv = time.monotonic()

        async def _send_loop() -> None:
            while True:
                # Drain state
                try:
                    state = self._state_q.get_nowait()
                    await ws.send(state.encode())
                except asyncio.QueueEmpty:
                    pass
                # Drain one camera frame
                try:
                    frame = self._frame_q.get_nowait()
                    await ws.send(frame.encode())
                except asyncio.QueueEmpty:
                    pass
                # Drain reset ack
                try:
                    rack = self._reset_ack_q.get_nowait()
                    await ws.send(rack.encode())
                except asyncio.QueueEmpty:
                    pass
                await asyncio.sleep(0.002)

        async def _recv_loop() -> None:
            nonlocal last_hb_recv
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=_HB_DEADLINE)
                except asyncio.TimeoutError:
                    log.warning("Heartbeat deadline exceeded from %s", addr)
                    return
                mtype = message_type(raw)
                decoded = decode(raw)

                if mtype == "joint_command":
                    try:
                        validate_joint_command(decoded, NUM_JOINTS)
                    except ValueError as exc:
                        await ws.send(Error(code="bad_command",
                                            message=str(exc)).encode())
                        continue
                    self._sim.submit_command(decoded)

                elif mtype == "reset":
                    self._sim.submit_reset(decoded)

                elif mtype == "pause":
                    self._sim.submit_pause(bool(decoded.get("paused", True)))

                elif mtype == "heartbeat":
                    last_hb_recv = time.monotonic()
                    await ws.send(HeartbeatAck(
                        echo_ns=decoded.get("sent_ns", 0)
                    ).encode())

                elif mtype == "heartbeat_ack":
                    last_hb_recv = time.monotonic()

                elif mtype == "scene_load":
                    scene_doc = decoded.get("scene_document", {})
                    req_id = decoded.get("request_id", "")
                    accepted, rev, warnings, error = self._load_scene(scene_doc)
                    await ws.send(SceneAck(
                        request_id=req_id,
                        accepted=accepted,
                        scene_revision=rev,
                        warnings=warnings,
                        error=error,
                    ).encode())

                elif mtype == "disconnect":
                    log.info("Client %s requested disconnect", addr)
                    return

        async def _heartbeat_loop() -> None:
            import time as _time
            from protocol import Heartbeat as _HB
            while True:
                await asyncio.sleep(_HB_INTERVAL)
                try:
                    await ws.send(_HB().encode())
                except Exception:
                    return

        try:
            await asyncio.gather(
                _send_loop(),
                _recv_loop(),
                _heartbeat_loop(),
                return_exceptions=False,
            )
        except websockets.exceptions.ConnectionClosed:
            log.info("Client %s disconnected", addr)
        except Exception as exc:
            log.exception("Error in handler for %s: %s", addr, exc)
        finally:
            self._connected_ws = None
            log.info("Handler exited for %s", addr)

    def _load_scene(
        self, scene_doc: dict
    ) -> tuple[bool, str, list, str]:
        """Stub: scene loading wired to scene_compiler (R12-402)."""
        try:
            import hashlib, json as _json
            rev = hashlib.sha256(
                _json.dumps(scene_doc, sort_keys=True).encode()
            ).hexdigest()[:12]
            self._sim.scene_revision = rev
            return True, rev, [], ""
        except Exception as exc:
            return False, "", [], str(exc)

    # ── Entry point ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        self._state_q = asyncio.Queue(maxsize=10)
        self._frame_q = asyncio.Queue(maxsize=4)
        self._reset_ack_q = asyncio.Queue(maxsize=4)
        self._loop = asyncio.get_running_loop()

        sim_thread = threading.Thread(
            target=self._sim_thread, daemon=True, name="sim-loop"
        )
        sim_thread.start()

        log.info("WebSocket server listening on ws://%s:%d", self._host, self._port)
        async with websockets.serve(self._handle_connection, self._host, self._port):
            try:
                await asyncio.Future()   # run forever
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass

        log.info("Stopping simulation…")
        self._shutdown.set()
        sim_thread.join(timeout=3.0)
        log.info("Server stopped")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reachy 1.2 native MuJoCo server")
    ap.add_argument("--model", default=str(_DEFAULT_MODEL))
    ap.add_argument("--host", default=_DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=_DEFAULT_PORT)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    server = ReachyMujocoServer(args.model, args.host, args.port)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
