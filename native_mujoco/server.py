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
from typing import Any, Dict, Mapping, Optional, Sequence

import mujoco
import numpy as np
import websockets
import websockets.exceptions

from actuator import ActuatorController
from gripper import GripperModel
from joint_map import JOINT_TABLE, NUM_JOINTS
from objects import ObjectTracker
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
    message_type,
    validate_joint_command,
)
from calibration import (
    StereoCalibrationProfile,
    apply_to_model,
    load_calibration,
    synthetic_defaults,
)
from recorder import Recorder
from renderer import StereoRenderer, jpeg_to_b64
from sensor_effects import EffectConfig, SensorEffectPipeline

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

    def __init__(
        self,
        model: mujoco.MjModel,
        tracked_ids: Optional[Sequence[str]] = None,
        interactive_specs: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
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

        # R12-501: actuator/compliance model owns ctrl, gains and force limits.
        self._reset_physics()
        self.controller = ActuatorController(model)
        self.controller.sync_targets_to_current(self.data)
        # R12-502: gripper/contact model reads contacts for grasp & force state.
        self.gripper = GripperModel(model)
        # R12-503: dynamic object tracking (free-joint scene objects).
        self.objects = ObjectTracker(model, self.data, tracked_ids=tracked_ids)
        self.objects.capture_initial(self.data)
        # R12-504: interactive controls (buttons/switches/levers).
        from interactive import InteractiveController
        self.interactive = InteractiveController(
            model, self.data, interactive_specs or []
        )

    def _reset_physics(self) -> None:
        """Reset to the home keyframe, keeping free-joint objects at their
        MJCF scene poses (the 21-DOF keyframe would otherwise zero them)."""
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        for jid in range(self.model.njnt):
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                adr = self.model.jnt_qposadr[jid]
                self.data.qpos[adr:adr + 7] = self.model.qpos0[adr:adr + 7]
        mujoco.mj_forward(self.model, self.data)  # propagate to xpos/contacts

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
            self._reset_physics()
            self.step = 0
            self.controller.sync_targets_to_current(self.data)
            # R12-503: seeded, deterministic object placement.
            seed = reset_req.get("seed")
            self.objects.reset(
                self.data,
                seed=seed,
                jitter_m=float(reset_req.get("jitter_m", 0.0)),
            )
            self.interactive.reset(self.data)
            mujoco.mj_forward(self.model, self.data)
            reset_id = reset_req.get("request_id", "")

        if cmd is not None:
            tgt = cmd.get("target_rad", [])
            mask = cmd.get("mask")
            compliant = cmd.get("compliant")            # optional list[bool|None]
            speed = cmd.get("speed_limit_rad_s")        # optional list[float|None]
            torque = cmd.get("torque_limit_percent")    # optional list[float|None]
            for entry in JOINT_TABLE:
                idx = entry.mjcf_index
                if mask is not None and not (mask and mask[idx]):
                    continue
                if idx < len(tgt):
                    self.controller.set_goal_position(idx, tgt[idx])
                if compliant is not None and compliant[idx] is not None:
                    self.controller.set_compliant(idx, compliant[idx])
                if speed is not None and speed[idx] is not None:
                    self.controller.set_speed_limit(idx, speed[idx])
                if torque is not None and torque[idx] is not None:
                    self.controller.set_torque_limit(idx, torque[idx])
            self._cmd_seq = cmd.get("seq", self._cmd_seq)

        return reset_id

    def control_step(self, dt: float) -> None:
        """Apply the actuator/compliance model for the upcoming mj_step."""
        self.controller.apply(self.data, dt)
        # R12-504: toggle logic + bistable snap torque, applied before mj_step.
        self.interactive.update(self.data)

    def snapshot_interactive(self) -> list:
        """Interactive control on/off states for the state message."""
        return self.interactive.states(self.data)

    def snapshot_joints(self) -> list:
        joints = []
        for entry in JOINT_TABLE:
            i = entry.mjcf_index
            st = self.controller.state[i]
            joints.append({
                "name": entry.sdk_name,
                "uid": entry.uid,
                "position_rad": float(self.data.qpos[i]),
                "velocity_rad_s": float(self.data.qvel[i]),
                "effort": float(self.data.actuator_force[i]),
                "compliant": bool(st.compliant),
                "saturated": self.controller.is_saturated(self.data, i),
            })
        return joints

    def snapshot_grippers(self):
        """Return (grippers, force_sensors) lists for the state message."""
        states = self.gripper.update(self.data)
        grippers = []
        force_sensors = []
        for side, st in states.items():
            grippers.append({
                "side": side,
                "grasping": st.grasping,
                "grip_force_n": st.grip_force_n,
                "grasped_geoms": st.grasped_geoms,
            })
            force_sensors.append({
                "uid": st.sensor_uid,
                "force": st.grip_force_n,
            })
        return grippers, force_sensors

    def copy_data(self) -> mujoco.MjData:
        """Deep copy of MjData for use in the render thread."""
        d = mujoco.MjData(self.model)
        mujoco.mj_copyData(d, self.model, self.data)
        return d


class ReachyMujocoServer:

    def __init__(
        self,
        model_path: str,
        host: str,
        port: int,
        scene_path: Optional[str] = None,
        calibration: Optional[StereoCalibrationProfile] = None,
        enable_depth: bool = False,
        enable_seg: bool = False,
        effects: Optional[EffectConfig] = None,
        record_dir: Optional[str] = None,
    ) -> None:
        self._calibration = calibration
        self._enable_depth = enable_depth
        self._enable_seg = enable_seg
        self._effects = effects or EffectConfig()
        self._record_dir = record_dir

        tracked_ids = None
        interactive_specs = None
        if scene_path:
            log.info("Loading scene: %s (into model %s)", scene_path, model_path)
            import yaml
            from objects import build_scene_model_xml
            from scene_compiler import tracked_object_ids
            from scene_compiler import interactive_specs as _interactive_specs
            # Validate for safety (raises on unsafe paths / bad schema); the
            # compiler consumes the raw dict since it needs full physics fields.
            try:
                from scene_loader import load_scene
                load_scene(scene_path)
            except ImportError:
                log.warning("scene_loader unavailable; skipping validation")
            with open(scene_path) as f:
                scene_doc = yaml.safe_load(f)
            xml = build_scene_model_xml(scene_doc, model_path)
            self._model = mujoco.MjModel.from_xml_string(xml)
            tracked_ids = tracked_object_ids(scene_doc)
            interactive_specs = _interactive_specs(scene_doc)
            log.info("Scene loaded: %d tracked objects, %d interactive controls",
                     len(tracked_ids), len(interactive_specs))
        else:
            log.info("Loading model: %s", model_path)
            self._model = mujoco.MjModel.from_xml_path(model_path)

        # Apply calibration intrinsics (fov_y) to model cameras.
        if self._calibration is None:
            self._calibration = synthetic_defaults(_CAM_WIDTH, _CAM_HEIGHT)
            log.info("Calibration: using synthetic defaults (fov_y=%.1f°)",
                     self._calibration.left_camera.fov_y_deg)
        else:
            log.info("Calibration: %s (fov_y=%.1f°)",
                     self._calibration.provenance,
                     self._calibration.left_camera.fov_y_deg)
        apply_to_model(self._calibration, self._model)

        self._sim = SimState(
            self._model,
            tracked_ids=tracked_ids,
            interactive_specs=interactive_specs,
        )
        self._host = host
        self._port = port

        self._seq = 0
        self._cam_seq = {"left_camera": 0, "right_camera": 0}
        self._shutdown = threading.Event()
        self._connected_ws: Optional[Any] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._recorder: Optional[Recorder] = None   # set in _sim_thread

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

        renderer = StereoRenderer(
            self._model,
            width=_CAM_WIDTH, height=_CAM_HEIGHT,
            enable_depth=self._enable_depth,
            enable_seg=self._enable_seg,
        )
        self._renderer = renderer

        # Per-camera sensor effect pipelines (one each, not shared across cameras)
        effect_pipelines = {
            "left_camera":  SensorEffectPipeline(self._effects),
            "right_camera": SensorEffectPipeline(self._effects),
        }

        # Recorder (R12-603)
        sim_start = time.monotonic()
        if self._record_dir:
            self._recorder = Recorder.new(
                self._record_dir,
                {
                    "model_path": str(_DEFAULT_MODEL),
                    "scene_path": None,
                    "calibration_provenance": (
                        self._calibration.provenance if self._calibration else "none"
                    ),
                    "depth_enabled": self._enable_depth,
                    "seg_enabled": self._enable_seg,
                    "effects": {
                        "blur_sigma": self._effects.blur_sigma,
                        "noise_std": self._effects.noise_std,
                        "drop_probability": self._effects.drop_probability,
                        "latency_ms": self._effects.latency_ms,
                    },
                },
            )
            log.info("Recording to: %s", self._recorder.run_dir)

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
                self._sim.control_step(dt)   # R12-501 actuator/compliance model
                mujoco.mj_step(self._model, self._sim.data)
                self._sim.step += 1

            # State push
            if self._sim.step % state_every == 0 and self._loop:
                state = self._build_state()
                asyncio.run_coroutine_threadsafe(
                    self._state_q.put(state), self._loop
                )
                if self._recorder is not None:
                    import json as _json
                    self._recorder.record_state(_json.loads(state.encode()))

            # Camera render
            if self._sim.step % cam_every == 0 and self._loop:
                data_copy = self._sim.copy_data()
                frames = renderer.render_stereo(data_copy)
                for cam_name, fr in frames.items():
                    # Apply sensor effects (R12-602)
                    pipe = effect_pipelines[cam_name]
                    jpeg = pipe.apply_pixels(fr.jpeg_bytes)
                    if jpeg is None:
                        continue   # frame dropped by effect pipeline
                    if self._effects.latency_ms > 0.0:
                        pipe.push_latency(jpeg)
                        jpeg = pipe.pop_ready()
                        if jpeg is None:
                            continue   # frame still in latency buffer

                    self._cam_seq[cam_name] += 1
                    cam_msg = CameraFrame(
                        camera=cam_name,
                        seq=self._cam_seq[cam_name],
                        sim_step=self._sim.step,
                        sim_time_s=float(self._sim.data.time),
                        scene_revision=self._sim.scene_revision,
                        width=fr.width,
                        height=fr.height,
                        jpeg_b64=jpeg_to_b64(jpeg),
                        render_us=fr.render_us,
                        depth_b64=fr.depth_b64,
                        seg_b64=fr.seg_b64,
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

        if self._recorder is not None:
            wall = time.monotonic() - sim_start
            self._recorder.finalize(total_steps=self._sim.step, duration_s=wall)
            log.info("Recording finalized: %s", self._recorder.run_dir)

        renderer.close()
        log.info("Sim thread stopped")

    def _build_state(self) -> State:
        self._seq += 1
        grippers, force_sensors = self._sim.snapshot_grippers()
        return State(
            seq=self._seq,
            sim_step=self._sim.step,
            sim_time_s=float(self._sim.data.time),
            scene_revision=self._sim.scene_revision,
            paused=self._sim.paused,
            joints=self._sim.snapshot_joints(),
            objects=self._sim.objects.poses_as_dicts(self._sim.data),
            grippers=grippers,
            force_sensors=force_sensors,
            interactive=self._sim.snapshot_interactive(),
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
                    if self._recorder is not None:
                        self._recorder.record_command(decoded)

                elif mtype == "reset":
                    self._sim.submit_reset(decoded)
                    if self._recorder is not None:
                        self._recorder.record_reset(
                            decoded.get("seed"), self._sim.step
                        )

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
        """Runtime scene replacement is not yet implemented atomically.

        Accepting a scene_load here would set a new revision without actually
        swapping the MuJoCo model, creating false provenance (clients believe a
        new scene is active while the old physical world remains loaded).
        Reject until atomic model/scene swap is implemented and tested.
        """
        return False, "", [], "restart_required: runtime scene_load is not supported; restart the server with the desired scene"

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
    ap.add_argument("--scene", default=None,
                    help="scene YAML to compile into the model (R12-503)")
    ap.add_argument("--host", default=_DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=_DEFAULT_PORT)
    ap.add_argument("--log-level", default="INFO")
    # R12-600: calibration
    ap.add_argument("--calibration", default=None,
                    help="camera calibration YAML file (R12-600); "
                         "defaults to synthetic_defaults")
    # R12-601: depth and segmentation
    ap.add_argument("--depth", action="store_true",
                    help="include depth map in camera_frame messages (R12-601)")
    ap.add_argument("--segmentation", action="store_true",
                    help="include body-ID segmentation in camera_frame messages (R12-601)")
    # R12-602: sensor effects
    ap.add_argument("--effects", default=None,
                    help="sensor effect config YAML file (R12-602)")
    # R12-603: recording
    ap.add_argument("--record", default=None, metavar="DIR",
                    help="record states+commands to timestamped run dir under DIR (R12-603)")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Load calibration profile
    calibration: Optional[StereoCalibrationProfile] = None
    if args.calibration:
        calibration = load_calibration(args.calibration)
        log.info("Loaded calibration from %s (%s)", args.calibration,
                 calibration.provenance)

    # Load sensor effects config
    effects: Optional[EffectConfig] = None
    if args.effects:
        effects = EffectConfig.from_yaml(args.effects)
        log.info("Loaded sensor effects from %s", args.effects)

    server = ReachyMujocoServer(
        args.model, args.host, args.port,
        scene_path=args.scene,
        calibration=calibration,
        enable_depth=args.depth,
        enable_seg=args.segmentation,
        effects=effects,
        record_dir=args.record,
    )
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
