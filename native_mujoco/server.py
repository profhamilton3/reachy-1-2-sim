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
import sys
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
    PlaceAck,
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
    ControlAck,
)
from calibration import (
    StereoCalibrationProfile,
    apply_to_model,
    load_calibration,
    synthetic_defaults,
)
from recorder import Recorder
from renderer import StereoRenderer, jpeg_to_b64
from distortion import LensDistorter, auto_margin
from zoom import ZoomLevel, fov_y_for_level, parse_level
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
#: Longest an execution lease may be held before it expires on its own.  A
#: pick-and-place arc is tens of seconds; anything near this is a wedge.
_MAX_CONTROL_TTL_S = 300.0
_CAM_WIDTH    = int(os.environ.get("REACHY_SIM_CAMERA_WIDTH", "640"))
_CAM_HEIGHT   = int(os.environ.get("REACHY_SIM_CAMERA_HEIGHT", "480"))
# Distortion source-render margin.  Unset = auto (smallest factor with no dark
# corners).  Set to 1.0 to render at the output size and accept dark corners.
_CAM_MARGIN = (
    float(os.environ["REACHY_SIM_DISTORTION_MARGIN"])
    if os.environ.get("REACHY_SIM_DISTORTION_MARGIN") else None
)


_MAX_PENDING_PLACES = 256


class SimState:
    """Mutable simulation state — owned by the sim thread."""

    def __init__(
        self,
        model: mujoco.MjModel,
        tracked_ids: Optional[Sequence[str]] = None,
        interactive_specs: Optional[Sequence[Mapping[str, Any]]] = None,
        scene_doc: Optional[Mapping[str, Any]] = None,
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
        self._pending_places: list[Dict[str, Any]] = []
        self._place_results: list[Dict[str, Any]] = []

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
        # R12-607: runtime placement onto grid cells.  Built LAST, so the home
        # pose it records for each object is the settled scene pose rather than
        # whatever qpos held before _reset_physics ran.
        self.placer = None
        if scene_doc:
            from placement import ObjectPlacer
            self.placer = ObjectPlacer(model, self.data, scene_doc)

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

    def submit_place(self, msg: Dict[str, Any]) -> None:
        # A list, not a slot: placements are discrete events a client may fire
        # several of in a row (set up a board, then look at it), and dropping
        # all but the last would silently build the wrong scene.  Joint commands
        # can coalesce because only the newest target matters; these cannot.
        #
        # Capped because this is network-facing: a client that submits faster
        # than the sim drains would otherwise grow the list without bound.  The
        # cap is far above any real board (ten objects) so it only ever trips on
        # a runaway, and it drops the newest rather than silently discarding the
        # placements already queued ahead of it.
        with self._lock:
            if len(self._pending_places) >= _MAX_PENDING_PLACES:
                self._place_results.append({
                    "_conn_id": msg.get("_conn_id"),
                    "request_id": str(msg.get("request_id", "")),
                    "accepted": False,
                    "error": f"placement queue full ({_MAX_PENDING_PLACES} "
                             f"pending); the sim has not drained yet"})
                return
            self._pending_places.append(msg)

    # --- Called by sim thread each step ---

    def apply_pending(self) -> Optional[Dict[str, Any]]:
        """Apply queued commands; return {"request_id", "_conn_id"} of the
        reset that occurred, so the ack can be routed to whoever asked for
        it rather than broadcast to whichever connection polls first."""
        with self._lock:
            pause = self._pending_pause
            cmd = self._pending_cmd
            reset_req = self._pending_reset
            self._pending_pause = None
            self._pending_cmd = None
            self._pending_reset = None

        if pause is not None:
            self.paused = pause

        reset_info = None
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
            reset_info = {
                "request_id": reset_req.get("request_id", ""),
                "_conn_id": reset_req.get("_conn_id"),
            }

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

        self._apply_places()
        return reset_info

    _RESHAPE_KEYS = frozenset({"size", "radius", "length", "rgba", "mass"})

    def _apply_places(self) -> None:
        """Run queued placements.  Sim thread only — writes qpos and model."""
        with self._lock:
            pending, self._pending_places = self._pending_places, []
        if not pending:
            return
        from placement import PlacementError
        for msg in pending:
            rid = str(msg.get("request_id", ""))
            conn = {"_conn_id": msg.get("_conn_id")}
            if self.placer is None:
                self._place_results.append({
                    **conn, "request_id": rid, "accepted": False,
                    "error": "placement unavailable: server was started without "
                             "a scene, so there are no grid cells and no objects"})
                continue
            try:
                oid = str(msg.get("object_id", ""))
                shape = msg.get("reshape")
                if shape:
                    unknown = sorted(set(shape) - self._RESHAPE_KEYS)
                    if unknown:
                        raise PlacementError(
                            f"reshape does not accept {unknown}; "
                            f"valid keys are {sorted(self._RESHAPE_KEYS)}")
                    self.placer.reshape(oid, **shape)
                cell = msg.get("cell")
                if cell is None:
                    placed = self.placer.stow(oid)
                else:
                    placed = self.placer.place(
                        oid, str(cell),
                        yaw_deg=float(msg.get("yaw_deg", 0.0)),
                        allow_unreachable=bool(msg.get("allow_unreachable", False)),
                        allow_occupied=bool(msg.get("allow_occupied", False)),
                    )
            except (PlacementError, ValueError, TypeError) as exc:
                self._place_results.append({
                    **conn, "request_id": rid, "accepted": False, "error": str(exc)})
            else:
                self._place_results.append({
                    **conn, "request_id": rid, "accepted": True,
                    "placement": placed.as_dict()})

    def drain_place_results(self) -> list[Dict[str, Any]]:
        with self._lock:
            out, self._place_results = self._place_results, []
        return out

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
        enable_distortion: bool = False,
    ) -> None:
        self._calibration = calibration
        self._enable_distortion = enable_distortion
        self._zoom_level = ZoomLevel.INTER   # the calibrated level
        self._pending_zoom: Optional[ZoomLevel] = None
        self._enable_depth = enable_depth
        self._enable_seg = enable_seg
        self._effects = effects or EffectConfig()
        self._record_dir = record_dir
        self._model_path = model_path        # actual path used — not _DEFAULT_MODEL
        self._scene_path = scene_path        # actual scene path (or None)

        tracked_ids = None
        interactive_specs = None
        scene_doc = None
        if scene_path:
            log.info("Loading scene: %s (into model %s)", scene_path, model_path)
            from objects import build_scene_model_xml
            from scene_compiler import tracked_object_ids
            from scene_compiler import interactive_specs as _interactive_specs
            # Resolve `extends:` FIRST, then validate the RESULT.  A child
            # scene is not a complete document on its own — it inherits `world`,
            # the cameras and several hundred lines of measured geometry — so
            # validating the raw file rejects every scene that uses inheritance,
            # and validating only the child would leave the parent's objects
            # unchecked.  This is not hypothetical: between #37 and #38 the
            # format gained `extends:` while the schema and this call site did
            # not, and the server could not load FWDCenterLabSiva at all.
            from scene_io import load_scene as _resolve_scene
            scene_doc = _resolve_scene(scene_path)
            # Safety validation (raises on unsafe mesh paths / bad schema); the
            # compiler consumes the raw dict since it needs full physics fields.
            # scene_loader lives at the repo root, and BOTH launch scripts cd
            # into native_mujoco/ before starting this server — so this import
            # has always failed there and the safety validation has always been
            # skipped, announced only as a WARNING nobody was reading.  Put the
            # root on the path rather than keep a check that never runs.
            _repo_root = str(pathlib.Path(__file__).resolve().parents[1])
            if _repo_root not in sys.path:
                sys.path.append(_repo_root)
            try:
                from scene_loader import load_scene as _validate_scene
                _validate_scene(scene_path, document=scene_doc)
            except ImportError:
                log.warning("scene_loader unavailable; skipping validation")
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

        # Opt-in post-render barrel distortion (see native_mujoco/distortion.py).
        # Built from the SAME profile that set fov_y, so the warp and the field
        # of view can never disagree.
        self._distorters: Dict[str, LensDistorter] = {}
        if self._enable_distortion:
            cams = (
                ("left_camera", self._calibration.left_camera),
                ("right_camera", self._calibration.right_camera),
            )
            # One shared margin: a single mujoco.Renderer serves both cameras,
            # so they must render at the same source size.  Take the wider of
            # the two requirements — a margin larger than a camera needs only
            # costs pixels, while one too small reintroduces its dark corners.
            margin = 1.0
            if _CAM_MARGIN is not None:
                margin = _CAM_MARGIN
            else:
                try:
                    margin = max(auto_margin(i, _CAM_WIDTH, _CAM_HEIGHT) for _, i in cams)
                except ValueError as exc:
                    log.warning("auto margin failed (%s); falling back to 1.0 "
                                "— frames will have dark corners", exc)
            for cam_name, intr in cams:
                d = LensDistorter(intr, _CAM_WIDTH, _CAM_HEIGHT, margin=margin)
                if d.is_identity:
                    log.warning(
                        "Distortion requested but calibration profile '%s' has zero "
                        "radial coefficients for %s — frames will be unchanged",
                        self._calibration.provenance, cam_name)
                self._distorters[cam_name] = d
            sample = self._distorters["left_camera"]
            log.info(
                "Lens distortion ENABLED (profile: %s, margin %.2f, "
                "source render %dx%d @ %.1f° -> output %dx%d, corners %s)",
                self._calibration.provenance, margin,
                *sample.source_size, sample.source_fov_y_deg,
                _CAM_WIDTH, _CAM_HEIGHT,
                "filled" if sample.fully_covered else "DARK")

        self._sim = SimState(
            self._model,
            tracked_ids=tracked_ids,
            interactive_specs=interactive_specs,
            scene_doc=scene_doc,
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
        # State and camera frames are BROADCASTS: every connected client is
        # entitled to all of them.  They were single shared queues drained by
        # whichever connection's send loop polled first, which silently SPLIT
        # the stream — measured, one client got 13.2 camera_frame/s and two got
        # 7.8 each.  So opening a second client (a browser panel, a notebook)
        # halved the Docker bridge's frame rate and slowed RViz, with nothing
        # anywhere saying why.  One queue per connection, fanned out below.
        self._state_qs: dict = {}
        self._frame_qs: dict = {}
        # One queue PER CONNECTION, keyed by connection id, for both reset and
        # placement acks.  A single shared queue is wrong here: the server
        # accepts concurrent clients, each with its own send loop, so whichever
        # loop called get_nowait() first took the ack — a notebook's placement
        # ack would be delivered to the Docker bridge, which discards it, and
        # the notebook would wait forever for a placement that had in fact
        # already happened.  Observed exactly that; reset_ack had the identical
        # bug (#84) since it was built the same way and never revisited when
        # place_ack was fixed.
        self._reset_ack_qs: dict = {}
        self._place_ack_qs: dict = {}
        self._next_conn_id = 0
        # Execution lease (issue #51).  None, or a dict with conn_id /
        # client_id / motion_client_id / expires_at.  Only ever touched from
        # the asyncio loop, so it needs no lock; keeping it off the sim thread
        # is also what lets a refusal be answered on the socket that caused it.
        self._control: Optional[dict] = None

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
            distorters=self._distorters,
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
                self._build_recorder_manifest(),
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

            reset_info = self._sim.apply_pending()

            if not self._sim.paused:
                self._sim.control_step(dt)   # R12-501 actuator/compliance model
                mujoco.mj_step(self._model, self._sim.data)
                self._sim.step += 1

            # State push
            if self._sim.step % state_every == 0 and self._loop:
                state = self._build_state()
                asyncio.run_coroutine_threadsafe(
                    self._broadcast(self._state_qs, state), self._loop
                )
                if self._recorder is not None:
                    import json as _json
                    self._recorder.record_state(_json.loads(state.encode()))

            # Camera render
            if self._sim.step % cam_every == 0 and self._loop:
                data_copy = self._sim.copy_data()
                frames = renderer.render_stereo(data_copy)
                for cam_name, fr in frames.items():
                    # R12-605: apply a pending zoom on this thread, where the
                    # renderer's cam_fovy and distortion maps are owned.
                    if self._pending_zoom is not None:
                        lvl, self._pending_zoom = self._pending_zoom, None
                        fov = fov_y_for_level(
                            lvl, _CAM_WIDTH, _CAM_HEIGHT,
                            self._calibration.left_camera.fov_y_deg)
                        try:
                            renderer.set_zoom(fov)
                        except ValueError as exc:
                            log.warning("Zoom to %s refused: %s", lvl.value, exc)
                        else:
                            self._zoom_level = lvl
                            log.info("Zoom now %s (fov_y %.1f°)%s", lvl.value, fov,
                                     "" if lvl in (ZoomLevel.INTER, ZoomLevel.ZERO)
                                     else " — barrel profile approximate off the "
                                          "calibrated level")

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
                        self._broadcast(self._frame_qs, cam_msg), self._loop
                    )

            # Reset ack — routed to the connection that asked, not broadcast.
            if reset_info is not None and self._loop:
                queue = self._reset_ack_qs.get(reset_info.get("_conn_id"))
                if queue is None:
                    # The client disconnected between asking and landing.  The
                    # reset still happened — it is a world change, not a reply
                    # — so this only drops the receipt (same call place_ack
                    # already makes, and for the same reason).
                    pass
                else:
                    ack = ResetAck(
                        request_id=reset_info["request_id"],
                        sim_step=self._sim.step,
                        scene_revision=self._sim.scene_revision,
                    )
                    asyncio.run_coroutine_threadsafe(queue.put(ack), self._loop)

            # Placement acks (R12-607).  Carry the sim step so a client can line
            # the next camera frame up with a placement whose pose it knows.
            if self._loop:
                for res in self._sim.drain_place_results():
                    conn_id = res.pop("_conn_id", None)
                    queue = self._place_ack_qs.get(conn_id)
                    if queue is None:
                        # The client disconnected between asking and landing.
                        # The placement still happened — it is a world change,
                        # not a reply — so this only drops the receipt.
                        continue
                    pack = PlaceAck(sim_step=self._sim.step, **res)
                    asyncio.run_coroutine_threadsafe(queue.put(pack), self._loop)

        if self._recorder is not None:
            wall = time.monotonic() - sim_start
            self._recorder.finalize(total_steps=self._sim.step, duration_s=wall)
            log.info("Recording finalized: %s", self._recorder.run_dir)

        renderer.close()
        log.info("Sim thread stopped")

    async def _broadcast(self, queues: dict, msg) -> None:
        """Put `msg` on EVERY connected client's queue, newest wins.

        Runs on the event loop (scheduled from the sim thread), because
        asyncio.Queue is not thread-safe.

        On overflow the OLDEST entry is dropped rather than the newest, and
        rather than awaiting space: these are live streams, so a late frame or
        a stale joint sample is worth less than the current one, and a slow
        client must not be able to stall the producer for everyone else.  The
        previous shared queue used a blocking put(), so one wedged consumer
        could back the whole thing up.
        """
        for q in list(queues.values()):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

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

    # ── Execution lease ──────────────────────────────────────────────────
    #
    # One holder at a time.  While held, the scene is frozen against edits from
    # every client, and joint_command is accepted only from the named mover.
    #
    # This has to live here rather than in a client.  The browser panel reaches
    # this server on its own socket, so disabling a button in one page stops
    # nothing: a place_object arriving mid-grasp teleports the object out of a
    # closing gripper, and a reset restarts the world underneath a trajectory
    # already in flight.  The refusal belongs where every client's message
    # actually arrives.

    def _active_control(self) -> Optional[dict]:
        """The lease, or None — clearing it first if its TTL has run out.

        A holder that wedges or dies without a clean close must not freeze the
        scene until someone restarts the simulator, so the lease is time-bounded
        and every read is where expiry gets noticed.
        """
        control = self._control
        if control is not None and time.monotonic() >= control["expires_at"]:
            log.info("Execution lease held by %r expired", control["client_id"])
            self._control = None
            return None
        return control

    def _release_control(self, conn_id: int) -> bool:
        held = self._active_control()
        if held is not None and held["conn_id"] == conn_id:
            log.info("Execution lease released by %r", held["client_id"])
            self._control = None
            return True
        return False

    def _control_ack(self, request_id: str, granted: bool,
                     error: str = "") -> ControlAck:
        held = self._active_control()
        return ControlAck(
            request_id=request_id,
            granted=granted,
            held=held is not None,
            holder=held["client_id"] if held else "",
            motion_client_id=held["motion_client_id"] if held else "",
            expires_in_s=(round(held["expires_at"] - time.monotonic(), 2)
                          if held else 0.0),
            error=error,
        )

    def _blocked_by_control(self, mtype: str, client_id: str) -> Optional[str]:
        """Why this message is refused right now, or None to let it through."""
        held = self._active_control()
        if held is None:
            return None
        if mtype in ("place_object", "scene_load", "reset"):
            return (f"{mtype} is refused while {held['client_id']!r} holds the "
                    f"execution lease")
        if mtype == "joint_command":
            mover = held["motion_client_id"]
            # An empty mover means the lease named nobody, so nobody may move.
            # Comparing directly would instead match every client that sent no
            # client_id in its hello, handing the arm to any anonymous client.
            if not mover:
                return ("joint_command is refused: the lease held by "
                        f"{held['client_id']!r} names no motion client")
            if client_id != mover:
                return (f"joint_command is refused: {mover!r} is the motion "
                        f"client for the lease held by {held['client_id']!r}")
        return None

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
            client_id = str(msg.get("client_id") or "")
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
            # Advertised so a client can tell an arbitrating server from an
            # older one rather than assuming its lease request was honoured.
            capabilities={"execution_lease": True},
        )
        await ws.send(ack.encode())
        self._connected_ws = ws
        conn_id = self._next_conn_id
        self._next_conn_id += 1
        self._place_ack_qs[conn_id] = asyncio.Queue(maxsize=32)
        self._reset_ack_qs[conn_id] = asyncio.Queue(maxsize=4)
        self._state_qs[conn_id] = asyncio.Queue(maxsize=10)
        self._frame_qs[conn_id] = asyncio.Queue(maxsize=4)
        log.info("Handshake complete with %s", addr)

        last_hb_recv = time.monotonic()

        async def _send_loop() -> None:
            while True:
                # Drain state
                try:
                    state = self._state_qs[conn_id].get_nowait()
                    await ws.send(state.encode())
                except asyncio.QueueEmpty:
                    pass
                # Drain one camera frame
                try:
                    frame = self._frame_qs[conn_id].get_nowait()
                    await ws.send(frame.encode())
                except asyncio.QueueEmpty:
                    pass
                # Drain THIS connection's reset ack
                try:
                    rack = self._reset_ack_qs[conn_id].get_nowait()
                    await ws.send(rack.encode())
                except (asyncio.QueueEmpty, KeyError):
                    pass
                # Drain THIS connection's placement acks
                try:
                    pack = self._place_ack_qs[conn_id].get_nowait()
                    await ws.send(pack.encode())
                except (asyncio.QueueEmpty, KeyError):
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

                refusal = self._blocked_by_control(mtype, client_id)
                if refusal is not None:
                    await ws.send(Error(code="control_held",
                                        message=refusal).encode())
                    continue

                if mtype == "acquire_control":
                    held = self._active_control()
                    if held is not None and held["conn_id"] != conn_id:
                        await ws.send(self._control_ack(
                            decoded.get("request_id", ""), False,
                            f"lease already held by {held['client_id']!r}",
                        ).encode())
                        continue
                    ttl = float(decoded.get("ttl_s") or 120.0)
                    ttl = max(1.0, min(ttl, _MAX_CONTROL_TTL_S))
                    self._control = {
                        "conn_id": conn_id,
                        "client_id": str(decoded.get("client_id") or client_id),
                        "motion_client_id": str(
                            decoded.get("motion_client_id") or ""),
                        "expires_at": time.monotonic() + ttl,
                        "reason": str(decoded.get("reason") or ""),
                    }
                    log.info("Execution lease granted to %r (mover %r, %.0fs)",
                             self._control["client_id"],
                             self._control["motion_client_id"], ttl)
                    await ws.send(self._control_ack(
                        decoded.get("request_id", ""), True).encode())
                    continue

                if mtype == "release_control":
                    self._release_control(conn_id)
                    await ws.send(self._control_ack(
                        decoded.get("request_id", ""), False).encode())
                    continue

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
                    decoded["_conn_id"] = conn_id
                    self._sim.submit_reset(decoded)
                    if self._recorder is not None:
                        self._recorder.record_reset(
                            decoded.get("seed"), self._sim.step
                        )

                elif mtype == "pause":
                    self._sim.submit_pause(bool(decoded.get("paused", True)))

                elif mtype == "zoom_command":
                    # R12-605. Applied on the render thread's next frame, not
                    # here: cam_fovy and the distortion maps are the renderer's
                    # state, and mutating them from the websocket task would
                    # race a render in progress.
                    try:
                        lvl = parse_level(decoded.get("level", "inter"))
                    except ValueError as exc:
                        await ws.send(Error(message=str(exc)).encode())
                    else:
                        self._pending_zoom = lvl
                        log.info("Zoom command: %s", lvl.value)

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

                elif mtype == "place_object":
                    # Queued for the sim thread; the ack comes back from there
                    # once it has actually run, so an accepted ack means the
                    # object is on the board, not that the request parsed.
                    decoded["_conn_id"] = conn_id
                    self._sim.submit_place(decoded)

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
            # A holder that drops its socket must not leave the scene frozen.
            self._release_control(conn_id)
            self._place_ack_qs.pop(conn_id, None)
            self._reset_ack_qs.pop(conn_id, None)
            self._state_qs.pop(conn_id, None)
            self._frame_qs.pop(conn_id, None)
            log.info("Handler exited for %s", addr)

    def _build_recorder_manifest(self) -> dict:
        """Build a provenance manifest with actual model/scene paths and versions.

        Records immutable identity for the compiled world so Epic 8 can reject
        experience from incompatible model/scene/physics identities.
        """
        import hashlib
        import platform
        import sys as _sys

        def _sha256(path: Optional[str]) -> Optional[str]:
            if not path:
                return None
            try:
                return hashlib.sha256(
                    pathlib.Path(path).read_bytes()
                ).hexdigest()
            except OSError:
                return None

        mujoco_version = getattr(mujoco, "__version__", "unknown")
        return {
            "model_path": self._model_path,
            "model_sha256": _sha256(self._model_path),
            "scene_path": self._scene_path,
            "scene_sha256": _sha256(self._scene_path),
            "scene_revision": self._sim.scene_revision,
            "mujoco_version": mujoco_version,
            "python_version": _sys.version,
            "platform": platform.platform(),
            "protocol_version": PROTOCOL_VERSION,
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
        }

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
    ap.add_argument("--distortion", action="store_true",
                    help="apply the calibration profile's barrel distortion to "
                         "rendered frames (and to depth/segmentation, so labels "
                         "stay aligned). Off by default: renders are ground "
                         "truth for the collision and evaluation paths.")
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
        enable_distortion=args.distortion,
        record_dir=args.record,
    )
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
