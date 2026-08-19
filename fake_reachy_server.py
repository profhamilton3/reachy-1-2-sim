"""Standalone fake gRPC server implementing reachy-sdk-api v1 for simulator use.

Replaces the ROS reachy_bringup fake mode. Provides realistic joint states for
a full Reachy 1.2 (both arms + head) without any ROS infrastructure.

Epic 1 (R12-100 through R12-103):
  - KinematicBackend (kinematic_backend.py) owns one authoritative 50 Hz loop.
  - StreamJointsState reads immutable snapshots — never advances state.
  - Motion convergence is invariant to subscriber count.
  - JSON file written via atomic rename to prevent partial reads (R12-103).

Domain models and backend logic are in kinematic_backend.py (no gRPC dep) so
they can be unit-tested on the host without installing grpcio.
"""

import logging
import math
import os
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import grpc
import numpy as np
from google.protobuf.empty_pb2 import Empty
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue, UInt32Value
from scipy.optimize import minimize

from reachy_sdk_api import (
    arm_kinematics_pb2,
    arm_kinematics_pb2_grpc,
    camera_reachy_pb2,
    camera_reachy_pb2_grpc,
    fan_pb2,
    fan_pb2_grpc,
    head_kinematics_pb2,
    head_kinematics_pb2_grpc,
    joint_pb2,
    joint_pb2_grpc,
    kinematics_pb2,
    sensor_pb2,
    sensor_pb2_grpc,
)

from camera_fixture import CameraFixture, CameraFrame, LEFT, RIGHT, frame_file_writer

from kinematic_backend import (
    JOINT_DEFS,
    JointCommand,
    JointSample,
    KinematicBackend,
    SimulationSnapshot,
    state_file_writer,
)

from mujoco_remote_backend import KinematicBridge, MujocoRemoteBackend

_BACKEND_ENV = "REACHY_SIM_BACKEND"
_MUJOCO_LEFT_JPG  = pathlib.Path("/tmp/reachy_left.jpg")
_MUJOCO_RIGHT_JPG = pathlib.Path("/tmp/reachy_right.jpg")
_SCENE_OVERRIDES = os.environ.get(
    "REACHY_SIM_SCENE_OVERRIDES", "/tmp/reachy_scene_overrides.json"
)
_INTERACTIVE_STATE = os.environ.get(
    "REACHY_SIM_INTERACTIVE_STATE", "/tmp/reachy_interactive_state.json"
)
_RESET_REQUEST = os.environ.get(
    "REACHY_SIM_RESET_REQUEST", "/tmp/reachy_reset_request"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [fake-reachy] %(message)s")
log = logging.getLogger(__name__)

PORT = 50051

import json
import time


def object_overrides_writer(remote, path: str = _SCENE_OVERRIDES, hz: float = 15.0) -> None:
    """Publish live physics object poses to the scene-marker overrides file.

    In the mujoco-remote backend the objects physically move; this mirrors their
    tracked world positions into /tmp/reachy_scene_overrides.json so the RViz
    scene markers follow the real objects (the same file the demo writes for
    fake attachment in kinematic mode).  {object_id: [x, y, z]}.
    """
    period = 1.0 / hz
    tmp = path + ".tmp"
    while True:
        try:
            snap = remote.latest_snapshot()
            objs = getattr(snap, "objects", None) or {}
            if objs:
                data = {
                    oid: list(o.get("pos_xyz", [0.0, 0.0, 0.0]))
                    for oid, o in objs.items()
                }
                with open(tmp, "w") as f:
                    json.dump(data, f)
                os.replace(tmp, path)
        except Exception:
            pass
        time.sleep(period)


def interactive_state_writer(remote, path: str = _INTERACTIVE_STATE, hz: float = 15.0) -> None:
    """Publish live interactive-control on/off states to a JSON file.

    Mirrors the physics server's button/switch/lever states into
    /tmp/reachy_interactive_state.json so the practice script (and any RViz
    overlay) can read them for visible + closed-loop feedback.
    {control_id: {"type": ..., "on": bool, "value": float}}.
    """
    period = 1.0 / hz
    tmp = path + ".tmp"
    while True:
        try:
            snap = remote.latest_snapshot()
            controls = getattr(snap, "interactive", None) or {}
            data = {
                cid: {
                    "type": c.get("type", ""),
                    "on": bool(c.get("on", False)),
                    "value": float(c.get("value", 0.0)),
                }
                for cid, c in controls.items()
            }
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, path)
        except Exception:
            pass
        time.sleep(period)


def reset_watcher(remote, path: str = _RESET_REQUEST, hz: float = 10.0) -> None:
    """Trigger a physics reset when a client (e.g. the demo) drops a sentinel
    file.  A file-based signal avoids threading a reset RPC through gRPC."""
    period = 1.0 / hz
    while True:
        try:
            if os.path.exists(path):
                os.remove(path)
                remote.request_reset()
        except Exception:
            pass
        time.sleep(period)


class _MujocoRemoteCameraAdapter:
    """Serve MuJoCo camera frames from the tmp files written by MujocoRemoteBackend.

    Presents the same latest_frame()/render_single_frame() interface as
    CameraFixture so FakeCameraService works with either backend unchanged.
    """

    def __init__(self) -> None:
        self._seqs: dict = {LEFT: 0, RIGHT: 0}
        self._last: dict = {LEFT: None, RIGHT: None}

    def latest_frame(self, cam_id: int) -> Optional[CameraFrame]:
        path = _MUJOCO_LEFT_JPG if cam_id == LEFT else _MUJOCO_RIGHT_JPG
        try:
            data = path.read_bytes()
        except OSError:
            return self._last[cam_id]
        self._seqs[cam_id] += 1
        frame = CameraFrame(
            camera_id=cam_id,
            sequence=self._seqs[cam_id],
            wall_time_ns=time.time_ns(),
            width=0,
            height=0,
            jpeg_bytes=data,
        )
        self._last[cam_id] = frame
        return frame

    def render_single_frame(self, cam_id: int) -> CameraFrame:
        frame = self.latest_frame(cam_id)
        if frame is not None:
            return frame
        # No frame from the native server yet — generate a dark placeholder.
        import io
        from PIL import Image as _PIL
        buf = io.BytesIO()
        _PIL.new("RGB", (320, 240), (10, 10, 30)).save(buf, format="JPEG", quality=70)
        return CameraFrame(
            camera_id=cam_id, sequence=0,
            wall_time_ns=time.time_ns(), width=320, height=240,
            jpeg_bytes=buf.getvalue(),
        )

    def start(self) -> None:
        pass  # frames arrive via MujocoRemoteBackend._ingest_camera_frame

_FORCE_SENSOR_DEFS = [
    ("r_force_gripper", 1),
    ("l_force_gripper", 2),
]

_FAN_DEFS = [
    ("r_arm_fan",  1),
    ("l_arm_fan",  2),
    ("head_fan",   3),
]


def _ts() -> Timestamp:
    t = time.time()
    return Timestamp(seconds=int(t), nanos=int((t % 1) * 1e9))


def _sample_to_proto(s: JointSample) -> joint_pb2.JointState:
    return joint_pb2.JointState(
        name=s.name,
        uid=UInt32Value(value=s.uid),
        present_position=FloatValue(value=s.present_position),
        present_speed=FloatValue(value=s.present_speed),
        present_load=FloatValue(value=s.present_load),
        temperature=FloatValue(value=s.temperature),
        compliant=BoolValue(value=s.compliant),
        goal_position=FloatValue(value=s.goal_position),
        speed_limit=FloatValue(value=s.speed_limit),
        torque_limit=FloatValue(value=s.torque_limit),
    )


# ── gRPC adapters ─────────────────────────────────────────────────────────────

class FakeJointService(joint_pb2_grpc.JointServiceServicer):
    """gRPC adapter — reads backend snapshots, submits commands. Never mutates state."""

    def __init__(self, backend) -> None:
        self._backend = backend

    def _resolve(self, ids, snapshot: SimulationSnapshot) -> List[JointSample]:
        if not ids:
            return list(snapshot.joints.values())
        result: List[JointSample] = []
        for jid in ids:
            field_name = jid.WhichOneof("id")
            if field_name == "uid":
                for s in snapshot.joints.values():
                    if s.uid == jid.uid:
                        result.append(s)
                        break
            elif field_name == "name" and jid.name in snapshot.joints:
                result.append(snapshot.joints[jid.name])
        return result

    def GetAllJointsId(self, request, context):
        snap = self._backend.latest_snapshot()
        names = list(snap.joints.keys())
        uids = [snap.joints[n].uid for n in names]
        return joint_pb2.JointsId(names=names, uids=uids)

    def GetJointsState(self, request, context):
        snap = self._backend.latest_snapshot()
        samples = self._resolve(request.ids, snap)
        ids = [joint_pb2.JointId(uid=s.uid) for s in samples]
        states = [_sample_to_proto(s) for s in samples]
        return joint_pb2.JointsState(ids=ids, states=states, timestamp=_ts())

    def StreamJointsState(self, request, context):
        """Yield snapshots at requested frequency. Reads backend — never advances state."""
        freq = max(request.publish_frequency, 1.0)
        dt = 1.0 / freq
        init_snap = self._backend.latest_snapshot()
        requested_names = [s.name for s in self._resolve(request.request.ids, init_snap)]
        while context.is_active():
            snap = self._backend.latest_snapshot()
            samples = [snap.joints[n] for n in requested_names if n in snap.joints]
            ids = [joint_pb2.JointId(uid=s.uid) for s in samples]
            states = [_sample_to_proto(s) for s in samples]
            yield joint_pb2.JointsState(ids=ids, states=states, timestamp=_ts())
            time.sleep(dt)

    def SendJointsCommands(self, request, context):
        self._submit_batch(request)
        return joint_pb2.JointsCommandAck(success=True)

    def StreamJointsCommands(self, request_iterator, context):
        for batch in request_iterator:
            self._submit_batch(batch)
        return joint_pb2.JointsCommandAck(success=True)

    def _submit_batch(self, batch) -> None:
        for jcmd in batch.commands:
            field_name = jcmd.id.WhichOneof("id")
            if field_name == "uid":
                uid: Optional[int] = jcmd.id.uid
            elif field_name == "name":
                uid = self._backend.uid_by_name(jcmd.id.name)
            else:
                uid = None
            if uid is None:
                continue
            self._backend.submit_command(JointCommand(
                uid=uid,
                goal_position=jcmd.goal_position.value if jcmd.HasField("goal_position") else None,
                compliant=jcmd.compliant.value if jcmd.HasField("compliant") else None,
                speed_limit=jcmd.speed_limit.value if jcmd.HasField("speed_limit") else None,
                torque_limit=jcmd.torque_limit.value if jcmd.HasField("torque_limit") else None,
            ))


class FakeSensorService(sensor_pb2_grpc.SensorServiceServicer):
    def __init__(self, remote_backend=None) -> None:
        self._sensors = {
            uid: {"name": n, "uid": uid}
            for n, uid in _FORCE_SENSOR_DEFS
        }
        # When a MujocoRemoteBackend is provided, read live gripper forces from it.
        self._remote = remote_backend

    def _force(self, uid: int) -> float:
        if self._remote is not None:
            return self._remote.latest_snapshot().force_sensors.get(uid, 0.0)
        return 0.0

    def GetAllForceSensorsId(self, request, context):
        names = [s["name"] for s in self._sensors.values()]
        uids  = [s["uid"]  for s in self._sensors.values()]
        return sensor_pb2.SensorsId(names=names, uids=uids)

    def GetSensorsState(self, request, context):
        sensors = [self._sensors[sid.uid] for sid in request.ids if sid.uid in self._sensors]
        ids = [sensor_pb2.SensorId(uid=s["uid"]) for s in sensors]
        states = [
            sensor_pb2.SensorState(
                force_sensor_state=sensor_pb2.ForceSensorState(force=self._force(s["uid"]))
            )
            for s in sensors
        ]
        return sensor_pb2.SensorsState(ids=ids, states=states, timestamp=_ts())

    def StreamSensorStates(self, request, context):
        sensors = [
            self._sensors[sid.uid]
            for sid in request.request.ids
            if sid.uid in self._sensors
        ]
        freq = max(request.publish_frequency, 1.0)
        dt = 1.0 / freq
        while context.is_active():
            ids = [sensor_pb2.SensorId(uid=s["uid"]) for s in sensors]
            states = [
                sensor_pb2.SensorState(
                    force_sensor_state=sensor_pb2.ForceSensorState(force=self._force(s["uid"]))
                )
                for s in sensors
            ]
            yield sensor_pb2.SensorsState(ids=ids, states=states, timestamp=_ts())
            time.sleep(dt)


class FakeHeadKinematicsService(head_kinematics_pb2_grpc.HeadKinematicsServicer):
    """Minimal analytic head kinematics — converts quaternion to neck RPY."""

    _NECK_UIDS = [30, 31, 32]  # neck_roll, neck_pitch, neck_yaw

    def ComputeHeadIK(self, request, context):
        q = request.q
        roll, pitch, yaw = _quat_to_rpy(q.w, q.x, q.y, q.z)
        ids = [joint_pb2.JointId(uid=uid) for uid in self._NECK_UIDS]
        neck_pos = kinematics_pb2.JointPosition(ids=ids, positions=[roll, pitch, yaw])
        return head_kinematics_pb2.HeadIKSolution(success=True, neck_position=neck_pos)

    def ComputeHeadFK(self, request, context):
        positions = request.neck_position.positions
        roll  = positions[0] if len(positions) > 0 else 0.0
        pitch = positions[1] if len(positions) > 1 else 0.0
        yaw   = positions[2] if len(positions) > 2 else 0.0
        w, x, y, z = _rpy_to_quat(roll, pitch, yaw)
        return head_kinematics_pb2.HeadFKSolution(
            success=True,
            q=kinematics_pb2.Quaternion(w=w, x=x, y=y, z=z),
        )


# ── Arm kinematics ────────────────────────────────────────────────────────────
# Approximate Reachy 1.2 geometry.  At q=0 the arm hangs straight down (-Z).

_R_SHOULDER_Y = -0.19
_L_SHOULDER_Y =  0.19
_UPPER_ARM    =  0.28
_FOREARM      =  0.25

_R_ARM_LIMITS = [
    (-3.14,  1.57), (-1.57,  0.17), (-2.09,  2.09),
    (-2.35,  0.0),  (-2.62,  2.62), (-0.79,  0.79), (-0.79,  0.79),
]
_L_ARM_LIMITS = [
    (-3.14,  1.57), (-0.17,  1.57), (-2.09,  2.09),
    (-2.35,  0.0),  (-2.62,  2.62), (-0.79,  0.79), (-0.79,  0.79),
]


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], dtype=float)

def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=float)

def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)

def _tx(x, y, z):
    T = np.eye(4); T[0,3], T[1,3], T[2,3] = x, y, z; return T


def _arm_fk(q: np.ndarray, side: str) -> np.ndarray:
    shoulder_y = _R_SHOULDER_Y if side == "right" else _L_SHOULDER_Y
    roll_sign  = 1.0 if side == "right" else -1.0
    T = _tx(0.0, shoulder_y, 0.0)
    T = T @ _ry(q[0]) @ _rx(roll_sign * q[1]) @ _rz(q[2])
    T = T @ _tx(0, 0, -_UPPER_ARM)
    T = T @ _ry(q[3]) @ _rz(q[4])
    T = T @ _tx(0, 0, -_FOREARM)
    T = T @ _ry(q[5]) @ _rx(q[6])
    return T


def _arm_ik(target: np.ndarray, side: str, q0=None):
    bounds = _R_ARM_LIMITS if side == "right" else _L_ARM_LIMITS
    q_init = np.zeros(7) if q0 is None else np.asarray(q0, dtype=float)[:7]
    def cost(q):
        T = _arm_fk(q, side)
        return np.sum((T[:3,3] - target[:3,3])**2) + 0.1*np.sum((T[:3,:3] - target[:3,:3])**2)
    res = minimize(cost, q_init, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-6})
    return res.x, bool(res.fun < 0.01)


class FakeArmKinematicsService(arm_kinematics_pb2_grpc.ArmKinematicsServicer):
    def ComputeArmIK(self, request, context):
        side = "left" if request.target.side == arm_kinematics_pb2.LEFT else "right"
        target = np.array(list(request.target.pose.data), dtype=float).reshape(4, 4)
        q0 = list(request.q0.positions) or None
        joints, success = _arm_ik(target, side, q0)
        if not success:
            log.warning("ArmIK: no solution for target %s", target[:3, 3])
        arm_side = arm_kinematics_pb2.LEFT if side == "left" else arm_kinematics_pb2.RIGHT
        arm_pos = arm_kinematics_pb2.ArmJointPosition(
            side=arm_side,
            positions=kinematics_pb2.JointPosition(positions=joints.tolist()),
        )
        return arm_kinematics_pb2.ArmIKSolution(success=success, arm_position=arm_pos)

    def ComputeArmFK(self, request, context):
        side = "left" if request.arm_position.side == arm_kinematics_pb2.LEFT else "right"
        q = np.array(list(request.arm_position.positions.positions), dtype=float)
        T = _arm_fk(q[:7], side)
        arm_side = arm_kinematics_pb2.LEFT if side == "left" else arm_kinematics_pb2.RIGHT
        end_eff = arm_kinematics_pb2.ArmEndEffector(
            side=arm_side,
            pose=kinematics_pb2.Matrix4x4(data=T.flatten().tolist()),
        )
        return arm_kinematics_pb2.ArmFKSolution(success=True, end_effector=end_eff)


def _quat_to_rpy(w, x, y, z):
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp  = 2*(w*y - z*x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def _rpy_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll/2),  math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
    return (cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy)


class FakeFanService(fan_pb2_grpc.FanControllerServiceServicer):
    def __init__(self) -> None:
        self._fans = {uid: {"name": n, "uid": uid, "on": False} for n, uid in _FAN_DEFS}

    def GetAllFansId(self, request, context):
        return fan_pb2.FansId(
            names=[f["name"] for f in self._fans.values()],
            uids=[f["uid"] for f in self._fans.values()],
        )

    def GetFansState(self, request, context):
        fans = [self._fans[fid.uid] for fid in request.ids if fid.uid in self._fans]
        return fan_pb2.FansState(
            ids=[fan_pb2.FanId(uid=f["uid"]) for f in fans],
            states=[fan_pb2.FanState(on=f["on"]) for f in fans],
        )

    def SendFansCommands(self, request, context):
        for fc in request.commands:
            if fc.id.uid in self._fans:
                self._fans[fc.id.uid]["on"] = fc.state.on
        return Empty()


# ── Camera service ─────────────────────────────────────────────────────────────

_STREAM_FRAME_HZ = 15.0  # max frame rate for StreamImage


class FakeCameraService(camera_reachy_pb2_grpc.CameraServiceServicer):
    """gRPC adapter over CameraFixture.

    GetImage returns the latest buffered frame.
    StreamImage yields frames at _STREAM_FRAME_HZ until the client cancels.
    Zoom/focus RPCs return stable simulated values with success=True.
    """

    def __init__(self, fixture: CameraFixture) -> None:
        self._fixture = fixture
        # Simulated zoom/focus state — optical effects not modelled.
        self._zoom_level = camera_reachy_pb2.ZoomLevelPossibilities.ZERO
        self._zoom_speed = 10000
        self._left_focus = 0
        self._right_focus = 0
        self._left_zoom = 0
        self._right_zoom = 0

    def _camera_id(self, request_camera) -> int:
        return LEFT if request_camera.id == camera_reachy_pb2.CameraId.LEFT else RIGHT

    def _get_frame_bytes(self, cam_id: int) -> bytes:
        frame = self._fixture.latest_frame(cam_id)
        if frame is None:
            frame = self._fixture.render_single_frame(cam_id)
        return frame.jpeg_bytes

    # ── unary RPCs ──────────────────────────────────────────────────────────

    def GetImage(self, request, context):
        cam_id = self._camera_id(request.camera)
        return camera_reachy_pb2.Image(data=self._get_frame_bytes(cam_id))

    def GetZoomLevel(self, request, context):
        return camera_reachy_pb2.ZoomLevel(level=self._zoom_level)

    def GetZoomSpeed(self, request, context):
        return camera_reachy_pb2.ZoomSpeed(speed=self._zoom_speed)

    def SendZoomCommand(self, request, context):
        which = request.WhichOneof("command")
        if which == "level_command":
            self._zoom_level = request.level_command.level
        elif which == "speed_command":
            self._zoom_speed = request.speed_command.speed
        # homing_command resets to zero
        elif which == "homing_command":
            self._zoom_level = camera_reachy_pb2.ZoomLevelPossibilities.ZERO
        return camera_reachy_pb2.ZoomCommandAck(success=True)

    def GetZoomFocus(self, request, context):
        from google.protobuf.wrappers_pb2 import UInt32Value
        return camera_reachy_pb2.ZoomFocusMessage(
            left_focus=UInt32Value(value=self._left_focus),
            right_focus=UInt32Value(value=self._right_focus),
            left_zoom=UInt32Value(value=self._left_zoom),
            right_zoom=UInt32Value(value=self._right_zoom),
        )

    def SetZoomFocus(self, request, context):
        if request.HasField("left_focus"):
            self._left_focus = request.left_focus.value
        if request.HasField("right_focus"):
            self._right_focus = request.right_focus.value
        if request.HasField("left_zoom"):
            self._left_zoom = request.left_zoom.value
        if request.HasField("right_zoom"):
            self._right_zoom = request.right_zoom.value
        return camera_reachy_pb2.ZoomCommandAck(success=True)

    def StartAutofocus(self, request, context):
        return camera_reachy_pb2.ZoomCommandAck(success=True)

    def StopAutofocus(self, request, context):
        return camera_reachy_pb2.ZoomCommandAck(success=True)

    # ── streaming RPC ────────────────────────────────────────────────────────

    def StreamImage(self, request, context):
        cam_id = self._camera_id(request.request.camera)
        dt = 1.0 / _STREAM_FRAME_HZ
        last_seq = -1
        while context.is_active():
            frame = self._fixture.latest_frame(cam_id)
            if frame is not None and frame.sequence != last_seq:
                last_seq = frame.sequence
                yield camera_reachy_pb2.Image(data=frame.jpeg_bytes)
            time.sleep(dt)


# ── Server entrypoint ─────────────────────────────────────────────────────────

def serve() -> None:
    backend_type = os.environ.get(_BACKEND_ENV, "kinematic").lower()

    if backend_type == "mujoco-remote":
        mujoco = MujocoRemoteBackend()
        mujoco.start()
        joint_backend = KinematicBridge(mujoco)
        sensor_backend = mujoco
        camera = _MujocoRemoteCameraAdapter()
        camera.start()
        threading.Thread(
            target=object_overrides_writer, args=(mujoco,), daemon=True,
            name="object-overrides-writer",
        ).start()
        threading.Thread(
            target=interactive_state_writer, args=(mujoco,), daemon=True,
            name="interactive-state-writer",
        ).start()
        threading.Thread(
            target=reset_watcher, args=(mujoco,), daemon=True,
            name="reset-watcher",
        ).start()
        log.info("Backend: mujoco-remote → %s", mujoco._url)
    else:
        if backend_type not in ("kinematic", "fixture"):
            log.warning("Unknown backend %r — falling back to kinematic", backend_type)
        kinematic = KinematicBackend()
        kinematic.start()
        joint_backend = kinematic
        sensor_backend = None
        camera = CameraFixture()
        camera.start()
        threading.Thread(
            target=frame_file_writer, args=(camera,), daemon=True, name="frame-writer"
        ).start()
        log.info("Backend: %s", backend_type)

    threading.Thread(
        target=state_file_writer, args=(joint_backend,), daemon=True, name="state-writer"
    ).start()

    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    joint_pb2_grpc.add_JointServiceServicer_to_server(FakeJointService(joint_backend), server)
    sensor_pb2_grpc.add_SensorServiceServicer_to_server(FakeSensorService(sensor_backend), server)
    fan_pb2_grpc.add_FanControllerServiceServicer_to_server(FakeFanService(), server)
    head_kinematics_pb2_grpc.add_HeadKinematicsServicer_to_server(
        FakeHeadKinematicsService(), server
    )
    arm_kinematics_pb2_grpc.add_ArmKinematicsServicer_to_server(
        FakeArmKinematicsService(), server
    )
    camera_reachy_pb2_grpc.add_CameraServiceServicer_to_server(
        FakeCameraService(camera), server
    )

    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    snap = joint_backend.latest_snapshot()
    log.info("Fake Reachy gRPC server listening on port %d", PORT)
    log.info("Joints: %s", list(snap.joints.keys()))
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
