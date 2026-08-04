"""Standalone fake gRPC server implementing reachy-sdk-api v1 for simulator use.

Replaces the ROS reachy_bringup fake mode. Provides realistic joint states for
a full Reachy 1.2 (both arms + head) without any ROS infrastructure.

Note: JointState fields use google.protobuf wrapper types (FloatValue, BoolValue,
UInt32Value), not plain Python scalars.
"""

import json
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import grpc
import numpy as np
from google.protobuf.empty_pb2 import Empty
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue, UInt32Value
from scipy.optimize import minimize

from reachy_sdk_api import (
    arm_kinematics_pb2,
    arm_kinematics_pb2_grpc,
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [fake-reachy] %(message)s")
log = logging.getLogger(__name__)

PORT = 50051

# Reachy 1.2: right arm (IDs 10-17), left arm (IDs 20-27), head (IDs 30-34)
_JOINT_DEFS = [
    ("r_shoulder_pitch", 10),
    ("r_shoulder_roll", 11),
    ("r_arm_yaw", 12),
    ("r_elbow_pitch", 13),
    ("r_forearm_yaw", 14),
    ("r_wrist_pitch", 15),
    ("r_wrist_roll", 16),
    ("r_gripper", 17),
    ("l_shoulder_pitch", 20),
    ("l_shoulder_roll", 21),
    ("l_arm_yaw", 22),
    ("l_elbow_pitch", 23),
    ("l_forearm_yaw", 24),
    ("l_wrist_pitch", 25),
    ("l_wrist_roll", 26),
    ("l_gripper", 27),
    ("neck_roll", 30),
    ("neck_pitch", 31),
    ("neck_yaw", 32),
    ("l_antenna", 33),
    ("r_antenna", 34),
]

_FORCE_SENSOR_DEFS = [
    ("r_force_gripper", 1),
    ("l_force_gripper", 2),
]

_FAN_DEFS = [
    ("r_arm_fan", 1),
    ("l_arm_fan", 2),
    ("head_fan", 3),
]


def _ts():
    t = time.time()
    return Timestamp(seconds=int(t), nanos=int((t % 1) * 1e9))


class JointState:
    def __init__(self, name: str, uid: int):
        self.name = name
        self.uid = uid
        # All positions are in radians (SDK converts to/from degrees)
        self.present_position = 0.0
        self.goal_position = 0.0
        self.present_speed = 0.0
        self.present_load = 0.0
        self.temperature = 35.0
        # Antenna joints (UIDs 33-34) are always stiff on physical Reachy — no turn_on needed
        self.compliant = uid not in (33, 34)
        self.speed_limit = 0.0
        self.torque_limit = 100.0

    def to_proto(self) -> joint_pb2.JointState:
        return joint_pb2.JointState(
            name=self.name,
            uid=UInt32Value(value=self.uid),
            present_position=FloatValue(value=self.present_position),
            present_speed=FloatValue(value=self.present_speed),
            present_load=FloatValue(value=self.present_load),
            temperature=FloatValue(value=self.temperature),
            compliant=BoolValue(value=self.compliant),
            goal_position=FloatValue(value=self.goal_position),
            speed_limit=FloatValue(value=self.speed_limit),
            torque_limit=FloatValue(value=self.torque_limit),
        )


class FakeJointService(joint_pb2_grpc.JointServiceServicer):
    def __init__(self, joints: Dict[int, JointState]):
        self._joints = joints  # uid → JointState
        self._by_name = {j.name: j for j in joints.values()}

    def _resolve(self, ids):
        if not ids:
            return list(self._joints.values())
        result = []
        for jid in ids:
            field = jid.WhichOneof("id")
            if field == "uid" and jid.uid in self._joints:
                result.append(self._joints[jid.uid])
            elif field == "name" and jid.name in self._by_name:
                result.append(self._by_name[jid.name])
        return result

    def GetAllJointsId(self, request, context):
        names = [j.name for j in self._joints.values()]
        uids = [j.uid for j in self._joints.values()]
        return joint_pb2.JointsId(names=names, uids=uids)

    def GetJointsState(self, request, context):
        joints = self._resolve(request.ids)
        ids = [joint_pb2.JointId(uid=j.uid) for j in joints]
        states = [j.to_proto() for j in joints]
        return joint_pb2.JointsState(ids=ids, states=states, timestamp=_ts())

    def StreamJointsState(self, request, context):
        joints = self._resolve(request.request.ids)
        freq = max(request.publish_frequency, 1.0)
        dt = 1.0 / freq
        while context.is_active():
            # Smoothly move present_position toward goal_position (fake servo)
            for j in joints:
                if not j.compliant:
                    diff = j.goal_position - j.present_position
                    j.present_position += diff * min(dt * 5.0, 1.0)
            ids = [joint_pb2.JointId(uid=j.uid) for j in joints]
            states = [j.to_proto() for j in joints]
            yield joint_pb2.JointsState(ids=ids, states=states, timestamp=_ts())
            time.sleep(dt)

    def SendJointsCommands(self, request, context):
        self._apply_commands(request)
        return joint_pb2.JointsCommandAck(success=True)

    def StreamJointsCommands(self, request_iterator, context):
        # Client-streaming RPC: consume the command stream, return one ack at end
        for cmd in request_iterator:
            self._apply_commands(cmd)
        return joint_pb2.JointsCommandAck(success=True)

    def _apply_commands(self, cmd):
        for jcmd in cmd.commands:
            field = jcmd.id.WhichOneof("id")
            if field == "uid":
                j = self._joints.get(jcmd.id.uid)
            elif field == "name":
                j = self._by_name.get(jcmd.id.name)
            else:
                continue
            if j is None:
                continue
            if jcmd.HasField("goal_position"):
                j.goal_position = jcmd.goal_position.value
            if jcmd.HasField("compliant"):
                j.compliant = jcmd.compliant.value
            if jcmd.HasField("torque_limit"):
                j.torque_limit = jcmd.torque_limit.value
            if jcmd.HasField("speed_limit"):
                j.speed_limit = jcmd.speed_limit.value


class FakeSensorService(sensor_pb2_grpc.SensorServiceServicer):
    def __init__(self):
        self._sensors = {uid: {"name": n, "uid": uid, "force": 0.0} for n, uid in _FORCE_SENSOR_DEFS}

    def GetAllForceSensorsId(self, request, context):
        names = [s["name"] for s in self._sensors.values()]
        uids = [s["uid"] for s in self._sensors.values()]
        return sensor_pb2.SensorsId(names=names, uids=uids)

    def GetSensorsState(self, request, context):
        sensors = [self._sensors[sid.uid] for sid in request.ids if sid.uid in self._sensors]
        ids = [sensor_pb2.SensorId(uid=s["uid"]) for s in sensors]
        states = [sensor_pb2.SensorState(force_sensor_state=sensor_pb2.ForceSensorState(force=s["force"])) for s in sensors]
        return sensor_pb2.SensorsState(ids=ids, states=states, timestamp=_ts())

    def StreamSensorStates(self, request, context):
        sensors = [self._sensors[sid.uid] for sid in request.request.ids if sid.uid in self._sensors]
        freq = max(request.publish_frequency, 1.0)
        dt = 1.0 / freq
        while context.is_active():
            ids = [sensor_pb2.SensorId(uid=s["uid"]) for s in sensors]
            states = [sensor_pb2.SensorState(force_sensor_state=sensor_pb2.ForceSensorState(force=s["force"])) for s in sensors]
            yield sensor_pb2.SensorsState(ids=ids, states=states, timestamp=_ts())
            time.sleep(dt)


class FakeHeadKinematicsService(head_kinematics_pb2_grpc.HeadKinematicsServicer):
    """Minimal analytic head kinematics — converts quaternion to neck RPY."""

    # Neck joint UIDs matching JOINT_DEFS
    _NECK_UIDS = [30, 31, 32]  # neck_roll, neck_pitch, neck_yaw

    def ComputeHeadIK(self, request, context):
        q = request.q
        # Convert quaternion to euler (roll, pitch, yaw) analytically
        roll, pitch, yaw = _quat_to_rpy(q.w, q.x, q.y, q.z)
        ids = [joint_pb2.JointId(uid=uid) for uid in self._NECK_UIDS]
        neck_pos = kinematics_pb2.JointPosition(
            ids=ids,
            positions=[roll, pitch, yaw],
        )
        return head_kinematics_pb2.HeadIKSolution(success=True, neck_position=neck_pos)

    def ComputeHeadFK(self, request, context):
        positions = request.neck_position.positions
        roll = positions[0] if len(positions) > 0 else 0.0
        pitch = positions[1] if len(positions) > 1 else 0.0
        yaw = positions[2] if len(positions) > 2 else 0.0
        w, x, y, z = _rpy_to_quat(roll, pitch, yaw)
        return head_kinematics_pb2.HeadFKSolution(
            success=True,
            q=kinematics_pb2.Quaternion(w=w, x=x, y=y, z=z),
        )


# ── Arm kinematics ────────────────────────────────────────────────────────────
# Approximate Reachy 1.2 geometry.  Joint axes match the URDF convention:
#   shoulder_pitch → Y,  shoulder_roll → X,  arm/forearm_yaw → Z,
#   elbow_pitch → Y,  wrist_pitch → Y,  wrist_roll → X.
# At q=0 the arm hangs straight down (-Z).

_R_SHOULDER_Y = -0.19   # right shoulder lateral offset (m)
_L_SHOULDER_Y =  0.19   # left  shoulder lateral offset (m)
_UPPER_ARM    =  0.28   # shoulder → elbow (m)
_FOREARM      =  0.25   # elbow → wrist   (m)

# Joint limits [min, max] in radians
_R_ARM_LIMITS = [
    (-3.14,  1.57),   # shoulder_pitch
    (-1.57,  0.17),   # shoulder_roll (right arm: negative = outward)
    (-2.09,  2.09),   # arm_yaw
    (-2.35,  0.0),    # elbow_pitch (always bends same direction)
    (-2.62,  2.62),   # forearm_yaw
    (-0.79,  0.79),   # wrist_pitch
    (-0.79,  0.79),   # wrist_roll
]
_L_ARM_LIMITS = [
    (-3.14,  1.57),
    (-0.17,  1.57),   # shoulder_roll (left arm: positive = outward)
    (-2.09,  2.09),
    (-2.35,  0.0),
    (-2.62,  2.62),
    (-0.79,  0.79),
    (-0.79,  0.79),
]


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], dtype=float)


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=float)


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)


def _tx(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    return T


def _arm_fk(q: np.ndarray, side: str) -> np.ndarray:
    """Forward kinematics for Reachy 1.2 7-DOF arm. q in radians."""
    shoulder_y = _R_SHOULDER_Y if side == "right" else _L_SHOULDER_Y
    roll_sign  = 1.0 if side == "right" else -1.0
    T = _tx(0.0, shoulder_y, 0.0)
    T = T @ _ry(q[0])                          # shoulder_pitch
    T = T @ _rx(roll_sign * q[1])              # shoulder_roll (mirrored for left)
    T = T @ _rz(q[2])                          # arm_yaw
    T = T @ _tx(0, 0, -_UPPER_ARM)             # upper arm link
    T = T @ _ry(q[3])                          # elbow_pitch
    T = T @ _rz(q[4])                          # forearm_yaw
    T = T @ _tx(0, 0, -_FOREARM)              # forearm link
    T = T @ _ry(q[5])                          # wrist_pitch
    T = T @ _rx(q[6])                          # wrist_roll
    return T


def _arm_ik(target: np.ndarray, side: str,
            q0: Optional[List[float]] = None) -> tuple:
    """Numerical IK via scipy L-BFGS-B. Returns (joints_rad, success)."""
    bounds = _R_ARM_LIMITS if side == "right" else _L_ARM_LIMITS
    q_init = np.zeros(7) if q0 is None else np.asarray(q0, dtype=float)[:7]

    def cost(q):
        T = _arm_fk(q, side)
        pos = np.sum((T[:3, 3] - target[:3, 3]) ** 2)
        rot = np.sum((T[:3, :3] - target[:3, :3]) ** 2)
        return pos + 0.1 * rot

    res = minimize(cost, q_init, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-6})
    return res.x, bool(res.fun < 0.01)


class FakeArmKinematicsService(arm_kinematics_pb2_grpc.ArmKinematicsServicer):
    """Numerical IK/FK for both arms using scipy optimization."""

    def ComputeArmIK(self, request, context):
        side = "left" if request.target.side == arm_kinematics_pb2.LEFT else "right"
        target = np.array(list(request.target.pose.data), dtype=float).reshape(4, 4)
        q0_vals = list(request.q0.positions)
        q0 = q0_vals if q0_vals else None
        joints, success = _arm_ik(target, side, q0)
        arm_side = arm_kinematics_pb2.LEFT if side == "left" else arm_kinematics_pb2.RIGHT
        joint_pos = kinematics_pb2.JointPosition(positions=joints.tolist())
        arm_pos = arm_kinematics_pb2.ArmJointPosition(side=arm_side, positions=joint_pos)
        if not success:
            log.warning("ArmIK: no solution found for target %s", target[:3, 3])
        return arm_kinematics_pb2.ArmIKSolution(success=success, arm_position=arm_pos)

    def ComputeArmFK(self, request, context):
        side = "left" if request.arm_position.side == arm_kinematics_pb2.LEFT else "right"
        q = np.array(list(request.arm_position.positions.positions), dtype=float)
        T = _arm_fk(q[:7], side)
        pose = kinematics_pb2.Matrix4x4(data=T.flatten().tolist())
        arm_side = arm_kinematics_pb2.LEFT if side == "left" else arm_kinematics_pb2.RIGHT
        end_eff = arm_kinematics_pb2.ArmEndEffector(side=arm_side, pose=pose)
        return arm_kinematics_pb2.ArmFKSolution(success=True, end_effector=end_eff)


def _quat_to_rpy(w, x, y, z):
    """Quaternion → (roll, pitch, yaw) in radians — intrinsic xyz."""
    roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp = 2*(w*y - z*x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def _rpy_to_quat(roll, pitch, yaw):
    """(roll, pitch, yaw) in radians → quaternion (w, x, y, z)."""
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return w, x, y, z


class FakeFanService(fan_pb2_grpc.FanControllerServiceServicer):
    def __init__(self):
        self._fans = {uid: {"name": n, "uid": uid, "on": False} for n, uid in _FAN_DEFS}

    def GetAllFansId(self, request, context):
        names = [f["name"] for f in self._fans.values()]
        uids = [f["uid"] for f in self._fans.values()]
        return fan_pb2.FansId(names=names, uids=uids)

    def GetFansState(self, request, context):
        fans = [self._fans[fid.uid] for fid in request.ids if fid.uid in self._fans]
        ids = [fan_pb2.FanId(uid=f["uid"]) for f in fans]
        states = [fan_pb2.FanState(on=f["on"]) for f in fans]
        return fan_pb2.FansState(ids=ids, states=states)

    def SendFansCommands(self, request, context):
        for fc in request.commands:
            if fc.id.uid in self._fans:
                self._fans[fc.id.uid]["on"] = fc.state.on
        return Empty()


_STATE_FILE = "/tmp/reachy_joints.json"


def _state_writer(joints: Dict[int, "JointState"]):
    """Write joint positions to a JSON file at 30 Hz for the ROS bridge to read."""
    dt = 1.0 / 30.0
    while True:
        state = {j.name: j.present_position for j in joints.values()}
        try:
            with open(_STATE_FILE, "w") as f:
                json.dump(state, f)
        except Exception:
            pass
        time.sleep(dt)


def _interpolation_loop(joints: Dict[int, "JointState"]):
    """Background servo sim: move present_position toward goal_position at 50 Hz."""
    dt = 1.0 / 50.0
    while True:
        for j in joints.values():
            if not j.compliant:
                diff = j.goal_position - j.present_position
                j.present_position += diff * min(dt * 5.0, 1.0)
        time.sleep(dt)


def serve():
    joints = {uid: JointState(name, uid) for name, uid in _JOINT_DEFS}

    threading.Thread(target=_state_writer, args=(joints,), daemon=True).start()
    threading.Thread(target=_interpolation_loop, args=(joints,), daemon=True).start()

    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    joint_pb2_grpc.add_JointServiceServicer_to_server(FakeJointService(joints), server)
    sensor_pb2_grpc.add_SensorServiceServicer_to_server(FakeSensorService(), server)
    fan_pb2_grpc.add_FanControllerServiceServicer_to_server(FakeFanService(), server)
    head_kinematics_pb2_grpc.add_HeadKinematicsServicer_to_server(FakeHeadKinematicsService(), server)
    arm_kinematics_pb2_grpc.add_ArmKinematicsServicer_to_server(FakeArmKinematicsService(), server)

    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    log.info(f"Fake Reachy gRPC server listening on port {PORT}")
    log.info(f"Joints: {[j.name for j in joints.values()]}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
