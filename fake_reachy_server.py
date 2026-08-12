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

from kinematic_backend import (
    JOINT_DEFS,
    JointCommand,
    JointSample,
    KinematicBackend,
    SimulationSnapshot,
    state_file_writer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [fake-reachy] %(message)s")
log = logging.getLogger(__name__)

PORT = 50051

import time

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

    def __init__(self, backend: KinematicBackend) -> None:
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
    def __init__(self) -> None:
        self._sensors = {
            uid: {"name": n, "uid": uid, "force": 0.0}
            for n, uid in _FORCE_SENSOR_DEFS
        }

    def GetAllForceSensorsId(self, request, context):
        names = [s["name"] for s in self._sensors.values()]
        uids  = [s["uid"]  for s in self._sensors.values()]
        return sensor_pb2.SensorsId(names=names, uids=uids)

    def GetSensorsState(self, request, context):
        sensors = [self._sensors[sid.uid] for sid in request.ids if sid.uid in self._sensors]
        ids = [sensor_pb2.SensorId(uid=s["uid"]) for s in sensors]
        states = [
            sensor_pb2.SensorState(
                force_sensor_state=sensor_pb2.ForceSensorState(force=s["force"])
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
                    force_sensor_state=sensor_pb2.ForceSensorState(force=s["force"])
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


# ── Server entrypoint ─────────────────────────────────────────────────────────

def serve() -> None:
    backend = KinematicBackend()
    backend.start()

    threading.Thread(
        target=state_file_writer, args=(backend,), daemon=True, name="state-writer"
    ).start()

    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    joint_pb2_grpc.add_JointServiceServicer_to_server(FakeJointService(backend), server)
    sensor_pb2_grpc.add_SensorServiceServicer_to_server(FakeSensorService(), server)
    fan_pb2_grpc.add_FanControllerServiceServicer_to_server(FakeFanService(), server)
    head_kinematics_pb2_grpc.add_HeadKinematicsServicer_to_server(
        FakeHeadKinematicsService(), server
    )
    arm_kinematics_pb2_grpc.add_ArmKinematicsServicer_to_server(
        FakeArmKinematicsService(), server
    )

    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    snap = backend.latest_snapshot()
    log.info("Fake Reachy gRPC server listening on port %d", PORT)
    log.info("Joints: %s", list(snap.joints.keys()))
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
