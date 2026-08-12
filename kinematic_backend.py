"""Pure-Python kinematic simulation backend — no gRPC dependency.

Domain models (R12-100) and KinematicBackend (R12-101, R12-102) live here so
they can be imported and unit-tested on any host without installing grpcio or
reachy-sdk-api.  The gRPC adapter in fake_reachy_server.py imports from here.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

log = logging.getLogger(__name__)

_STEP_HZ = 50.0        # authoritative simulation rate (Hz)
_CMD_QUEUE_MAX = 64    # bounded command queue depth
_STATE_FILE = "/tmp/reachy_joints.json"

# Reachy 1.2: right arm (IDs 10-17), left arm (IDs 20-27), head (IDs 30-34)
JOINT_DEFS = [
    ("r_shoulder_pitch", 10),
    ("r_shoulder_roll",  11),
    ("r_arm_yaw",        12),
    ("r_elbow_pitch",    13),
    ("r_forearm_yaw",    14),
    ("r_wrist_pitch",    15),
    ("r_wrist_roll",     16),
    ("r_gripper",        17),
    ("l_shoulder_pitch", 20),
    ("l_shoulder_roll",  21),
    ("l_arm_yaw",        22),
    ("l_elbow_pitch",    23),
    ("l_forearm_yaw",    24),
    ("l_wrist_pitch",    25),
    ("l_wrist_roll",     26),
    ("l_gripper",        27),
    ("neck_roll",        30),
    ("neck_pitch",       31),
    ("neck_yaw",         32),
    ("l_antenna",        33),
    ("r_antenna",        34),
]


# ── Domain models (R12-100) ───────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class JointCommand:
    """Command submitted to the backend for one joint."""
    uid: int
    goal_position: Optional[float] = None   # radians; None = no change
    compliant: Optional[bool] = None
    speed_limit: Optional[float] = None
    torque_limit: Optional[float] = None


@dataclass(frozen=True, slots=True)
class JointSample:
    """Immutable snapshot of one joint at a single simulation step."""
    name: str
    uid: int
    present_position: float   # radians
    goal_position: float      # radians
    present_speed: float
    present_load: float
    temperature: float
    compliant: bool
    speed_limit: float
    torque_limit: float


@dataclass(frozen=True)
class SimulationSnapshot:
    """Immutable, coherent view of all joints from one simulation step."""
    sequence: int
    sim_step: int
    sim_time_s: float
    wall_time_ns: int
    joints: Mapping[str, JointSample]   # name → JointSample


# ── KinematicBackend (R12-101, R12-102) ──────────────────────────────────────

class _MutableJoint:
    """Internal mutable joint state — only written inside the backend loop."""
    __slots__ = ("name", "uid", "present_position", "goal_position",
                 "present_speed", "present_load", "temperature",
                 "compliant", "speed_limit", "torque_limit")

    def __init__(self, name: str, uid: int) -> None:
        self.name = name
        self.uid = uid
        self.present_position: float = 0.0
        self.goal_position: float = 0.0
        self.present_speed: float = 0.0
        self.present_load: float = 0.0
        self.temperature: float = 35.0
        # Antenna joints (UIDs 33-34) are always stiff on physical Reachy.
        self.compliant: bool = uid not in (33, 34)
        self.speed_limit: float = 0.0
        self.torque_limit: float = 100.0

    def sample(self) -> JointSample:
        return JointSample(
            name=self.name, uid=self.uid,
            present_position=self.present_position,
            goal_position=self.goal_position,
            present_speed=self.present_speed,
            present_load=self.present_load,
            temperature=self.temperature,
            compliant=self.compliant,
            speed_limit=self.speed_limit,
            torque_limit=self.torque_limit,
        )


class KinematicBackend:
    """Authoritative kinematic simulation backend.

    Concurrency contract:
    - Exactly one thread (started by start()) writes joint state.
    - All other threads read via latest_snapshot() — an atomic reference.
    - Commands enter via a bounded queue; the loop drains them at each step.
    - Snapshot reference assignment is GIL-safe in CPython.
    """

    def __init__(self) -> None:
        self._joints: Dict[int, _MutableJoint] = {
            uid: _MutableJoint(name, uid) for name, uid in JOINT_DEFS
        }
        self._uid_by_name: Dict[str, int] = {
            j.name: j.uid for j in self._joints.values()
        }
        self._cmd_queue: queue.Queue[JointCommand] = queue.Queue(maxsize=_CMD_QUEUE_MAX)
        self._sequence: int = 0
        self._sim_step: int = 0
        self._sim_time_s: float = 0.0
        self._snapshot: SimulationSnapshot = self._build_snapshot()

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self._step_loop, daemon=True, name="kinematic-loop")
        t.start()
        return t

    def submit_command(self, cmd: JointCommand) -> None:
        """Non-blocking. Logs and drops on queue overflow."""
        try:
            self._cmd_queue.put_nowait(cmd)
        except queue.Full:
            log.warning("Command queue full — dropping command for uid=%d", cmd.uid)

    def uid_by_name(self, name: str) -> Optional[int]:
        return self._uid_by_name.get(name)

    def latest_snapshot(self) -> SimulationSnapshot:
        """Return the latest immutable snapshot. Safe to call from any thread."""
        return self._snapshot

    def _drain_commands(self) -> None:
        while True:
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                break
            j = self._joints.get(cmd.uid)
            if j is None:
                continue
            if cmd.goal_position is not None:
                j.goal_position = cmd.goal_position
            if cmd.compliant is not None:
                j.compliant = cmd.compliant
            if cmd.speed_limit is not None:
                j.speed_limit = cmd.speed_limit
            if cmd.torque_limit is not None:
                j.torque_limit = cmd.torque_limit

    def _advance(self, dt: float) -> None:
        """First-order servo model: move present_position toward goal_position."""
        for j in self._joints.values():
            if not j.compliant:
                diff = j.goal_position - j.present_position
                j.present_position += diff * min(dt * 5.0, 1.0)

    def _build_snapshot(self) -> SimulationSnapshot:
        return SimulationSnapshot(
            sequence=self._sequence,
            sim_step=self._sim_step,
            sim_time_s=self._sim_time_s,
            wall_time_ns=time.monotonic_ns(),
            joints={j.name: j.sample() for j in self._joints.values()},
        )

    def _step_loop(self) -> None:
        dt = 1.0 / _STEP_HZ
        while True:
            t0 = time.monotonic()
            self._drain_commands()
            self._advance(dt)
            self._sim_step += 1
            self._sim_time_s += dt
            self._sequence += 1
            self._snapshot = self._build_snapshot()
            elapsed = time.monotonic() - t0
            sleep_for = max(0.0, dt - elapsed)
            if sleep_for > 0.0:
                time.sleep(sleep_for)


# ── State file writer (R12-103) ───────────────────────────────────────────────

def state_file_writer(backend: KinematicBackend, path: Optional[str] = None) -> None:
    """Write joint positions atomically to JSON for the ROS bridge at 30 Hz.

    Uses write-then-rename so the bridge never reads a partial file.
    Skips writes when the snapshot sequence has not advanced.
    `path` overrides _STATE_FILE — useful for tests to avoid module-level patching.
    """
    dest = path if path is not None else _STATE_FILE
    tmp = dest + ".tmp"
    dt = 1.0 / 30.0
    last_seq = -1
    while True:
        snap = backend.latest_snapshot()
        if snap.sequence != last_seq:
            state = {name: s.present_position for name, s in snap.joints.items()}
            try:
                with open(tmp, "w") as f:
                    json.dump(state, f)
                os.replace(tmp, dest)   # POSIX atomic rename
            except Exception as exc:
                log.warning("State file write failed: %s", exc)
            last_seq = snap.sequence
        time.sleep(dt)
