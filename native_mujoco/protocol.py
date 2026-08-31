"""
R12-400: Versioned WebSocket protocol for the native MuJoCo server.

All messages are JSON objects with a "type" field.  The server binds on
127.0.0.1:8765 by default.  The Docker compatibility core connects as a
WebSocket client via ws://host.docker.internal:8765.

Protocol version: PROTOCOL_VERSION = 1

Handshake sequence
------------------
1. Client sends   hello
2. Server replies hello_ack  (or error if version mismatch)
3. Server begins  pushing state at ~sim_fps
4. Client may send joint_command / reset / pause at any time
5. Either side sends heartbeat; other must reply with heartbeat_ack
6. Server sends shutdown on clean exit; client sends disconnect

Message catalogue
-----------------
Client → Server:
  hello           version negotiation
  joint_command   set target joint positions (radians)
  scene_load      load a scene document
  reset           reset simulation state
  pause           pause or resume stepping
  heartbeat       liveness check
  heartbeat_ack   reply to server's heartbeat
  disconnect      clean shutdown request

Server → Client:
  hello_ack       confirm connection and capabilities
  state           snapshot: joints + object poses + metadata
  camera_frame    one JPEG-encoded camera image
  scene_ack       result of scene_load
  reset_ack       result of reset
  heartbeat       liveness check
  heartbeat_ack   reply to client's heartbeat
  error           non-fatal error report
  shutdown        server is stopping
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

PROTOCOL_VERSION = 1
_MAX_IMAGE_BYTES = 4 * 1024 * 1024   # 4 MiB sanity limit per camera frame
_MAX_SCENE_BYTES = 256 * 1024        # 256 KiB scene payload limit


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _now_ns() -> int:
    return time.monotonic_ns()


def encode(msg: Dict[str, Any]) -> str:
    return json.dumps(msg, separators=(",", ":"))


def decode(raw: str) -> Dict[str, Any]:
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Client → Server messages
# ---------------------------------------------------------------------------

@dataclass
class Hello:
    type: str = field(default="hello", init=False)
    protocol_version: int = PROTOCOL_VERSION
    client_id: str = ""

    def encode(self) -> str:
        return encode(asdict(self))


@dataclass
class JointCommand:
    """Full 21-element lists in MJCF joint order (from joint_map).

    Only target_rad is required.  The optional per-joint lists carry the v1 SDK
    actuator/compliance semantics (R12-501); each element may be None to leave
    that joint's current setting unchanged.
    """
    type: str = field(default="joint_command", init=False)
    seq: int = 0
    target_rad: List[float] = field(default_factory=list)
    mask: Optional[List[bool]] = None  # None = apply all; True element = apply
    compliant: Optional[List[Optional[bool]]] = None
    speed_limit_rad_s: Optional[List[Optional[float]]] = None
    torque_limit_percent: Optional[List[Optional[float]]] = None

    def encode(self) -> str:
        return encode(asdict(self))


@dataclass
class SceneLoad:
    type: str = field(default="scene_load", init=False)
    scene_document: Dict[str, Any] = field(default_factory=dict)
    request_id: str = ""

    def encode(self) -> str:
        d = asdict(self)
        payload = encode(d)
        if len(payload) > _MAX_SCENE_BYTES:
            raise ValueError(
                f"Scene payload {len(payload)} B exceeds {_MAX_SCENE_BYTES} B limit"
            )
        return payload


@dataclass
class Reset:
    type: str = field(default="reset", init=False)
    seed: Optional[int] = None
    request_id: str = ""

    def encode(self) -> str:
        return encode(asdict(self))


@dataclass
class Pause:
    type: str = field(default="pause", init=False)
    paused: bool = True

    def encode(self) -> str:
        return encode(asdict(self))


@dataclass
class ZoomCommand:
    """Set the motorised zoom level on one or both cameras (R12-605).

    level is an SDK ZoomLevelPossibilities name: in / out / inter / zero.
    camera is "left_camera", "right_camera", or "" for both.  The physical
    lens moves both eyes independently, but a stereo pair at mismatched zoom
    has no usable disparity, so "" is the normal case.
    """
    type: str = field(default="zoom_command", init=False)
    level: str = "inter"
    camera: str = ""

    def encode(self) -> str:
        return encode(asdict(self))


@dataclass
class Heartbeat:
    type: str = field(default="heartbeat", init=False)
    sent_ns: int = field(default_factory=_now_ns)

    def encode(self) -> str:
        return encode(asdict(self))


@dataclass
class HeartbeatAck:
    type: str = field(default="heartbeat_ack", init=False)
    echo_ns: int = 0   # echoes sent_ns from the peer's heartbeat

    def encode(self) -> str:
        return encode(asdict(self))


@dataclass
class Disconnect:
    type: str = field(default="disconnect", init=False)
    reason: str = ""

    def encode(self) -> str:
        return encode(asdict(self))


# ---------------------------------------------------------------------------
# Server → Client messages
# ---------------------------------------------------------------------------

@dataclass
class HelloAck:
    type: str = field(default="hello_ack", init=False)
    protocol_version: int = PROTOCOL_VERSION
    server_version: str = "0.1.0"
    sim_fps: float = 500.0
    camera_fps: float = 15.0
    num_joints: int = 21
    camera_names: List[str] = field(default_factory=lambda: ["left_camera", "right_camera"])
    capabilities: Dict[str, Any] = field(default_factory=dict)

    def encode(self) -> str:
        return encode(asdict(self))


@dataclass
class JointSample:
    position_rad: float
    velocity_rad_s: float = 0.0
    effort: float = 0.0


@dataclass
class ObjectPose:
    object_id: str
    pos_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class State:
    """Authoritative simulation snapshot pushed to clients."""
    type: str = field(default="state", init=False)
    seq: int = 0
    sim_step: int = 0
    sim_time_s: float = 0.0
    wall_time_ns: int = field(default_factory=_now_ns)
    cmd_seq: int = 0         # last applied joint_command seq
    scene_revision: str = ""
    paused: bool = False
    # joints: list parallel to JOINT_TABLE order (21 entries)
    joints: List[Dict[str, Any]] = field(default_factory=list)
    objects: List[Dict[str, Any]] = field(default_factory=list)
    # R12-502: per-gripper grasp/force state and SDK force-sensor readings.
    grippers: List[Dict[str, Any]] = field(default_factory=list)
    force_sensors: List[Dict[str, Any]] = field(default_factory=list)
    # R12-504: interactive control on/off states ({id, type, on, value}).
    interactive: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def encode(self) -> str:
        return encode(asdict(self))


@dataclass
class CameraFrame:
    """One rendered camera frame.

    jpeg_b64 is always present (base64-encoded JPEG).
    depth_b64 and seg_b64 are empty strings when not enabled (R12-601).
    """
    type: str = field(default="camera_frame", init=False)
    camera: str = ""
    seq: int = 0
    sim_step: int = 0
    sim_time_s: float = 0.0
    wall_time_ns: int = field(default_factory=_now_ns)
    scene_revision: str = ""
    width: int = 640
    height: int = 480
    jpeg_b64: str = ""   # base64(JPEG bytes)
    render_us: int = 0   # render latency in microseconds
    # R12-601: optional research products ("" = not enabled for this run)
    depth_b64: str = ""  # base64(float16 H×W raw bytes), values in metres
    seg_b64: str = ""    # base64(uint16 H×W raw bytes), values are MuJoCo body IDs

    def encode(self) -> str:
        return encode(asdict(self))


@dataclass
class SceneAck:
    type: str = field(default="scene_ack", init=False)
    request_id: str = ""
    accepted: bool = False
    scene_revision: str = ""
    warnings: List[str] = field(default_factory=list)
    error: str = ""

    def encode(self) -> str:
        return encode(asdict(self))


@dataclass
class ResetAck:
    type: str = field(default="reset_ack", init=False)
    request_id: str = ""
    sim_step: int = 0
    scene_revision: str = ""

    def encode(self) -> str:
        return encode(asdict(self))


@dataclass
class Error:
    type: str = field(default="error", init=False)
    code: str = ""
    message: str = ""

    def encode(self) -> str:
        return encode(asdict(self))


@dataclass
class Shutdown:
    type: str = field(default="shutdown", init=False)
    reason: str = ""

    def encode(self) -> str:
        return encode(asdict(self))


# ---------------------------------------------------------------------------
# Dispatch helper
# ---------------------------------------------------------------------------

_CLIENT_TYPES = {
    "hello", "joint_command", "scene_load",
    "reset", "pause", "heartbeat", "heartbeat_ack", "disconnect",
    "zoom_command",
}
_SERVER_TYPES = {
    "hello_ack", "state", "camera_frame", "scene_ack",
    "reset_ack", "heartbeat", "heartbeat_ack", "error", "shutdown",
}


def message_type(raw: str) -> str:
    """Fast path: parse only the type field."""
    msg = json.loads(raw)
    return msg.get("type", "")


def validate_joint_command(msg: Mapping[str, Any], num_joints: int = 21) -> None:
    """Raise ValueError if a decoded joint_command is structurally invalid."""
    tgt = msg.get("target_rad")
    if not isinstance(tgt, list):
        raise ValueError("joint_command.target_rad must be a list")
    if len(tgt) != num_joints:
        raise ValueError(
            f"joint_command.target_rad length {len(tgt)} != {num_joints}"
        )
    mask = msg.get("mask")
    if mask is not None:
        if not isinstance(mask, list) or len(mask) != num_joints:
            raise ValueError(
                f"joint_command.mask must be null or a bool list of length {num_joints}"
            )


def validate_image_payload(b64_str: str) -> None:
    """Raise ValueError if a base64 image payload is too large."""
    estimated = len(b64_str) * 3 // 4
    if estimated > _MAX_IMAGE_BYTES:
        raise ValueError(
            f"Camera frame payload ~{estimated // 1024} KiB exceeds "
            f"{_MAX_IMAGE_BYTES // 1024} KiB limit"
        )
