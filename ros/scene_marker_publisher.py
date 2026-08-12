"""ROS 2 node: publishes scene objects as a transient-local MarkerArray.

Usage inside the container:
    python3 /opt/ros_nodes/scene_marker_publisher.py \
        --scene /opt/scenes/tabletop.example.yaml

Publishes to /reachy_sim/scene/markers with TRANSIENT_LOCAL durability so
late-joining RViz instances receive the full marker set without waiting for
the next publish cycle.

Each scene object maps to one Marker.  Geometry kind → RViz marker type:
    box       → CUBE
    sphere    → SPHERE
    cylinder  → CYLINDER
    capsule   → CYLINDER  (approximation; height = total capsule length)
    plane     → CUBE      (flat slab, scale z = 0.01 m)
    mesh      → MESH_RESOURCE

Poses use the scene's orientation_wxyz [w, x, y, z] convention.
The RViz marker quaternion field order is also w, x, y, z in the
geometry_msgs/Quaternion message, so no reordering is needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow importing scene_loader from the repo root whether running inside or
# outside the container.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scene_loader import (  # noqa: E402
    SceneDocument,
    SceneGeometry,
    SceneObject,
    load_scene,
)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
    from std_msgs.msg import ColorRGBA, Header
    from visualization_msgs.msg import Marker, MarkerArray

    _ROS_AVAILABLE = True
except ImportError:  # pragma: no cover — only missing outside the container
    _ROS_AVAILABLE = False


_PUBLISH_HZ = 15.0   # fast enough for smooth object-follows-gripper animation
_NAMESPACE = "scene"
_DEFAULT_OVERRIDES = "/tmp/reachy_scene_overrides.json"


def _read_overrides(path: str) -> dict:
    """Read {object_id: [x, y, z]} pose overrides written by the demo.

    Used to animate a grasped object's marker so it follows the gripper in the
    kinematic/RViz view (the kinematic backend has no grasp physics).  Returns
    an empty dict if the file is absent or malformed.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _marker_type(kind: str) -> int:
    if not _ROS_AVAILABLE:
        return 0
    mapping: dict[str, int] = {
        "box": Marker.CUBE,
        "sphere": Marker.SPHERE,
        "cylinder": Marker.CYLINDER,
        "capsule": Marker.CYLINDER,
        "plane": Marker.CUBE,
        "mesh": Marker.MESH_RESOURCE,
    }
    return mapping.get(kind, Marker.CUBE)


def _marker_scale(geo: SceneGeometry) -> tuple[float, float, float]:
    kind = geo.kind
    if kind == "box":
        sx, sy, sz = geo.size or (0.1, 0.1, 0.1)
        return float(sx), float(sy), float(sz)
    if kind == "sphere":
        r = float(geo.radius or 0.05)
        return r * 2, r * 2, r * 2
    if kind == "cylinder":
        r = float(geo.radius or 0.05)
        h = float(geo.length or 0.1)
        return r * 2, r * 2, h
    if kind == "capsule":
        r = float(geo.radius or 0.05)
        l = float(geo.length or 0.1)
        return r * 2, r * 2, l + 2 * r
    if kind == "plane":
        sx, sy, _ = geo.size or (1.0, 1.0, 0.0)
        return float(sx), float(sy), 0.01
    if kind == "mesh":
        sx, sy, sz = geo.scale or (1.0, 1.0, 1.0)
        return float(sx), float(sy), float(sz)
    return 0.1, 0.1, 0.1


def _build_marker(
    obj: SceneObject, marker_id: int, frame_id: str, stamp: object
) -> "Marker":
    m = Marker()
    m.header = Header()
    m.header.frame_id = frame_id
    m.header.stamp = stamp

    m.ns = _NAMESPACE
    m.id = marker_id
    m.action = Marker.ADD
    m.type = _marker_type(obj.geometry.kind)

    pos = obj.pose.position
    m.pose = Pose()
    m.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
    q = obj.pose.orientation_wxyz
    m.pose.orientation = Quaternion(w=float(q[0]), x=float(q[1]), y=float(q[2]), z=float(q[3]))

    sx, sy, sz = _marker_scale(obj.geometry)
    m.scale = Vector3(x=sx, y=sy, z=sz)

    r, g, b, a = obj.material.rgba
    m.color = ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))

    if obj.geometry.kind == "mesh" and obj.geometry.path:
        m.mesh_resource = "file://" + obj.geometry.path
        m.mesh_use_embedded_materials = True

    m.frame_locked = False
    return m


class SceneMarkerPublisher(Node):  # type: ignore[misc]
    def __init__(self, scene_path: str, overrides_path: str = _DEFAULT_OVERRIDES) -> None:
        super().__init__("scene_marker_publisher")
        self._overrides_path = overrides_path

        transient_local_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._pub = self.create_publisher(
            MarkerArray,
            "/reachy_sim/scene/markers",
            transient_local_qos,
        )

        self._scene: SceneDocument = load_scene(scene_path)
        self.get_logger().info(
            f"Scene loaded: '{self._scene.name}', "
            f"{len(self._scene.objects)} objects, "
            f"frame_id='{self._scene.frame_id}'"
        )

        self._timer = self.create_timer(1.0 / _PUBLISH_HZ, self._publish)
        self._publish()

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        frame_id = self._scene.frame_id
        overrides = _read_overrides(self._overrides_path)
        markers = []
        for idx, obj in enumerate(self._scene.objects):
            m = _build_marker(obj, idx, frame_id, stamp)
            ov = overrides.get(obj.id)
            if ov and len(ov) >= 3:
                m.pose.position = Point(x=float(ov[0]), y=float(ov[1]), z=float(ov[2]))
            markers.append(m)
        msg = MarkerArray()
        msg.markers = markers
        self._pub.publish(msg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish scene objects as RViz markers"
    )
    parser.add_argument(
        "--scene",
        default=os.environ.get(
            "REACHY_SIM_SCENE", "/opt/scenes/tabletop.example.yaml"
        ),
        help="Path to scene YAML file",
    )
    parser.add_argument(
        "--overrides",
        default=os.environ.get("REACHY_SIM_SCENE_OVERRIDES", _DEFAULT_OVERRIDES),
        help="JSON file of {object_id: [x,y,z]} live pose overrides",
    )
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = SceneMarkerPublisher(args.scene, args.overrides)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
