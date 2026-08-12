"""ROS 2 node: publishes left/right camera frames as CompressedImage + CameraInfo.

R12-302 — reads JPEG frames from the atomic files written by frame_file_writer()
and re-publishes them as ROS messages.  This keeps the ROS node decoupled from
the gRPC server: both read from the same filesystem-level frame files, so they
always serve frames from a consistent source.

Topics published (per camera):
    /reachy_sim/{left,right}_camera/image_compressed  (sensor_msgs/CompressedImage)
    /reachy_sim/{left,right}_camera/camera_info        (sensor_msgs/CameraInfo)

QoS: SENSOR_DATA profile (BEST_EFFORT, VOLATILE) at the configured FPS.
     Late-joining subscribers see the next live frame, not a cached one.

Camera intrinsics are synthetic defaults (no physical calibration yet).
R12-600 will wire in a calibration-file loader.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSPresetProfiles
    from sensor_msgs.msg import CameraInfo, CompressedImage
    from std_msgs.msg import Header

    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False

_LEFT_FILE = "/tmp/reachy_left.jpg"
_RIGHT_FILE = "/tmp/reachy_right.jpg"

_WIDTH_DEFAULT = 640
_HEIGHT_DEFAULT = 480

# Synthetic focal length for a rough 60° HFOV at 640 px
_FX_DEFAULT = 554.26
_FY_DEFAULT = 554.26


def _synthetic_camera_info(frame_id: str, width: int, height: int) -> "CameraInfo":
    """Return a CameraInfo with synthetic pinhole defaults.

    Not calibrated to any physical camera.  Replace with R12-600 loader output.
    """
    cx = width / 2.0
    cy = height / 2.0
    fx = _FX_DEFAULT * (width / _WIDTH_DEFAULT)
    fy = _FY_DEFAULT * (height / _HEIGHT_DEFAULT)

    info = CameraInfo()
    info.header.frame_id = frame_id
    info.width = width
    info.height = height
    info.distortion_model = "plumb_bob"
    info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    info.k = [fx, 0.0, cx,
               0.0, fy, cy,
               0.0, 0.0, 1.0]
    info.r = [1.0, 0.0, 0.0,
               0.0, 1.0, 0.0,
               0.0, 0.0, 1.0]
    info.p = [fx, 0.0, cx, 0.0,
               0.0, fy, cy, 0.0,
               0.0, 0.0, 1.0, 0.0]
    return info


class CameraPublisher(Node):  # type: ignore[misc]
    """Reads JPEG frame files and publishes ROS camera messages at a configured FPS."""

    def __init__(self, fps: float, width: int, height: int) -> None:
        super().__init__("reachy_camera_publisher")

        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value

        self._left_pub = self.create_publisher(
            CompressedImage, "/reachy_sim/left_camera/image_compressed", sensor_qos
        )
        self._right_pub = self.create_publisher(
            CompressedImage, "/reachy_sim/right_camera/image_compressed", sensor_qos
        )
        self._left_info_pub = self.create_publisher(
            CameraInfo, "/reachy_sim/left_camera/camera_info", sensor_qos
        )
        self._right_info_pub = self.create_publisher(
            CameraInfo, "/reachy_sim/right_camera/camera_info", sensor_qos
        )

        self._left_info = _synthetic_camera_info("left_optical", width, height)
        self._right_info = _synthetic_camera_info("right_optical", width, height)

        self._drop_count = 0
        self._publish_count = 0
        self._last_left_mtime: float = 0.0
        self._last_right_mtime: float = 0.0

        self._timer = self.create_timer(1.0 / fps, self._publish)

    def _read_jpeg(self, path: str) -> bytes:
        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError:
            return b""

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()

        # Only republish when the file has changed (mtime check avoids redundant messages)
        try:
            left_mtime = os.stat(_LEFT_FILE).st_mtime
            right_mtime = os.stat(_RIGHT_FILE).st_mtime
        except OSError:
            self._drop_count += 1
            return

        if left_mtime == self._last_left_mtime and right_mtime == self._last_right_mtime:
            return  # no new frame yet; skip

        left_data = self._read_jpeg(_LEFT_FILE)
        right_data = self._read_jpeg(_RIGHT_FILE)

        if not left_data or not right_data:
            self._drop_count += 1
            return

        for data, pub, info_pub, info, frame_id in (
            (left_data,  self._left_pub,  self._left_info_pub,  self._left_info,  "left_optical"),
            (right_data, self._right_pub, self._right_info_pub, self._right_info, "right_optical"),
        ):
            img = CompressedImage()
            img.header = Header()
            img.header.stamp = stamp
            img.header.frame_id = frame_id
            img.format = "jpeg"
            img.data = list(data)   # CompressedImage.data is uint8[]
            pub.publish(img)

            info.header.stamp = stamp
            info_pub.publish(info)

        self._last_left_mtime = left_mtime
        self._last_right_mtime = right_mtime
        self._publish_count += 1

        if self._publish_count % 150 == 0:
            self.get_logger().info(
                f"Camera publisher: {self._publish_count} frames, "
                f"{self._drop_count} drops"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reachy camera ROS publisher")
    parser.add_argument("--fps", type=float, default=float(
        os.environ.get("REACHY_SIM_CAMERA_FPS", "15")
    ))
    parser.add_argument("--width", type=int, default=int(
        os.environ.get("REACHY_SIM_CAMERA_WIDTH", str(_WIDTH_DEFAULT))
    ))
    parser.add_argument("--height", type=int, default=int(
        os.environ.get("REACHY_SIM_CAMERA_HEIGHT", str(_HEIGHT_DEFAULT))
    ))
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = CameraPublisher(fps=args.fps, width=args.width, height=args.height)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
