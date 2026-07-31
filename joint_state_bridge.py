"""ROS 2 node: reads joint positions from fake_reachy_server and publishes /joint_states.

Runs as its own process so rclpy is on the main thread with no threading conflicts.
fake_reachy_server writes /tmp/reachy_joints.json at 30 Hz; this node reads and
publishes at the same rate so RViz animates in response to SDK commands.
"""

import json
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

STATE_FILE = "/tmp/reachy_joints.json"
RATE_HZ = 30.0


class JointStateBridge(Node):
    def __init__(self):
        super().__init__("joint_state_bridge")
        self._pub = self.create_publisher(JointState, "joint_states", 10)
        self.create_timer(1.0 / RATE_HZ, self._publish)
        self.get_logger().info(f"Joint state bridge started, reading {STATE_FILE}")

    def _publish(self):
        try:
            with open(STATE_FILE) as f:
                state: dict = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        for name, position in state.items():
            msg.name.append(name)
            msg.position.append(position)
        self._pub.publish(msg)


def main():
    rclpy.init()
    node = JointStateBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
