#!/usr/bin/env python3
"""Run INSIDE the container, stack up: does the RViz marker topic carry every PART
body with its compound parts?

    python3 check_rviz_markers.py <scene.yaml> [--override]

Subscribes to /reachy_sim/scene/markers (transient local), takes one fresh
MarkerArray, and compares it with markers worked out HERE, by plain arithmetic
from the resolved scene (scene_io), not by calling the publisher's own
primitive_views: one marker per primitive of each PART object, at object position
+ part offset (every PART is parked with identity orientation), with the part's
quaternion, size and the object's colour.  Also checks the total count and unique
ids.  With --override it moves one holed part through the live pose-override file
and checks that all its markers move together, then restores the file.
"""
import json
import os
import sys
import time

sys.path[:0] = ["/opt"]
import scene_io                                                     # noqa: E402

import rclpy                                                        # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from visualization_msgs.msg import Marker, MarkerArray              # noqa: E402

TOPIC = "/reachy_sim/scene/markers"
OVERRIDES = os.environ.get("REACHY_SIM_SCENE_OVERRIDES", "/tmp/reachy_scene_overrides.json")
TOL = 1e-6


def primitives(obj):
    """(marker type, scale, position, quat) for each primitive of a parked object."""
    geo = obj["geometry"]
    pos = obj["pose"]["position"]
    assert obj["pose"]["orientation_wxyz"] == [1.0, 0.0, 0.0, 0.0], obj["id"]
    parts = geo["parts"] if geo["kind"] == "compound" else [
        dict(geo, pos=[0, 0, 0], quat_wxyz=[1, 0, 0, 0])]
    for p in parts:
        if p["kind"] == "box":
            kind, scale = Marker.CUBE, list(p["size"])
        else:
            kind, scale = Marker.CYLINDER, [2 * p["radius"], 2 * p["radius"], p["length"]]
        yield kind, scale, [a + b for a, b in zip(pos, p["pos"])], list(p["quat_wxyz"])


def matches(m, kind, scale, pos, quat, rgba):
    got = ([m.scale.x, m.scale.y, m.scale.z],
           [m.pose.position.x, m.pose.position.y, m.pose.position.z],
           [m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z],
           [m.color.r, m.color.g, m.color.b, m.color.a])
    return (m.type == kind and m.action == Marker.ADD and m.ns == "scene"
            and m.header.frame_id == "pedestal"
            and all(abs(a - b) <= TOL for a, b in zip(got[0] + got[1] + got[2], scale + pos + quat))
            and all(abs(a - b) <= 1e-5 for a, b in zip(got[3], rgba)))     # float32 colour


def receive(node, wait_s=20.0, settle_s=1.0):
    """The latest MarkerArray published after `settle_s` of listening."""
    got = []
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)
    sub = node.create_subscription(MarkerArray, TOPIC, got.append, qos)
    start = time.monotonic()
    while time.monotonic() - start < wait_s and (not got or time.monotonic() - start < settle_s):
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_subscription(sub)
    assert got, "nothing on " + TOPIC
    return got[-1].markers


def main():
    doc = scene_io.load_scene(sys.argv[1])
    objects = doc["objects"]
    parts = [o for o in objects if o["id"].startswith("part")]
    expected_total = sum(len(o["geometry"]["parts"]) if o["geometry"]["kind"] == "compound" else 1
                         for o in objects)
    rclpy.init()
    node = rclpy.create_node("check_rviz_markers")
    failures = []
    try:
        markers = receive(node)
        ids = [m.id for m in markers]
        print("scene %s: %d markers on %s (expected %d), ids unique: %s"
              % (doc["name"], len(markers), TOPIC, expected_total, len(set(ids)) == len(ids)))
        if len(markers) != expected_total or len(set(ids)) != len(ids):
            failures.append("marker count or ids")
        used = set()
        for obj in parts:
            want = list(primitives(obj))
            found = 0
            for kind, scale, pos, quat in want:
                hit = next((m for m in markers if m.id not in used and
                            matches(m, kind, scale, pos, quat, obj["material"]["rgba"])), None)
                if hit is not None:
                    used.add(hit.id)
                    found += 1
            colour = tuple(round(v, 3) for v in obj["material"]["rgba"][:3])
            print("  %-10s %-8s primitives %2d/%2d  rgb %s" %
                  (obj["id"], obj["geometry"]["kind"], found, len(want), colour))
            if found != len(want):
                failures.append(obj["id"])
        total_prims = sum(len(list(primitives(o))) for o in parts)
        print("PART bodies: %d; PART primitives matched: %d/%d" % (len(parts), len(used), total_prims))

        if "--override" in sys.argv:
            obj = next(o for o in parts if o["id"] == "part121_a")
            moved = [0.432, 0.0, 0.774925]
            had = os.path.exists(OVERRIDES)
            before = open(OVERRIDES).read() if had else None
            try:
                with open(OVERRIDES, "w") as f:
                    json.dump({obj["id"]: moved}, f)
                time.sleep(0.5)
                markers = receive(node, settle_s=1.5)
                shifted = dict(obj, pose=dict(obj["pose"], position=moved))
                n = sum(any(matches(m, *p, obj["material"]["rgba"]) for m in markers)
                        for p in primitives(shifted))
                print("override of %s to %s: %d/%d markers moved together" %
                      (obj["id"], moved, n, len(obj["geometry"]["parts"])))
                if n != len(obj["geometry"]["parts"]):
                    failures.append("override")
            finally:
                if had:
                    with open(OVERRIDES, "w") as f:
                        f.write(before)
                else:
                    os.remove(OVERRIDES)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print("RESULT:", "PASS" if not failures else "FAIL %s" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
