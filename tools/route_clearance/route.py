"""Reading a tabletop route document (arm_routes.v1) as a wire contract, and expanding a route into the legs that are
flown: its own legs out, then the same legs in reverse back to the start (`return: reverse`, design section 3.1).

The tabletop's loader (`motion/arm_routes.py`) is the strict validator. This module checks only what the clearance
run relies on, and leaves joint-limit judgement to the verdict, so an out-of-limit pose is reported as a failing
record rather than a crash.
"""
import json
import math
from dataclasses import dataclass

from .arm_route_hash import route_hash
from .limits import ARM_LIMITS_DEG

SCHEMA = "arm_routes.v1"
ARM7 = ("r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw", "r_elbow_pitch", "r_forearm_yaw", "r_wrist_pitch",
        "r_wrist_roll")
GRIPPER = "r_gripper"

#: Design section 4.2 / OD-A1: a leg lasts max(min_duration_s, largest arm-joint change / this).
MAX_AVG_SPEED_DEG_S = 10.0


class RouteError(ValueError):
    """The route document cannot be flown; the message names where."""


@dataclass(frozen=True)
class Leg:
    index: int                  # position in the flown sequence, 0-based
    direction: str              # "out" or "return"
    start: str                  # pose names
    to: str
    q_from: dict                # degrees, the seven arm joints plus r_gripper
    q_to: dict
    duration_s: float
    near_board: bool


def load_doc(path):
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise RouteError("document: schema must be %r" % SCHEMA)
    return doc


def _pose(doc, name, gripper_deg):
    try:
        joints = doc["poses"][name]["joints_deg"]
    except KeyError:
        raise RouteError("poses.%s: missing" % name)
    if sorted(joints) != sorted(ARM7):
        raise RouteError("poses.%s.joints_deg: must name exactly %s" % (name, list(ARM7)))
    q = {joint: float(joints[joint]) for joint in ARM7}
    q[GRIPPER] = float(gripper_deg)
    return q


def leg_duration(q_from, q_to, min_duration_s):
    largest = max(abs(q_to[j] - q_from[j]) for j in ARM7)
    return round(max(float(min_duration_s), largest / MAX_AVG_SPEED_DEG_S), 2)


def expand(doc, route_name):
    """The flown legs. A pose's gripper is the one its arriving leg commands; the start pose keeps the first leg's.
    A return leg retraces an outbound leg backwards, with that leg's duration and near_board flag."""
    try:
        route = doc["routes"][route_name]
    except KeyError:
        raise RouteError("routes.%s: no such route" % route_name)
    if route.get("return") != "reverse":
        raise RouteError("routes.%s.return: only 'reverse' is flown" % route_name)
    legs = route.get("legs")
    if not isinstance(legs, list) or not legs:
        raise RouteError("routes.%s.legs: must be a non-empty list" % route_name)
    names = [route["start"]] + [leg["to"] for leg in legs]
    grips = [legs[0]["gripper_deg"]] + [leg["gripper_deg"] for leg in legs]
    poses = [_pose(doc, n, g) for n, g in zip(names, grips)]
    out = []
    for i, leg in enumerate(legs):
        out.append(dict(direction="out", start=names[i], to=names[i + 1], q_from=poses[i], q_to=poses[i + 1],
                        duration_s=leg_duration(poses[i], poses[i + 1], leg["min_duration_s"]),
                        near_board=bool(leg.get("near_board", False))))
    for back in reversed(out[:]):
        out.append(dict(back, direction="return", start=back["to"], to=back["start"], q_from=back["q_to"],
                        q_to=back["q_from"]))
    return [Leg(index=i, **leg) for i, leg in enumerate(out)]


def waypoints(legs):
    """(name, q) for each distinct pose of the route, in flown order."""
    seen = {}
    for leg in legs:
        for name, q in ((leg.start, leg.q_from), (leg.to, leg.q_to)):
            seen.setdefault(name, q)
    return list(seen.items())


def near_board_poses(legs):
    return sorted({leg.to if leg.direction == "out" else leg.start for leg in legs if leg.near_board})


def limit_violations(name, q, limits=ARM_LIMITS_DEG):
    """Reasons a waypoint is outside the gate's limits (design section 5 item 3), one per joint."""
    bad = []
    for joint in ARM7 + (GRIPPER,):
        lo, hi = limits[joint]
        if not (math.isfinite(q[joint]) and lo <= q[joint] <= hi):
            bad.append("waypoint %s: %s %+.2f deg is outside the limits [%s, %s]" % (name, joint, q[joint], lo, hi))
    return bad


def hash_of(doc, route_name, limits=ARM_LIMITS_DEG):
    return route_hash(route_name, doc, limits)
