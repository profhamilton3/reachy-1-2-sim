"""Goal and compliance reporting through the REAL client/bridge path, offline.

    reachy_sdk 0.7.0 (ReachySDK, Joint, trajectory.goto, rig_motion.fly_route)
      -> loopback gRPC -> fake_reachy_server.FakeJointService
      -> KinematicBridge(MujocoRemoteBackend) running its own websocket session
      -> tests/integration/native_stub.NativeStub (toy plant with sag, logs commands)

Nothing here is mocked between the SDK and the websocket.  No Docker, no native
MuJoCo server, no hardware, no motion.

Background: IITG-Reachy-Project/outputs/analysis-2026-09-23-goal-feedback-verification-report.md.
The fake server ignored `requested_fields`, and the bridge reported
goal_position = present.  reachy_sdk caches every field it is sent, starts goto
from its cached goal, and sends that cache at pop time -- so goto started from
the sagged pose, and setpoints were echoed back as the present pose.  The
check ids (C1, C5, C6, C7, C9) are that report's §4.

SDK-path tests skip if reachy_sdk / grpc / websockets are missing.  Set
REQUIRE_REACHY_SDK=1 to make them fail instead of skipping.
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, os.path.join(ROOT, "src"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

if os.environ.get("REQUIRE_REACHY_SDK") == "1":
    import grpc  # noqa: F401
    import numpy as np
    import reachy_sdk  # noqa: F401
    import websockets  # noqa: F401
else:
    grpc = pytest.importorskip("grpc")
    np = pytest.importorskip("numpy")
    pytest.importorskip("reachy_sdk")
    pytest.importorskip("websockets")
    pytest.importorskip("scipy")

import grpc  # noqa: E402
from reachy_sdk import ReachySDK  # noqa: E402
from reachy_sdk_api import fan_pb2_grpc, joint_pb2, joint_pb2_grpc, sensor_pb2_grpc  # noqa: E402

import mujoco_remote_backend as mrb  # noqa: E402
from fake_reachy_server import FakeFanService, FakeJointService, FakeSensorService  # noqa: E402
from kinematic_backend import JOINT_DEFS, KinematicBackend  # noqa: E402
from mujoco_remote_backend import KinematicBridge, MujocoRemoteBackend  # noqa: E402
from native_stub import IDX, NativeStub  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.tasks import rig_motion  # noqa: E402

import reset_watcher as rw  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")

ARM8 = R.ARM7 + ("r_gripper",)
ARM_IDX = [IDX[n] for n in ARM8]
TOL = 1e-5  # rad: "the same setpoint" after the SDK's float32 round trip
# goto stops at the last 100 Hz tick before `duration`, so its final setpoint
# misses the goal by the minimum-jerk residual (~1e-5 of the move), and the next
# goto ramps across that.  "Constant" (report §4 C4/C5/C7) allows it; the sag
# these tests detect is 1e-2 rad.
END_TOL = 1e-4


def _skew(start, goal, seconds, lag=0.04):
    """Per joint, how far a goto may already have moved by its first logged
    command: the bridge merges 100 Hz SDK writes into 20 ms batches, so that
    command can be the SDK's 2nd-4th tick (report §4: per-joint skew)."""
    x = min(1.0, lag / seconds)
    m = 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5
    return [abs(g - s) * m + TOL for s, g in zip(start, goal)]

# Sag on every right-arm joint, most on the shoulder, as gravity does.
SAG = {"r_shoulder_pitch": 0.03, "r_shoulder_roll": 0.01, "r_arm_yaw": 0.005,
       "r_elbow_pitch": 0.015, "r_forearm_yaw": 0.002, "r_wrist_pitch": 0.01,
       "r_wrist_roll": 0.005, "r_gripper": 0.008}


# ── rig ──────────────────────────────────────────────────────────────────────

class Rig:
    def __init__(self, stub: NativeStub, backend=None):
        self.stub = stub
        self.backend = backend
        self.server = grpc.server(ThreadPoolExecutor(max_workers=48))
        joint_backend = KinematicBridge(backend) if isinstance(backend, MujocoRemoteBackend) else backend
        joint_pb2_grpc.add_JointServiceServicer_to_server(FakeJointService(joint_backend), self.server)
        sensor_pb2_grpc.add_SensorServiceServicer_to_server(
            FakeSensorService(backend if isinstance(backend, MujocoRemoteBackend) else None), self.server)
        fan_pb2_grpc.add_FanControllerServiceServicer_to_server(FakeFanService(), self.server)
        self.port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        self.clients = []

    def client(self) -> ReachySDK:
        c = ReachySDK(host="127.0.0.1", sdk_port=self.port, camera_port=self.port,
                      restart_port=self.port)
        time.sleep(0.2)
        self.clients.append(c)
        return c

    def raw(self):
        return joint_pb2_grpc.JointServiceStub(grpc.insecure_channel(f"127.0.0.1:{self.port}"))

    def close(self):
        self.server.stop(0)
        if isinstance(self.backend, MujocoRemoteBackend):
            self.backend.stop()
        if self.stub is not None:
            self.stub.stop()


def _wait(pred, timeout=5.0, what="condition"):
    t = time.monotonic() + timeout
    while time.monotonic() < t:
        if pred():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture
def make_rig(monkeypatch):
    monkeypatch.setattr(mrb, "_RECONNECT_DELAY", 0.1)
    rigs = []

    def _make(pose=None, bias=None, compliant=True):
        stub = NativeStub(pose=pose, bias=bias, compliant=compliant).start()
        backend = MujocoRemoteBackend(url=stub.url())
        backend.start()
        _wait(lambda: len(stub.states) > 3 and backend.connection_state == mrb.ConnectionState.READY,
              what="bridge connected and receiving states")
        rig = Rig(stub, backend)
        rigs.append(rig)
        return rig

    yield _make
    for r in rigs:
        r.close()


def _commands(stub, a, b=None):
    with stub.lock:
        return list(stub.commands[a:b])


def _ncmd(stub):
    with stub.lock:
        return len(stub.commands)


class Recorder:
    """A `move` for fly_route that is the real `sdk_move`, bracketed so each
    goto / re-stream pass can be cut out of the stub's command log."""

    def __init__(self, stub):
        self.stub = stub
        self.calls = []   # (pose, seconds, first_idx, last_idx_exclusive)

    def __call__(self, arm, pose, seconds):
        a = _ncmd(self.stub)
        rig_motion.sdk_move(arm, pose, seconds)
        time.sleep(0.12)   # let the bridge's 20 ms sends land
        self.calls.append((dict(pose), seconds, a, _ncmd(self.stub)))


def _rad(pose):
    return [math.radians(pose[n]) for n in ARM8]


def _arm(cmd):
    return [cmd["target"][i] for i in ARM_IDX]


def _echoes(stub, cmds):
    """New right-arm target values bit-equal to float32(position) of a state
    the stub sent in the 0.5 s before the command (report §4, C9)."""
    with stub.lock:
        states = list(stub.states)
    n = 0
    prev = None
    for c in cmds:
        tgt = _arm(c)
        recent = [s["pos"] for s in states if c["t"] - 0.5 <= s["t"] <= c["t"]]
        for k, i in enumerate(ARM_IDX):
            if prev is not None and tgt[k] == prev[k]:
                continue
            if any(float(np.float32(p[i])) == tgt[k] for p in recent):
                n += 1
        prev = tgt
    return n


HOVERISH = dict(R.HOVER)
REST_SHUTISH = dict(R.REST_SHUT)
RESTISH = dict(R.REST)


# ── field filtering ──────────────────────────────────────────────────────────

def test_stream_honours_requested_fields(make_rig):
    rig = make_rig(bias=SAG)
    stub = rig.raw()
    req = joint_pb2.StreamJointsRequest(
        request=joint_pb2.JointsStateRequest(
            ids=[joint_pb2.JointId(uid=u) for _, u in JOINT_DEFS],
            requested_fields=[joint_pb2.JointField.PRESENT_POSITION,
                              joint_pb2.JointField.PRESENT_SPEED,
                              joint_pb2.JointField.PRESENT_LOAD]),
        publish_frequency=100)
    it = stub.StreamJointsState(req)
    msg = next(it)
    it.cancel()
    for st in msg.states:
        assert st.HasField("present_position")
        assert not st.HasField("goal_position"), "stream sent a field the client did not ask for"
        assert not st.HasField("compliant")
    full = stub.GetJointsState(joint_pb2.JointsStateRequest(
        ids=[joint_pb2.JointId(uid=10)], requested_fields=[joint_pb2.JointField.ALL]))
    st = full.states[0]
    for f in ("name", "uid", "present_position", "goal_position", "compliant", "temperature"):
        assert st.HasField(f) if f != "name" else st.name, f


# ── C1 / C9: goto starts from the last goal; no echoes ───────────────────────

def test_goto_starts_from_last_goal_not_present(make_rig):
    rig = make_rig(bias=SAG)
    reachy = rig.client()
    reachy.turn_on("r_arm")
    time.sleep(0.3)
    rec = Recorder(rig.stub)
    rec(reachy.r_arm, HOVERISH, 1.0)
    time.sleep(0.3)                       # fly_route's settle_s: nothing streams
    rec(reachy.r_arm, REST_SHUTISH, 1.0)
    (_, _, a1, b1), (_, _, a2, b2) = rec.calls
    last1 = _arm(_commands(rig.stub, a1, b1)[-1])
    first2 = _arm(_commands(rig.stub, a2, b2)[0])
    assert max(abs(x - y) for x, y in zip(last1, _rad(HOVERISH))) < 1e-3, \
        "the first goto did not end on its goal"
    # N2 (PR #141 re-review, 2026-09-24): _skew's own tolerance is TOL
    # (1e-5 rad) for a joint _skew judges to have barely moved yet at
    # `lag` -- that is only float32-round-trip margin, no allowance for
    # scheduling jitter in when this test's OWN observer (not the bridge)
    # samples the first logged command.  A measured run failed one such
    # joint at 1.055e-05 rad against a 1.0006e-05 rad bound, a 5% overshoot
    # on an assertion meant to catch a ~1e-2 rad sag (100x END_TOL) --
    # i.e. real jitter, not the bug.  Flooring the bound at END_TOL
    # (1e-4 rad) keeps the movement-based term for joints that actually
    # move (it already exceeds END_TOL there) while giving barely-moving
    # joints the same justified floor test_kinematic_backend_through_the_
    # same_path's B3 fix uses, with the same 100x margin below the sag
    # this test targets.
    allow = [max(a, END_TOL) for a in _skew(last1, _rad(REST_SHUTISH), 1.0)]
    worst = max(range(8), key=lambda k: abs(first2[k] - last1[k]) - allow[k])
    assert abs(first2[worst] - last1[worst]) < allow[worst], (
        f"C1: the second goto started {math.degrees(first2[worst] - last1[worst]):+.3f} deg "
        f"away from the last goal on {ARM8[worst]} -- it started from the sagged pose")


def test_no_echoes_under_sag(make_rig):
    rig = make_rig(bias=SAG)
    reachy = rig.client()
    reachy.turn_on("r_arm")
    time.sleep(0.3)
    rec = Recorder(rig.stub)
    rec(reachy.r_arm, HOVERISH, 1.0)      # the first goto may start at the (still) present pose
    rec(reachy.r_arm, REST_SHUTISH, 1.5)
    rec(reachy.r_arm, HOVERISH, 1.5)
    a = rec.calls[1][2]
    cmds = _commands(rig.stub, a)
    assert len(cmds) > 50
    n = _echoes(rig.stub, cmds)
    assert n == 0, f"C9: {n} setpoints were the measured position sent back as a target"


# ── C5 / C6: re-stream passes and holds, through the real fly_route ──────────

def test_restream_pass_is_constant(make_rig):
    # The elbow sags 6.9 deg, past TRACK_TOL (6), so fly_route re-streams; the
    # guard tolerance (20) then accepts the waypoint.  Synthetic route: the real
    # runner, not rig_routes.PLACE_ROUTE.
    rig = make_rig(bias=dict(SAG, r_elbow_pitch=math.radians(6.9)))
    reachy = rig.client()
    reachy.turn_on("r_arm")
    time.sleep(0.3)
    route = (R.Waypoint("A", HOVERISH, 1.0, 20.0, ("r_elbow_pitch",)),
             R.Waypoint("B", REST_SHUTISH, 1.0, 20.0, ("r_elbow_pitch",)))
    rec = Recorder(rig.stub)
    flown = rig_motion.fly_route(reachy.r_arm, route, settle_s=0.3, retries=2,
                                 settle_pass_s=0.3, move=rec)
    assert flown == ["A", "B"]
    # fly_route re-streams up to retries+1 times; a pass is a move to the same
    # pose as the move before it.
    calls = rec.calls
    passes = [c for i, c in enumerate(calls) if i and calls[i - 1][0] == c[0]]
    assert len(passes) >= 2, "the elbow bias should force re-stream passes"
    for pose, secs, a, b in passes:
        want = _rad(pose)
        for c in _commands(rig.stub, a, b):
            got = _arm(c)
            dev = max(abs(x - y) for x, y in zip(got, want))
            assert dev < END_TOL, (f"C5: a re-stream pass to the same waypoint moved the target "
                               f"({math.degrees(dev):.3f} deg off it) -- it started from the present pose")
    # C6: nothing new is commanded between one waypoint's last pass and the next goto
    ib = next(i for i, c in enumerate(calls) if c[0] == REST_SHUTISH)
    hold = _commands(rig.stub, calls[ib - 1][3], calls[ib][2])
    assert all(_arm(c) == _arm(hold[0]) for c in hold)


# ── C7: gripper sequencing is preserved ───────────────────────────────────────

def test_gripper_sequencing_preserved(make_rig):
    rig = make_rig(bias=SAG)
    reachy = rig.client()
    reachy.turn_on("r_arm")
    time.sleep(0.3)
    present_ish = dict(R.PRESENT)   # gripper OPEN
    route = (R.Waypoint("HOVER", HOVERISH, 1.0, 20.0),
             R.Waypoint("REST_SHUT", REST_SHUTISH, 1.0, 20.0),   # arm only
             R.Waypoint("REST", RESTISH, 1.0, 20.0),             # gripper only
             R.Waypoint("PRESENT", present_ish, 1.0, 20.0),      # arm only, gripper stays OPEN
             R.Waypoint("REST_SHUT2", REST_SHUTISH, 1.0, 20.0))  # arm AND gripper together
    rec = Recorder(rig.stub)
    rig_motion.fly_route(reachy.r_arm, route, settle_s=0.2, retries=0, move=rec)
    g = ARM8.index("r_gripper")
    # One goto per waypoint, then any re-stream passes to the same pose.
    by_name, k = {}, -1
    for i, (pose, secs, a, b) in enumerate(rec.calls):
        if i == 0 or rec.calls[i - 1][0] != pose:
            k += 1
        by_name.setdefault(route[k].name, []).extend(_commands(rig.stub, a, b))
    for name in ("REST_SHUT", "PRESENT"):
        grips = [_arm(c)[g] for c in by_name[name]]
        assert max(grips) - min(grips) < END_TOL, \
            f"C7: the gripper target moved {math.degrees(max(grips) - min(grips)):.3f} deg during arm-only {name}"
    for k in range(7):
        arm = [_arm(c)[k] for c in by_name["REST"]]
        assert max(arm) - min(arm) < END_TOL, \
            f"C7: {ARM8[k]} moved {math.degrees(max(arm) - min(arm)):.3f} deg during the gripper-only REST waypoint"
    both = by_name["REST_SHUT2"]
    assert len({_arm(c)[g] for c in both}) > 5 and len({_arm(c)[3] for c in both}) > 5, \
        "PRESENT -> REST_SHUT moves the arm and the gripper in one goto"


# ── compliance ────────────────────────────────────────────────────────────────

def test_stiffening_after_hand_move_holds_in_place(make_rig):
    rig = make_rig(bias=SAG)
    reachy = rig.client()
    reachy.turn_on("r_arm")
    time.sleep(0.3)
    rig_motion.sdk_move(reachy.r_arm, HOVERISH, 1.0)
    time.sleep(0.2)
    reachy.turn_off("r_arm")
    _wait(lambda: rig.stub.compliant[IDX["r_elbow_pitch"]], what="elbow compliant")
    moved = math.radians(-100.0)
    rig.stub.move_by_hand("r_elbow_pitch", moved)
    time.sleep(0.2)
    full = rig.raw().GetJointsState(joint_pb2.JointsStateRequest(
        ids=[joint_pb2.JointId(uid=13)], requested_fields=[joint_pb2.JointField.ALL])).states[0]
    assert full.compliant.value is True, "compliance must be reported from the native state"
    assert abs(full.goal_position.value - moved) < 1e-6, "a compliant joint's goal follows its pose"
    reachy.turn_on("r_arm")
    _wait(lambda: not rig.stub.compliant[IDX["r_elbow_pitch"]], what="elbow stiff")
    with rig.stub.lock:
        tgt = rig.stub.target[IDX["r_elbow_pitch"]]
    assert abs(tgt - moved) < 1e-6, (
        f"re-stiffening drove the elbow to {math.degrees(tgt):.1f} deg, not where it was "
        f"left ({math.degrees(moved):.1f})")


# ── reset ─────────────────────────────────────────────────────────────────────

def _reset(rig):
    ev = rig.backend.request_reset()
    assert ev.wait(3), "no reset_ack"
    _wait(lambda: rig.backend._last_target is not None, what="post-reset baseline")


def _goal(rig, name):
    uid = dict(JOINT_DEFS)[name]
    return rig.raw().GetJointsState(joint_pb2.JointsStateRequest(
        ids=[joint_pb2.JointId(uid=uid)],
        requested_fields=[joint_pb2.JointField.ALL])).states[0].goal_position.value


def test_reset_reports_post_reset_pose_and_new_client_starts_there(make_rig):
    rig = make_rig(bias=SAG)
    reachy = rig.client()
    reachy.turn_on("r_arm")
    time.sleep(0.3)
    rig_motion.sdk_move(reachy.r_arm, HOVERISH, 1.0)
    time.sleep(0.2)
    _reset(rig)
    time.sleep(0.2)
    with rig.stub.lock:
        post_target = list(rig.stub.target)       # native: goals = pose at reset
    g = _goal(rig, "r_shoulder_pitch")
    assert abs(g - post_target[IDX["r_shoulder_pitch"]]) < 1e-6
    assert abs(g - math.radians(HOVERISH["r_shoulder_pitch"])) > 0.1, \
        "reported goal must be the post-reset baseline, not the pre-reset command"
    # (b) a client connecting now starts its first goto from that baseline
    fresh = rig.client()
    a = _ncmd(rig.stub)
    rig_motion.sdk_move(fresh.r_arm, REST_SHUTISH, 0.6)
    time.sleep(0.12)
    first = _arm(_commands(rig.stub, a)[0])
    base = [post_target[i] for i in ARM_IDX]
    allow = _skew(base, _rad(REST_SHUTISH), 0.6)
    assert all(abs(x - y) < e for x, y, e in zip(first, base, allow))


def test_turn_on_after_reset_refreshes_a_live_client(make_rig):
    """What the panel does: reset, then turn_on, then move.  SDK turn_on sets
    its local goal to the present pose, so a live client is not stale."""
    rig = make_rig(bias=SAG)
    reachy = rig.client()
    reachy.turn_on("r_arm")
    time.sleep(0.3)
    rig_motion.sdk_move(reachy.r_arm, HOVERISH, 1.0)
    time.sleep(0.2)   # let the goto's last setpoints reach the stub: commands
    #                   still queued at a reset are held and sent AFTER it
    _reset(rig)
    time.sleep(0.2)
    reachy.turn_on("r_arm")
    time.sleep(0.1)
    with rig.stub.lock:
        base = [rig.stub.initial[i] for i in ARM_IDX]
    a = _ncmd(rig.stub)
    rig_motion.sdk_move(reachy.r_arm, REST_SHUTISH, 0.6)
    time.sleep(0.12)
    first = _arm(_commands(rig.stub, a)[0])
    # Starts where the arm is after the reset (give or take the sag since),
    # not from the pre-reset HOVER goal.
    k = ARM8.index("r_shoulder_pitch")
    assert abs(first[k] - base[k]) < 0.05
    assert abs(first[k] - math.radians(HOVERISH["r_shoulder_pitch"])) > 0.5


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN LIMITATION: the stream no longer overwrites the SDK's goal cache, so "
    "a client that stays connected across a reset and moves WITHOUT turn_on "
    "starts its goto from its own pre-reset goal -- as it would on hardware "
    "if something else moved the arm.  Reconnect, or turn_on, after a reset."))
def test_live_client_goto_after_reset_without_turn_on(make_rig):
    rig = make_rig(bias=SAG)
    reachy = rig.client()
    reachy.turn_on("r_arm")
    time.sleep(0.3)
    rig_motion.sdk_move(reachy.r_arm, HOVERISH, 1.0)
    time.sleep(0.2)   # let the goto's last setpoints reach the stub: commands
    #                   still queued at a reset are held and sent AFTER it
    _reset(rig)
    time.sleep(0.2)
    with rig.stub.lock:
        base = [rig.stub.target[i] for i in ARM_IDX]
    a = _ncmd(rig.stub)
    rig_motion.sdk_move(reachy.r_arm, REST_SHUTISH, 0.6)
    time.sleep(0.12)
    first = _arm(_commands(rig.stub, a)[0])
    allow = _skew(base, _rad(REST_SHUTISH), 0.6)
    assert all(abs(x - y) < e for x, y, e in zip(first, base, allow))


def test_reset_refused_does_not_wedge_commands_or_falsely_ack(make_rig, tmp_path):
    """PR #141 review, B1: the native server can refuse a reset with
    `control_held` while an execution lease is held (the panel executor
    takes one), sending back an error and no ack -- and it keeps streaming
    state throughout, exactly like NativeStub with refuse_resets set. A
    refused reset must not hold every later command forever, must not grow
    the queue without bound, and reset_watcher must not tell a caller
    (reset.sh, request_reset_and_wait()) that it succeeded."""
    rig = make_rig(bias=SAG)
    reachy = rig.client()
    reachy.turn_on("r_arm")
    time.sleep(0.2)
    rig.backend._reset_timeout = 0.3   # keep the test fast
    rig.stub.refuse_resets = True

    req = tmp_path / "reset_request"
    ack = tmp_path / "reset_ack"
    req.write_text("41")
    result = rw.reset_watcher_step(rig.backend, req, ack)

    assert result == "41"
    assert not ack.exists(), "a refused reset must not publish a success ack"
    assert rig.stub.refusals >= 1
    assert rig.backend.connection_state == mrb.ConnectionState.ABORTED

    # Commands must not be held forever: once the abort clears
    # _awaiting_reset, the next state (still streaming) re-seeds and the
    # bridge drains _pending_cmds on its next 20 ms tick.  The goto streams
    # at 100 Hz for its whole duration, so _pending_cmds legitimately holds
    # a few items between send ticks while it runs -- that's normal
    # draining, not a wedge. What the review's probe actually found was
    # UNBOUNDED growth (every command stayed queued, forever): track the
    # high-water mark across the goto rather than requiring exact zero at
    # one sleep-timed instant, which is sensitive to scheduling jitter.
    a = _ncmd(rig.stub)
    max_pending = 0
    deadline = time.monotonic() + 1.5
    rig_motion.sdk_move(reachy.r_arm, REST_SHUTISH, 0.4)
    while time.monotonic() < deadline:
        with rig.backend._lock:
            max_pending = max(max_pending, len(rig.backend._pending_cmds))
        time.sleep(0.02)
    assert _ncmd(rig.stub) > a, "commands must reach native after a refused reset"
    assert max_pending < 20, \
        f"_pending_cmds grew to {max_pending} -- looks wedged, not draining"

    # A later, unrefused reset must still succeed and publish its ack.
    rig.stub.refuse_resets = False
    req2 = tmp_path / "reset_request2"
    ack2 = tmp_path / "reset_ack2"
    req2.write_text("42")
    result2 = rw.reset_watcher_step(rig.backend, req2, ack2)
    assert result2 == "42"
    assert ack2.read_text() == "42"


# ── reconnect ─────────────────────────────────────────────────────────────────

def test_bridge_reconnect_reseeds_from_new_session(make_rig):
    rig = make_rig(bias=SAG)
    reachy = rig.client()
    reachy.turn_on("r_arm")
    time.sleep(0.3)
    rig_motion.sdk_move(reachy.r_arm, HOVERISH, 1.0)
    time.sleep(0.2)
    n_conn = rig.backend._reconnect_count
    new_pose = {"r_shoulder_pitch": math.radians(10.0), "l_elbow_pitch": math.radians(-30.0)}
    rig.stub.restart(new_pose)                     # a new native process
    _wait(lambda: rig.backend._reconnect_count > n_conn
          and abs(_goal(rig, "r_shoulder_pitch") - math.radians(10.0)) < 1e-6,
          what="bridge reconnected and re-seeded from the new session")
    assert abs(_goal(rig, "r_shoulder_pitch") - math.radians(10.0)) < 1e-6
    assert abs(_goal(rig, "l_elbow_pitch") - math.radians(-30.0)) < 1e-6
    # The first command of the new session must not replay the old session's
    # targets for joints it does not name.
    a = _ncmd(rig.stub)
    reachy.r_arm.r_gripper.goal_position = 10.0
    _wait(lambda: _ncmd(rig.stub) > a, what="a command in the new session")
    first = _commands(rig.stub, a)[0]["target"]
    assert abs(first[IDX["r_shoulder_pitch"]] - math.radians(10.0)) < 1e-6
    assert abs(first[IDX["l_elbow_pitch"]] - math.radians(-30.0)) < 1e-6


def test_new_client_mid_run_starts_from_last_goal(make_rig):
    rig = make_rig(bias=SAG)
    first_client = rig.client()
    first_client.turn_on("r_arm")
    time.sleep(0.3)
    rig_motion.sdk_move(first_client.r_arm, HOVERISH, 1.0)
    time.sleep(0.3)                               # the arm sags below HOVER
    second = rig.client()
    assert abs(second.r_arm.r_shoulder_pitch.goal_position - HOVERISH["r_shoulder_pitch"]) < 1e-3
    a = _ncmd(rig.stub)
    rig_motion.sdk_move(second.r_arm, REST_SHUTISH, 0.6)
    time.sleep(0.12)
    first = _arm(_commands(rig.stub, a)[0])
    # NB2 (PR #141 re-review, 2026-09-24, issue #142): same justification as
    # N2's fix to test_goto_starts_from_last_goal_not_present above -- _skew's
    # own TOL (1e-5 rad) floor is only float32-round-trip margin, with no
    # allowance for this test's own sampling jitter. Failed 1/20 isolated
    # runs at aa83903 (assert at this line). Flooring at END_TOL (1e-4 rad)
    # keeps the same 100x margin below the ~1e-2 rad sag this test targets.
    allow = [max(a, END_TOL) for a in _skew(_rad(HOVERISH), _rad(REST_SHUTISH), 0.6)]
    assert all(abs(x - y) < e for x, y, e in zip(first, _rad(HOVERISH), allow))


# ── the kinematic backend through the same path ──────────────────────────────

def test_kinematic_backend_through_the_same_path():
    """The kinematic backend always reported its true goal, but the unfiltered
    stream still overwrote the SDK's `compliant` register between turn_on's
    write and its send -- so turn_on could be lost here too (it was, on
    8c0dad2: the elbow stayed compliant).  With the fields honoured it holds."""
    kin = KinematicBackend()
    kin.start()
    rig = Rig(None, kin)
    try:
        reachy = rig.client()
        reachy.turn_on("r_arm")
        time.sleep(0.2)
        rig_motion.sdk_move(reachy.r_arm, HOVERISH, 1.0)
        time.sleep(1.0)
        for n in ("r_shoulder_pitch", "r_elbow_pitch", "r_wrist_pitch"):
            assert abs(getattr(reachy.r_arm, n).present_position - HOVERISH[n]) < 1.0
        st = rig.raw().GetJointsState(joint_pb2.JointsStateRequest(
            ids=[joint_pb2.JointId(uid=13)], requested_fields=[joint_pb2.JointField.ALL])).states[0]
        # B3 (PR #141 review): compare in radians against END_TOL, not a
        # hardcoded 1e-3 degrees (1.7e-5 rad) -- goto's own end-of-move
        # residual is ~2.3e-5 rad, bigger than that bound, which made this
        # assertion flaky. END_TOL (1e-4 rad) exists for exactly this.
        assert abs(st.goal_position.value - math.radians(HOVERISH["r_elbow_pitch"])) < END_TOL
        assert st.compliant.value is False
    finally:
        rig.server.stop(0)
