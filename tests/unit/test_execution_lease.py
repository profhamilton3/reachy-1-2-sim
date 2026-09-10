"""Issue #51: the simulator's execution lease.

Exercises the arbitration rules directly against a MujocoServer instance rather
than over a socket.  The rules are pure decisions about who may send what, and
testing them here keeps them in the offline tier — no MuJoCo model, no GL, no
websocket — while the round trip itself is covered by the contract tier.

The lease exists because the browser panel reaches the simulator on its own
socket.  Disabling a button in one page cannot stop a `place_object` arriving
mid-grasp, so the refusal has to live where every client's message lands.
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../native_mujoco"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    AcquireControl,
    ControlAck,
    ReleaseControl,
    decode,
)


class _Server:
    """The lease slice of ReachyMujocoServer, without compiling a model.

    That class's __init__ loads a physics model and starts threads, neither of
    which these rules touch.  Binding the real methods onto a bare object
    exercises the shipped code without that cost — if a method moves or changes
    shape, this fails rather than silently testing a copy of it.
    """

    def __init__(self):
        import server as server_module
        self._control = None
        cls = server_module.ReachyMujocoServer
        for name in ("_active_control", "_release_control", "_control_ack",
                     "_blocked_by_control"):
            setattr(self, name, getattr(cls, name).__get__(self, cls))

    def grant(self, conn_id=1, client_id="command-panel",
              motion_client_id="docker-core", ttl_s=120.0):
        self._control = {
            "conn_id": conn_id,
            "client_id": client_id,
            "motion_client_id": motion_client_id,
            "expires_at": time.monotonic() + ttl_s,
            "reason": "test",
        }


@pytest.fixture
def srv():
    return _Server()


# ---------------------------------------------------------------------------
# Nothing is refused while the lease is free
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mtype", [
    "place_object", "scene_load", "reset", "joint_command", "pause",
    "heartbeat", "zoom_command",
])
def test_an_unheld_lease_blocks_nothing(srv, mtype):
    assert srv._blocked_by_control(mtype, "anyone") is None


# ---------------------------------------------------------------------------
# Scene edits are refused for everyone while a lease is held
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mtype", ["place_object", "scene_load", "reset"])
def test_scene_edits_are_refused_while_held(srv, mtype):
    srv.grant()
    for who in ("camera-panel", "notebook", "docker-core", "command-panel"):
        # Even the lease holder and the mover are refused: place_object during
        # a grasp teleports the object out of a closing gripper regardless of
        # who sent it.
        refusal = srv._blocked_by_control(mtype, who)
        assert refusal is not None
        assert mtype in refusal
        assert "command-panel" in refusal


def test_the_browser_panels_placement_is_refused_by_name(srv):
    srv.grant()
    refusal = srv._blocked_by_control("place_object", "camera-panel")
    assert "place_object is refused" in refusal
    assert "execution lease" in refusal


# ---------------------------------------------------------------------------
# joint_command is restricted to the named mover
# ---------------------------------------------------------------------------

def test_only_the_named_motion_client_may_command_joints(srv):
    srv.grant(motion_client_id="docker-core")
    assert srv._blocked_by_control("joint_command", "docker-core") is None
    for other in ("camera-panel", "notebook", "command-panel", ""):
        refusal = srv._blocked_by_control("joint_command", other)
        assert refusal is not None
        assert "docker-core" in refusal


def test_the_holder_need_not_be_the_mover(srv):
    """The panel holds the lease; motion arrives through the SDK bridge.

    They are different connections, so a design that assumed the acquirer was
    the mover could not express this arrangement at all.
    """
    srv.grant(client_id="command-panel", motion_client_id="docker-core")
    assert srv._blocked_by_control("joint_command", "docker-core") is None
    assert srv._blocked_by_control("joint_command", "command-panel") is not None


def test_a_lease_with_no_mover_refuses_all_joint_commands(srv):
    srv.grant(motion_client_id="")
    for who in ("docker-core", "camera-panel", ""):
        assert srv._blocked_by_control("joint_command", who) is not None


# ---------------------------------------------------------------------------
# Messages that are never arbitrated
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mtype", [
    "heartbeat", "heartbeat_ack", "disconnect", "pause", "zoom_command",
    "acquire_control", "release_control",
])
def test_liveness_and_lease_messages_are_never_blocked(srv, mtype):
    srv.grant()
    # Blocking heartbeats would make the server drop the very client holding
    # the lease six seconds later.
    assert srv._blocked_by_control(mtype, "camera-panel") is None


# ---------------------------------------------------------------------------
# Expiry and release
# ---------------------------------------------------------------------------

def test_an_expired_lease_stops_blocking_and_is_cleared(srv):
    srv.grant(ttl_s=-0.01)          # already past
    assert srv._blocked_by_control("place_object", "camera-panel") is None
    assert srv._control is None     # cleared, not merely ignored


def test_a_live_lease_is_not_cleared(srv):
    srv.grant(ttl_s=60)
    assert srv._active_control() is not None
    assert srv._control is not None


def test_release_only_works_for_the_holding_connection(srv):
    srv.grant(conn_id=7)
    assert srv._release_control(9) is False
    assert srv._control is not None
    assert srv._release_control(7) is True
    assert srv._control is None


def test_releasing_an_unheld_lease_is_harmless(srv):
    assert srv._release_control(1) is False


# ---------------------------------------------------------------------------
# The ack a client reads
# ---------------------------------------------------------------------------

def test_ack_describes_a_held_lease(srv):
    srv.grant(client_id="command-panel", motion_client_id="docker-core",
              ttl_s=120)
    ack = srv._control_ack("req-1", True)
    assert ack.granted is True
    assert ack.held is True
    assert ack.holder == "command-panel"
    assert ack.motion_client_id == "docker-core"
    assert 0 < ack.expires_in_s <= 120
    assert ack.error == ""


def test_ack_for_a_refused_acquire_names_the_holder(srv):
    srv.grant(client_id="someone-else")
    ack = srv._control_ack("req-2", False, "lease already held by 'someone-else'")
    assert ack.granted is False
    assert ack.held is True
    assert ack.holder == "someone-else"
    assert "someone-else" in ack.error


def test_ack_after_release_reports_no_holder(srv):
    ack = srv._control_ack("req-3", False)
    assert ack.held is False
    assert ack.holder == ""
    assert ack.expires_in_s == 0.0


# ---------------------------------------------------------------------------
# Wire shape
# ---------------------------------------------------------------------------

def test_acquire_and_release_round_trip():
    acq = AcquireControl(client_id="command-panel",
                         motion_client_id="docker-core",
                         reason="task abc", ttl_s=90.0, request_id="r1")
    got = decode(acq.encode())
    assert got["type"] == "acquire_control"
    assert got["motion_client_id"] == "docker-core"
    assert got["ttl_s"] == 90.0

    rel = decode(ReleaseControl(request_id="r1").encode())
    assert rel["type"] == "release_control"

    ack = decode(ControlAck(request_id="r1", granted=True, held=True,
                            holder="command-panel").encode())
    assert ack["type"] == "control_ack"
    assert ack["granted"] is True


def test_the_message_catalogue_lists_every_type_the_server_handles():
    """The dispatch sets had drifted: place_object and place_ack were missing.

    A catalogue that silently omits shipped message types is worse than none,
    because it reads as an authoritative list.
    """
    import protocol
    for name in ("hello", "joint_command", "place_object", "scene_load",
                 "reset", "pause", "heartbeat", "heartbeat_ack", "disconnect",
                 "zoom_command", "acquire_control", "release_control"):
        assert name in protocol._CLIENT_TYPES, name
    for name in ("hello_ack", "state", "camera_frame", "scene_ack", "place_ack",
                 "reset_ack", "heartbeat", "heartbeat_ack", "error",
                 "shutdown", "control_ack"):
        assert name in protocol._SERVER_TYPES, name


def test_protocol_version_is_unchanged():
    """Adding messages must not break existing clients.

    The lease is additive: a client that never sends acquire_control behaves
    exactly as before, so the version does not move.
    """
    assert PROTOCOL_VERSION == 1
