"""Issue #50: the task routes with a live simulator link behind them.

Uses a stub link rather than a real simulator so the suite stays offline, but
drives the real server over HTTP so the `/capabilities` shape and the
invalidation path are checked where the browser actually meets them.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../web"))
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))

import camera_server  # noqa: E402
from panel_routes import SESSION_COOKIE  # noqa: E402
from panel_sim_link import SimSnapshot  # noqa: E402


class StubLink:
    """Stands in for SimLink: same two methods the server actually calls."""

    def __init__(self):
        self._snapshot = None
        self.stopped = False

    def push(self, objects, *, scene_revision="rev-1", sim_step=1000):
        self._snapshot = SimSnapshot(
            sim_step=sim_step, sim_time_s=sim_step * 0.002,
            scene_revision=scene_revision, objects=dict(objects),
            received_at=time.monotonic(),
        )

    def clear(self):
        self._snapshot = None

    def snapshot(self):
        return self._snapshot

    def stop(self):
        self.stopped = True

    @property
    def status(self):
        snap = self._snapshot
        return {"state": "connected" if snap else "connecting",
                "live": snap is not None, "detail": "stub",
                "url": "ws://stub", "age_s": 0.0 if snap else None,
                "scene_revision": snap.scene_revision if snap else "",
                "sim_step": snap.sim_step if snap else 0}


@pytest.fixture(scope="module")
def live():
    scene_path = os.path.abspath(
        os.path.join(_HERE, "../../scenes/FWDCenterLabSivaPool.yaml")
    )
    os.environ["REACHY_SIM_SCENE_FILE"] = scene_path
    camera_server._SCENE_FILE = scene_path
    camera_server._PANEL = None

    link = StubLink()
    httpd = camera_server._ThreadingHTTPServer(
        ("127.0.0.1", 0), camera_server._Handler
    )
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host = f"127.0.0.1:{httpd.server_address[1]}"

    # Build the panel, then swap in the stub so no real socket is opened.
    urllib.request.urlopen(f"http://{host}/capabilities", timeout=5).read()
    if camera_server._SIM_LINK is not None:
        camera_server._SIM_LINK.stop()
    camera_server._SIM_LINK = link
    camera_server._PANEL.link = link
    try:
        yield host, link
    finally:
        httpd.shutdown()
        httpd.server_close()
        if camera_server._PANEL is not None:
            camera_server._PANEL.shutdown()
        camera_server._PANEL = None
        camera_server._SIM_LINK = None


class Client:
    def __init__(self, host):
        self.host = host
        self.cookie = ""
        self.get("/")

    def get(self, path):
        return self._call("GET", path, None)

    def post(self, path, body):
        return self._call("POST", path, body)

    def _call(self, method, path, body):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"http://{self.host}{path}", data=data,
                                     method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
            req.add_header("Origin", f"http://{self.host}")
        if self.cookie:
            req.add_header("Cookie", f"{SESSION_COOKIE}={self.cookie}")
        try:
            resp = urllib.request.urlopen(req, timeout=5)
            status, raw, hdrs = resp.status, resp.read(), resp.headers
        except urllib.error.HTTPError as exc:
            status, raw, hdrs = exc.code, exc.read(), exc.headers
        sc = hdrs.get("Set-Cookie", "")
        if SESSION_COOKIE in sc:
            self.cookie = sc.split(f"{SESSION_COOKIE}=", 1)[1].split(";")[0]
        try:
            return status, json.loads(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return status, raw

    def settle(self, task_id, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            status, body = self.get(f"/tasks/{task_id}")
            assert status == 200, body
            if body["state"] != "planning":
                return body
            time.sleep(0.02)
        raise AssertionError("task stayed in planning")


def cell_pos(host, name, z_offset=0.05):
    scene = json.loads(
        urllib.request.urlopen(f"http://{host}/scene", timeout=5).read()
    )
    cell = next(c for c in scene["cells"] if c["name"] == name)
    return (cell["x"], cell["y"], cell["top_z"] + z_offset)


# ---------------------------------------------------------------------------

def test_capabilities_report_the_link_state(live):
    host, link = live
    link.clear()
    c = Client(host)
    _, body = c.get("/capabilities")
    assert body["sim_link"]["live"] is False
    assert body["capabilities"]["live_scene"] is False

    link.push({"soda_can": cell_pos(host, "r1c1")})
    _, body = c.get("/capabilities")
    assert body["sim_link"]["live"] is True
    assert body["sim_link"]["scene_revision"] == "rev-1"
    assert body["capabilities"]["live_scene"] is True
    # Still planning-only: seeing the board is not being able to move on it.
    assert body["capabilities"]["live_execution"] is False
    assert body["capabilities"]["execution_mode"] == "planning_only"


def test_a_placed_can_resolves_the_worked_example_to_a_proposal(live):
    host, link = live
    link.push({"soda_can": cell_pos(host, "r1c1")})
    c = Client(host)

    _, task = c.post("/tasks", {"text": "Put the recycle item in the bin."})
    task = c.settle(task["task_id"])
    # Still no bin in this scene, so still a clarification -- but now it can
    # name the object without asking, because it can see the board.
    assert task["state"] == "needs_clarification"
    assert "no bin or tray configured" in task["events"][-1]["text"]
    assert "r1c1" not in task["events"][-1]["choices"]

    _, task = c.post(f"/tasks/{task['task_id']}/reply",
                     {"question_id": task["question_id"], "text": "r2c2"})
    task = c.settle(task["task_id"])
    assert task["state"] == "awaiting_confirmation"
    p = task["proposal"]
    assert p["target_id"] == "soda_can"           # resolved, not asked
    assert p["destination"] == "cell:r2c2"
    assert p["legacy_destination"] is None
    assert p["state_evidence"]["target_cell"] == "r1c1"
    c.post(f"/tasks/{task['task_id']}/cancel", {})


def test_moving_the_target_invalidates_an_open_proposal(live):
    host, link = live
    link.push({"soda_can": cell_pos(host, "r1c1")})
    c = Client(host)

    _, task = c.post("/tasks", {"text": "put soda_can on r2c2"})
    task = c.settle(task["task_id"])
    p = task["proposal"]

    # Someone uses the board controls while the proposal is on screen.
    link.push({"soda_can": cell_pos(host, "r1c3")})

    status, body = c.post(f"/tasks/{task['task_id']}/confirm",
                          {"plan_id": p["plan_id"],
                           "plan_version": p["plan_version"]})
    assert status == 409
    assert body["code"] == "plan_invalidated"

    _, after = c.get(f"/tasks/{task['task_id']}")
    assert after["state"] == "failed"
    assert after["proposal"] is None


def test_advancing_sim_step_alone_still_confirms(live):
    host, link = live
    link.push({"soda_can": cell_pos(host, "r1c1")}, sim_step=1000)
    c = Client(host)

    _, task = c.post("/tasks", {"text": "put soda_can on r2c2"})
    task = c.settle(task["task_id"])
    p = task["proposal"]

    # Same world, a quarter of a million steps later.
    link.push({"soda_can": cell_pos(host, "r1c1")}, sim_step=250_000)

    status, done = c.post(f"/tasks/{task['task_id']}/confirm",
                          {"plan_id": p["plan_id"],
                           "plan_version": p["plan_version"]})
    assert status == 200, done
    assert done["state"] == "confirmed_no_motion"
    assert done["detail"] == "Plan confirmed; no movement performed."


def test_losing_the_link_invalidates_an_open_proposal(live):
    host, link = live
    link.push({"soda_can": cell_pos(host, "r1c1")})
    c = Client(host)

    _, task = c.post("/tasks", {"text": "put soda_can on r2c2"})
    task = c.settle(task["task_id"])
    p = task["proposal"]

    link.clear()          # simulator went away

    status, body = c.post(f"/tasks/{task['task_id']}/confirm",
                          {"plan_id": p["plan_id"],
                           "plan_version": p["plan_version"]})
    assert status == 409
    assert "lost my link" in body["error"]


def test_a_can_left_in_the_pool_is_not_a_candidate(live):
    host, link = live
    # Where FWDCenterLabSivaPool actually parks the can.
    link.push({"soda_can": (0.55, 0.95, 0.0575)})
    c = Client(host)

    _, task = c.post("/tasks", {"text": "put the recycle item on r2c2"})
    task = c.settle(task["task_id"])
    assert task["state"] == "needs_clarification"
    assert "No recyclable object is on the board" in task["events"][-1]["text"]
    c.post(f"/tasks/{task['task_id']}/cancel", {})


def test_a_tasks_mode_agrees_with_capabilities(live):
    """The two must never disagree in the same second.

    `/capabilities` computed the live answer while `GET /tasks/{id}` reported
    the value fixed at construction, so the badge could read "live execution"
    while the task it described said "planning_only".
    """
    host, link = live
    link.push({"soda_can": cell_pos(host, "r1c1")})
    c = Client(host)

    class YesExecutor:
        def available(self, proposal=None):
            return True, ""

    saved = camera_server._PANEL.executor
    camera_server._PANEL.executor = YesExecutor()
    try:
        _, caps = c.get("/capabilities")
        _, task = c.post("/tasks", {"text": "put soda_can on r2c2"})
        assert caps["capabilities"]["execution_mode"] == "live_simulation"
        assert task["mode"] == caps["capabilities"]["execution_mode"]
        assert task["capabilities"]["live_execution"] is True
        c.post(f"/tasks/{task['task_id']}/cancel", {})
    finally:
        camera_server._PANEL.executor = saved

    # And back again once the executor is gone.
    _, caps = c.get("/capabilities")
    _, task = c.post("/tasks", {"text": "put soda_can on r2c2"})
    assert caps["capabilities"]["execution_mode"] == "planning_only"
    assert task["mode"] == "planning_only"
    c.post(f"/tasks/{task['task_id']}/cancel", {})


def test_camera_and_scene_routes_are_unaffected_by_the_link(live):
    host, link = live
    link.clear()
    c = Client(host)
    assert c.get("/status")[0] == 200
    status, scene = c.get("/scene")
    assert status == 200 and scene["objects"] and scene["cells"]
    body = urllib.request.urlopen(f"http://{host}/", timeout=5).read()
    assert b"TALK TO REACHY" in body and b"THE BOARD" in body
