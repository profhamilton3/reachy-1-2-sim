"""Issue #49: the panel's HTTP task routes, driven against a real server.

Runs camera_server's own handler on an ephemeral port rather than calling the
route functions directly, because half of what is being checked lives in the
HTTP layer: the session cookie, the origin refusal, the body-size cap, and the
fact that adding POST did not break the existing GETs.
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
from panel_routes import (  # noqa: E402
    SESSION_COOKIE,
    origin_allowed,
    parse_body,
    session_from_cookie,
)
from tasks import TaskError  # noqa: E402


@pytest.fixture(scope="module")
def server():
    """camera_server bound to an ephemeral loopback port."""
    os.environ["REACHY_SIM_SCENE_FILE"] = os.path.abspath(
        os.path.join(_HERE, "../../scenes/FWDCenterLabSivaPool.yaml")
    )
    camera_server._SCENE_FILE = os.environ["REACHY_SIM_SCENE_FILE"]
    camera_server._PANEL = None

    httpd = camera_server._ThreadingHTTPServer(("127.0.0.1", 0), camera_server._Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host = f"127.0.0.1:{httpd.server_address[1]}"
    try:
        yield host
    finally:
        httpd.shutdown()
        httpd.server_close()
        if camera_server._PANEL is not None:
            camera_server._PANEL.shutdown()


class Client:
    """Minimal cookie-keeping client; urllib has no session of its own."""

    def __init__(self, host):
        self.host = host
        self.cookie = ""

    def get(self, path, headers=None):
        return self._call("GET", path, None, headers)

    def post(self, path, body, headers=None, origin=True):
        h = dict(headers or {})
        h["Content-Type"] = "application/json"
        if origin:
            h.setdefault("Origin", f"http://{self.host}")
        return self._call("POST", path, body, h)

    def _call(self, method, path, body, headers):
        data = None
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
        req = urllib.request.Request(f"http://{self.host}{path}", data=data,
                                     method=method)
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        if self.cookie:
            req.add_header("Cookie", f"{SESSION_COOKIE}={self.cookie}")
        try:
            resp = urllib.request.urlopen(req, timeout=5)
            status, raw, hdrs = resp.status, resp.read(), resp.headers
        except urllib.error.HTTPError as exc:
            status, raw, hdrs = exc.code, exc.read(), exc.headers
        set_cookie = hdrs.get("Set-Cookie", "")
        if SESSION_COOKIE in set_cookie:
            self.cookie = set_cookie.split(f"{SESSION_COOKIE}=", 1)[1].split(";")[0]
        try:
            return status, json.loads(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return status, raw

    def open_session(self):
        self.get("/")
        assert self.cookie
        return self

    def settle(self, task_id, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            status, body = self.get(f"/tasks/{task_id}")
            assert status == 200, body
            if body["state"] != "planning":
                return body
            time.sleep(0.02)
        raise AssertionError("task stayed in planning")


@pytest.fixture
def client(server):
    return Client(server).open_session()


# ---------------------------------------------------------------------------
# The page still works
# ---------------------------------------------------------------------------

def test_index_still_serves_the_page_and_issues_a_session(server):
    c = Client(server)
    status, body = c.get("/")
    assert status == 200
    assert b"TALK TO REACHY" in body
    assert b"LEFT CAMERA" in body and b"THE BOARD" in body
    assert c.cookie


def test_existing_routes_are_untouched(client):
    status, body = client.get("/status")
    assert status == 200 and "left_age_ms" in body

    status, body = client.get("/scene")
    assert status == 200
    assert body["objects"] and body["cells"]
    assert body["ws_port"] == camera_server._SIM_WS_PORT

    assert client.get("/nope")[0] == 404


def test_a_refresh_reuses_the_session_rather_than_minting_one(server):
    c = Client(server).open_session()
    first = c.cookie
    c.get("/")
    assert c.cookie == first


def test_capabilities_report_planning_only(client):
    status, body = client.get("/capabilities")
    assert status == 200
    caps = body["capabilities"]
    assert caps["live_execution"] is False
    assert caps["execution_mode"] == "planning_only"
    assert caps["semantic_source"] == "scene_data"


# ---------------------------------------------------------------------------
# Request hygiene
# ---------------------------------------------------------------------------

def test_cross_origin_post_is_refused(client):
    status, body = client.post("/tasks", {"text": "put soda_can on r2c2"},
                               headers={"Origin": "http://evil.example"})
    assert status == 403
    assert body["code"] == "bad_origin"


def test_post_without_an_origin_is_refused(client):
    status, body = client.post("/tasks", {"text": "put soda_can on r2c2"},
                               origin=False)
    assert status == 403


def test_post_without_a_session_is_refused(server):
    c = Client(server)          # never fetched "/", so holds no cookie
    status, body = c.post("/tasks", {"text": "put soda_can on r2c2"})
    assert status == 403
    assert body["code"] == "no_session"


def test_oversized_body_is_refused(client):
    status, body = client.post("/tasks", b'{"text": "' + b"x" * 9000 + b'"}')
    assert status == 413
    assert body["code"] == "body_too_large"


def test_malformed_json_is_refused_readably(client):
    status, body = client.post("/tasks", b"{not json")
    assert status == 400
    assert "not valid JSON" in body["error"]


def test_non_object_body_is_refused(client):
    status, body = client.post("/tasks", b'["a list"]')
    assert status == 400


def test_empty_text_creates_no_task(client):
    status, body = client.post("/tasks", {"text": "   "})
    assert status == 400
    assert body["code"] == "empty_text"


def test_post_to_an_unknown_path_is_404(client):
    assert client.post("/tasks/abc/detonate", {})[0] == 404
    assert client.post("/elsewhere", {})[0] == 404


def test_confirm_requires_an_integer_plan_version(client):
    status, task = client.post("/tasks", {"text": "put soda_can on r2c2"})
    task = client.settle(task["task_id"])
    status, body = client.post(f"/tasks/{task['task_id']}/confirm",
                               {"plan_id": task["proposal"]["plan_id"],
                                "plan_version": "2"})
    assert status == 400
    assert body["code"] == "bad_body"


# ---------------------------------------------------------------------------
# The worked example, end to end
# ---------------------------------------------------------------------------

def test_missing_bin_produces_clarification_and_no_motion(client):
    status, task = client.post("/tasks", {"text": "Put the recycle item in the bin."})
    assert status == 202
    assert task["state"] == "planning"

    task = client.settle(task["task_id"])
    assert task["state"] == "needs_clarification"
    assert "no bin or tray configured" in task["events"][-1]["text"]
    assert task["question_id"]
    assert task["proposal"] is None
    assert task["mode"] == "planning_only"


def test_explicit_command_proposes_then_confirms_without_motion(client):
    status, task = client.post("/tasks", {"text": "put soda_can on r2c2"})
    task = client.settle(task["task_id"])
    assert task["state"] == "awaiting_confirmation"

    p = task["proposal"]
    assert p["target_id"] == "soda_can"
    assert p["destination"] == "cell:r2c2"
    assert p["execution_mode"] == "planning_only"
    assert p["semantic_source"] == "scene_data"
    assert p["scene_name"] == "FWDCenterLabSivaPool"

    status, done = client.post(f"/tasks/{task['task_id']}/confirm",
                               {"plan_id": p["plan_id"],
                                "plan_version": p["plan_version"]})
    assert status == 200
    assert done["state"] == "confirmed_no_motion"
    assert done["detail"] == "Plan confirmed; no movement performed."


def test_a_reply_must_quote_the_current_question(client):
    status, task = client.post("/tasks", {"text": "Put the recycle item in the bin."})
    task = client.settle(task["task_id"])

    status, body = client.post(f"/tasks/{task['task_id']}/reply",
                               {"question_id": "stale", "text": "r2c2"})
    assert status == 409
    assert body["code"] == "stale_question"

    status, body = client.post(f"/tasks/{task['task_id']}/reply",
                               {"question_id": task["question_id"], "text": "r2c2"})
    assert status == 202


def test_duplicate_confirmation_is_refused(client):
    status, task = client.post("/tasks", {"text": "put soda_can on r2c2"})
    task = client.settle(task["task_id"])
    p = task["proposal"]
    body = {"plan_id": p["plan_id"], "plan_version": p["plan_version"]}
    assert client.post(f"/tasks/{task['task_id']}/confirm", body)[0] == 200
    status, err = client.post(f"/tasks/{task['task_id']}/confirm", body)
    assert status == 409
    assert err["code"] == "already_confirmed"


def test_another_session_cannot_see_or_confirm_the_task(server, client):
    status, task = client.post("/tasks", {"text": "put soda_can on r2c2"})
    task = client.settle(task["task_id"])

    other = Client(server).open_session()
    assert other.cookie != client.cookie
    assert other.get(f"/tasks/{task['task_id']}")[0] == 404
    status, body = other.post(
        f"/tasks/{task['task_id']}/confirm",
        {"plan_id": task["proposal"]["plan_id"],
         "plan_version": task["proposal"]["plan_version"]},
    )
    assert status == 404


def test_cancel_frees_the_session(client):
    status, task = client.post("/tasks", {"text": "put soda_can on r2c2"})
    tid = task["task_id"]
    status, done = client.post(f"/tasks/{tid}/cancel", {})
    assert status == 200 and done["state"] == "cancelled"
    assert client.post("/tasks", {"text": "put red_cube on r1c1"})[0] == 202


def test_a_retried_submission_does_not_open_two_tasks(client):
    body = {"text": "put soda_can on r2c2", "client_request_id": "retry-1"}
    a = client.post("/tasks", body)[1]
    b = client.post("/tasks", body)[1]
    assert a["task_id"] == b["task_id"]


def test_second_concurrent_task_is_refused(client):
    client.post("/tasks", {"text": "put soda_can on r2c2"})
    status, body = client.post("/tasks", {"text": "put red_cube on r1c1"})
    assert status == 409
    assert body["code"] == "task_in_progress"


def test_transcript_is_returned_as_text_not_markup(client):
    nasty = "put <img src=x onerror=alert(1)> on r2c2"
    status, task = client.post("/tasks", {"text": nasty})
    task = client.settle(task["task_id"])
    # Stored and echoed verbatim as data — no escaping, no stripping, no
    # interpretation.  The page is what renders it, and it does so with
    # textContent, so markup in a message can never become markup on the page.
    assert task["events"][0]["text"] == nasty
    assert nasty.split(" on ")[0][4:] in task["events"][-1]["text"]
    assert task["state"] == "needs_clarification"


# ---------------------------------------------------------------------------
# Helper units
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("origin,host,ok", [
    ("http://127.0.0.1:8080", "127.0.0.1:8080", True),
    ("https://box.local:8080", "box.local:8080", True),
    ("http://127.0.0.1:8081", "127.0.0.1:8080", False),
    ("http://evil.example", "127.0.0.1:8080", False),
    ("", "127.0.0.1:8080", False),
    ("null", "127.0.0.1:8080", False),
    ("http://127.0.0.1:8080", "", False),
    # A host that merely ends with the real one must not pass.
    ("http://evil.com/127.0.0.1:8080", "127.0.0.1:8080", False),
])
def test_origin_allowed(origin, host, ok):
    assert origin_allowed(origin, host) is ok


@pytest.mark.parametrize("header,expected", [
    (f"{SESSION_COOKIE}=abc123", "abc123"),
    (f"other=1; {SESSION_COOKIE}=abc123; more=2", "abc123"),
    ("other=1", ""),
    ("", ""),
    (f"{SESSION_COOKIE}=", ""),
    (f"{SESSION_COOKIE}=has spaces", ""),
    (f"{SESSION_COOKIE}=" + "x" * 200, ""),
])
def test_session_from_cookie(header, expected):
    assert session_from_cookie(header) == expected


def test_parse_body_limits_and_errors():
    assert parse_body(b'{"a": 1}') == {"a": 1}
    for raw in (b"", b"{bad", b"x" * (9 * 1024)):
        with pytest.raises(TaskError):
            parse_body(raw)
