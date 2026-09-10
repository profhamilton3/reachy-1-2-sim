"""Same-origin HTTP task routes for the command panel (issue #49).

Thin on purpose.  Everything here is request shape — routing, ownership,
origin, body limits — and everything about what a task *means* lives in
tasks.TaskCoordinator.  Handlers do no planning: `POST /tasks` returns as soon
as the task is queued, so a slow planner never occupies the HTTP thread that
the MJPEG streams and `/status` polling share.

WHY ORIGIN IS CHECKED AND NOT CORS
----------------------------------
These routes are mutating and unauthenticated beyond a session cookie, and the
server is reachable from the browser on localhost.  There is no CORS header to
relax, so the only thing a cross-site page needs is for the POST to be
processed at all; a `fetch` from another origin still sends `Origin`, so
rejecting a mismatched one closes it.  A missing `Origin` on a mutating route
is rejected too rather than trusted.

WHY THE SESSION COOKIE IS HttpOnly
----------------------------------
The panel's JavaScript never needs to read it — the browser attaches it on its
own — and the brief asks that no credential be exposed to page script.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional, Tuple

from panel_planner import DeterministicPlanner
from tasks import Capabilities, TaskCoordinator, TaskError

#: Cookie the panel is identified by.  Not a security boundary against a user
#: who controls the browser — it separates concurrent panels, and stops one
#: session confirming another's plan.
SESSION_COOKIE = "reachy_panel_session"

#: Largest task request body accepted.  A command is ~2 KB of text at most.
MAX_BODY_BYTES = 8 * 1024

Response = Tuple[int, Dict[str, Any]]


class PanelRoutes:
    """Routing table for `/capabilities` and `/tasks...`."""

    def __init__(self, scene_provider: Callable[[], Any],
                 capabilities: Optional[Capabilities] = None) -> None:
        self.capabilities = capabilities or Capabilities()
        self.coordinator = TaskCoordinator(
            DeterministicPlanner(scene_provider), capabilities=self.capabilities
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _task_id(path: str, suffix: str = "") -> Optional[str]:
        """Extract the id from /tasks/<id> or /tasks/<id>/<suffix>."""
        parts = [p for p in path.strip("/").split("/") if p]
        if not parts or parts[0] != "tasks":
            return None
        if suffix:
            if len(parts) != 3 or parts[2] != suffix:
                return None
        elif len(parts) != 2:
            return None
        return parts[1]

    # -- GET ---------------------------------------------------------------

    def handle_get(self, path: str, session_id: str) -> Optional[Response]:
        if path == "/capabilities":
            return 200, {"capabilities": self.capabilities.as_dict()}

        task_id = self._task_id(path)
        if task_id is None:
            return None
        try:
            task = self.coordinator.get(session_id, task_id)
        except TaskError as exc:
            return _error(exc)
        return 200, task.as_dict(self.capabilities)

    # -- POST --------------------------------------------------------------

    def handle_post(self, path: str, session_id: str,
                    payload: Any) -> Optional[Response]:
        if not isinstance(payload, dict):
            return 400, {"error": "Expected a JSON object.", "code": "bad_body"}

        try:
            if path == "/tasks":
                task = self.coordinator.submit(
                    session_id,
                    payload.get("text", ""),
                    str(payload.get("client_request_id", ""))[:64],
                )
                return 202, task.as_dict(self.capabilities)

            task_id = self._task_id(path, "reply")
            if task_id:
                task = self.coordinator.reply(
                    session_id, task_id,
                    str(payload.get("question_id", "")),
                    payload.get("text", ""),
                )
                return 202, task.as_dict(self.capabilities)

            task_id = self._task_id(path, "confirm")
            if task_id:
                version = payload.get("plan_version")
                if not isinstance(version, int) or isinstance(version, bool):
                    return 400, {"error": "plan_version must be an integer.",
                                 "code": "bad_body"}
                task = self.coordinator.confirm(
                    session_id, task_id, str(payload.get("plan_id", "")), version
                )
                return 200, task.as_dict(self.capabilities)

            task_id = self._task_id(path, "cancel")
            if task_id:
                task = self.coordinator.cancel(session_id, task_id)
                return 200, task.as_dict(self.capabilities)
        except TaskError as exc:
            return _error(exc)

        return None

    def shutdown(self) -> None:
        self.coordinator.shutdown()


def _error(exc: TaskError) -> Response:
    return exc.status, {"error": exc.message, "code": exc.code}


def parse_body(raw: bytes) -> Any:
    """Decode a request body, or raise TaskError with a readable message."""
    if len(raw) > MAX_BODY_BYTES:
        raise TaskError("That request is too large.", 413, "body_too_large")
    if not raw:
        raise TaskError("Expected a JSON body.", 400, "bad_body")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise TaskError("That request body is not valid JSON.", 400, "bad_body")


def origin_allowed(origin: str, host: str) -> bool:
    """True when a mutating request came from this very page.

    Compared against the request's own Host header rather than a configured
    allowlist: the page is served on whatever host:port the operator published
    the container on, and hard-coding localhost would break the documented
    "page and sim on different machines" case for no gain.
    """
    if not origin or not host:
        return False
    for scheme in ("http://", "https://"):
        if origin.startswith(scheme):
            return origin[len(scheme):] == host
    return False


def session_from_cookie(cookie_header: str) -> str:
    """Read our session cookie out of a Cookie header, ignoring the rest."""
    if not cookie_header:
        return ""
    for part in cookie_header.split(";"):
        name, _, value = part.strip().partition("=")
        if name == SESSION_COOKIE:
            # Only our own opaque token shape is accepted; anything else is
            # treated as no session rather than trusted as an id.
            value = value.strip()
            if value and len(value) <= 128 and all(
                c.isalnum() or c in "-_" for c in value
            ):
                return value
    return ""
