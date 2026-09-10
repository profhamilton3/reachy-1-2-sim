"""Task coordinator for the browser panel's "Talk to Reachy" card (issue #49).

Owns the conversation and task lifecycle that sits behind the same-origin HTTP
routes in camera_server.py.  Deliberately stdlib-only: this module runs inside
the Docker compatibility container next to the camera page, which the
container's supervisord starts with a bare `python3` and no extra wheels.

WHAT THIS IS NOT
----------------
It does not move the robot and it does not talk to the simulator.  Stage 1 is
planning-only by construction: `confirm` lands a task in `confirmed_no_motion`
and nothing else happens.  A live executor arrives in issue #51 and announces
itself through `Capabilities`, so the browser only ever shows execution
controls that something can actually service.

WHY A GENERATION COUNTER
------------------------
Planning runs on a worker thread and a planner can be slow or wedged.  A reply,
a cancel, or a planning timeout therefore has to be able to abandon an
in-flight job whose thread cannot be killed.  Each of those bumps
`Task.generation`, and a worker result is applied only when the generation it
captured at submit time still matches.  A late answer to a superseded question
is dropped rather than reviving a cancelled task or overwriting a newer one.

WHY VERSIONS ARE SEPARATE FROM GENERATIONS
------------------------------------------
`generation` guards worker results; `version` guards the browser.  Every state
transition bumps `version`, and a proposal records the version it was built at
as `plan_version`.  Confirming quotes both the plan id and that version, so a
Confirm click that was rendered before some other change cannot be applied
afterwards.  Two ids because they answer different questions: one is "is this
worker's answer still wanted", the other is "is the button the human clicked
still describing the world".
"""

from __future__ import annotations

import secrets
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Limits.  All bounded on purpose — an unbounded queue or transcript in a
# long-lived container is a slow leak nobody watches.
# ---------------------------------------------------------------------------

MAX_TEXT_CHARS = 2000          # one message; the HTTP layer caps the body too
MAX_EVENTS = 40                # transcript entries retained per task
MAX_TASKS = 200                # total tasks retained before oldest are dropped
PLANNING_TIMEOUT_S = 20.0      # a planner slower than this is declared failed
TASK_TTL_S = 30 * 60           # idle task lifetime before it expires
MAX_QUEUED_JOBS = 8            # backpressure: refuse rather than pile up


class TaskState(str, Enum):
    planning = "planning"
    needs_clarification = "needs_clarification"
    awaiting_confirmation = "awaiting_confirmation"
    confirmed_no_motion = "confirmed_no_motion"
    executing = "executing"
    completed = "completed"
    cancelled = "cancelled"
    failed = "failed"
    expired = "expired"


#: States from which no further transition is possible.
TERMINAL_STATES = frozenset({
    TaskState.confirmed_no_motion,
    TaskState.completed,
    TaskState.cancelled,
    TaskState.failed,
    TaskState.expired,
})

#: States that still count against a session's one-active-task budget.
ACTIVE_STATES = frozenset({
    TaskState.planning,
    TaskState.needs_clarification,
    TaskState.awaiting_confirmation,
    TaskState.executing,
})


class TaskError(Exception):
    """A request the coordinator refuses, carrying the HTTP status to use."""

    def __init__(self, message: str, status: int = 400, code: str = "bad_request"):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code


# ---------------------------------------------------------------------------
# Conversation and proposals
# ---------------------------------------------------------------------------

@dataclass
class ConversationEvent:
    """One turn in the transcript.  `text` is always plain text, never markup."""

    role: str                  # "user" | "reachy"
    text: str
    at: float = field(default_factory=time.time)
    # Set on a reachy turn that asks something.  A reply must quote it, so an
    # answer typed against a stale question cannot be applied to a new one.
    question_id: str = ""
    choices: List[str] = field(default_factory=list)
    #: Which slot this question is about — "which_cell", "which_object".  The
    #: planner knows this at the moment it asks and used to throw it away, then
    #: re-derive it from the answer's wording on the next turn.  Recording it
    #: is what stops an answer landing in the slot nobody asked about (#59).
    slot: str = ""

    def as_dict(self) -> dict:
        d = {"role": self.role, "text": self.text, "at": round(self.at, 3)}
        if self.question_id:
            d["question_id"] = self.question_id
            d["choices"] = list(self.choices)
            if self.slot:
                d["slot"] = self.slot
        return d


@dataclass
class Proposal:
    """A validated, confirmable action.

    `destination` is the reference string (`cell:r2c2`, `destination:left_tray`)
    and `legacy_destination` is the IITG `Destination` enum value when — and
    only when — one genuinely exists.  None there is a real answer, not a gap
    to fill: see DestinationRef.legacy_destination in panel_scene.py.

    `state_evidence` is whatever the planner needs in order to decide later
    whether this plan still describes the world.  The coordinator never reads
    it; it only hands it back to the revalidator at confirm time.  Keeping the
    shape out of here is what lets the scene layer change its mind about what
    evidence is worth recording without touching the state machine.
    """

    plan_id: str
    plan_version: int
    task_type: str
    #: Absent on the abilities that genuinely have neither.  None here is a
    #: real answer, the same way `legacy_destination` None is — a wave has no
    #: target and no destination, and writing "" or "__none__" into these
    #: would put a value meaning ABSENT into fields whose readers (the live
    #: revalidator, the executor's cell lookup, the browser card) all assume
    #: PRESENT.  `__post_init__` holds pick_place to both.
    target_id: Optional[str] = None
    destination: Optional[str] = None
    brief_reason: str = ""
    #: Which arm.  Always known, never guessed: the validated corridor is
    #: right-arm geometry through a rig that is not symmetric.
    arm: str = "right"
    #: The cell an ability points AT — not a destination to place into, so it
    #: does not inherit pick-and-place's "must be empty" rule.
    cell: Optional[str] = None
    #: The object an ability points AT, distinct from a pick target.
    object_id: Optional[str] = None
    #: Which measured motion this plan flies, and which version of it.
    route: str = ""
    route_version: int = 0
    #: The posture the route may be entered from, so the executor can refuse an
    #: unsupported start rather than discover it in flight.
    expected_start_posture: str = ""
    requires_confirmation: bool = True
    execution_mode: str = "planning_only"
    semantic_source: str = "scene_data"
    scene_name: str = ""
    summary: str = ""
    destination_kind: str = ""
    destination_label: str = ""
    legacy_destination: Optional[str] = None
    state_evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A pick-and-place plan without both of these is not a plan, and the
        # cheapest place to notice is here rather than in the executor with the
        # lease held.
        if self.task_type == "pick_place" and not (self.target_id
                                                   and self.destination):
            raise ValueError(
                "a pick_place proposal needs a target_id and a destination"
            )

    def as_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "plan_version": self.plan_version,
            "task_type": self.task_type,
            "target_id": self.target_id,
            "destination": self.destination,
            "arm": self.arm,
            "cell": self.cell,
            "object_id": self.object_id,
            "route": self.route,
            "route_version": self.route_version,
            "expected_start_posture": self.expected_start_posture,
            "destination_kind": self.destination_kind,
            "destination_label": self.destination_label,
            "legacy_destination": self.legacy_destination,
            "brief_reason": self.brief_reason,
            "requires_confirmation": self.requires_confirmation,
            "execution_mode": self.execution_mode,
            "semantic_source": self.semantic_source,
            "scene_name": self.scene_name,
            "summary": self.summary,
            "state_evidence": dict(self.state_evidence),
        }


@dataclass
class PlannerOutcome:
    """What a planner may return.  Anything else is a programming error."""

    kind: str      # "clarification" | "proposal" | "reply" | "unsupported"
    message: str = ""
    choices: List[str] = field(default_factory=list)
    proposal: Optional[Proposal] = None
    #: On a clarification, the slot being asked about.  Carried onto the
    #: transcript event so the next turn can fill that slot instead of
    #: guessing which one the answer was for.
    slot: str = ""


@dataclass
class PlannerRequest:
    text: str
    history: List[ConversationEvent]


Planner = Callable[[PlannerRequest], PlannerOutcome]


@dataclass
class Capabilities:
    """What the server can actually do, so the UI never offers more.

    `live_execution` stays False until issue #51 lands a validated executor.
    The panel keys its Confirm-button wording off this rather than assuming.
    """

    live_execution: bool = False
    execution_mode: str = "planning_only"
    semantic_source: str = "scene_data"
    max_text_chars: int = MAX_TEXT_CHARS

    def as_dict(self) -> dict:
        return {
            "live_execution": self.live_execution,
            "execution_mode": self.execution_mode,
            "semantic_source": self.semantic_source,
            "max_text_chars": self.max_text_chars,
        }


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id: str
    session_id: str
    state: TaskState = TaskState.planning
    version: int = 1
    generation: int = 1
    events: List[ConversationEvent] = field(default_factory=list)
    proposal: Optional[Proposal] = None
    # Set once a confirmation has been consumed, so a second one is refused
    # rather than quietly re-applied.
    confirmed_plan_id: str = ""
    #: Set by cancel() while executing.  The executor reads it at a phase
    #: boundary; the task is not cancelled until the executor says it stopped.
    cancel_requested: bool = False
    #: What the executor observed, when one ran.  Empty otherwise — it is
    #: evidence, so it exists only when something actually looked.
    execution_evidence: Dict[str, Any] = field(default_factory=dict)
    question_id: str = ""
    client_request_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    deadline: float = 0.0
    detail: str = ""

    def touch(self) -> None:
        self.version += 1
        self.updated_at = time.time()

    def add_event(self, ev: ConversationEvent) -> None:
        self.events.append(ev)
        while len(self.events) > MAX_EVENTS:
            self.events.pop(0)

    def as_dict(self, caps: Capabilities) -> dict:
        return {
            "task_id": self.task_id,
            "state": self.state.value,
            "version": self.version,
            "detail": self.detail,
            "question_id": self.question_id,
            "events": [e.as_dict() for e in self.events],
            "proposal": self.proposal.as_dict() if self.proposal else None,
            "execution_evidence": dict(self.execution_evidence),
            "cancel_requested": self.cancel_requested,
            "mode": caps.execution_mode,
            "capabilities": caps.as_dict(),
        }


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class TaskCoordinator:
    """Owns tasks, sessions, and the worker pool that runs planning.

    Every public method takes the caller's `session_id` and refuses to touch a
    task owned by anyone else — with 404 rather than 403, so one session cannot
    probe another's task ids.
    """

    def __init__(
        self,
        planner: Planner,
        *,
        capabilities: Optional[Capabilities] = None,
        revalidate: Optional[Callable[[Proposal], Tuple[bool, str]]] = None,
        executor: Any = None,
        aside: Optional[Callable[[str], Optional["PlannerOutcome"]]] = None,
        max_workers: int = 2,
        planning_timeout_s: float = PLANNING_TIMEOUT_S,
        task_ttl_s: float = TASK_TTL_S,
    ) -> None:
        self._planner = planner
        # Called under the lock at confirm time, before the confirmation is
        # consumed.  A plan is only worth confirming if it still describes the
        # world, and the world moves on its own here.
        self._revalidate = revalidate
        # Whatever can actually move the robot, or None.  The coordinator never
        # decides that a thing is executable — it asks.
        self._executor = executor
        # Answers a message that needs no task at all, or None if it is not one
        # of those.  Called OUTSIDE the lock and required to be fast and
        # side-effect free: it decides whether "Hello" should disturb the
        # session's one active task, so it must not itself be able to.
        self._aside = aside
        self._caps = capabilities or Capabilities()
        self._timeout = planning_timeout_s
        self._ttl = task_ttl_s
        self._lock = threading.RLock()
        self._tasks: Dict[str, Task] = {}
        self._by_session: Dict[str, str] = {}          # session -> active task id
        self._by_client_req: Dict[str, str] = {}       # session+crid -> task id
        self._queued = 0
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="panel-planner"
        )

    # -- lifecycle ---------------------------------------------------------

    @property
    def capabilities(self) -> Capabilities:
        return self._caps

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)

    @staticmethod
    def new_session_id() -> str:
        return secrets.token_urlsafe(24)

    # -- public API --------------------------------------------------------

    def submit(self, session_id: str, text: str, client_request_id: str = "") -> Task:
        """Validate, create a task, and queue planning.  Never blocks on the planner."""
        text = _clean_text(text)
        if not text:
            raise TaskError("Type a command first.", 400, "empty_text")
        if len(text) > MAX_TEXT_CHARS:
            raise TaskError(
                f"That message is {len(text)} characters; the limit is {MAX_TEXT_CHARS}.",
                413, "text_too_long",
            )

        # Before anything is created or any lock is taken: is this a message
        # that needs no task?  A greeting has nothing to confirm, nothing to
        # clarify and nothing to move, so making it the session's one active
        # task would put it in the way of real work — and typing "Hello" while
        # a plan is awaiting confirmation would either be refused with "one
        # task at a time" or, worse, replace the plan.
        aside = self._aside(text) if self._aside is not None else None
        if aside is not None and aside.kind == "reply":
            return self._aside_task_locked(session_id, text, aside,
                                           client_request_id)

        with self._lock:
            self._sweep_locked()

            # A retried POST (double click, flaky network) must not open a
            # second task.  Same session + same client id returns the original.
            crid_key = f"{session_id}:{client_request_id}"
            if client_request_id and crid_key in self._by_client_req:
                existing = self._tasks.get(self._by_client_req[crid_key])
                if existing is not None:
                    return existing

            active_id = self._by_session.get(session_id)
            if active_id:
                active = self._tasks.get(active_id)
                if active is not None and active.state in ACTIVE_STATES:
                    raise TaskError(
                        "One task at a time. Finish or cancel the current one first.",
                        409, "task_in_progress",
                    )

            if self._queued >= MAX_QUEUED_JOBS:
                raise TaskError(
                    "The planner is busy. Try again in a moment.", 503, "planner_busy"
                )

            task = Task(task_id=uuid.uuid4().hex, session_id=session_id)
            task.client_request_id = client_request_id
            task.add_event(ConversationEvent(role="user", text=text))
            task.deadline = time.time() + self._timeout
            self._tasks[task.task_id] = task
            self._by_session[session_id] = task.task_id
            if client_request_id:
                self._by_client_req[crid_key] = task.task_id
            self._trim_locked()
            self._queue_locked(task, text)
            return task

    def get(self, session_id: str, task_id: str) -> Task:
        with self._lock:
            self._sweep_locked()
            return self._owned_locked(session_id, task_id)

    def reply(self, session_id: str, task_id: str, question_id: str, text: str) -> Task:
        """Answer the *current* question.  A stale question_id is refused."""
        text = _clean_text(text)
        if not text:
            raise TaskError("Type an answer first.", 400, "empty_text")
        if len(text) > MAX_TEXT_CHARS:
            raise TaskError(
                f"That message is {len(text)} characters; the limit is {MAX_TEXT_CHARS}.",
                413, "text_too_long",
            )

        with self._lock:
            self._sweep_locked()
            task = self._owned_locked(session_id, task_id)
            if task.state is not TaskState.needs_clarification:
                raise TaskError(
                    f"This task is not waiting for an answer (state: {task.state.value}).",
                    409, "not_awaiting_reply",
                )
            if question_id != task.question_id:
                raise TaskError(
                    "That answer was for an earlier question. Read the latest one and answer again.",
                    409, "stale_question",
                )

            task.add_event(ConversationEvent(role="user", text=text))
            task.state = TaskState.planning
            task.question_id = ""
            task.detail = ""
            task.generation += 1          # abandon anything still in flight
            task.deadline = time.time() + self._timeout
            task.touch()
            self._queue_locked(task, text)
            return task

    def confirm(self, session_id: str, task_id: str, plan_id: str, plan_version: int) -> Task:
        """Consume a confirmation exactly once, for exactly this plan."""
        with self._lock:
            self._sweep_locked()
            task = self._owned_locked(session_id, task_id)

            if task.confirmed_plan_id:
                raise TaskError(
                    "This plan was already confirmed.", 409, "already_confirmed"
                )
            if task.state is not TaskState.awaiting_confirmation or task.proposal is None:
                raise TaskError(
                    f"There is no plan awaiting confirmation (state: {task.state.value}).",
                    409, "not_awaiting_confirmation",
                )
            if plan_id != task.proposal.plan_id:
                raise TaskError(
                    "That confirmation is for a different plan.", 409, "plan_mismatch"
                )
            if plan_version != task.proposal.plan_version:
                raise TaskError(
                    "The plan changed since that button was drawn. Review the current plan and confirm again.",
                    409, "stale_plan_version",
                )

            # Last gate before the confirmation is spent.  Everything above
            # checks that the human confirmed the plan we think they saw; this
            # checks that the plan still matches the world.
            if self._revalidate is not None:
                ok, why = self._revalidate(task.proposal)
                if not ok:
                    task.state = TaskState.failed
                    task.detail = why or "The plan no longer matches the scene."
                    task.proposal = None
                    task.generation += 1
                    task.add_event(ConversationEvent(role="reachy", text=task.detail))
                    task.touch()
                    self._release_locked(task)
                    raise TaskError(task.detail, 409, "plan_invalidated")

            task.confirmed_plan_id = plan_id
            task.generation += 1

            # Whether anything can move is the executor's answer, not a flag
            # set here.  Asking it per proposal is what lets the panel say
            # "nothing can pick from off the board" instead of the useless
            # "execution unavailable".
            can_execute, why_not = (
                self._executor.available(task.proposal) if self._executor
                else (False, "no execution adapter is installed")
            )

            if not can_execute:
                task.state = TaskState.confirmed_no_motion
                # This sentence is load-bearing and exact: it is the panel's
                # promise that confirming moved nothing.  The reason goes in
                # its own turn rather than blurring that line.
                task.detail = "Plan confirmed; no movement performed."
                task.add_event(ConversationEvent(
                    role="reachy", text="Plan confirmed; no movement performed."
                ))
                if why_not:
                    task.add_event(ConversationEvent(
                        role="reachy", text=f"I did not move because {why_not}."
                    ))
                task.touch()
                self._release_locked(task)
                return task

            task.state = TaskState.executing
            task.detail = "Executing in simulation."
            task.add_event(ConversationEvent(
                role="reachy", text="Executing in simulation."
            ))
            self._queued += 1
            self._pool.submit(self._run_executor, task.task_id, task.generation,
                              task.proposal)

            # NOT released here: an executing task is still the session's
            # active one, so a second command cannot be submitted while the
            # arm is moving.  The worker releases it when the motion lands.
            task.touch()
            return task

    def cancel(self, session_id: str, task_id: str) -> Task:
        with self._lock:
            self._sweep_locked()
            task = self._owned_locked(session_id, task_id)
            if task.state in TERMINAL_STATES:
                return task
            if task.state is TaskState.executing:
                # The arm is moving.  Ask it to stop, and say so — but do NOT
                # call this cancelled yet.  The executor stops at the next
                # phase boundary and parks the arm; only when it acknowledges
                # that does the task become cancelled.  Claiming a motion has
                # stopped before the thing doing it agrees is the failure mode
                # the brief names outright.
                if not task.cancel_requested:
                    task.cancel_requested = True
                    task.detail = "Stopping; waiting for the arm to come to rest."
                    task.add_event(ConversationEvent(
                        role="reachy",
                        text="Stopping; waiting for the arm to come to rest.",
                    ))
                    task.touch()
                return task
            task.state = TaskState.cancelled
            task.detail = "Cancelled."
            task.generation += 1                     # abandon in-flight planning
            task.proposal = None
            task.question_id = ""
            task.add_event(ConversationEvent(role="reachy", text="Cancelled."))
            task.touch()
            self._release_locked(task)
            return task

    # -- internals ---------------------------------------------------------

    def _aside_task_locked(self, session_id: str, text: str,
                           outcome: "PlannerOutcome",
                           client_request_id: str = "") -> Task:
        """A finished task that was never the session's active one.

        It exists so the page has something to render the exchange into, and
        it is deliberately absent from `_by_session`: the session's active task
        — a pending clarification, a plan awaiting confirmation, an arm in
        motion — is not touched, not re-versioned, and not replaced.

        It IS in `_by_client_req`, though.  Skipping that was a real gap: a
        double-clicked "Hello", or a POST retried over a flaky link, put two
        greetings in the transcript, which is exactly the invariant the
        client-request id was added to hold.  Not being the active task and
        not being idempotent are unrelated properties, and this needs both.
        """
        with self._lock:
            self._sweep_locked()
            crid_key = f"{session_id}:{client_request_id}"
            if client_request_id and crid_key in self._by_client_req:
                existing = self._tasks.get(self._by_client_req[crid_key])
                if existing is not None:
                    return existing

            task = Task(task_id=uuid.uuid4().hex, session_id=session_id)
            task.client_request_id = client_request_id
            task.deadline = time.time() + self._ttl
            task.add_event(ConversationEvent(role="user", text=text))
            task.add_event(ConversationEvent(role="reachy", text=outcome.message))
            task.state = TaskState.completed
            task.detail = outcome.message
            task.touch()
            self._tasks[task.task_id] = task
            if client_request_id:
                self._by_client_req[crid_key] = task.task_id
            self._trim_locked()
            return task

    def _owned_locked(self, session_id: str, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        # 404 for both "gone" and "someone else's": a different status would
        # let one session confirm the existence of another's task.
        if task is None or task.session_id != session_id:
            raise TaskError("No such task.", 404, "not_found")
        return task

    def _release_locked(self, task: Task) -> None:
        if self._by_session.get(task.session_id) == task.task_id:
            del self._by_session[task.session_id]

    def _queue_locked(self, task: Task, text: str) -> None:
        gen = task.generation
        history = list(task.events)
        self._queued += 1
        self._pool.submit(self._run_planner, task.task_id, gen, text, history)

    def _run_planner(self, task_id: str, gen: int, text: str, history: List[ConversationEvent]) -> None:
        """Worker thread.  Runs the planner off the request path, then applies
        the result only if it is still wanted."""
        try:
            outcome = self._planner(PlannerRequest(text=text, history=history))
        except Exception as exc:                     # a planner must not kill the pool
            outcome = PlannerOutcome(
                kind="unsupported",
                message=f"The planner failed: {exc.__class__.__name__}.",
            )
        finally:
            with self._lock:
                self._queued = max(0, self._queued - 1)

        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.generation != gen:
                return                               # superseded, cancelled, or timed out
            if task.state is not TaskState.planning:
                return
            self._apply_locked(task, outcome)

    def _run_executor(self, task_id: str, gen: int, proposal: Proposal) -> None:
        """Worker thread: run the motion, then record what actually happened.

        Progress is written to `detail` rather than added as transcript turns:
        a pick-and-place reports a dozen phases, and a conversation that filled
        up with them would bury the exchange that started it.
        """
        def should_cancel() -> bool:
            with self._lock:
                task = self._tasks.get(task_id)
                return bool(task and task.cancel_requested)

        def on_phase(name: str) -> None:
            with self._lock:
                task = self._tasks.get(task_id)
                if task is not None and task.generation == gen:
                    task.detail = f"Executing: {name}."
                    task.touch()

        try:
            result = self._executor.execute(
                proposal, should_cancel=should_cancel, on_phase=on_phase
            )
        except Exception as exc:                     # never kill the pool
            result = _failed_result(
                f"the executor raised {exc.__class__.__name__}: {exc}"
            )
        finally:
            with self._lock:
                self._queued = max(0, self._queued - 1)

        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.generation != gen:
                return
            status = getattr(result, "status", "failed")
            detail = getattr(result, "detail", "") or ""
            # Only these three, and only from the executor's own report.  A
            # task never becomes completed because the trajectory finished.
            task.state = {
                "completed": TaskState.completed,
                "cancelled": TaskState.cancelled,
            }.get(status, TaskState.failed)
            task.detail = detail or task.state.value
            task.execution_evidence = dict(getattr(result, "evidence", None) or {})
            task.add_event(ConversationEvent(role="reachy", text=task.detail))
            task.touch()
            self._release_locked(task)

    def _apply_locked(self, task: Task, outcome: PlannerOutcome) -> None:
        if outcome.kind == "clarification":
            task.state = TaskState.needs_clarification
            task.question_id = f"{task.task_id}:{task.version}"
            task.detail = ""
            task.add_event(ConversationEvent(
                role="reachy",
                text=outcome.message,
                question_id=task.question_id,
                choices=list(outcome.choices),
                slot=outcome.slot,
            ))
        elif outcome.kind == "reply":
            # A successful answer with nothing to do.  Terminal on arrival: no
            # confirmation to seek, no executor to ask, no lease to take.  It
            # is a distinct kind rather than a friendly `unsupported` because
            # the panel shows failures in red, and "Hello" is not a failure.
            task.state = TaskState.completed
            task.detail = outcome.message
            task.add_event(ConversationEvent(role="reachy", text=outcome.message))
            self._release_locked(task)
        elif outcome.kind == "proposal" and outcome.proposal is not None:
            proposal = outcome.proposal
            proposal.plan_version = task.version
            # Ask the executor about THIS plan, not the configured default.
            # It is the same question confirm() will ask, so the card cannot
            # promise motion that Confirm then declines to perform — or say
            # "no motion" on a plan that is about to move the arm.
            can_execute = (
                self._executor.available(proposal)[0] if self._executor else False
            )
            proposal.execution_mode = (
                "live_simulation" if can_execute else "planning_only"
            )
            task.proposal = proposal
            task.state = TaskState.awaiting_confirmation
            task.question_id = ""
            task.detail = ""
            if outcome.message:
                task.add_event(ConversationEvent(role="reachy", text=outcome.message))
        else:
            task.state = TaskState.failed
            task.detail = outcome.message or "That request is not supported."
            task.add_event(ConversationEvent(role="reachy", text=task.detail))
            self._release_locked(task)
        task.touch()

    def _sweep_locked(self) -> None:
        """Fail timed-out planning and expire idle tasks.

        Called on every request rather than from a timer thread: the container
        has few clients, and a sweep that only runs when someone is looking is
        one less thread to shut down cleanly.
        """
        now = time.time()
        for task in self._tasks.values():
            if task.state is TaskState.planning and task.deadline and now > task.deadline:
                task.state = TaskState.failed
                task.detail = "The planner did not answer in time."
                task.generation += 1                 # discard the late result
                task.add_event(ConversationEvent(role="reachy", text=task.detail))
                task.touch()
                self._release_locked(task)
            elif task.state not in TERMINAL_STATES and now - task.updated_at > self._ttl:
                task.state = TaskState.expired
                task.detail = "This task expired after being idle."
                task.generation += 1
                task.touch()
                self._release_locked(task)

    def _trim_locked(self) -> None:
        if len(self._tasks) <= MAX_TASKS:
            return
        for tid, _ in sorted(self._tasks.items(), key=lambda kv: kv[1].updated_at):
            if len(self._tasks) <= MAX_TASKS:
                break
            task = self._tasks[tid]
            if task.state in ACTIVE_STATES:
                continue
            del self._tasks[tid]
            self._by_client_req = {
                k: v for k, v in self._by_client_req.items() if v != tid
            }


class _FailedResult:
    """Stand-in when the executor itself raised, so the caller still gets a shape."""

    status = "failed"
    evidence: Dict[str, Any] = {}

    def __init__(self, detail: str) -> None:
        self.detail = detail


def _failed_result(detail: str) -> _FailedResult:
    return _FailedResult(detail)


def _clean_text(text: Any) -> str:
    """Trim, and drop control characters that would corrupt a transcript line.

    Tabs and newlines survive because the input is multiline on purpose.
    Nothing here is HTML escaping — the browser renders every turn with
    textContent, so escaping at this layer would only double-encode.
    """
    if not isinstance(text, str):
        return ""
    kept = [c for c in text if c in "\t\n" or ord(c) >= 32]
    return "".join(kept).strip()
