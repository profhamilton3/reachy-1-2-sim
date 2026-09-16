"""Single source of truth for Stage 1 execution timing and base-tree
provenance constants. The notebook generator and `leg.sh`'s DUR both read
from here; the timing literals baked into the notebook's cell 1 are
compared against this module by `test_notebook_constants_match_plan`."""
from __future__ import annotations

import pathlib
import sys
from typing import Dict, NamedTuple, Tuple

#: Self-contained on purpose (project memory: "Test Path Gotcha" -- no
#: conftest.py, and a file can pass only via sys.path leakage from another
#: test module run earlier in the same session). This module now reads
#: `rig_routes` to derive route budgets, so it puts `src` on `sys.path`
#: itself rather than depending on a caller's PYTHONPATH -- mirroring
#: `scripts/e1_tail_check.py`'s own handling.
_SRC = pathlib.Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from reachy_ai.motion import rig_routes as _R  # noqa: E402

LEAD_IN_S = 3.0
COMPLIANCE_TIMEOUT_S = 3.0

#: The parked-tail window `e1_tail_check.check` requires at the END of a
#: leg's recording -- named here so it is a visible term in the budget
#: rather than folded invisibly into `MARGIN_S` (PR #120 review, M1/W3).
#: Must match `e1_tail_check.check`'s own default `window_s`; that
#: agreement is asserted by `test_parked_tail_matches_e1_tail_check_default`.
PARKED_TAIL_S = 3.0

#: Generic slack beyond every modelled term below (polling jitter in the
#: notebook's `wait_for`, log flushing, scheduling delays) -- NOT where the
#: parked tail or convergence retries live any more. Was 10.0 when it had
#: to cover the parked tail implicitly; now that PARKED_TAIL_S is its own
#: term, this is pure margin.
MARGIN_S = 5.0

# ── Route budgets, derived from the route tables, not observed ─────────────
#
# `ROUTE_BUDGET_S` used to be a single number (20.0) "rounded up" from
# `elapsed_s: 19.221` in a Stage 1 probe record. That record is an ABORT:
# the arm reached the second of twelve RAISE_TO_SIDE waypoints and stopped
# with `r_elbow_pitch` 39 deg off (PR #120 review, M1). It measures one
# waypoint's full retry-to-failure cost (GRIP_SHUT 3.5s + BACK's nominal 3.0s
# + `converge`'s six re-stream passes 0.6*(1..6)=12.6s = 19.1s), not a route.
# No route below has ever been flown to completion, so none of these numbers
# come from an observed flight -- they come from the waypoint chains
# themselves plus the runner's own retry/settle policy, read from the
# runners rather than re-typed as a second copy of their constants.
#
# Each budget = nominal waypoint time (`sum(wp.seconds)`)
#             + this runner's per-waypoint settle overhead (0 for
#               `primitives.fly`, which has none; `settle_s` * waypoint count
#               for `rig_motion.fly_route`, which sleeps after every one)
#             + ONE waypoint's worst-case convergence-retry allowance.
#
# THAT LAST TERM IS DELIBERATELY NOT WORST-CASE FOR EVERY WAYPOINT. Budgeting
# every waypoint's full retry schedule would run RAISE_TO_SIDE past three
# minutes; nothing observed justifies that, and nothing here claims it. This
# is an explicit, bounded, EXPERIMENTAL budget for "at most one bad waypoint
# per leg" -- not proof the route completes. If a real flight needs more
# than one waypoint's worth of retries, the recorder window can still run
# out before the route (or the required parked tail) finishes. That must
# show up as a failed `e1_tail_check` (`PARKED_AT_<T>=NO`) and be treated as
# an INCOMPLETE leg -- never reported as full-route coverage. `leg.sh`
# already enforces this in `end` mode (a failed tail check writes
# `control/stop`); this module and the notebook do not weaken that.

#: Which runner flies each route, so the settle/retry policy below is read
#: off the runner actually used (see `plan.CYCLES`), not guessed per route.
ROUTE_RUNNER: Dict[str, str] = {
    "RAISE_TO_SIDE": "primitives_fly",
    "LOWER_TO_REST": "rig_motion_fly_route",
    "PLACE_ROUTE": "rig_motion_fly_route",
    "LIFT_TO_PRESENT": "rig_motion_fly_route",
}

#: `primitives.fly` calls `primitives.converge(..., passes=6)` for every
#: waypoint (`src/reachy_ai/motion/primitives.py`, `fly()`); `converge`'s own
#: re-stream schedule is `0.6 * (1 + k)` for `k in range(passes)`. No sleep
#: between waypoints.
_PRIMITIVES_FLY_PASSES = 6
_PRIMITIVES_FLY_STEP_S = 0.6

#: `rig_motion.fly_route` (`src/reachy_ai/tasks/rig_motion.py`) defaults to
#: `retries=5` (so `range(retries + 1)` is 6 possible re-stream passes) at
#: `settle_pass_s * (1 + k)` each, and sleeps `settle_s` after every waypoint
#: regardless of whether it converged on the first pass.
_RIG_MOTION_FLY_ROUTE_PASSES = 6
_RIG_MOTION_FLY_ROUTE_STEP_S = 0.8
_RIG_MOTION_FLY_ROUTE_SETTLE_S = 0.3


def _one_waypoint_retry_worst_s(passes: int, step_s: float) -> float:
    """`step_s * (1 + k)` summed over `k in range(passes)` -- the full
    re-stream schedule a single waypoint runs through if it never converges
    (what a route stops on, at the waypoint's own tolerance, rather than
    what it silently absorbs)."""
    return step_s * sum(1 + k for k in range(passes))


#: One waypoint's worst-case retry cost, by runner.
RUNNER_RETRY_WORST_S: Dict[str, float] = {
    "primitives_fly": _one_waypoint_retry_worst_s(
        _PRIMITIVES_FLY_PASSES, _PRIMITIVES_FLY_STEP_S),
    "rig_motion_fly_route": _one_waypoint_retry_worst_s(
        _RIG_MOTION_FLY_ROUTE_PASSES, _RIG_MOTION_FLY_ROUTE_STEP_S),
}


def _route_waypoints(route_name: str) -> Tuple:
    return getattr(_R, route_name)


def _nominal_route_s(route_name: str) -> float:
    """`sum(wp.seconds)` over the route's own waypoint chain."""
    return sum(wp.seconds for wp in _route_waypoints(route_name))


def _settle_overhead_s(route_name: str) -> float:
    if ROUTE_RUNNER[route_name] == "rig_motion_fly_route":
        return _RIG_MOTION_FLY_ROUTE_SETTLE_S * len(_route_waypoints(route_name))
    return 0.0


def _route_budget_s(route_name: str) -> float:
    runner = ROUTE_RUNNER[route_name]
    return (_nominal_route_s(route_name) + _settle_overhead_s(route_name)
            + RUNNER_RETRY_WORST_S[runner])


#: Per-route motion budget, seconds -- see the derivation above.
ROUTE_BUDGET_S: Dict[str, float] = {
    name: _route_budget_s(name) for name in ROUTE_RUNNER
}

#: No route has ever been flown to completion (see the note above); this is
#: still all four, honestly, rather than crediting RAISE_TO_SIDE with a
#: flight it never finished.
UNFLOWN_ROUTES: Tuple[str, ...] = (
    "RAISE_TO_SIDE", "LOWER_TO_REST", "PLACE_ROUTE", "LIFT_TO_PRESENT")


class Leg(NamedTuple):
    name: str
    route: str
    tool: str
    desc: str
    check: str


CYCLES: Dict[str, Tuple[Leg, ...]] = {
    "S1a": (
        Leg("setup_a", "RAISE_TO_SIDE", "primitives.raise_to_side",
            "HOME->PRESENT", "PLACE_ROUTE_start"),
        Leg("flight_a", "LOWER_TO_REST", "rig_motion.from_present",
            "PRESENT->REST", "PRESENT"),
    ),
    "S1b": (
        Leg("flight_b", "PLACE_ROUTE", "rig_motion.deploy_to_rest",
            "HOME->REST", "PLACE_ROUTE_start"),
    ),
    "S1c": (
        Leg("setup_c", "PLACE_ROUTE", "rig_motion.deploy_to_rest",
            "HOME->REST", "PLACE_ROUTE_start"),
        Leg("flight_c", "LIFT_TO_PRESENT", "rig_motion.to_present",
            "REST->PRESENT", "REST"),
    ),
}

#: One call per tool name, matching what the previous generator (#117)
#: rendered into each motion cell.
TOOL_TO_CALL: Dict[str, str] = {
    "primitives.raise_to_side": "primitives.raise_to_side(reachy.r_arm)",
    "rig_motion.from_present": (
        "rig_motion.from_present(reachy.r_arm, "
        "on_phase=lambda *a: phases.append([time.monotonic_ns(), *map(str, a)]))"),
    "rig_motion.deploy_to_rest": (
        "rig_motion.deploy_to_rest(reachy.r_arm, "
        "on_phase=lambda *a: phases.append([time.monotonic_ns(), *map(str, a)]))"),
    "rig_motion.to_present": (
        "rig_motion.to_present(reachy.r_arm, "
        "on_phase=lambda *a: phases.append([time.monotonic_ns(), *map(str, a)]))"),
}

#: The sha this branch is actually built on, and the moment it landed -- the
#: binding check's floor for `manifest.started_at` (W4). A server tree at
#: `4d727c0` (PR #118's merge, the cmd_seq fix) alone is not enough: it has
#: no `code_sha` at all, because `native_mujoco/recorder.py` did not stamp
#: one until `3d7fc74` (this PR's own first commit, "fresh-server code
#: provenance in the run manifest"). Pointing this at `4d727c0` made the
#: binding cell's honest "manifest has no code_sha" refusal look like a
#: mystery STOP for exactly the servers PR #117's flow would have used (PR
#: #120 review, M2). `3d7fc74` is a descendant of `4d727c0`, so requiring it
#: still implies the cmd_seq fix.
REQUIRED_SHA = "3d7fc744f81eae14c39b70f22fab3c95fd02626d"
MERGE_TIME_ISO = "2026-09-16T02:20:18Z"


def leg_duration_s(route: str, *, lead_in_s: float = LEAD_IN_S,
                    compliance_timeout_s: float = COMPLIANCE_TIMEOUT_S) -> float:
    return (lead_in_s + compliance_timeout_s + ROUTE_BUDGET_S[route]
            + PARKED_TAIL_S + MARGIN_S)


def cycle_durations(cycle: str, *, lead_in_s: float = LEAD_IN_S,
                     compliance_timeout_s: float = COMPLIANCE_TIMEOUT_S) -> Dict[str, float]:
    return {leg.name: leg_duration_s(leg.route, lead_in_s=lead_in_s,
                                      compliance_timeout_s=compliance_timeout_s)
            for leg in CYCLES[cycle]}
