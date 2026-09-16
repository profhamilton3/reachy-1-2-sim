"""Single source of truth for Stage 1 execution timing and base-tree
provenance constants. The notebook generator and `leg.sh`'s DUR both read
from here; the timing literals baked into the notebook's cell 1 are
compared against this module by `test_notebook_constants_match_plan`."""
from __future__ import annotations

from typing import Dict, NamedTuple, Tuple

LEAD_IN_S = 3.0
COMPLIANCE_TIMEOUT_S = 3.0

#: Slack on top of LEAD_IN_S + COMPLIANCE_TIMEOUT_S + ROUTE_BUDGET_S so a
#: legitimate slow-apply attempt never runs the route into the recorder's
#: tail (W3, PR #118 re-review).
MARGIN_S = 10.0

#: Per-route motion budget, seconds. Only RAISE_TO_SIDE was actually
#: attempted in Stage 1 -- docs/reviews/probes-2026-09-15-e1-stage1-b4/
#: control/setup_a_done, elapsed_s=19.221186666 before the abort at BACK
#: (read-only; that tree is untouched by this branch). LOWER_TO_REST,
#: PLACE_ROUTE and LIFT_TO_PRESENT were never flown. Per the assignment
#: ("if a route was never flown, use the largest flown one and say so"),
#: all three share RAISE_TO_SIDE's observed budget, rounded up to 20s.
ROUTE_BUDGET_S: Dict[str, float] = {
    "RAISE_TO_SIDE": 20.0,
    "LOWER_TO_REST": 20.0,
    "PLACE_ROUTE": 20.0,
    "LIFT_TO_PRESENT": 20.0,
}
UNFLOWN_ROUTES: Tuple[str, ...] = ("LOWER_TO_REST", "PLACE_ROUTE", "LIFT_TO_PRESENT")


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

#: The sha this branch is built on (PR #118's merge to main) and the moment
#: it landed -- the binding check's floor for `manifest.started_at` (W4): a
#: server started before this instant cannot have a3ee2c9's cmd_seq fix.
REQUIRED_SHA = "4d727c02b1e65d6b42bc2819c2ec7f6dc40daae6"
MERGE_TIME_ISO = "2026-09-16T01:35:22Z"


def leg_duration_s(route: str, *, lead_in_s: float = LEAD_IN_S,
                    compliance_timeout_s: float = COMPLIANCE_TIMEOUT_S) -> float:
    return lead_in_s + compliance_timeout_s + ROUTE_BUDGET_S[route] + MARGIN_S


def cycle_durations(cycle: str, *, lead_in_s: float = LEAD_IN_S,
                     compliance_timeout_s: float = COMPLIANCE_TIMEOUT_S) -> Dict[str, float]:
    return {leg.name: leg_duration_s(leg.route, lead_in_s=lead_in_s,
                                      compliance_timeout_s=compliance_timeout_s)
            for leg in CYCLES[cycle]}
