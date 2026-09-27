"""Initial-state comparison (plan §5 P4/P6, §7.4).

The "stiff-zero" start-variant classification stays with the existing
``scripts/e1_stage1/plan.py::start_variant_gate`` -- this module reads its
output and never re-implements the classification. Everything else here
(the cross-cycle deviation and the fixed-``sim_step`` comparison) is
report-only, per the plan: "1e-6 rad is reported, never a gate." Only the
compliance-vector check is a gate.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
for _p in (_REPO / "src", _REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reachy_ai.motion.rig_routes import R_JOINTS  # noqa: E402


class StartVariantGateUnavailable(RuntimeError):
    """``e1_stage1.plan`` could not be imported (offline environments that
    lack the package on ``sys.path``); callers treat this as evidence
    incomplete, never as a silent pass."""


def start_variant_gate(ctrl_dir, cycle: str):
    """Thin pass-through to ``e1_stage1.plan.start_variant_gate`` -- see
    that function's own docstring for the ``(ok, reason, doc)`` contract.
    Imported lazily so importing this module never requires ``scripts/``
    to be on ``sys.path`` until this specific call is made."""
    try:
        from e1_stage1 import plan as _plan
    except ImportError as exc:
        raise StartVariantGateUnavailable(str(exc)) from exc
    return _plan.start_variant_gate(ctrl_dir, cycle)


@dataclass
class InitialStateComparison:
    deviation_rad: Dict[str, float]              # vs cycle 1's first post-reset state; report-only
    max_deviation_rad: float
    fixed_step_deviation_rad: Optional[Dict[str, float]]  # report-only
    compliance_ok: bool                          # gate
    compliance_detail: str


def compare_initial_states(
    state21_a: Sequence[float], state21_b: Sequence[float],
) -> Dict[str, float]:
    """Per-joint deviation of ``state21_b`` from ``state21_a`` (radians),
    JOINT_ORDER (``evidence._JOINT_ORDER``). Report-only -- callers must
    not turn this into a pass/fail on their own."""
    from tools.goalfix_cmp.evidence import _JOINT_ORDER
    return {name: float(state21_b[i] - state21_a[i]) for i, name in enumerate(_JOINT_ORDER)}


def check_compliance(
    compliant21: Sequence[bool], expected_compliant21: Sequence[bool],
) -> Tuple[bool, str]:
    from tools.goalfix_cmp.evidence import _JOINT_ORDER
    mismatches = [name for i, name in enumerate(_JOINT_ORDER)
                  if bool(compliant21[i]) != bool(expected_compliant21[i])]
    if mismatches:
        return False, f"compliance mismatch on {mismatches}"
    return True, ""
