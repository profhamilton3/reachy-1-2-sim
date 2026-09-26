"""The minimum-jerk time-scaling `s(tau)` a `goto(..., MINIMUM_JERK)` flies,
and its inverse -- needed by both `echo.py` (path-coincidence, C2-style
agreement) and `pathcheck.py` (C2/C3/C4). Implemented locally per the
assignment ("implement the formula locally"); the optional
`test_goalfix_cmp_echo.py::test_min_jerk_matches_reachy_sdk` pins it against
`reachy_sdk.trajectory`'s own interpolation when that package is importable
(``importorskip``), skipped under the system interpreter.
"""
from __future__ import annotations


def s_of_tau(tau: float) -> float:
    """The canonical quintic minimum-jerk profile, tau in [0, 1] -> [0, 1]."""
    tau = 0.0 if tau < 0.0 else (1.0 if tau > 1.0 else tau)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def tau_of_s(s: float, iters: int = 60) -> float:
    """Inverse of ``s_of_tau`` by bisection (``s_of_tau`` is monotone
    non-decreasing on [0, 1], so this always converges)."""
    s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if s_of_tau(mid) < s:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def pose_at(a: float, b: float, tau: float) -> float:
    """The commanded value of one joint at ``tau`` in [0, 1] of a
    ``goto(a -> b, MINIMUM_JERK)``."""
    return a + s_of_tau(tau) * (b - a)


def implied_tau(a: float, b: float, value: float) -> float:
    """``tau`` such that ``pose_at(a, b, tau) == value``, for a joint that
    actually moves (``a != b``). Ill-conditioned near either end -- callers
    exclude values within the tolerance the plan/report specify (0.05 deg)
    of ``a``/``b`` before trusting this."""
    if a == b:
        raise ValueError("implied_tau is undefined when start == goal")
    return tau_of_s((value - a) / (b - a))
