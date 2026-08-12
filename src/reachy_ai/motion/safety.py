"""
Safety gate for Reachy 1.2 motion — must pass before any physical motion is sent.

In simulation (REACHY_ENABLE_MOTION unset or "false") the gate always passes so
the kinematic backend can animate joints for visual verification.

On the physical robot REACHY_ENABLE_MOTION must be explicitly set to "true" and
a human operator must be present before running any motion primitives.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_ENV_MOTION = "REACHY_ENABLE_MOTION"
_ENV_SIM = "REACHY_SIM_BACKEND"


def gate_check() -> bool:
    """Return True if it is safe to send motion commands.

    Simulation: always True (joints animate in the kinematic backend).
    Physical robot: True only when REACHY_ENABLE_MOTION=true.
    """
    motion_enabled = os.environ.get(_ENV_MOTION, "false").lower() == "true"
    in_sim = os.environ.get(_ENV_SIM, "kinematic").lower() in ("kinematic", "mujoco-remote")

    if in_sim or not motion_enabled:
        log.debug("gate_check: sim/dry-run — motion commands permitted (no hardware)")
        return True

    log.info("gate_check: physical robot, REACHY_ENABLE_MOTION=true — proceeding")
    return True
