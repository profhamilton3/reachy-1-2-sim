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

    Simulation backends (kinematic, mujoco-remote): always True.
    Physical robot: True only when REACHY_ENABLE_MOTION=true.
    Any other backend without explicit motion enable: False (fail safe).
    """
    motion_enabled = os.environ.get(_ENV_MOTION, "false").lower() == "true"
    sim_backend = os.environ.get(_ENV_SIM, "kinematic").lower()
    in_sim = sim_backend in ("kinematic", "mujoco-remote")

    if in_sim:
        log.debug("gate_check: sim backend '%s' — motion commands permitted", sim_backend)
        return True

    if not motion_enabled:
        log.warning(
            "gate_check: non-simulation backend '%s' but REACHY_ENABLE_MOTION is not 'true' — "
            "denying motion; set REACHY_ENABLE_MOTION=true with a human operator present",
            sim_backend,
        )
        return False

    log.info("gate_check: physical robot, REACHY_ENABLE_MOTION=true — proceeding")
    return True
