"""Simulator contact-model setting: the MuJoCo no-slip post-processing pass.

The model file (``model/reachy_1_2.xml``) sets ``noslip_iterations="10"`` as
the simulator default (ADR-0005, #55).  This module validates an optional
override (``--noslip-iterations N``; 0 restores the previous behaviour),
applies it to a compiled model, and describes the EFFECTIVE setting so it
can be logged, advertised in the handshake and recorded in manifests.

Simulator-only: nothing here validates the physical robot.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

NOSLIP_ITERATIONS_MAX = 50

_SOLVERS = {0: "PGS", 1: "CG", 2: "Newton"}
_CONES = {0: "pyramidal", 1: "elliptic"}


def validate_noslip_iterations(value: Any) -> int:
    """Return ``value`` as an int in 0..NOSLIP_ITERATIONS_MAX or raise ValueError."""
    msg = (f"noslip_iterations must be an integer in 0..{NOSLIP_ITERATIONS_MAX}, "
           f"got {value!r}")
    if isinstance(value, bool):
        raise ValueError(msg)
    if isinstance(value, int):
        n = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(msg)
        n = int(value)
    elif isinstance(value, str):
        try:
            n = int(value.strip())
        except ValueError:
            raise ValueError(msg) from None
    else:
        raise ValueError(msg)
    if not 0 <= n <= NOSLIP_ITERATIONS_MAX:
        raise ValueError(msg)
    return n


def apply_contact_model(model, noslip_iterations: Optional[Any] = None) -> Dict[str, Any]:
    """Apply an optional override to a compiled model; return the effective setting.

    ``noslip_iterations=None`` leaves the model file's value in force.
    """
    model_default = int(model.opt.noslip_iterations)
    source = "model"
    if noslip_iterations is not None:
        n = validate_noslip_iterations(noslip_iterations)
        model.opt.noslip_iterations = n
        source = "override"
    return {
        "noslip_iterations": int(model.opt.noslip_iterations),
        "noslip_tolerance": float(model.opt.noslip_tolerance),
        "source": source,
        "model_default_noslip_iterations": model_default,
        "solver": _SOLVERS.get(int(model.opt.solver), str(int(model.opt.solver))),
        "cone": _CONES.get(int(model.opt.cone), str(int(model.opt.cone))),
        "impratio": float(model.opt.impratio),
        "iterations": int(model.opt.iterations),
    }


def physics_profile_id(contact_model: Optional[Dict[str, Any]]) -> str:
    """``"default"`` unless the effective setting differs from the model file's.

    ``model_sha256`` hashes only the file, so an override must change the
    recorded physics identity some other way.
    """
    if not contact_model:
        return "default"
    if contact_model["noslip_iterations"] == contact_model["model_default_noslip_iterations"]:
        return "default"
    return f"noslip_iterations={contact_model['noslip_iterations']}"


def describe_contact_model(cm: Dict[str, Any]) -> str:
    """One-line startup log text."""
    src = ("model default" if cm["source"] == "model"
           else f"override; model default {cm['model_default_noslip_iterations']}")
    return (f"Contact model: noslip_iterations={cm['noslip_iterations']} ({src}) "
            f"tolerance={cm['noslip_tolerance']:g}")
