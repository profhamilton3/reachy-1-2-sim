"""Clearance decomposition: planned -> commanded -> realised (plan §7.3).

Reuses ``scripts/measure_route_clearance.py``'s already-tested clearance
math as-is, never modified: ``planned_clearance_by_link``,
``validated_samples``, ``PLANNED_APERTURE_POLICY``, ``full_route_poses``,
``_CAPSULE_NAMES_BY_HAND`` -- the same ``sys.path`` pattern
``outputs/analysis-2026-09-23-corrected-campaign/campaign.py`` uses. The
COMMANDED path has no equivalent there (that module only ever looked at
recorder logs, never raw ``commands.jsonl``), so ``_worst_per_link_over_samples``
below reimplements ``_worst_clearance_over_samples``'s aggregation loop at
PER-LINK granularity (which that function does not expose -- it pools every
capsule of a hand together) using the same underlying primitives
(``link_capsules`` + ``scene.clearances``), exactly the pattern
``campaign.py``'s own docstring says it had to duplicate for the same
reason.

``indep_wrist_ball`` is ported from
IITG-Reachy-Project/outputs/analysis-2026-09-23-corrected-campaign/hover_wristball.py
(equivalence test in tests/unit/test_goalfix_cmp_equivalence.py).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
for _p in (_REPO / "src", _REPO / "scripts", _REPO / "native_mujoco"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reachy_ai.motion.rig_routes import ARM7  # noqa: E402
from reachy_ai.motion.kinematics import link_capsules  # noqa: E402
from reachy_ai.scene.awareness import SceneModel  # noqa: E402
import measure_route_clearance as M  # noqa: E402

HAND_MODES = M.HAND_MODES
ApertureDataError = M.ApertureDataError


def _worst_per_link_over_samples(
    q7_and_gripper: Sequence[Tuple[Sequence[float], Optional[float]]],
    scene: SceneModel, hand: str, link: str,
) -> Dict[str, float]:
    """Worst clearance for exactly one named link, over a sequence of
    (q7, gripper_deg) samples -- the same per-link filtering
    ``planned_clearance_by_link`` applies to its own planned-path samples,
    applied here to a commanded or realised sample stream instead."""
    worst: Dict[str, float] = {}
    for q7, gripper_deg in q7_and_gripper:
        caps = [c for c in link_capsules(q7, "right", gripper_deg, hand=hand) if c[0] == link]
        for oid, c in scene.clearances(caps).items():
            if worst.get(oid) is None or c.distance < worst[oid]:
                worst[oid] = c.distance
    return worst


#: native_mujoco's states/commands are MJCF-native radians
#: (native_mujoco/joint_map.py); measure_route_clearance.py -- and
#: link_capsules underneath it -- expect the SDK's native degrees (the same
#: convention rig_routes.py's pose constants are written in, e.g.
#: ``HOVER = pose(r_shoulder_pitch=-40.0, ...)``). Every sample crossing
#: that boundary is converted here, once, rather than silently passing
#: radians into a function that will call ``np.radians`` on them again.
def commanded_samples(targets8_rad: Sequence[Sequence[float]]) -> List[Tuple[List[float], float]]:
    """(q7_deg, gripper_deg) from each command's own 8 right-arm targets
    (radians, native_mujoco convention) -- "commanded" per plan §7.3: the
    gripper aperture the command itself asked for, not what the hand
    actually reached."""
    return [(list(np.degrees(t[:7])), float(np.degrees(t[7]))) for t in targets8_rad]


def realised_samples_from_states(
    position_rad8: Sequence[Sequence[float]],
) -> Tuple[List[Tuple[List[float], float]], List[int]]:
    """(q7_deg, gripper_deg) from realised state positions (radians), through
    ``validated_samples`` (schema 3) so a missing/invalid gripper reading
    raises ``ApertureDataError`` -- never imputed -- exactly as it would
    for a recorder log. Returns (samples, assumed_indices)."""
    schema_samples = []
    for row in position_rad8:
        joints = {j: float(np.degrees(row[i])) for i, j in enumerate(ARM7)}
        if len(row) > 7 and row[7] is not None:
            joints["r_gripper"] = float(np.degrees(row[7]))
        schema_samples.append({"joints": joints, "wall_time_ns": 0, "t": 0.0})
    return M.validated_samples(schema_samples, schema_version=M.LOG_SCHEMA_VERSION)


@dataclass
class LinkDelta:
    hand: str
    link: str
    object_id: str
    planned_cm: float
    commanded_cm: float
    realised_cm: float

    @property
    def delta_cmd_cm(self) -> float:
        return self.planned_cm - self.commanded_cm

    @property
    def delta_trk_cm(self) -> float:
        return self.commanded_cm - self.realised_cm


def compute_deltas(
    route: str, waypoints, n_planned_samples: int, scene: SceneModel,
    commanded_q7_gripper: Sequence[Tuple[Sequence[float], float]],
    realised_q7_gripper: Sequence[Tuple[Sequence[float], float]],
) -> List[LinkDelta]:
    """Per (hand, link, object) triple: planned/commanded/realised worst
    clearance in cm (``scene.clearances`` itself returns metres; converted
    here so ``LinkDelta``'s ``_cm``-suffixed fields mean what they say, the
    same convention this codebase's other clearance tooling -- campaign.py,
    hover_wristball.py -- uses throughout). ``waypoints`` is passed straight through to
    ``planned_clearance_by_link`` (``None`` for the route's own
    ``FOOTPRINT_LEGS`` scope, or ``full_route_poses(route)`` for the whole
    named route)."""
    out: List[LinkDelta] = []
    for hand in HAND_MODES:
        for link in sorted(M._CAPSULE_NAMES_BY_HAND[hand]):
            planned = M.planned_clearance_by_link(
                route, n_planned_samples, scene, link, hand=hand, waypoints=waypoints)
            commanded = _worst_per_link_over_samples(commanded_q7_gripper, scene, hand, link)
            realised = _worst_per_link_over_samples(realised_q7_gripper, scene, hand, link)
            for oid in sorted(set(planned) | set(commanded) | set(realised)):
                if oid not in planned or oid not in commanded or oid not in realised:
                    continue
                out.append(LinkDelta(hand, link, oid, 100 * planned[oid],
                                      100 * commanded[oid], 100 * realised[oid]))
    return out


# ---------------------------------------------------------------------------
# Independent wrist_ball cross-check (ported from hover_wristball.py)
# ---------------------------------------------------------------------------

def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def indep_wrist_ball(J: Dict[str, float], center: Sequence[float], size: Sequence[float]) -> float:
    """Independent FK + box-SDF cross-check of the wrist_ball clearance
    against a box object, in cm -- ported verbatim (arg order, constants,
    formula) from ``hover_wristball.py``'s function of the same name."""
    q = np.radians([J[j] for j in ARM7])
    sh = np.array([0, -0.19, 1.0])
    Rm = _ry(q[0]) @ _rx(q[1]) @ _rz(q[2])
    el = sh + Rm @ np.array([0, 0, -0.28])
    Rm = Rm @ _ry(q[3]) @ _rz(q[4])
    wr = el + Rm @ np.array([0, 0, -0.25])
    d = np.abs(wr - np.array(center)) - np.array(size) / 2
    return 100 * (np.linalg.norm(np.maximum(d, 0)) + min(d.max(), 0) - 0.028)


def cross_check_wrist_ball(
    q7_and_gripper: Sequence[Tuple[Sequence[float], Optional[float]]],
    scene: SceneModel, object_id: str, hand: str = "shells",
) -> float:
    """max |repo - independent| wrist_ball clearance (cm) over a sample
    stream, mirroring hover_wristball.py's own A3 check."""
    obj = scene.get(object_id)
    worst_diff = 0.0
    for q7, gripper_deg in q7_and_gripper:
        caps = [c for c in link_capsules(q7, "right", gripper_deg, hand=hand) if c[0] == "wrist_ball"]
        repo_cm = 100 * scene.clearances(caps)[object_id].distance
        J = dict(zip(ARM7, q7))
        J["r_gripper"] = gripper_deg if gripper_deg is not None else 0.0
        indep_cm = indep_wrist_ball(J, obj.center, obj.size)
        worst_diff = max(worst_diff, abs(repo_cm - indep_cm))
    return worst_diff
