"""The forearm-footprint check's pure computation (#82, extracted for #56).

This used to live inside ``web/panel_executor.py::_footprint_refusal``.  It is
pure geometry, so the same numbers can be asked for by the panel's
``available()`` and by the pick/place job preflight
(``tasks.pick_place_live.preflight_pick_place``) without a second copy that
could drift.

WHAT THIS CHECK COVERS, EXACTLY -- and nothing more is claimed for it:

  * Only the legs listed in ``rig_routes.FOOTPRINT_LEGS`` for each route:
    RAISE_TO_SIDE is HOVER -> REST_SHUT -> REST and STOW_FROM_SIDE is
    PRESENT -> REST_SHUT -> HOVER.  The rest of each measured route is covered
    by that route's own rig validation, not by this live-object check.
  * Manipulable objects only: ``path_clearances`` with its default ids, so no
    rig fixtures.
  * The wider commanded aperture of each leg's two ends, judged by the
    capsule radius ``hand_radius`` actually produces (the encoding is
    inverted: SHUT = +20 is a NARROWER hand than OPEN = -45).
  * The tube hand model, the default.

Pure geometry: ``arm=None`` -- nothing here touches the SDK.
"""

from __future__ import annotations

from typing import Dict, Sequence

from reachy_ai.motion import rig_routes as R
from reachy_ai.motion.kinematics import CartesianPlanner, hand_radius
from reachy_ai.scene.awareness import Clearance, SceneModel


def route_footprint_clearances(model: SceneModel,
                               route_names: Sequence[str]) -> Dict[str, Clearance]:
    """Worst modelled clearance per manipulable object over ``route_names``.

    Routes without a ``FOOTPRINT_LEGS`` entry contribute nothing.  The result
    maps object id to the ``Clearance`` (distance, link, point) of that
    object's tightest approach over every leg of every named route; callers
    compare ``.distance`` against ``rig_routes.FOOTPRINT_MARGIN``.
    """
    legs = [R.FOOTPRINT_LEGS[r] for r in route_names if r in R.FOOTPRINT_LEGS]
    if not legs:
        return {}

    # arm=None: every clearance method is pure geometry over the scene and
    # never touches the SDK, which is what makes this callable here rather
    # than only from the motion process that actually owns an arm.
    planner = CartesianPlanner(arm=None, scene=model, side="right")

    worst: Dict[str, Clearance] = {}
    for waypoints in legs:
        for a, b in zip(waypoints, waypoints[1:]):
            qa = [a[j] for j in R.ARM7]
            qb = [b[j] for j in R.ARM7]
            # The waypoints carry the commanded gripper for each end of this
            # leg.  Which one is "wider" is a question about the capsule
            # radius `hand_radius` actually produces, not about the raw
            # commanded degrees: the encoding is inverted (#82/A2 -- SHUT =
            # +20 is a NARROWER hand than OPEN = -45), so comparing the
            # degrees directly picks the wrong endpoint.  Using the wider of
            # the two for the whole leg is the conservative choice already
            # documented in FOOTPRINT_LEGS -- never narrower than either
            # commanded end.
            gripper_deg = max(a["r_gripper"], b["r_gripper"], key=hand_radius)
            for oid, c in planner.path_clearances(
                    qa, qb, gripper_deg=gripper_deg).items():
                if oid not in worst or c.distance < worst[oid].distance:
                    worst[oid] = c
    return worst
