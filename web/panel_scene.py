"""Scene view the command panel's planner is allowed to reason over (issue #49).

The `/scene` route serves grid cells and placeable object ids — enough to draw
the board, not enough to plan against.  It carries no tags, so nothing there can
tell `soda_can` from `foam_block`, and no live poses, so nothing there can tell
an object sitting on r2c2 from one parked on the floor in the pool.

This module builds the richer view, and is explicit about the half it cannot
fill in yet.  `ObjectView.on_board` is None in stage 1 and means *unknown*, not
False; the planner must never read a None as "not on the table".  Issue #50
supplies live object poses from the simulator's `state` messages and turns
those Nones into answers.

TAGS ARE MATCHED EXACTLY
------------------------
`soda_can` is tagged `recyclable` and `foam_block` is tagged `non-recyclable`.
A substring test says the foam block is recyclable and sends the wrong object
to the wrong place, so membership is the only test used here.

DESTINATIONS ARE TAG-DRIVEN, NOT NAMED
--------------------------------------
FWDCenterLabMCC tags its trays `[tray, destination, placement-target,
sort-target]`, and FWDCenterLabSiva deliberately drops both trays via
`drop_objects`.  So "which destinations exist" is answered by looking for the
`destination` tag, and in the Siva/Pool scenes the honest answer is none.  That
is a fact to report, not a gap to paper over by resolving "the bin" to an
absent tray.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

#: An object is a candidate destination when it carries this tag.
DESTINATION_TAG = "destination"

#: Tags that answer "is this recyclable".  Exact membership, never substring.
RECYCLABLE_TAG = "recyclable"
NON_RECYCLABLE_TAG = "non-recyclable"


@dataclass
class ObjectView:
    object_id: str
    tags: List[str] = field(default_factory=list)
    semantic_class: str = ""
    #: True/False once live poses are available; None means "not yet known".
    on_board: Optional[bool] = None
    #: Grid cell the object currently occupies, when live poses say so.
    cell: Optional[str] = None

    @property
    def is_recyclable(self) -> bool:
        return RECYCLABLE_TAG in self.tags

    @property
    def is_non_recyclable(self) -> bool:
        return NON_RECYCLABLE_TAG in self.tags


@dataclass
class CellView:
    name: str
    reachable: bool = True
    distance_m: float = 0.0
    occupant: Optional[str] = None


@dataclass
class DestinationView:
    """A real, tagged place to put something.  Never a grid cell.

    Grid cells are also valid targets but are a different kind: they are
    coordinates on the board, not containers, and the panel says so rather than
    calling one a bin.
    """

    destination_id: str
    tags: List[str] = field(default_factory=list)
    label: str = ""


@dataclass
class SceneView:
    name: str = ""
    objects: Dict[str, ObjectView] = field(default_factory=dict)
    cells: Dict[str, CellView] = field(default_factory=dict)
    destinations: Dict[str, DestinationView] = field(default_factory=dict)
    #: False while poses come from the scene file rather than the simulator.
    live: bool = False
    error: str = ""

    # -- queries the planner uses -----------------------------------------

    def reachable_cells(self) -> List[str]:
        return sorted(n for n, c in self.cells.items() if c.reachable)

    def recyclables(self) -> List[ObjectView]:
        return [o for o in self.objects.values() if o.is_recyclable]

    def tabletop_recyclables(self) -> List[ObjectView]:
        """Recyclable objects known to be on the board.

        `on_board is True` and not merely truthy: stage 1 leaves it None, and a
        None must not be read as either answer.
        """
        return [o for o in self.recyclables() if o.on_board is True]

    @property
    def has_destination(self) -> bool:
        return bool(self.destinations)


def _tags(obj: Mapping[str, Any]) -> List[str]:
    raw = obj.get("tags") or []
    return [str(t) for t in raw if isinstance(t, (str, int, float))]


def scene_view_from_doc(doc: Mapping[str, Any], cells: Mapping[str, Any],
                        *, placeable: Optional[List[str]] = None) -> SceneView:
    """Build a SceneView from a resolved scene document and its placement cells.

    `cells` is whatever `placement.cells_from_scene()` returned — accessed
    duck-typed so this module does not have to import the native simulator
    package, which is not on the path inside the compatibility container.
    """
    view = SceneView(name=str(doc.get("name", "")))

    allowed = set(placeable) if placeable is not None else None
    for obj in doc.get("objects", []) or []:
        if not isinstance(obj, Mapping):
            continue
        oid = str(obj.get("id", "")).strip()
        if not oid:
            continue
        tags = _tags(obj)
        if DESTINATION_TAG in tags:
            view.destinations[oid] = DestinationView(
                destination_id=oid, tags=tags,
                label=str(obj.get("semantic_class") or oid),
            )
            continue
        if allowed is not None and oid not in allowed:
            continue
        view.objects[oid] = ObjectView(
            object_id=oid, tags=tags,
            semantic_class=str(obj.get("semantic_class") or ""),
        )

    for name, cell in (cells or {}).items():
        view.cells[str(name)] = CellView(
            name=str(name),
            reachable=bool(getattr(cell, "reachable", True)),
            distance_m=round(float(getattr(cell, "shoulder_distance_m", 0.0)), 3),
        )
    return view
