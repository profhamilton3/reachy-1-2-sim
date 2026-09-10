"""Scene view the command panel's planner is allowed to reason over (issue #49).

The `/scene` route serves grid cells and placeable object ids — enough to draw
the board, not enough to plan against.  It carries no tags, so nothing there can
tell `soda_can` from `foam_block`, and no live poses, so nothing there can tell
an object sitting on r2c2 from one parked on the floor in the pool.

This module builds the richer view.  `ObjectView.on_board` is None until a
live snapshot is applied, and None means *unknown*, not False — the planner
must never read one as "not on the table".  `apply_snapshot()` (issue #50)
turns those Nones into answers using the poses the simulator pushes, and an
object the snapshot does not mention stays None rather than becoming False.

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
    #: Live world position, when a snapshot has been applied.
    position: Optional[tuple] = None

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
    # Geometry, carried so occupancy can be decided here rather than guessed.
    x: float = 0.0
    y: float = 0.0
    top_z: float = 0.0
    half_extent: float = 0.0

    def contains(self, pos) -> bool:
        """Is an object at `pos` sitting on this cell?

        Deliberately the same predicate as `ObjectPlacer.occupant_of` in
        native_mujoco/placement.py.  The panel and the simulator disagreeing
        about what "on r2c2" means would be the worst kind of bug here: both
        would be self-consistent and the proposal would still be wrong.
        """
        return (abs(pos[0] - self.x) <= self.half_extent
                and abs(pos[1] - self.y) <= self.half_extent
                and pos[2] > self.top_z - 0.01)


#: Destination values IITG's `planner/schema.py` enum can represent.  Grid
#: cells are NOT among them, which is why proposals carry a DestinationRef and
#: the legacy value is offered only when one genuinely exists.
LEGACY_DESTINATIONS = frozenset(
    {"left_tray", "right_tray", "handover_zone", "point_only"}
)


@dataclass(frozen=True)
class DestinationRef:
    """Where a plan puts something, as a validated reference.

    Two kinds, kept apart on purpose.  A `destination` is a tagged place in the
    scene — a tray, later a bin.  A `cell` is a coordinate on the board: a
    valid target, but not a container, and the panel says so rather than
    calling one a bin.
    """

    kind: str            # "cell" | "destination"
    ref_id: str
    label: str = ""

    def as_str(self) -> str:
        return f"{self.kind}:{self.ref_id}"

    def describe(self) -> str:
        return f"grid cell {self.ref_id}" if self.kind == "cell" else (
            self.label or self.ref_id)

    def legacy_destination(self) -> Optional[str]:
        """The IITG `Destination` enum value, or None when there is none.

        None is the answer for every grid cell, and returning it is the whole
        point of this adapter.  The enum has no way to say "r2c2", and the one
        value that would type-check — `left_tray` — names a tray this scene
        deliberately removed.  A caller that needs a legacy value must handle
        None, not accept a substitute.
        """
        if self.kind == "destination" and self.ref_id in LEGACY_DESTINATIONS:
            return self.ref_id
        return None


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
    scene_revision: str = ""
    sim_step: int = 0
    error: str = ""

    # -- queries the planner uses -----------------------------------------

    def reachable_cells(self) -> List[str]:
        return sorted(n for n, c in self.cells.items() if c.reachable)

    def available_cells(self) -> List[str]:
        """Reachable and unoccupied — the cells worth offering someone.

        Before live poses, `occupant` is None everywhere and this is the same
        list as `reachable_cells`.  Once a snapshot is applied it stops
        offering a cell the next turn would only refuse.
        """
        return sorted(n for n, c in self.cells.items()
                      if c.reachable and not c.occupant)

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
            x=float(getattr(cell, "x", 0.0)),
            y=float(getattr(cell, "y", 0.0)),
            top_z=float(getattr(cell, "top_z", 0.0)),
            half_extent=float(getattr(cell, "half_extent", 0.0)),
        )
    return view


def apply_snapshot(view: SceneView, snapshot) -> SceneView:
    """Fold live object poses into a scene view, in place.

    `snapshot` is a panel_sim_link.SimSnapshot, or None.  With None the view is
    left exactly as it was — unknown stays unknown, and the planner goes on
    refusing to assert what is on the board.  That is the honest reading of a
    dropped link; the alternative, treating "no data" as "nothing there", would
    have the panel confidently report an empty board while the simulator shows
    a full one.
    """
    if snapshot is None:
        return view

    for obj in view.objects.values():
        pos = snapshot.objects.get(obj.object_id)
        if pos is None:
            # Not in the snapshot at all: still unknown, not absent.  A pool
            # object the tracker does not report is a gap in our knowledge.
            continue
        obj.position = pos
        obj.cell = next(
            (c.name for c in view.cells.values() if c.contains(pos)), None
        )
        obj.on_board = obj.cell is not None

    for cell in view.cells.values():
        cell.occupant = next(
            (o.object_id for o in view.objects.values() if o.cell == cell.name),
            None,
        )

    view.live = True
    view.scene_revision = snapshot.scene_revision
    view.sim_step = snapshot.sim_step
    return view
