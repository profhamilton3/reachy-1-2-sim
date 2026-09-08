"""Scene loading with single-parent inheritance.

A scene file may name a parent:

    extends: FWDCenterLabMCC.yaml
    name: FWDCenterLabSiva
    objects:
      - id: red_cube
        material: {rgba: [0.26, 0.26, 0.25, 1.0]}

and inherit everything it does not restate.  This exists because the measured
geometry in FWDCenterLabMCC.yaml — board extents, grid pitch, cell centres,
reachability, and several hundred lines of provenance for how each number was
arrived at — is the expensive part of a scene, and a variant that differs only
in what colour the objects are should not fork it.  Two copies of a measurement
is one copy that will silently go stale.

MERGE RULES

  * Mappings merge key by key, the child winning.
  * `objects` merges BY `id`: an entry whose id already exists patches that
    object (recursively); an unknown id is appended.  Ordering follows the
    parent, with new objects last.
  * `drop_objects: [id, ...]` removes inherited objects outright.
  * Any other list REPLACES the inherited one.  Element-wise merging of an
    anonymous list has no well-defined identity to merge on, and guessing is
    worse than making the child restate it.

Chains are followed to any depth; a cycle raises rather than hanging.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any

import yaml

_MAX_DEPTH = 8


class SceneLoadError(Exception):
    pass


def _merge(parent: Any, child: Any) -> Any:
    """Child wins, except that mappings merge and `objects` merges by id."""
    if not (isinstance(parent, Mapping) and isinstance(child, Mapping)):
        return child

    out = dict(parent)
    for key, value in child.items():
        if key == "objects":
            out[key] = _merge_objects(parent.get(key) or [], value or [])
        elif key in out:
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _merge_objects(parent: list, child: list) -> list:
    by_id = {o["id"]: dict(o) for o in parent if isinstance(o, Mapping) and "id" in o}
    order = list(by_id)
    extra = [o for o in parent if not (isinstance(o, Mapping) and "id" in o)]

    for obj in child:
        if not isinstance(obj, Mapping) or "id" not in obj:
            extra.append(obj)
            continue
        oid = obj["id"]
        if oid in by_id:
            by_id[oid] = _merge(by_id[oid], obj)
        else:
            by_id[oid] = dict(obj)
            order.append(oid)
    return [by_id[o] for o in order] + extra


def load_scene(path: str | pathlib.Path, _depth: int = 0) -> dict:
    """Read a scene YAML, resolving `extends` and `drop_objects`."""
    if _depth > _MAX_DEPTH:
        raise SceneLoadError(
            f"scene inheritance deeper than {_MAX_DEPTH} at {path!s} — "
            f"almost certainly a cycle")

    p = pathlib.Path(path).resolve()
    try:
        doc = yaml.safe_load(p.read_text())
    except FileNotFoundError as exc:
        raise SceneLoadError(f"scene not found: {p}") from exc
    if not isinstance(doc, Mapping):
        raise SceneLoadError(f"scene {p} is not a mapping")
    doc = dict(doc)

    parent_ref = doc.pop("extends", None)
    dropped = doc.pop("drop_objects", None) or []

    if parent_ref is not None:
        parent_path = (p.parent / str(parent_ref)).resolve()
        if parent_path == p:
            raise SceneLoadError(f"scene {p} extends itself")
        doc = _merge(load_scene(parent_path, _depth + 1), doc)

    if dropped:
        drop = set(dropped)
        objects = doc.get("objects") or []
        kept = [o for o in objects
                if not (isinstance(o, Mapping) and o.get("id") in drop)]
        missing = drop - {o.get("id") for o in objects if isinstance(o, Mapping)}
        if missing:
            raise SceneLoadError(
                f"{p.name}: drop_objects names {sorted(missing)}, which the "
                f"inherited scene does not contain — a typo here would "
                f"silently leave the object in the scene")
        doc["objects"] = kept

    return doc
