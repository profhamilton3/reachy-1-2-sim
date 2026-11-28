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

TWO MORE THINGS LIVE HERE, because this is the one scene module that the native
server, the offline tools and the container image all already have:

  * `color_randomization:` is applied after the chain is resolved, so every
    consumer of one file sees the same colours.
  * the rules and the bounding box of `geometry.kind: compound` (#212).
"""

from __future__ import annotations

import hashlib
import math
import pathlib
from collections.abc import Mapping
from typing import Any

import yaml

_MAX_DEPTH = 8

MAX_COMPOUND_PARTS = 64
_QUAT_TOL = 0.01              # the same tolerance scene_loader applies to poses
_CENTRE_TOL_M = 1e-5          # part positions are written to 0.1 um
_PART_KEYS = {
    "box": frozenset({"kind", "size", "pos", "quat_wxyz"}),
    "cylinder": frozenset({"kind", "radius", "length", "pos", "quat_wxyz"}),
}
_RANDOMIZATION_KEYS = frozenset({"tag", "seed", "palette"})
_MAX_PALETTE = 64


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


# ── Compound geometry ───────────────────────────────────────────────────────

def _numbers(value: Any, n: int, what: str) -> list:
    if (not isinstance(value, (list, tuple)) or len(value) != n
            or not all(_is_number(v) for v in value)):
        raise SceneLoadError(f"{what} must be {n} finite numbers, got {value!r}")
    return [float(v) for v in value]


def _is_number(v: Any) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))


def rotation_matrix(quat_wxyz: Any) -> list:
    """3x3 rotation matrix (row major) of a [w, x, y, z] quaternion."""
    n = math.sqrt(sum(float(v) ** 2 for v in quat_wxyz))
    w, x, y, z = (float(v) / n for v in quat_wxyz)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def compound_parts(geometry: Mapping[str, Any], what: str = "compound") -> list:
    """The checked parts of a compound geometry, numbers as floats.

    Unknown and missing fields fail.  This repeats what the JSON schema says on
    purpose: the server skips the schema when jsonschema is not installed, and
    the compiler must never silently drop a field.
    """
    if not isinstance(geometry, Mapping) or geometry.get("kind") != "compound":
        raise SceneLoadError(f"{what}: not a compound geometry")
    unknown = sorted(set(geometry) - {"kind", "parts"})
    if unknown:
        raise SceneLoadError(f"{what}: compound geometry has unknown field(s) {unknown}")
    parts = geometry.get("parts")
    if not isinstance(parts, list) or not 1 <= len(parts) <= MAX_COMPOUND_PARTS:
        raise SceneLoadError(
            f"{what}: compound geometry needs a list of 1..{MAX_COMPOUND_PARTS} parts")

    out = []
    for i, part in enumerate(parts):
        where = f"{what}: parts[{i}]"
        if not isinstance(part, Mapping):
            raise SceneLoadError(f"{where} is not a mapping")
        kind = part.get("kind")
        if kind not in _PART_KEYS:
            raise SceneLoadError(
                f"{where}: kind {kind!r}; a part is one of {sorted(_PART_KEYS)}")
        unknown = sorted(set(part) - _PART_KEYS[kind])
        if unknown:
            raise SceneLoadError(f"{where}: unknown field(s) {unknown} on a {kind} part")
        missing = sorted(_PART_KEYS[kind] - set(part))
        if missing:
            raise SceneLoadError(f"{where}: missing field(s) {missing} on a {kind} part")

        checked = {"kind": kind,
                   "pos": _numbers(part["pos"], 3, f"{where}: pos"),
                   "quat_wxyz": _numbers(part["quat_wxyz"], 4, f"{where}: quat_wxyz")}
        norm = math.sqrt(sum(v * v for v in checked["quat_wxyz"]))
        if abs(norm - 1.0) >= _QUAT_TOL:
            raise SceneLoadError(f"{where}: quat_wxyz norm {norm:.6f} is not unit")
        if kind == "box":
            checked["size"] = _numbers(part["size"], 3, f"{where}: size")
            if min(checked["size"]) <= 0:
                raise SceneLoadError(f"{where}: size must be positive")
        else:
            for key in ("radius", "length"):
                checked[key] = _numbers([part[key]], 1, f"{where}: {key}")[0]
                if checked[key] <= 0:
                    raise SceneLoadError(f"{where}: {key} must be positive")
        out.append(checked)
    return out


def compound_bounds(geometry: Mapping[str, Any], what: str = "compound") -> tuple:
    """((x, y, z) low, (x, y, z) high) of a compound in the object frame."""
    lo, hi = [math.inf] * 3, [-math.inf] * 3
    for part in compound_parts(geometry, what):
        rot = rotation_matrix(part["quat_wxyz"])
        for axis in range(3):
            if part["kind"] == "box":
                half = sum(abs(rot[axis][j]) * part["size"][j] / 2.0 for j in range(3))
            else:
                # The cylinder's own axis is column 2; its rim spreads over the rest.
                along = rot[axis][2]
                half = (part["radius"] * math.sqrt(max(0.0, 1.0 - along * along))
                        + part["length"] / 2.0 * abs(along))
            lo[axis] = min(lo[axis], part["pos"][axis] - half)
            hi[axis] = max(hi[axis], part["pos"][axis] + half)
    return tuple(lo), tuple(hi)


def compound_size(geometry: Mapping[str, Any], what: str = "compound") -> tuple:
    """Full extents [x, y, z] of a compound's bounding box."""
    lo, hi = compound_bounds(geometry, what)
    return tuple(h - l for l, h in zip(lo, hi))


def check_compound_object(obj: Mapping[str, Any]) -> list:
    """Every rule a compound OBJECT must pass; returns its checked parts."""
    what = f"Object '{obj.get('id', '?')}'"
    geometry = obj.get("geometry") or {}
    parts = compound_parts(geometry, what)
    if obj.get("articulation"):
        raise SceneLoadError(f"{what}: articulation is not supported on a compound object")
    lo, hi = compound_bounds(geometry, what)
    off = [(l + h) / 2.0 for l, h in zip(lo, hi)]
    if max(abs(v) for v in off) > _CENTRE_TOL_M:
        # Placement, the planner and truth export all read an object as
        # centre +/- size/2.
        raise SceneLoadError(
            f"{what}: a compound's bounding box must be centred on the object "
            f"origin; it is centred on {[round(v, 6) for v in off]}")
    physics = obj.get("physics") or {}
    if physics.get("dynamic"):
        mass = physics.get("mass")
        if not _is_number(mass) or mass <= 0:
            raise SceneLoadError(
                f"{what}: a dynamic compound object needs a positive physics.mass; "
                "it becomes the body's total mass")
    return parts


# ── Colour randomisation ────────────────────────────────────────────────────

def _randomized_index(seed: int, object_id: str, n: int) -> int:
    # SHA-256 rather than `random`: a seed written into a run record must give
    # the same colours on any Python, whatever order the objects are in.
    digest = hashlib.sha256(f"{seed}:{object_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % n


def apply_color_randomization(doc: dict) -> dict:
    """Repaint the tagged objects of a RESOLVED scene.  No block, no change."""
    if "color_randomization" not in doc:
        return doc
    block = doc["color_randomization"]
    if not isinstance(block, Mapping):
        raise SceneLoadError("color_randomization must be a mapping")
    unknown = sorted(set(block) - _RANDOMIZATION_KEYS)
    missing = sorted(_RANDOMIZATION_KEYS - set(block))
    if unknown or missing:
        raise SceneLoadError(
            f"color_randomization: unknown field(s) {unknown}, missing {missing}")
    tag, seed, palette = block["tag"], block["seed"], block["palette"]
    if not isinstance(tag, str) or not tag:
        raise SceneLoadError("color_randomization.tag must be a non-empty string")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2 ** 32:
        raise SceneLoadError(
            f"color_randomization.seed must be an integer in 0..2^32-1, got {seed!r}")
    if not isinstance(palette, list) or not 2 <= len(palette) <= _MAX_PALETTE:
        raise SceneLoadError(
            f"color_randomization.palette must list 2..{_MAX_PALETTE} rgba colours")
    colours = []
    for i, entry in enumerate(palette):
        rgba = _numbers(entry, 4, f"color_randomization.palette[{i}]")
        if not all(0.0 <= v <= 1.0 for v in rgba):
            raise SceneLoadError(f"color_randomization.palette[{i}] must be in 0..1")
        colours.append(rgba)

    objects, matched = [], 0
    for obj in doc.get("objects") or []:
        if isinstance(obj, Mapping) and tag in (obj.get("tags") or []):
            matched += 1
            material = dict(obj.get("material") or {})
            material["rgba"] = list(
                colours[_randomized_index(seed, str(obj.get("id")), len(colours))])
            obj = dict(obj, material=material)
        objects.append(obj)
    if not matched:
        raise SceneLoadError(
            f"color_randomization.tag {tag!r} matches no object; the scene would "
            "look randomised and not be")
    return dict(doc, objects=objects)


def color_randomization_record(doc: Mapping[str, Any]) -> dict | None:
    """What a run manifest records of a RESOLVED scene's randomisation:
    {tag, seed, colors: {object id: rgba}}, or None if it has none."""
    block = doc.get("color_randomization")
    if not block:
        return None
    return {
        "tag": block["tag"],
        "seed": block["seed"],
        "colors": {str(o["id"]): list(o["material"]["rgba"])
                   for o in doc.get("objects") or []
                   if block["tag"] in (o.get("tags") or [])},
    }


# ── Loading ─────────────────────────────────────────────────────────────────

def load_scene(path: str | pathlib.Path, _depth: int = 0) -> dict:
    """Read a scene YAML, resolving `extends`, `drop_objects` and, once the
    whole chain is merged, `color_randomization`."""
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

    # Only on the finished document: a child may override just the seed.
    return apply_color_randomization(doc) if _depth == 0 else doc
