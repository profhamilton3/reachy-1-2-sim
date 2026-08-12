"""Scene YAML loader and validator (R12-200).

Loads a renderer-independent scene YAML file, validates it against
scenes/scene.schema.json, enforces custom rules (unique IDs, quaternion norm,
path safety), and returns an immutable SceneDocument.

No ROS or gRPC dependency — importable and testable on any host.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import yaml

try:
    import jsonschema
    _JSONSCHEMA_AVAILABLE = True
except ImportError:
    _JSONSCHEMA_AVAILABLE = False

# Schema lives next to scenes/ at the repo root.
_SCHEMA_PATH = Path(__file__).parent / "scenes" / "scene.schema.json"
_QUAT_TOL = 0.01   # |norm - 1| must be < this


class SceneValidationError(ValueError):
    """Raised when a scene file fails validation with an actionable message."""


# ── Domain types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScenePose:
    position: Tuple[float, float, float]
    orientation_wxyz: Tuple[float, float, float, float]   # always in w,x,y,z form


@dataclass(frozen=True)
class SceneGeometry:
    kind: str                                        # box, sphere, cylinder, capsule, plane, mesh
    size: Optional[Tuple[float, float, float]] = None
    radius: Optional[float] = None
    length: Optional[float] = None
    path: Optional[str] = None                       # resolved absolute path for mesh
    scale: Optional[Tuple[float, float, float]] = None


@dataclass(frozen=True)
class SceneMaterial:
    rgba: Tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)


@dataclass(frozen=True)
class SceneObject:
    id: str
    geometry: SceneGeometry
    pose: ScenePose
    material: SceneMaterial
    tracked: bool
    dynamic: bool
    semantic_class: Optional[str]


@dataclass(frozen=True)
class SceneDocument:
    """Validated, immutable scene ready for RViz/backend consumption."""
    name: str
    frame_id: str
    seed: int
    source_path: str                          # absolute path of the YAML file
    objects: Tuple[SceneObject, ...]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _rpy_to_wxyz(roll: float, pitch: float, yaw: float
                 ) -> Tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2),  math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2),   math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _parse_pose(pose_dict: dict, obj_id: str) -> ScenePose:
    pos = tuple(float(v) for v in pose_dict["position"])
    if len(pos) != 3:
        raise SceneValidationError(f"Object '{obj_id}': position must have 3 components")

    if "orientation_wxyz" in pose_dict:
        raw = pose_dict["orientation_wxyz"]
        q = tuple(float(v) for v in raw)
        norm = math.sqrt(sum(v * v for v in q))
        if abs(norm - 1.0) >= _QUAT_TOL:
            raise SceneValidationError(
                f"Object '{obj_id}': quaternion norm {norm:.6f} is not unit "
                f"(tolerance {_QUAT_TOL}). Got {list(q)}."
            )
        ori = q
    elif "rpy" in pose_dict:
        rpy = pose_dict["rpy"]
        ori = _rpy_to_wxyz(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    else:
        ori = (1.0, 0.0, 0.0, 0.0)

    return ScenePose(position=pos, orientation_wxyz=ori)


def _safe_mesh_path(raw_path: str, scene_root: Path, obj_id: str) -> str:
    """Resolve raw_path relative to scene_root; raise on path traversal."""
    if raw_path.startswith(("http://", "https://", "ftp://")):
        raise SceneValidationError(
            f"Object '{obj_id}': remote mesh URLs are not allowed: '{raw_path}'"
        )
    resolved = (scene_root / raw_path).resolve()
    approved = scene_root.resolve()
    try:
        resolved.relative_to(approved)
    except ValueError:
        raise SceneValidationError(
            f"Object '{obj_id}': mesh path '{raw_path}' escapes the scene root "
            f"'{approved}'. Path traversal is not allowed."
        )
    return str(resolved)


def _parse_geometry(geo_dict: dict, obj_id: str, scene_root: Path) -> SceneGeometry:
    kind = geo_dict["kind"]
    size = tuple(float(v) for v in geo_dict["size"]) if "size" in geo_dict else None
    radius = float(geo_dict["radius"]) if "radius" in geo_dict else None
    length = float(geo_dict["length"]) if "length" in geo_dict else None
    scale = (tuple(float(v) for v in geo_dict["scale"])
              if "scale" in geo_dict else None)
    path = None
    if kind == "mesh":
        raw = geo_dict.get("path", "")
        path = _safe_mesh_path(raw, scene_root, obj_id)
    return SceneGeometry(
        kind=kind, size=size, radius=radius, length=length, path=path, scale=scale
    )


def _parse_material(mat_dict: Optional[dict]) -> SceneMaterial:
    if mat_dict is None:
        return SceneMaterial()
    rgba = mat_dict.get("rgba")
    if rgba is not None:
        return SceneMaterial(rgba=tuple(float(v) for v in rgba))
    return SceneMaterial()


def _parse_object(obj_dict: dict, scene_root: Path) -> SceneObject:
    obj_id = obj_dict["id"]
    geometry = _parse_geometry(obj_dict["geometry"], obj_id, scene_root)
    pose = _parse_pose(obj_dict["pose"], obj_id)
    material = _parse_material(obj_dict.get("material"))
    physics = obj_dict.get("physics", {})
    return SceneObject(
        id=obj_id,
        geometry=geometry,
        pose=pose,
        material=material,
        tracked=bool(obj_dict.get("tracked", False)),
        dynamic=bool(physics.get("dynamic", False)),
        semantic_class=obj_dict.get("semantic_class"),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def load_scene(
    path: Union[str, Path],
    scene_root: Optional[Path] = None,
) -> SceneDocument:
    """Load and validate a scene YAML file.

    Args:
        path:       Absolute or relative path to the scene YAML.
        scene_root: Directory used as the approved root for mesh asset resolution.
                    Defaults to the directory containing the YAML file.

    Returns:
        Validated SceneDocument.

    Raises:
        SceneValidationError: descriptive error with field path on any failure.
        FileNotFoundError:    if the YAML file does not exist.
    """
    yaml_path = Path(path).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Scene file not found: {yaml_path}")

    effective_root = (scene_root or yaml_path.parent).resolve()

    # ── Load YAML safely ─────────────────────────────────────────────────────
    try:
        with open(yaml_path) as f:
            doc = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise SceneValidationError(f"YAML parse error in '{yaml_path}': {exc}") from exc

    if not isinstance(doc, dict):
        raise SceneValidationError(f"Scene file must be a YAML mapping, got {type(doc).__name__}")

    # ── Pre-flight path safety (runs before jsonschema for clear messages) ───
    for obj in doc.get("objects", []) if isinstance(doc, dict) else []:
        if not isinstance(obj, dict):
            continue
        geo = obj.get("geometry", {}) if isinstance(obj.get("geometry"), dict) else {}
        if geo.get("kind") == "mesh":
            raw = str(geo.get("path", ""))
            obj_id = obj.get("id", "?")
            if raw.startswith(("http://", "https://", "ftp://")):
                raise SceneValidationError(
                    f"Object '{obj_id}': remote mesh URLs are not allowed: '{raw}'"
                )
            if "../" in raw:
                raise SceneValidationError(
                    f"Object '{obj_id}': mesh path '{raw}' contains path traversal "
                    f"('../') which is not allowed."
                )

    # ── JSON Schema validation ────────────────────────────────────────────────
    if _JSONSCHEMA_AVAILABLE:
        if not _SCHEMA_PATH.exists():
            raise SceneValidationError(
                f"Scene schema not found at '{_SCHEMA_PATH}'. "
                "Run 'git status' to verify the scenes/ directory is present."
            )
        with open(_SCHEMA_PATH) as f:
            schema = json.load(f)
        try:
            jsonschema.validate(doc, schema)
        except jsonschema.ValidationError as exc:
            path_str = " → ".join(str(p) for p in exc.absolute_path) or "<root>"
            raise SceneValidationError(
                f"Schema validation failed at '{path_str}': {exc.message}"
            ) from exc
    else:
        # Minimal manual check when jsonschema is not installed.
        if doc.get("schema_version") != "1.0":
            raise SceneValidationError(
                f"Unsupported schema_version: {doc.get('schema_version')!r}. "
                "Expected '1.0'."
            )

    # ── Custom validation (beyond JSON Schema) ────────────────────────────────
    objects_raw = doc.get("objects", [])

    # Unique IDs
    seen_ids: set[str] = set()
    for obj in objects_raw:
        obj_id = obj.get("id", "")
        if obj_id in seen_ids:
            raise SceneValidationError(
                f"Duplicate object id '{obj_id}'. Every object must have a unique id."
            )
        seen_ids.add(obj_id)

    # Parse each object (including quaternion norm and path safety checks).
    parsed_objects = tuple(_parse_object(obj, effective_root) for obj in objects_raw)

    return SceneDocument(
        name=doc["name"],
        frame_id=doc.get("frame_id", "pedestal"),
        seed=int(doc.get("seed", 0)),
        source_path=str(yaml_path),
        objects=parsed_objects,
    )
