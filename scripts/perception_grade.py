#!/usr/bin/env python3
"""Grade the robot team's detector and stereo depth against simulator truth (#186, #182).

    python3 scripts/perception_grade.py --manifest layouts.json --out-dir out/

`layouts.json` (paths are relative to it and must stay below its directory):

    {"layouts": [{"name": "A_default",
                  "truth_before": "A/truth_before.json",   # scripts/export_truth.py
                  "truth_after":  "A/truth_after.json",
                  "pairs": ["A/pair1.zip", "A/pair2.zip"]}]}   # viewer single-pair bundles

Each pair is a bundle (or an unpacked folder) from the tabletop stereo viewer holding
`analysis.json` and `depth.json`.  Writes `result.json` and `report.md`.

A pair entry that ends in `.json` is instead a `PerceivedScene` document (perceived_scene.v1, made by the tabletop
builder, validated by native_mujoco/perceived_scene.py), graded with the same metrics: the 40 mm gate is on the
measured `surface_point_m` (so the same objects match with a surface correction on or off), and `centre_m` is
graded against the true centre.  Cells are graded from the scene's own cell reports.  A layout may not mix forms.
`--bias-loo` adds a leave-one-layout-out "measured-bias.v1" evaluation (see `bias_loo`).

Rules, as decided on #186:
* The scene must be static across a layout.  Truth is read before and after that
  layout's captures; if the scene revision changed or any object moved more than
  1 mm, the whole layout is rejected.
* Grading is in 3D in the URDF torso frame.  Truth (scene frame) is shifted by the
  `arm_base_in_world` the export records, (0.206, 0, 1.15) (ADR-0010).
* A detection matches a truth object when its 3D position is within ONE radius
  (default 40 mm) of the object's visible-surface point: the centre moved toward
  the camera by the object's half-extent along the view ray.  The viewer measures
  the visible surface, not the centre, so this is the point it can be right about.
  Raw centre error, signed along-ray error and horizontal (x, y) error against the
  true centre are reported separately; the horizontal one is the pick-and-place number.
* In view = the truth centre projects inside both images, at the recorded neck
  angles, with the profile's own lens model.
* Truth objects not standing on a grid cell (the parked floor pool is in view at board_view)
  are reported under `off_board`, outside the headline, unless --score-off-board is given.
* Objects the detector has no class for (can, foam, ...) are not misses; a detection
  that lands on one is an "OOV hit", not a correct detection.
* `empty` detections are graded against grid cells, not positions.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import pathlib
import statistics
import sys
import zipfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "native_mujoco"))

import perceived_scene as ps                                # noqa: E402
import truth_export as te                                   # noqa: E402

SCHEMA_VERSION = "perception_grade.v1"
VOCAB = ("cube", "cylinder")                # the deployed detector's object classes
EMPTY = "empty"

MATCH_RADIUS_M = 0.040
IN_VIEW_MARGIN_PX = 16.0
BAND_EDGES_M = (0.45, 0.50, 0.55, 0.60)     # camera-to-object distance bands
MOVE_TOL_M = 0.001                          # static-scene bracket

MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_ZIP_BYTES = 32 * 1024 * 1024
_PROFILE_KEYS = ("lens", "resolution", "focal_px", "warp_fx_px", "radial", "baseline_m",
                 "neck_origin_xyz_m", "neck_origin_pitch_rad", "right_camera_in_head_m")
_NECK = ("neck_roll", "neck_pitch", "neck_yaw")
# Head frame +x forward, +y left, +z up  <-  optical +x right, +y down, +z forward.
_OPTICAL_TO_HEAD = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)


#: Sim stereo profiles a PerceivedScene can name by `rig.intrinsics_sha256` (= StereoGeometry.identity["id"], the first
#: 16 hex digits of the SHA-256 of the identity JSON without the id).  The document carries no lens model, so the grader
#: needs it; an unknown id is refused.  `test_perception_grade.py` re-derives the id from this dict.
KNOWN_SIM_PROFILES: Dict[str, Dict[str, Any]] = {
    "771392183ddc6c62": {
        "baseline_m": 0.0725, "focal_px": [407.0, 398.8], "geometry": "URDF camera positions and axes",
        "lens": "distorted", "name": "measured_2026_08_27 + scene URDF", "neck_origin_pitch_rad": 0.174,
        "neck_origin_xyz_m": [0.015, 0, 0.095], "radial": [[-0.3163, 0.1027], [-0.3895, 0.1838]],
        "resolution": [640, 480], "right_camera_in_head_m": [0.0033, -0.03625, 0.061], "source": "sim",
        "warp_fx_px": [408.4, 400.9]},
}


class GradeError(ValueError):
    """An input cannot be graded; the message names the file and the reason."""


# ---------------------------------------------------------------------------
# Small geometry
# ---------------------------------------------------------------------------

def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def quat_matrix(q: Sequence[float]) -> np.ndarray:
    """Rotation matrix of a unit quaternion given as [w, x, y, z]."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Rig:
    """The sim stereo head at one neck pose: torso <-> camera, and the lens.

    Mirrors the viewer's `camera_to_torso` (reachy-tabletop-ai `stereo_depth.py`, @ fec44b6):
    torso = R(y(pitch0) x(roll) y(pitch) z(yaw)) . (O . p_optical + camera_in_head) + neck_origin,
    angles in degrees as read back.  This is the inverse of it, plus the left eye.
    """

    def __init__(self, profile: Mapping[str, Any], neck_deg: Mapping[str, float]):
        missing = [k for k in _PROFILE_KEYS if k not in profile]
        if missing:
            raise GradeError("depth.json profile lacks " + ", ".join(missing))
        self.distorted = profile["lens"] == "distorted"
        self.size = tuple(int(v) for v in profile["resolution"])
        self.focal = [float(v) for v in profile["focal_px"]]          # [left, right] (= fy)
        self.warp_fx = [float(v) for v in profile["warp_fx_px"]]
        self.radial = [[float(v) for v in k] for k in profile["radial"]]
        roll, pitch, yaw = (math.radians(float(neck_deg[n])) for n in _NECK)
        self.R = _ry(float(profile["neck_origin_pitch_rad"])) @ _rx(roll) @ _ry(pitch) @ _rz(yaw)
        self.pivot = np.array(profile["neck_origin_xyz_m"], dtype=float)
        right = np.array(profile["right_camera_in_head_m"], dtype=float)
        # The left camera sits one baseline to the left of the right one (optical -x).
        left = right + _OPTICAL_TO_HEAD @ np.array([-float(profile["baseline_m"]), 0.0, 0.0])
        self._head_offset = {"left": left, "right": right}

    def origin(self, eye: str) -> np.ndarray:
        return self.R @ self._head_offset[eye] + self.pivot

    def to_camera(self, p_torso: Sequence[float], eye: str) -> np.ndarray:
        head = self.R.T @ (np.asarray(p_torso, dtype=float) - self.pivot)
        return _OPTICAL_TO_HEAD.T @ (head - self._head_offset[eye])

    def to_torso(self, p_cam: Sequence[float], eye: str = "right") -> np.ndarray:
        return self.R @ (_OPTICAL_TO_HEAD @ np.asarray(p_cam, dtype=float)
                         + self._head_offset[eye]) + self.pivot

    def project(self, p_torso: Sequence[float], eye: str) -> Optional[Tuple[float, float]]:
        """Pixel of a torso-frame point in one eye's image, or None if behind the camera."""
        x, y, z = self.to_camera(p_torso, eye)
        if z < 0.05:
            return None
        i = 0 if eye == "left" else 1
        x, y = x / z, y / z
        fx = self.warp_fx[i] if self.distorted else self.focal[i]
        if self.distorted:
            k1, k2 = self.radial[i]
            d = 1 + k1 * (x * x + y * y) + k2 * (x * x + y * y) ** 2
            x, y = x * d, y * d
        return fx * x + self.size[0] / 2, self.focal[i] * y + self.size[1] / 2

    def in_view(self, p_torso: Sequence[float], margin_px: float) -> bool:
        for eye in ("left", "right"):
            px = self.project(p_torso, eye)
            if px is None or not (margin_px <= px[0] < self.size[0] - margin_px
                                  and margin_px <= px[1] < self.size[1] - margin_px):
                return False
        return True


def half_extent_along(obj: Mapping[str, Any], direction: np.ndarray) -> float:
    """Half the object's extent along a world direction (its support function)."""
    v = quat_matrix(obj["quat_wxyz"]).T @ direction
    if obj["shape"] == "box":
        return float(np.sum(np.abs(v) * np.array(obj["size_m"]) / 2))
    return float(abs(v[2]) * obj["size_m"][2] / 2
                 + obj["radius_m"] * math.sqrt(max(0.0, 1 - v[2] * v[2])))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_limited(path: pathlib.Path, limit: int) -> bytes:
    if path.stat().st_size > limit:
        raise GradeError(f"{path.name}: larger than {limit} bytes")
    return path.read_bytes()


def load_truth(path: pathlib.Path) -> Tuple[dict, str]:
    raw = _read_limited(path, te.MAX_DOC_BYTES)
    try:
        return te.validate_truth(json.loads(raw)), _sha256(raw)
    except (json.JSONDecodeError, te.TruthError) as exc:
        raise GradeError(f"{path.name}: {exc}") from exc


def load_pair(path: pathlib.Path) -> Tuple[dict, str]:
    """A viewer single-pair bundle (.zip, checked against its manifest) or an unpacked folder.

    Returns ({"pair": ..., "analysis": ..., "depth": ...}, sha256 of the zip or "")."""
    files: Dict[str, bytes] = {}
    digest = ""
    if path.is_dir():
        for name in ("pair.json", "analysis.json", "depth.json"):
            if (path / name).is_file():
                files[name] = _read_limited(path / name, MAX_JSON_BYTES)
    else:
        raw = _read_limited(path, MAX_ZIP_BYTES)
        digest = _sha256(raw)
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                names = zf.namelist()
                if "manifest.json" not in names:
                    raise GradeError(f"{path.name}: manifest.json is missing")
                listed = {e["name"]: e["sha256"] for e in
                          json.loads(zf.read("manifest.json")).get("files", [])}
                for name in ("pair.json", "analysis.json", "depth.json"):
                    if name in names:
                        info = zf.getinfo(name)
                        if info.file_size > MAX_JSON_BYTES:
                            raise GradeError(f"{path.name}: {name} is larger than {MAX_JSON_BYTES} bytes")
                        data = zf.read(name)
                        if listed.get(name) != _sha256(data):
                            raise GradeError(f"{path.name}: {name} does not match its manifest SHA-256")
                        files[name] = data
        except zipfile.BadZipFile as exc:
            raise GradeError(f"{path.name}: not a zip file") from exc
    try:
        parsed = {n[:-5]: json.loads(b) for n, b in files.items()}
    except json.JSONDecodeError as exc:
        raise GradeError(f"{path.name}: {exc}") from exc
    return parsed, digest


def _neck_from(parsed: Mapping[str, Any]) -> Dict[str, float]:
    head = (parsed.get("depth") or {}).get("head_pose_used")
    if not head:
        head = ((parsed.get("pair") or {}).get("head_context") or {}).get("joints")
    out = {}
    for n in _NECK:
        v = (head or {}).get(n, {}).get("present_deg") if isinstance(head, dict) else None
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            raise GradeError("no neck read-back for the capture")
        out[n] = float(v)
    return out


# ---------------------------------------------------------------------------
# One layout
# ---------------------------------------------------------------------------

def check_bracket(before: Mapping[str, Any], after: Mapping[str, Any],
                  move_tol_m: float = MOVE_TOL_M) -> Tuple[bool, List[str], float]:
    """Static-scene test over the whole bracket.  (ok, reasons, largest move in metres)."""
    reasons: List[str] = []
    for key in ("scene_name", "scene_revision"):
        if before[key] != after[key]:
            reasons.append(f"{key} changed: {before[key]!r} -> {after[key]!r}")
    a = {o["id"]: o for o in before["objects"]}
    b = {o["id"]: o for o in after["objects"]}
    if set(a) != set(b):
        reasons.append("object set changed: " + ", ".join(sorted(set(a) ^ set(b))))
    worst = 0.0
    for oid in sorted(set(a) & set(b)):
        move = float(np.linalg.norm(np.subtract(a[oid]["pos_xyz"], b[oid]["pos_xyz"])))
        worst = max(worst, move)
        if move > move_tol_m:
            reasons.append(f"{oid} moved {move * 1000:.1f} mm")
    return not reasons, reasons, worst


def _err(p: np.ndarray, delta: np.ndarray, t: Mapping[str, Any]) -> dict:
    """Errors of a position `p` (delta = p - true centre) for one truth object, in mm.

    gate/centre/horizontal are magnitudes; along_ray_mm (+ = farther than the centre) and x/y/z_mm (torso frame,
    detection minus truth) are SIGNED, so a bias can be learned.  `toward_camera` is the unit vector from the truth
    centre to the camera, kept so a bias can be projected on the ray later."""
    return {"gate_mm": float(np.linalg.norm(p - t["surface"])) * 1000,
            "centre_mm": float(np.linalg.norm(delta)) * 1000,
            "along_ray_mm": float(delta @ -t["u"]) * 1000,
            "horizontal_mm": float(np.linalg.norm(delta[:2])) * 1000,
            "x_mm": float(delta[0]) * 1000, "y_mm": float(delta[1]) * 1000, "z_mm": float(delta[2]) * 1000,
            "toward_camera": [round(float(v), 6) for v in t["u"]]}


def _box_contains(box: Sequence[float], px: Optional[Tuple[float, float]]) -> bool:
    return px is not None and box[0] <= px[0] <= box[0] + box[2] and box[1] <= px[1] <= box[1] + box[3]


def grade_pair(truth: Mapping[str, Any], parsed: Mapping[str, Any], *,
               radius_m: float = MATCH_RADIUS_M, margin_px: float = IN_VIEW_MARGIN_PX) -> dict:
    """Grade one captured pair against the layout's truth.  Raises GradeError if ungradable."""
    analysis, depth = parsed.get("analysis"), parsed.get("depth")
    if not isinstance(analysis, dict) or not isinstance(analysis.get("detections"), list):
        raise GradeError("analysis.json missing (press Analyze)")
    if not isinstance(depth, dict) or not isinstance(depth.get("profile"), dict):
        raise GradeError("depth.json missing (press Measure in 3D)")
    neck = _neck_from(parsed)
    rig = Rig(depth["profile"], neck)
    base = np.array(truth["arm_base_in_world"], dtype=float)
    cam = rig.origin("right")

    dets = analysis["detections"]
    meas = {m.get("detection_index"): m for m in depth.get("measurements") or []}

    # Truth in the torso frame, with the point the viewer can be right about.
    objs = []
    for o in truth["objects"]:
        c = np.array(o["pos_xyz"], dtype=float) - base
        if not rig.in_view(c, margin_px):
            continue
        to_cam = cam - c
        dist = float(np.linalg.norm(to_cam))
        u = to_cam / dist
        objs.append({"o": o, "centre": c, "u": u, "dist": dist,
                     "surface": c + u * half_extent_along(o, u)})

    def xyz_of(i: int) -> Optional[np.ndarray]:
        m = meas.get(i) or {}
        v = m.get("xyz_torso_m")
        if m.get("status") == "measured" and isinstance(v, list) and len(v) == 3 \
                and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v):
            return np.array(v, dtype=float)
        return None

    obj_dets = [i for i, d in enumerate(dets) if d.get("label") in VOCAB]
    used_det, used_obj = set(), set()
    matched: List[Tuple[int, int, str]] = []                 # (det, truth idx, "3d"|"2d")

    pairs3d = sorted(
        (float(np.linalg.norm(xyz_of(i) - t["surface"])), i, j)
        for i in obj_dets if xyz_of(i) is not None for j, t in enumerate(objs))
    for dist, i, j in pairs3d:
        if dist <= radius_m and i not in used_det and j not in used_obj:
            used_det.add(i), used_obj.add(j), matched.append((i, j, "3d"))

    # Detections with no 3D position: match by 2D, so class and recall still count.
    pairs2d = []
    for i in obj_dets:
        if i in used_det or xyz_of(i) is not None:
            continue
        x, y, w, h = dets[i]["box_xywh"]
        for j, t in enumerate(objs):
            if j in used_obj:
                continue
            px = rig.project(t["centre"], "right")
            if _box_contains(dets[i]["box_xywh"], px):
                pairs2d.append((math.hypot(px[0] - (x + w / 2), px[1] - (y + h / 2)), i, j))
    for _, i, j in sorted(pairs2d):
        if i not in used_det and j not in used_obj:
            used_det.add(i), used_obj.add(j), matched.append((i, j, "2d"))

    matches, oov_hits, misses = [], [], []
    for i, j, via in matched:
        t, d = objs[j], dets[i]
        o = t["o"]
        if o["kind"] not in VOCAB:
            oov_hits.append({"object_id": o["id"], "kind": o["kind"], "label": d["label"],
                             "det_index": i, "on_board": o["cell"] is not None})
            continue
        rec = {"object_id": o["id"], "kind": o["kind"], "cell": o["cell"], "label": d["label"],
               "correct": d["label"] == o["kind"], "via": via, "on_board": o["cell"] is not None, "det_index": i,
               "confidence": d.get("confidence"), "distance_m": round(t["dist"], 4), "err": None}
        p = xyz_of(i)
        if p is not None:
            delta = p - t["centre"]
            rec["err"] = _err(p, delta, t)
        matches.append(rec)
    matched_objs = {j for _, j, _ in matched}
    for j, t in enumerate(objs):
        o = t["o"]
        if j in matched_objs or o["kind"] not in VOCAB:
            continue
        px = rig.project(t["centre"], "right")
        covering = [dets[i]["label"] for i in obj_dets if _box_contains(dets[i]["box_xywh"], px)]
        misses.append({"object_id": o["id"], "kind": o["kind"], "cell": o["cell"],
                       "on_board": o["cell"] is not None, "distance_m": round(t["dist"], 4),
                       "box_covers_label": covering[0] if covering else None})

    false_pos = []
    for i in obj_dets:
        if i in used_det:
            continue
        p = xyz_of(i)
        near = None
        if p is not None and objs:
            k = min(range(len(objs)), key=lambda j: np.linalg.norm(p - objs[j]["surface"]))
            near = (objs[k]["o"]["id"], round(float(np.linalg.norm(p - objs[k]["surface"])) * 1000, 1))
        false_pos.append({"det_index": i, "label": dets[i]["label"],
                          "confidence": dets[i].get("confidence"), "measured": p is not None,
                          "nearest_truth": near[0] if near else None,
                          "nearest_mm": near[1] if near else None})

    # Cells: occupant from the truth, projected centre in both eyes.
    occupant: Dict[str, List[str]] = {}
    for o in truth["objects"]:
        if o["cell"]:
            occupant.setdefault(o["cell"], []).append(o["kind"])
    cell_info = {}
    for c in truth["cells"]:
        centre = np.array([c["x"], c["y"], c["top_z"]]) - base
        if rig.in_view(centre, margin_px):
            cell_info[c["name"]] = {"centre": centre, "px": rig.project(centre, "right")}

    empty_dets = []
    assigned_empty: Dict[str, int] = {}
    for i, d in enumerate(dets):
        if d.get("label") != EMPTY:
            continue
        x, y, w, h = d["box_xywh"]
        near = [(math.hypot(v["px"][0] - (x + w / 2), v["px"][1] - (y + h / 2)), n)
                for n, v in cell_info.items() if _box_contains(d["box_xywh"], v["px"])]
        cell = min(near)[1] if near else None
        rec = {"det_index": i, "cell": cell, "confidence": d.get("confidence"),
               "correct": None if cell is None else cell not in occupant}
        if cell is not None:
            assigned_empty[cell] = assigned_empty.get(cell, 0) + 1
        empty_dets.append(rec)

    by_obj_cell = {m["cell"]: m for m in matches if m["cell"]}
    cells = {}
    for name in sorted(cell_info):
        kinds = occupant.get(name)
        if kinds:
            truth_s = "+".join(sorted(kinds))
            m = by_obj_cell.get(name)
            if m is not None:
                reported = m["label"]
                outcome = "correct" if m["correct"] else "wrong"
            elif name in assigned_empty:
                reported, outcome = EMPTY, "wrong"
            else:
                reported, outcome = "unknown", "unknown"
            if all(k not in VOCAB for k in kinds):
                outcome = "oov"
        else:
            truth_s = EMPTY
            reported = EMPTY if name in assigned_empty else "unknown"
            outcome = "correct" if reported == EMPTY else "unknown"
        cells[name] = {"truth": truth_s, "reported": reported, "outcome": outcome}

    return {"neck_deg": neck, "in_view": sorted(t["o"]["id"] for t in objs),
            "n_detections": len(dets), "matches": matches, "oov_hits": oov_hits,
            "misses": misses, "false_positives": false_pos, "empty_dets": empty_dets,
            "cells": cells, "profile_id": depth["profile"].get("id"),
            "detector_model": (analysis.get("model") or {}).get("sha256")
            or (analysis.get("model") or {}).get("name")}


def load_scene(path: pathlib.Path) -> Tuple[Any, str]:
    """A PerceivedScene document (perceived_scene.v1).  Returns (scene, sha256 of the file)."""
    raw = _read_limited(path, MAX_JSON_BYTES)
    try:
        return ps.loads(raw.decode("utf-8")), _sha256(raw)
    except (UnicodeDecodeError, ps.SceneContractError) as exc:
        raise GradeError(f"{path.name}: {exc}") from exc


def grade_scene(truth: Mapping[str, Any], scene: Any, *, neck_deg: Optional[Sequence[float]] = None,
                radius_m: float = MATCH_RADIUS_M, margin_px: float = IN_VIEW_MARGIN_PX) -> dict:
    """Grade one PerceivedScene against the layout's truth, in the same result shape as `grade_pair`.

    Differences from a viewer pair, all because the document is the builder's output rather than raw detections:
    * the gate (`radius_m`) is on `surface_point_m`, the raw stereo median, against the truth visible-surface point,
      so the same objects match whatever surface correction the builder applied; `centre_m` (after the correction)
      is graded against the true centre for centre / horizontal / along-ray / x, y, z;
    * `objects` and `off_board` are both candidate detections (the parked pool is in view);
    * there are no `empty` boxes: a cell the scene reports `empty` is an empty detection on that cell, and every
      cell outcome comes from the scene's own cell reports (`unknown` when it has no board);
    * misses carry no `box_covers_label`: the document has no boxes;
    * the head pose is the objects' `source.neck_deg`.  A scene with no object has none, so the manifest entry may give
      `neck_deg` (the pair's own read-back); when both exist they must agree.
    """
    if scene.rig.source != "sim":
        raise GradeError("only a sim PerceivedScene can be graded against simulator truth")
    for key in ("scene_sha256", "scene_revision"):
        mine, theirs = getattr(scene.rig, key), truth.get(key)
        if mine is not None and theirs is not None and mine != theirs:
            raise GradeError(f"scene {key} {mine!r} differs from the truth's {theirs!r}")
    profile = KNOWN_SIM_PROFILES.get(scene.rig.intrinsics_sha256)
    if profile is None:
        raise GradeError(f"unknown sim stereo profile {scene.rig.intrinsics_sha256!r}: the document carries no lens model")
    perceived = list(scene.objects) + list(scene.off_board)
    poses = {tuple(o.source.neck_deg) for o in perceived}
    if len(poses) > 1:
        raise GradeError("the scene's objects were captured at different head poses")
    if neck_deg is not None:
        if len(neck_deg) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in neck_deg):
            raise GradeError("neck_deg must be three finite numbers (roll, pitch, yaw)")
        if poses and max(abs(a - b) for a, b in zip(neck_deg, next(iter(poses)))) > 0.01:
            raise GradeError("the manifest's neck_deg differs from the scene's")
    elif poses:
        neck_deg = next(iter(poses))
    else:
        raise GradeError("the scene lists no object, so its head pose is unknown: give neck_deg in the manifest entry")
    neck = dict(zip(_NECK, (float(v) for v in neck_deg)))
    rig = Rig(profile, neck)
    base = np.array(truth["arm_base_in_world"], dtype=float)
    cam = rig.origin("right")

    objs = []
    for o in truth["objects"]:
        c = np.array(o["pos_xyz"], dtype=float) - base
        if not rig.in_view(c, margin_px):
            continue
        to_cam = cam - c
        dist = float(np.linalg.norm(to_cam))
        u = to_cam / dist
        objs.append({"o": o, "centre": c, "u": u, "dist": dist, "surface": c + u * half_extent_along(o, u)})

    surface = [np.array(d.surface_point_m, dtype=float) for d in perceived]
    centre = [np.array(d.centre_m, dtype=float) for d in perceived]
    used_det, used_obj = set(), set()
    matched: List[Tuple[int, int]] = []
    for dist, i, j in sorted((float(np.linalg.norm(surface[i] - t["surface"])), i, j)
                             for i in range(len(perceived)) for j, t in enumerate(objs)):
        if dist <= radius_m and i not in used_det and j not in used_obj:
            used_det.add(i), used_obj.add(j), matched.append((i, j))

    matches, oov_hits, misses = [], [], []
    for i, j in matched:
        t, d = objs[j], perceived[i]
        o = t["o"]
        if o["kind"] not in VOCAB:
            oov_hits.append({"object_id": o["id"], "kind": o["kind"], "label": d.kind,
                             "det_index": i, "on_board": o["cell"] is not None})
            continue
        matches.append({"object_id": o["id"], "kind": o["kind"], "cell": o["cell"], "label": d.kind,
                        "correct": d.kind == o["kind"], "via": "3d", "on_board": o["cell"] is not None,
                        "det_index": i, "confidence": d.kind_confidence, "distance_m": round(t["dist"], 4),
                        "err": _err_centre(surface[i], centre[i], t),
                        "bounds_provisional": d.bounds_provisional,
                        "perceived_cell": d.cell})
    for j, t in enumerate(objs):
        o = t["o"]
        if j in used_obj or o["kind"] not in VOCAB:
            continue
        misses.append({"object_id": o["id"], "kind": o["kind"], "cell": o["cell"], "on_board": o["cell"] is not None,
                       "distance_m": round(t["dist"], 4), "box_covers_label": None, "no_boxes": True})
    false_pos = []
    for i in range(len(perceived)):
        if i in used_det:
            continue
        near = None
        if objs:
            k = min(range(len(objs)), key=lambda j: np.linalg.norm(surface[i] - objs[j]["surface"]))
            near = (objs[k]["o"]["id"], round(float(np.linalg.norm(surface[i] - objs[k]["surface"])) * 1000, 1))
        false_pos.append({"det_index": i, "label": perceived[i].kind, "confidence": perceived[i].kind_confidence,
                          "measured": True, "nearest_truth": near[0] if near else None,
                          "nearest_mm": near[1] if near else None})

    occupant: Dict[str, List[str]] = {}
    for o in truth["objects"]:
        if o["cell"]:
            occupant.setdefault(o["cell"], []).append(o["kind"])
    in_view_cells = set()
    for c in truth["cells"]:
        if rig.in_view(np.array([c["x"], c["y"], c["top_z"]]) - base, margin_px):
            in_view_cells.add(c["name"])
    reported = {c.name: c.state for c in scene.cells}          # {} without a board: every cell is unknown
    empty_dets = [{"det_index": None, "cell": n, "confidence": None, "correct": n not in occupant}
                  for n in sorted(in_view_cells) if reported.get(n) == EMPTY]
    cells = {}
    for name in sorted(in_view_cells):
        state = reported.get(name, "unknown")
        kinds = occupant.get(name)
        if kinds:
            truth_s = "+".join(sorted(kinds))
            outcome = "unknown" if state == "unknown" else "correct" if state == truth_s else "wrong"
            if all(k not in VOCAB for k in kinds):
                outcome = "oov"
        else:
            truth_s = EMPTY
            outcome = "correct" if state == EMPTY else "unknown" if state == "unknown" else "wrong"
        cells[name] = {"truth": truth_s, "reported": state, "outcome": outcome}

    return {"neck_deg": neck, "in_view": sorted(t["o"]["id"] for t in objs),
            "n_detections": len(perceived), "matches": matches, "oov_hits": oov_hits,
            "misses": misses, "false_positives": false_pos, "empty_dets": empty_dets,
            "cells": cells, "profile_id": scene.rig.intrinsics_sha256,
            "detector_model": scene.pipeline.detector_model_sha256,
            "perceived": {"scene_id": scene.scene_id, "surface_correction_id": scene.pipeline.surface_correction_id,
                          "bounds_model_id": scene.pipeline.bounds_model_id, "has_board": scene.board is not None,
                          "pair_id": perceived[0].source.pair_id if perceived else None}}


def _err_centre(surface_pt: np.ndarray, centre_pt: np.ndarray, t: Mapping[str, Any]) -> dict:
    """`_err` for a perceived object: the gate on its surface point, everything else on its (corrected) centre."""
    out = _err(centre_pt, centre_pt - t["centre"], t)
    out["gate_mm"] = float(np.linalg.norm(surface_pt - t["surface"])) * 1000
    return out


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _stats(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    v = np.asarray(values, dtype=float)
    return {"n": len(v), "mean": round(float(v.mean()), 1), "median": round(float(statistics.median(v)), 1),
            "p95": round(float(np.percentile(v, 95)), 1), "max": round(float(v.max()), 1)}


def _signed_stats(values: Sequence[float]) -> dict:
    """Signed errors: the mean is the bias; min and max show the spread (a p95 of a signed value means little)."""
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    v = np.asarray(values, dtype=float)
    return {"n": len(v), "mean": round(float(v.mean()), 1), "median": round(float(statistics.median(v)), 1),
            "std": round(float(v.std()), 1), "min": round(float(v.min()), 1), "max": round(float(v.max()), 1)}


def _band(distance_m: float, edges: Sequence[float] = BAND_EDGES_M) -> str:
    lo = None
    for e in edges:
        if distance_m < e:
            return f"<{e:.2f} m" if lo is None else f"{lo:.2f}-{e:.2f} m"
        lo = e
    return f">={edges[-1]:.2f} m"


def _err_block(matches: Sequence[Mapping[str, Any]]) -> dict:
    block = {k: _stats([m["err"][k] for m in matches]) for k in
             ("gate_mm", "centre_mm", "horizontal_mm", "along_ray_mm")}
    block.update({k: _signed_stats([m["err"][k] for m in matches if k in m["err"]])
                  for k in ("x_mm", "y_mm", "z_mm")})
    block["along_ray_signed_mm"] = _signed_stats([m["err"]["along_ray_mm"] for m in matches])
    return block


def summarize(pairs: Sequence[Mapping[str, Any]], score_off_board: bool = False) -> dict:
    """Headline metrics.  Truth objects not standing on a grid cell (the parked floor pool is in
    view at board_view) are reported in `off_board` and left out of the headline unless asked."""
    every_match = [m for p in pairs for m in p["matches"]]
    every_oov = [h for p in pairs for h in p["oov_hits"]]
    every_miss = [m for p in pairs for m in p["misses"]]
    keep = (lambda r: True) if score_off_board else (lambda r: r["on_board"])
    matches = [m for m in every_match if keep(m)]
    oov = [h for h in every_oov if keep(h)]
    misses = [m for m in every_miss if keep(m)]
    fps = [f for p in pairs for f in p["false_positives"]]
    off_board = {"scored_in_headline": score_off_board,
                 "in_view": sum(1 for r in every_match + every_oov + every_miss if not r["on_board"]),
                 "correct": sum(1 for m in every_match if not m["on_board"] and m["correct"]),
                 "wrong_class": sum(1 for m in every_match if not m["on_board"] and not m["correct"]),
                 "oov_hits": sum(1 for h in every_oov if not h["on_board"]),
                 "missed": sum(1 for m in every_miss if not m["on_board"])}
    empties = [e for p in pairs for e in p["empty_dets"]]

    per_kind = {}
    for k in VOCAB:
        n = sum(1 for m in matches if m["kind"] == k) + sum(1 for m in misses if m["kind"] == k)
        correct = sum(1 for m in matches if m["kind"] == k and m["correct"])
        labelled = (sum(1 for m in matches if m["label"] == k) + sum(1 for h in oov if h["label"] == k)
                    + sum(1 for f in fps if f["label"] == k))
        per_kind[k] = {"truth_in_view": n, "correct": correct, "detections": labelled,
                       "recall": round(correct / n, 3) if n else None,
                       "precision": round(sum(1 for m in matches if m["label"] == k and m["correct"])
                                          / labelled, 3) if labelled else None}
    confusion: Dict[str, Dict[str, int]] = {k: {} for k in VOCAB}
    for m in matches:
        confusion[m["kind"]][m["label"]] = confusion[m["kind"]].get(m["label"], 0) + 1
    for m in misses:
        confusion[m["kind"]]["missed"] = confusion[m["kind"]].get("missed", 0) + 1

    per_object: Dict[str, dict] = {}
    for m in matches:
        o = per_object.setdefault(m["object_id"], {"kind": m["kind"], "seen": 0, "correct": 0,
                                                   "wrong_class": 0, "missed": 0})
        o["seen"] += 1
        o["correct" if m["correct"] else "wrong_class"] += 1
    for m in misses:
        o = per_object.setdefault(m["object_id"], {"kind": m["kind"], "seen": 0, "correct": 0,
                                                   "wrong_class": 0, "missed": 0})
        o["seen"] += 1
        o["missed"] += 1
    for o in per_object.values():
        o["recall"] = round(o["correct"] / o["seen"], 3)

    graded = [e for e in empties if e["correct"] is not None]
    empty = {"detections": len(empties), "unassigned": len(empties) - len(graded),
             "correct": sum(1 for e in graded if e["correct"]),
             "false_positive": sum(1 for e in graded if not e["correct"])}
    empty["precision"] = round(empty["correct"] / len(graded), 3) if graded else None

    cell_outcomes: Dict[str, Dict[str, int]] = {}
    for p in pairs:
        for name, c in p["cells"].items():
            tally = cell_outcomes.setdefault(name, {})
            tally[c["outcome"]] = tally.get(c["outcome"], 0) + 1

    with3d = [m for m in matches if m["err"]]
    by_cell: Dict[str, list] = {}
    by_band: Dict[str, list] = {}
    for m in with3d:
        by_cell.setdefault(m["cell"] or "(off board)", []).append(m)
        by_band.setdefault(_band(m["distance_m"]), []).append(m)
    in_vocab_matches = len(matches)
    return {
        "pairs": len(pairs),
        "per_kind": per_kind, "confusion": confusion, "per_object": per_object,
        "class_accuracy": round(sum(1 for m in matches if m["correct"]) / in_vocab_matches, 3)
        if in_vocab_matches else None,
        "false_positives": len(fps), "oov_hits": len(oov), "misses": len(misses),
        "matched_2d_only": sum(1 for m in matches if m["via"] == "2d"),
        "empty": empty, "cell_outcomes": cell_outcomes, "off_board": off_board,
        "error_3d_mm": _err_block(with3d),
        "error_3d_by_cell_mm": {c: _err_block(v) for c, v in sorted(by_cell.items())},
        "error_3d_by_distance_mm": {b: _err_block(v) for b, v in sorted(by_band.items())},
        "size_error": {"available": False,
                       "reason": "the viewer's analysis carries no size estimate"},
    }


# ---------------------------------------------------------------------------
# measured-bias.v1: a correction learned from graded rows, graded leave-one-layout-out
# ---------------------------------------------------------------------------

def bias_loo(layouts: Sequence[Mapping[str, Any]], score_off_board: bool = False) -> dict:
    """"measured-bias.v1": subtract the per-kind signed mean error (torso x, y, z) from graded positions.

    To avoid fitting and testing on the same data it is fitted leave-one-layout-out: for each accepted layout, the
    mean signed error per truth kind over every OTHER layout's rows, applied to that layout's rows.  A kind with no
    row in the other layouts has no fit; its rows are counted in `n_unfitted` and left out of `after`.  `before`
    is the same rows uncorrected, so the two blocks compare like with like.  The input rows are the headline rows
    (on board unless `score_off_board`) with a 3D position; repeated pairs of one layout are counted as they are
    in the headline, and `n_distinct` counts distinct (layout, object) sightings.
    """
    rows = []
    for lay in layouts:
        if not lay["accepted"]:
            continue
        for pair in lay["pairs"]:
            for m in pair["matches"]:
                e = m["err"]
                if e and "x_mm" in e and (score_off_board or m["on_board"]):
                    rows.append({"layout": lay["name"], "kind": m["kind"], "object_id": m["object_id"],
                                 "delta": np.array([e["x_mm"], e["y_mm"], e["z_mm"]]) / 1000.0,
                                 "u": np.array(e["toward_camera"]), "distance_m": m["distance_m"]})
    names = sorted({r["layout"] for r in rows})
    folds, before, after, unfitted = [], [], [], 0

    def errors(r, delta):
        return {"centre_mm": float(np.linalg.norm(delta)) * 1000, "horizontal_mm": float(np.linalg.norm(delta[:2])) * 1000,
                "along_ray_mm": float(delta @ -r["u"]) * 1000, "x_mm": float(delta[0]) * 1000,
                "y_mm": float(delta[1]) * 1000, "z_mm": float(delta[2]) * 1000}

    for name in names:
        fit = {}
        for k in VOCAB:
            d = [r["delta"] for r in rows if r["layout"] != name and r["kind"] == k]
            if d:
                fit[k] = np.mean(d, axis=0)
        graded = [r for r in rows if r["layout"] == name]
        folds.append({"layout": name, "n_graded": len(graded),
                      "n_fit": {k: sum(1 for r in rows if r["layout"] != name and r["kind"] == k) for k in VOCAB},
                      "bias_mm": {k: [round(float(v) * 1000, 1) for v in vec] for k, vec in fit.items()}})
        for r in graded:
            if r["kind"] not in fit:
                unfitted += 1
                continue
            before.append(errors(r, r["delta"]))
            after.append(errors(r, r["delta"] - fit[r["kind"]]))

    def block(items):
        out = {k: _stats([e[k] for e in items]) for k in ("centre_mm", "horizontal_mm", "along_ray_mm")}
        out.update({k: _signed_stats([e[k] for e in items]) for k in ("x_mm", "y_mm", "z_mm")})
        return out

    overall = {k: [round(float(v) * 1000, 1) for v in np.mean([r["delta"] for r in rows if r["kind"] == k], axis=0)]
               for k in VOCAB if any(r["kind"] == k for r in rows)}
    return {"scheme": "measured-bias.v1", "fit": "leave-one-layout-out, per truth kind, mean signed x/y/z error",
            "n_rows": len(rows), "n_distinct": len({(r["layout"], r["object_id"]) for r in rows}),
            "n_graded": len(after), "n_unfitted": unfitted, "folds": folds,
            "bias_mm_all_rows": overall, "before": block(before), "after": block(after)}


# ---------------------------------------------------------------------------
# Whole run
# ---------------------------------------------------------------------------

def _resolve(root: pathlib.Path, rel: Any) -> pathlib.Path:
    if not isinstance(rel, str) or not rel or "://" in rel:
        raise GradeError(f"bad path {rel!r} in the manifest")
    p = (root / rel).resolve()
    if root != p and root not in p.parents:
        raise GradeError(f"{rel!r} is outside the manifest's directory")
    if not p.exists():
        raise GradeError(f"{rel!r} does not exist")
    return p


def load_manifest(path: pathlib.Path) -> Tuple[dict, pathlib.Path]:
    root = path.resolve().parent
    try:
        doc = json.loads(_read_limited(path, MAX_JSON_BYTES))
    except json.JSONDecodeError as exc:
        raise GradeError(f"{path.name}: {exc}") from exc
    if not isinstance(doc, dict) or set(doc) != {"layouts"} or not isinstance(doc["layouts"], list) \
            or not doc["layouts"]:
        raise GradeError("manifest must be {\"layouts\": [...]} with at least one layout")
    names = set()
    for lay in doc["layouts"]:
        if not isinstance(lay, dict) or set(lay) != {"name", "truth_before", "truth_after", "pairs"} \
                or not isinstance(lay["pairs"], list) or not lay["pairs"]:
            raise GradeError("each layout needs exactly: name, truth_before, truth_after, pairs[]")
        if not isinstance(lay["name"], str) or not lay["name"] or lay["name"] in names:
            raise GradeError(f"layout name {lay['name']!r} is empty or repeated")
        names.add(lay["name"])
    return doc, root


def grade_run(manifest: Mapping[str, Any], root: pathlib.Path, *,
              radius_m: float = MATCH_RADIUS_M, margin_px: float = IN_VIEW_MARGIN_PX,
              move_tol_m: float = MOVE_TOL_M, score_off_board: bool = False, bias_loo_eval: bool = False) -> dict:
    layouts, rejected_pairs, ok_pairs = [], [], []
    inputs = []
    profile_ids, models, corrections = set(), set(), set()
    for lay in manifest["layouts"]:
        before, h_before = load_truth(_resolve(root, lay["truth_before"]))
        after, h_after = load_truth(_resolve(root, lay["truth_after"]))
        ok, reasons, worst = check_bracket(before, after, move_tol_m)
        entry = {"name": lay["name"], "accepted": ok, "bracket": {
            "ok": ok, "reasons": reasons, "max_move_mm": round(worst * 1000, 3),
            "scene_name": before["scene_name"], "scene_revision": before["scene_revision"],
            "sim_step_before": before["sim_step"], "sim_step_after": after["sim_step"]},
            "pairs": [], "summary": None}
        in_files = []
        forms = {isinstance(r, dict) or (isinstance(r, str) and r.endswith(".json")) for r in lay["pairs"]}
        if len(forms) > 1:
            raise GradeError(f"layout {lay['name']!r} mixes viewer bundles and PerceivedScene .json documents")
        for entry_in in lay["pairs"]:
            extra = {}
            if isinstance(entry_in, dict):       # {"scene": "x.json", "neck_deg": [roll, pitch, yaw]}
                if set(entry_in) - {"scene", "neck_deg"} or "scene" not in entry_in:
                    raise GradeError("a scene entry is a path or {\"scene\": path, \"neck_deg\": [...]}")
                extra, rel = {"neck_deg": entry_in.get("neck_deg")}, entry_in["scene"]
            else:
                rel = entry_in
            p = _resolve(root, rel)
            try:
                if True in forms:
                    scene, digest = load_scene(p)
                    in_files.append({"path": rel, "sha256": digest})
                    if not ok:
                        raise GradeError("layout rejected (scene not static)")
                    res = grade_scene(before, scene, radius_m=radius_m, margin_px=margin_px, **extra)
                    parsed = {"pair": {"pair_id": res["perceived"]["pair_id"]}}
                    corrections.add(res["perceived"]["surface_correction_id"])
                else:
                    parsed, digest = load_pair(p)
                    in_files.append({"path": rel, "sha256": digest})
                    if not ok:
                        raise GradeError("layout rejected (scene not static)")
                    res = grade_pair(before, parsed, radius_m=radius_m, margin_px=margin_px)
            except (GradeError, KeyError, TypeError, ValueError, IndexError) as exc:
                rejected_pairs.append({"layout": lay["name"], "pair": rel, "reason": str(exc)})
                continue
            res["pair"] = rel
            res["pair_id"] = (parsed.get("pair") or {}).get("pair_id")
            entry["pairs"].append(res)
            profile_ids.add(res["profile_id"]), models.add(res["detector_model"])
        if ok and entry["pairs"]:
            entry["summary"] = summarize(entry["pairs"], score_off_board)
            ok_pairs.extend(entry["pairs"])
        layouts.append(entry)
        inputs.append({"layout": lay["name"], "truth_before_sha256": h_before,
                       "truth_after_sha256": h_after, "scene_sha256": before["scene_sha256"],
                       "pairs": in_files})
    result = {
        "schema_version": SCHEMA_VERSION,
        "config": {"match_radius_m": radius_m, "in_view_margin_px": margin_px,
                   "distance_band_edges_m": list(BAND_EDGES_M), "static_move_tolerance_m": move_tol_m,
                   "gate": "detection 3D position to the truth visible-surface point",
                   "frame": "URDF torso; truth shifted by arm_base_in_world from the export",
                   "classes_graded": list(VOCAB), "score_off_board": score_off_board, "empty_graded_against": "grid cells"},
        "detector_models": sorted(m for m in models if m), "profile_ids": sorted(p for p in profile_ids if p),
        "inputs": inputs, "layouts": layouts, "rejected_pairs": rejected_pairs,
        "overall": summarize(ok_pairs, score_off_board) if ok_pairs else None,
    }
    if corrections:
        result["config"]["input_form"] = "perceived_scene.v1"
        result["config"]["surface_corrections"] = sorted(corrections)
    if bias_loo_eval:
        result["measured_bias_v1"] = bias_loo(layouts, score_off_board)
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    return "-" if v is None else str(v)


def _err_rows(block: Mapping[str, Mapping[str, Any]]) -> List[str]:
    rows = ["| group | n | gate mean | gate p95 | centre median | horizontal median | horizontal p95 | along-ray median |",
            "|---|---|---|---|---|---|---|---|"]
    for name, e in block.items():
        rows.append(f"| {name} | {e['gate_mm']['n']} | {_fmt(e['gate_mm']['mean'])} | {_fmt(e['gate_mm']['p95'])} "
                    f"| {_fmt(e['centre_mm']['median'])} | {_fmt(e['horizontal_mm']['median'])} "
                    f"| {_fmt(e['horizontal_mm']['p95'])} | {_fmt(e['along_ray_mm']['median'])} |")
    return rows


def _signed_rows(block: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Signed mean error per axis (torso x forward, y left, z up) and along the ray: a bias if it is not near 0."""
    rows = ["| group | n | x mean | y mean | z mean | along-ray mean | z min..max | along-ray min..max |",
            "|---|---|---|---|---|---|---|---|"]
    for name, e in block.items():
        if "x_mm" not in e:
            continue
        z, a = e["z_mm"], e["along_ray_signed_mm"]
        rows.append(f"| {name} | {e['x_mm']['n']} | {_fmt(e['x_mm']['mean'])} | {_fmt(e['y_mm']['mean'])} "
                    f"| {_fmt(z['mean'])} | {_fmt(a['mean'])} | {_fmt(z['min'])}..{_fmt(z['max'])} "
                    f"| {_fmt(a['min'])}..{_fmt(a['max'])} |")
    return rows


def _summary_md(s: Mapping[str, Any]) -> List[str]:
    out = [f"Pairs graded: {s['pairs']}. Class accuracy on matched objects: {_fmt(s['class_accuracy'])}. "
           f"Misses: {s['misses']}. False positives: {s['false_positives']}. OOV hits: {s['oov_hits']}. "
           f"Matched by 2D only (no 3D): {s['matched_2d_only']}.", "",
           "| class | in view | correct | detections | recall | precision |", "|---|---|---|---|---|---|"]
    for k, v in s["per_kind"].items():
        out.append(f"| {k} | {v['truth_in_view']} | {v['correct']} | {v['detections']} "
                   f"| {_fmt(v['recall'])} | {_fmt(v['precision'])} |")
    ob = s["off_board"]
    out += ["", f"Off-board truth in view (parked floor pool; {'scored above' if ob['scored_in_headline'] else 'not in the headline'}): "
            f"{ob['in_view']} sightings, {ob['correct']} correct, {ob['wrong_class']} wrong class, "
            f"{ob['oov_hits']} OOV, {ob['missed']} missed."]
    out += ["", "Confusion (truth → reported): " + "; ".join(
        f"{k}: " + ", ".join(f"{lab} {n}" for lab, n in sorted(v.items())) for k, v in s["confusion"].items() if v), ""]
    e = s["empty"]
    out += [f"`empty` detections: {e['detections']} ({e['correct']} correct, {e['false_positive']} false "
            f"positive, {e['unassigned']} not on a cell). Precision {_fmt(e['precision'])}.", "",
            "| object | kind | seen | correct | wrong class | missed | recall |", "|---|---|---|---|---|---|---|"]
    for oid, o in sorted(s["per_object"].items()):
        out.append(f"| {oid} | {o['kind']} | {o['seen']} | {o['correct']} | {o['wrong_class']} "
                   f"| {o['missed']} | {o['recall']} |")
    out += ["", "Per cell (outcome counts over pairs):", "", "| cell | outcomes |", "|---|---|"]
    for name, t in sorted(s["cell_outcomes"].items()):
        out.append(f"| {name} | " + ", ".join(f"{k} {v}" for k, v in sorted(t.items())) + " |")
    out += ["", "3D error in mm (matched objects with a 3D position). `gate` is to the visible-surface point; "
            "`centre`, `horizontal` (x, y) and signed `along-ray` (+ = farther than the centre) are to the true centre.",
            ""] + _err_rows({"all": s["error_3d_mm"]}) + [""]
    out += ["By cell:", ""] + _err_rows(s["error_3d_by_cell_mm"]) + ["", "By camera distance:", ""] \
        + _err_rows(s["error_3d_by_distance_mm"]) + [""]
    out += ["Signed error in mm, detection minus truth centre (torso x forward, y left, z up; along-ray + = farther "
            "than the centre). The mean is the bias.", ""] + _signed_rows({"all": s["error_3d_mm"]}) + [""]
    out += ["Signed, by camera distance:", ""] + _signed_rows(s["error_3d_by_distance_mm"]) + [""]
    out.append(f"Size error: not available ({s['size_error']['reason']}).")
    return out


def _bias_md(b: Mapping[str, Any]) -> List[str]:
    out = [f"Fit: {b['fit']}. Rows: {b['n_rows']} ({b['n_distinct']} distinct sightings); graded {b['n_graded']}, "
           f"no fit for {b['n_unfitted']}.", "",
           "| layout | graded | fit rows (cube / cylinder) | bias mm cube (x, y, z) | bias mm cylinder (x, y, z) |",
           "|---|---|---|---|---|"]
    for f in b["folds"]:
        out.append(f"| {f['layout']} | {f['n_graded']} | {f['n_fit'].get('cube', 0)} / {f['n_fit'].get('cylinder', 0)} "
                   f"| {_fmt(f['bias_mm'].get('cube'))} | {_fmt(f['bias_mm'].get('cylinder'))} |")
    out += ["", "Error in mm over the same rows, before and after:", "",
            "| | n | centre median | horizontal median | horizontal p95 | along-ray median | z mean |",
            "|---|---|---|---|---|---|---|"]
    for name in ("before", "after"):
        e = b[name]
        out.append(f"| {name} | {e['centre_mm']['n']} | {_fmt(e['centre_mm']['median'])} "
                   f"| {_fmt(e['horizontal_mm']['median'])} | {_fmt(e['horizontal_mm']['p95'])} "
                   f"| {_fmt(e['along_ray_mm']['median'])} | {_fmt(e['z_mm']['mean'])} |")
    return out


def _miss_line(m):
    """One report line for a miss; says plainly when no detection box covers the object."""
    if m.get("no_boxes"):
        return f"- miss: {m['object_id']} ({m['kind']}, {m['cell']}); the scene lists no object near it"
    covered = (f"a box covers it labelled {m['box_covers_label']}" if m["box_covers_label"] is not None
               else "no detection box covers it")
    return f"- miss: {m['object_id']} ({m['kind']}, {m['cell']}); {covered}"


def render_report(result: Mapping[str, Any]) -> str:
    c = result["config"]
    lines = ["# Perception stage 1 grading report (#186)", "",
             f"- Match radius: {c['match_radius_m'] * 1000:g} mm, on the truth visible-surface point; "
             f"in-view margin {c['in_view_margin_px']:g} px; distance bands {c['distance_band_edges_m']} m.",
             f"- Detector model: {', '.join(result['detector_models']) or '-'}; "
             f"stereo profile: {', '.join(result['profile_ids']) or '-'}.", ""]
    if c.get("input_form"):
        lines[-1:] = [f"- Input: PerceivedScene documents ({c['input_form']}), surface correction "
                      f"{', '.join(c['surface_corrections'])}. The gate is on `surface_point_m`; centre, horizontal, "
                      f"along-ray and the signed errors are on `centre_m`, against the true centre.", ""]
    rej = result["rejected_pairs"]
    bad = [lay for lay in result["layouts"] if not lay["accepted"]]
    if bad or rej:
        lines += ["## Rejected input", ""]
        for lay in bad:
            lines.append(f"- Layout **{lay['name']}** rejected, scene not static: " + "; ".join(lay["bracket"]["reasons"]))
        for r in rej:
            if not (r["reason"].startswith("layout rejected")):
                lines.append(f"- Pair `{r['pair']}` ({r['layout']}): {r['reason']}")
        lines.append("")
    lines += ["## Overall", ""]
    lines += _summary_md(result["overall"]) if result["overall"] else ["No pair was graded."]
    if result.get("measured_bias_v1"):
        lines += ["", "## measured-bias.v1 (leave-one-layout-out)", ""] + _bias_md(result["measured_bias_v1"])
    for lay in result["layouts"]:
        lines += ["", f"## Layout {lay['name']}", "",
                  f"Static bracket: {'ok' if lay['accepted'] else 'REJECTED'} (largest move "
                  f"{lay['bracket']['max_move_mm']} mm; scene `{lay['bracket']['scene_name']}` "
                  f"revision `{lay['bracket']['scene_revision']}`).", ""]
        lines += _summary_md(lay["summary"]) if lay["summary"] else ["Not graded."]
        for p in lay["pairs"]:
            if p["misses"] or p["false_positives"]:
                lines += ["", f"Pair `{p['pair']}`:"]
                lines += [_miss_line(m) for m in p["misses"]]
                lines += [f"- false positive: {f['label']} #{f['det_index']}, nearest truth {f['nearest_truth']} "
                          f"at {f['nearest_mm']} mm" for f in p["false_positives"]]
    lines += ["", "## Not modelled", "",
              "- Optical effects of the sim camera (exposure, background) differ from the lab's; see #182 findings 3-5.",
              "- The in-view test uses the profile's lens model; objects near the edge may flip in or out.",
              "- Each pair is an independent sample of a static layout; repeated pairs of one layout are not independent scenes."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--match-radius-mm", type=float, default=MATCH_RADIUS_M * 1000)
    ap.add_argument("--score-off-board", action="store_true",
                    help="count in-view objects that are not on a grid cell (floor pool) in the headline")
    ap.add_argument("--in-view-margin-px", type=float, default=IN_VIEW_MARGIN_PX)
    ap.add_argument("--bias-loo", action="store_true",
                    help="also grade measured-bias.v1: a per-kind signed-mean correction fitted leave-one-layout-out")
    args = ap.parse_args(argv)
    try:
        manifest, root = load_manifest(args.manifest)
        result = grade_run(manifest, root, radius_m=args.match_radius_mm / 1000,
                           margin_px=args.in_view_margin_px,
                           score_off_board=args.score_off_board, bias_loo_eval=args.bias_loo)
    except (GradeError, OSError) as exc:
        print(f"perception_grade: {exc}", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "report.md").write_text(render_report(result))
    rejected = [lay["name"] for lay in result["layouts"] if not lay["accepted"]]
    print(f"wrote {args.out_dir / 'result.json'} and report.md; "
          f"{result['overall']['pairs'] if result['overall'] else 0} pairs graded, "
          f"{len(result['rejected_pairs'])} rejected", file=sys.stderr)
    if rejected:
        print("WARNING: layouts rejected (scene not static): " + ", ".join(rejected), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
