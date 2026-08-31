"""Synthetic labelled-image generator for detector training (R12-606).

Renders the scene from randomised viewpoints and object placements, and derives
bounding boxes from the segmentation buffer rather than from hand annotation.
Every box is exact by construction: it is the extent of the pixels MuJoCo
attributes to that body, not a human's estimate of where the object is.

WHY THIS EXISTS
    reachy-tabletop-ai #12 needs a detection dataset. The annotated set that
    exists is a CLASSIFICATION set -- one class per folder, no boxes -- because
    the Coral imprinting flow it was built for never needed boxes. Turning it
    into a detector dataset means annotating, and 8 images augmented to 25-30
    is nowhere near enough to train a detector. Rendering solves the volume and
    the annotation cost at once.

WHAT IS RANDOMISED, AND WHY EACH ONE MATTERS
    object placement  over the reachable grid cells, so the detector does not
                      learn "the cube is always bottom-right".
    head pitch/yaw    NOT a nicety. At the head's zero pose the near row and
                      both trays fall outside the frame entirely; they only
                      appear past about 20 deg of pitch. A generator pinned to
                      one head pose would silently never produce a sample
                      containing them.
    zoom              likewise not a nicety. The rig rails are labelled at
                      zoom OUT and at no other level, and a target is 2.4x
                      larger at zoom IN. One FOV yields a detector that only
                      works at one FOV.
    lighting          so the detector keys on shape, not on one exposure.

WHAT IS NOT RANDOMISED
    Camera intrinsics, beyond zoom. The calibration is a measurement, and
    jittering it would teach the detector a lens that does not exist.

DISTORTION
    Off by default, matching the renderer. Pass distort=True to render frames
    that match the raw camera, which is what a detector destined for the real
    feed should train on. The warp is applied to the segmentation buffer with
    the same map, so boxes stay aligned; and the margin is set automatically so
    an object near the frame edge is not lost in a dark corner.

OUTPUT
    <out_dir>/images/<stem>.jpg                     the frame
    <out_dir>/labels/<stem>.txt                     YOLO: cls cx cy w h, normalised
    <out_dir>/annotations/<stem>.json               boxes in pixels + full pose
    <out_dir>/classes.txt                           class index -> name
    <out_dir>/manifest.json                         provenance for the whole run

    The JSON sidecar carries the head pose, zoom level and object poses for
    every frame, so a sample can be reproduced or audited later. A dataset you
    cannot trace back to the state that produced it is not evidence.
"""

from __future__ import annotations

import json
import math
import pathlib
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np
from calibration import StereoCalibrationProfile, apply_to_model
from distortion import LensDistorter, auto_margin
from renderer import StereoRenderer, decode_seg
from zoom import ZoomLevel, all_levels

# Zoom levels to sample, and how often.  ZoomLevel.ZERO is deliberately EXCLUDED:
# it is the SDK's "unset" and renders identically to INTER, so sampling the enum
# uniformly silently put half of every dataset at one field of view.
#
# INTER is still weighted highest, but now on purpose: it is the only level with
# a measured barrel profile (see native_mujoco/zoom.py), so frames there are the
# most faithful.  IN and OUT carry a correct field of view and an approximate
# distortion, and are worth having precisely because the rig rails are labelled
# only at OUT and a target is 2.4x larger at IN.
_ZOOM_SAMPLE_LEVELS = [ZoomLevel.INTER, ZoomLevel.OUT, ZoomLevel.IN]
_ZOOM_SAMPLE_WEIGHTS = [0.5, 0.25, 0.25]

# A box smaller than this is a sliver of a mostly-occluded object; training on
# it teaches the detector noise.
_MIN_BOX_PIXELS = 60
# An object clipped to a thin strip at the frame edge is likewise not a sample.
_MIN_BOX_SIDE_PX = 4


@dataclass(frozen=True)
class BoxLabel:
    """One object's extent in one frame, in pixels."""
    object_id: str
    semantic_class: str
    class_index: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    pixel_count: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min + 1

    @property
    def height(self) -> int:
        return self.y_max - self.y_min + 1

    def to_yolo(self, img_w: int, img_h: int) -> str:
        cx = (self.x_min + self.x_max + 1) / 2.0 / img_w
        cy = (self.y_min + self.y_max + 1) / 2.0 / img_h
        return (f"{self.class_index} {cx:.6f} {cy:.6f} "
                f"{self.width / img_w:.6f} {self.height / img_h:.6f}")


@dataclass(frozen=True)
class FrameRecord:
    """Everything needed to reproduce or audit one rendered sample."""
    stem: str
    camera: str
    width: int
    height: int
    zoom_level: str
    fov_y_deg: float
    distorted: bool
    neck_pitch_deg: float
    neck_yaw_deg: float
    neck_roll_deg: float
    object_poses: dict[str, list[float]]
    boxes: list[BoxLabel] = field(default_factory=list)


def _bodies_for(model: mujoco.MjModel, ids: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for oid in ids:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, oid)
        if bid < 0:
            raise ValueError(f"object {oid!r} has no body in the compiled model")
        out[oid] = bid
    return out


def boxes_from_segmentation(
    seg: np.ndarray,
    body_ids: dict[str, int],
    classes: dict[str, str],
    class_index: dict[str, int],
    min_pixels: int = _MIN_BOX_PIXELS,
    min_side: int = _MIN_BOX_SIDE_PX,
) -> list[BoxLabel]:
    """Derive exact boxes from a body-ID segmentation map.

    Objects absent from the frame produce no box at all — an empty label file
    is a valid negative sample, and inventing a box for something off-screen
    would be worse than having none.
    """
    labels: list[BoxLabel] = []
    for oid, bid in body_ids.items():
        mask = seg == bid
        count = int(mask.sum())
        if count < min_pixels:
            continue
        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        if (x1 - x0 + 1) < min_side or (y1 - y0 + 1) < min_side:
            continue
        cls = classes[oid]
        labels.append(BoxLabel(
            object_id=oid, semantic_class=cls, class_index=class_index[cls],
            x_min=x0, y_min=y0, x_max=x1, y_max=y1, pixel_count=count,
        ))
    return labels


class DatasetGenerator:
    """Renders labelled frames from a compiled scene.

    Args:
        model, scene_doc: the compiled MuJoCo model and the scene YAML it came
            from.  The scene supplies semantic_class and the placement cells.
        calibration: profile whose fov_y anchors the zoom levels.
        width, height: output frame size.
        distort: apply the measured barrel distortion (with an automatic margin
            so edge objects are not lost to dark corners).
        seed: RNG seed.  Recorded in the manifest; a dataset that cannot be
            regenerated is not reproducible.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        scene_doc: dict,
        calibration: StereoCalibrationProfile,
        width: int = 640,
        height: int = 480,
        distort: bool = False,
        seed: int = 0,
        object_tag: str = "detector-target",
    ) -> None:
        self._model = model
        self._scene = scene_doc
        self._cal = calibration
        self._w, self._h = width, height
        self._distort = distort
        self._rng = random.Random(seed)
        self._seed = seed

        objects = scene_doc.get("objects") or []
        self._targets = [o["id"] for o in objects
                         if object_tag in (o.get("tags") or [])]
        if not self._targets:
            raise ValueError(f"no objects tagged {object_tag!r} in the scene")
        self._classes = {o["id"]: o["semantic_class"] for o in objects
                         if o["id"] in self._targets}
        self._class_names = sorted(set(self._classes.values()))
        self._class_index = {c: i for i, c in enumerate(self._class_names)}
        self._body_ids = _bodies_for(model, self._targets)

        # Placement cells, minus the ones the right arm cannot reach: a sample
        # showing an object where the robot could never act on it is not useful
        # for a system whose next step is grasping.
        self._cells = {
            o["id"]: list(o["pose"]["position"])
            for o in objects
            if o.get("semantic_class") == "grid.cell"
            and o["id"] not in ("cell_r3c1", "cell_r3c2")
        }

        self._movable = [o for o in objects
                         if o["id"] in self._targets
                         and (o.get("physics") or {}).get("dynamic")]

        apply_to_model(calibration, model)
        self._zoom_fov = all_levels(width, height, calibration.left_camera.fov_y_deg)

        distorters = None
        if distort:
            margin = max(auto_margin(i, width, height)
                         for i in (calibration.left_camera, calibration.right_camera))
            distorters = {
                "left_camera": LensDistorter(calibration.left_camera, width, height, margin),
                "right_camera": LensDistorter(calibration.right_camera, width, height, margin),
            }
        self._renderer = StereoRenderer(
            model, width=width, height=height,
            enable_seg=True, distorters=distorters,
        )

        self._neck = {}
        for j in ("neck_pitch", "neck_yaw", "neck_roll"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jid >= 0:
                self._neck[j] = (int(model.jnt_qposadr[jid]),
                                 float(model.jnt_range[jid][0]),
                                 float(model.jnt_range[jid][1]))

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    def _sample_head(self, data: mujoco.MjData) -> dict[str, float]:
        """Pitch is drawn from the range that actually frames the board.

        The joint allows -45.8 to +64.7 deg, but below ~15 deg the near row and
        both trays leave the frame entirely, so sampling the full range would
        spend most of it rendering the far wall.
        """
        chosen: dict[str, float] = {}
        for jname, (adr, lo, hi) in self._neck.items():
            if jname == "neck_pitch":
                val = math.radians(self._rng.uniform(15.0, 55.0))
            elif jname == "neck_yaw":
                val = math.radians(self._rng.uniform(-12.0, 12.0))
            else:
                val = math.radians(self._rng.uniform(-6.0, 6.0))
            val = max(lo, min(hi, val))
            data.qpos[adr] = val
            chosen[jname] = math.degrees(val)
        return chosen

    def _place_objects(self, data: mujoco.MjData) -> dict[str, list[float]]:
        """Put each movable target on a distinct reachable cell."""
        poses: dict[str, list[float]] = {}
        cells = list(self._cells.values())
        self._rng.shuffle(cells)
        for obj, cell in zip(self._movable, cells):
            oid = obj["id"]
            jname = f"{oid}__j"
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            adr = int(self._model.jnt_qposadr[jid])
            rest_z = obj["pose"]["position"][2]      # already surface + half height
            jitter = 0.02
            x = cell[0] + self._rng.uniform(-jitter, jitter)
            y = cell[1] + self._rng.uniform(-jitter, jitter)
            yaw = self._rng.uniform(-math.pi, math.pi)
            data.qpos[adr:adr + 3] = [x, y, rest_z]
            data.qpos[adr + 3:adr + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
            poses[oid] = [x, y, rest_z]
        return poses

    def render_one(
        self,
        data: mujoco.MjData,
        stem: str,
        camera: str = "left_camera",
        zoom: ZoomLevel | None = None,
    ) -> tuple[bytes, FrameRecord]:
        """Render one randomised sample.  Returns (jpeg_bytes, record)."""
        head = self._sample_head(data)
        poses = self._place_objects(data)
        mujoco.mj_forward(self._model, data)

        lvl = zoom or self._rng.choices(
            _ZOOM_SAMPLE_LEVELS, weights=_ZOOM_SAMPLE_WEIGHTS, k=1)[0]
        fov = self._zoom_fov[lvl]
        self._renderer.set_zoom(fov)

        frame = self._renderer.render_stereo(data, cameras=(camera,))[camera]
        seg = decode_seg(frame.seg_b64, self._h, self._w)
        boxes = boxes_from_segmentation(
            seg, self._body_ids, self._classes, self._class_index)

        return frame.jpeg_bytes, FrameRecord(
            stem=stem, camera=camera, width=self._w, height=self._h,
            zoom_level=lvl.value, fov_y_deg=round(fov, 3),
            distorted=self._distort,
            neck_pitch_deg=round(head.get("neck_pitch", 0.0), 3),
            neck_yaw_deg=round(head.get("neck_yaw", 0.0), 3),
            neck_roll_deg=round(head.get("neck_roll", 0.0), 3),
            object_poses={k: [round(v, 5) for v in p] for k, p in poses.items()},
            boxes=boxes,
        )

    def generate(
        self,
        out_dir: str | pathlib.Path,
        count: int,
        camera: str = "left_camera",
    ) -> dict:
        """Render `count` samples and write images, labels and provenance."""
        out = pathlib.Path(out_dir)
        for sub in ("images", "labels", "annotations"):
            (out / sub).mkdir(parents=True, exist_ok=True)

        (out / "classes.txt").write_text(
            "\n".join(self._class_names) + "\n", encoding="utf-8")

        data = mujoco.MjData(self._model)
        records: list[FrameRecord] = []
        empty = 0
        for i in range(count):
            stem = f"{i:06d}"
            jpeg, rec = self.render_one(data, stem, camera=camera)
            (out / "images" / f"{stem}.jpg").write_bytes(jpeg)
            (out / "labels" / f"{stem}.txt").write_text(
                "\n".join(b.to_yolo(self._w, self._h) for b in rec.boxes) + "\n"
                if rec.boxes else "", encoding="utf-8")
            (out / "annotations" / f"{stem}.json").write_text(
                json.dumps(asdict(rec), indent=2), encoding="utf-8")
            if not rec.boxes:
                empty += 1
            records.append(rec)

        manifest = {
            "format_version": 1,
            "seed": self._seed,
            "count": count,
            "camera": camera,
            "resolution": [self._w, self._h],
            "distorted": self._distort,
            "classes": self._class_names,
            "calibration_provenance": self._cal.provenance,
            "scene_name": self._scene.get("name"),
            "empty_frames": empty,
            "boxes_per_class": {
                c: sum(1 for r in records for b in r.boxes if b.semantic_class == c)
                for c in self._class_names
            },
            "zoom_histogram": {
                lv.value: sum(1 for r in records if r.zoom_level == lv.value)
                for lv in _ZOOM_SAMPLE_LEVELS
            },
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def close(self) -> None:
        self._renderer.close()
