#!/usr/bin/env python3
"""sim #212: place the sixteen PART objects through a RUNNING native server.

    python3 fly_parts.py --out OUT_DIR [--port 8799]

For each colour variant (family, black, one generated random seed) it starts
`mjpython native_mujoco/server.py --record` with the launcher's default
calibration, then, as a WebSocket client: turns the head to gaze.BOARD_VIEW,
places one of each design on eight cells, waits 2 s of SIM time, reads the
poses, saves the robot's left camera frame, stows them, and (for the family
scene) does the same with the other eight.  It renders free-camera views of the
same settled poses offline, including close-ups of the two holed blocks, and
composes contact_sheet.png.  Raw numbers go to results.json.  It exits non-zero
unless every part is still, upright and on its cell.

Reads no STL/STEP file.
"""
import argparse, asyncio, base64, json, math, pathlib, shutil, subprocess, sys, tempfile, time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "native_mujoco"), str(REPO / "scripts"), str(REPO / "src")]

import mujoco                                           # noqa: E402
import numpy as np                                      # noqa: E402
import websockets                                       # noqa: E402
from PIL import Image, ImageDraw                        # noqa: E402

from make_part_colour_scene import write_scene          # noqa: E402
from objects import build_scene_model_xml               # noqa: E402
from placement import cells_from_scene                  # noqa: E402
from reachy_ai.motion import gaze                       # noqa: E402
from scene_io import load_scene                         # noqa: E402

DESIGNS = ["110", "111", "120", "121", "210", "211", "220", "221"]
# Short parts on the row nearest the robot, tall ones behind, so none is hidden.
CELLS = {"110": "r1c1", "111": "r1c2", "210": "r1c3", "211": "r2c1",
         "120": "r3c1", "121": "r2c2", "220": "r2c3", "221": "r3c3"}
RANDOM_SEED = 20261127
SIM_SECONDS = 2.0
NECK_INDEX = {"neck_roll": 16, "neck_pitch": 17, "neck_yaw": 18}
CALIBRATION = REPO / "scenes" / "calibration_measured_2026_08_27.yaml"


async def drive(url, doc, batches, out, label):
    cells = cells_from_scene(doc)
    results = {}
    async with websockets.connect(url, max_size=16 * 1024 * 1024, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "hello", "protocol_version": 1, "client_id": "fly-parts-212"}))
        latest = {}

        async def recv_until(pred, timeout=30.0):
            end = time.monotonic() + timeout
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), end - time.monotonic()))
                t = msg.get("type")
                if t == "heartbeat":
                    await ws.send(json.dumps({"type": "heartbeat_ack", "echo_ns": msg.get("sent_ns", 0)}))
                elif t == "state":
                    latest["state"] = msg
                elif t == "error":
                    raise RuntimeError(msg)
                if pred(msg):
                    return msg

        await recv_until(lambda m: m.get("type") == "state")
        target = [0.0] * 21
        for joint, deg in gaze.BOARD_VIEW.items():
            target[NECK_INDEX[joint]] = math.radians(deg)
        mask = [i in NECK_INDEX.values() for i in range(21)]
        await ws.send(json.dumps({"type": "joint_command", "seq": 1, "target_rad": target, "mask": mask}))
        t0 = latest["state"]["sim_time_s"]
        await recv_until(lambda m: m.get("type") == "state" and m["sim_time_s"] >= t0 + 3.0)

        async def place(oid, cell):
            rid = f"{oid}-{cell}"
            await ws.send(json.dumps({"type": "place_object", "object_id": oid, "cell": cell, "request_id": rid}))
            ack = await recv_until(lambda m: m.get("type") == "place_ack" and m.get("request_id") == rid)
            if not ack["accepted"]:
                raise RuntimeError(f"{oid} on {cell}: {ack['error']}")
            return ack

        for batch_index, ids in enumerate(batches):
            last = None
            for oid in ids:
                last = await place(oid, CELLS[oid[4:7]])
            first_step = latest["state"]["sim_step"]
            start = await recv_until(lambda m: m.get("type") == "state" and m["sim_step"] >= last["sim_step"])
            end = await recv_until(lambda m: m.get("type") == "state"
                                   and m["sim_time_s"] >= start["sim_time_s"] + SIM_SECONDS, timeout=60)
            # A second reading 0.25 s later: moving or not is measured, not assumed.
            later = await recv_until(lambda m: m.get("type") == "state"
                                     and m["sim_time_s"] >= end["sim_time_s"] + 0.25)
            frame = await recv_until(lambda m: m.get("type") == "camera_frame" and m["camera"] == "left_camera"
                                     and m["sim_step"] >= later["sim_step"])
            (out / f"{label}_left_camera_{'ab'[batch_index]}.jpg").write_bytes(base64.b64decode(frame["jpeg_b64"]))
            poses = {o["object_id"]: o for o in end["objects"]}
            after = {o["object_id"]: o for o in later["objects"]}
            for oid in ids:
                cell = cells[CELLS[oid[4:7]]]
                obj = next(o for o in doc["objects"] if o["id"] == oid)
                half = obj["pose"]["position"][2]       # parked on the floor: z is half the height
                x, y, z = poses[oid]["pos_xyz"]
                w, qx, qy, qz = poses[oid]["quat_wxyz"]
                results[oid] = {
                    "cell": cell.name,
                    "sim_seconds_after_placement": round(end["sim_time_s"] - start["sim_time_s"], 3),
                    "xy_error_mm": round(1000 * math.hypot(x - cell.x, y - cell.y), 3),
                    "rest_height_error_mm": round(1000 * (z - (0.740 + half)), 3),
                    "tilt_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (qx*qx + qy*qy))))), 4),
                    "moved_in_next_0p25s_mm": round(1000 * math.dist((x, y, z), after[oid]["pos_xyz"]), 4),
                    "pose": poses[oid],
                }
            render_free(doc, {oid: poses[oid] for oid in ids}, out, f"{label}_{'ab'[batch_index]}")
            for oid in ids:
                rid = f"stow-{oid}"
                await ws.send(json.dumps({"type": "place_object", "object_id": oid, "cell": None, "request_id": rid}))
                await recv_until(lambda m: m.get("type") == "place_ack" and m.get("request_id") == rid)
    return results


def render_free(doc, poses, out, label):
    """Two free-camera views of the poses the live server reported, with the
    ROBOT HIDDEN: its torso and arm stand between a free camera and the board.
    The left_camera frames above are the unaltered ones."""
    robot = mujoco.MjModel.from_xml_path(str(REPO / "native_mujoco" / "model" / "reachy_1_2.xml"))
    robot_bodies = {mujoco.mj_id2name(robot, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, robot.nbody)}
    model = mujoco.MjModel.from_xml_string(build_scene_model_xml(doc))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            a = model.jnt_qposadr[j]
            data.qpos[a:a + 7] = model.qpos0[a:a + 7]
    for oid, p in poses.items():
        a = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, oid + "__j")]
        data.qpos[a:a + 7] = p["pos_xyz"] + p["quat_wxyz"]
    mujoco.mj_forward(model, data)
    for g in range(model.ngeom):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) in robot_bodies:
            model.geom_rgba[g][3] = 0.0
            model.geom_matid[g] = -1
    renderer = mujoco.Renderer(model, 480, 640)
    views = {"top": ((0.432, 0.0, 0.76), 0, -89.9, 0.78),
             "oblique": ((0.432, 0.0, 0.76), 35, -38, 0.80)}
    # The round bore of a holed block, seen from straight above and at a slant.
    for oid, height in (("part111_a", 0.0254), ("part121_a", 0.06985)):
        if oid in poses:
            x, y, z = poses[oid]["pos_xyz"]
            views[f"closeup_{oid}_top"] = ((x, y, z + height / 2), 0, -89.9, 0.14)
            views[f"closeup_{oid}_slant"] = ((x, y, z + height / 2), 20, -52, 0.16)
    for view, (lookat, azimuth, elevation, distance) in views.items():
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
        renderer.update_scene(data, camera=cam)
        Image.fromarray(renderer.render()).save(out / f"{label}_{view}.png")
    renderer.close()


def run_variant(label, scene, batches, out, port):
    mjpython = shutil.which("mjpython") or str(pathlib.Path(sys.executable).with_name("mjpython"))
    with tempfile.TemporaryDirectory() as runs:
        log = out / f"{label}_server.log"
        with open(log, "w") as fh:
            proc = subprocess.Popen(
                [mjpython, "server.py", "--scene", str(scene), "--port", str(port), "--record", runs,
                 "--calibration", str(CALIBRATION)],
                cwd=REPO / "native_mujoco", stdout=fh, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 90
            while "WebSocket server listening" not in log.read_text():
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("server did not start:\n" + log.read_text())
                time.sleep(0.2)
            results = asyncio.run(drive(f"ws://127.0.0.1:{port}", load_scene(scene), batches, out, label))
        finally:
            proc.send_signal(2)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        manifest = json.loads(next(pathlib.Path(runs).glob("run_*/manifest.json")).read_text())
    return {"scene": scene.name, "parts": results,
            "manifest_color_randomization": manifest.get("color_randomization"),
            "mujoco_version": manifest.get("mujoco_version")}


def contact_sheet(out, labels):
    """family / black / random rows, then the holed-block close-ups."""
    cell_w, cell_h, caption = 640, 480, 24
    rows = len(labels) * (cell_h + caption)
    sheet = Image.new("RGB", (2 * cell_w, rows + caption + 240), "white")
    draw = ImageDraw.Draw(sheet)
    for row, label in enumerate(labels):
        y = row * (cell_h + caption)
        draw.text((8, y + 6), f"{label}  |  left: robot left_camera frame from the live server at board_view "
                  "(unaltered)   right: free camera, same settled poses, ROBOT HIDDEN", fill="black")
        sheet.paste(Image.open(out / f"{label}_left_camera_a.jpg").convert("RGB"), (0, y + caption))
        sheet.paste(Image.open(out / f"{label}_a_oblique.png").convert("RGB"), (cell_w, y + caption))
    draw.text((8, rows + 6), "Holed blocks close up (free camera, robot hidden), the round 16-sided bore:  "
              "family PART121 from above | family PART111 slant | black PART111 from above | "
              "random PART121 slant", fill="black")
    shots = [f"{labels[0]}_a_closeup_part121_a_top.png", f"{labels[0]}_a_closeup_part111_a_slant.png",
             f"{labels[1]}_a_closeup_part111_a_top.png", f"{labels[2]}_a_closeup_part121_a_slant.png"]
    for i, name in enumerate(shots):
        crop = Image.open(out / name).convert("RGB").crop((160, 120, 480, 360))
        sheet.paste(crop, (i * 320, rows + caption))
    sheet.save(out / "contact_sheet.png", optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--port", type=int, default=8799)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    a = [f"part{d}_a" for d in DESIGNS]
    b = [f"part{d}_b" for d in DESIGNS]
    with tempfile.TemporaryDirectory(dir=REPO / "scenes") as gen:
        variants = [("family", REPO / "scenes" / "FWDCenterLabSivaPool.yaml", [a, b]),
                    ("black", REPO / "scenes" / "FWDCenterLabSivaPoolBlack.yaml", [a]),
                    (f"random_seed{RANDOM_SEED}", write_scene(RANDOM_SEED, pathlib.Path(gen)), [a])]
        report = {label: run_variant(label, scene, batches, out, args.port)
                  for label, scene, batches in variants}
    contact_sheet(out, [label for label, _, _ in variants])
    full = {"head_deg": gaze.BOARD_VIEW, "calibration": CALIBRATION.name, "variants": report}
    (out / "results.json").write_text(json.dumps(full, indent=1) + "\n")
    for label, r in report.items():
        for oid, m in r["parts"].items():
            print(f"{label:22s} {oid} {m['cell']} t={m['sim_seconds_after_placement']}s "
                  f"xy={m['xy_error_mm']}mm z={m['rest_height_error_mm']}mm tilt={m['tilt_deg']}deg "
                  f"moved={m['moved_in_next_0p25s_mm']}mm")
        print(label, "manifest color_randomization seed:",
              (r["manifest_color_randomization"] or {}).get("seed"))
    readings = [m for r in report.values() for m in r["parts"].values()]
    bad = [m for m in readings
           if not (m["sim_seconds_after_placement"] >= SIM_SECONDS and m["xy_error_mm"] < 2.0
                   and abs(m["rest_height_error_mm"]) < 0.5 and m["tilt_deg"] < 1.0
                   and m["moved_in_next_0p25s_mm"] < 0.05)]
    family = set(report["family"]["parts"])
    print(f"{len(readings)} readings, {len(bad)} not still/upright/on-cell; "
          f"family scene covers {len(family)} of 16 parts")
    print("worst: xy {:.3f} mm, height {:.3f} mm, tilt {:.4f} deg, moved {:.4f} mm".format(
        *(max(abs(m[k]) for m in readings) for k in
          ("xy_error_mm", "rest_height_error_mm", "tilt_deg", "moved_in_next_0p25s_mm"))))
    sys.exit(1 if bad or len(family) != 16 else 0)


if __name__ == "__main__":
    main()
