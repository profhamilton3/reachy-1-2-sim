#!/usr/bin/env python3
"""Does a recessed (hidden) surface of a holed part show through in the robot's camera at board_view?

The segmentation buffer uses the same depth test as the colour render.  On the top of a part's full-height
faces it must name only those full-height geoms.  Reuses measure_rendered_colours.py's world and pixel placing.
"""
import os, sys, pathlib, math, tempfile
os.chdir(tempfile.mkdtemp())     # MuJoCo writes MUJOCO_LOG.TXT to the working directory
repo = pathlib.Path(__file__).resolve().parents[3]
sys.path[:0] = [str(repo/"native_mujoco"), str(repo/"src"), str(repo/"docs/reviews/2026-11-27-part-set-212")]
import mujoco, numpy as np
import measure_rendered_colours as M
doc = M.load_scene(repo/"scenes/FWDCenterLabSivaPool.yaml")
srv = M.ReachyMujocoServer(str(repo/"native_mujoco/model/reachy_1_2.xml"), "127.0.0.1", 0, scene_path=str(repo/"scenes/FWDCenterLabSivaPool.yaml"), calibration=M.load_calibration(str(M.CALIBRATION)))
m = srv._model
sim = M.SimState(m, tracked_ids=M.tracked_object_ids(doc), interactive_specs=M.interactive_specs(doc), scene_doc=doc)
def step(s):
    for _ in range(int(round(s/m.opt.timestep))):
        sim.apply_pending(); sim.control_step(m.opt.timestep); mujoco.mj_step(m, sim.data)
neck = {e.sdk_name: e.mjcf_index for e in M.JOINT_TABLE if e.sdk_name.startswith("neck")}
for j, deg in M.gaze.BOARD_VIEW.items(): sim.controller.set_goal_position(neck[j], math.radians(deg))
step(4)
flat = mujoco.Renderer(m, 480, 640)
cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "left_camera")
f = 240 / math.tan(math.radians(m.cam_fovy[cam])/2)
print("znear*extent =", m.vis.map.znear*m.stat.extent, "m   zfar*extent =", m.vis.map.zfar*m.stat.extent, "m")
for oid, wallh in (("part111_a", 0.0254), ("part121_a", 0.06985), ("part211_a", None), ("part221_a", None)):
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, oid)
    obj = next(o for o in doc["objects"] if o["id"] == oid)
    parts = obj["geometry"]["parts"]; full = max(p["size"][2] for p in parts)
    first = int(m.body_geomadr[body])
    full_geoms = {first+i for i,p in enumerate(parts) if abs(p["size"][2]-full) < 1e-9}
    tot = bad = 0
    for cell in M.CELLS:
        sim.submit_place({"object_id": oid, "cell": cell}); sim.apply_pending(); assert sim.drain_place_results()[0]["accepted"]; step(0.6)
        top, side, inner = M.face_pixels(m, sim.data, flat, body, wallh is None, full/2)
        flat.update_scene(sim.data, camera="left_camera"); flat.enable_segmentation_rendering(); seg = flat.render()[:,:,0]; flat.disable_segmentation_rendering()
        flat.enable_depth_rendering(); depth = flat.render().astype(float); flat.disable_depth_rendering()
        v,u = np.nonzero(top); d = depth[v,u]
        pc = np.stack([(u+.5-320)/f*d, -(v+.5-240)/f*d, -d],axis=1)
        loc = (sim.data.cam_xpos[cam] + pc@sim.data.cam_xmat[cam].reshape(3,3).T - sim.data.xpos[body]) @ sim.data.xmat[body].reshape(3,3)
        if wallh is not None:     # the two long walls' top face: |y| >= 13.0 mm, clear of the seam
            region = np.abs(loc[:,1]) > 0.0130
        else:                     # middle of each full-height ring segment
            th = np.arctan2(loc[:,1], loc[:,0]); off = np.abs((th + math.pi/16) % (math.pi/8) - math.pi/16)
            sector = np.round(th/(math.pi/8)).astype(int) % 2 == 0
            region = sector & (off < math.radians(6)) & (np.hypot(loc[:,0],loc[:,1]) > 0.016)
        g = seg[v[region], u[region]]
        tot += len(g); bad += int(sum(int(x) not in full_geoms for x in g))
        sim.placer.stow(oid)
    print(f"{oid}: {bad}/{tot} left-camera pixels on a full-height top face show a recessed part instead ({100*bad/max(tot,1):.1f} %), over 9 cells")
