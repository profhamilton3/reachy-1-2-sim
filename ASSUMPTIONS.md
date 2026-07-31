# Open Assumptions — Verify Before Final Build

## 3. gRPC API package
**Status:** ✓ CONFIRMED 2026-07-31  
`reachy-sdk-api` (no "2") — installed via pip, supports Reachy 1.2.  
`reachy2-sdk-api` is reference only and has no role in this project.

---

## 1. reachy_bringup fake mode launch command
**Status:** ✓ CONFIRMED 2026-07-31 by Siva
```bash
ros2 launch reachy_bringup reachy.launch.py fake:=true start_sdk_server:=true
```

## 2. SDK server GitHub repo URL
**Status:** ✓ CONFIRMED 2026-07-31 by Siva  
- Server (ROS 2 / bringup): `https://github.com/pollen-robotics/reachy_sdk_server_2021`  
- Python client SDK: `https://github.com/pollen-robotics/reachy-sdk` (installed via pip)

## 3. URDF / robot description source
**Status:** ⚠️ BLOCKED — `pollen-robotics/reachy-description` is a private repo (clone fails with auth error).  
**Options:**
  - A) The description package may already be inside `reachy_sdk_server_2021` — confirm with Siva.
  - B) Siva provides the correct public repo URL or a local copy of the URDF.  
**Current behaviour:** Robot state publisher is set to `autorestart=false` — if no description launch is found, RViz2 opens but shows no robot model. All other services (JupyterLab, SDK server, noVNC) work normally.  
**File to update:** `supervisord.conf` → `[program:robot-state-publisher]` command

## 4. Ubuntu version on physical Reachy 1.2
**Status:** ✓ CONFIRMED 2026-07-31 — Ubuntu 20.04, ROS 2 Foxy  
**Impact:** No changes needed — Dockerfile already targets `ros:foxy`.

## 5. gRPC port
**Status:** ✓ CONFIRMED 2026-07-31 — port 50051, all default ports correct.
