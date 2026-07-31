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
**Status:** ⚠️ BLOCKED — `pollen-robotics/reachy-description` is a private repo.  
RViz2 starts but shows no robot model. All other services work normally.  
To resolve: Siva to provide a public URDF source or copy the description package into this repo.  
Update `supervisord.conf` `[program:robot-state-publisher]` command when resolved.

## 6. reachy_bringup fake mode
**Status:** ✓ RESOLVED 2026-07-31 — NOT needed in Docker.  
`reachy_bringup` is a package on the physical Reachy only (not in `reachy_sdk_server_2021`).  
Replaced by `fake_reachy_server.py`: standalone Python gRPC server on port 50051 implementing
the full reachy-sdk-api v1 surface. Phase 0 smoke tests confirmed passing.

## 4. Ubuntu version on physical Reachy 1.2
**Status:** ✓ CONFIRMED 2026-07-31 — Ubuntu 20.04, ROS 2 Foxy  
**Impact:** No changes needed — Dockerfile already targets `ros:foxy`.

## 5. gRPC port
**Status:** ✓ CONFIRMED 2026-07-31 — port 50051, all default ports correct.
