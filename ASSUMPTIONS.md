# Open Assumptions — Verify Before Final Build

## 3. gRPC API repo name
**Status:** ⚠️ PROVIDED but needs verification  
**Provided URL:** `https://github.com/pollen-robotics/reachy2-sdk-api`  
**Concern:** This repo is named `reachy2-sdk-api` (Reachy 2), but our robot is Reachy 1.2. The older API repo is `pollen-robotics/reachy-sdk-api`. Confirm with Siva whether `reachy2-sdk-api` is intentionally used by `reachy_sdk_server_2021`, or if `reachy-sdk-api` (no "2") is the correct one.  
**Impact:** Wrong API version would cause gRPC protobuf mismatches between client and server.  
**File to update:** `Dockerfile` → gRPC api git clone line

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

## 3. reachy-description launch file
**Assumed:** `ros2 launch reachy_description reachy_description.launch.py`  
**Verify:** Confirm launch file exists and publishes `/robot_description` topic.  
**File to update if wrong:** `supervisord.conf` → `[program:robot-state-publisher]` command

## 4. Ubuntu version on physical Reachy 1.2
**Status:** ✓ CONFIRMED 2026-07-31 — Ubuntu 20.04, ROS 2 Foxy  
**Impact:** No changes needed — Dockerfile already targets `ros:foxy`.

## 5. gRPC port
**Assumed:** `50051` (standard gRPC default)  
**Verify:** Confirm `reachy_sdk_server` listens on 50051 in fake mode.
