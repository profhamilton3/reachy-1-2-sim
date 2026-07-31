# Open Assumptions — Verify Before Final Build

## 1. reachy_sdk_server fake mode launch command
**Assumed:**
```bash
ros2 launch reachy_sdk_server reachy_sdk_server.launch.py fake:=true
```
**Verify with Siva:** Is the launch file name correct? Is the fake-mode parameter `fake:=true` or something else (e.g. `fake_hardware:=true`, `--ros-args -p fake:=true`)?  
**File to update if wrong:** `supervisord.conf` → `[program:reachy-sdk-server]` command

## 2. reachy_sdk_server GitHub repo URL
**Assumed:** `https://github.com/pollen-robotics/reachy_sdk_server.git`  
**Verify:** Confirm exact repo name (underscore vs dash). Check Pollen's GitHub org.  
**File to update if wrong:** `Dockerfile` → `git clone` line

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
