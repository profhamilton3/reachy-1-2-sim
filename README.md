# Reachy 1.2 Simulator

Browser-accessible Docker environment for developing and testing code against a **simulated Reachy 1.2 robot** — no physical hardware required.

> **SDK:** `reachy_sdk` (v1) — NOT `reachy2_sdk`. This image matches the physical Reachy 1.2 at IITG/FWD Center.

## What's inside

| Service | URL | Description |
|---|---|---|
| JupyterLab | http://localhost:8888 | Write and run Python SDK code |
| RViz2 / noVNC | http://localhost:6080 | 3D robot visualization in browser |
| SDK gRPC server | localhost:50051 | Fake Reachy 1.2 (no hardware needed) |

All services start automatically when the container launches.

## Quick start

```bash
# 1. Build and start
docker compose up --build

# 2. Open JupyterLab in your browser
open http://localhost:8888

# 3. Open RViz2 visualization
open http://localhost:6080

# 4. Run the Phase 0 smoke test notebook
#    → notebooks/test_motion.ipynb
```

## Stop / restart

```bash
docker compose down        # stop and remove container
docker compose up          # start again (no rebuild)
docker compose up --build  # rebuild after Dockerfile changes
```

## Connecting from host Python (optional)

You can also connect to the simulated robot from Python running on your Mac:

```python
from reachy_sdk import ReachySDK
reachy = ReachySDK(host='localhost')  # port 50051 forwarded from container
```

Install the SDK on your host:
```bash
pip install reachy-sdk==0.7.0
```

## Host-side smoke test (no Jupyter needed)

```bash
pip install reachy-sdk==0.7.0
python3 scripts/smoke_test_host.py     # must pass before any PR merge
```

Exits 0 on success, nonzero on failure. Safe to run against the live container — no destructive motion, only bounded test commands.

## Architecture

The gRPC service on port 50051 is a **custom Python fake server** (`fake_reachy_server.py`), not Pollen's `reachy_sdk_server` running in fake mode. It implements the Reachy v1 gRPC API with kinematic joint simulation (no physics or camera rendering).

```
Docker container (linux/amd64, Ubuntu 20.04, ROS 2 Foxy)
├── Xvfb :1              virtual display
├── Fluxbox              window manager
├── x11vnc → port 5900
├── noVNC/websockify → 127.0.0.1:6080    ← browser RViz2
├── fake_reachy_server.py → 127.0.0.1:50051  ← custom kinematic fake server
├── joint_state_bridge.py                publishes /joint_states to ROS
├── robot_state_publisher                publishes URDF TF for RViz2
├── rviz2                                3D visualization on virtual display
└── JupyterLab → 127.0.0.1:8888         ← browser coding
```

All processes managed by **supervisord**. Logs at `/var/log/supervisor/`.

**Planned extensions** (see `docs/ROADMAP.md`): scene YAML files, v1 camera API, native macOS MuJoCo backend for physics/rendering.

## Notebooks

| Notebook | Purpose |
|---|---|
| `test_motion.ipynb` | Phase 0 smoke test — connects to fake server, reads joints, moves arm, tests gripper and head |

## Viewing logs

```bash
# All services
docker exec reachy-1-2-sim tail -f /var/log/supervisor/supervisord.log

# Specific service
docker exec reachy-1-2-sim tail -f /var/log/supervisor/reachy_sdk_server.log
docker exec reachy-1-2-sim tail -f /var/log/supervisor/jupyter.log
docker exec reachy-1-2-sim tail -f /var/log/supervisor/rviz2.log
```

## Open assumptions

See [ASSUMPTIONS.md](ASSUMPTIONS.md) for items pending confirmation from Siva (especially the fake-mode launch command and Ubuntu version on the physical robot).

## Related

- [reachy-tabletop-ai](https://github.com/profhamilton3/reachy-tabletop-ai) — main project repo
- [Pollen Robotics reachy-sdk](https://github.com/pollen-robotics/reachy-sdk)
- [Pollen Robotics reachy-description](https://github.com/pollen-robotics/reachy-description)
