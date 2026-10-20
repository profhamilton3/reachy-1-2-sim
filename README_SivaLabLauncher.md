# SivaLabLauncher — running the Reachy 1.2 simulator on a Mac

`SivaLabLauncher.sh` starts the simulator in the lab's default setup:
- the **FWDCenterLabSivaPool** scene (the October lab layout);
- **MuJoCo physics**;
- the **measured camera calibration with lens distortion**;
- the **command panel** able to move the simulated arm.

It runs two pieces that must both be up:

```text
Your Mac (native, Apple Silicon)            Docker container (linux/amd64)
┌──────────────────────────────┐            ┌──────────────────────────────────┐
│ MuJoCo physics server        │◀─ ws:8765 ─│ Reachy SDK gRPC (:50051)         │
│ native_mujoco/server.py      │            │ RViz in the browser (:6080)      │
│ started with `mjpython`      │            │ Cameras + command panel (:8080)  │
└──────────────────────────────┘            │ JupyterLab (:8888)               │
                                            └──────────────────────────────────┘
```

If either piece is missing, the sim looks "up" but does nothing useful. Most problems below come down to that.

---

## 1. One-time setup

Tested target: **macOS on Apple Silicon (M1/M2/M3/M4)**. An Intel Mac has not been tested.

### 1.1 Command-line tools and Homebrew

```bash
xcode-select --install          # git, make, compilers (skip if already installed)
# Homebrew, if you don't have it: https://brew.sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 1.2 Python for the native MuJoCo server

The server runs **on the Mac, not in Docker**, through `mjpython`, MuJoCo's macOS launcher.
- `mjpython` needs a **framework build** of Python. Homebrew's Python and the python.org installer are framework builds.
- A default **pyenv** or **conda** Python usually is **not**. See Troubleshooting T3.

```bash
brew install python@3.13        # 3.13 or 3.14; the lock file was produced on 3.14
python3.13 --version
```

### 1.3 Docker Desktop

1. Install **Docker Desktop for Mac (Apple Silicon)**: https://www.docker.com/products/docker-desktop/
2. Start it once and accept the prompts.
3. In **Settings → General**, turn on **"Use Rosetta for x86_64/amd64 emulation on Apple Silicon"**. The sim image is `linux/amd64` (ROS 2 Foxy), and Rosetta makes it much faster.
4. In **Settings → Resources**, give Docker at least **4 CPUs and 8 GB of memory**.
5. Check it works:
   ```bash
   docker version
   docker compose version     # must print v2.x; the old `docker-compose` is not used
   ```

### 1.4 Get the repository

```bash
cd ~
git clone https://github.com/profhamilton3/reachy-1-2-sim.git
cd reachy-1-2-sim
```

The repository is private. If the clone asks for credentials or fails, make sure your GitHub account has access and you're signed in (GitHub Desktop, `gh auth login`, or an SSH key).

### 1.5 Python environment for the native server

Create it **inside the repo**, so it's easy to find later:

```bash
cd ~/reachy-1-2-sim
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r native_mujoco/requirements.lock
pip install "pyyaml==6.0.3"     # the server needs PyYAML, which requirements.lock doesn't list
```

Check it:

```bash
which mjpython                      # → …/reachy-1-2-sim/.venv/bin/mjpython
mjpython -c "import mujoco, yaml, websockets, numpy, PIL; print('mujoco', mujoco.__version__)"
```

`.venv` is only for the Mac side (Docker doesn't use it), and `.gitignore` already excludes it.

### 1.6 The `.env` file (settings and keys)

`docker compose` reads a file named **`.env`** in the repo folder automatically.

```bash
cd ~/reachy-1-2-sim
cp .env.example .env
open -e .env        # or any editor
```

What to set:

| Setting | Value | Notes |
|---|---|---|
| `REACHY_SIM_BACKEND` | `mujoco-remote` | Physics. The launcher sets this too. |
| `REACHY_SIM_SCENE_FILE` | `/opt/scenes/FWDCenterLabSivaPool.yaml` | The October lab scene. The launcher sets this too. |
| `REACHY_PANEL_EXECUTOR` | `1` | Confirm in the panel moves the simulated arm. |
| `REACHY_PANEL_EPISODES` | `1` | Records each panel action under `runs/`. |
| `REACHY_PANEL_RECIPES` | `1` | Harmless until something has been promoted. |
| `REACHY_PANEL_LANGUAGE` | `1` | The natural-language panel. It needs the key below. |
| `ANTHROPIC_API_KEY` | `sk-ant-…` | Remove the leading `#` and paste your key. Without it, the panel still works with its fixed commands. |

**Keep `.env` private:**
- It is listed in `.gitignore`. **Never commit it, paste it into chats, or put it in screenshots.**
- `docker inspect` on your own Mac can show the key, which is normal for environment variables.

Settings the launcher sets on its own command line override `.env`. Anything it doesn't set, such as the key, comes from `.env`.

### 1.7 Build the Docker image (first time only)

The launcher would build the image on first use, but building it on its own first means you can watch for errors:

```bash
cd ~/reachy-1-2-sim
docker compose build          # 15–45 minutes the first time (amd64 image on Apple Silicon)
```

Rebuild only when `Dockerfile`, `requirements.txt`, `entrypoint.sh` or `supervisord.conf` changes. A plain `git pull` of Python code doesn't need a rebuild: `src/`, `web/`, `scripts/`, `scenes/` and the server files are mounted into the container live.

### 1.8 Make the launcher executable

```bash
chmod +x ~/reachy-1-2-sim/SivaLabLauncher.sh
```

---

## 2. Launching

Always launch from a terminal **with the venv active**. The launcher calls `mjpython` from your `PATH`.

```bash
cd ~/reachy-1-2-sim
source .venv/bin/activate
./SivaLabLauncher.sh
```

What it does, in order:
1. Stops any old MuJoCo server listening on port 8765.
2. Starts the native server in the background, with its log at `/tmp/reachy_mujoco_server.log`.
3. Waits up to 15 s for port 8765 to open, or fails with a pointer to the log.
4. Starts the container (`docker compose up -d`) with physics, the October scene, the panel executor and the language panel.
5. Restarts the container once, to clear stale joint goals.

The first time, macOS may ask **"Do you want the application python to accept incoming network connections?"** Click **Allow**. If you deny it, the container can't reach the server (T6).

**Open in a browser:**

| What | URL |
|---|---|
| Cameras + command panel | http://localhost:8080 |
| RViz (3D robot view) | http://localhost:6080 |
| JupyterLab | http://localhost:8888 |
| SDK (from Python) | `ReachySDK(host='localhost')`, port 50051 |

**It's working when:**
- The camera page shows live left and right images of the board.
- The robot is at REST: its hand lies on the board's right margin, outside the grid.
- The panel status reads connected, and a simple request such as "put your arm away" completes.
- RViz shows the robot and the board.

---

## 3. Stopping (kill) and restarting

| Situation | Do this |
|---|---|
| Stop everything | `./SivaLabLauncher.sh -KILL`. This takes the container down and kills whatever listens on port 8765. |
| Restart cleanly | `./SivaLabLauncher.sh -KILL`, then `./SivaLabLauncher.sh` (venv active). |
| After `git pull` | Restart cleanly. Rebuild first only if a file listed in 1.7 changed. |
| The arm is stuck or the panel refuses everything | Restart cleanly. The start pose is REST, and "put your arm away" returns there. |
| The Mac went to sleep or Docker Desktop restarted | Restart cleanly. The container keeps retrying a missing server but won't recover by itself in every case. |
| Only the container misbehaves | `docker compose restart reachy-sim`, keeping the native server. |
| Free disk space later | `docker image prune`. Old images pile up after rebuilds. |

**Check what's running:**

```bash
lsof -iTCP:8765 -sTCP:LISTEN              # the native server (should show a python process)
docker compose ps                          # the container (should be "running")
docker compose exec reachy-sim supervisorctl status   # services inside the container
```

---

## 4. Troubleshooting

Start with the two logs:

```bash
tail -n 80 /tmp/reachy_mujoco_server.log            # native MuJoCo server
docker compose logs --tail 200 reachy-sim           # container
```

**T1. `mjpython: command not found`**
- The venv isn't active in this terminal. Run `source ~/reachy-1-2-sim/.venv/bin/activate` and launch again.
- `which mjpython` should point into `.venv`.

**T2. "✖ Native server did not start — see /tmp/reachy_mujoco_server.log"**
- Read the log. The usual causes:
  - `ModuleNotFoundError: No module named 'yaml'`: run `pip install "pyyaml==6.0.3"` in the venv (1.5).
  - `No module named 'mujoco'` (or `websockets`, `numpy`, `PIL`): run `pip install -r native_mujoco/requirements.lock` in the venv.
  - A framework-build error: see T3.
  - `Address already in use`: see T4.
  - Scene or calibration file not found: run the launcher from the repo checkout. The paths are relative to the script's folder, so check that `scenes/FWDCenterLabSivaPool.yaml` exists, and run `git pull` if it doesn't.
- On a slow first start, the 15 s wait can expire before MuJoCo finishes compiling. Run the launcher again; the second start is faster.

**T3. `mjpython` complains about a framework build, or segfaults immediately**
- Use Homebrew's `python@3.13` (or python.org's installer) for the venv, not pyenv or conda.
- Delete and recreate the venv:
  ```bash
  rm -rf .venv && /opt/homebrew/bin/python3.13 -m venv .venv
  ```
  Then repeat 1.5.

**T4. Port already in use (8765, 8080, 6080, 8888 or 50051)**
- Find the owner: `lsof -iTCP:<port> -sTCP:LISTEN`.
- Common causes: a local JupyterLab on 8888, an old sim container, or another copy of the server.
- Stop it, or run `./SivaLabLauncher.sh -KILL` and `docker ps` to find stray containers (`docker stop <id>`).

**T5. `xargs: illegal option -- r` in the launcher's output**
- Some macOS versions' `xargs` don't accept `-r`. The launcher ignores the error (`|| true`), but an old server might not get killed.
- Kill it by hand, then launch again:
  ```bash
  kill -9 $(lsof -tiTCP:8765 -sTCP:LISTEN)
  ```

**T6. The camera page is blank or shows "degraded" or "disconnected"; RViz shows the robot but nothing moves**
The container is up, but it can't reach the native server.
- **Is the server running?** `lsof -iTCP:8765 -sTCP:LISTEN`. If not, see T2.
- **The macOS firewall:** System Settings → Network → Firewall → Options. Allow incoming connections for `python`/`mjpython`, or turn off "Block all incoming connections".
- **Docker Desktop must be the engine.** The container reaches your Mac at `host.docker.internal`, which Docker Desktop provides; Colima or another engine may need extra setup.
- **Order matters:** restart cleanly so the server starts before the container (section 3).

**T7. The image build fails or takes forever**
- Check Rosetta (1.3) and the memory setting (at least 8 GB).
- Retry `docker compose build`. Network hiccups during `apt`/`pip` are the usual cause.
- Stuck on old layers? Run `docker compose build --no-cache`. This is slow, but clean.
- `no space left on device`: run `docker system df`, then `docker image prune` (or `docker system prune` if you're sure).

**T8. `Cannot connect to the Docker daemon`**
- Docker Desktop isn't running. Start it and wait for "Engine running", then launch again.

**T9. `docker: 'compose' is not a docker command`**
- Docker Desktop is too old, or only the legacy `docker-compose` is installed. Update Docker Desktop.

**T10. The panel's language requests don't work, but its fixed commands do**
- The key is missing or wrong in `.env` (`ANTHROPIC_API_KEY=…`, without a leading `#`). Fix it, then restart cleanly.
- The container log mentions the language adapter at start-up if no key was found.

**T11. Jupyter or the SDK can't connect to `localhost:50051`**
- The container isn't running (`docker compose ps`), or port 50051 is taken (T4).
- From host Python, use the pinned SDK: `pip install reachy-sdk==0.7.0`, in a separate venv from `.venv` if you like.

**T12. The robot starts in a strange pose, or the arm is "inside" the board**
- You're on an old checkout. Run `git pull` on `main`, then restart cleanly.
- The October layout (PR #194) starts the arm at REST on the board's right margin.

**T13. Permission denied running the launcher**
- Run `chmod +x SivaLabLauncher.sh`, or run it as `bash SivaLabLauncher.sh`.

**T14. Everything looks up, but behaviour is old**
- The container mounts code live, but the native server only reloads on restart. Restart cleanly after any `git pull`.

**Still stuck?** Collect these and send them to the team. Remove your API key from anything you share.

```bash
sw_vers; uname -m
docker version; docker compose version
which mjpython; mjpython -c "import mujoco; print(mujoco.__version__)"
git -C ~/reachy-1-2-sim log --oneline -1
tail -n 80 /tmp/reachy_mujoco_server.log
docker compose logs --tail 200 reachy-sim
```

---

## Appendix A — Using Docker's AI assistant (Gordon)

Docker Desktop includes an AI assistant called **Ask Gordon**. It can read your containers, images and logs and explain errors in plain language. It has been genuinely helpful for problems on the Docker side of this setup.

**Turn it on:**
- In Docker Desktop, open **Settings**, find Ask Gordon (under **Beta features**, called "Features in development" in some versions), and enable it.
- Menu names change between Docker Desktop releases. If you can't find it, search Docker Desktop's settings for "Gordon".

**Use it:**
- **In Docker Desktop:** open the **Ask Gordon** panel and type a question.
- **In a terminal**, if your version supports the CLI:
  ```bash
  docker ai "why is my reachy-1-2-sim container restarting?"
  ```

**Good questions for this setup:**
- "Look at the logs of container reachy-1-2-sim and tell me why it can't connect to ws://host.docker.internal:8765."
- "Why does `docker compose build` fail for this project's Dockerfile?"
- "Which process is using port 8080 / 6080 / 8888 / 50051 on my Mac?"
- "My image is linux/amd64 on Apple Silicon. Is Rosetta emulation on, and is it being used?"
- "How much disk do my Docker images use, and what's safe to prune?"

**Cautions:**
- Gordon sees the Docker side only. It can't see the **native MuJoCo server** on your Mac. For that, read `/tmp/reachy_mujoco_server.log` (T2) yourself.
- **Never paste your `.env` or API key** into Gordon or any chat.
- Read a suggested command before you run it, especially anything with `prune`, `rm` or `--force`.
- If its advice conflicts with this README on project-specific points (ports, scene, launcher steps), follow this README.

---

## Appendix B — Quick reference

```bash
# first time: sections 1.1–1.8
cd ~/reachy-1-2-sim && source .venv/bin/activate

./SivaLabLauncher.sh            # start
./SivaLabLauncher.sh -KILL      # stop

tail -f /tmp/reachy_mujoco_server.log        # native server log
docker compose logs -f reachy-sim            # container log
lsof -iTCP:8765 -sTCP:LISTEN                 # is the native server up?
docker compose ps                            # is the container up?
```
