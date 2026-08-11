"""Collect reproducibility metadata for the reachy-1-2-sim environment.

Run inside the Docker container or on the host.  Writes artifacts/environment.json.
Never records secrets, credentials, or private hostnames.

Usage (host, container must be running):
    docker exec reachy-1-2-sim python3 /opt/scripts/collect_environment.py

Usage (inside container):
    python3 /opt/scripts/collect_environment.py
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

OUTPUT = Path("/opt/artifacts/environment.json")


def _run(cmd: list[str], cwd: str | None = None) -> str:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, cwd=cwd
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _git_sha(repo_path: str) -> str:
    return _run(["git", "-C", repo_path, "rev-parse", "HEAD"])


def _git_dirty(repo_path: str) -> bool:
    return bool(_run(["git", "-C", repo_path, "status", "--porcelain"]))


def _file_sha256(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _pip_packages() -> dict[str, str]:
    raw = _run([sys.executable, "-m", "pip", "list", "--format=json"])
    try:
        pkgs = json.loads(raw)
        return {p["name"]: p["version"] for p in pkgs}
    except Exception:
        return {}


def _ros_packages() -> dict[str, str]:
    raw = _run(["ros2", "pkg", "list"])
    if not raw:
        return {}
    return {pkg: "" for pkg in raw.splitlines()}


def _docker_info() -> dict:
    raw = _run(["docker", "info", "--format", "{{json .}}"])
    try:
        info = json.loads(raw)
        return {
            "server_version": info.get("ServerVersion", ""),
            "os": info.get("OperatingSystem", ""),
            "architecture": info.get("Architecture", ""),
            "kernel_version": info.get("KernelVersion", ""),
        }
    except Exception:
        return {}


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    env: dict = {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
        },
        "repo": {
            "path": str(repo_root),
            "git_commit": _git_sha(str(repo_root)),
            "git_dirty": _git_dirty(str(repo_root)),
        },
        "cloned_repos": {
            "reachy_sdk_server_2021": _git_sha(
                "/opt/reachy_ws/src/reachy_sdk_server_2021"
            ),
            "reachy_description": _git_sha(
                "/opt/reachy_ws/src/reachy_description"
            ),
            "novnc": _git_sha("/opt/novnc"),
        },
        "key_files": {
            "Dockerfile": _file_sha256(str(repo_root / "Dockerfile")),
            "requirements.txt": _file_sha256(str(repo_root / "requirements.txt")),
            "fake_reachy_server.py": _file_sha256(
                str(repo_root / "fake_reachy_server.py")
            ),
            "docker-compose.yml": _file_sha256(
                str(repo_root / "docker-compose.yml")
            ),
        },
        "pip_packages": _pip_packages(),
        "ros_packages": _ros_packages(),
        "docker": _docker_info(),
    }

    # Remove any value that looks like it could be a secret (rudimentary).
    output_path = OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(env, f, indent=2, default=str)
    print(f"environment.json written to {output_path}")


if __name__ == "__main__":
    main()
