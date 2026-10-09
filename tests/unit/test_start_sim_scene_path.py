"""Runs `test_start_sim_scene_path.sh` under pytest (issue #115 item 3).

The shell test exercises `scripts/lib/scene_path.sh` directly -- offline,
no docker -- but nothing invoked it, so a regression there passed every
suite run.
"""
import pathlib
import shutil
import subprocess

import pytest

_SCRIPT = pathlib.Path(__file__).with_name("test_start_sim_scene_path.sh")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_scene_path_shell_test_passes():
    result = subprocess.run(["bash", str(_SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout
