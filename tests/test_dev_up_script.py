"""Wave 11 smoke tests for scripts/dev_up.sh.

Full execution of the script requires Docker + a network connection to
pull images, neither of which is available in CI. These tests guard the
basics that any CI runner *can* check:

- the file exists and is executable,
- bash can parse it (``bash -n``),
- shellcheck (if installed) finds no issues,
- ``--help`` exits 0 and renders the usage block.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dev_up.sh"


def test_dev_up_script_exists_and_is_executable() -> None:
    assert SCRIPT.is_file(), f"{SCRIPT} is missing"
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} is not executable"


def test_dev_up_script_parses_with_bash() -> None:
    """``bash -n`` performs syntax checking without executing the script."""
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_dev_up_script_help_renders_usage() -> None:
    """``--help`` short-circuits before any docker/curl call."""
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "dev_up.sh" in result.stdout
    assert "--no-seed" in result.stdout


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_dev_up_script_passes_shellcheck() -> None:
    """When shellcheck is available, the script must pass it cleanly."""
    result = subprocess.run(
        ["shellcheck", "--severity=warning", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
