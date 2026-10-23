"""Native Bend engine parity: compile bend/main.bend, run it on the parity fixture, compare with PyTorch."""
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "parity"
# GROOT_BEND overrides the compiler command, e.g. "bun /path/to/bend/bend2/main.ts"
BEND = shlex.split(os.environ.get("GROOT_BEND") or shutil.which("bend") or str(Path.home() / ".bend" / "bin" / "bend"))


@pytest.mark.skipif(not Path(BEND[-1]).exists() or not (FIXTURE / "model.gbw").exists(), reason="bend toolchain or fixture missing")
def test_native_forward_parity(tmp_path):
    binary = tmp_path / "groot_native"
    build = subprocess.run([*BEND, str(ROOT / "bend" / "main.bend"), "-o", str(binary)], capture_output=True, text=True,
                           timeout=600, env={**os.environ, "BEND_NO_TELEMETRY": "1"})
    assert binary.exists(), build.stdout + build.stderr
    run = subprocess.run([str(binary), str(FIXTURE / "model.gbw"), str(FIXTURE / "req.bin"), "--threads", "4", "--gpu", "off"],
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    out = np.array([float(x) for x in run.stdout.split()])
    ref = np.load(FIXTURE / "ref.npy")
    assert out.shape == ref.shape
    assert np.abs(out - ref).max() < 2e-3, np.abs(out - ref).max()
