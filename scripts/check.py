"""Dependency-free repository quality gate.

Runs every regression module in a clean headless configuration, then parses every Python source
file without creating bytecode. Invoke from any directory with ``uv run python scripts/check.py``.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_MODULES = (
    "tests.test_pipeline",
    "tests.test_plugins",
    "tests.test_mts_uniaxial",
    "tests.test_pressure_strain",
)
SOURCE_ROOTS = ("app", "plugins", "tests", "scripts")


def main() -> int:
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="ecmtracker-check-") as temporary_root:
        env["MPLCONFIGDIR"] = str(Path(temporary_root) / "matplotlib")
        env["XDG_CACHE_HOME"] = str(Path(temporary_root) / "cache")
        for index, module in enumerate(TEST_MODULES):
            config_dir = Path(temporary_root) / f"config-{index}"
            config_dir.mkdir()
            env["TRACKER_CONFIG_DIR"] = str(config_dir)
            print(f"\n== {module} ==", flush=True)
            subprocess.run(
                [sys.executable, "-m", module],
                cwd=ROOT,
                env=env,
                check=True,
            )

    files = sorted(
        path
        for source_root in SOURCE_ROOTS
        for path in (ROOT / source_root).rglob("*.py")
    )
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"\nParsed {len(files)} Python files. All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
