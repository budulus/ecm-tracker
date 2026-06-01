"""Build a distributable macOS app of ECM Tracker with Nuitka.

RUN THIS ON A MAC. Nuitka cannot cross-compile — a macOS binary can only be produced
on macOS, with Xcode command-line tools installed (`xcode-select --install`).

Mirrors build.py (Windows) but emits, under dist/:
  * ECMTracker-<version>-macos.zip   — zipped ECMTracker.app (ditto-archived)
  * ECMTracker-<version>.dmg          — drag-to-Applications disk image

As on Windows the app is COMPILED while the plugins/ folder ships as plain Python
*inside the bundle* next to the binary (Contents/MacOS/plugins), so the frozen-path
logic in app/plugins/manager.py (__compiled__.containing_dir) finds the loose plugins
unchanged — no manager.py change is needed for macOS.

Usage (from repo root, on a Mac):
    uv sync                          # resolve macOS wheels first (see notes)
    uv run python build_macos.py

First-run friction we expect to iterate on (none of this is verified yet — it cannot
be tested from the Windows machine where it was written):
  * The uv lock is currently Windows-tuned (`required-environments = win32` + PyQt5/Qt
    pins in pyproject.toml). On Apple Silicon you'll likely need to relax those pins so
    `uv sync` resolves arm64 wheels. Verify before building.
  * App icon: drop an `assets/app_icon.icns` and it's used automatically. To make one
    from a 1024px PNG:
        mkdir icon.iconset
        sips -z 512 512 app.png --out icon.iconset/icon_512x512.png   # (+ other sizes)
        iconutil -c icns icon.iconset -o assets/app_icon.icns
    Without it the build still works, just with a generic icon.
  * Signing: Nuitka ad-hoc signs by default (required for arm64 apps to launch at all).
    For client distribution without Gatekeeper warnings, set a Developer ID:
        MACOS_SIGN_IDENTITY="Developer ID Application: You (TEAMID)" uv run python build_macos.py
    then notarize the .dmg separately (`xcrun notarytool submit` + `xcrun stapler staple`).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD_DIR = Path(tempfile.gettempdir()) / "ecmtracker-build"   # outside Dropbox (see build.py)
DIST_DIR = ROOT / "dist"
ENTRY = ROOT / "run.py"
ICNS = ROOT / "assets" / "app_icon.icns"

APP_NAME = "ECMTracker"        # .app bundle + binary name
DISPLAY_NAME = "ECM Tracker"   # .dmg volume name / human-facing
PUBLISHER = "Senecell"

_DONT_COPY = shutil.ignore_patterns("__pycache__", "*.pyc", "*.tmp.*")


def _version() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    return data["project"]["version"]


def _nuitka_cmd(version: str) -> list[str]:
    cmd = [
        sys.executable, "-m", "nuitka",
        "--mode=standalone",
        "--macos-create-app-bundle",
        f"--macos-app-name={APP_NAME}",
        f"--macos-app-version={version}",
        "--enable-plugin=pyqt5",
        "--include-package=app",                # whole app package ...
        "--include-package-data=app",           # ... plus its runtime-loaded SVG icons
        "--include-package=scipy",              # plugin-only: core never imports it
        "--include-package=matplotlib",         # plugin-only: core never imports it
        "--include-package-data=cv2",           # ship opencv data defensively
        "--assume-yes-for-downloads",
        "--remove-output",
        f"--output-dir={BUILD_DIR}",
        f"--output-filename={APP_NAME}",        # name of the Contents/MacOS binary
    ]
    if ICNS.exists():
        cmd.append(f"--macos-app-icon={ICNS}")
    sign = os.environ.get("MACOS_SIGN_IDENTITY")
    if sign:
        cmd.append(f"--macos-sign-identity={sign}")   # else Nuitka ad-hoc signs
    cmd.append(str(ENTRY))
    return cmd


def _stage_app() -> Path:
    """Normalize the bundle name to ECMTracker.app and drop loose plugins/ inside it."""
    apps = list(BUILD_DIR.glob("*.app"))
    if not apps:
        raise SystemExit("Nuitka produced no .app bundle in the build dir.")
    target = BUILD_DIR / f"{APP_NAME}.app"
    if apps[0] != target:
        if target.exists():
            shutil.rmtree(target)
        apps[0].rename(target)
    # Plugins sit next to the binary so manager.py's __compiled__.containing_dir logic
    # finds them with no platform branch (same contract as the Windows build).
    plugins_dest = target / "Contents" / "MacOS" / "plugins"
    if plugins_dest.exists():
        shutil.rmtree(plugins_dest)
    shutil.copytree(ROOT / "plugins", plugins_dest, ignore=_DONT_COPY)
    return target


def _make_zip(app: Path, version: str) -> Path:
    """Archive the .app with ditto (preserves symlinks/permissions/signature; plain zip can't)."""
    out = DIST_DIR / f"{APP_NAME}-{version}-macos.zip"
    out.unlink(missing_ok=True)
    subprocess.run(["ditto", "-c", "-k", "--keepParent", str(app), str(out)], check=True)
    return out


def _make_dmg(app: Path, version: str) -> Path:
    """Build a drag-to-Applications .dmg with hdiutil (built into macOS, no extra tools)."""
    out = DIST_DIR / f"{APP_NAME}-{version}.dmg"
    out.unlink(missing_ok=True)
    stage = BUILD_DIR / "dmg"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    subprocess.run(["ditto", str(app), str(stage / app.name)], check=True)  # ditto, not cp -R
    os.symlink("/Applications", stage / "Applications")                     # drag target
    subprocess.run([
        "hdiutil", "create", "-volname", DISPLAY_NAME,
        "-srcfolder", str(stage), "-ov", "-format", "UDZO", str(out),
    ], check=True)
    return out


def main() -> int:
    if sys.platform != "darwin":
        raise SystemExit("build_macos.py must be run on macOS; Nuitka cannot cross-compile.")
    version = _version()
    print(f"Building {DISPLAY_NAME} {version} for macOS ...")
    if not ICNS.exists():
        print(f"  ! {ICNS.name} not found — building with a generic icon (see header to make one).")
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    DIST_DIR.mkdir(exist_ok=True)

    subprocess.run(_nuitka_cmd(version), check=True, cwd=ROOT)

    app = _stage_app()
    print(f"  app   -> {app}")
    print(f"  zip   -> {_make_zip(app, version)}")
    print(f"  dmg   -> {_make_dmg(app, version)}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
