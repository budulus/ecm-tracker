"""Per-user persistent settings (JSON). Used to remember parameter-dialog defaults across
sessions. Deliberately Qt-free so it can be exercised headlessly.

The config directory honors TRACKER_CONFIG_DIR (tests point this at a temp dir), then
XDG_CONFIG_HOME, then ~/.config; the file is always settings.json inside feature_tracker/.
"""
import json
import os
from typing import Optional

from app.core.atomic_io import atomic_open


def _reject_nonfinite(token: str):
    raise ValueError(f"Non-finite JSON number {token!r} is not supported")


def _config_dir() -> str:
    override = os.environ.get("TRACKER_CONFIG_DIR")
    if override:
        return override
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config"
    )
    return os.path.join(base, "feature_tracker")


def _config_path() -> str:
    return os.path.join(_config_dir(), "settings.json")


def load_settings() -> dict:
    path = _config_path()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f, parse_constant=_reject_nonfinite)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    with atomic_open(_config_path()) as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def get_section(name: str) -> Optional[dict]:
    section = load_settings().get(name)
    return section if isinstance(section, dict) else None


def update_section(name: str, data: dict) -> None:
    settings = load_settings()
    settings[name] = data
    save_settings(settings)
