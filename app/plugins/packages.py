"""Qt-free operations on plugin package folders: listing, static metadata, zip install, migration.

A plugin is a folder ``<plugins>/<plugin_id>/`` containing ``__init__.py``; folders starting with
``.`` or ``_`` are never plugins (``_sdk`` holds the author kit). Everything here works on paths
only, so the GUI manager stays a thin presentation layer and these rules are unit-testable.
"""
from __future__ import annotations

import ast
import keyword
import logging
import re
import shutil
import stat
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)

# Archive members that are packaging noise, never part of a plugin.
_JUNK_PARTS = {"__MACOSX", "__pycache__"}
_JUNK_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}


class PluginPackageError(Exception):
    """A plugin archive is unusable or would conflict with an installed plugin."""


@dataclass(frozen=True)
class ArchivePlugin:
    """One plugin found inside a zip archive."""

    plugin_id: str  # target folder name in the plugins directory
    prefix: str  # archive path of the plugin's folder ("" for a flat archive), "/"-terminated


def is_plugin_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and not path.name.startswith((".", "_"))
        and (path / "__init__.py").is_file()
    )


def plugin_ids(root: Path) -> List[str]:
    """Folder names of every plugin package directly inside ``root`` (sorted; [] if missing)."""
    if not root.is_dir():
        return []
    return sorted(entry.name for entry in root.iterdir() if is_plugin_dir(entry))


def read_static_metadata(plugin_dir: Path) -> Dict[str, object]:
    """Best-effort ``name``/``description``/``order`` of a plugin WITHOUT importing it.

    Reads literal ``NAME``/``DESCRIPTION``/``ORDER`` class attributes from ``__init__.py``,
    preferring the class named by ``PLUGIN = ...``. Used to list disabled plugins and to label
    plugins that fail to import; returns ``{}`` when nothing can be read.
    """
    try:
        tree = ast.parse((plugin_dir / "__init__.py").read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return {}
    exported = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Name)
            and any(isinstance(t, ast.Name) and t.id == "PLUGIN" for t in node.targets)
        ):
            exported = node.value.id
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if exported is not None:
        classes = [c for c in classes if c.name == exported] or classes
    for cls in classes:
        meta: Dict[str, object] = {}
        for stmt in cls.body:
            if not (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Constant)
            ):
                continue
            key, value = stmt.targets[0].id, stmt.value.value
            if key in ("NAME", "DESCRIPTION") and isinstance(value, str):
                meta[key.lower()] = value
            elif key == "ORDER" and type(value) is int:
                meta["order"] = value
        if str(meta.get("name", "")).strip():
            return meta
    return {}


def sanitize_plugin_id(name: str) -> str:
    """Turn a folder/archive name into an importable plugin id (``my-plugin`` -> ``my_plugin``)."""
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip()).strip("_")
    if not cleaned:
        raise PluginPackageError(f"Cannot derive a plugin folder name from {name!r}.")
    if cleaned[0].isdigit():
        cleaned = "plugin_" + cleaned
    if keyword.iskeyword(cleaned):
        cleaned += "_plugin"
    return cleaned


# ---- zip archives ----------------------------------------------------------------------------
def _member_path(info: zipfile.ZipInfo) -> str | None:
    """Normalized "/"-joined path of a regular file member, None for directories and junk.

    Raises on anything that could escape the destination folder (zip-slip) or is a symlink.
    """
    raw = info.filename.replace("\\", "/")
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise PluginPackageError(f"Unsafe absolute path in archive: {info.filename!r}")
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if ".." in parts:
        raise PluginPackageError(f"Unsafe path in archive: {info.filename!r}")
    if stat.S_ISLNK(info.external_attr >> 16):
        raise PluginPackageError(f"Symbolic links are not allowed in plugin archives: {info.filename!r}")
    if info.is_dir() or not parts:
        return None
    if _JUNK_PARTS.intersection(parts) or parts[-1] in _JUNK_FILES or parts[-1].endswith(".pyc"):
        return None
    return "/".join(parts)


def _packages_under(files: Iterable[str], prefix: str) -> List[str]:
    """Names of the plugin folders directly under ``prefix`` (each must hold ``__init__.py``)."""
    found = set()
    for path in files:
        if not path.startswith(prefix):
            continue
        rest = path[len(prefix):].split("/")
        if len(rest) == 2 and rest[1] == "__init__.py" and not rest[0].startswith((".", "_")):
            found.add(rest[0])
    return sorted(found)


def find_archive_plugins(files: Iterable[str], archive_stem: str) -> List[ArchivePlugin]:
    """Locate the plugin(s) in an archive's file list.

    Accepted layouts, in order: a flat archive (``__init__.py`` at the root; named after the
    archive), one or more plugin folders at the root, or the same inside a single wrapper folder
    (as produced by "Download ZIP" on code-hosting sites).
    """
    files = list(files)
    if "__init__.py" in files:
        return [ArchivePlugin(sanitize_plugin_id(archive_stem), "")]
    prefix = ""
    names = _packages_under(files, prefix)
    if not names:
        tops = {path.split("/")[0] for path in files if "/" in path}
        if len(tops) == 1 and all("/" in path for path in files):
            prefix = next(iter(tops)) + "/"
            names = _packages_under(files, prefix)
    if not names:
        raise PluginPackageError(
            "No plugin found in the archive. A plugin is a folder containing __init__.py "
            "(zip that folder, not just its contents' parent)."
        )
    plugins = [ArchivePlugin(sanitize_plugin_id(name), f"{prefix}{name}/") for name in names]
    ids = [p.plugin_id for p in plugins]
    if len(set(ids)) != len(ids):
        raise PluginPackageError("The archive contains plugin folders whose names collide: " + ", ".join(names))
    return plugins


def inspect_zip(zip_path) -> List[ArchivePlugin]:
    """Validate a plugin archive and return the plugins it would install (nothing is written)."""
    plugins, _ = _read_archive(zip_path)
    return plugins


def _read_archive(zip_path) -> Tuple[List[ArchivePlugin], List[Tuple[zipfile.ZipInfo, str]]]:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = []
            for info in archive.infolist():
                path = _member_path(info)
                if path is not None:
                    members.append((info, path))
    except (zipfile.BadZipFile, OSError) as exc:
        raise PluginPackageError(f"Not a readable zip archive: {exc}") from exc
    plugins = find_archive_plugins([path for _, path in members], Path(zip_path).stem)
    return plugins, members


def install_zip(zip_path, plugins_dir: Path, *, overwrite: bool = False) -> List[str]:
    """Install every plugin in ``zip_path`` into ``plugins_dir``; return the installed ids.

    Files are extracted to a temporary folder outside ``plugins_dir`` first (the plugins folder
    may be cloud-synced, which locks half-written files), then each plugin folder is moved into
    place. Existing folders are replaced only with ``overwrite=True``: the previous version is
    renamed aside (a dot-folder discovery ignores) and deleted only once every plugin is in
    place; if any step fails, every folder is restored to its previous state.
    """
    plugins, members = _read_archive(zip_path)
    plugins_dir = Path(plugins_dir)
    conflicts = [p.plugin_id for p in plugins if (plugins_dir / p.plugin_id).exists()]
    if conflicts and not overwrite:
        raise PluginPackageError("Already installed: " + ", ".join(conflicts))
    plugins_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="ecm-plugin-install-"))
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for plugin in plugins:
                for info, path in members:
                    if not path.startswith(plugin.prefix):
                        continue
                    relative = PurePosixPath(path[len(plugin.prefix):]).parts
                    out = scratch.joinpath(plugin.plugin_id, *relative)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        backups: Dict[str, Path] = {}
        placed: List[str] = []
        try:
            for plugin in plugins:
                target = plugins_dir / plugin.plugin_id
                if target.exists():
                    backup = plugins_dir / f".{plugin.plugin_id}.previous-{uuid.uuid4().hex[:8]}"
                    target.rename(backup)  # Same filesystem: atomic, never a half-moved folder.
                    backups[plugin.plugin_id] = backup
                placed.append(plugin.plugin_id)  # Before the move: a cross-drive copy can fail midway.
                shutil.move(str(scratch / plugin.plugin_id), str(target))
        except BaseException:
            _roll_back(plugins_dir, placed, backups)
            raise
        for backup in backups.values():
            shutil.rmtree(backup, ignore_errors=True)
        return [p.plugin_id for p in plugins]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _roll_back(plugins_dir: Path, placed: List[str], backups: Dict[str, Path]) -> None:
    try:
        for plugin_id in placed:
            if (plugins_dir / plugin_id).exists():
                shutil.rmtree(plugins_dir / plugin_id)
        for plugin_id, backup in backups.items():
            backup.rename(plugins_dir / plugin_id)
    except OSError:
        logger.exception(
            "Could not fully roll back a failed plugin install; previous versions are kept in %s as %s",
            plugins_dir, ", ".join(b.name for b in backups.values()),
        )


def migrate_plugins(source: Path, plugins_dir: Path) -> Tuple[List[str], List[str]]:
    """Move plugin folders from an old location into ``plugins_dir``.

    Returns ``(moved, skipped)``; a folder is skipped (left untouched) when ``plugins_dir``
    already holds a plugin of the same name or it cannot be moved.
    """
    source, plugins_dir = Path(source), Path(plugins_dir)
    if not source.is_dir() or (plugins_dir.exists() and source.resolve() == plugins_dir.resolve()):
        return [], []
    moved: List[str] = []
    skipped: List[str] = []
    for plugin_id in plugin_ids(source):
        target = plugins_dir / plugin_id
        if target.exists():
            skipped.append(plugin_id)
            continue
        try:
            plugins_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source / plugin_id), str(target))
            moved.append(plugin_id)
        except OSError:
            logger.exception("Could not move plugin %s from %s", plugin_id, source)
            skipped.append(plugin_id)
    return moved, skipped
