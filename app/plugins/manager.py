"""Plugin discovery, enable/disable state, installation, and lifecycle.

Every plugin lives in ONE folder, ``plugins/`` next to the application (the repo root in a source
checkout, the executable's directory in a frozen build). Plugins that ship with the app and ones a
user installs are treated identically: each is a package folder imported as ``plugins.<folder>``
that exposes a :class:`~app.plugins.api.TrackerPlugin` subclass via a module-level
``PLUGIN = MyPlugin`` (a lone subclass is auto-detected as a fallback).

Each plugin is enabled or disabled from the Plugin Manager window (Plugins → Manage Plugins…);
the choice persists in ``settings.json`` and new plugins start enabled. Disabled plugins are never
imported (their name/description are read statically) and get no side-pane button. The manager
builds the side-pane launch buttons and the Plugins menu, installs plugins from zip archives,
instantiates plugins on demand, keeps their windows alive, and isolates failures so one broken
plugin never takes down the app.
"""
from __future__ import annotations

import importlib
import inspect
import logging
import shutil
import sys
import traceback
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Type

from PySide6.QtCore import QFile, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QWidget,
)

from app.core import settings
from app.plugins import packages
from app.plugins.api import API_VERSION, PluginContext, TrackerPlugin
from app.plugins.manager_dialog import PluginManagerDialog

logger = logging.getLogger(__name__)

def _project_root() -> Path:
    """Locate the directory that holds the ``plugins/`` folder.

    In a source checkout that's the repo root, two levels up from this file
    (.../tracker/app/plugins/manager.py -> .../tracker). In a Nuitka-standalone
    build the source tree is gone and ``plugins/`` ships next to the executable,
    so anchor on the compiled binary's directory (``__compiled__.containing_dir``)
    — a compiled module's ``__file__`` points inside the baked-in package tree,
    not the dist folder. Nuitka does not set ``sys.frozen``, hence the
    ``__compiled__`` probe.
    """
    compiled = globals().get("__compiled__")
    if compiled is not None:
        return Path(compiled.containing_dir)
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _project_root()
PLUGINS_DIR = PROJECT_ROOT / "plugins"  # the one and only plugin folder
SETTINGS_SECTION = "plugins"  # settings.json: {"disabled": [plugin_id, ...]}
TRASH_NAME = "Recycle Bin" if sys.platform == "win32" else "Trash"


def _legacy_user_plugins_dir() -> Path:
    """Per-user plugin folder of earlier releases; its plugins are moved into PLUGINS_DIR."""
    return Path(settings._config_dir()) / "plugins"


def _disabled_ids() -> Set[str]:
    ids = (settings.get_section(SETTINGS_SECTION) or {}).get("disabled", [])
    return {i for i in ids if isinstance(i, str)} if isinstance(ids, list) else set()


def _save_disabled_ids(ids: Iterable[str]) -> None:
    section = settings.get_section(SETTINGS_SECTION) or {}
    section["disabled"] = sorted(set(ids))
    settings.update_section(SETTINGS_SECTION, section)


def _plugin_traceback(exc: BaseException) -> str:
    """Format a load error, keeping only frames inside the plugins folder (the author's code)
    plus the exception itself — host and importlib frames would bury the failing plugin line."""
    def in_plugins(filename: str) -> bool:
        try:
            return Path(filename).resolve().is_relative_to(PLUGINS_DIR.resolve())
        except (OSError, ValueError):
            return False

    report = traceback.TracebackException.from_exception(exc)
    report.stack = traceback.StackSummary.from_list(
        [frame for frame in report.stack if in_plugins(frame.filename)]
    )
    return "".join(report.format())


def _move_to_trash(path: Path) -> bool:
    """Move a folder to the OS trash / Recycle Bin; False if the platform refused."""
    return bool(QFile.moveToTrash(str(path)))


def _clear_layout(layout) -> None:
    """Delete all widgets currently in a layout (used to rebuild the plugins panel)."""
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


@dataclass
class PluginRecord:
    """One discovered plugin folder."""

    plugin_id: str  # the folder name, also the settings key suffix
    name: str
    description: str = ""
    cls: Optional[Type[TrackerPlugin]] = None
    error: Optional[str] = None  # import/discovery error message, if any
    instance: Optional[TrackerPlugin] = None
    window: Optional[QWidget] = field(default=None, repr=False)
    context: Optional[PluginContext] = field(default=None, repr=False)
    path: Optional[Path] = None  # the plugin's folder
    enabled: bool = True  # False: listed in the manager only, never imported
    order: int = TrackerPlugin.ORDER


def _sort_key(rec: PluginRecord):
    # ORDER (ascending), ties broken by display name; the side pane and manager share it.
    return (rec.order, rec.name.lower())


class PluginManager:
    """Owns the set of installed plugins, their enabled state, and their menu/windows."""

    def __init__(self, main_window):
        self._window = main_window
        self._records: Dict[str, PluginRecord] = {}
        self._menu: Optional[QMenu] = None
        self._panel_layout = None
        self._dialog: Optional[PluginManagerDialog] = None

    @property
    def plugins_dir(self) -> Path:
        return PLUGINS_DIR

    def records(self) -> List[PluginRecord]:
        """Every plugin in the plugins folder (enabled or not), in side-pane order."""
        return list(self._records.values())

    # ---- discovery ------------------------------------------------------
    def discover(self) -> List[PluginRecord]:
        """(Re)scan the plugins folder and return the records (also stored on the manager).

        Enabled plugins are imported; disabled ones only have their metadata read."""
        self._unload_all()
        self._records.clear()
        self._migrate_legacy_plugins()
        self._ensure_package()
        disabled = _disabled_ids()
        records = [
            self._disabled_record(pid) if pid in disabled else self._load_record(pid)
            for pid in packages.plugin_ids(PLUGINS_DIR)
        ]
        for rec in sorted(records, key=_sort_key):
            self._records[rec.plugin_id] = rec
        return self.records()

    @staticmethod
    def _ensure_package() -> None:
        """Make ``plugins`` an importable package rooted at PLUGINS_DIR.

        Standalone Nuitka wipes ``sys.path`` and ignores ``PYTHONPATH``, and the folder may lack
        an ``__init__.py``, so the package module is created when the import fails; its
        ``__path__`` is pinned to PLUGINS_DIR either way."""
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        importlib.invalidate_caches()
        try:
            package = importlib.import_module("plugins")
        except ModuleNotFoundError as exc:
            if exc.name != "plugins":
                raise
            package = types.ModuleType("plugins")
            package.__package__ = "plugins"
            sys.modules["plugins"] = package
        package.__path__ = [str(PLUGINS_DIR)]

    def _migrate_legacy_plugins(self) -> None:
        legacy = _legacy_user_plugins_dir()
        moved, skipped = packages.migrate_plugins(legacy, PLUGINS_DIR)
        if moved:
            self._notify(f"Moved plugin(s) {', '.join(moved)} from {legacy} into {PLUGINS_DIR}.")
        if skipped:
            self._notify(
                f"Plugin folder(s) {', '.join(skipped)} in {legacy} were not moved: a plugin with "
                f"the same name is already in {PLUGINS_DIR}. Merge or delete them by hand."
            )

    def _load_record(self, plugin_id: str) -> PluginRecord:
        path = PLUGINS_DIR / plugin_id
        try:
            module = importlib.import_module(f"plugins.{plugin_id}")
            cls = self._find_plugin_class(module)
            if cls is None:
                return self._failed_record(
                    plugin_id, "No TrackerPlugin subclass found (expose PLUGIN = YourClass)."
                )
            if type(cls.ORDER) is not int:
                raise TypeError("ORDER must be an integer")
            if not isinstance(cls.NAME, str) or not cls.NAME.strip():
                raise TypeError("NAME must be a nonempty string")
            if not isinstance(cls.DESCRIPTION, str):
                raise TypeError("DESCRIPTION must be a string")
            if type(cls.API_VERSION) is not int or cls.API_VERSION != API_VERSION:
                raise ValueError(f"Unsupported API_VERSION; host supports {API_VERSION}")
            return PluginRecord(
                plugin_id, cls.NAME, cls.DESCRIPTION, cls=cls, path=path, order=cls.ORDER
            )
        except Exception as exc:
            return self._failed_record(plugin_id, _plugin_traceback(exc))

    @staticmethod
    def _failed_record(plugin_id: str, error: str) -> PluginRecord:
        path = PLUGINS_DIR / plugin_id
        meta = packages.read_static_metadata(path)
        return PluginRecord(
            plugin_id, str(meta.get("name", plugin_id)), str(meta.get("description", "")),
            error=error, path=path, order=int(meta.get("order", TrackerPlugin.ORDER)),
        )

    @staticmethod
    def _disabled_record(plugin_id: str) -> PluginRecord:
        path = PLUGINS_DIR / plugin_id
        meta = packages.read_static_metadata(path)
        return PluginRecord(
            plugin_id, str(meta.get("name", plugin_id)), str(meta.get("description", "")),
            path=path, enabled=False, order=int(meta.get("order", TrackerPlugin.ORDER)),
        )

    @staticmethod
    def _find_plugin_class(module) -> Optional[Type[TrackerPlugin]]:
        explicit = getattr(module, "PLUGIN", None)
        if explicit is not None:
            if inspect.isclass(explicit) and explicit is not TrackerPlugin and issubclass(explicit, TrackerPlugin):
                return explicit
            raise TypeError("PLUGIN must name a TrackerPlugin subclass")
        candidates = set()
        for obj in vars(module).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, TrackerPlugin)
                and obj is not TrackerPlugin
                and obj.__module__.startswith(module.__name__)
            ):
                candidates.add(obj)
        if len(candidates) > 1:
            raise ValueError("Ambiguous plugin classes; expose PLUGIN = YourClass.")
        return next(iter(candidates), None)

    def _rescan(self, plugin_ids: Iterable[str], disabled: Optional[Set[str]] = None) -> None:
        """Re-read individual plugins from disk (after install/remove/toggle), leaving the rest
        alone. ``disabled`` overrides the persisted set (so a failed settings write still
        applies the user's choice for this session)."""
        self._ensure_package()
        disabled = _disabled_ids() if disabled is None else disabled
        present = set(packages.plugin_ids(PLUGINS_DIR))  # Real folder-name case, not exists().
        for plugin_id in plugin_ids:
            rec = self._records.pop(plugin_id, None)
            if rec is not None:
                self._unload(rec)
            self._purge_modules(plugin_id)
            if plugin_id in present:
                self._records[plugin_id] = (
                    self._disabled_record(plugin_id) if plugin_id in disabled
                    else self._load_record(plugin_id)
                )
        self._records = {r.plugin_id: r for r in sorted(self._records.values(), key=_sort_key)}
        self._refresh_views()

    @staticmethod
    def _purge_modules(plugin_id: str) -> None:
        """Drop a plugin's cached modules so its next import reads the files on disk."""
        prefix = f"plugins.{plugin_id}"
        for name in [m for m in sys.modules if m == prefix or m.startswith(prefix + ".")]:
            del sys.modules[name]

    # ---- enable / disable -----------------------------------------------
    def set_enabled(self, plugin_id: str, enabled: bool) -> Optional[PluginRecord]:
        """Enable (import it + show its button) or disable (unload it + hide its button) one
        plugin and persist the choice. Other plugins and their open windows are untouched."""
        rec = self._records.get(plugin_id)
        if rec is None or rec.enabled == enabled:
            return rec
        self._rescan([plugin_id], self._persist_enabled([plugin_id], enabled))
        return self._records.get(plugin_id)

    def _persist_enabled(self, plugin_ids: Iterable[str], enabled: bool) -> Set[str]:
        """Save the new disabled set and return it (even when saving failed)."""
        disabled = _disabled_ids()
        disabled = disabled - set(plugin_ids) if enabled else disabled | set(plugin_ids)
        try:
            _save_disabled_ids(disabled)
        except OSError as exc:
            self._notify(f"Could not save the plugin settings: {exc}")
        return disabled

    # ---- install / remove -----------------------------------------------
    def install_from_zip(self, parent=None, zip_path: Optional[str] = None) -> List[str]:
        """Install the plugin(s) in a ``.zip`` into the plugins folder, enable them, and show
        them in the Plugin Manager. Asks for the archive when ``zip_path`` is None. Returns the
        installed plugin ids (empty when cancelled or failed)."""
        parent = parent if parent is not None else self._window
        if zip_path is None:
            zip_path, _ = QFileDialog.getOpenFileName(
                parent, "Install Plugin from Zip", "", "Plugin archives (*.zip);;All files (*)"
            )
            if not zip_path:
                return []
        try:
            found = packages.inspect_zip(zip_path)
        except packages.PluginPackageError as exc:
            QMessageBox.warning(parent, "Install Plugin", str(exc))
            return []
        ids = [p.plugin_id for p in found]
        existing = [pid for pid in ids if (PLUGINS_DIR / pid).exists()]
        # On a case-insensitive filesystem "Zones" replaces the record of folder "zones".
        lowered = {pid.lower() for pid in ids}
        replaced = [rid for rid in self._records if rid.lower() in lowered and rid not in ids]
        if existing:
            clashing = {pid.lower() for pid in existing}
            names = ", ".join(
                rec.name for rec in self._records.values() if rec.plugin_id.lower() in clashing
            ) or ", ".join(existing)
            answer = QMessageBox.question(
                parent, "Install Plugin",
                f"Already installed: {names}.\n\nReplace with the version from the archive? "
                "Its open windows will be closed.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return []
        for pid in existing + replaced:  # Close windows and drop cached modules before files change.
            if pid in self._records:
                self._unload(self._records[pid])
            self._purge_modules(pid)
        try:
            packages.install_zip(zip_path, PLUGINS_DIR, overwrite=bool(existing))
        except (packages.PluginPackageError, OSError) as exc:
            self._rescan(existing + replaced)  # Rolled back on disk; reload the previous versions.
            QMessageBox.critical(parent, "Install Plugin", f"Could not install the plugin:\n\n{exc}")
            return []
        self._rescan(ids + replaced, self._persist_enabled(ids, True))
        self._notify("Installed plugin(s): " + ", ".join(
            self._records[pid].name if pid in self._records else pid for pid in ids
        ))
        self.open_manager(select=ids[0])
        return ids

    def remove(self, plugin_id: str, parent=None) -> bool:
        """Confirm, unload, and move one plugin's folder to the trash. True if it is gone."""
        rec = self._records.get(plugin_id)
        if rec is None:
            return False
        parent = parent if parent is not None else self._window
        path = PLUGINS_DIR / plugin_id
        yes, cancel = QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.Cancel
        answer = QMessageBox.question(
            parent, "Remove Plugin",
            f"Remove “{rec.name}”?\n\nIts folder will be moved to the {TRASH_NAME}:\n{path}",
            yes | cancel, cancel,
        )
        if answer != yes:
            return False
        self._unload(rec)
        self._purge_modules(plugin_id)
        removed = _move_to_trash(path)
        if not removed and QMessageBox.warning(
            parent, "Remove Plugin",
            f"The folder could not be moved to the {TRASH_NAME}.\n\nDelete it permanently?\n{path}",
            yes | cancel, cancel,
        ) == yes:
            try:
                shutil.rmtree(path)
                removed = True
            except OSError as exc:
                QMessageBox.critical(parent, "Remove Plugin", f"Could not delete the plugin folder:\n\n{exc}")
        if removed:
            disabled = _disabled_ids()
            if plugin_id in disabled:
                self._persist_enabled([plugin_id], True)  # Forget its state; the folder is gone.
            self._notify(f"Removed plugin: {rec.name}")
        self._rescan([plugin_id])  # Drops the record, or reloads a plugin that was kept.
        return removed

    # ---- menu -----------------------------------------------------------
    def build_menu(self, menu: QMenu) -> None:
        """Populate ``menu`` with the plugin utility actions (manage, install, folder, reload).

        Launching is done from the per-plugin buttons in the side pane (see ``build_panel``).
        """
        self._menu = menu
        menu.clear()
        menu.addAction("Manage Plugins…").triggered.connect(lambda: self.open_manager())
        menu.addAction("Install Plugin from Zip…").triggered.connect(lambda: self.install_from_zip())
        menu.addSeparator()
        menu.addAction("Open Plugins Folder…").triggered.connect(self.open_folder)
        menu.addAction("Reload Plugins").triggered.connect(self.reload)

    def _refresh_menu(self) -> None:
        if self._menu is not None:
            self.build_menu(self._menu)

    # ---- side-pane panel ------------------------------------------------
    def build_panel(self, layout) -> None:
        """Populate a vertical layout with one launch button per ENABLED plugin (the pane card)."""
        self._panel_layout = layout
        _clear_layout(layout)
        enabled = [rec for rec in self._records.values() if rec.enabled]
        if not enabled:
            empty = QLabel("No plugins enabled" if self._records else "No plugins installed")
            empty.setEnabled(False)
            layout.addWidget(empty)
        for rec in enabled:
            button = QPushButton(rec.name)
            button.setObjectName("pluginButton")
            if rec.error is not None:
                button.setEnabled(False)
                button.setText(f"⚠ {rec.name}")
                button.setToolTip(rec.error)
            else:
                button.setToolTip(rec.description)
                button.clicked.connect(
                    lambda _checked=False, pid=rec.plugin_id: self.launch(pid)
                )
            layout.addWidget(button)
        manage = QPushButton("Manage plugins…")
        manage.setObjectName("pluginManageButton")
        manage.setToolTip("Enable, disable, install, or remove plugins")
        manage.clicked.connect(lambda _checked=False: self.open_manager())
        layout.addWidget(manage)

    def _refresh_panel(self) -> None:
        if self._panel_layout is not None:
            self.build_panel(self._panel_layout)

    def _refresh_views(self) -> None:
        self._refresh_panel()
        if self._dialog is not None:
            self._dialog.refresh()

    # ---- Plugin Manager window -------------------------------------------
    def open_manager(self, select: Optional[str] = None) -> PluginManagerDialog:
        """Show the Plugin Manager window (raising it if open), optionally selecting a plugin."""
        if self._dialog is None:
            dialog = PluginManagerDialog(self, self._window)
            dialog.finished.connect(self._manager_closed)
            dialog.finished.connect(dialog.deleteLater)
            self._dialog = dialog
        self._dialog.refresh(select)
        self._dialog.show()
        self._dialog.raise_()
        self._dialog.activateWindow()
        return self._dialog

    def _manager_closed(self, _result=None) -> None:
        self._dialog = None

    # ---- launch / lifecycle --------------------------------------------
    def launch(self, plugin_id: str) -> None:
        """Instantiate (once) and launch a plugin; re-focus its window if already open.

        A window that was merely closed keeps its instance: ``launch()`` is called again, so
        plugins that cache their window keep their state. A destroyed window (e.g.
        ``WA_DeleteOnClose``) releases the old instance and creates a fresh one."""
        rec = self._records.get(plugin_id)
        if rec is None or rec.cls is None:
            return
        if rec.window is not None:
            try:
                if rec.window.isVisible():
                    rec.window.raise_()
                    rec.window.activateWindow()
                    return
            except RuntimeError:
                rec.window = None  # Native widget is gone; context still needs disposal.
                # WA_DeleteOnClose has already destroyed the native window.
                self._unload(rec)
        previous = rec.window
        try:
            if rec.instance is None:
                rec.context = PluginContext(self._window, plugin_id)
                rec.instance = rec.cls(rec.context)
            window = rec.instance.launch()
            if window is not None and not isinstance(window, QWidget):
                raise TypeError("launch() must return a QWidget or None")
        except Exception:
            self._unload(rec)
            QMessageBox.critical(
                self._window,
                f"Plugin error: {rec.name}",
                "The plugin raised an error:\n\n" + traceback.format_exc(limit=6),
            )
            return
        if isinstance(window, QWidget):
            if previous is not None and window is not previous:
                try:
                    previous.deleteLater()  # The plugin built a new window; drop the closed one.
                except RuntimeError:
                    pass
            rec.window = window
        session = getattr(self._window, "session", None)
        if session is not None:  # a macro being recorded replays this as app.launch_plugin(...)
            session.record_command("app", "launch_plugin", plugin_id)

    def shutdown(self) -> None:
        """Release every plugin resource and the manager window when the host closes."""
        self._unload_all()
        dialog, self._dialog = self._dialog, None
        if dialog is not None:
            try:
                dialog.close()
                dialog.deleteLater()
            except RuntimeError:
                pass

    def _unload_all(self) -> None:
        for rec in self._records.values():
            self._unload(rec)

    def reload(self) -> None:
        """Unload all plugins and re-scan (drops cached modules so code edits take effect)."""
        self._unload_all()
        for name in [m for m in sys.modules if m == "plugins" or m.startswith("plugins.")]:
            del sys.modules[name]
        self.discover()
        self._refresh_menu()
        self._refresh_views()

    @staticmethod
    def _unload(rec: PluginRecord) -> None:
        if rec.instance is not None:
            try:
                rec.instance.on_unload()
            except Exception:
                logger.exception("Plugin %s failed during unload", rec.plugin_id)
        if rec.window is not None:
            try:
                rec.window.close()
                rec.window.deleteLater()
            except Exception:
                logger.exception("Plugin window %s failed to close", rec.plugin_id)
        if rec.context is not None:
            rec.context.dispose()
        rec.context = None
        rec.instance = None
        rec.window = None

    def open_folder(self, folder: Optional[Path] = None) -> None:
        """Open the plugins folder (or one plugin's folder) in the system file browser."""
        folder = PLUGINS_DIR if folder is None else folder
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _notify(self, message: str) -> None:
        logger.info(message)
        status_bar = getattr(self._window, "statusBar", None)
        if callable(status_bar):
            status_bar().showMessage(message, 8000)
