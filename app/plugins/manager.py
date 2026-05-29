"""Plugin discovery, loading, and lifecycle.

Plugins live in one folder each under the project-root ``plugins/`` directory and are imported
as ``plugins.<folder>``. Each is expected to expose a :class:`~app.plugins.api.TrackerPlugin`
subclass via a module-level ``PLUGIN = MyPlugin`` (a lone subclass is auto-detected as a
fallback). The manager builds the ``Plugins`` menu, instantiates plugins on demand, keeps their
windows alive, and isolates failures so one broken plugin never takes down the app.
"""
from __future__ import annotations

import importlib
import inspect
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Type

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QMenu, QMessageBox, QWidget

from app.plugins.api import PluginContext, TrackerPlugin

# project root = .../tracker ; this file is .../tracker/app/plugins/manager.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLUGINS_DIR = PROJECT_ROOT / "plugins"


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


class PluginManager:
    """Owns the set of installed plugins and their menu/windows."""

    def __init__(self, main_window):
        self._window = main_window
        self._records: Dict[str, PluginRecord] = {}
        self._menu: Optional[QMenu] = None

    # ---- discovery ------------------------------------------------------
    def discover(self) -> List[PluginRecord]:
        """(Re)scan ``plugins/`` and return the records (also stored on the manager)."""
        self._records.clear()
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        if not PLUGINS_DIR.is_dir():
            return []
        for entry in sorted(PLUGINS_DIR.iterdir()):
            if not entry.is_dir() or entry.name.startswith((".", "_")):
                continue
            if not (entry / "__init__.py").exists():
                continue
            self._records[entry.name] = self._load_record(entry.name)
        return list(self._records.values())

    def _load_record(self, plugin_id: str) -> PluginRecord:
        try:
            module = importlib.import_module(f"plugins.{plugin_id}")
            cls = self._find_plugin_class(module)
            if cls is None:
                return PluginRecord(
                    plugin_id, plugin_id,
                    error="No TrackerPlugin subclass found (expose PLUGIN = YourClass).",
                )
            name = getattr(cls, "NAME", plugin_id) or plugin_id
            return PluginRecord(plugin_id, name, getattr(cls, "DESCRIPTION", ""), cls=cls)
        except Exception:
            return PluginRecord(plugin_id, plugin_id, error=traceback.format_exc(limit=4))

    @staticmethod
    def _find_plugin_class(module) -> Optional[Type[TrackerPlugin]]:
        explicit = getattr(module, "PLUGIN", None)
        if inspect.isclass(explicit) and issubclass(explicit, TrackerPlugin):
            return explicit
        for obj in vars(module).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, TrackerPlugin)
                and obj is not TrackerPlugin
                and obj.__module__.startswith(module.__name__)
            ):
                return obj
        return None

    # ---- menu -----------------------------------------------------------
    def build_menu(self, menu: QMenu) -> None:
        """Populate ``menu`` with one entry per plugin, plus folder/reload utilities."""
        self._menu = menu
        menu.clear()
        records = list(self._records.values())
        if not records:
            placeholder = menu.addAction("No plugins installed")
            placeholder.setEnabled(False)
        for rec in records:
            action = menu.addAction(rec.name)
            if rec.error is not None:
                action.setEnabled(False)
                action.setText(f"⚠ {rec.name} (failed to load)")
                action.setToolTip(rec.error)
            else:
                action.setToolTip(rec.description)
                action.triggered.connect(lambda _checked, pid=rec.plugin_id: self.launch(pid))
        menu.addSeparator()
        menu.addAction("Open Plugins Folder…").triggered.connect(self._open_folder)
        menu.addAction("Reload Plugins").triggered.connect(self.reload)

    def _refresh_menu(self) -> None:
        if self._menu is not None:
            self.build_menu(self._menu)

    # ---- launch / lifecycle --------------------------------------------
    def launch(self, plugin_id: str) -> None:
        """Instantiate (once) and launch a plugin; re-focus its window if already open."""
        rec = self._records.get(plugin_id)
        if rec is None or rec.cls is None:
            return
        if rec.window is not None and rec.window.isVisible():
            rec.window.raise_()
            rec.window.activateWindow()
            return
        try:
            if rec.instance is None:
                rec.instance = rec.cls(PluginContext(self._window, plugin_id))
            window = rec.instance.launch()
        except Exception:
            QMessageBox.critical(
                self._window,
                f"Plugin error: {rec.name}",
                "The plugin raised an error:\n\n" + traceback.format_exc(limit=6),
            )
            return
        if isinstance(window, QWidget):
            rec.window = window

    def reload(self) -> None:
        """Unload all plugins and re-scan (drops cached modules so code edits take effect)."""
        for rec in self._records.values():
            self._unload(rec)
        for name in [m for m in sys.modules if m == "plugins" or m.startswith("plugins.")]:
            del sys.modules[name]
        self.discover()
        self._refresh_menu()

    @staticmethod
    def _unload(rec: PluginRecord) -> None:
        if rec.instance is not None:
            try:
                rec.instance.on_unload()
            except Exception:
                pass
        if rec.window is not None:
            try:
                rec.window.close()
            except Exception:
                pass
        rec.instance = None
        rec.window = None

    def _open_folder(self) -> None:
        PLUGINS_DIR.mkdir(exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(PLUGINS_DIR)))
