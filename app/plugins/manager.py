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
from PyQt5.QtWidgets import QLabel, QMenu, QMessageBox, QPushButton, QWidget

from app.plugins.api import PluginContext, TrackerPlugin

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
PLUGINS_DIR = PROJECT_ROOT / "plugins"


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


class PluginManager:
    """Owns the set of installed plugins and their menu/windows."""

    def __init__(self, main_window):
        self._window = main_window
        self._records: Dict[str, PluginRecord] = {}
        self._menu: Optional[QMenu] = None
        self._panel_layout = None

    # ---- discovery ------------------------------------------------------
    def discover(self) -> List[PluginRecord]:
        """(Re)scan ``plugins/`` and return the records (also stored on the manager)."""
        self._records.clear()
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        if not PLUGINS_DIR.is_dir():
            return []
        records: List[PluginRecord] = []
        for entry in sorted(PLUGINS_DIR.iterdir()):
            if not entry.is_dir() or entry.name.startswith((".", "_")):
                continue
            if not (entry / "__init__.py").exists():
                continue
            records.append(self._load_record(entry.name))
        # Order by the plugin's ORDER (ascending), ties broken by display name. Records that
        # failed to load (cls is None) fall back to the default ORDER and sort by name.
        records.sort(key=lambda r: (getattr(r.cls, "ORDER", TrackerPlugin.ORDER), r.name.lower()))
        for rec in records:
            self._records[rec.plugin_id] = rec
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
        """Populate ``menu`` with the plugin utility actions (open folder, reload).

        Launching is done from the per-plugin buttons in the side pane (see ``build_panel``).
        """
        self._menu = menu
        menu.clear()
        menu.addAction("Open Plugins Folder…").triggered.connect(self._open_folder)
        menu.addAction("Reload Plugins").triggered.connect(self.reload)

    def _refresh_menu(self) -> None:
        if self._menu is not None:
            self.build_menu(self._menu)

    # ---- side-pane panel ------------------------------------------------
    def build_panel(self, layout) -> None:
        """Populate a vertical layout with one launch button per plugin (the pane card)."""
        self._panel_layout = layout
        _clear_layout(layout)
        records = list(self._records.values())
        if not records:
            empty = QLabel("No plugins installed")
            empty.setEnabled(False)
            layout.addWidget(empty)
            return
        for rec in records:
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

    def _refresh_panel(self) -> None:
        if self._panel_layout is not None:
            self.build_panel(self._panel_layout)

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
        self._refresh_panel()

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
