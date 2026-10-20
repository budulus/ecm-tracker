"""The Plugin Manager window: one checkable list of every plugin in the plugins folder.

Checking a plugin enables it (it is imported and gets a launch button in the side pane);
unchecking disables it (it is unloaded, its button disappears, and it is not imported on later
starts). The window also installs plugins from zip archives and removes them. All state and every
operation live in the :class:`~app.plugins.manager.PluginManager`; this dialog only presents them
and calls back into it (duck-typed, so ``manager`` can import this module without a cycle).
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFontDatabase
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

_ID_ROLE = Qt.ItemDataRole.UserRole
_MUTED = QColor("#8a9099")  # theme "muted": disabled plugins read as inactive
_ZWSP = "​"  # zero-width space: lets a word-wrapped QLabel break a long path
_STATUS = {  # state -> (label, color)
    "enabled": ("Enabled", "#15803d"),
    "disabled": ("Disabled — not loaded", "#6b727a"),
    "error": ("Enabled — failed to load", "#dc2626"),
}


def _wrappable_path(path) -> str:
    """A path that wraps at its separators instead of forcing its label wide."""
    return str(path).replace("\\", "\\" + _ZWSP).replace("/", "/" + _ZWSP)


class PluginManagerDialog(QDialog):
    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self._manager = manager
        self.setWindowTitle("Plugin Manager")
        self.setModal(False)
        self.resize(780, 480)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search plugins")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_filter)
        self.install_btn = QPushButton("Install from Zip…")
        self.install_btn.setToolTip("Install a plugin from a .zip archive into the plugins folder")
        self.install_btn.clicked.connect(lambda: self._manager.install_from_zip(self))
        top = QHBoxLayout()
        top.addWidget(self.search, 1)
        top.addWidget(self.install_btn)

        # ---- left: the checkable plugin list -------------------------------
        self.list = QListWidget()
        self.list.setObjectName("pluginList")
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.currentItemChanged.connect(lambda *_: self._show_details())
        hint = QLabel("Checked plugins are loaded and get a button in the side pane.")
        hint.setObjectName("pluginHint")
        hint.setWordWrap(True)
        left = QWidget()
        left.setMinimumWidth(220)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.list, 1)
        left_layout.addWidget(hint)

        # ---- right: details of the selected plugin --------------------------
        self.name_label = QLabel()
        self.name_label.setObjectName("pluginTitle")
        self.name_label.setWordWrap(True)
        self.status_label = QLabel()
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.description_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.folder_label = QLabel()
        self.folder_label.setObjectName("pluginHint")
        self.folder_label.setWordWrap(True)
        self.folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.error_view = QPlainTextEdit()
        self.error_view.setObjectName("pluginError")
        self.error_view.setReadOnly(True)
        self.error_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)  # keep traceback shape
        self.error_view.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.show_folder_btn = QPushButton("Show in Folder")
        self.show_folder_btn.clicked.connect(self._show_folder)
        self.remove_btn = QPushButton("Remove…")
        self.remove_btn.setToolTip("Move this plugin's folder to the trash")
        self.remove_btn.clicked.connect(self._remove)
        actions = QHBoxLayout()
        actions.addWidget(self.show_folder_btn)
        actions.addWidget(self.remove_btn)
        actions.addStretch(1)

        self.details = QWidget()
        details_layout = QVBoxLayout(self.details)
        details_layout.setContentsMargins(12, 0, 0, 0)
        details_layout.setSpacing(8)
        details_layout.addWidget(self.name_label)
        details_layout.addWidget(self.status_label)
        details_layout.addWidget(self.description_label)
        details_layout.addWidget(self.folder_label)
        details_layout.addWidget(self.error_view, 4)
        details_layout.addStretch(1)
        details_layout.addLayout(actions)
        self.empty_label = QLabel(
            "The plugins folder has no plugins yet.\n\n"
            "Use “Install from Zip…”, or copy a plugin folder into the plugins folder "
            "and choose Reload All."
        )
        self.empty_label.setObjectName("pluginHint")
        self.empty_label.setWordWrap(True)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.details, 1)
        right_layout.addWidget(self.empty_label, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([300, 480])

        # ---- footer ---------------------------------------------------------
        self.folder_path_label = QLabel()
        self.folder_path_label.setObjectName("pluginHint")
        self.folder_path_label.setWordWrap(True)
        self.folder_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        open_btn = QPushButton("Open Plugins Folder")
        open_btn.clicked.connect(lambda: self._manager.open_folder())
        reload_btn = QPushButton("Reload All")
        reload_btn.setToolTip("Re-import every enabled plugin from disk (closes plugin windows)")
        reload_btn.clicked.connect(lambda: self._manager.reload())
        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_box.rejected.connect(self.reject)
        footer = QHBoxLayout()
        footer.addWidget(open_btn)
        footer.addWidget(reload_btn)
        footer.addStretch(1)
        footer.addWidget(close_box)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.folder_path_label)
        layout.addLayout(footer)

        # Enter in the search box must not click a button (install / close).
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)

    # ---- population -------------------------------------------------------
    def refresh(self, select: Optional[str] = None) -> None:
        """Rebuild the list from the manager, keeping the selection (or moving it to ``select``)."""
        current = select or self._current_id()
        self.list.blockSignals(True)
        self.list.clear()
        for rec in self._manager.records():
            item = QListWidgetItem()
            item.setData(_ID_ROLE, rec.plugin_id)
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable
            )
            self._decorate(item, rec)
            self.list.addItem(item)
        self.list.blockSignals(False)
        self.folder_path_label.setText(f"Plugins folder: {_wrappable_path(self._manager.plugins_dir)}")
        self.folder_path_label.setToolTip(str(self._manager.plugins_dir))
        if select is not None and self.search.text() and self._item(select) is not None:
            self.search.clear()  # A requested plugin (e.g. just installed) must not be filtered out.
        self._apply_filter()
        self._select(current)

    @staticmethod
    def _decorate(item: QListWidgetItem, rec) -> None:
        item.setText(f"⚠  {rec.name}" if rec.error else rec.name)
        item.setCheckState(Qt.CheckState.Checked if rec.enabled else Qt.CheckState.Unchecked)
        tip = (rec.error.strip().splitlines() or [""])[-1] if rec.error else rec.description
        item.setToolTip(tip)
        item.setData(Qt.ItemDataRole.ForegroundRole, None if rec.enabled else _MUTED)

    def _apply_filter(self) -> None:
        needle = self.search.text().strip().lower()
        records = {rec.plugin_id: rec for rec in self._manager.records()}
        for row in range(self.list.count()):
            item = self.list.item(row)
            rec = records.get(item.data(_ID_ROLE))
            haystack = f"{rec.name} {rec.plugin_id} {rec.description}".lower() if rec else ""
            item.setHidden(bool(needle) and needle not in haystack)
        current = self.list.currentItem()
        if current is None or current.isHidden():
            self._select(None)
        else:
            self._show_details()

    def _item(self, plugin_id: Optional[str]) -> Optional[QListWidgetItem]:
        for row in range(self.list.count()):
            if self.list.item(row).data(_ID_ROLE) == plugin_id:
                return self.list.item(row)
        return None

    def _select(self, plugin_id: Optional[str]) -> None:
        """Select ``plugin_id`` if it is visible, else the first visible plugin (or nothing)."""
        visible = [self.list.item(r) for r in range(self.list.count()) if not self.list.item(r).isHidden()]
        match = next((item for item in visible if item.data(_ID_ROLE) == plugin_id), None)
        match = match or (visible[0] if visible else None)
        if match is None:
            self.list.setCurrentRow(-1)
        else:
            self.list.setCurrentItem(match)
        self._show_details()

    def _current_id(self) -> Optional[str]:
        item = self.list.currentItem()
        return item.data(_ID_ROLE) if item is not None and not item.isHidden() else None

    def _current_record(self):
        plugin_id = self._current_id()
        return next((r for r in self._manager.records() if r.plugin_id == plugin_id), None)

    def _show_details(self) -> None:
        rec = self._current_record()
        self.details.setVisible(rec is not None)
        self.empty_label.setVisible(self.list.count() == 0)
        if rec is None:
            return
        state = "error" if rec.error else "enabled" if rec.enabled else "disabled"
        label, color = _STATUS[state]
        self.name_label.setText(rec.name)
        self.status_label.setText(label)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: 600;")
        self.description_label.setText(rec.description or "No description.")
        path = rec.path or self._manager.plugins_dir / rec.plugin_id
        self.folder_label.setText(f"Folder: {_wrappable_path(path)}")
        self.folder_label.setToolTip(str(path))
        self.error_view.setVisible(bool(rec.error))
        self.error_view.setPlainText(rec.error or "")

    # ---- actions ----------------------------------------------------------
    def _on_item_changed(self, item: QListWidgetItem) -> None:
        plugin_id = item.data(_ID_ROLE)
        enabled = item.checkState() == Qt.CheckState.Checked
        self.list.setCurrentItem(item)
        # Deferred: applying the change rebuilds this list, which must not happen while the
        # item is still emitting its own change signal.
        QTimer.singleShot(0, self, lambda: self._manager.set_enabled(plugin_id, enabled))

    def _remove(self) -> None:
        rec = self._current_record()
        if rec is not None:
            self._manager.remove(rec.plugin_id, self)

    def _show_folder(self) -> None:
        rec = self._current_record()
        if rec is not None:
            self._manager.open_folder(rec.path or self._manager.plugins_dir / rec.plugin_id)
