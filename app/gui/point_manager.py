"""The Point Manager: a non-modal window listing every tracker point with two-way selection sync
to the canvas and a bulk-delete action.

The dialog reports a delete request and exposes its current selection; the MainWindow owns the
canvas, the interaction slot, and the actual mutation (the same division of labour as the Cleanup
dialog). It works in both modes:

- **pre-track** — lists the seed points in ``state.features`` (one "i / N" column, no metrics);
  delete removes them directly (matches the Delete tool's seed path, not undoable).
- **post-track** — lists the *active* tracked points with FB-mean and OpenCV-mean error columns;
  delete routes through ``MainWindow.apply_keep_mask`` (undoable).

Selecting rows highlights the matching markers red on the canvas; clicking points on the canvas
selects the matching rows. ``_row_to_index`` is the single row→point-index translation used by the
selection readout, the red overlay, and delete, so the two systems can never disagree.
"""
import numpy as np
from PyQt5.QtCore import QItemSelection, QItemSelectionModel, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from app.core.cleanup import compute_metrics
from app.gui.point_tools import nearest_point

SELECT_COLOR = QColor(255, 60, 60)  # the app's standard "drop / special status" red


class PointSelectInteraction:
    """Duck-typed canvas interaction installed while the Point Manager is open: forwards a
    left-click (with its keyboard modifiers) to the dialog, which resolves it to a row. Holds no
    state and mutates nothing — selection lives in the dialog's table."""

    def __init__(self, dialog: "PointManagerDialog"):
        self._dialog = dialog

    def on_press(self, image_pt, event) -> None:
        modifiers = event.modifiers() if event is not None else Qt.NoModifier
        self._dialog.handle_canvas_click(image_pt, modifiers)

    def on_cancel(self) -> None:
        pass


class PointManagerDialog(QDialog):
    deleteRequested = pyqtSignal()

    def __init__(self, window, parent=None):
        super().__init__(parent if parent is not None else window)
        self._window = window
        self._row_to_index: list = []   # row -> point index (global post-track; features row pre-track)
        self._syncing = False           # guard: suppress selection echo while we drive the table
        self._metrics = None            # cached Metrics post-track, else None

        self.setWindowTitle("Point Manager")
        self.setModal(False)

        self.table = QTableWidget(0, 1)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)

        self.delete_btn = QPushButton("Delete selected")
        self.delete_btn.clicked.connect(self.deleteRequested)
        self.delete_btn.setEnabled(False)

        close_box = QDialogButtonBox(QDialogButtonBox.Close)
        close_box.rejected.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addWidget(self.delete_btn)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(buttons)
        layout.addWidget(close_box)
        self.resize(320, 480)

    # ---- table population -------------------------------------------------
    def rebuild(self) -> None:
        """Single source of truth for the table contents and ``_row_to_index``. Re-derives the row
        map from the current state, preserves still-valid selection (by point index), and drops
        stale rows. Selection signals are suppressed (``_syncing``) so repopulation doesn't echo."""
        state = self._window.state
        prev = set(self.selected_point_indices())

        result = state.result
        post_track = result is not None and state.active_mask is not None

        self._syncing = True
        try:
            if post_track:
                self._row_to_index = [int(p) for p in np.where(state.active_mask)[0]]
                h, w = state.image_size()
                self._metrics = compute_metrics(result, state.roi, (h, w))
                headers = ["Point", "FB mean", "OpenCV mean"]
            else:
                feats = state.features
                self._row_to_index = list(range(len(feats))) if feats is not None else []
                self._metrics = None
                headers = ["Point"]

            n = len(self._row_to_index)
            self.table.setColumnCount(len(headers))
            self.table.setHorizontalHeaderLabels(headers)
            self.table.setRowCount(n)
            for r, p in enumerate(self._row_to_index):
                self.table.setItem(r, 0, QTableWidgetItem(f"{r + 1} / {n}"))
                if post_track:
                    self.table.setItem(r, 1, self._metric_item(result.fb_mean_error[p]))
                    self.table.setItem(r, 2, self._metric_item(self._metrics.mean_err_fw[p]))

            if prev:
                self._select_rows([r for r, p in enumerate(self._row_to_index) if p in prev])
        finally:
            self._syncing = False

        self.table.resizeColumnToContents(0)
        self._update_delete_enabled()
        self._window.canvas.update()

    @staticmethod
    def _metric_item(value) -> QTableWidgetItem:
        text = f"{value:.2f}" if np.isfinite(value) else "∞"
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return item

    # ---- selection readout ------------------------------------------------
    def selected_point_indices(self) -> list:
        """Selected rows mapped through ``_row_to_index`` to point indices. The bridge used by both
        delete and the red overlay."""
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
        return [self._row_to_index[r] for r in rows if 0 <= r < len(self._row_to_index)]

    def _select_rows(self, rows) -> None:
        """Replace the selection with exactly ``rows`` (applied as whole-row ranges in one go)."""
        sm = self.table.selectionModel()
        sel = QItemSelection()
        last_col = self.table.columnCount() - 1
        for r in rows:
            sel.select(self.table.model().index(r, 0), self.table.model().index(r, last_col))
        sm.select(sel, QItemSelectionModel.ClearAndSelect)

    def _on_selection_changed(self, *_args) -> None:
        # list -> canvas: just repaint the overlay (which reads the live selection). Never writes
        # back to the canvas, so there is no loop with handle_canvas_click.
        if self._syncing:
            return
        self._update_delete_enabled()
        self._window.canvas.update()

    def _update_delete_enabled(self) -> None:
        self.delete_btn.setEnabled(bool(self.table.selectionModel().selectedRows()))

    # ---- canvas -> list ---------------------------------------------------
    def handle_canvas_click(self, image_pt, modifiers) -> None:
        """Resolve a canvas click to a row and apply modifier semantics (none = select only,
        Ctrl = toggle, Shift = range-extend). Wrapped in ``_syncing`` so the resulting
        selectionChanged doesn't double-fire the overlay."""
        coords, candidates = self._click_targets()
        i = nearest_point(
            self._window.canvas,
            self._window.state.display_params["marker_size"],
            image_pt,
            coords,
            candidates,
        ) if coords is not None else -1

        ctrl = bool(modifiers & Qt.ControlModifier)
        shift = bool(modifiers & Qt.ShiftModifier)

        self._syncing = True
        try:
            if i < 0:
                if not (ctrl or shift):
                    self.table.clearSelection()
            else:
                row = self._row_to_index.index(i)
                self._apply_click(row, ctrl, shift)
        finally:
            self._syncing = False

        self._update_delete_enabled()
        self._window.canvas.update()

    def _click_targets(self):
        """``(coords, candidates)`` for the hit test in the current mode, or ``(None, None)`` when
        no points are clickable on the current frame."""
        state = self._window.state
        result = state.result
        if result is not None and state.active_mask is not None:
            if not state.current_in_range:
                return None, None
            coords = result.coords_fw[state.global_to_cut(state.current_index)]
            return coords, list(self._row_to_index)
        # pre-track: seed points only exist (and are only drawn) on the reference frame
        if state.features is not None and state.current_index == state.reference_index:
            return state.features, list(range(len(state.features)))
        return None, None

    def _apply_click(self, row: int, ctrl: bool, shift: bool) -> None:
        sm = self.table.selectionModel()
        last_col = self.table.columnCount() - 1
        index = self.table.model().index(row, 0)
        if ctrl:
            row_range = QItemSelection(index, self.table.model().index(row, last_col))
            sm.select(row_range, QItemSelectionModel.Toggle | QItemSelectionModel.Rows)
            sm.setCurrentIndex(index, QItemSelectionModel.NoUpdate)
        elif shift:
            anchor = sm.currentIndex().row()
            if anchor < 0:
                anchor = row
            self._select_rows(range(min(anchor, row), max(anchor, row) + 1))
        else:
            self._select_rows([row])
            sm.setCurrentIndex(index, QItemSelectionModel.NoUpdate)

    # ---- red overlay (registered for the dialog's lifetime by MainWindow) --
    def _paint_selected(self, painter, canvas) -> None:
        """Draw a red ring around each selected point at its current-frame position. Reading the
        live selection and current-frame coords each paint makes the highlight follow both
        selection changes and frame scrubbing for free."""
        sel = self.selected_point_indices()
        if not sel:
            return
        state = self._window.state
        coords = None
        result = state.result
        if result is not None and state.active_mask is not None:
            if state.current_in_range:
                coords = result.coords_fw[state.global_to_cut(state.current_index)]
        elif state.features is not None and state.current_index == state.reference_index:
            coords = state.features
        if coords is None:
            return

        radius = state.display_params["marker_size"] + 1
        painter.setPen(QPen(SELECT_COLOR, 2))
        painter.setBrush(Qt.NoBrush)
        for p in sel:
            if 0 <= p < len(coords):
                x, y = coords[p]
                painter.drawEllipse(canvas.image_to_screen(float(x), float(y)), radius, radius)
