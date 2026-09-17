"""Image selection and a self-contained, non-destructive alignment preview."""
from __future__ import annotations

import os
from dataclasses import replace
import math

import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt, Signal, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QTransform, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from app.core.pair_transform import PairAlignment, alignment_crop
from app.gui.canvas_view import bgr_to_qimage

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"


class ImagePairDialog(QDialog):
    """Select roles explicitly instead of inferring them from sorted filenames."""

    def __init__(self, start_dir="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open Image Pair")
        self.resize(650, 180)
        self._start_dir = start_dir
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose reference and destination images. Adjust their alignment in the next step."))
        form = QFormLayout()
        self.path_fields = []
        for title in ("Reference", "Destination"):
            row = QHBoxLayout()
            field = QLineEdit()
            field.setPlaceholderText(f"{title} image path")
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda _checked=False, f=field, t=title: self._browse(f, t))
            row.addWidget(field, 1)
            row.addWidget(browse)
            form.addRow(title, row)
            self.path_fields.append(field)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.open_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.open_button.setText("Align Images…")
        self.open_button.setEnabled(False)
        for field in self.path_fields:
            field.textChanged.connect(self._update_ready)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def paths(self):
        return [field.text().strip() for field in self.path_fields]

    def _update_ready(self):
        self.open_button.setEnabled(all(self.paths))

    def _browse(self, field, title):
        path, _ = QFileDialog.getOpenFileName(self, f"Choose {title} Image",
                                            field.text() or self._start_dir, IMAGE_FILTER)
        if path:
            field.setText(path)
            self._start_dir = os.path.dirname(path)


class AlignmentCanvas(QWidget):
    """Move only the destination; navigation transforms never alter scientific geometry."""

    translationChanged = Signal(float, float)
    alignmentChanged = Signal()
    overlapChanged = Signal()
    pivotPickingChanged = Signal()

    def __init__(self, reference, destination, translation=(0, 0), parent=None, *, alignment=None):
        super().__init__(parent)
        self.reference = bgr_to_qimage(reference)
        self.destination = bgr_to_qimage(destination)
        self._shapes = (reference.shape, destination.shape)
        self.alignment = alignment or PairAlignment(translation=tuple(translation))
        self._overlap = alignment_crop(*self._shapes, self.alignment)
        self._crop_timer = QTimer(self)
        self._crop_timer.setSingleShot(True)
        self._crop_timer.setInterval(120)
        self._crop_timer.timeout.connect(self.finish_geometry)
        self.opacity = 0.5
        self._zoom = 1.0
        self._pan = QPointF()
        self._drag_origin = None
        self._pan_origin = None
        self._space_pan = False
        self._handle = None
        self.picking_pivot = False
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(320, 240)
        self.fit_view()

    @property
    def overlap(self):
        return self._overlap or (0, 0, 0, 0)

    @property
    def crop_pending(self):
        return self._overlap is None

    @property
    def translation(self):
        return self.alignment.translation

    def destination_corners(self):
        w, h = self.destination.width(), self.destination.height()
        corners = np.array([[-.5, -.5], [w-.5, -.5], [w-.5, h-.5], [-.5, h-.5]])
        matrix = self.alignment.matrix(self._shapes[1])
        return corners @ matrix[:, :2].T + matrix[:, 2]

    def rotation_handle(self):
        corners = self.destination_corners()
        center = self.image_to_screen(*corners.mean(axis=0))
        edge = self.image_to_screen(*corners[:2].mean(axis=0))
        vector = edge - center
        length = math.hypot(vector.x(), vector.y())
        return edge + vector * (30 / max(length, 1e-9))

    def _image_bounds(self):
        polygon = QPolygonF([QPointF(*p) for p in self.destination_corners()])
        return QRectF(-.5, -.5, self.reference.width(), self.reference.height()).united(polygon.boundingRect())

    def fit_view(self):
        # Freeze these bounds during dragging so the reference does not jump or resize.
        self._bounds = self._image_bounds()
        self._zoom = 1.0
        self._pan = QPointF()
        self.update()

    def _mapping(self):
        # Leave room for the rotation handle outside the image footprint.
        scale = min(max(1, self.width() - 72) / self._bounds.width(),
                    max(1, self.height() - 96) / self._bounds.height()) * self._zoom
        transform = QTransform()
        transform.translate(self.width() / 2 + self._pan.x() - self._bounds.center().x() * scale,
                            self.height() / 2 + self._pan.y() - self._bounds.center().y() * scale)
        transform.scale(scale, scale)
        return transform

    def image_to_screen(self, x, y):
        return self._mapping().map(QPointF(x, y))

    def screen_to_image(self, point):
        return self._mapping().inverted()[0].map(point)

    def set_translation(self, dx, dy):
        self.set_alignment(replace(self.alignment, translation=(dx, dy)))

    def set_scale_percent(self, percent):
        self.set_alignment(self.alignment.with_scale_percent(percent, self._shapes[1]))

    def set_mode(self, mode):
        self.set_alignment(self.alignment.with_mode(mode, self._shapes[1]))

    def pivot_position(self):
        if self.alignment.pivot is None:
            return None
        return self.alignment.matrix(self._shapes[1]) @ np.array([*self.alignment.pivot, 1])

    def set_pivot_picking(self, enabled):
        if self.picking_pivot == enabled:
            return
        self.picking_pivot = enabled
        if enabled:
            self.setFocus()
        self._update_cursor()
        self.pivotPickingChanged.emit()

    def clear_pivot(self):
        self.set_pivot_picking(False)
        self.set_alignment(replace(self.alignment, pivot=None))

    def _update_cursor(self):
        if self._pan_origin is not None:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self._drag_origin is not None:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        elif self._space_pan:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self.picking_pivot:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()

    def set_alignment(self, alignment):
        if alignment == self.alignment:
            return
        geometry_changed = not self.alignment.same_geometry(alignment)
        self.alignment = alignment
        if geometry_changed:
            self._overlap = None
            if alignment.integer_translation_only:
                self.finish_geometry()
            elif self._drag_origin is None:
                self._crop_timer.start()
        self.translationChanged.emit(*self.translation)
        self.alignmentChanged.emit()
        self.update()

    def finish_geometry(self):
        self._crop_timer.stop()
        self._overlap = alignment_crop(*self._shapes, self.alignment)
        self.overlapChanged.emit()
        self.update()

    def reset_alignment(self):
        self.set_pivot_picking(False)
        self.set_alignment(PairAlignment(mode=self.alignment.mode,
                                         rotation_enabled=self.alignment.rotation_enabled))

    def set_opacity(self, percent):
        self.opacity = percent / 100.0
        self.update()

    def zoom_at(self, factor, point=None):
        point = point if point is not None else QPointF(self.width() / 2, self.height() / 2)
        before = self.screen_to_image(point)
        self._zoom = max(0.1, min(100.0, self._zoom * factor))
        self._pan += point - self._mapping().map(before)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#20252b"))
        painter.setTransform(self._mapping())
        painter.drawImage(QPointF(-.5, -.5), self.reference)
        painter.setOpacity(self.opacity)
        matrix = self.alignment.matrix(self._shapes[1])
        transform = QTransform(matrix[0, 0], matrix[1, 0], matrix[0, 1], matrix[1, 1],
                               matrix[0, 2], matrix[1, 2])
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setTransform(transform * self._mapping())
        painter.drawImage(QPointF(-.5, -.5), self.destination)
        painter.setTransform(self._mapping())
        painter.setOpacity(1)
        pen = QPen(QColor("#99a7b5"), 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawRect(QRectF(-.5, -.5, self.reference.width(), self.reference.height()))
        painter.drawPolygon(QPolygonF([QPointF(*p) for p in self.destination_corners()]))
        x, y, width, height = self.overlap
        if width and height:
            pen = QPen(QColor("#67e8a5"), 2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawRect(QRectF(x-.5, y-.5, width, height))
        painter.resetTransform()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.setBrush(QColor("#2563eb"))
        if self.alignment.mode != "translation":
            for corner in self.destination_corners():
                point = self.image_to_screen(*corner)
                painter.drawRect(QRectF(point.x()-5, point.y()-5, 10, 10))
        if self.alignment.rotation_enabled:
            handle = self.rotation_handle()
            edge = self.image_to_screen(*self.destination_corners()[:2].mean(axis=0))
            painter.drawLine(edge, handle)
            painter.setBrush(QColor("#f59e0b"))
            painter.drawEllipse(handle, 6, 6)
        pivot = self.pivot_position()
        if pivot is not None:
            point = self.image_to_screen(*pivot)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            # Draw in screen coordinates so the feature remains visible at every zoom.
            for color, width in (("#20252b", 4), ("#facc15", 2)):
                painter.setPen(QPen(QColor(color), width))
                painter.drawEllipse(point, 7, 7)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    painter.drawLine(point + QPointF(dx * 4, dy * 4),
                                     point + QPointF(dx * 11, dy * 11))
        painter.end()

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton) or (
            event.button() == Qt.MouseButton.LeftButton and self._space_pan
        ):
            self._pan_origin = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            if self.picking_pivot:
                point = self.screen_to_image(event.position())
                matrix = self.alignment.matrix(self._shapes[1])
                pivot = np.linalg.solve(matrix[:, :2], np.array([point.x(), point.y()]) - matrix[:, 2])
                h, w = self._shapes[1][:2]
                if -.5 <= pivot[0] <= w - .5 and -.5 <= pivot[1] <= h - .5:
                    self.set_alignment(replace(self.alignment, pivot=tuple(pivot)))
                    self.set_pivot_picking(False)
                event.accept()
                return
            self._crop_timer.stop()
            self._drag_origin = self.screen_to_image(event.position())
            self._start_translation = self.translation
            self._start_alignment = self.alignment
            self._handle = None
            if self.alignment.mode != "translation":
                for index, corner in enumerate(self.destination_corners()):
                    delta = event.position() - self.image_to_screen(*corner)
                    if math.hypot(delta.x(), delta.y()) <= 10:
                        self._handle = index
                        break
            if self.alignment.rotation_enabled:
                delta = event.position() - self.rotation_handle()
                if math.hypot(delta.x(), delta.y()) <= 10:
                    self._handle = "rotation"
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._pan_origin is not None:
            self._pan += event.position() - self._pan_origin
            self._pan_origin = event.position()
            self.update()
        elif self._drag_origin is not None:
            point = self.screen_to_image(event.position())
            try:
                if self._handle == "rotation":
                    h, w = self._shapes[1][:2]
                    center = np.array([(w-1)/2, (h-1)/2]) + self._start_translation
                    a = math.atan2(self._drag_origin.y()-center[1], self._drag_origin.x()-center[0])
                    b = math.atan2(point.y()-center[1], point.x()-center[0])
                    self.set_alignment(replace(self._start_alignment,
                                               angle_degrees=self._start_alignment.angle_degrees + math.degrees(b-a)))
                elif self._handle is not None:
                    self.set_alignment(self._start_alignment.drag_corner(
                        self._shapes[1], self._handle, (point.x(), point.y())))
                else:
                    delta = point - self._drag_origin
                    self.set_translation(self._start_translation[0] + round(delta.x()),
                                         self._start_translation[1] + round(delta.y()))
            except ValueError:
                pass  # Keep the last valid geometry when a handle would fold/collapse the image.
        event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_origin = self._pan_origin = None
        self._handle = None
        if self.crop_pending:
            self.finish_geometry()
        self._update_cursor()
        event.accept()

    def wheelEvent(self, event):
        self.zoom_at(1.2 if event.angleDelta().y() > 0 else 1 / 1.2, event.position())
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self.picking_pivot:
            self.set_pivot_picking(False)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Space:
            self._space_pan = True
            self._update_cursor()
            event.accept()
            return
        direction = {Qt.Key.Key_Left: (-1, 0), Qt.Key.Key_Right: (1, 0),
                     Qt.Key.Key_Up: (0, -1), Qt.Key.Key_Down: (0, 1)}.get(event.key())
        if direction is not None:
            step = 10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
            self.set_translation(self.translation[0] + direction[0] * step,
                                 self.translation[1] + direction[1] * step)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan = False
            self._update_cursor()
            event.accept()
        else:
            super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self._space_pan = False
        self._drag_origin = self._pan_origin = None
        self._handle = None
        if self.crop_pending:
            self._crop_timer.start()
        self._update_cursor()
        super().focusOutEvent(event)


class PairAlignmentDialog(QDialog):
    def __init__(self, sequence, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Align Image Pair")
        self.resize(1080, 800)
        layout = QVBoxLayout(self)
        reference, destination = (sequence.source_bgr(i) for i in range(2))
        names = QLabel(f"Reference: {os.path.basename(sequence.paths[0])}  ·  {reference.shape[1]} × {reference.shape[0]} px\n"
                       f"Destination: {os.path.basename(sequence.paths[1])}  ·  {destination.shape[1]} × {destination.shape[0]} px")
        names.setTextFormat(Qt.TextFormat.PlainText)
        names.setWordWrap(True)
        names.setToolTip("\n".join(sequence.paths))
        layout.addWidget(names)
        instructions = QLabel("Drag to move destination · Arrows: 1 px · Shift+arrows: 10 px · "
                              "Wheel: zoom · Space+drag or right-drag: pan")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        self.canvas = AlignmentCanvas(reference, destination, alignment=sequence.alignment)
        layout.addWidget(self.canvas, 1)

        deformation = QHBoxLayout()
        deformation.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        for label, mode in (("Translation", "translation"), ("Uniform Scale", "scale"), ("Affine", "affine")):
            self.mode_combo.addItem(label, mode)
        deformation.addWidget(self.mode_combo)
        deformation.addWidget(QLabel("Uniform scale"))
        self.scale_percent = QDoubleSpinBox()
        self.scale_percent.setDecimals(4)
        self.scale_percent.setRange(.0001, 1000000)
        self.scale_percent.setSuffix(" %")
        self.scale_percent.setSingleStep(1)
        self.scale_percent.setToolTip("Scale about the selected pivot, or the image center if none is set. "
                                     "In Affine mode, preserve the current stretch/shear proportions.")
        deformation.addWidget(self.scale_percent)
        self.rotation_check = QCheckBox("Rotation")
        deformation.addWidget(self.rotation_check)
        self.angle_degrees = QDoubleSpinBox()
        self.angle_degrees.setRange(-180, 180)
        self.angle_degrees.setDecimals(3)
        self.angle_degrees.setSuffix("°")
        self.angle_degrees.setWrapping(True)
        self.angle_degrees.setToolTip("Clockwise rotation about the destination center; excluded from restored deformation.")
        deformation.addWidget(self.angle_degrees)
        self.set_pivot_button = QPushButton("Set Pivot")
        self.set_pivot_button.setCheckable(True)
        self.set_pivot_button.setAutoDefault(False)
        self.set_pivot_button.setToolTip("Click an aligned feature to hold it fixed during scaling and affine adjustments.")
        self.set_pivot_button.toggled.connect(self.canvas.set_pivot_picking)
        deformation.addWidget(self.set_pivot_button)
        self.clear_pivot_button = QPushButton("Clear Pivot")
        self.clear_pivot_button.setAutoDefault(False)
        self.clear_pivot_button.setToolTip("Restore center/opposite-corner scaling without moving the image.")
        self.clear_pivot_button.clicked.connect(self.canvas.clear_pivot)
        deformation.addWidget(self.clear_pivot_button)
        deformation.addStretch()
        layout.addLayout(deformation)
        self.mode_hint = QLabel()
        self.mode_hint.setWordWrap(True)
        layout.addWidget(self.mode_hint)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("X offset"))
        self.x_offset = QDoubleSpinBox()
        self.x_offset.setDecimals(3)
        self.x_offset.setRange(-1e9, 1e9)
        self.x_offset.setSuffix(" px")
        controls.addWidget(self.x_offset)
        controls.addWidget(QLabel("Y offset"))
        self.y_offset = QDoubleSpinBox()
        self.y_offset.setDecimals(3)
        self.y_offset.setRange(-1e9, 1e9)
        self.y_offset.setSuffix(" px")
        controls.addWidget(self.y_offset)
        self.x_offset.setToolTip("Positive X moves the destination right")
        self.y_offset.setToolTip("Positive Y moves the destination down")
        reset = QPushButton("Reset Alignment")
        reset.clicked.connect(self.canvas.reset_alignment)
        controls.addWidget(reset)
        controls.addStretch()
        for title, callback in (("−", lambda: self.canvas.zoom_at(1 / 1.2)),
                                ("+", lambda: self.canvas.zoom_at(1.2)),
                                ("Fit View", self.canvas.fit_view)):
            button = QPushButton(title)
            button.setAutoDefault(False)
            button.clicked.connect(callback)
            if title in ("−", "+"):
                button.setFixedWidth(36)
                button.setToolTip("Zoom out" if title == "−" else "Zoom in")
            controls.addWidget(button)
        reset.setAutoDefault(False)
        layout.addLayout(controls)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Destination opacity"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_label = QLabel("50%")
        self.opacity_label.setMinimumWidth(40)
        opacity_row.addWidget(self.opacity_slider, 1)
        opacity_row.addWidget(self.opacity_label)
        self.opacity_slider.valueChanged.connect(self.canvas.set_opacity)
        self.opacity_slider.valueChanged.connect(lambda value: self.opacity_label.setText(f"{value}%"))
        layout.addLayout(opacity_row)

        self.overlap_label = QLabel()
        layout.addWidget(self.overlap_label)
        note = QLabel("Both images will be cropped to the green valid rectangle. "
                      "Tracking measures residual motion. The Affine Zone Tool can restore alignment scale/shear. "
                      "Originals stay unchanged.")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.apply_button.setText("Apply Alignment")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.canvas.alignmentChanged.connect(self._sync_shift)
        self.canvas.overlapChanged.connect(self._sync_overlap)
        self.canvas.pivotPickingChanged.connect(self._sync_pivot)
        self.x_offset.valueChanged.connect(self._spin_changed)
        self.y_offset.valueChanged.connect(self._spin_changed)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.rotation_check.toggled.connect(lambda enabled: self.canvas.set_alignment(
            replace(self.alignment, rotation_enabled=enabled)))
        self.angle_degrees.valueChanged.connect(lambda value: self.canvas.set_alignment(
            replace(self.alignment, angle_degrees=value)))
        self.scale_percent.valueChanged.connect(self.canvas.set_scale_percent)
        self._sync_shift()
        self.canvas.setFocus()

    @property
    def translation(self):
        return self.canvas.translation

    @property
    def alignment(self):
        return self.canvas.alignment

    def _mode_changed(self):
        self.canvas.set_mode(self.mode_combo.currentData())

    def _spin_changed(self):
        self.canvas.set_translation(self.x_offset.value(), self.y_offset.value())

    def _sync_shift(self):
        dx, dy = self.translation
        for spin, value in ((self.x_offset, dx), (self.y_offset, dy),
                            (self.scale_percent, self.alignment.scale_percent),
                            (self.angle_degrees, self.alignment.angle_degrees)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(self.alignment.mode))
        self.mode_combo.blockSignals(False)
        self.rotation_check.blockSignals(True)
        self.rotation_check.setChecked(self.alignment.rotation_enabled)
        self.rotation_check.blockSignals(False)
        self.angle_degrees.setEnabled(self.alignment.rotation_enabled)
        self.scale_percent.setEnabled(self.alignment.mode != "translation")
        self._sync_pivot()
        self._sync_overlap()

    def _sync_pivot(self):
        picking = self.canvas.picking_pivot
        self.set_pivot_button.blockSignals(True)
        self.set_pivot_button.setChecked(picking)
        self.set_pivot_button.blockSignals(False)
        self.clear_pivot_button.setEnabled(picking or self.alignment.pivot is not None)
        if picking:
            self.mode_hint.setText("Click a feature on the destination to set its scaling pivot. Esc cancels; zoom and pan remain available.")
            return
        anchor = "the selected pivot" if self.alignment.pivot is not None else "the opposite corner"
        hint = {"translation": "Drag to translate the destination.",
                "scale": f"Drag a square corner handle to scale uniformly; {anchor} stays fixed.",
                "affine": f"Drag a square corner handle to stretch/shear; {anchor} stays fixed and parallel edges remain parallel."}
        self.mode_hint.setText(hint[self.alignment.mode] + (
            " The yellow marker follows the feature during translation and rotation."
            if self.alignment.pivot is not None else " Set Pivot to keep an aligned feature fixed while scaling.") + (
            " Drag the round orange handle to rotate about the image center." if self.alignment.rotation_enabled else ""))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self.canvas.picking_pivot:
            self.canvas.set_pivot_picking(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def _sync_overlap(self):
        x, y, width, height = self.canvas.overlap
        valid = width > 0 and height > 0
        self.apply_button.setEnabled(valid)
        self.overlap_label.setText(
            f"Shared overlap: {width} × {height} px · Reference crop origin: ({x}, {y})"
            if valid else ("Calculating valid overlap…" if self.canvas.crop_pending else
                           "No shared overlap. Move the destination closer to the reference.")
        )

    def accept(self):
        if self.apply_button.isEnabled():
            super().accept()
