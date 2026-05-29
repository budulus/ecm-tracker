import cv2
import numpy as np
from PyQt5.QtCore import QEvent, QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPolygonF, QTransform
from PyQt5.QtWidgets import QWidget

from app.models.project_state import ProjectState


def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    """Convert a 3-channel BGR uint8 array to an independent (copied) QImage."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return image.copy()  # detach from the numpy buffer


class CanvasView(QWidget):
    """Displays the current frame with image-space overlays.

    All overlay geometry is kept in image coordinates; the image->widget QTransform is the
    single place where image space is mapped to screen space (rebuilt on every paint so it
    tracks widget resizing).
    """

    imageClicked = pyqtSignal(QPointF)  # emitted with image-space coordinates on left click

    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self._state = state
        self._qimage: QImage = QImage()
        self._img_w = 0
        self._img_h = 0
        self._transform = QTransform()
        self._preview_keep_mask = None  # set by cleanup preview: bool (P,) over active points
        self._zoom = 1.0  # user zoom on top of the fit scale
        self._pan = QPointF(0.0, 0.0)  # screen-space pan offset
        self._panning = False
        self._last_pan_pos = QPointF(0.0, 0.0)
        self._pan_tool = False  # hand-tool mode (toolbar toggle): left-drag pans
        self._space_pan = False  # Spacebar held: temporary left-drag pan
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    # ---- navigation -----------------------------------------------------
    def _zoom_at(self, factor: float, center: QPointF) -> None:
        """Scale by `factor` while keeping the image point under `center` fixed on screen."""
        before = self.screen_to_image(center)
        self._zoom = max(0.1, min(self._zoom * factor, 50.0))
        self._transform = self._build_transform()
        after = self._transform.map(before)
        self._pan += center - after
        self.update()

    def zoom_in(self) -> None:
        self._zoom_at(1.25, QPointF(self.rect().center()))

    def zoom_out(self) -> None:
        self._zoom_at(0.8, QPointF(self.rect().center()))

    def set_pan_tool(self, enabled: bool) -> None:
        """Toggle the hand tool: while on, a left-drag pans instead of placing points."""
        self._pan_tool = enabled
        self._apply_nav_cursor()

    def _apply_nav_cursor(self) -> None:
        """Reflect the current navigation mode in the cursor (open hand when pannable)."""
        if self._panning:
            return  # mid-drag: keep the closed-hand cursor set by the press handler
        if self._pan_tool or self._space_pan:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.unsetCursor()

    def set_preview_mask(self, mask) -> None:
        """Set a per-point keep mask for the cleanup preview (None clears it)."""
        self._preview_keep_mask = mask
        self.update()

    def refresh(self) -> None:
        """Rebuild the cached frame image from the current state and repaint."""
        seq = self._state.sequence
        if seq is None:
            self._qimage = QImage()
            self._img_w = self._img_h = 0
        else:
            bgr = seq.load_bgr(self._state.current_index)
            self._img_h, self._img_w = bgr.shape[:2]
            self._qimage = bgr_to_qimage(bgr)
        self.update()

    # ---- coordinate mapping --------------------------------------------
    def _fit_transform(self) -> QTransform:
        scale = min(self.width() / self._img_w, self.height() / self._img_h)
        offset_x = (self.width() - self._img_w * scale) / 2.0
        offset_y = (self.height() - self._img_h * scale) / 2.0
        transform = QTransform()
        transform.translate(offset_x, offset_y)
        transform.scale(scale, scale)
        return transform

    def _build_transform(self) -> QTransform:
        """image -> screen = fit (aspect-preserving) then user zoom/pan (in screen space)."""
        if self._img_w == 0 or self._img_h == 0:
            return QTransform()
        view = QTransform()
        view.translate(self._pan.x(), self._pan.y())
        view.scale(self._zoom, self._zoom)
        return self._fit_transform() * view

    def image_to_screen(self, x: float, y: float) -> QPointF:
        return self._transform.map(QPointF(x, y))

    def screen_to_image(self, point: QPointF) -> QPointF:
        inverted, ok = self._transform.inverted()
        return inverted.map(point) if ok else QPointF(point)

    # ---- input ----------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if self._qimage.isNull():
            super().mousePressEvent(event)
            return
        if event.button() == Qt.LeftButton:
            if self._pan_tool or self._space_pan:
                self._panning = True
                self._last_pan_pos = QPointF(event.pos())
                self.setCursor(Qt.ClosedHandCursor)
            else:
                self.imageClicked.emit(self.screen_to_image(QPointF(event.pos())))
        elif event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._panning = True
            self._last_pan_pos = QPointF(event.pos())
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            pos = QPointF(event.pos())
            self._pan += pos - self._last_pan_pos
            self._last_pan_pos = pos
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._panning and event.button() in (
            Qt.LeftButton,
            Qt.MiddleButton,
            Qt.RightButton,
        ):
            self._panning = False
            self._apply_nav_cursor()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.reset_view()
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event) -> None:
        if self._qimage.isNull():
            return
        pixel = event.pixelDelta()
        ctrl = bool(event.modifiers() & Qt.ControlModifier)  # Cmd on macOS
        # Zoom for a mouse wheel (no pixelDelta) or when Cmd/Ctrl is held; a bare two-finger
        # trackpad scroll (pixelDelta present, no modifier) pans like a native image viewer.
        if pixel.isNull() or ctrl:
            delta = event.angleDelta().y() or pixel.y()
            if delta == 0:
                return
            self._zoom_at(1.25 if delta > 0 else 0.8, QPointF(event.pos()))
        else:
            self._pan += QPointF(pixel)
            self.update()

    def event(self, e):
        # macOS sends pinch as a native gesture, not a wheel event.
        if e.type() == QEvent.NativeGesture and e.gestureType() == Qt.ZoomNativeGesture:
            if not self._qimage.isNull():
                self._zoom_at(1.0 + e.value(), QPointF(e.pos()))
            return True
        return super().event(e)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_pan = True
            self._apply_nav_cursor()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_pan = False
            self._apply_nav_cursor()
            return
        super().keyReleaseEvent(event)

    # ---- painting -------------------------------------------------------
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        if self._qimage.isNull():
            return
        self._transform = self._build_transform()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setTransform(self._transform)
        painter.drawImage(0, 0, self._qimage)

        # Overlays are drawn in screen space (after resetting the transform) so pen widths
        # and marker sizes stay constant regardless of image scale.
        painter.resetTransform()
        self._draw_roi(painter)
        self._draw_features(painter)
        self._draw_tracked(painter)

    def _draw_roi(self, painter: QPainter) -> None:
        roi = self._state.roi
        if roi is None or not roi.corners:
            return
        screen_pts = [self.image_to_screen(x, y) for x, y in roi.corners]

        painter.setPen(QPen(QColor(255, 215, 0), 2))
        if roi.is_complete:
            painter.setBrush(QBrush(QColor(255, 215, 0, 40)))
            painter.drawPolygon(QPolygonF(screen_pts))
        else:
            painter.setBrush(Qt.NoBrush)
            if len(screen_pts) >= 2:
                painter.drawPolyline(QPolygonF(screen_pts))

        painter.setBrush(QBrush(QColor(255, 0, 255)))
        painter.setPen(QPen(QColor(255, 0, 255), 1))
        for pt in screen_pts:
            painter.drawEllipse(pt, 4, 4)

    def _draw_features(self, painter: QPainter) -> None:
        """Draw the raw reference-frame seed points (only before tracking exists)."""
        feats = self._state.features
        if feats is None or len(feats) == 0 or self._state.result is not None:
            return
        if self._state.current_index != self._state.reference_index:
            return
        painter.setPen(QPen(QColor(0, 255, 255), 1))
        painter.setBrush(QBrush(QColor(0, 255, 255)))
        for x, y in feats:
            painter.drawEllipse(self.image_to_screen(x, y), 2.5, 2.5)

    def _draw_tracked(self, painter: QPainter) -> None:
        """Draw tracked positions for the current frame, with a motion trail from the previous
        frame. Kept points are green; cleanup-preview drops are red; already-removed points are
        not drawn."""
        state = self._state
        result = state.result
        if result is None or not state.current_in_range:
            return
        cut = state.global_to_cut(state.current_index)
        coords = result.coords_fw[cut]
        prev = result.coords_fw[cut - 1] if cut > 0 else None
        active = state.active_mask
        preview = self._preview_keep_mask

        green = QColor(0, 220, 0)
        red = QColor(255, 60, 60)
        for p in range(result.n_points):
            if active is not None and not active[p]:
                continue
            keep = True if preview is None else bool(preview[p])
            color = green if keep else red
            here = self.image_to_screen(coords[p][0], coords[p][1])
            if prev is not None:
                trail = QColor(color)
                trail.setAlpha(140)
                painter.setPen(QPen(trail, 1))
                painter.drawLine(
                    self.image_to_screen(prev[p][0], prev[p][1]), here
                )
            painter.setPen(QPen(color, 1))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(here, 2.5, 2.5)
