import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPolygonF, QTransform
from PySide6.QtWidgets import QWidget

from app.models.project_state import ProjectState


def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    """Convert a 3-channel BGR uint8 array to an independent (copied) QImage."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return image.copy()  # detach from the numpy buffer


class CanvasView(QWidget):
    """Displays the current frame with image-space overlays.

    All overlay geometry is kept in image coordinates; the image->widget QTransform is the
    single place where image space is mapped to screen space (rebuilt on every paint so it
    tracks widget resizing).
    """

    imageClicked = Signal(QPointF)  # emitted with image-space coordinates on left click

    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self._state = state
        self._qimage: QImage = QImage()
        self._img_w = 0
        self._img_h = 0
        self._transform = QTransform()
        self._preview_keep_mask = None  # set by cleanup preview: bool (P,) over active points
        # Plugin extension points (see app/plugins/api.py). Overlays are callables
        # fn(painter, canvas) drawn in screen space after the built-in overlays; the optional
        # interaction handler receives image-space mouse events while installed.
        self._overlays: list = []
        self._interaction = None
        self._zoom = 1.0  # user zoom on top of the fit scale
        self._pan = QPointF(0.0, 0.0)  # screen-space pan offset
        self._panning = False
        self._last_pan_pos = QPointF(0.0, 0.0)
        self._pan_tool = False  # hand-tool mode (toolbar toggle): left-drag pans
        self._space_pan = False  # Spacebar held: temporary left-drag pan
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()

    def set_preview_mask(self, mask) -> None:
        """Set a per-point keep mask for the cleanup preview (None clears it)."""
        self._preview_keep_mask = mask
        self.update()

    # ---- plugin extension points ---------------------------------------
    def add_overlay(self, fn) -> None:
        """Register a custom overlay painter ``fn(painter, canvas)`` (drawn in screen space,
        after the built-in overlays). Idempotent. Call ``update()`` / ``request_redraw`` to show."""
        if fn not in self._overlays:
            self._overlays.append(fn)
            self.update()

    def remove_overlay(self, fn) -> None:
        """Unregister a previously added overlay painter (no-op if not registered)."""
        if fn in self._overlays:
            self._overlays.remove(fn)
            self.update()

    def set_interaction(self, handler) -> None:
        """Install a mouse-interaction handler. While set, left-button press/move/release are
        delivered to it (in image coordinates) instead of the default ROI click. Replacing an
        existing handler calls its ``on_cancel`` first."""
        if self._interaction is handler:
            return
        self.clear_interaction()
        self._interaction = handler

    def clear_interaction(self) -> None:
        """Remove the active interaction handler, calling its ``on_cancel`` if present."""
        handler, self._interaction = self._interaction, None
        if handler is not None:
            cancel = getattr(handler, "on_cancel", None)
            if cancel is not None:
                try:
                    cancel()
                except Exception:  # a broken handler must not wedge the canvas
                    pass

    def _dispatch_interaction(self, name: str, event) -> bool:
        """Send an image-space mouse event to the active interaction handler. Returns True if a
        handler consumed it. Exceptions are swallowed so a buggy plugin can't crash the canvas."""
        handler = self._interaction
        if handler is None:
            return False
        method = getattr(handler, name, None)
        if method is None:
            return True  # handler is active but doesn't care about this event
        try:
            method(self.screen_to_image(event.position()), event)
        except Exception:
            pass
        return True

    def _right_press_consumed(self, event) -> bool:
        """Give the active interaction a chance to handle a right-click (e.g. N-Gon close).

        Unlike ``_dispatch_interaction``, this returns False when the handler lacks
        ``on_right_press`` so right-drag still pans for tools/plugins that don't opt in. Do not
        fold this back into ``_dispatch_interaction`` — the missing-method default is opposite.
        """
        handler = self._interaction
        if handler is None:
            return False
        method = getattr(handler, "on_right_press", None)
        if method is None:
            return False
        try:
            method(self.screen_to_image(event.position()), event)
        except Exception:
            pass
        return True

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
        if event.button() == Qt.MouseButton.LeftButton:
            if self._pan_tool or self._space_pan:
                self._panning = True
                self._last_pan_pos = event.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            elif not self._dispatch_interaction("on_press", event):
                self.imageClicked.emit(self.screen_to_image(event.position()))
        elif event.button() == Qt.MouseButton.RightButton and self._right_press_consumed(event):
            pass  # consumed by the active interaction (e.g. N-Gon close)
        elif event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._panning = True
            self._last_pan_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            pos = event.position()
            self._pan += pos - self._last_pan_pos
            self._last_pan_pos = pos
            self.update()
        elif not self._qimage.isNull():
            self._dispatch_interaction("on_move", event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._panning and event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.RightButton,
        ):
            self._panning = False
            self._apply_nav_cursor()
        elif event.button() == Qt.MouseButton.LeftButton:
            self._dispatch_interaction("on_release", event)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.reset_view()
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event) -> None:
        if self._qimage.isNull():
            return
        pixel = event.pixelDelta()
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)  # Cmd on macOS
        # Zoom for a mouse wheel (no pixelDelta) or when Cmd/Ctrl is held; a bare two-finger
        # trackpad scroll (pixelDelta present, no modifier) pans like a native image viewer.
        if pixel.isNull() or ctrl:
            delta = event.angleDelta().y() or pixel.y()
            if delta == 0:
                return
            self._zoom_at(1.25 if delta > 0 else 0.8, event.position())
        else:
            self._pan += QPointF(pixel)
            self.update()

    def event(self, e):
        # macOS sends pinch as a native gesture, not a wheel event.
        if e.type() == QEvent.Type.NativeGesture and e.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
            if not self._qimage.isNull():
                self._zoom_at(1.0 + e.value(), e.position())
            return True
        return super().event(e)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan = True
            self._apply_nav_cursor()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
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
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setTransform(self._transform)
        painter.drawImage(0, 0, self._qimage)

        # Overlays are drawn in screen space (after resetting the transform) so pen widths
        # and marker sizes stay constant regardless of image scale.
        painter.resetTransform()
        self._draw_roi(painter)
        self._draw_features(painter)
        self._draw_tracked(painter)
        self._draw_overlays(painter)

    def _draw_overlays(self, painter: QPainter) -> None:
        """Paint each registered plugin overlay in screen space. A failing overlay is removed so
        one buggy plugin can't break every subsequent paint."""
        for fn in list(self._overlays):
            painter.save()
            try:
                fn(painter, self)
            except Exception:
                self._overlays.remove(fn)
            finally:
                painter.restore()

    def _draw_roi(self, painter: QPainter) -> None:
        if not self._state.display_params["show_roi"]:
            return
        roi = self._state.roi
        if roi is None or not roi.corners:
            return
        screen_pts = [self.image_to_screen(x, y) for x, y in roi.corners]

        painter.setPen(QPen(QColor(255, 215, 0), 2))
        if roi.is_complete:
            painter.setBrush(QBrush(QColor(255, 215, 0, 40)))
            painter.drawPolygon(QPolygonF(screen_pts))
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if len(screen_pts) >= 2:
                painter.drawPolyline(QPolygonF(screen_pts))

        # Vertex handles only for low-vertex shapes; a circle (64-gon) reads as its outline.
        if len(screen_pts) <= 16:
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
        display = self._state.display_params
        if not display["show_markers"]:
            return
        radius = display["marker_size"]
        cyan = QColor(0, 255, 255)
        cyan.setAlpha(round(255 * display["marker_opacity"] / 100))
        painter.setPen(QPen(cyan, 1))
        painter.setBrush(QBrush(cyan))
        for x, y in feats:
            painter.drawEllipse(self.image_to_screen(x, y), radius, radius)

    def _draw_tracked(self, painter: QPainter) -> None:
        """Draw tracked positions for the current frame, with a motion trail from the previous
        frame. Kept points are green; cleanup-preview drops are red; already-removed points are
        not drawn."""
        state = self._state
        result = state.result
        if result is None or not state.current_in_range:
            return
        display = state.display_params
        if not display["show_markers"]:
            return
        radius = display["marker_size"]
        alpha = round(255 * display["marker_opacity"] / 100)
        trail_alpha = round(140 * display["marker_opacity"] / 100)
        show_box = display["show_window_box"]

        cut = state.global_to_cut(state.current_index)
        coords = result.coords_fw[cut]
        prev = result.coords_fw[cut - 1] if cut > 0 else None
        active = state.active_mask
        preview = self._preview_keep_mask

        # Marker/trail pens are constant per keep-state, so build them once rather than
        # allocating a QColor/QPen/QBrush per point inside the loop.
        def _pens(rgb):
            c = QColor(*rgb)
            c.setAlpha(alpha)
            t = QColor(*rgb)
            t.setAlpha(trail_alpha)
            return QPen(c, 1), QBrush(c), QPen(t, 1)

        kept_pen, kept_brush, kept_trail = _pens((0, 220, 0))
        drop_pen, drop_brush, drop_trail = _pens((255, 60, 60))

        box_pen = None
        if show_box:
            # Use the win_size tracking actually used (recorded on the result), not the live
            # param, which the user may have edited after tracking without re-running.
            half = (result.win_size or state.lk_params["win_size"]) / 2.0
            box_color = QColor(0, 220, 0)
            box_color.setAlpha(alpha)
            box_pen = QPen(box_color, 1)

        for p in range(result.n_points):
            if active is not None and not active[p]:
                continue
            keep = True if preview is None else bool(preview[p])
            pen, brush, trail_pen = (
                (kept_pen, kept_brush, kept_trail) if keep
                else (drop_pen, drop_brush, drop_trail)
            )
            x, y = coords[p][0], coords[p][1]
            here = self.image_to_screen(x, y)
            if show_box:
                # The window box is an image-space region, so map its corners through the
                # transform: it scales with zoom (unlike the constant-size markers).
                tl = self.image_to_screen(x - half, y - half)
                br = self.image_to_screen(x + half, y + half)
                painter.setPen(box_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(QRectF(tl, br))
            if prev is not None:
                painter.setPen(trail_pen)
                painter.drawLine(self.image_to_screen(prev[p][0], prev[p][1]), here)
            painter.setPen(pen)
            painter.setBrush(brush)
            painter.drawEllipse(here, radius, radius)
