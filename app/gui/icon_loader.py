"""Render and tint bundled SVG icons into themed ``QIcon``s.

The SVGs in ``icons/`` are authored in plain black; this module re-colors them at
load time via a ``SourceIn`` composition so a single source file can serve the
dark normal state, the white accent (primary-action) state, and a faded disabled
state. Results are memoized, and rendering is done at the device pixel ratio so
icons stay crisp on HiDPI (Retina) displays.

Qt-only — no third-party dependency. ``QtSvg`` ships with the PyQt5 wheel.
"""

import os

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtWidgets import QApplication

_ICON_DIR = os.path.join(os.path.dirname(__file__), "icons")

# Theme colors for icon glyphs (kept in sync with theme.LIGHT_QSS).
NORMAL = "#3a3f44"   # default toolbar/menu glyph
ACCENT = "#ffffff"   # glyph on the accent "Run" button
DISABLED = "#b3b9c0"  # faded glyph for disabled actions

_cache: dict = {}


def _device_pixel_ratio() -> float:
    app = QApplication.instance()
    return float(app.devicePixelRatio()) if app is not None else 1.0


def _tinted_pixmap(name: str, color: str, size: int, dpr: float) -> QPixmap:
    """Render ``icons/<name>.svg`` at ``size`` px and recolor it to ``color``."""
    renderer = QSvgRenderer(os.path.join(_ICON_DIR, f"{name}.svg"))
    px = max(1, int(round(size * dpr)))
    pixmap = QPixmap(px, px)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    renderer.render(painter, QRectF(0, 0, px, px))
    # Replace the rendered (black) glyph with the requested color, preserving alpha.
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), QColor(color))
    painter.end()

    pixmap.setDevicePixelRatio(dpr)
    return pixmap


def load_icon(name: str, color: str = NORMAL, size: int = 20) -> QIcon:
    """Return a themed ``QIcon`` for ``icons/<name>.svg``.

    The icon carries an explicit Normal pixmap (``color``) and a Disabled pixmap
    (faded) so greyed-out actions read correctly. Memoized by
    ``(name, color, size, dpr)``.
    """
    dpr = _device_pixel_ratio()
    key = (name, color, size, dpr)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    icon = QIcon()
    icon.addPixmap(_tinted_pixmap(name, color, size, dpr), QIcon.Normal, QIcon.On)
    icon.addPixmap(_tinted_pixmap(name, color, size, dpr), QIcon.Normal, QIcon.Off)
    icon.addPixmap(_tinted_pixmap(name, DISABLED, size, dpr), QIcon.Disabled, QIcon.On)
    icon.addPixmap(_tinted_pixmap(name, DISABLED, size, dpr), QIcon.Disabled, QIcon.Off)
    _cache[key] = icon
    return icon
