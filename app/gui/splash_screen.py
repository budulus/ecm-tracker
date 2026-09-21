"""Lightweight startup branding, usable before the main window is imported."""

import os

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QGuiApplication,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication, QSplashScreen

from app.gui.icon_loader import _ICON_DIR


def _splash_pixmap(dpr: float) -> QPixmap:
    width, height = 600, 360
    pixmap = QPixmap(round(width * dpr), round(height * dpr))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        panel = QPainterPath()
        panel.addRoundedRect(QRectF(0, 0, width, height), 20, 20)
        painter.setClipPath(panel)

        background = QLinearGradient(0, 0, 0, height)
        background.setColorAt(0, QColor("#3b82f6"))
        background.setColorAt(1, QColor("#1d4ed8"))
        painter.fillPath(panel, background)

        glow = QRadialGradient(460, 40, 370)
        glow.setColorAt(0, QColor(103, 232, 249, 42))
        glow.setColorAt(1, QColor(103, 232, 249, 0))
        painter.fillPath(panel, glow)

        # Render the original SVG directly so its colors and detail survive HiDPI.
        logo = QSvgRenderer(os.path.join(_ICON_DIR, "app_icon.svg"))
        logo.render(painter, QRectF(244, 62, 112, 112))

        title_font = QFont(QApplication.font())
        title_font.setPixelSize(36)
        title_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(title_font)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(QRectF(32, 194, 536, 54), Qt.AlignmentFlag.AlignCenter, "ECM Tracker")

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(103, 232, 249, 190))
        painter.drawRoundedRect(QRectF(282, 268, 36, 3), 1.5, 1.5)

        credit_font = QFont(QApplication.font())
        credit_font.setPixelSize(13)
        credit_font.setWeight(QFont.Weight.Normal)
        painter.setFont(credit_font)
        painter.setPen(QColor("#dbeafe"))
        painter.drawText(
            QRectF(32, 309, 536, 24),
            Qt.AlignmentFlag.AlignCenter,
            "Senecell AG, 2026",
        )
    finally:
        painter.end()
    return pixmap


class SplashScreen(QSplashScreen):
    """A static splash that stays visible until startup finishes, even if clicked."""

    def __init__(self) -> None:
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        super().__init__(screen, _splash_pixmap(screen.devicePixelRatio()))
        self.setWindowTitle("ECM Tracker")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # Keep the app-wide light QWidget background out of the rounded corners.
        self.setStyleSheet("QSplashScreen { background: transparent; }")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # QSplashScreen normally hides on click, leaving a blank wait at startup.
        event.accept()
