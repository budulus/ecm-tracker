"""One-off: render app/gui/icons/app_icon.svg -> assets/app_icon.ico (multi-res).

Dev tool only -- not imported at runtime. Pillow is pulled in ephemerally, so it
never lands in pyproject.toml. Run after editing the SVG:

    uv run --with pillow python scripts/make_app_ico.py
"""
import io
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QBuffer, QIODevice, QRectF, Qt
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtWidgets import QApplication
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SVG = os.path.join(ROOT, "app", "gui", "icons", "app_icon.svg")
OUT = os.path.join(ROOT, "assets", "app_icon.ico")


def main() -> None:
    app = QApplication(sys.argv)  # keep referenced: an unreferenced one is GC'd
    pixmap = QPixmap(256, 256)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    QSvgRenderer(SVG).render(painter, QRectF(0, 0, 256, 256))
    painter.end()

    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    pixmap.save(buf, "PNG")
    img = Image.open(io.BytesIO(bytes(buf.data()))).convert("RGBA")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    img.save(
        OUT,
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
