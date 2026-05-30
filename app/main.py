import sys

from PyQt5.QtWidgets import QApplication

from app.gui.main_window import MainWindow
from app.gui.theme import apply_theme


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("ECM Tracker")
    apply_theme(app)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
