import sys

from PySide6.QtWidgets import QApplication

from app.gui.icon_loader import load_app_icon
from app.gui.main_window import MainWindow
from app.gui.theme import apply_theme


def main() -> int:
    if sys.platform == "win32":
        # Give the process an explicit app id so Windows groups it under "ECM
        # Tracker" and shows our icon on the taskbar instead of Python's.
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Senecell.ECMTracker")
    # Must be set *before* QApplication construction: on macOS the native
    # application menu ("About <name>", Hide, Quit) is built during construction
    # and, when running unbundled (dev), takes its name from applicationName()
    # — which otherwise defaults to argv[0]'s basename ("main.py").
    QApplication.setApplicationName("ECM Tracker")
    app = QApplication(sys.argv)
    app.setWindowIcon(load_app_icon())
    apply_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
