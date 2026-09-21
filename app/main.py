import sys

from PySide6.QtWidgets import QApplication

from app.gui.icon_loader import load_app_icon
from app.gui.splash_screen import SplashScreen
from app.gui.theme import apply_theme


def main() -> int:
    if "--check-plugin" in sys.argv:
        from app.plugins.conformance import main as check_main
        return check_main(sys.argv[sys.argv.index("--check-plugin") + 1:])
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
    splash = SplashScreen()
    try:
        splash.show()
        app.processEvents()

        # Paint the splash before importing OpenCV, NumPy, and the plugin UI.
        from app.gui.main_window import MainWindow

        window = MainWindow()
        window.show()
        splash.finish(window)
    finally:
        splash.close()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
