import sys

USAGE = """\
usage: tracker                          start the GUI
       tracker --script FILE.py [ARGS]  start the GUI and run a script in it (output in the console)
       tracker --run FILE.py [ARGS]     run a script headless, without any window
       tracker --check-plugin DIR       check a plugin package against the SDK contract
"""


def main() -> int:
    if "--help" in sys.argv[1:2] or "-h" in sys.argv[1:2]:
        print(USAGE)
        return 0
    if "--run" in sys.argv:
        # Headless: imported before PySide6 so a script never pays for (or needs) a display.
        from app.scripting.cli import main as run_main
        return run_main(sys.argv[sys.argv.index("--run") + 1:])
    if "--check-plugin" in sys.argv:
        from app.plugins.conformance import main as check_main
        return check_main(sys.argv[sys.argv.index("--check-plugin") + 1:])
    startup_script = None
    if "--script" in sys.argv:
        index = sys.argv.index("--script")
        startup_script = sys.argv[index + 1:]
        del sys.argv[index:]
        if not startup_script:
            print(USAGE, file=sys.stderr)
            return 2

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from app.gui.icon_loader import load_app_icon
    from app.gui.splash_screen import SplashScreen
    from app.gui.theme import apply_theme

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
    if startup_script:
        path, *args = startup_script
        QTimer.singleShot(0, lambda: window.scripting.run_file(path, args=args))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
