# Launching ECM Tracker on Windows

Double-click **Start ECM Tracker.bat** in the application folder. It finds the
application relative to its own location, so no paths need editing. A command
window may appear briefly and will close after launching the application.

## Desktop shortcut with the application icon

1. Right-click **Start ECM Tracker.bat** and choose **Show more options > Send to >
   Desktop (create shortcut)** (or **Send to > Desktop** on Windows 10).
2. Right-click the new desktop shortcut and open **Properties > Change Icon**.
3. Browse to **assets\app_icon.ico** inside the application folder, select the
   icon, and apply the changes. If Windows says the batch file contains no icons,
   dismiss that message and browse to the icon file.
4. Rename the shortcut to **ECM Tracker**.

The custom icon belongs to the shortcut. Keep the batch file in the application
folder; create a shortcut on each computer after placing the folder where you
want it. Recreate the shortcut if you move the application folder later.

## First-time setup on another computer

1. If `uv` is not installed, double-click **Install uv.bat**. It downloads and
   runs the [official uv installer](https://docs.astral.sh/uv/getting-started/installation/),
   so an internet connection is required. If `uv` is already available, it skips
   installation. The window stays open so you can read the result or any errors.
2. Open a new terminal in the application folder after installation (close and
   reopen any existing terminals to pick up the updated PATH), then run `uv sync`
   once to create that computer's Python environment and install the dependencies.
3. Double-click **Start ECM Tracker.bat**.

Share the source application folder without your `.venv`; each computer needs
its own environment. The launcher uses the local `.venv` and does not install
dependencies automatically.
