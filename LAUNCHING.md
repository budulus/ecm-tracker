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

Install `uv`, then open a terminal in the application folder and run `uv sync`
once to create that computer's Python environment and install the dependencies.
Share the source application folder without your `.venv`; each computer needs
its own environment. The launcher uses the local `.venv` and does not install
dependencies automatically.
