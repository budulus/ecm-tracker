# Writing ECM Tracker plugins

Start with [PLUGIN_CONTRACT.md](PLUGIN_CONTRACT.md). It defines the complete supported
workflow: data shapes and validity, frame/point identity, mutations, events, Qt ownership,
math conventions, installation, and tests.

This folder is the app's one plugin folder: every subfolder with an `__init__.py` is a plugin
(folders starting with `_` or `.` are not). Copy [_sdk/example](_sdk/example/__init__.py) here
under a new name (**Plugins → Open Plugins Folder**), then **Reload Plugins**.
The example uses only the public SDK; you do not need to read the host source.

**Plugins → Manage Plugins…** lists every plugin: check/uncheck to enable or disable it
(disabled plugins are not loaded and have no side-pane button), **Install from Zip…** to add
one, **Remove…** to move one to the trash. To share a plugin, zip its folder.

[Offline API reference](plugin-api.html) is generated from the same contract and Python
signatures/docstrings. [API_REFERENCE.txt](_sdk/API_REFERENCE.txt) is the plain-text version.
Exact host dependency versions and the release lockfile are in [_sdk](_sdk).

Source checkout:

    uv run python run.py --check-plugin plugins/_sdk/example --report plugin-check.json
    uv run python -m unittest discover -s plugins/_sdk/example -p test_*.py

Installed executable:

    ECMTracker.exe --check-plugin PATH_TO_PLUGIN --report plugin-check.json

The smoke test executes trusted plugin code; it is not a sandbox or a physics validator.
Use only PySide6 and bundled dependencies. Installing packages into an unrelated Python/uv
environment does not add them to a frozen host.

Updating the app replaces the files of the plugins it ships with and leaves every other
plugin folder alone; to customize an included plugin, copy it under a new folder name.
