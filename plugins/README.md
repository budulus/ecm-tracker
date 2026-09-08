# Writing ECM Tracker plugins

Start with [PLUGIN_CONTRACT.md](PLUGIN_CONTRACT.md). It defines the complete supported
workflow: data shapes and validity, frame/point identity, mutations, events, Qt ownership,
math conventions, installation, and tests.

Copy [_sdk/example](_sdk/example/__init__.py) into **Plugins → Open Plugins Folder**.
The example uses only the public SDK; you do not need to read the host source.

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

User plugins live outside the installation and override bundled names after Reload Plugins.
Legacy plugin folders remain readable; new releases do not overwrite them.
