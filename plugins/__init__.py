"""Installed ECM Tracker plugins.

Each subdirectory of this package is one plugin (a Python package with its own ``__init__.py``
that exposes ``PLUGIN = YourTrackerPluginSubclass``). The app discovers them at startup, lists
them in Plugins → Manage Plugins…, and gives each enabled one a side-pane button. See
``plugins/README.md`` to write your own.
"""
