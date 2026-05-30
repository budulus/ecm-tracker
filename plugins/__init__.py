"""Installed ECM Tracker plugins.

Each subdirectory of this package is one plugin (a Python package with its own ``__init__.py``
that exposes ``PLUGIN = YourTrackerPluginSubclass``). The app discovers them at startup and
lists them in the Plugins menu. See ``plugins/README.md`` to write your own.
"""
