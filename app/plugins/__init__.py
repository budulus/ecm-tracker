"""Feature Tracker plugin SDK.

Public API for plugin authors (everything you need is in :mod:`app.plugins.api`)::

    from app.plugins import TrackerPlugin, PluginContext, CanvasInteraction

See ``plugins/README.md`` for the author's guide and the bundled example plugins.
"""
from app.plugins.api import (
    CanvasInteraction,
    PluginContext,
    PluginSignals,
    TrackerPlugin,
)

__all__ = [
    "TrackerPlugin",
    "PluginContext",
    "CanvasInteraction",
    "PluginSignals",
]
