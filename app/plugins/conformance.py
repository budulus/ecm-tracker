"""Explicit, trusted-code plugin smoke test, also usable in the packaged executable."""
from __future__ import annotations
import ast
import importlib.util
import json
import sys
import uuid
from pathlib import Path
from PySide6.QtWidgets import QApplication
from app.gui.main_window import MainWindow
from app.plugins.api import PluginContext
from app.plugins.manager import PluginManager, PluginRecord


def check_plugin(path):
    """Run a trusted plugin through empty-session launch, close, relaunch and unload.

    This executes arbitrary plugin code; it is NOT a sandbox or a scientific correctness test.
    Interactive/file-dependent operations need the author's own tests.
    """
    path = Path(path).resolve()
    if not (path / "__init__.py").is_file():
        raise ValueError("Expected a plugin folder containing __init__.py")
    for source in path.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            modules = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                       else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            if any(m.startswith("app.") and m not in ("app.plugins", "app.plugins.api", "app.plugins.analysis", "app.plugins.testing") for m in modules):
                raise ValueError(f"Unsupported host import in {source.name}; use app.plugins")
    name = "_ecm_check_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py",
                                                 submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    app = QApplication.instance() or QApplication([])
    host = MainWindow()
    ctx = PluginContext(host, name)
    record = None
    try:
        spec.loader.exec_module(module)
        cls = PluginManager._find_plugin_class(module)
        if cls is None:
            raise ValueError("Expose PLUGIN = YourPlugin")
        if type(cls.ORDER) is not int or not isinstance(cls.NAME, str) or not cls.NAME.strip():
            raise ValueError("NAME and ORDER metadata are invalid")
        if type(cls.API_VERSION) is not int or cls.API_VERSION != 1 or not isinstance(cls.DESCRIPTION, str):
            raise ValueError("Unsupported API_VERSION or DESCRIPTION")
        record = PluginRecord(name, cls.NAME, cls=cls, context=ctx)
        record.instance = cls(ctx)
        for _ in range(2):
            record.window = record.instance.launch()
            app.processEvents()
            host.signals.sequence_changed.emit()
            host.signals.result_changed.emit()
            host.signals.mask_changed.emit()
            host.signals.range_changed.emit()
            host.signals.seeds_changed.emit()
            host.signals.frame_changed.emit(0)
            if record.window is not None:
                record.window.close()
            app.processEvents()
        return {"plugin": cls.NAME, "api_version": cls.API_VERSION, "smoke_test": "passed"}
    finally:
        if record is not None:
            PluginManager._unload(record)
        ctx.dispose()
        host.close()
        host.deleteLater()
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plugin_dir")
    parser.add_argument("--report", help="Optional JSON report file (useful for GUI executables)")
    args = parser.parse_args(argv)
    try:
        report = check_plugin(args.plugin_dir)
        code = 0
    except Exception as exc:
        report = {"error": str(exc)}
        code = 1
    if args.report:
        from app.core.atomic_io import atomic_open
        with atomic_open(args.report) as handle:
            json.dump(report, handle, indent=2)
    if sys.stdout is not None:
        print(json.dumps(report))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
