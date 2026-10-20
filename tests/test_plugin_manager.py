"""Regressions for the unified plugin folder, enable/disable state, and the Plugin Manager.

Every test points the manager at a throwaway plugins folder (never the repo's ``plugins/``)
and a throwaway config dir, so installs/removals and the persisted enabled state are isolated.
Run with:  QT_QPA_PLATFORM=offscreen uv run python -m tests.test_plugin_manager
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

import app.plugins.manager as manager_module
from app.plugins import packages
from app.plugins.packages import PluginPackageError

REPO_PLUGINS = Path(__file__).resolve().parents[1] / "plugins"

PLUGIN_SOURCE = '''\
from PySide6.QtWidgets import QWidget
from app.plugins import TrackerPlugin


class Plugin(TrackerPlugin):
    NAME = {name!r}
    DESCRIPTION = {description!r}
    ORDER = {order}

    def launch(self):
        window = QWidget()
        window.show()
        return window


PLUGIN = Plugin
'''


def write_plugin(root: Path, plugin_id: str, name: str, order: int = 100, description: str = "") -> Path:
    folder = root / plugin_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "__init__.py").write_text(
        PLUGIN_SOURCE.format(name=name, description=description or f"{name} plugin", order=order),
        encoding="utf-8",
    )
    return folder


def write_zip(path: Path, files: dict) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return path


def plugin_zip(path: Path, prefix: str, name: str) -> Path:
    return write_zip(path, {prefix + "__init__.py": PLUGIN_SOURCE.format(name=name, description="", order=50),
                            prefix + "helpers.py": "VALUE = 1\n"})


def flush_deletes():
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


# ---- Qt-free package operations ------------------------------------------------------------
class PackageOperations(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ecm-packages-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plugins = self.root / "plugins"

    def test_archive_layouts(self):
        find = packages.find_archive_plugins
        self.assertEqual(find(["zones/__init__.py", "zones/a.py", "README.md"], "x"),
                         [packages.ArchivePlugin("zones", "zones/")])
        self.assertEqual(find(["__init__.py", "a.py"], "my-plugin-1.0"),
                         [packages.ArchivePlugin("my_plugin_1_0", "")])
        self.assertEqual(find(["repo-main/zones/__init__.py", "repo-main/README.md"], "repo-main"),
                         [packages.ArchivePlugin("zones", "repo-main/zones/")])
        self.assertEqual([p.plugin_id for p in find(["a/__init__.py", "b/__init__.py", "_sdk/x/__init__.py"], "x")],
                         ["a", "b"])
        with self.assertRaises(PluginPackageError):
            find(["docs/readme.md", "zones/a.py"], "x")
        with self.assertRaises(PluginPackageError):
            find(["a-b/__init__.py", "a_b/__init__.py"], "x")  # Both sanitize to a_b.

    def test_unsafe_archives_are_rejected_and_junk_ignored(self):
        for bad in ("../evil/__init__.py", "/abs/__init__.py", "C:/win/__init__.py", "ok/../../x.py"):
            archive = write_zip(self.root / "bad.zip", {bad: "x = 1\n"})
            with self.assertRaises(PluginPackageError, msg=bad):
                packages.inspect_zip(archive)
        link = zipfile.ZipInfo("zones/link.py")
        link.external_attr = 0o120777 << 16
        with zipfile.ZipFile(self.root / "link.zip", "w") as archive:
            archive.writestr("zones/__init__.py", "")
            archive.writestr(link, "/etc/passwd")
        with self.assertRaises(PluginPackageError):
            packages.inspect_zip(self.root / "link.zip")
        (self.root / "not.zip").write_text("plain text", encoding="utf-8")
        with self.assertRaises(PluginPackageError):
            packages.inspect_zip(self.root / "not.zip")
        junk = write_zip(self.root / "junk.zip", {
            "zones/__init__.py": "", "zones/__pycache__/a.cpython-312.pyc": "", "__MACOSX/zones/._a": "",
            "zones/.DS_Store": "",
        })
        packages.install_zip(junk, self.plugins)
        self.assertEqual(sorted(p.name for p in (self.plugins / "zones").iterdir()), ["__init__.py"])

    def test_sanitize_plugin_id(self):
        self.assertEqual(packages.sanitize_plugin_id("my-plugin"), "my_plugin")
        self.assertEqual(packages.sanitize_plugin_id("3d view"), "plugin_3d_view")
        self.assertEqual(packages.sanitize_plugin_id("class"), "class_plugin")
        with self.assertRaises(PluginPackageError):
            packages.sanitize_plugin_id("---")

    def test_static_metadata(self):
        meta = packages.read_static_metadata(REPO_PLUGINS / "affine_zones")
        self.assertEqual((meta["name"], meta["order"]), ("Affine Zone Tool", 20))
        folder = self.plugins / "two"
        folder.mkdir(parents=True)
        (folder / "__init__.py").write_text(
            "class Helper:\n    NAME = 'helper'\n\nclass Real:\n    NAME = 'Real one'\n    ORDER = 7\n\nPLUGIN = Real\n",
            encoding="utf-8")
        self.assertEqual(packages.read_static_metadata(folder), {"name": "Real one", "order": 7})
        (folder / "__init__.py").write_text("def broken(:\n", encoding="utf-8")
        self.assertEqual(packages.read_static_metadata(folder), {})

    def test_install_conflict_overwrite_and_rollback(self):
        write_plugin(self.plugins, "zones", "Old")
        archive = plugin_zip(self.root / "zones.zip", "zones/", "New")
        with self.assertRaises(PluginPackageError):
            packages.install_zip(archive, self.plugins)
        self.assertIn("'Old'", (self.plugins / "zones/__init__.py").read_text(encoding="utf-8"))

        # A failure while moving the second plugin restores the first plugin's old version.
        both = write_zip(self.root / "both.zip", {"zones/__init__.py": "NEW = 1\n", "zz/__init__.py": ""})
        real_move = packages.shutil.move
        def flaky_move(src, dst):
            if Path(dst).name == "zz":
                raise OSError("injected")
            return real_move(src, dst)
        with patch.object(packages.shutil, "move", side_effect=flaky_move):
            with self.assertRaises(OSError):
                packages.install_zip(both, self.plugins, overwrite=True)
        self.assertIn("'Old'", (self.plugins / "zones/__init__.py").read_text(encoding="utf-8"))
        self.assertEqual(sorted(p.name for p in self.plugins.iterdir()), ["zones"])  # No zz, no backups.

        self.assertEqual(packages.install_zip(archive, self.plugins, overwrite=True), ["zones"])
        self.assertIn("'New'", (self.plugins / "zones/__init__.py").read_text(encoding="utf-8"))
        self.assertTrue((self.plugins / "zones/helpers.py").is_file())
        self.assertEqual(sorted(p.name for p in self.plugins.iterdir()), ["zones"])

    def test_migrate_moves_new_plugins_and_skips_conflicts(self):
        legacy = self.root / "legacy"
        write_plugin(legacy, "mine", "Mine")
        write_plugin(legacy, "zones", "Override")
        write_plugin(self.plugins, "zones", "Shipped")
        (legacy / "_private").mkdir()
        moved, skipped = packages.migrate_plugins(legacy, self.plugins)
        self.assertEqual((moved, skipped), (["mine"], ["zones"]))
        self.assertTrue((self.plugins / "mine/__init__.py").is_file())
        self.assertFalse((legacy / "mine").exists())
        self.assertTrue((legacy / "zones").exists())  # Never overwrite; left for the user.
        self.assertEqual(packages.migrate_plugins(self.root / "missing", self.plugins), ([], []))


# ---- manager + GUI ---------------------------------------------------------------------------
class ManagerBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ecm-manager-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = self.root / "config"
        self.plugins = self.root / "plugins"
        write_plugin(self.plugins, "alpha", "Alpha", order=10)
        write_plugin(self.plugins, "beta", "Beta", order=20)
        for active in (
            patch.dict(os.environ, {"TRACKER_CONFIG_DIR": str(self.config)}),
            patch.dict(sys.modules),  # Restores the real ``plugins`` package afterwards.
            patch.object(manager_module, "PLUGINS_DIR", self.plugins),
        ):
            active.start()
            self.addCleanup(active.stop)
        for name in [m for m in sys.modules if m == "plugins" or m.startswith("plugins.")]:
            del sys.modules[name]

    def disable(self, *plugin_ids):
        self.config.mkdir(parents=True, exist_ok=True)
        (self.config / "settings.json").write_text(
            json.dumps({"plugins": {"disabled": list(plugin_ids)}}), encoding="utf-8")

    def saved_disabled(self):
        data = json.loads((self.config / "settings.json").read_text(encoding="utf-8"))
        return data["plugins"]["disabled"]

    def host(self):
        from app.gui.main_window import MainWindow
        host = MainWindow()
        self.addCleanup(flush_deletes)
        self.addCleanup(host.deleteLater)
        self.addCleanup(host.close)
        return host

    @staticmethod
    def panel_texts(host):
        layout = host.plugin_manager._panel_layout
        return [layout.itemAt(i).widget().text() for i in range(layout.count())
                if isinstance(layout.itemAt(i).widget(), QPushButton)]

    @staticmethod
    def item(dialog, plugin_id):
        return next(dialog.list.item(r) for r in range(dialog.list.count())
                    if dialog.list.item(r).data(Qt.ItemDataRole.UserRole) == plugin_id)

    def test_disabled_plugin_is_listed_but_not_imported_or_shown(self):
        self.disable("beta")
        host = self.host()
        records = {r.plugin_id: r for r in host.plugin_manager.records()}
        self.assertEqual(set(records), {"alpha", "beta"})
        self.assertFalse(records["beta"].enabled)
        self.assertEqual(records["beta"].name, "Beta")  # Read statically, not imported.
        self.assertNotIn("plugins.beta", sys.modules)
        self.assertIn("plugins.alpha", sys.modules)
        texts = self.panel_texts(host)
        self.assertIn("Alpha", texts)
        self.assertNotIn("Beta", texts)
        self.assertIn("Manage plugins…", texts)

    def test_new_plugins_start_enabled(self):
        host = self.host()
        self.assertTrue(all(r.enabled and r.cls is not None for r in host.plugin_manager.records()))
        self.assertEqual([r.plugin_id for r in host.plugin_manager.records()], ["alpha", "beta"])

    def test_toggling_in_manager_updates_panel_and_persists(self):
        self.disable("beta")
        host = self.host()
        manager = host.plugin_manager
        manager.launch("alpha")
        alpha_window = manager._records["alpha"].window
        dialog = manager.open_manager()

        self.item(dialog, "beta").setCheckState(Qt.CheckState.Checked)
        QCoreApplication.processEvents()
        self.assertTrue(manager._records["beta"].enabled)
        self.assertIsNotNone(manager._records["beta"].cls)
        self.assertIn("Beta", self.panel_texts(host))
        self.assertEqual(self.saved_disabled(), [])
        self.assertTrue(alpha_window.isVisible())  # Enabling beta leaves alpha's window alone.

        manager.launch("beta")
        beta_window = manager._records["beta"].window
        self.item(dialog, "alpha").setCheckState(Qt.CheckState.Unchecked)
        QCoreApplication.processEvents()
        self.assertFalse(manager._records["alpha"].enabled)
        self.assertNotIn("Alpha", self.panel_texts(host))
        self.assertNotIn("plugins.alpha", sys.modules)
        self.assertEqual(self.saved_disabled(), ["alpha"])
        self.assertTrue(beta_window.isVisible())
        self.assertEqual(self.item(dialog, "alpha").checkState(), Qt.CheckState.Unchecked)

        # The choice survives a restart (new manager reading the persisted state).
        records = {r.plugin_id: r for r in manager_module.PluginManager(None).discover()}
        self.assertFalse(records["alpha"].enabled)
        self.assertTrue(records["beta"].enabled)

    def test_toggle_applies_even_when_settings_cannot_be_saved(self):
        host = self.host()
        with patch.object(manager_module, "_save_disabled_ids", side_effect=OSError("read-only")):
            rec = host.plugin_manager.set_enabled("alpha", False)
        self.assertFalse(rec.enabled)
        self.assertNotIn("Alpha", self.panel_texts(host))

    def test_case_only_clash_replaces_instead_of_duplicating(self):
        probe = self.root / "CaseProbe"
        probe.mkdir()
        if not (self.root / "caseprobe").exists():
            self.skipTest("case-sensitive filesystem")
        host = self.host()
        archive = plugin_zip(self.root / "Alpha.zip", "Alpha/", "Alpha upper")
        with patch("app.plugins.manager.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
            self.assertEqual(host.plugin_manager.install_from_zip(zip_path=str(archive)), ["Alpha"])
        ids = [r.plugin_id for r in host.plugin_manager.records()]
        self.assertEqual(sorted(ids), ["Alpha", "beta"])
        self.assertEqual(host.plugin_manager._records["Alpha"].name, "Alpha upper")

    def test_reload_keeps_disabled_state(self):
        self.disable("alpha")
        host = self.host()
        host.plugin_manager.reload()
        self.assertFalse(host.plugin_manager._records["alpha"].enabled)
        self.assertNotIn("Alpha", self.panel_texts(host))

    def test_install_from_zip_enables_and_shows_plugin(self):
        self.disable("gamma")  # A stale disabled entry must not hide a fresh install.
        host = self.host()
        archive = plugin_zip(self.root / "gamma-main.zip", "gamma-main/gamma/", "Gamma")
        self.assertEqual(host.plugin_manager.install_from_zip(zip_path=str(archive)), ["gamma"])
        rec = host.plugin_manager._records["gamma"]
        self.assertTrue(rec.enabled)
        self.assertIsNone(rec.error)
        self.assertTrue((self.plugins / "gamma" / "helpers.py").is_file())
        self.assertIn("Gamma", self.panel_texts(host))
        self.assertEqual(self.saved_disabled(), [])
        dialog = host.plugin_manager._dialog
        self.assertIsNotNone(dialog)
        self.assertEqual(dialog._current_id(), "gamma")
        self.assertEqual([r.plugin_id for r in host.plugin_manager.records()], ["alpha", "beta", "gamma"])

    def test_install_over_existing_needs_confirmation(self):
        host = self.host()
        manager = host.plugin_manager
        manager.launch("alpha")
        window = manager._records["alpha"].window
        archive = plugin_zip(self.root / "alpha.zip", "alpha/", "Alpha 2")
        with patch("app.plugins.manager.QMessageBox.question", return_value=QMessageBox.StandardButton.Cancel):
            self.assertEqual(manager.install_from_zip(zip_path=str(archive)), [])
        self.assertEqual(manager._records["alpha"].name, "Alpha")
        self.assertTrue(window.isVisible())
        with patch("app.plugins.manager.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
            self.assertEqual(manager.install_from_zip(zip_path=str(archive)), ["alpha"])
        self.assertEqual(manager._records["alpha"].name, "Alpha 2")  # Fresh import of the new code.
        self.assertFalse(window.isVisible())
        self.assertIn("Alpha 2", self.panel_texts(host))

    def test_invalid_zip_warns_and_changes_nothing(self):
        host = self.host()
        archive = write_zip(self.root / "docs.zip", {"docs/readme.md": "hi"})
        with patch("app.plugins.manager.QMessageBox.warning") as warning:
            self.assertEqual(host.plugin_manager.install_from_zip(zip_path=str(archive)), [])
        warning.assert_called_once()
        self.assertEqual(sorted(p.name for p in self.plugins.iterdir()), ["alpha", "beta"])

    def test_remove_moves_folder_to_trash(self):
        self.disable("beta")
        host = self.host()
        manager = host.plugin_manager
        trash = self.root / "trash"
        def fake_trash(path):
            trash.mkdir(exist_ok=True)
            Path(path).rename(trash / Path(path).name)
            return True
        with patch.object(manager_module, "_move_to_trash", side_effect=fake_trash), \
             patch("app.plugins.manager.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
            self.assertTrue(manager.remove("beta"))
        self.assertFalse((self.plugins / "beta").exists())
        self.assertTrue((trash / "beta" / "__init__.py").is_file())
        self.assertNotIn("beta", manager._records)
        self.assertEqual(self.saved_disabled(), [])

        with patch("app.plugins.manager.QMessageBox.question", return_value=QMessageBox.StandardButton.Cancel):
            self.assertFalse(manager.remove("alpha"))
        self.assertTrue((self.plugins / "alpha").is_dir())
        self.assertIn("Alpha", self.panel_texts(host))

    def test_legacy_user_plugins_move_into_plugins_folder(self):
        legacy = self.config / "plugins"
        write_plugin(legacy, "legacy_one", "Legacy One")
        write_plugin(legacy, "alpha", "Alpha override")
        host = self.host()
        records = {r.plugin_id: r for r in host.plugin_manager.records()}
        self.assertEqual(records["legacy_one"].name, "Legacy One")
        self.assertTrue((self.plugins / "legacy_one" / "__init__.py").is_file())
        self.assertEqual(records["alpha"].name, "Alpha")  # The existing folder wins; nothing overwritten.
        self.assertTrue((legacy / "alpha").is_dir())

    def test_broken_plugin_is_reported_with_static_name(self):
        folder = write_plugin(self.plugins, "broken", "Broken Tool")
        with open(folder / "__init__.py", "a", encoding="utf-8") as handle:
            handle.write("raise RuntimeError('boom')\n")
        host = self.host()
        rec = host.plugin_manager._records["broken"]
        self.assertIn("boom", rec.error)
        self.assertEqual(rec.name, "Broken Tool")
        self.assertIn("⚠ Broken Tool", self.panel_texts(host))
        dialog = host.plugin_manager.open_manager(select="broken")
        self.assertFalse(dialog.error_view.isHidden())
        self.assertIn("boom", dialog.error_view.toPlainText())

    def test_manager_search_filters_list(self):
        host = self.host()
        dialog = host.plugin_manager.open_manager()
        dialog.search.setText("bet")
        visible = [dialog.list.item(r).data(Qt.ItemDataRole.UserRole) for r in range(dialog.list.count())
                   if not dialog.list.item(r).isHidden()]
        self.assertEqual(visible, ["beta"])
        self.assertEqual(dialog._current_id(), "beta")

    def test_panel_manage_button_opens_manager(self):
        host = self.host()
        layout = host.plugin_manager._panel_layout
        button = next(layout.itemAt(i).widget() for i in range(layout.count())
                      if isinstance(layout.itemAt(i).widget(), QPushButton)
                      and layout.itemAt(i).widget().text() == "Manage plugins…")
        button.click()
        self.assertIsNotNone(host.plugin_manager._dialog)
        self.assertTrue(host.plugin_manager._dialog.isVisible())
        host.plugin_manager._dialog.reject()
        self.assertIsNone(host.plugin_manager._dialog)


if __name__ == "__main__":
    unittest.main(verbosity=2)
