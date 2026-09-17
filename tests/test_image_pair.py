"""Pair geometry, reproducibility, residual tracking, and Qt workflow regressions."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from app.core.image_pair import AlignedImagePairSequence, load_pair_setup, save_pair_setup
from app.core.pair_transform import PairAlignment
from app.core.image_sequence import ImageSequence
from app.core.roi import ROI
from app.core.tracking import DEFAULT_LK, track
from app.gui.main_window import MainWindow
from app.gui.pair_alignment import ImagePairDialog, PairAlignmentDialog
from app.plugins.api import PluginContext


class ImagePairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ecm-pair-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        config = patch.dict(os.environ, {"TRACKER_CONFIG_DIR": str(self.root / "config")})
        config.start()
        self.addCleanup(config.stop)
        self.reference = np.random.default_rng(42).integers(0, 256, (100, 140, 3), dtype=np.uint8)
        self.destination = self.reference.copy()

    def pair(self, shift=(0, 0)):
        # Filenames deliberately sort in the opposite order to the selected roles.
        paths = [str(self.root / "z_reference.png"), str(self.root / "a_destination.png")]
        for path, image in zip(paths, (self.reference, self.destination)):
            self.assertTrue(cv2.imwrite(path, image))
        sequence = AlignedImagePairSequence(paths, *shift)
        self.assertTrue(sequence.validate_all())
        return sequence

    def host(self, sequence):
        host = MainWindow()
        self.addCleanup(host.deleteLater)
        self.addCleanup(host.close)
        self.assertTrue(host._load_sequence_candidate(sequence, str(self.root)))
        return host

    def dialog(self, sequence):
        dialog = PairAlignmentDialog(sequence)
        self.addCleanup(dialog.deleteLater)
        self.addCleanup(dialog.close)
        dialog.show()
        self.app.processEvents()
        return dialog

    def test_exact_crops_unequal_sizes_and_all_shift_signs(self):
        self.destination = np.arange(120 * 110 * 3, dtype=np.uint8).reshape(120, 110, 3)
        pair = self.pair()
        for dx, dy in ((0, 0), (13, 7), (-13, -7), (13, -7), (-13, 7)):
            with self.subTest(shift=(dx, dy)):
                candidate = pair.with_translation(dx, dy)
                candidate.validate_all()
                x0, y0 = max(0, dx), max(0, dy)
                x1, y1 = min(140, 110 + dx), min(100, 120 + dy)
                self.assertEqual(candidate.crop, (x0, y0, x1 - x0, y1 - y0))
                np.testing.assert_array_equal(candidate.load_bgr(0), self.reference[y0:y1, x0:x1])
                np.testing.assert_array_equal(candidate.load_bgr(1), self.destination[y0-dy:y1-dy, x0-dx:x1-dx])
                self.assertEqual(candidate.load_bgr(0).shape, candidate.load_bgr(1).shape)
                self.assertFalse(candidate.load_bgr(1).flags.writeable)
                self.assertTrue(candidate.load_bgr(1).flags.c_contiguous)
        with self.assertRaisesRegex(ValueError, "dimensions changed"):
            ImageSequence(pair.paths).validate_all()

    def test_invalid_geometry_and_indices(self):
        pair = self.pair()
        for shift in ((140, 0), (-140, 0), (0, 100), (0, -100)):
            with self.assertRaisesRegex(ValueError, "do not overlap"):
                pair.with_translation(*shift).validate_all()
        for value in (0.5, True, float("nan")):
            with self.assertRaises(ValueError):
                pair.with_translation(value, 0)
        for index in (-1, 2):
            with self.assertRaises(IndexError):
                pair.load_bgr(index)

    def test_shared_normalization_and_guarded_cached_frames(self):
        self.reference = np.full((100, 140), 25700, dtype=np.uint16)
        self.destination = np.full((120, 140, 4), 25700, dtype=np.uint16)
        pair = self.pair((1, -1))
        self.assertTrue(np.all(pair.load_bgr(0) == 100))
        self.assertTrue(np.all(pair.load_bgr(1) == 100))
        path = pair.paths[1]
        stat = os.stat(path)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        with self.assertRaisesRegex(OSError, "changed on disk"):
            pair.load_bgr(0)  # Even the other cached frame checks the full pair.
        with self.assertRaises(OSError):
            save_pair_setup(self.root / "bad", pair)

    def test_fingerprint_binds_alignment_and_order(self):
        self.reference.fill(70)
        self.destination.fill(90)
        pair = self.pair((3, 0))
        changed = pair.with_translation(-3, 0)
        changed.validate_all()
        np.testing.assert_array_equal(pair.load_bgr(0), changed.load_bgr(0))
        np.testing.assert_array_equal(pair.load_bgr(1), changed.load_bgr(1))
        self.assertNotEqual(pair.fingerprint, changed.fingerprint)
        reversed_pair = AlignedImagePairSequence(pair.paths[::-1], *pair.translation)
        reversed_pair.validate_all()
        self.assertNotEqual(pair.fingerprint, reversed_pair.fingerprint)

    def test_setup_roundtrip_relocation_and_source_change(self):
        pair = self.pair((-9, 4))
        path = Path(save_pair_setup(self.root / "setup", pair))
        self.assertTrue(path.name.endswith(".ecmpair.json"))
        metadata = json.loads(path.read_text())
        self.assertEqual(metadata["sources"], ["z_reference.png", "a_destination.png"])
        restored = load_pair_setup(path)
        self.assertEqual(restored.fingerprint, pair.fingerprint)
        moved = self.root / "moved"
        moved.mkdir()
        for source in [path, *map(Path, pair.paths)]:
            shutil.copy(source, moved / source.name)
        relocated = load_pair_setup(moved / path.name)
        self.assertEqual(relocated.fingerprint, pair.fingerprint)
        self.assertEqual(Path(relocated.paths[0]).parent, moved)
        cv2.imwrite(str(moved / "a_destination.png"), np.zeros_like(self.destination))
        with self.assertRaisesRegex(ValueError, "source images have changed"):
            load_pair_setup(moved / path.name)

    def test_corrupt_setup_missing_sources_and_atomic_save(self):
        pair = self.pair()
        path = Path(save_pair_setup(self.root / "setup", pair))
        original = path.read_bytes()
        metadata = json.loads(original)
        for key, value in (("translation", [True, 0]), ("crop", [0, 0, 1, 1]),
                           ("version", True), ("sources", ["missing.png", pair.paths[1]]),
                           ("source_fingerprints", []), ("sequence_fingerprint", "invalid")):
            bad = dict(metadata)
            bad[key] = value
            path.write_text(json.dumps(bad))
            with self.subTest(field=key), self.assertRaises((ValueError, OSError)):
                load_pair_setup(path)
        path.write_bytes(original)
        with patch("app.core.atomic_io.os.replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                save_pair_setup(path, pair)
        self.assertEqual(path.read_bytes(), original)

    def test_two_frame_tracking_measures_residual_motion(self):
        gray = np.random.default_rng(51).integers(0, 256, (240, 320), dtype=np.uint8)
        gray = cv2.GaussianBlur(gray, (5, 5), 1)
        self.reference = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.destination = cv2.warpAffine(self.reference, np.float32([[1, 0, 23], [0, 1, -12]]), (320, 240))
        pair = self.pair((-21, 11))  # Residual motion should be (+2, -1).
        first = pair.load_gray(0)
        mask = np.zeros_like(first)
        mask[30:-30, 30:-30] = 255
        seeds = cv2.goodFeaturesToTrack(first, 100, .03, 10, mask=mask).reshape(-1, 2)
        result = track(pair, 0, 1, seeds, DEFAULT_LK)
        self.assertEqual(result.n_frames, 2)
        valid = result.status_fw[1].astype(bool)
        self.assertGreater(valid.mean(), .95)
        residual = np.median(result.coords_fw[1, valid] - result.coords_fw[0, valid], axis=0)
        np.testing.assert_allclose(residual, [2, -1], atol=.15)
        self.assertLess(np.median(result.fb_mean_error[valid]), .3)

    def test_dialog_opacity_keyboard_crop_and_unchanged_source(self):
        pair = self.pair()
        dialog = self.dialog(pair)
        self.assertEqual(dialog.opacity_slider.value(), 50)
        for opacity in (0, 100):
            dialog.opacity_slider.setValue(opacity)
            self.assertEqual(dialog.canvas.opacity, opacity / 100)
        QTest.keyClick(dialog.canvas, Qt.Key.Key_Right)
        QTest.keyClick(dialog.canvas, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(dialog.translation, (1, 10))
        self.assertEqual((dialog.x_offset.value(), dialog.y_offset.value()), (1, 10))
        dialog.x_offset.setValue(140)
        self.assertFalse(dialog.apply_button.isEnabled())
        dialog.accept()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)
        dialog.x_offset.setValue(2)
        self.assertTrue(dialog.apply_button.isEnabled())
        self.assertEqual(pair.translation, (0, 0))
        dialog.reject()
        self.assertEqual(pair.translation, (0, 0))

    def test_opacity_renders_reference_and_destination_endpoints(self):
        self.reference[:] = (15, 25, 35)
        self.destination[:] = (100, 110, 120)
        dialog = self.dialog(self.pair())
        position = dialog.canvas.image_to_screen(60, 40).toPoint()
        for opacity, expected_rgb in ((0, (35, 25, 15)), (100, (120, 110, 100))):
            dialog.opacity_slider.setValue(opacity)
            pixel = dialog.canvas.grab().toImage().pixelColor(position)
            self.assertEqual((pixel.red(), pixel.green(), pixel.blue()), expected_rgb)

    def test_drag_is_in_image_pixels_at_multiple_zoom_levels(self):
        dialog = self.dialog(self.pair())
        canvas = dialog.canvas
        for zoom in (0.5, 1, 2):
            canvas.set_translation(0, 0)
            canvas.fit_view()
            canvas.zoom_at(zoom)
            self.app.processEvents()
            start = canvas.image_to_screen(60, 45).toPoint()
            end = canvas.image_to_screen(67, 41).toPoint()
            QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
            QTest.mouseMove(canvas, end)
            QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=end)
            self.assertEqual(canvas.translation, (7, -4))
            before = canvas.image_to_screen(60, 45)
            QTest.mousePress(canvas, Qt.MouseButton.RightButton, pos=start)
            QTest.mouseMove(canvas, start + (end - start))
            QTest.mouseRelease(canvas, Qt.MouseButton.RightButton, pos=end)
            self.assertEqual(canvas.translation, (7, -4))
            after = canvas.image_to_screen(60, 45)
            self.assertGreater((after - before).manhattanLength(), 1)

    def test_alignment_transaction_cancel_noop_confirm_and_reset(self):
        pair = self.pair()
        host = self.host(pair)
        host.state.roi = ROI([(5, 5), (90, 5), (90, 80), (5, 80)])
        host.state.features = np.array([[40, 40]], dtype=np.float32)
        host.state.result = track(pair, 0, 1, host.state.features, DEFAULT_LK)
        host.state.active_mask = np.ones(1, dtype=bool)
        host.state.undo_stack = [np.ones(1, dtype=bool)]
        old_revision = host.state.revision
        old_result = host.state.result
        events = []
        host.signals.sequence_changed.connect(lambda: events.append(host.state.sequence))

        def run(shift, accepted, confirm):
            with patch("app.gui.main_window.PairAlignmentDialog") as cls, patch.object(
                QMessageBox, "question", return_value=confirm
            ) as question:
                cls.return_value.exec.return_value = accepted
                cls.return_value.alignment = PairAlignment(translation=shift)
                host._adjust_pair_alignment()
                return question.call_count

        self.assertEqual(run((2, 3), QDialog.DialogCode.Rejected, QMessageBox.StandardButton.Yes), 0)
        self.assertEqual(run((0, 0), QDialog.DialogCode.Accepted, QMessageBox.StandardButton.Yes), 0)
        self.assertIs(host.state.result, old_result)
        self.assertEqual(host.state.revision, old_revision)
        self.assertEqual(run((2, 3), QDialog.DialogCode.Accepted, QMessageBox.StandardButton.No), 1)
        self.assertIs(host.state.sequence, pair)
        self.assertIs(host.state.result, old_result)
        self.assertFalse(events)
        self.assertEqual(run((2, 3), QDialog.DialogCode.Accepted, QMessageBox.StandardButton.Yes), 1)
        self.assertEqual(host.state.sequence.translation, (2, 3))
        self.assertIsNone(host.state.roi)
        self.assertIsNone(host.state.features)
        self.assertIsNone(host.state.result)
        self.assertIsNone(host.state.active_mask)
        self.assertEqual(host.state.undo_stack, [])
        self.assertEqual(len(events), 1)

    def test_confirmation_rechecks_session_and_source_changes(self):
        pair = self.pair()
        host = self.host(pair)
        seeds = np.array([[40, 40]], dtype=np.float32)
        host.state.features = seeds
        with patch("app.gui.main_window.PairAlignmentDialog") as cls:
            cls.return_value.exec.return_value = QDialog.DialogCode.Accepted
            cls.return_value.alignment = PairAlignment(translation=(2, 3))

            def change_revision(*args):
                host.state.touch()
                return QMessageBox.StandardButton.Yes

            with patch.object(QMessageBox, "question", side_effect=change_revision):
                host._adjust_pair_alignment()
            self.assertIs(host.state.sequence, pair)
            self.assertIs(host.state.features, seeds)

            def change_file(*args):
                stat = os.stat(pair.paths[0])
                os.utime(pair.paths[0], ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
                return QMessageBox.StandardButton.Yes

            with patch.object(QMessageBox, "question", side_effect=change_file), patch.object(
                QMessageBox, "critical"
            ) as error:
                host._adjust_pair_alignment()
                self.assertEqual(error.call_count, 1)
            self.assertIs(host.state.sequence, pair)
            self.assertIs(host.state.features, seeds)

    def test_file_menu_open_align_save_and_reopen_without_alignment(self):
        pair = self.pair()
        host = self.host(pair)
        with patch("app.gui.main_window.ImagePairDialog") as chooser, patch(
            "app.gui.main_window.PairAlignmentDialog"
        ) as align:
            chooser.return_value.exec.return_value = QDialog.DialogCode.Accepted
            chooser.return_value.paths = pair.paths
            align.return_value.exec.return_value = QDialog.DialogCode.Rejected
            host._open_image_pair()
            self.assertIs(host.state.sequence, pair)
            align.return_value.exec.return_value = QDialog.DialogCode.Accepted
            align.return_value.alignment = PairAlignment(translation=(-6, 8))
            host._open_image_pair()
            self.assertEqual(host.state.sequence.translation, (-6, 8))
            self.assertEqual((host.state.reference_index, host.state.last_index, host.state.current_index), (0, 1, 0))
        path = str(self.root / "chosen.ecmpair.json")
        with patch("app.gui.main_window.QFileDialog.getSaveFileName", return_value=(path, "")):
            host._save_pair_setup()  # Saving works before defining any points.
        self.assertEqual(host.state.pair_setup_path, path)
        with patch("app.gui.main_window.PairAlignmentDialog") as align:
            align.return_value.exec.return_value = QDialog.DialogCode.Accepted
            align.return_value.alignment = PairAlignment(translation=(-7, 8))
            host._adjust_pair_alignment()
        self.assertEqual(host.state.pair_setup_path, path)
        with patch("app.gui.main_window.QFileDialog.getOpenFileName", return_value=(path, "")), patch(
            "app.gui.main_window.PairAlignmentDialog"
        ) as align:
            host._open_pair_setup()
            align.assert_not_called()
        self.assertEqual(host.state.sequence.translation, (-6, 8))
        self.assertEqual(host.state.pair_setup_path, path)

    def test_validation_cancellation_and_source_change_during_callback(self):
        pair = self.pair()
        host = self.host(pair)
        candidate = pair.with_translation(2, 3)
        with patch.object(candidate, "validate_all", return_value=False):
            self.assertFalse(host._load_sequence_candidate(candidate, str(self.root)))
        self.assertIs(host.state.sequence, pair)
        self.assertFalse(candidate.validate_all(lambda *_: True))
        self.assertIsNone(candidate.fingerprint)

        def change_after_last_frame(done, total):
            if done == total:
                stat = os.stat(pair.paths[0])
                os.utime(pair.paths[0], ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            return False

        with self.assertRaises(OSError):
            candidate.validate_all(change_after_last_frame)

    def test_save_reload_trackers_plugins_export_and_normal_load(self):
        pair = self.pair((3, -4))
        host = self.host(pair)
        self.assertTrue(host.save_pair_setup_action.isEnabled())
        self.assertTrue(host.adjust_pair_action.isEnabled())
        host.state.features = np.array([[30, 30], [50, 50]], dtype=np.float32)
        host.state.result = track(pair, 0, 1, host.state.features, DEFAULT_LK)
        host.state.active_mask = np.ones(2, dtype=bool)
        host.state.touch()
        tracker_path = str(self.root / "trackers.npz")
        setup_path = save_pair_setup(self.root / "setup", pair)
        host.save_trackers_to(tracker_path)
        restored = load_pair_setup(setup_path)
        self.assertTrue(host._load_sequence_candidate(restored, str(self.root), setup_path=setup_path))
        host.load_trackers_from(tracker_path)
        ctx = PluginContext(host)
        self.addCleanup(ctx.dispose)
        self.assertEqual(ctx.frame_paths, tuple(pair.paths))
        self.assertEqual(ctx.image_size(), (96, 137))
        np.testing.assert_array_equal(ctx.frame_bgr(1), restored.load_bgr(1))
        np.testing.assert_array_equal(ctx.tracks().coords, host.state.result.coords_fw)
        with patch("app.gui.main_window.QFileDialog.getSaveFileName", return_value=(str(self.root / "coords.npy"), "")):
            host._export()
        np.testing.assert_array_equal(np.load(self.root / "coords.npy"), ctx.tracks().coords)
        changed = restored.with_translation(4, -4)
        host._load_sequence_candidate(changed, str(self.root))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            host.load_trackers_from(tracker_path)
        self.assertTrue(host._load_paths(pair.paths, str(self.root)))
        self.assertIsNone(host.state.pair_setup_path)
        self.assertFalse(host.adjust_pair_action.isEnabled())
        self.assertFalse(host.save_pair_setup_action.isEnabled())

    def test_bad_pair_loading_keeps_session_and_selection_preserves_roles(self):
        pair = self.pair()
        host = self.host(pair)
        chooser = ImagePairDialog(str(self.root))
        self.addCleanup(chooser.deleteLater)
        for field, path in zip(chooser.path_fields, pair.paths):
            field.setText(path)
        self.assertEqual(chooser.paths, pair.paths)
        self.assertTrue(chooser.open_button.isEnabled())
        candidate = AlignedImagePairSequence([pair.paths[0], str(self.root / "missing.png")])
        with patch.object(QMessageBox, "critical"):
            self.assertFalse(host._load_sequence_candidate(candidate, str(self.root)))
        self.assertIs(host.state.sequence, pair)
        with patch.object(QMessageBox, "critical"), patch(
            "app.gui.main_window.QFileDialog.getOpenFileName", return_value=(str(self.root / "missing.ecmpair.json"), "")
        ):
            host._open_pair_setup()
        self.assertIs(host.state.sequence, pair)


if __name__ == "__main__":
    unittest.main()
