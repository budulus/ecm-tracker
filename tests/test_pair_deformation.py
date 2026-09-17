"""Scientific composition, resampling, persistence, and pair deformation UI regressions."""
import csv
from dataclasses import replace
import hashlib
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

from app.core.affine import fit_affine, principal_stretches
from app.core.image_pair import AlignedImagePairSequence, load_pair_setup, save_pair_setup
from app.core.pair_transform import PairAlignment, alignment_crop, compose_alignment_affines, largest_valid_rectangle
from app.core.result import TrackerResult
from app.core.tracking import DEFAULT_LK, track
from app.gui.main_window import MainWindow
from app.gui.pair_alignment import PairAlignmentDialog
from app.plugins.api import PluginContext


class PairDeformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ecm-deformation-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        env = patch.dict(os.environ, {"TRACKER_CONFIG_DIR": str(self.root / "config")})
        env.start()
        self.addCleanup(env.stop)

    def pair(self, alignment=PairAlignment(), reference=None, destination=None):
        reference = np.full((100, 140, 3), 180, np.uint8) if reference is None else reference
        destination = reference.copy() if destination is None else destination
        paths = [str(self.root / "ref.png"), str(self.root / "dst.png")]
        for path, image in zip(paths, (reference, destination)):
            self.assertTrue(cv2.imwrite(path, image))
        pair = AlignedImagePairSequence(paths, alignment=alignment)
        pair.validate_all()
        return pair

    def host(self, pair):
        host = MainWindow()
        self.addCleanup(host.deleteLater)
        self.addCleanup(host.close)
        self.assertTrue(host._load_sequence_candidate(pair, str(self.root)))
        return host

    def dialog(self, pair):
        dialog = PairAlignmentDialog(pair)
        self.addCleanup(dialog.deleteLater)
        self.addCleanup(dialog.close)
        dialog.show()
        self.app.processEvents()
        return dialog

    def test_noncommuting_composition_all_modes_and_rotations(self):
        F = np.array([[1.23, .21], [.09, .88]])
        for mode in ("translation", "scale", "affine"):
            for enabled in (False, True):
                with self.subTest(mode=mode, rotation=enabled):
                    a = PairAlignment(mode=mode, scale=.8, affine_stretch=((.82, .11), (.11, 1.13)),
                                      rotation_enabled=enabled, angle_degrees=27, translation=(13, -9))
                    residual = a.linear @ F
                    corrected = compose_alignment_affines(residual, a.correction, np.eye(2))
                    np.testing.assert_allclose(corrected, a.rotation @ F, atol=1e-12)
                    np.testing.assert_allclose(principal_stretches(corrected)[:2], principal_stretches(F)[:2])
                    np.testing.assert_allclose(compose_alignment_affines(np.eye(2), a.correction, a.correction), np.eye(2), atol=1e-12)
        self.assertFalse(np.allclose(residual @ a.correction, corrected))
        np.testing.assert_array_equal(PairAlignment(rotation_enabled=True, angle_degrees=37).correction, np.eye(2))
        np.testing.assert_allclose(PairAlignment(mode="scale", scale=.8).correction, np.eye(2)*1.25)

    def test_affine_corners_anchor_and_retain_rotation(self):
        shape = (100, 140, 3)
        points = np.array([[-.5, -.5], [139.5, -.5], [139.5, 99.5], [-.5, 99.5]])
        for mode in ("scale", "affine"):
            a = PairAlignment(mode=mode, rotation_enabled=True, angle_degrees=23)
            initial = points @ a.linear.T + a.matrix(shape)[:, 2]
            for corner in range(4):
                opposite = (corner+2) % 4
                target = initial[corner] + (.08 * (initial[corner]-initial[opposite]) if mode == "scale" else [7, -3])
                b = a.drag_corner(shape, corner, target)
                final = points @ b.linear.T + b.matrix(shape)[:, 2]
                np.testing.assert_allclose(final[opposite], initial[opposite], atol=1e-12)
                np.testing.assert_allclose(final[corner], target, atol=1e-12)
                np.testing.assert_allclose(final[0]+final[2], final[1]+final[3], atol=1e-12)
                self.assertEqual(b.angle_degrees, a.angle_degrees)
                self.assertGreater(np.linalg.eigvalsh(b.stretch).min(), 0)

    def test_invalid_deformation_and_metadata_rejected(self):
        for kwargs in ({"scale": 0}, {"scale": -1}, {"scale": True}, {"scale": float("nan")},
                       {"translation": (True, 0)}, {"rotation_enabled": 1}, {"angle_degrees": float("inf")},
                       {"affine_stretch": ((1, 0), (0, -1))}, {"affine_stretch": ((1, 1), (0, 1))},
                       {"affine_stretch": ((1, 0), (0, 1e-10))}, {"mode": "perspective"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PairAlignment(**kwargs)
        with self.assertRaises(ValueError):
            PairAlignment.from_metadata({"scale": 1})
        with self.assertRaises(ValueError):
            PairAlignment(mode="scale").drag_corner((100, 140), 0, (200, 200))

    def test_pivot_stays_fixed_for_percentage_and_mode_changes(self):
        shape = (100, 140, 3)
        pivot = (31.25, 42.75)
        point = np.array([*pivot, 1])
        for mode in ("translation", "scale", "affine"):
            a = PairAlignment(mode=mode, scale=.83, affine_stretch=((.9, .1), (.1, 1.2)),
                              rotation_enabled=True, angle_degrees=27, translation=(7.3, -4.1), pivot=pivot)
            anchor = a.matrix(shape) @ point
            for percent in (60, 115, 90):
                b = a.with_scale_percent(percent, shape)
                np.testing.assert_allclose(b.matrix(shape) @ point, anchor, atol=1e-12)
                if mode == "affine":
                    np.testing.assert_allclose(b.stretch / a.stretch, percent / a.scale_percent)
                self.assertEqual(b.angle_degrees, a.angle_degrees)
            for next_mode in ("affine", "scale", "translation", mode):
                a = a.with_mode(next_mode, shape)
                np.testing.assert_allclose(a.matrix(shape) @ point, anchor, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "shape"):
            PairAlignment(mode="scale", pivot=pivot).with_scale_percent(80)
        a = PairAlignment(mode="scale", translation=(7.3, -4.1))
        self.assertEqual(a.with_scale_percent(80).translation, a.translation)

    def test_pivot_corner_drags_and_degenerate_handle(self):
        shape = (100, 140)
        corners = np.array([[-.5, -.5], [139.5, -.5], [139.5, 99.5], [-.5, 99.5]])
        for mode in ("scale", "affine"):
            a = PairAlignment(mode=mode, scale=.9, affine_stretch=((.9, .07), (.07, 1.1)),
                              rotation_enabled=True, angle_degrees=-21, translation=(3.2, -1.7),
                              pivot=(37.25, 62.75))
            point = np.array([*a.pivot, 1])
            anchor = a.matrix(shape) @ point
            initial = corners @ a.linear.T + a.matrix(shape)[:, 2]
            for corner in range(4):
                target = initial[corner] + (.08 * (initial[corner] - anchor) if mode == "scale" else [2, -1])
                b = a.drag_corner(shape, corner, target)
                final = corners @ b.linear.T + b.matrix(shape)[:, 2]
                np.testing.assert_allclose(b.matrix(shape) @ point, anchor, atol=1e-12)
                np.testing.assert_allclose(final[corner], target, atol=1e-12)
                np.testing.assert_allclose(final[0] + final[2], final[1] + final[3], atol=1e-12)
                self.assertEqual(b.angle_degrees, a.angle_degrees)
                self.assertGreater(np.linalg.eigvalsh(b.stretch).min(), 0)
                coincident = replace(a, pivot=tuple(corners[corner]))
                self.assertIs(coincident.drag_corner(shape, corner, target), coincident)
        a = PairAlignment(mode="scale", pivot=(20, 30))
        with self.assertRaises(ValueError):
            a.drag_corner(shape, 0, a.pivot)

    def test_pivot_metadata_validation_and_legacy_fields(self):
        for pivot in (True, 2, "12", (), (1,), (1, 2, 3), (True, 0), (float("nan"), 0),
                      (0, float("inf")), ("2", 3)):
            with self.subTest(pivot=pivot), self.assertRaises(ValueError):
                PairAlignment(pivot=pivot)
        a = PairAlignment(pivot=[12.5, 34.25])
        self.assertEqual(a.pivot, (12.5, 34.25))
        self.assertEqual(PairAlignment.from_metadata(a.metadata()), a)
        legacy = a.metadata()
        del legacy["pivot"]
        self.assertEqual(PairAlignment.from_metadata(legacy), PairAlignment())
        for missing in legacy:
            invalid = dict(legacy)
            del invalid[missing]
            with self.assertRaises(ValueError):
                PairAlignment.from_metadata(invalid)
        with self.assertRaises(ValueError):
            PairAlignment.from_metadata(dict(legacy, unexpected=1))

    def test_pivot_setup_roundtrip_and_metadata_only_identity(self):
        a = PairAlignment(mode="affine", affine_stretch=((.9, .05), (.05, 1.1)),
                          rotation_enabled=True, angle_degrees=13, translation=(4, -2))
        pair = self.pair(a)
        edited = pair.with_alignment(replace(a, pivot=(25.25, 40.75)))
        edited.validate_all()
        self.assertEqual(edited.fingerprint, pair.fingerprint)
        self.assertEqual(edited.crop, pair.crop)
        np.testing.assert_array_equal(edited.load_bgr(1), pair.load_bgr(1))
        path = Path(save_pair_setup(self.root / "pivot", edited))
        restored = load_pair_setup(path)
        self.assertEqual(restored.alignment, edited.alignment)
        self.assertEqual(self.dialog(restored).alignment.pivot, edited.alignment.pivot)
        metadata = json.loads(path.read_text())
        del metadata["alignment"]["pivot"]
        path.write_text(json.dumps(metadata))
        legacy = load_pair_setup(path)
        self.assertEqual(legacy.alignment, a)
        self.assertEqual(legacy.fingerprint, edited.fingerprint)

    def test_pivot_only_apply_and_cancel_preserve_tracking(self):
        pair = self.pair()
        host = self.host(pair)
        host.state.features = np.array([[30, 30]], np.float32)
        result = host.state.result = track(pair, 0, 1, host.state.features, DEFAULT_LK)
        revision = host.state.revision
        for pivot in ((30.5, 40.25), (55, 60), None):
            with patch("app.gui.main_window.PairAlignmentDialog") as cls, patch.object(QMessageBox, "question") as question:
                cls.return_value.exec.return_value = QDialog.DialogCode.Accepted
                cls.return_value.alignment = replace(pair.alignment, pivot=pivot)
                host._adjust_pair_alignment()
                question.assert_not_called()
            self.assertEqual(host.state.sequence.alignment.pivot, pivot)
            self.assertIs(host.state.result, result)
            self.assertEqual(host.state.revision, revision)
            self.assertEqual(host.state.sequence.fingerprint, pair.fingerprint)
        with patch("app.gui.main_window.PairAlignmentDialog") as cls:
            cls.return_value.exec.return_value = QDialog.DialogCode.Rejected
            cls.return_value.alignment = replace(pair.alignment, pivot=(11, 22))
            host._adjust_pair_alignment()
        self.assertIsNone(host.state.sequence.alignment.pivot)
        self.assertIs(host.state.result, result)

    def test_pivot_picking_navigation_cancel_clear_and_reset(self):
        a = PairAlignment(mode="affine", affine_stretch=((.85, .05), (.05, 1.1)),
                          rotation_enabled=True, angle_degrees=17, translation=(4.2, -3.1))
        dialog = self.dialog(self.pair(a))
        canvas = dialog.canvas
        for zoom in (.5, 1.8):
            canvas.fit_view()
            canvas.zoom_at(zoom)
            QTest.mouseClick(dialog.set_pivot_button, Qt.MouseButton.LeftButton)
            self.assertTrue(canvas.picking_pivot)
            self.assertEqual(canvas.cursor().shape(), Qt.CursorShape.CrossCursor)
            # Both pan gestures remain navigation even while a pick is armed.
            for button in (Qt.MouseButton.RightButton, Qt.MouseButton.LeftButton):
                if button == Qt.MouseButton.LeftButton:
                    QTest.keyPress(canvas, Qt.Key.Key_Space)
                start = canvas.rect().center()
                end = start + QPointF(13, -9).toPoint()
                QTest.mousePress(canvas, button, pos=start)
                QTest.mouseMove(canvas, end)
                QTest.mouseRelease(canvas, button, pos=end)
                if button == Qt.MouseButton.LeftButton:
                    QTest.keyRelease(canvas, Qt.Key.Key_Space)
                self.assertTrue(canvas.picking_pivot)
                self.assertTrue(canvas.alignment.same_geometry(a))
                self.assertEqual(canvas.cursor().shape(), Qt.CursorShape.CrossCursor)
            target = a.matrix((100, 140)) @ np.array([45.25, 35.75, 1])
            click = canvas.image_to_screen(*target).toPoint()
            QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=click)
            QTest.mouseMove(canvas, click + QPointF(15, 8).toPoint())
            QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=click)
            self.assertFalse(canvas.picking_pivot)
            self.assertFalse(dialog.set_pivot_button.isChecked())
            self.assertTrue(canvas.alignment.same_geometry(a))
            np.testing.assert_allclose(canvas.alignment.pivot, (45.25, 35.75), atol=.5 / zoom)
            self.assertFalse(canvas.crop_pending)
            previous = canvas.alignment
            QTest.mouseClick(dialog.set_pivot_button, Qt.MouseButton.LeftButton)
            outside = a.matrix((100, 140)) @ np.array([-10, 40, 1])
            QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=canvas.image_to_screen(*outside).toPoint())
            self.assertTrue(canvas.picking_pivot)
            self.assertEqual(canvas.alignment, previous)
            QTest.keyClick(canvas, Qt.Key.Key_Escape)
            self.assertFalse(canvas.picking_pivot)
            self.assertTrue(dialog.isVisible())
            self.assertEqual(canvas.alignment, previous)
            QTest.mouseClick(dialog.clear_pivot_button, Qt.MouseButton.LeftButton)
            self.assertEqual(canvas.alignment, a)
            self.assertFalse(dialog.clear_pivot_button.isEnabled())
        canvas.set_alignment(replace(a, pivot=(40, 30)))
        canvas.set_pivot_picking(True)
        canvas.reset_alignment()
        self.assertIsNone(canvas.alignment.pivot)
        self.assertFalse(canvas.picking_pivot)
        self.assertFalse(dialog.set_pivot_button.isChecked())

    def test_dialog_pivot_scale_modes_and_feature_motion(self):
        a = PairAlignment(mode="scale", scale=.9, affine_stretch=((.85, .08), (.08, 1.1)),
                          rotation_enabled=True, angle_degrees=17, translation=(4, -2), pivot=(35.25, 40.75))
        dialog = self.dialog(self.pair(a))
        canvas = dialog.canvas
        anchor = canvas.pivot_position().copy()
        dialog.scale_percent.setValue(105)
        np.testing.assert_allclose(canvas.pivot_position(), anchor, atol=1e-12)
        for mode in ("affine", "translation", "scale"):
            dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(mode))
            np.testing.assert_allclose(canvas.pivot_position(), anchor, atol=1e-12)
        QTest.keyClick(canvas, Qt.Key.Key_Right)
        np.testing.assert_allclose(canvas.pivot_position(), anchor + [1, 0], atol=1e-12)
        before_rotation = canvas.alignment
        dialog.angle_degrees.setValue(38)
        self.assertEqual(canvas.translation, before_rotation.translation)
        expected = replace(before_rotation, angle_degrees=38).matrix((100, 140)) @ np.array([*a.pivot, 1])
        np.testing.assert_allclose(canvas.pivot_position(), expected, atol=1e-12)
        dialog.scale_percent.setValue(85)
        np.testing.assert_allclose(canvas.pivot_position(), expected, atol=1e-12)
        canvas.finish_geometry()
        applied = self.pair().with_alignment(canvas.alignment)
        applied.validate_all()
        self.assertEqual(canvas.overlap, applied.crop)

    def test_pivot_handle_drags_at_multiple_zooms(self):
        dialog = self.dialog(self.pair())
        canvas = dialog.canvas
        for mode in ("scale", "affine"):
            for zoom in (.5, 1.5):
                for corner in range(4):
                    a = PairAlignment(mode=mode, rotation_enabled=True, angle_degrees=11,
                                      translation=(3, -2), pivot=(45.25, 35.75))
                    canvas.set_alignment(a)
                    canvas.fit_view()
                    canvas.zoom_at(zoom)
                    anchor = canvas.pivot_position().copy()
                    initial = canvas.destination_corners()[corner]
                    target = initial + (.08 * (initial - anchor) if mode == "scale" else [3, -2])
                    start, end = (canvas.image_to_screen(*p).toPoint() for p in (initial, target))
                    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
                    QTest.mouseMove(canvas, end)
                    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=end)
                    np.testing.assert_allclose(canvas.pivot_position(), anchor, atol=1e-12)
                    np.testing.assert_allclose(canvas.destination_corners()[corner], target, atol=.6 / zoom)
                    self.assertFalse(canvas.crop_pending)

    def test_crop_is_maximal_and_ties_deterministic(self):
        # Compare the optimized interval scan against exhaustive rectangle enumeration.
        for shape in ((7, 10), (10, 7)):
            for angle in (0, 17, 43, -31):
                a = PairAlignment(mode="affine", affine_stretch=((.9, .1), (.1, .85)),
                                  rotation_enabled=True, angle_degrees=angle, translation=(.4, -.3))
                mask = cv2.warpAffine(np.ones(shape, np.float32), a.matrix(shape), shape[::-1]) == 1
                best, key = (0, 0, 0, 0), (0, 0, 0, 0)
                for y in range(shape[0]):
                    for x in range(shape[1]):
                        for h in range(1, shape[0]-y+1):
                            for w in range(1, shape[1]-x+1):
                                candidate_key = (w*h, -y, -x, w)
                                if candidate_key > key and mask[y:y+h, x:x+w].all():
                                    best, key = (x, y, w, h), candidate_key
                self.assertEqual(largest_valid_rectangle(mask), best)
                self.assertEqual(alignment_crop(shape, shape, a), best)

    def test_no_border_fill_and_single_original_resampling(self):
        a = PairAlignment(mode="affine", affine_stretch=((.87, .09), (.09, 1.12)),
                          rotation_enabled=True, angle_degrees=19, translation=(2.3, -1.7))
        pair = self.pair(a)
        self.assertTrue((pair.load_bgr(0) == 180).all())
        self.assertTrue((pair.load_bgr(1) == 180).all())
        b = replace(a, angle_degrees=-13, translation=(0, 0))
        edited = pair.with_alignment(b)
        edited.validate_all()
        direct = AlignedImagePairSequence(pair.paths, alignment=b)
        direct.validate_all()
        np.testing.assert_array_equal(edited.load_bgr(1), direct.load_bgr(1))
        self.assertEqual(edited.fingerprint, direct.fingerprint)
        self.assertFalse(edited.load_bgr(1).flags.writeable)
        with self.assertRaisesRegex(ValueError, "do not overlap"):
            pair.with_alignment(replace(a, translation=(10000, 10000))).validate_all()

    def test_warp_matches_preview_landmarks_and_overlap(self):
        image = np.zeros((100, 140, 3), np.uint8)
        cv2.circle(image, (45, 35), 5, (0, 0, 255), -1)
        a = PairAlignment(mode="affine", affine_stretch=((.87, .08), (.08, 1.1)),
                          rotation_enabled=True, angle_degrees=19, translation=(2, -1), pivot=(80.25, 60.75))
        a = a.with_scale_percent(95, image.shape)
        pair = self.pair(a, image, image)
        dialog = self.dialog(pair)
        dialog.opacity_slider.setValue(100)
        self.assertEqual(dialog.canvas.overlap, pair.crop)
        target = a.matrix(image.shape) @ np.array([45, 35, 1])
        point = dialog.canvas.image_to_screen(*target).toPoint()
        color = dialog.canvas.grab().toImage().pixelColor(point)
        self.assertGreater(color.red(), 240)
        self.assertLess(color.green(), 10)
        x, y, w, h = pair.crop
        full = cv2.warpAffine(image, a.matrix(image.shape), (140, 100))
        np.testing.assert_array_equal(pair.load_bgr(1), full[y:y+h, x:x+w])

    def test_transformed_tracker_roundtrip_and_identity_fallback(self):
        a = PairAlignment(mode="scale", scale=.9, rotation_enabled=True, angle_degrees=12)
        pair = self.pair(a)
        host = self.host(pair)
        host.state.features = np.array([[20, 20], [40, 40], [50, 20]], np.float32)
        host.state.result = track(pair, 0, 1, host.state.features, DEFAULT_LK)
        host.state.active_mask = np.ones(3, bool)
        setup = save_pair_setup(self.root / "tracked", pair)
        trackers = str(self.root / "trackers.npz")
        host.save_trackers_to(trackers)
        host._load_sequence_candidate(load_pair_setup(setup), str(self.root))
        host.load_trackers_from(trackers)
        self.assertIsNotNone(host.state.result)
        changed = pair.with_alignment(replace(a, scale=.91))
        changed.validate_all()
        host._load_sequence_candidate(changed, str(self.root))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            host.load_trackers_from(trackers)
        host._load_paths(pair.paths, str(self.root))
        ctx = PluginContext(host, "identity-check")
        self.addCleanup(ctx.dispose)
        for index in (0, 1):
            np.testing.assert_array_equal(ctx.alignment_affine(index), np.eye(2))

    def test_legacy_fingerprint_and_v1_setup(self):
        pair = self.pair(PairAlignment(translation=(-9, 4)))
        fingerprints = []
        for i in range(2):
            source = pair.source_bgr(i)
            digest = hashlib.sha256(str(source.shape).encode("ascii"))
            digest.update(memoryview(source))
            fingerprints.append(digest.hexdigest())
        digest = hashlib.sha256(b"ecmtracker-aligned-pair-v1")
        digest.update(json.dumps([tuple(fingerprints), (-9, 4), pair.crop]).encode("ascii"))
        for i in range(2):
            digest.update(memoryview(pair.load_bgr(i)))
        self.assertEqual(pair.fingerprint, digest.hexdigest())
        path = Path(save_pair_setup(self.root / "legacy", pair))
        metadata = json.loads(path.read_text())
        metadata["version"] = 1
        del metadata["alignment"]
        path.write_text(json.dumps(metadata))
        restored = load_pair_setup(path)
        self.assertEqual(restored.fingerprint, pair.fingerprint)
        np.testing.assert_array_equal(restored.alignment.correction, np.eye(2))

    def test_v2_roundtrip_presets_relocation_and_fingerprint(self):
        a = PairAlignment(mode="scale", scale=.83, affine_stretch=((.9, .1), (.1, 1.2)),
                          rotation_enabled=True, angle_degrees=-21, translation=(4.3, 7.1))
        pair = self.pair(a)
        path = Path(save_pair_setup(self.root / "setup", pair))
        restored = load_pair_setup(path)
        self.assertEqual(restored.alignment, a)
        self.assertEqual(restored.fingerprint, pair.fingerprint)
        np.testing.assert_array_equal(restored.load_bgr(1), pair.load_bgr(1))
        changed_preset = pair.with_alignment(replace(a, affine_stretch=((1, 0), (0, 1))))
        changed_preset.validate_all()
        self.assertEqual(changed_preset.fingerprint, pair.fingerprint)
        changed_geometry = pair.with_alignment(replace(a, scale=.84))
        changed_geometry.validate_all()
        self.assertNotEqual(changed_geometry.fingerprint, pair.fingerprint)
        moved = self.root / "moved"
        moved.mkdir()
        for source in (path, *map(Path, pair.paths)):
            shutil.copy(source, moved/source.name)
        self.assertEqual(load_pair_setup(moved/path.name).fingerprint, pair.fingerprint)
        metadata = json.loads(path.read_text())
        metadata["alignment"]["scale"] = .84
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            load_pair_setup(path)

    def test_dialog_presets_rotation_reset_and_invalid_overlap(self):
        dialog = self.dialog(self.pair())
        dialog.mode_combo.setCurrentIndex(1)
        dialog.scale_percent.setValue(80)
        dialog.rotation_check.setChecked(True)
        dialog.angle_degrees.setValue(23)
        dialog.mode_combo.setCurrentIndex(2)
        dialog.canvas.set_alignment(replace(dialog.alignment, affine_stretch=((.8, .1), (.1, 1.2))))
        affine = dialog.alignment.affine_stretch
        dialog.scale_percent.setValue(110)
        np.testing.assert_allclose(dialog.alignment.stretch / np.sqrt(np.linalg.det(dialog.alignment.stretch)),
                                   np.array(affine) / np.sqrt(np.linalg.det(affine)))
        affine = dialog.alignment.affine_stretch
        dialog.mode_combo.setCurrentIndex(0)
        np.testing.assert_array_equal(dialog.alignment.correction, np.eye(2))
        dialog.mode_combo.setCurrentIndex(1)
        self.assertEqual(dialog.scale_percent.value(), 80)
        dialog.mode_combo.setCurrentIndex(2)
        self.assertEqual(dialog.alignment.affine_stretch, affine)
        dialog.rotation_check.setChecked(False)
        self.assertEqual(dialog.alignment.angle_degrees, 23)
        np.testing.assert_array_equal(dialog.alignment.rotation, np.eye(2))
        dialog.rotation_check.setChecked(True)
        self.assertEqual(dialog.alignment.angle_degrees, 23)
        dialog.canvas.set_translation(10000, 0)
        dialog.canvas.finish_geometry()
        self.assertFalse(dialog.apply_button.isEnabled())
        dialog.canvas.reset_alignment()
        self.assertEqual(dialog.translation, (0, 0))
        self.assertEqual(dialog.alignment.scale, 1)
        self.assertEqual(dialog.alignment.angle_degrees, 0)
        np.testing.assert_array_equal(dialog.alignment.stretch, np.eye(2))
        self.assertTrue(dialog.apply_button.isEnabled())

    def test_handle_drag_at_multiple_zooms_and_rotation_handle(self):
        dialog = self.dialog(self.pair(PairAlignment(mode="affine")))
        canvas = dialog.canvas
        for zoom in (.5, 1, 2):
            canvas.set_alignment(PairAlignment(mode="affine"))
            canvas.fit_view()
            canvas.zoom_at(zoom)
            initial = canvas.destination_corners()
            start = canvas.image_to_screen(*initial[0]).toPoint()
            end = canvas.image_to_screen(*(initial[0] + [9, 4])).toPoint()
            QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
            QTest.mouseMove(canvas, end)
            QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=end)
            final = canvas.destination_corners()
            np.testing.assert_allclose(final[2], initial[2], atol=1e-9)
            np.testing.assert_allclose(final[0], initial[0]+[9, 4], atol=.6/zoom)
        canvas.set_alignment(PairAlignment(rotation_enabled=True))
        canvas.fit_view()
        center = canvas.image_to_screen(*canvas.destination_corners().mean(axis=0))
        start = canvas.rotation_handle()
        delta = start-center
        end = center + QPointF(-delta.y(), delta.x())
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start.toPoint())
        QTest.mouseMove(canvas, end.toPoint())
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=end.toPoint())
        self.assertAlmostEqual(canvas.alignment.angle_degrees, 90, delta=.5)
        np.testing.assert_array_equal(canvas.alignment.correction, np.eye(2))

    def test_metadata_only_edit_preserves_result_and_geometry_edit_resets(self):
        pair = self.pair()
        host = self.host(pair)
        host.state.features = np.array([[30, 30]], np.float32)
        result = host.state.result = track(pair, 0, 1, host.state.features, DEFAULT_LK)
        revision = host.state.revision
        a = PairAlignment(mode="affine", scale=.8, angle_degrees=12)
        with patch("app.gui.main_window.PairAlignmentDialog") as cls, patch.object(QMessageBox, "question") as question:
            cls.return_value.exec.return_value = QDialog.DialogCode.Accepted
            cls.return_value.alignment = a
            host._adjust_pair_alignment()
            question.assert_not_called()
        self.assertIs(host.state.result, result)
        self.assertEqual(host.state.revision, revision)
        self.assertEqual(host.state.sequence.alignment, a)
        self.assertEqual(host.state.sequence.fingerprint, pair.fingerprint)
        with patch("app.gui.main_window.PairAlignmentDialog") as cls, patch.object(
            QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
        ):
            cls.return_value.exec.return_value = QDialog.DialogCode.Accepted
            cls.return_value.alignment = replace(a, affine_stretch=((.9, .05), (.05, 1.1)))
            host._adjust_pair_alignment()
        self.assertIsNone(host.state.result)

    def test_tracking_restores_known_deformation_after_prealignment(self):
        rng = np.random.default_rng(123)
        reference = cv2.GaussianBlur(rng.integers(0, 256, (320, 420, 3), dtype=np.uint8), (5, 5), 1)
        F = np.array([[1.08, .09], [-.02, .95]])
        center = np.array([209.5, 159.5])
        destination = cv2.warpAffine(reference, np.column_stack((F, center-F@center)), (420, 320))
        u, singular, vt = np.linalg.svd(np.linalg.inv(F))
        R, U = u @ vt, vt.T @ np.diag(singular) @ vt
        # Deliberately leave small residual scale rather than testing only an identity fit.
        a = PairAlignment(mode="affine", affine_stretch=tuple(map(tuple, U*1.005)), rotation_enabled=True,
                          angle_degrees=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))))
        pair = self.pair(a, reference, destination)
        gray = pair.load_gray(0)
        mask = np.zeros_like(gray)
        mask[35:-35, 35:-35] = 255
        seeds = cv2.goodFeaturesToTrack(gray, 120, .03, 12, mask=mask).reshape(-1, 2)
        result = track(pair, 0, 1, seeds, DEFAULT_LK)
        valid = result.status_fw[1].astype(bool)
        self.assertGreater(valid.mean(), .95)
        fit = fit_affine(result.coords_fw[0], result.coords_fw[1], valid)
        corrected = compose_alignment_affines(fit[1], a.correction, np.eye(2))
        np.testing.assert_allclose(corrected, R @ F, atol=.002)
        np.testing.assert_allclose(principal_stretches(corrected)[:2], principal_stretches(F)[:2], atol=.002)

    def test_plugin_outputs_and_readonly_api_agree(self):
        from plugins.affine_zones.zones import AffineZonesWindow, Zone, default_zone_color

        a = PairAlignment(mode="affine", affine_stretch=((.85, .08), (.08, 1.1)),
                          rotation_enabled=True, angle_degrees=17)
        host = self.host(self.pair(a))
        ctx = PluginContext(host, "affine_zones")
        self.addCleanup(ctx.dispose)
        C = ctx.alignment_affine(1)
        self.assertFalse(C.flags.writeable)
        C.setflags(write=True)
        C[:] = 0
        np.testing.assert_allclose(ctx.alignment_affine(1), a.correction)
        np.testing.assert_array_equal(ctx.alignment_affine(0), np.eye(2))
        for index in (-1, 2, True, .5):
            with self.assertRaises(IndexError):
                ctx.alignment_affine(index)
        ref = np.array([[20, 20], [40, 20], [60, 20], [20, 40], [40, 40], [60, 40]], np.float32)
        residual = np.array([[1.02, .04], [.01, .97]])
        coords = np.stack((ref, ref @ residual.T + [1, -1]))
        status = np.ones((2, len(ref)), np.uint8)
        errors = np.zeros_like(status, np.float32)
        host.state.features = ref
        host.state.result = TrackerResult(0, 1, coords, status, errors, coords, status, errors,
                                          np.zeros(len(ref)), np.zeros(len(ref)))
        host.state.active_mask = np.ones(len(ref), bool)
        host.state.current_index = 1
        win = AffineZonesWindow(ctx)
        self.addCleanup(win.deleteLater)
        self.addCleanup(win.dispose)
        self.addCleanup(win.close)
        win.zones.append(Zone([(5, 5), (80, 5), (80, 60), (5, 60)], default_zone_color(0)))
        win._refresh()
        self.assertTrue(win.include_alignment.isChecked())
        win._open_plot()
        win._open_gauge()
        gauge = win._gauge_window
        gauge.slider.setValue(1)
        path = self.root / "stretches.csv"
        for include in (True, False):
            win.include_alignment.setChecked(include)
            expected = a.correction @ residual if include else residual
            np.testing.assert_allclose(win._get_fits()[0][1], expected, atol=1e-6)
            lam1, lam2, v1, v2 = principal_stretches(expected)
            self.assertAlmostEqual(float(win.table.item(0, 3).text()), lam1, places=4)
            plotted = win._plot_window.ax.lines[0].get_ydata()
            np.testing.assert_allclose(plotted, [1, lam1], atol=1e-6)
            self.assertEqual(len(gauge.canvas._entries), 1)
            entry = gauge.canvas._entries[0]
            np.testing.assert_allclose(entry[3:5], (lam1, lam2), atol=1e-6)
            expected_dir = np.linalg.inv(a.correction) @ v1 if include else v1
            np.testing.assert_allclose(entry[1], expected_dir, atol=1e-6)
            with patch("plugins.affine_zones.zones.QFileDialog.getSaveFileName", return_value=(str(path), "")):
                win._export()
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(float(rows[1]["lambda1"]), lam1, places=6)
            self.assertAlmostEqual(float(rows[1]["v1x"]), v1[0], delta=1e-6)
        self.assertFalse(ctx.get_settings()["include_alignment_affines"])
        # A different tracked reference uses its own correction, so F(reference)=I.
        host.state.reference_index = 1
        win.include_alignment.blockSignals(True)
        win.include_alignment.setChecked(True)
        win.include_alignment.blockSignals(False)
        fit = win.fit_zone(win.zones[0], ref, ref, 0)
        np.testing.assert_allclose(fit[1], np.eye(2), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
