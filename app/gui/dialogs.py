from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
)

from app.core import settings


def _add_buttons(dialog, form, section_getter, section_name):
    """Add OK / Cancel plus a 'Save as defaults' action that persists the dialog's values
    with a brief non-blocking confirmation on the button itself."""
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    save_btn = buttons.addButton("Save as defaults", QDialogButtonBox.ActionRole)

    def _save():
        settings.update_section(section_name, section_getter())
        save_btn.setText("Saved ✓")
        QTimer.singleShot(1500, lambda: save_btn.setText("Save as defaults"))

    save_btn.clicked.connect(_save)
    form.addRow(buttons)


class CornerDetectionDialog(QDialog):
    """Edit Shi-Tomasi / Harris parameters for cv2.goodFeaturesToTrack."""

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Corner Detection")

        self.max_corners = QSpinBox()
        self.max_corners.setRange(1, 1_000_000)
        self.max_corners.setValue(int(params["maxCorners"]))

        self.quality_level = QDoubleSpinBox()
        self.quality_level.setDecimals(4)
        self.quality_level.setRange(0.0001, 1.0)
        self.quality_level.setSingleStep(0.001)
        self.quality_level.setValue(float(params["qualityLevel"]))

        self.min_distance = QDoubleSpinBox()
        self.min_distance.setRange(0.0, 1000.0)
        self.min_distance.setSingleStep(1.0)
        self.min_distance.setValue(float(params["minDistance"]))

        self.block_size = QSpinBox()
        self.block_size.setRange(1, 99)
        self.block_size.setValue(int(params["blockSize"]))

        self.use_harris = QCheckBox()
        self.use_harris.setChecked(bool(params["useHarrisDetector"]))

        self.k = QDoubleSpinBox()
        self.k.setDecimals(4)
        self.k.setRange(0.0, 1.0)
        self.k.setSingleStep(0.01)
        self.k.setValue(float(params["k"]))

        form = QFormLayout(self)
        form.addRow("maxCorners", self.max_corners)
        form.addRow("qualityLevel", self.quality_level)
        form.addRow("minDistance", self.min_distance)
        form.addRow("blockSize", self.block_size)
        form.addRow("useHarrisDetector", self.use_harris)
        form.addRow("k (Harris)", self.k)

        _add_buttons(self, form, self.values, "shi_tomasi")

    def values(self) -> dict:
        return dict(
            maxCorners=self.max_corners.value(),
            qualityLevel=self.quality_level.value(),
            minDistance=self.min_distance.value(),
            blockSize=self.block_size.value(),
            useHarrisDetector=self.use_harris.isChecked(),
            k=self.k.value(),
        )


class TrackerDialog(QDialog):
    """Edit pyramidal Lucas-Kanade parameters for cv2.calcOpticalFlowPyrLK."""

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tracker")

        self.win_size = QSpinBox()
        self.win_size.setRange(3, 201)
        self.win_size.setSingleStep(2)
        self.win_size.setValue(int(params["win_size"]))

        self.max_level = QSpinBox()
        self.max_level.setRange(0, 10)
        self.max_level.setValue(int(params["max_level"]))

        self.max_iter = QSpinBox()
        self.max_iter.setRange(1, 1000)
        self.max_iter.setValue(int(params["max_iter"]))

        self.epsilon = QDoubleSpinBox()
        self.epsilon.setDecimals(4)
        self.epsilon.setRange(0.0001, 10.0)
        self.epsilon.setSingleStep(0.001)
        self.epsilon.setValue(float(params["epsilon"]))

        self.flags = QSpinBox()
        self.flags.setRange(0, 3)
        self.flags.setValue(int(params["flags"]))

        self.min_eig = QDoubleSpinBox()
        self.min_eig.setDecimals(6)
        self.min_eig.setRange(0.0, 1.0)
        self.min_eig.setSingleStep(0.0001)
        self.min_eig.setValue(float(params["min_eig_threshold"]))

        form = QFormLayout(self)
        form.addRow("winSize (square)", self.win_size)
        form.addRow("maxLevel", self.max_level)
        form.addRow("criteria max iter", self.max_iter)
        form.addRow("criteria epsilon", self.epsilon)
        form.addRow("flags", self.flags)
        form.addRow("minEigThreshold", self.min_eig)

        _add_buttons(self, form, self.values, "lk")

    def values(self) -> dict:
        return dict(
            win_size=self.win_size.value(),
            max_level=self.max_level.value(),
            max_iter=self.max_iter.value(),
            epsilon=self.epsilon.value(),
            flags=self.flags.value(),
            min_eig_threshold=self.min_eig.value(),
        )


class GridDialog(QDialog):
    """Edit regular-grid spacing."""

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Regular Grid")

        self.spacing_x = QSpinBox()
        self.spacing_x.setRange(1, 1000)
        self.spacing_x.setValue(int(params["spacing_x"]))

        self.spacing_y = QSpinBox()
        self.spacing_y.setRange(1, 1000)
        self.spacing_y.setValue(int(params["spacing_y"]))

        form = QFormLayout(self)
        form.addRow("spacing_x", self.spacing_x)
        form.addRow("spacing_y", self.spacing_y)

        _add_buttons(self, form, self.values, "grid")

    def values(self) -> dict:
        return dict(spacing_x=self.spacing_x.value(), spacing_y=self.spacing_y.value())
