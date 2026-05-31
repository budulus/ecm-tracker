"""Application-wide light theme.

A single QSS string applied on the ``QApplication`` so it covers the main window
*and* every dialog. Pair with ``app.setStyle("Fusion")`` first for consistent,
predictable rendering of the styled controls across macOS / Windows / Linux.

Palette
    window   #f4f5f7   surface  #ffffff   border  #d6dade
    text     #2b2f33   muted    #8a9099   accent  #2563eb
The icon glyph colors live in ``icon_loader`` (NORMAL/ACCENT/DISABLED) and are
kept visually in sync with this sheet.
"""

from PyQt5.QtWidgets import QApplication

from app.gui.icon_loader import _ICON_DIR as _ICON_DIR_RAW

# The bundled-icons dir as a QSS url() path: forward slashes so url(...) is valid on Windows
# too. Single source of truth is icon_loader._ICON_DIR.
_ICON_DIR = _ICON_DIR_RAW.replace("\\", "/")

LIGHT_QSS = """
* { font-size: 13px; }

QMainWindow, QDialog, QWidget { background: #f4f5f7; color: #2b2f33; }

/* ---- Toolbar ---------------------------------------------------------- */
QToolBar {
    background: #ffffff;
    border: none;
    border-bottom: 1px solid #d6dade;
    padding: 6px 8px;
    spacing: 4px;
}
QToolBar::separator {
    background: #e3e7ec;
    width: 1px;
    margin: 6px 10px;
}

QToolButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 6px 8px 4px 8px;
    color: #3a3f44;
    font-size: 11px;
}
QToolButton:hover { background: #eef1f6; }
QToolButton:pressed { background: #e3e8f0; }
QToolButton:checked {
    background: #e7efff;
    border: 1px solid #b9d2ff;
    color: #1d4ed8;
}
QToolButton:disabled { color: #b3b9c0; }

/* Dropdown menu buttons: reuse the spinbox chevron as the menu indicator */
QToolButton#menuButton::menu-indicator {
    image: url(@ICON_DIR@/chevron-down.svg);
    subcontrol-origin: padding;
    subcontrol-position: bottom right;
    width: 10px;
    height: 10px;
    right: 4px;
    bottom: 4px;
}

/* Accent primary action (Run Tracking) */
QToolButton#primaryAction {
    background: #2563eb;
    color: #ffffff;
    border: 1px solid #2563eb;
    font-weight: 600;
}
QToolButton#primaryAction:hover { background: #1d4ed8; border-color: #1d4ed8; }
QToolButton#primaryAction:pressed { background: #1a45c0; }
QToolButton#primaryAction:disabled {
    background: #e9ebef;
    color: #b3b9c0;
    border-color: #e9ebef;
}

/* Group caption under each toolbar section */
QLabel#toolGroupCaption {
    color: #9aa1aa;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1px;
    padding-top: 2px;
}

/* Sliders are intentionally left unstyled (native/Fusion default look). */

/* ---- Spin boxes / line edits ----------------------------------------- */
QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #ffffff;
    border: 1px solid #d6dade;
    border-radius: 6px;
    padding: 3px 6px;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}
QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus { border-color: #2563eb; }

/* Spin button sub-controls: once the box has a custom border, Qt stops
   drawing the native buttons/arrows, so they must be styled explicitly. The
   left divider + matching corner radii keep them inside the rounded border. */
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    background: #f4f5f7;
    border-left: 1px solid #d6dade;
    border-top-right-radius: 6px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    background: #f4f5f7;
    border-left: 1px solid #d6dade;
    border-bottom-right-radius: 6px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: #e9edf2; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: url(@ICON_DIR@/chevron-up.svg);
    width: 9px;
    height: 9px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: url(@ICON_DIR@/chevron-down.svg);
    width: 9px;
    height: 9px;
}

/* ---- Buttons (dialogs) ------------------------------------------------ */
QPushButton {
    background: #ffffff;
    border: 1px solid #cdd2d8;
    border-radius: 6px;
    padding: 5px 14px;
    color: #2b2f33;
}
QPushButton:hover { background: #f0f2f5; }
QPushButton:pressed { background: #e6e9ee; }
QPushButton:default {
    background: #2563eb;
    border-color: #2563eb;
    color: #ffffff;
    font-weight: 600;
}
QPushButton:default:hover { background: #1d4ed8; }
QPushButton:disabled { background: #f0f1f3; color: #b3b9c0; border-color: #e3e6ea; }

/* Plugin launch buttons (side-pane card): read as a left-aligned list. */
QPushButton#pluginButton { text-align: left; padding: 7px 12px; }

/* ---- Group boxes (cleanup dialog) ------------------------------------ */
QGroupBox {
    background: #ffffff;
    border: 1px solid #d6dade;
    border-radius: 8px;
    margin-top: 12px;
    padding: 10px 8px 8px 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: #6b727a;
    font-weight: 600;
}

/* ---- Splitter handle -------------------------------------------------- */
QSplitter::handle:horizontal { width: 6px; background: #f4f5f7; }
QSplitter::handle:horizontal:hover { background: #e3e7ec; }

/* ---- Checkboxes ------------------------------------------------------- */
QCheckBox { spacing: 6px; }
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #c4c9cf;
    border-radius: 4px;
    background: #ffffff;
}
QCheckBox::indicator:checked { background: #2563eb; border-color: #2563eb; }
QCheckBox::indicator:disabled { background: #f0f1f3; border-color: #e3e6ea; }

/* ---- Menus & status bar ---------------------------------------------- */
QMenuBar { background: #ffffff; border-bottom: 1px solid #d6dade; }
QMenuBar::item { padding: 4px 10px; background: transparent; }
QMenuBar::item:selected { background: #eef1f6; border-radius: 4px; }
QMenu { background: #ffffff; border: 1px solid #d6dade; padding: 4px; }
QMenu::item { padding: 5px 22px; border-radius: 4px; }
QMenu::item:selected { background: #eef1f6; }
QMenu::separator { height: 1px; background: #e3e7ec; margin: 4px 8px; }

QStatusBar { background: #ffffff; border-top: 1px solid #d6dade; color: #6b727a; }
QStatusBar QLabel { color: #6b727a; }
"""


def apply_theme(app: QApplication) -> None:
    """Apply the Fusion base style + light stylesheet to the whole application."""
    app.setStyle("Fusion")
    app.setStyleSheet(LIGHT_QSS.replace("@ICON_DIR@", _ICON_DIR))
