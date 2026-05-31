//! The plugin control-panel declaration API (Phase 3 slice 3g-c).
//!
//! Replaces the free-form Qt `QWidget` a Python plugin built in the original app: egui is
//! immediate-mode and embedded Python can't own egui widgets, so a plugin instead *declares* its
//! controls. It defines a `panel(self, builder)` method and calls into the host-provided
//! [`PanelBuilder`], which accumulates [`Control`]s; the host (GUI) renders them as an egui window,
//! owns the live values, and reports each change back via the plugin's `on_control(self, key,
//! value)` hook. This mirrors the `overlay(self, painter)` collect-then-render pattern (slice 3f) —
//! Python never holds a live widget handle.

use pyo3::prelude::*;

/// One declared control. The inline value is the plugin-declared *initial* value; after the panel
/// is built the host owns the live value (egui mutates it in place) and reports changes through the
/// plugin's `on_control(key, value)` hook. Keyed controls carry a stable `key` the plugin matches on.
#[derive(Clone, Debug)]
pub enum Control {
    /// Float slider over `[min, max]`, reports `on_control(key, float)`.
    Slider {
        key: String,
        label: String,
        value: f64,
        min: f64,
        max: f64,
    },
    /// Boolean checkbox, reports `on_control(key, bool)`.
    Checkbox {
        key: String,
        label: String,
        value: bool,
    },
    /// Action button, reports `on_control(key, None)` on click.
    Button { key: String, label: String },
    /// Static text (no interaction). Declared once; not updated after build.
    Label { text: String },
}

/// The new value of a changed control, dispatched to `on_control(self, key, value)`:
/// a slider → `Float`, a checkbox → `Bool`, a button click → `Click` (Python `None`).
#[derive(Clone, Copy, Debug)]
pub enum ControlValue {
    Float(f64),
    Bool(bool),
    Click,
}

/// The builder handed to a plugin's `panel(self, builder)`. Each method appends a [`Control`];
/// the host reads [`PanelBuilder::controls`] after the call (mirrors [`crate::overlay::OverlayPainter`]).
#[pyclass]
#[derive(Default)]
pub struct PanelBuilder {
    pub controls: Vec<Control>,
}

#[pymethods]
impl PanelBuilder {
    /// Add a float slider over `[min, max]` with initial `value`. Changes call
    /// `on_control(key, <new float>)`.
    #[pyo3(signature = (key, label, value, min, max))]
    fn slider(&mut self, key: String, label: String, value: f64, min: f64, max: f64) {
        self.controls.push(Control::Slider {
            key,
            label,
            value,
            min,
            max,
        });
    }

    /// Add a checkbox with initial `value`. Toggles call `on_control(key, <new bool>)`.
    #[pyo3(signature = (key, label, value=false))]
    fn checkbox(&mut self, key: String, label: String, value: bool) {
        self.controls.push(Control::Checkbox { key, label, value });
    }

    /// Add an action button. Clicks call `on_control(key, None)`.
    #[pyo3(signature = (key, label))]
    fn button(&mut self, key: String, label: String) {
        self.controls.push(Control::Button { key, label });
    }

    /// Add a static text label (no interaction).
    #[pyo3(signature = (text))]
    fn label(&mut self, text: String) {
        self.controls.push(Control::Label { text });
    }
}
