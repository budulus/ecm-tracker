//! Per-user persistent settings (JSON), for remembering parameter-dialog defaults.
//! Ports `app/core/settings.py`. Config dir resolves: `TRACKER_CONFIG_DIR` → `XDG_CONFIG_HOME`
//! → `~/.config`, then `/feature_tracker`; the file is always `settings.json`.

use serde_json::{Map, Value};
use std::path::PathBuf;

fn home_dir() -> PathBuf {
    // Python os.path.expanduser("~"): HOME on Unix, USERPROFILE on Windows.
    for var in ["HOME", "USERPROFILE"] {
        if let Ok(h) = std::env::var(var) {
            if !h.is_empty() {
                return PathBuf::from(h);
            }
        }
    }
    PathBuf::from(".")
}

fn config_dir() -> PathBuf {
    if let Ok(o) = std::env::var("TRACKER_CONFIG_DIR") {
        if !o.is_empty() {
            return PathBuf::from(o);
        }
    }
    let base = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".config"));
    base.join("feature_tracker")
}

fn config_path() -> PathBuf {
    config_dir().join("settings.json")
}

/// Whole settings document (empty map on missing/invalid file).
pub fn load_settings() -> Map<String, Value> {
    match std::fs::read_to_string(config_path()) {
        Ok(s) => match serde_json::from_str::<Value>(&s) {
            Ok(Value::Object(m)) => m,
            _ => Map::new(),
        },
        Err(_) => Map::new(),
    }
}

/// Persist the whole document (pretty, sorted keys — matches Python `indent=2, sort_keys=True`;
/// serde_json's default `Map` is a sorted `BTreeMap`).
pub fn save_settings(data: &Map<String, Value>) -> std::io::Result<()> {
    std::fs::create_dir_all(config_dir())?;
    let s = serde_json::to_string_pretty(&Value::Object(data.clone()))?;
    std::fs::write(config_path(), s)
}

/// One named section, if present and an object.
pub fn get_section(name: &str) -> Option<Value> {
    match load_settings().get(name) {
        Some(v @ Value::Object(_)) => Some(v.clone()),
        _ => None,
    }
}

/// Replace one section and persist.
pub fn update_section(name: &str, data: Value) -> std::io::Result<()> {
    let mut settings = load_settings();
    settings.insert(name.to_string(), data);
    save_settings(&settings)
}

/// Serialize a value and persist it as one section — the typed convenience over `update_section`
/// (used by the GUI's "Save as defaults"). Mirrors persisting a params dataclass to its section.
pub fn save_section<T: serde::Serialize>(name: &str, value: &T) -> std::io::Result<()> {
    let data = serde_json::to_value(value)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    update_section(name, data)
}
