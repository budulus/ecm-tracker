//! Settings persistence round-trip (own test binary → isolated TRACKER_CONFIG_DIR).

use ecm_core::settings::{get_section, load_settings, update_section};
use serde_json::json;

#[test]
fn settings_roundtrip() {
    let dir = std::env::temp_dir().join(format!("ecm_settings_{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::env::set_var("TRACKER_CONFIG_DIR", &dir);

    // Empty to start.
    assert!(load_settings().is_empty());
    assert!(get_section("lk").is_none());

    update_section("lk", json!({"win_size": 31, "max_level": 4})).unwrap();
    update_section("grid", json!({"spacing_x": 25})).unwrap();

    let lk = get_section("lk").unwrap();
    assert_eq!(lk["win_size"], 31);
    assert_eq!(lk["max_level"], 4);
    assert_eq!(get_section("grid").unwrap()["spacing_x"], 25);
    assert!(get_section("missing").is_none());

    // Persisted file exists with sorted keys ("grid" before "lk").
    let text = std::fs::read_to_string(dir.join("settings.json")).unwrap();
    assert!(text.find("\"grid\"").unwrap() < text.find("\"lk\"").unwrap());

    std::fs::remove_dir_all(&dir).ok();
}
