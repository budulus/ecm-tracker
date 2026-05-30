//! ProjectState dual-index system + validated setters (own test binary → isolated env).

use ecm_core::image_sequence::{discover_dir, ImageSequence};
use ecm_core::project_state::ProjectState;
use ecm_core::result::LkParams;
use std::path::{Path, PathBuf};

fn frames_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/frames")
}

#[test]
fn dual_index_and_setters() {
    // Deterministic defaults: point settings at an empty temp dir.
    let cfg = std::env::temp_dir().join(format!("ecm_ps_{}", std::process::id()));
    std::fs::remove_dir_all(&cfg).ok();
    std::env::set_var("TRACKER_CONFIG_DIR", &cfg);

    let mut st = ProjectState::new();
    assert!(!st.has_sequence());
    assert_eq!(st.total_images(), 0);
    assert_eq!(st.lk_params, LkParams::default());
    assert!(st.display_params.show_roi);

    let seq = ImageSequence::new(discover_dir(&frames_dir()).unwrap()).unwrap();
    st.load_sequence(seq, Some("fixtures".into()));
    assert_eq!(st.total_images(), 12);
    assert_eq!(st.reference_index, 0);
    assert_eq!(st.last_index, 11);
    assert_eq!(st.n_cut(), 12);
    assert!(st.on_reference_frame());
    assert_eq!(st.image_size().unwrap(), Some((240, 320))); // (height, width)

    // Dual-index conversions.
    st.set_reference(3);
    st.set_last(9);
    assert_eq!(st.n_cut(), 7);
    assert_eq!(st.global_to_cut(3), 0);
    assert_eq!(st.global_to_cut(9), 6);
    assert_eq!(st.global_to_cut(2), -1); // before the reference frame
    assert_eq!(st.cut_to_global(0), 3);
    assert_eq!(st.cut_to_global(6), 9);

    // Setter clamping + invariants (0 <= reference <= last < total).
    st.set_current(100);
    assert_eq!(st.current_index, 11);
    st.set_current(-5);
    assert_eq!(st.current_index, 0);
    st.set_reference(50); // cannot exceed last (9)
    assert_eq!(st.reference_index, 9);
    st.set_last(0); // cannot drop below reference (9)
    assert_eq!(st.last_index, 9);

    std::fs::remove_dir_all(&cfg).ok();
}
