//! Cleanup parity: Rust `compute_metrics` + `build_mask` must reproduce the Python
//! reference on the same tracked result (fixture built by tests/fixtures/gen_fixture.py).
//!
//! The Rust tracker is already bit-identical to Python (see tracking_parity.rs), so feeding
//! its result into the Rust metrics and comparing against Python's metrics is a valid check.

use ecm_core::cleanup::{build_mask, compute_metrics, BandFilter, Thresholds, BAND_CAP};
use ecm_core::image_sequence::{discover_dir, ImageSequence};
use ecm_core::result::LkParams;
use ecm_core::roi::Roi;
use ecm_core::tracking::track;
use ndarray::{Array1, Array2};
use std::path::{Path, PathBuf};

fn fx() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}
fn npy1f(name: &str) -> Array1<f32> {
    ndarray_npy::read_npy(fx().join(name)).unwrap()
}
fn npy1i(name: &str) -> Array1<i64> {
    ndarray_npy::read_npy(fx().join(name)).unwrap()
}
fn npy1b(name: &str) -> Vec<bool> {
    let a: Array1<u8> = ndarray_npy::read_npy(fx().join(name)).unwrap();
    a.iter().map(|&v| v != 0).collect()
}

fn max_diff(a: &[f32], b: &Array1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn rust_cleanup_matches_python() {
    // Reproduce the tracked result (matches the Python fixture bit-for-bit).
    let seq = ImageSequence::new(discover_dir(&fx().join("frames")).unwrap()).unwrap();
    let seeds_arr: Array2<f32> = ndarray_npy::read_npy(fx().join("seeds.npy")).unwrap();
    let seeds: Vec<(f32, f32)> = seeds_arr.rows().into_iter().map(|r| (r[0], r[1])).collect();
    let res = track(&seq, 0, 11, &seeds, &LkParams::default(), None)
        .unwrap()
        .unwrap();

    // Same ROI + image size gen_fixture.py used.
    let roi = Roi::new(vec![
        (40.0, 40.0),
        (280.0, 40.0),
        (280.0, 200.0),
        (40.0, 200.0),
    ]);
    let m = compute_metrics(&res, Some(&roi), (240, 320)).unwrap();

    // Integer counts + boolean flags: exact match.
    assert_eq!(m.fail_count_fw, npy1i("m_fail_fw.npy").to_vec(), "fail_fw");
    assert_eq!(m.fail_count_bw, npy1i("m_fail_bw.npy").to_vec(), "fail_bw");
    assert_eq!(m.left_image, npy1b("m_left_image.npy"), "left_image");
    assert_eq!(m.left_roi, npy1b("m_left_roi.npy"), "left_roi");

    // Float metrics: inf positions agree; finite values match (sub-1e-4).
    for (name, rust, py) in [
        ("max_err", &m.max_err_fw, npy1f("m_max_err.npy")),
        ("mean_err", &m.mean_err_fw, npy1f("m_mean_err.npy")),
        ("max_step", &m.max_step, npy1f("m_max_step.npy")),
    ] {
        for (j, (&r, p)) in rust.iter().zip(py.iter()).enumerate() {
            assert_eq!(r.is_finite(), p.is_finite(), "{name} finiteness at {j}");
        }
        let d = max_diff(rust, &py);
        eprintln!("{name}: max finite diff = {d:.6}");
        assert!(d < 1e-4, "{name} diff {d} >= 1e-4");
    }

    // build_mask with the same Thresholds gen_fixture.py used.
    let thr = Thresholds {
        fb_mean: BandFilter { enabled: true, hi: 0.5, cap: BAND_CAP },
        distance: BandFilter { enabled: true, hi: 5.0, cap: BAND_CAP },
        drop_left_image: true,
        ..Thresholds::default()
    };
    let keep = build_mask(&m, &thr);
    assert_eq!(keep, npy1b("mask_demo.npy"), "build_mask demo mask differs");
    eprintln!("build_mask kept {}/{} points", keep.iter().filter(|&&k| k).count(), keep.len());
}
