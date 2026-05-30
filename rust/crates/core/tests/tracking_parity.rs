//! Tracking parity: the Rust `track()` must reproduce the Python tracker's output on
//! the same synthetic frames + seed points (fixture built by tests/fixtures/gen_fixture.py).
//!
//! Both call the same OpenCV 4.11.0 `calcOpticalFlowPyrLK`, so coordinates should match to
//! sub-pixel tolerance. Regenerate the fixture with:
//!   uv run python rust/crates/core/tests/fixtures/gen_fixture.py

use ecm_core::image_sequence::ImageSequence;
use ecm_core::result::LkParams;
use ecm_core::tracking::track;
use ndarray::{Array1, Array2, Array3};
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn npy3(name: &str) -> Array3<f32> {
    ndarray_npy::read_npy(fixtures_dir().join(name)).unwrap()
}
fn npy2f(name: &str) -> Array2<f32> {
    ndarray_npy::read_npy(fixtures_dir().join(name)).unwrap()
}
fn npy2u8(name: &str) -> Array2<u8> {
    ndarray_npy::read_npy(fixtures_dir().join(name)).unwrap()
}
fn npy1(name: &str) -> Array1<f32> {
    ndarray_npy::read_npy(fixtures_dir().join(name)).unwrap()
}

/// Max abs difference between two (N,P,2) coord arrays, but only at positions the Python
/// run tracked successfully (`status==1`) — failed points carry stale positions forward.
fn max_coord_diff(a: &Array3<f32>, b: &Array3<f32>, status: &Array2<u8>) -> f32 {
    let mut m = 0.0f32;
    for t in 0..a.shape()[0] {
        for j in 0..a.shape()[1] {
            if status[[t, j]] == 1 {
                m = m.max((a[[t, j, 0]] - b[[t, j, 0]]).abs());
                m = m.max((a[[t, j, 1]] - b[[t, j, 1]]).abs());
            }
        }
    }
    m
}

#[test]
fn rust_tracking_matches_python() {
    let fx = fixtures_dir();

    // Same frames + seeds + params the Python reference used.
    let paths = ecm_core::image_sequence::discover_dir(&fx.join("frames")).unwrap();
    assert_eq!(paths.len(), 12, "expected 12 synthetic frames");
    let seq = ImageSequence::new(paths).unwrap();

    let seeds_arr = npy2f("seeds.npy");
    let seeds: Vec<(f32, f32)> = seeds_arr
        .rows()
        .into_iter()
        .map(|r| (r[0], r[1]))
        .collect();

    // ref=0, last=11, DEFAULT_LK — matches gen_fixture.py (asserted via fixture shapes below).
    let res = track(&seq, 0, 11, &seeds, &LkParams::default(), None)
        .unwrap()
        .expect("tracking should not be cancelled");

    // Reference arrays from the Python run.
    let py_fw = npy3("coords_fw.npy");
    let py_bw = npy3("coords_bw.npy");
    let py_sfw = npy2u8("status_fw.npy");
    let py_sbw = npy2u8("status_bw.npy");
    let py_fb_mean = npy1("fb_mean.npy");
    let py_fb_max = npy1("fb_max.npy");

    // Shapes line up (also pins ref/last/point-count assumptions).
    assert_eq!(res.coords_fw.shape(), py_fw.shape());
    assert_eq!(res.n_frames(), 12);
    assert_eq!(res.n_points(), seeds.len());

    // Status must match exactly (same algorithm + OpenCV version).
    assert_eq!(res.status_fw, py_sfw, "forward status differs");
    assert_eq!(res.status_bw, py_sbw, "backward status differs");

    // Coordinates match to sub-pixel tolerance (validates LK + the backward reindex).
    let d_fw = max_coord_diff(&res.coords_fw, &py_fw, &py_sfw);
    let d_bw = max_coord_diff(&res.coords_bw, &py_bw, &py_sbw);
    eprintln!("max coord diff: forward={d_fw:.6}px  backward={d_bw:.6}px");
    const ATOL: f32 = 1e-3;
    assert!(d_fw < ATOL, "forward coord diff {d_fw} >= {ATOL}");
    assert!(d_bw < ATOL, "backward coord diff {d_bw} >= {ATOL}");

    // FB errors: inf positions agree; finite values match closely.
    let mut max_fb = 0.0f32;
    for j in 0..res.n_points() {
        for (rust, py) in [
            (res.fb_mean_error[j], py_fb_mean[j]),
            (res.fb_max_error[j], py_fb_max[j]),
        ] {
            assert_eq!(rust.is_finite(), py.is_finite(), "fb finiteness differs at {j}");
            if py.is_finite() {
                max_fb = max_fb.max((rust - py).abs());
            }
        }
    }
    eprintln!("max FB-error diff: {max_fb:.6}");
    assert!(max_fb < ATOL, "fb error diff {max_fb} >= {ATOL}");

    // Ground-truth sanity: valid points tracked the known (dx,dy)=(2,1)/frame translation.
    // Median forward displacement reference->last should be ~ (2*11, 1*11) = (22, 11).
    let last = 11usize;
    let mut dxs = Vec::new();
    let mut dys = Vec::new();
    for j in 0..res.n_points() {
        if py_sfw[[last, j]] == 1 {
            dxs.push(res.coords_fw[[last, j, 0]] - res.coords_fw[[0, j, 0]]);
            dys.push(res.coords_fw[[last, j, 1]] - res.coords_fw[[0, j, 1]]);
        }
    }
    dxs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    dys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_dx = dxs[dxs.len() / 2];
    let med_dy = dys[dys.len() / 2];
    eprintln!("median displacement: ({med_dx:.3}, {med_dy:.3}) expected ~(22, 11)");
    assert!((med_dx - 22.0).abs() < 0.5, "median dx {med_dx} != ~22");
    assert!((med_dy - 11.0).abs() < 0.5, "median dy {med_dy} != ~11");
}
