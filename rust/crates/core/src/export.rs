//! Export cleaned forward coordinates to `.npy` (+ `sequence.txt`) and `.csv`.
//! Ports `app/core/export.py`. (CSV uses `\n` line endings rather than Python csv's `\r\n`.)

use ndarray::{Array3, Axis};
use ndarray_npy::write_npy;
use std::io::Write;
use std::path::{Path, PathBuf};

fn kept_indices(active_mask: &[bool]) -> Vec<usize> {
    active_mask
        .iter()
        .enumerate()
        .filter(|(_, &k)| k)
        .map(|(i, _)| i)
        .collect()
}

/// Forward coords for kept points: `(N, P_kept, 2)` f32.
fn kept_coords(coords_fw: &Array3<f32>, active_mask: &[bool]) -> Array3<f32> {
    coords_fw.select(Axis(1), &kept_indices(active_mask))
}

fn npy_err(e: ndarray_npy::WriteNpyError) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
}

pub struct NpyExport {
    pub coords_path: PathBuf,
    pub sequence_path: PathBuf,
    pub shape: (usize, usize, usize),
}

/// Write `coords.npy` (kept forward coords) + `sequence.txt` (`"<ref> <last>"`, global indices).
pub fn export(
    coords_fw: &Array3<f32>,
    active_mask: &[bool],
    reference_index: usize,
    last_index: usize,
    out_dir: &Path,
    filename: &str,
) -> std::io::Result<NpyExport> {
    let mut fname = filename.to_string();
    if !fname.ends_with(".npy") {
        fname.push_str(".npy");
    }
    let coords = kept_coords(coords_fw, active_mask);
    let coords_path = out_dir.join(&fname);
    write_npy(&coords_path, &coords).map_err(npy_err)?;

    let sequence_path = out_dir.join("sequence.txt");
    std::fs::write(&sequence_path, format!("{reference_index} {last_index}\n"))?;

    let s = coords.shape();
    Ok(NpyExport {
        coords_path,
        sequence_path,
        shape: (s[0], s[1], s[2]),
    })
}

/// Write kept forward coords as CSV: one row per frame, columns `filename,p1x,p1y,p2x,p2y,...`.
pub fn export_csv(
    coords_fw: &Array3<f32>,
    active_mask: &[bool],
    frame_names: &[String],
    out_dir: &Path,
    filename: &str,
) -> std::io::Result<(PathBuf, (usize, usize, usize))> {
    let mut fname = filename.to_string();
    if !fname.ends_with(".csv") {
        fname.push_str(".csv");
    }
    let coords = kept_coords(coords_fw, active_mask); // (N, P_kept, 2)
    let n = coords.shape()[0];
    let p = coords.shape()[1];

    let csv_path = out_dir.join(&fname);
    let mut f = std::fs::File::create(&csv_path)?;

    let mut header = String::from("filename");
    for pi in 0..p {
        header.push_str(&format!(",p{}x,p{}y", pi + 1, pi + 1)); // 1-based labels
    }
    writeln!(f, "{header}")?;

    for i in 0..n {
        let mut row = frame_names.get(i).cloned().unwrap_or_default();
        for j in 0..p {
            // Fixed 6-decimal format (matches the Python exporter).
            row.push_str(&format!(",{:.6},{:.6}", coords[[i, j, 0]], coords[[i, j, 1]]));
        }
        writeln!(f, "{row}")?;
    }
    Ok((csv_path, (n, p, 2)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn export_subsets_kept_points_and_roundtrips() {
        // 3 frames, 4 points; keep points 0 and 2.
        let coords = Array::from_shape_fn((3, 4, 2), |(t, j, c)| (t * 100 + j * 10 + c) as f32);
        let mask = vec![true, false, true, false];
        let dir = std::env::temp_dir().join(format!("ecm_export_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let out = export(&coords, &mask, 5, 16, &dir, "coords").unwrap();
        assert_eq!(out.shape, (3, 2, 2));
        assert_eq!(
            std::fs::read_to_string(&out.sequence_path).unwrap(),
            "5 16\n"
        );

        // Round-trip the npy and confirm it holds exactly the kept columns.
        let back: Array3<f32> = ndarray_npy::read_npy(&out.coords_path).unwrap();
        assert_eq!(back.shape(), &[3, 2, 2]);
        for t in 0..3 {
            for (out_j, src_j) in [0usize, 2usize].into_iter().enumerate() {
                assert_eq!(back[[t, out_j, 0]], coords[[t, src_j, 0]]);
                assert_eq!(back[[t, out_j, 1]], coords[[t, src_j, 1]]);
            }
        }

        let names: Vec<String> = (0..3).map(|i| format!("img_{i}.png")).collect();
        let (csv_path, shape) = export_csv(&coords, &mask, &names, &dir, "coords").unwrap();
        assert_eq!(shape, (3, 2, 2));
        let text = std::fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "filename,p1x,p1y,p2x,p2y");
        // frame 0: point0=(0,1), point2=(20,21)
        assert_eq!(lines[1], "img_0.png,0.000000,1.000000,20.000000,21.000000");

        std::fs::remove_dir_all(&dir).ok();
    }
}
