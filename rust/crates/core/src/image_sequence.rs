//! Image discovery + on-demand decode, normalized to 3-channel BGR / 1-channel gray
//! uint8. Ports `app/core/image_sequence.py`.
//!
//! NOTE: the Python version keeps an 8-entry LRU frame cache. That's a GUI-display
//! optimization (tracking loads each frame once per pass), so it's deferred — add it
//! back when the canvas re-displays frames in Phase 2.

use opencv::core::{Mat, CV_16U, CV_8U};
use opencv::prelude::*;
use opencv::{imgcodecs, imgproc};
use std::path::{Path, PathBuf};

const SUPPORTED_EXTS: &[&str] = &["png", "jpg", "jpeg", "bmp", "tif", "tiff"];

fn is_supported(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| SUPPORTED_EXTS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// One token of a numeric-aware sort key. Digit runs compare as numbers, so
/// "img_2" sorts before "img_10" (mirrors Python's `natural_sort_key`).
#[derive(PartialEq, Eq)]
enum NatTok {
    Num(u128),
    Txt(String),
}

impl Ord for NatTok {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        use NatTok::*;
        match (self, other) {
            (Num(a), Num(b)) => a.cmp(b),
            (Txt(a), Txt(b)) => a.cmp(b),
            (Num(_), Txt(_)) => Ordering::Less,
            (Txt(_), Num(_)) => Ordering::Greater,
        }
    }
}
impl PartialOrd for NatTok {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn natural_key(name: &str) -> Vec<NatTok> {
    let mut out = Vec::new();
    let mut chars = name.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            let mut s = String::new();
            while let Some(&d) = chars.peek() {
                if d.is_ascii_digit() {
                    s.push(d);
                    chars.next();
                } else {
                    break;
                }
            }
            match s.parse::<u128>() {
                Ok(n) => out.push(NatTok::Num(n)),
                Err(_) => out.push(NatTok::Txt(s)), // overflow: keep as text
            }
        } else {
            let mut s = String::new();
            while let Some(&d) = chars.peek() {
                if d.is_ascii_digit() {
                    break;
                }
                s.push(d.to_ascii_lowercase());
                chars.next();
            }
            out.push(NatTok::Txt(s));
        }
    }
    out
}

fn sort_natural(mut files: Vec<PathBuf>) -> Vec<PathBuf> {
    files.sort_by(|a, b| {
        let ka = natural_key(&a.file_name().unwrap_or_default().to_string_lossy());
        let kb = natural_key(&b.file_name().unwrap_or_default().to_string_lossy());
        ka.cmp(&kb)
    });
    files
}

/// Naturally sorted, supported image paths in `dir`.
pub fn discover_dir(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| is_supported(p))
        .collect();
    files = sort_natural(files);
    Ok(files)
}

/// Naturally sorted, supported subset of an explicit file list.
pub fn discover_files(files: &[PathBuf]) -> Vec<PathBuf> {
    sort_natural(files.iter().filter(|p| is_supported(p)).cloned().collect())
}

/// Coerce any decoded image to a 3-channel uint8 BGR `Mat` (deterministic across frames).
fn normalize_to_bgr_u8(raw: &Mat) -> opencv::Result<Mat> {
    // Depth -> uint8.
    let u8mat = if raw.depth() == CV_16U {
        let mut dst = Mat::default();
        raw.convert_to(&mut dst, CV_8U, 1.0 / 256.0, 0.0)?; // img // 256
        dst
    } else if raw.depth() != CV_8U {
        let mut dst = Mat::default();
        raw.convert_to(&mut dst, CV_8U, 1.0, 0.0)?; // saturating cast ~ np.clip(0,255)
        dst
    } else {
        raw.try_clone()?
    };

    // Channels -> 3 (BGR).
    match u8mat.channels() {
        1 => {
            let mut out = Mat::default();
            imgproc::cvt_color_def(&u8mat, &mut out, imgproc::COLOR_GRAY2BGR)?;
            Ok(out)
        }
        4 => {
            let mut out = Mat::default();
            imgproc::cvt_color_def(&u8mat, &mut out, imgproc::COLOR_BGRA2BGR)?;
            Ok(out)
        }
        _ => Ok(u8mat),
    }
}

/// Naturally ordered image paths with on-demand decoding.
pub struct ImageSequence {
    pub paths: Vec<PathBuf>,
}

impl ImageSequence {
    pub fn new(paths: Vec<PathBuf>) -> Result<Self, String> {
        if paths.is_empty() {
            return Err("ImageSequence requires at least one image path".into());
        }
        Ok(Self { paths })
    }

    pub fn len(&self) -> usize {
        self.paths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    /// Frame at `index` as a 3-channel uint8 BGR `Mat`.
    pub fn load_bgr(&self, index: usize) -> opencv::Result<Mat> {
        let path = self.paths[index]
            .to_str()
            .ok_or_else(|| opencv::Error::new(opencv::core::StsError, "non-utf8 image path"))?;
        let raw = imgcodecs::imread(path, imgcodecs::IMREAD_UNCHANGED)?;
        if raw.empty() {
            return Err(opencv::Error::new(
                opencv::core::StsError,
                format!("Failed to load image: {path}"),
            ));
        }
        normalize_to_bgr_u8(&raw)
    }

    /// Frame at `index` as a single-channel uint8 grayscale `Mat`.
    pub fn load_gray(&self, index: usize) -> opencv::Result<Mat> {
        let bgr = self.load_bgr(index)?;
        let mut gray = Mat::default();
        imgproc::cvt_color_def(&bgr, &mut gray, imgproc::COLOR_BGR2GRAY)?;
        Ok(gray)
    }
}
