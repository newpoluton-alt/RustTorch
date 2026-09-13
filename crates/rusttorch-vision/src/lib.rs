//! Decode local images and keep annotations aligned through geometric transforms.
//!
//! Images are RGB `Uint8` CHW tensors until [`VisionSample::to_float`]. Datasets
//! read local files only; no download or archive extraction happens implicitly.
//!
//! ```no_run
//! use rusttorch_data::{DataLoader, ResourceLimits};
//! use rusttorch_vision::ImageFolder;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let dataset = ImageFolder::open("images/train", ResourceLimits::default())?;
//! let mut loader = DataLoader::builder(dataset).batch_size(8).workers(2).build()?;
//! for batch in loader.iter() {
//!     let batch = batch?;
//!     // Same-sized inputs stack to [batch, 3, height, width].
//!     println!("{:?}", batch.images.size());
//! }
//! # Ok(()) }
//! ```
#![deny(missing_docs)]
#![doc = include_str!("../README.md")]

use image::{ImageDecoder, ImageReader};
use rusttorch_core::{Device, Kind, Tensor};
use rusttorch_data::{
    CollateError, Dataset, DefaultCollate, DefaultConvert, MemoryFootprint, PinMemory,
    ResourceLimits,
};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Cursor, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

/// Image decoding, annotation, I/O or tensor failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VisionError {
    /// A local file could not be read.
    #[error("{path}: {source}")]
    Io {
        /// Input path.
        path: PathBuf,
        /// Preserved I/O failure.
        #[source]
        source: std::io::Error,
    },
    /// Image codec failure.
    #[error(transparent)]
    Decode(#[from] image::ImageError),
    /// Invalid annotation JSON.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Invalid metadata or transform configuration.
    #[error("{0}")]
    Invalid(String),
    /// Resource ceiling was exceeded.
    #[error(transparent)]
    Limit(#[from] rusttorch_core::RustTorchError),
    /// LibTorch rejected tensor construction or transformation.
    #[error(transparent)]
    Tensor(#[from] tch::TchError),
}
/// Result returned by vision operations.
pub type Result<T> = std::result::Result<T, VisionError>;

/// An axis-aligned rectangle in continuous pixel-edge XYXY coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingBox {
    /// Left edge, inclusive coordinate.
    pub x_min: f64,
    /// Top edge.
    pub y_min: f64,
    /// Right edge.
    pub x_max: f64,
    /// Bottom edge.
    pub y_max: f64,
    /// Dataset category identifier, preserved without remapping.
    pub category: i64,
}
/// A keypoint in pixel-edge coordinates, with COCO visibility (0, 1 or 2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Keypoint {
    /// Horizontal coordinate.
    pub x: f64,
    /// Vertical coordinate.
    pub y: f64,
    /// Zero means absent; one means labeled and two visible.
    pub visibility: u8,
}
/// Owned CHW RGB bytes and associated classification/detection targets.
#[derive(Debug)]
pub struct VisionSample {
    /// RGB tensor shaped `[3, height, width]`; Uint8 or normalized Float.
    pub image: Tensor,
    /// Optional classification label.
    pub label: Option<i64>,
    /// Detection rectangles.
    pub boxes: Vec<BoundingBox>,
    /// Keypoints; geometric operations preserve their order.
    pub keypoints: Vec<Keypoint>,
    /// Optional integer segmentation mask shaped `[height, width]`.
    pub mask: Option<Tensor>,
}
impl VisionSample {
    /// Validates RGB shape and constructs a sample with no detection targets.
    pub fn new(image: Tensor, label: Option<i64>) -> Result<Self> {
        let value = Self {
            image,
            label,
            boxes: Vec::new(),
            keypoints: Vec::new(),
            mask: None,
        };
        value.validate()?;
        Ok(value)
    }
    /// Validates tensor/target shapes and finite in-bounds coordinates.
    pub fn validate(&self) -> Result<()> {
        let size = self.image.size();
        if size.len() != 3
            || size[0] != 3
            || size[1] <= 0
            || size[2] <= 0
            || !matches!(self.image.kind(), Kind::Uint8 | Kind::Float)
        {
            return Err(VisionError::Invalid(
                "expected nonempty RGB Uint8/Float [3,H,W]".into(),
            ));
        }
        let (w, h) = (size[2] as f64, size[1] as f64);
        for b in &self.boxes {
            if ![b.x_min, b.y_min, b.x_max, b.y_max]
                .iter()
                .all(|v| v.is_finite())
                || b.x_min < 0.
                || b.y_min < 0.
                || b.x_max > w
                || b.y_max > h
                || b.x_min >= b.x_max
                || b.y_min >= b.y_max
            {
                return Err(VisionError::Invalid("invalid bounding box".into()));
            }
        }
        for k in &self.keypoints {
            if !k.x.is_finite()
                || !k.y.is_finite()
                || k.visibility > 2
                || (k.visibility > 0 && (k.x < 0. || k.y < 0. || k.x > w || k.y > h))
            {
                return Err(VisionError::Invalid("invalid keypoint".into()));
            }
        }
        if let Some(mask) = &self.mask
            && (mask.size() != size[1..]
                || !matches!(mask.kind(), Kind::Uint8 | Kind::Int64 | Kind::Int)
                || mask.device() != self.image.device())
        {
            return Err(VisionError::Invalid(
                "mask must match image H,W, device and use integer labels".into(),
            ));
        }
        Ok(())
    }
    /// Converts RGB bytes to Float in `[0,1]`; Float inputs pass through.
    pub fn to_float(mut self) -> Result<Self> {
        self.validate()?;
        if self.image.kind() == Kind::Uint8 {
            self.image = self.image.f_to_kind(Kind::Float)?.f_div_scalar(255.)?;
        }
        Ok(self)
    }
    /// Resizes RGB with bilinear interpolation and masks with nearest-neighbor.
    /// All boxes/keypoints use the same scale factors. Does not mutate input storage.
    pub fn resize(mut self, height: u32, width: u32, limits: ResourceLimits) -> Result<Self> {
        self.validate()?;
        check_image(height, width, limits)?;
        let old = self.image.size();
        let kind = self.image.kind();
        let pixels = limits.tensor_elements(&[height as usize, width as usize])?;
        let per_pixel = 3 * kind.elt_size_in_bytes()
            + self
                .mask
                .as_ref()
                .map_or(0, |mask| mask.kind().elt_size_in_bytes());
        limits.check(
            "resized image and mask bytes",
            pixels
                .checked_mul(per_pixel)
                .ok_or_else(|| VisionError::Invalid("image and mask bytes overflow".into()))?,
            limits.max_decoded_bytes,
        )?;
        self.image = self
            .image
            .f_to_kind(Kind::Float)?
            .f_unsqueeze(0)?
            .f_upsample_bilinear2d([height as i64, width as i64], false, None, None)?
            .f_squeeze_dim(0)?;
        if kind == Kind::Uint8 {
            self.image = self.image.f_round()?.f_to_kind(kind)?;
        }
        if let Some(mask) = self.mask.take() {
            // Select integer coordinates directly so Int64 category IDs never
            // round-trip through floating-point mask values.
            let rows = Tensor::f_arange(height as i64, (Kind::Int64, mask.device()))?
                .f_mul_scalar(old[1])?
                .f_floor_divide_scalar(height as i64)?;
            let columns = Tensor::f_arange(width as i64, (Kind::Int64, mask.device()))?
                .f_mul_scalar(old[2])?
                .f_floor_divide_scalar(width as i64)?;
            self.mask = Some(mask.f_index_select(0, &rows)?.f_index_select(1, &columns)?);
        }
        let (sx, sy) = (width as f64 / old[2] as f64, height as f64 / old[1] as f64);
        for b in &mut self.boxes {
            b.x_min *= sx;
            b.x_max *= sx;
            b.y_min *= sy;
            b.y_max *= sy;
        }
        for k in &mut self.keypoints {
            k.x *= sx;
            k.y *= sy;
        }
        Ok(self)
    }
    /// Flips image, boxes and mask horizontally. Keypoint order is unchanged;
    /// callers must supply a left/right semantic permutation themselves.
    pub fn horizontal_flip(mut self) -> Result<Self> {
        self.validate()?;
        let width = self.image.size()[2] as f64;
        self.image = self.image.f_flip([2])?;
        self.mask = self.mask.map(|m| m.f_flip([1])).transpose()?;
        for b in &mut self.boxes {
            (b.x_min, b.x_max) = (width - b.x_max, width - b.x_min);
        }
        for k in &mut self.keypoints {
            k.x = width - k.x;
        }
        Ok(self)
    }
    /// Crops to a checked rectangle, clips boxes, removes empty boxes and marks
    /// outside keypoints absent. Mask and image use the identical crop.
    pub fn crop(mut self, top: u32, left: u32, height: u32, width: u32) -> Result<Self> {
        self.validate()?;
        let s = self.image.size();
        if height == 0
            || width == 0
            || u64::from(top) + u64::from(height) > s[1] as u64
            || u64::from(left) + u64::from(width) > s[2] as u64
        {
            return Err(VisionError::Invalid("crop outside image".into()));
        }
        self.image = self
            .image
            .f_narrow(1, top as i64, height as i64)?
            .f_narrow(2, left as i64, width as i64)?;
        self.mask = self
            .mask
            .map(|m| {
                m.f_narrow(0, top as i64, height as i64)?
                    .f_narrow(1, left as i64, width as i64)
            })
            .transpose()?;
        for b in &mut self.boxes {
            b.x_min = (b.x_min - left as f64).clamp(0., width as f64);
            b.x_max = (b.x_max - left as f64).clamp(0., width as f64);
            b.y_min = (b.y_min - top as f64).clamp(0., height as f64);
            b.y_max = (b.y_max - top as f64).clamp(0., height as f64);
        }
        self.boxes
            .retain(|b| b.x_min < b.x_max && b.y_min < b.y_max);
        for k in &mut self.keypoints {
            k.x -= left as f64;
            k.y -= top as f64;
            if k.x < 0. || k.x > width as f64 || k.y < 0. || k.y > height as f64 {
                k.visibility = 0;
            }
        }
        Ok(self)
    }
}
fn check_image(height: u32, width: u32, limits: ResourceLimits) -> Result<()> {
    if height == 0 || width == 0 {
        return Err(VisionError::Invalid(
            "image dimensions must be positive".into(),
        ));
    }
    limits.check("image height", height as usize, limits.max_image_dimension)?;
    limits.check("image width", width as usize, limits.max_image_dimension)?;
    let n = limits.tensor_elements(&[3, height as usize, width as usize])?;
    limits.check(
        "decoded image",
        n.checked_mul(4)
            .ok_or_else(|| VisionError::Invalid("image bytes overflow".into()))?,
        limits.max_decoded_bytes,
    )?;
    Ok(())
}
fn io(path: &Path, source: std::io::Error) -> VisionError {
    VisionError::Io {
        path: path.into(),
        source,
    }
}
/// Decodes a bounded PNG/JPEG into RGB Uint8 CHW, ignoring EXIF orientation.
pub fn decode_image(path: impl AsRef<Path>, limits: ResourceLimits) -> Result<VisionSample> {
    let path = path.as_ref();
    let bytes = limits
        .read_encoded(File::open(path).map_err(|e| io(path, e))?)
        .map_err(|e| io(path, e))?;
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| io(path, e))?;
    let mut native = image::Limits::default();
    native.max_image_width = Some(limits.max_image_dimension.min(u32::MAX as usize) as u32);
    native.max_image_height = native.max_image_width;
    native.max_alloc = Some(limits.max_decoded_bytes as u64);
    reader.limits(native);
    let decoder = reader.into_decoder()?;
    let (w, h) = decoder.dimensions();
    check_image(h, w, limits)?;
    let rgb = image::DynamicImage::from_decoder(decoder)?.into_rgb8();
    let tensor = Tensor::f_from_slice(rgb.as_raw())?
        .f_reshape([h as i64, w as i64, 3])?
        .f_permute([2, 0, 1])?;
    VisionSample::new(tensor, None)
}
/// Dense image batch with variable-size annotations retained per sample.
#[derive(Debug)]
pub struct VisionBatch {
    /// Same-shaped images stacked on a new leading batch axis.
    pub images: Tensor,
    /// Classification labels, in sample order.
    pub labels: Vec<Option<i64>>,
    /// Per-image rectangles.
    pub boxes: Vec<Vec<BoundingBox>>,
    /// Per-image keypoints.
    pub keypoints: Vec<Vec<Keypoint>>,
    /// Per-image optional masks.
    pub masks: Vec<Option<Tensor>>,
}
impl DefaultCollate for VisionSample {
    type Batch = VisionBatch;
    fn default_collate(samples: Vec<Self>) -> std::result::Result<VisionBatch, CollateError> {
        let mut images = Vec::with_capacity(samples.len());
        let mut labels = Vec::new();
        let mut boxes = Vec::new();
        let mut keypoints = Vec::new();
        let mut masks = Vec::new();
        for s in samples {
            images.push(s.image);
            labels.push(s.label);
            boxes.push(s.boxes);
            keypoints.push(s.keypoints);
            masks.push(s.mask);
        }
        Ok(VisionBatch {
            images: Tensor::default_collate(images)?,
            labels,
            boxes,
            keypoints,
            masks,
        })
    }
}
impl DefaultConvert for VisionSample {
    type Output = Self;
    fn default_convert(self) -> std::result::Result<Self, CollateError> {
        Ok(self)
    }
}
impl MemoryFootprint for VisionSample {
    fn resident_bytes(&self) -> usize {
        self.image
            .resident_bytes()
            .saturating_add(self.mask.resident_bytes())
            .saturating_add(self.label.resident_bytes())
            .saturating_add(
                self.boxes
                    .capacity()
                    .saturating_mul(std::mem::size_of::<BoundingBox>()),
            )
            .saturating_add(
                self.keypoints
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Keypoint>()),
            )
    }
}
impl MemoryFootprint for VisionBatch {
    fn resident_bytes(&self) -> usize {
        self.images
            .resident_bytes()
            .saturating_add(self.masks.resident_bytes())
            .saturating_add(
                self.labels
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Option<i64>>()),
            )
            .saturating_add(
                self.boxes
                    .iter()
                    .map(|b| {
                        b.capacity()
                            .saturating_mul(std::mem::size_of::<BoundingBox>())
                    })
                    .fold(0, usize::saturating_add),
            )
            .saturating_add(
                self.keypoints
                    .iter()
                    .map(|b| b.capacity().saturating_mul(std::mem::size_of::<Keypoint>()))
                    .fold(0, usize::saturating_add),
            )
    }
}
impl PinMemory for VisionSample {
    fn pin_memory(mut self, d: Device) -> rusttorch_core::Result<Self> {
        self.image = self.image.pin_memory(d)?;
        self.mask = self.mask.pin_memory(d)?;
        Ok(self)
    }
}
impl PinMemory for VisionBatch {
    fn pin_memory(mut self, d: Device) -> rusttorch_core::Result<Self> {
        self.images = self.images.pin_memory(d)?;
        self.masks = self.masks.pin_memory(d)?;
        Ok(self)
    }
}

/// Images under `root/class-name/image.png`, with sorted classes and paths.
/// Symlinks are rejected; traversal is bounded by the record limit.
#[derive(Debug)]
pub struct ImageFolder {
    files: Vec<(PathBuf, i64)>,
    classes: Vec<String>,
    limits: ResourceLimits,
}
impl ImageFolder {
    /// Scans a local class directory. Empty classes and unsupported files fail.
    pub fn open(root: impl AsRef<Path>, limits: ResourceLimits) -> Result<Self> {
        let root = root.as_ref();
        let mut dirs = entries(root, limits)?;
        dirs.sort();
        let mut classes = Vec::new();
        let mut files = Vec::new();
        for dir in dirs {
            let meta = fs::symlink_metadata(&dir).map_err(|e| io(&dir, e))?;
            if !meta.is_dir() || meta.file_type().is_symlink() {
                return Err(VisionError::Invalid(format!(
                    "expected class directory: {}",
                    dir.display()
                )));
            }
            let label = classes.len() as i64;
            classes.push(
                dir.file_name()
                    .and_then(|x| x.to_str())
                    .ok_or_else(|| VisionError::Invalid("class name is not UTF-8".into()))?
                    .to_owned(),
            );
            let mut paths = Vec::new();
            walk(&dir, &mut paths, limits, 0)?;
            if paths.is_empty() {
                return Err(VisionError::Invalid("empty image class".into()));
            }
            paths.sort();
            for path in paths {
                files.push((path, label));
                limits.check("image records", files.len(), limits.max_records)?;
            }
        }
        if classes.is_empty() {
            return Err(VisionError::Invalid("no image classes".into()));
        }
        Ok(Self {
            files,
            classes,
            limits,
        })
    }
    /// Sorted label-to-class-name mapping.
    pub fn classes(&self) -> &[String] {
        &self.classes
    }
}
fn entries(path: &Path, limits: ResourceLimits) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(path).map_err(|e| io(path, e))? {
        limits.check(
            "directory entries",
            paths.len().saturating_add(1),
            limits.max_records,
        )?;
        paths.push(entry.map_err(|e| io(path, e))?.path());
    }
    Ok(paths)
}

fn walk(path: &Path, out: &mut Vec<PathBuf>, limits: ResourceLimits, depth: usize) -> Result<()> {
    if depth > 64 {
        return Err(VisionError::Invalid(
            "image directory nesting exceeds 64".into(),
        ));
    }
    for p in entries(path, limits)? {
        let m = fs::symlink_metadata(&p).map_err(|e| io(&p, e))?;
        if m.file_type().is_symlink() {
            return Err(VisionError::Invalid("image symlink rejected".into()));
        }
        if m.is_dir() {
            walk(&p, out, limits, depth + 1)?;
        } else if m.is_file() {
            let ext = p
                .extension()
                .and_then(|x| x.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if !["png", "jpg", "jpeg"].contains(&ext.as_str()) {
                return Err(VisionError::Invalid(format!(
                    "unsupported image: {}",
                    p.display()
                )));
            }
            out.push(p);
            limits.check("image records", out.len(), limits.max_records)?;
        } else {
            return Err(VisionError::Invalid("nonregular image input".into()));
        }
    }
    Ok(())
}
impl Dataset for ImageFolder {
    fn get_batch_with_context(
        &self,
        indices: &[usize],
        context: &rusttorch_data::WorkerContext,
    ) -> Result<Vec<Self::Sample>> {
        indices
            .iter()
            .map(|&index| {
                context.check().map_err(|_| {
                    VisionError::Invalid("sample loading cancelled or timed out".into())
                })?;
                self.get(index)
            })
            .collect()
    }

    type Sample = VisionSample;
    type Error = VisionError;
    fn len(&self) -> usize {
        self.files.len()
    }
    fn get(&self, index: usize) -> Result<VisionSample> {
        let (p, label) = self
            .files
            .get(index)
            .ok_or_else(|| VisionError::Invalid("image index out of range".into()))?;
        let mut s = decode_image(p, self.limits)?;
        s.label = Some(*label);
        Ok(s)
    }
}

/// Local uncompressed MNIST IDX image/label pair, read one sample at a time.
#[derive(Debug)]
pub struct Mnist {
    images: PathBuf,
    labels: PathBuf,
    len: usize,
    height: u32,
    width: u32,
    limits: ResourceLimits,
}
impl Mnist {
    /// Validates IDX magic, matching counts, checked sizes and exact file lengths.
    pub fn open(
        images: impl AsRef<Path>,
        labels: impl AsRef<Path>,
        limits: ResourceLimits,
    ) -> Result<Self> {
        let (images, labels) = (images.as_ref(), labels.as_ref());
        let mut a = File::open(images).map_err(|e| io(images, e))?;
        let mut b = File::open(labels).map_err(|e| io(labels, e))?;
        let mut ah = [0u8; 16];
        a.read_exact(&mut ah).map_err(|e| io(images, e))?;
        let mut bh = [0u8; 8];
        b.read_exact(&mut bh).map_err(|e| io(labels, e))?;
        let read = |s: &[u8]| u32::from_be_bytes(s.try_into().expect("fixed header field"));
        let n = read(&ah[4..8]);
        let h = read(&ah[8..12]);
        let w = read(&ah[12..16]);
        if read(&ah[..4]) != 2051 || read(&bh[..4]) != 2049 || read(&bh[4..8]) != n {
            return Err(VisionError::Invalid(
                "invalid MNIST IDX magic or record counts".into(),
            ));
        }
        check_image(h, w, limits)?;
        limits.check("MNIST records", n as usize, limits.max_records)?;
        let size = (n as u64)
            .checked_mul(h as u64)
            .and_then(|n| n.checked_mul(w as u64))
            .and_then(|n| n.checked_add(16))
            .ok_or_else(|| VisionError::Invalid("MNIST length overflow".into()))?;
        if a.metadata().map_err(|e| io(images, e))?.len() != size
            || b.metadata().map_err(|e| io(labels, e))?.len() != n as u64 + 8
        {
            return Err(VisionError::Invalid("MNIST file length mismatch".into()));
        }
        Ok(Self {
            images: images.into(),
            labels: labels.into(),
            len: n as usize,
            height: h,
            width: w,
            limits,
        })
    }
}
impl Dataset for Mnist {
    fn get_batch_with_context(
        &self,
        indices: &[usize],
        context: &rusttorch_data::WorkerContext,
    ) -> Result<Vec<Self::Sample>> {
        indices
            .iter()
            .map(|&index| {
                context.check().map_err(|_| {
                    VisionError::Invalid("sample loading cancelled or timed out".into())
                })?;
                self.get(index)
            })
            .collect()
    }

    type Sample = VisionSample;
    type Error = VisionError;
    fn len(&self) -> usize {
        self.len
    }
    fn get(&self, index: usize) -> Result<VisionSample> {
        if index >= self.len {
            return Err(VisionError::Invalid("MNIST index out of range".into()));
        }
        let n = self
            .limits
            .tensor_elements(&[self.height as usize, self.width as usize])?;
        let mut data = vec![0; n];
        let mut a = File::open(&self.images).map_err(|e| io(&self.images, e))?;
        a.seek(SeekFrom::Start(16 + (index as u64) * (n as u64)))
            .map_err(|e| io(&self.images, e))?;
        a.read_exact(&mut data).map_err(|e| io(&self.images, e))?;
        let mut label = [0u8; 1];
        let mut b = File::open(&self.labels).map_err(|e| io(&self.labels, e))?;
        b.seek(SeekFrom::Start(8 + index as u64))
            .map_err(|e| io(&self.labels, e))?;
        b.read_exact(&mut label).map_err(|e| io(&self.labels, e))?;
        if label[0] > 9 {
            return Err(VisionError::Invalid("MNIST label outside 0..9".into()));
        }
        VisionSample::new(
            Tensor::f_from_slice(&data)?
                .f_reshape([1, self.height as i64, self.width as i64])?
                .f_repeat([3, 1, 1])?,
            Some(label[0] as i64),
        )
    }
}

/// One local CIFAR-10 or CIFAR-100 binary file; RGB planar bytes are preserved.
#[derive(Debug)]
pub struct Cifar {
    path: PathBuf,
    len: usize,
    cifar100: bool,
}
impl Cifar {
    /// Opens a binary file; CIFAR-100 returns the fine (second) class label.
    pub fn open(path: impl AsRef<Path>, cifar100: bool, limits: ResourceLimits) -> Result<Self> {
        let p = path.as_ref();
        check_image(32, 32, limits)?;
        let size = fs::metadata(p).map_err(|e| io(p, e))?.len();
        let stride = 3073 + u64::from(cifar100);
        if size % stride != 0 {
            return Err(VisionError::Invalid(
                "CIFAR file has a partial record".into(),
            ));
        }
        let len = usize::try_from(size / stride)
            .map_err(|_| VisionError::Invalid("CIFAR length overflow".into()))?;
        limits.check("CIFAR records", len, limits.max_records)?;
        Ok(Self {
            path: p.into(),
            len,
            cifar100,
        })
    }
}
impl Dataset for Cifar {
    fn get_batch_with_context(
        &self,
        indices: &[usize],
        context: &rusttorch_data::WorkerContext,
    ) -> Result<Vec<Self::Sample>> {
        indices
            .iter()
            .map(|&index| {
                context.check().map_err(|_| {
                    VisionError::Invalid("sample loading cancelled or timed out".into())
                })?;
                self.get(index)
            })
            .collect()
    }

    type Sample = VisionSample;
    type Error = VisionError;
    fn len(&self) -> usize {
        self.len
    }
    fn get(&self, index: usize) -> Result<VisionSample> {
        if index >= self.len {
            return Err(VisionError::Invalid("CIFAR index out of range".into()));
        }
        let stride = 3073 + usize::from(self.cifar100);
        let mut data = vec![0u8; stride];
        let mut f = File::open(&self.path).map_err(|e| io(&self.path, e))?;
        f.seek(SeekFrom::Start((index as u64) * (stride as u64)))
            .map_err(|e| io(&self.path, e))?;
        f.read_exact(&mut data).map_err(|e| io(&self.path, e))?;
        let label = data[usize::from(self.cifar100)];
        if label >= if self.cifar100 { 100 } else { 10 } {
            return Err(VisionError::Invalid("CIFAR label out of range".into()));
        }
        VisionSample::new(
            Tensor::f_from_slice(&data[1 + usize::from(self.cifar100)..])?
                .f_reshape([3, 32, 32])?,
            Some(label as i64),
        )
    }
}

#[derive(Deserialize)]
struct CocoFile {
    images: Vec<CocoImage>,
    annotations: Vec<CocoAnnotation>,
}
#[derive(Deserialize)]
struct CocoImage {
    id: i64,
    file_name: String,
    width: u32,
    height: u32,
}
#[derive(Deserialize)]
struct CocoAnnotation {
    image_id: i64,
    category_id: i64,
    bbox: [f64; 4],
    #[serde(default)]
    keypoints: Vec<f64>,
    #[serde(default)]
    segmentation: serde_json::Value,
}
/// COCO-style bounding-box/keypoint dataset. Polygon/RLE segmentation is rejected
/// explicitly; integer masks can be attached to [`VisionSample`] by callers.
pub struct CocoDetection {
    root: PathBuf,
    images: Vec<CocoImage>,
    annotations: BTreeMap<i64, Vec<CocoAnnotation>>,
    limits: ResourceLimits,
}
impl CocoDetection {
    /// Reads bounded local annotation JSON, validates image IDs and safe paths.
    pub fn open(
        root: impl AsRef<Path>,
        annotations: impl AsRef<Path>,
        limits: ResourceLimits,
    ) -> Result<Self> {
        let p = annotations.as_ref();
        let bytes = limits
            .read_encoded(File::open(p).map_err(|e| io(p, e))?)
            .map_err(|e| io(p, e))?;
        let parsed: CocoFile = serde_json::from_slice(&bytes)?;
        limits.check("COCO images", parsed.images.len(), limits.max_records)?;
        limits.check(
            "COCO annotations",
            parsed.annotations.len(),
            limits.max_records,
        )?;
        let root = fs::canonicalize(root.as_ref()).map_err(|e| io(root.as_ref(), e))?;
        let mut ids = BTreeMap::new();
        for im in &parsed.images {
            check_image(im.height, im.width, limits)?;
            limits.check(
                "COCO file name",
                im.file_name.len(),
                limits.max_string_bytes,
            )?;
            let path = fs::canonicalize(root.join(&im.file_name))
                .map_err(|e| io(Path::new(&im.file_name), e))?;
            if !path.starts_with(&root) || ids.insert(im.id, ()).is_some() {
                return Err(VisionError::Invalid(
                    "COCO duplicate image ID or escaping image path".into(),
                ));
            }
        }
        let mut by_image: BTreeMap<i64, Vec<CocoAnnotation>> = BTreeMap::new();
        for a in parsed.annotations {
            if !ids.contains_key(&a.image_id) || a.keypoints.len() % 3 != 0 {
                return Err(VisionError::Invalid(
                    "COCO unknown image or malformed keypoints".into(),
                ));
            }
            if !a.segmentation.is_null() && a.segmentation != serde_json::json!([]) {
                return Err(VisionError::Invalid(
                    "COCO segmentation decoding is not supported; provide an integer mask".into(),
                ));
            }
            by_image.entry(a.image_id).or_default().push(a);
        }
        Ok(Self {
            root,
            images: parsed.images,
            annotations: by_image,
            limits,
        })
    }
}
impl Dataset for CocoDetection {
    fn get_batch_with_context(
        &self,
        indices: &[usize],
        context: &rusttorch_data::WorkerContext,
    ) -> Result<Vec<Self::Sample>> {
        indices
            .iter()
            .map(|&index| {
                context.check().map_err(|_| {
                    VisionError::Invalid("sample loading cancelled or timed out".into())
                })?;
                self.get(index)
            })
            .collect()
    }

    type Sample = VisionSample;
    type Error = VisionError;
    fn len(&self) -> usize {
        self.images.len()
    }
    fn get(&self, index: usize) -> Result<VisionSample> {
        let im = self
            .images
            .get(index)
            .ok_or_else(|| VisionError::Invalid("COCO index out of range".into()))?;
        let path = fs::canonicalize(self.root.join(&im.file_name))
            .map_err(|e| io(Path::new(&im.file_name), e))?;
        if !path.starts_with(&self.root) {
            return Err(VisionError::Invalid("COCO image path escapes root".into()));
        }
        let mut s = decode_image(path, self.limits)?;
        if s.image.size()[1..] != [im.height as i64, im.width as i64] {
            return Err(VisionError::Invalid(
                "COCO image size differs from annotation".into(),
            ));
        }
        for a in self.annotations.get(&im.id).into_iter().flatten() {
            s.boxes.push(BoundingBox {
                x_min: a.bbox[0],
                y_min: a.bbox[1],
                x_max: a.bbox[0] + a.bbox[2],
                y_max: a.bbox[1] + a.bbox[3],
                category: a.category_id,
            });
            for k in a.keypoints.as_chunks::<3>().0 {
                if k[2].fract() != 0. || !(0. ..=2.).contains(&k[2]) {
                    return Err(VisionError::Invalid("COCO visibility must be 0,1,2".into()));
                }
                s.keypoints.push(Keypoint {
                    x: k[0],
                    y: k[1],
                    visibility: k[2] as u8,
                });
            }
        }
        s.validate()?;
        Ok(s)
    }
}
