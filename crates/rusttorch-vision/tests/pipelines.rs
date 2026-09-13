use rusttorch_core::{Kind, Tensor};
use rusttorch_data::{DataLoader, Dataset, MemoryFootprint, ResourceLimits};
use rusttorch_vision::*;
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "rusttorch-vision-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
#[test]
fn imagefolder_workers_decode_classify_and_bound_payload() {
    let t = Temp::new();
    for class in ["zebra", "ant"] {
        let p = t.0.join(class);
        std::fs::create_dir(&p).unwrap();
        std::fs::copy(fixture("rgb.png"), p.join("one.png")).unwrap();
    }
    let dataset = ImageFolder::open(&t.0, ResourceLimits::default()).unwrap();
    assert_eq!(dataset.classes(), ["ant", "zebra"]);
    let mut loader = DataLoader::builder(dataset)
        .batch_size(2)
        .workers(2)
        .prefetch_bytes(std::num::NonZeroUsize::new(1024).unwrap())
        .build()
        .unwrap();
    let batch = loader.iter().next().unwrap().unwrap();
    assert_eq!(batch.images.size(), [2, 3, 2, 2]);
    assert_eq!(batch.labels, [Some(0), Some(1)]);
    assert!(batch.resident_bytes() >= 24);
}
#[test]
fn target_geometry_tracks_image_mask_and_keypoints() {
    let mut s = decode_image(fixture("rgb.png"), Default::default()).unwrap();
    s.boxes.push(BoundingBox {
        x_min: 0.,
        y_min: 0.,
        x_max: 1.,
        y_max: 2.,
        category: 4,
    });
    s.keypoints.push(Keypoint {
        x: 0.5,
        y: 1.,
        visibility: 2,
    });
    s.mask = Some(Tensor::from_slice(&[1i64, 2, 3, 4]).reshape([2, 2]));
    let s = s.horizontal_flip().unwrap();
    assert_eq!(s.boxes[0].x_min, 1.);
    assert_eq!(s.keypoints[0].x, 1.5);
    assert_eq!(
        Vec::<Vec<i64>>::try_from(s.mask.as_ref().unwrap()).unwrap(),
        [[2, 1], [4, 3]]
    );
    let s = s
        .resize(4, 4, Default::default())
        .unwrap()
        .crop(0, 0, 4, 2)
        .unwrap()
        .to_float()
        .unwrap();
    assert!(s.boxes.is_empty());
    assert_eq!(s.image.size(), [3, 4, 2]);
    assert_eq!(s.image.kind(), Kind::Float);
    assert_eq!(s.keypoints[0].visibility, 0);
}
#[test]
fn named_dataset_families_read_real_local_formats() {
    let limits = ResourceLimits::default();
    let d = Mnist::open(fixture("images.idx"), fixture("labels.idx"), limits).unwrap();
    assert_eq!(d.len(), 2);
    let s = d.get(1).unwrap();
    assert_eq!(s.label, Some(8));
    assert_eq!(s.image.size(), [3, 2, 2]);
    assert!(d.get(2).is_err());
    let c = Cifar::open(fixture("cifar10.cifar"), false, limits).unwrap();
    assert_eq!(c.get(0).unwrap().label, Some(2));
    let c = Cifar::open(fixture("cifar100.cifar"), true, limits).unwrap();
    assert_eq!(c.get(0).unwrap().label, Some(17));
    let coco = CocoDetection::open(fixture(""), fixture("coco.json"), limits).unwrap();
    let s = coco.get(0).unwrap();
    assert_eq!(s.boxes[0].category, 4);
    assert_eq!(s.keypoints[0].visibility, 2);
}
#[test]
fn malformed_and_oversized_images_fail_without_panics() {
    let limits = ResourceLimits {
        max_image_dimension: 1,
        ..Default::default()
    };
    assert!(decode_image(fixture("rgb.png"), limits).is_err());
    let limits = ResourceLimits {
        max_encoded_bytes: 4,
        ..Default::default()
    };
    assert!(decode_image(fixture("rgb.png"), limits).is_err());
    let t = Temp::new();
    let path = t.0.join("bad.png");
    std::fs::write(&path, b"broken png").unwrap();
    assert!(decode_image(path, Default::default()).is_err());
    assert!(
        Mnist::open(
            fixture("labels.idx"),
            fixture("labels.idx"),
            Default::default()
        )
        .is_err()
    );
    assert!(
        decode_image(fixture("rgb.png"), Default::default())
            .unwrap()
            .crop(1, 1, 2, 2)
            .is_err()
    );
}

#[test]
fn jpeg_decodes_and_integer_masks_keep_large_category_ids() {
    let sample = decode_image(fixture("rgb.jpg"), Default::default()).unwrap();
    assert_eq!(sample.image.size(), [3, 2, 2]);
    let mut sample = decode_image(fixture("rgb.png"), Default::default()).unwrap();
    let category = (1i64 << 54) + 1;
    sample.mask = Some(Tensor::from_slice(&[category, 2, 3, 4]).reshape([2, 2]));
    let sample = sample.resize(4, 4, Default::default()).unwrap();
    assert_eq!(sample.mask.as_ref().unwrap().int64_value(&[0, 0]), category);
}

#[test]
fn vision_sample_footprint_includes_classification_label() {
    let mut sample = decode_image(fixture("rgb.png"), Default::default()).unwrap();
    let without_label = sample.resident_bytes();
    sample.label = Some(7);
    assert_eq!(
        sample.resident_bytes(),
        without_label + std::mem::size_of::<i64>()
    );
}

#[test]
fn resize_limit_covers_image_and_integer_mask_together() {
    let mut sample = decode_image(fixture("rgb.png"), Default::default())
        .unwrap()
        .to_float()
        .unwrap();
    sample.mask = Some(Tensor::zeros(
        [2, 2],
        (Kind::Int64, rusttorch_core::Device::Cpu),
    ));
    let limits = ResourceLimits {
        max_decoded_bytes: 3 * 4 * 4 * 4,
        ..Default::default()
    };
    assert!(sample.resize(4, 4, limits).is_err());
}
