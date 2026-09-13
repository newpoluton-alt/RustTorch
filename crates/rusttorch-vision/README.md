# RustTorch Vision

Turn local image folders, digit datasets and detection annotations into typed
training batches. Image decoding produces RGB `Uint8` tensors in **CHW** order;
`to_float()` converts bytes to `Float` values in `[0, 1]`. Decoding does not apply
EXIF orientation, color-profile conversion or automatic normalization.

## Installation

Install through `rusttorch = { version = "0.4", features = ["vision"] }`, which
exposes `rusttorch::vision`, or use the package directly:

```toml
[dependencies]
rusttorch-vision = { version = "0.4", features = ["download-libtorch"] }
rusttorch-data = "0.4"
```

The package uses the same LibTorch runtime as the rest of RustTorch. PNG/JPEG
codecs are included only when this optional package is selected.

## Features

The `vision` facade feature enables PNG/JPEG decoding, local dataset readers
and target-aware image transforms. The direct package has no default features.
`download-libtorch` obtains the tensor runtime; `doc-only` builds documentation
without native linking and cannot execute tensor operations.

## Native runtime

Tensor operations require **LibTorch 2.13.0**. Enable `download-libtorch`, or set
`LIBTORCH` to an extracted distribution and add its library directory to `PATH`
on Windows or the platform's shared-library search path. All RustTorch packages
in one application must share the same runtime. Documentation can be checked
with `cargo doc -p rusttorch-vision --no-default-features --features doc-only`.

## Example: Train from class folders

Arrange inputs as `train/cats/a.jpg`, `train/dogs/b.png`, and so on. Class names
and file paths are sorted, so label assignments are stable. Nested directories
inside a class are supported. Empty classes, symlinks and unsupported files
produce errors. Keep the dataset files unchanged during an epoch.

```no_run
use std::num::NonZeroUsize;
use rusttorch_data::{DataLoader, FnTransform, ResourceLimits, TaskContext};
use rusttorch_vision::{ImageFolder, VisionSample};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let limits = ResourceLimits::default();
    let dataset = ImageFolder::open("train", limits)?;
    println!("label names: {:?}", dataset.classes());
    let mut loader = DataLoader::builder(dataset)
        .shuffle(42)?
        .batch_size(16)
        .workers(2)
        .prefetch_bytes(NonZeroUsize::new(32 * 1024 * 1024).unwrap())
        .transform(FnTransform::new(move |image: VisionSample, _: &TaskContext| {
            image.resize(224, 224, limits)?.to_float()
        }))
        .build()?;

    for batch in loader.iter() {
        let batch = batch?;
        // Pass batch.images to an image model. The final batch may be smaller.
        assert_eq!(&batch.images.size()[1..], &[3, 224, 224]);
        assert_eq!(batch.labels.len(), batch.images.size()[0] as usize);
    }
    Ok(())
}
```

The default collator returns `VisionBatch`: a stacked image tensor plus ordered
per-image labels, boxes, keypoints and optional masks. Resize differing input
sizes before stacking. For variable-size detection images, explicitly select
`rusttorch_data::VecCollate` and process each owned sample separately.

## Keep detection targets aligned

Rectangles use continuous **XYXY pixel-edge coordinates**. Cropping clips boxes
and removes empty boxes; keypoints outside the crop become absent. Resizing
uses bilinear image interpolation and nearest-neighbor mask indexing, preserving
integer category IDs exactly. Transformations consume the sample and keep all
its targets together.

```no_run
use rusttorch_data::{Dataset, ResourceLimits};
use rusttorch_vision::CocoDetection;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let limits = ResourceLimits::default();
    let dataset = CocoDetection::open("images", "annotations.json", limits)?;
    let sample = dataset.get(0)?.resize(256, 256, limits)?;
    let sample = sample.crop(16, 16, 224, 224)?.to_float()?;
    println!("{} objects", sample.boxes.len());
    Ok(())
}
```

`horizontal_flip()` updates coordinates, boxes and masks. It preserves keypoint
order; a pose model with named left/right joints must additionally permute those
joint entries according to its own skeleton. COCO category IDs are preserved,
not automatically made contiguous. The COCO reader supports bounding boxes and
keypoints; polygon/RLE segmentation returns an explicit unsupported-data error.
Attach an integer `[H,W]` mask to `VisionSample.mask` for mask transforms.

## Read local MNIST and CIFAR records

```no_run
use rusttorch_data::{Dataset, ResourceLimits};
use rusttorch_vision::{Cifar, Mnist};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let limits = ResourceLimits::default();
    let digits = Mnist::open("train-images-idx3-ubyte", "train-labels-idx1-ubyte", limits)?;
    let digit = digits.get(0)?.to_float()?;
    assert_eq!(digit.image.size()[0], 3); // Grayscale repeated into RGB channels.

    let images = Cifar::open("data_batch_1.bin", false, limits)?;
    let first = images.get(0)?;
    assert_eq!(first.image.size(), [3, 32, 32]);
    // Pass true to read CIFAR-100 binary records and return the fine label.
    Ok(())
}
```

IDX inputs must be uncompressed and have matching header counts and exact file
sizes. CIFAR inputs must contain complete binary records. Both read one sample
at a time, validate label ranges and work with the same loader as image folders.
No dataset download or archive extraction happens implicitly.

## Supported formats and limits

| Input or operation | Contract | Executable evidence |
|---|---|---|
| PNG/JPEG | RGB byte decoding, explicit float conversion | `tests/pipelines.rs` |
| ImageFolder | Stable classes/paths, lazy worker decoding | `imagefolder_workers_decode_classify_and_bound_payload` |
| MNIST/CIFAR-10/CIFAR-100 | Local IDX/binary samples and labels | `named_dataset_families_read_real_local_formats` |
| COCO detection | Local boxes/keypoints and safe-root paths | `named_dataset_families_read_real_local_formats` |
| Resize/crop/flip | Image, boxes, keypoints and integer masks | `target_geometry_tracks_image_mask_and_keypoints` |

`ResourceLimits` checks encoded bytes, image dimensions, tensor elements,
decoded payload and record counts. Decoder scratch allocations belong to the
image library; these limits are not a process-RSS guarantee. Loader
`prefetch_bytes` bounds retained sample payload independently. Existing direct
`rusttorch::data` paths are unchanged; this package adds optional APIs.
