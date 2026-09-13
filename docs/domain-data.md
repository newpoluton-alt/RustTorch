# Build training batches from images, audio, text and tables

Choose a domain package to decode or prepare a sample, then use the same
RustTorch DataLoader for sampling, transformations, batching and workers. A
sample retains useful metadata: image targets, audio rates, token masks or
fitted table columns. Your training loop receives owned tensors with an explicit
shape and meaning.

## Choose the inputs your application uses

```toml
[dependencies]
rusttorch = { version = "0.4", features = ["vision", "audio", "text", "tabular", "download-libtorch"] }
```

All tensor packages use LibTorch **2.13.0**. Install it through
`download-libtorch` or configure an existing distribution with `LIBTORCH` and
the platform's library search path. These domain features are optional; the
basic RustTorch facade does not select their decoding dependencies.

| Input | Facade feature and module | Model-facing result | Full use cases |
|---|---|---|---|
| PNG/JPEG, ImageFolder, MNIST, CIFAR, COCO detection | `vision`, `rusttorch::vision` | RGB `[B,3,H,W]` with labels and per-image targets | [Vision guide](https://docs.rs/rusttorch-vision/latest/rusttorch_vision/) |
| Local video or compressed media | `codec`, `rusttorch::codec` | Owned timestamped RGB frames or audio blocks | [Codec guide](https://docs.rs/rusttorch-codec/latest/rusttorch_codec/) |
| WAV/FLAC, waveform augmentation and spectral features | `audio`, `rusttorch::audio` | Waveforms `[B,C,T]`, lengths and sample rate | [Audio guide](https://docs.rs/rusttorch-audio/latest/rusttorch_audio/) |
| Local tokenizer vocabularies and text | `text`, `rusttorch::text` | Int64 `[B,L]` tokens, attention and special-token masks | [Text guide](https://docs.rs/rusttorch-text/latest/rusttorch_text/) |
| CSV/JSONL and fitted preprocessing | `tabular`, `rusttorch::tabular` | Float numeric and Int64 categorical feature matrices | [Tabular guide](https://docs.rs/rusttorch-tabular/latest/rusttorch_tabular/) |
| Arrow IPC files and Parquet | `columnar`, `rusttorch::tabular` | The same table records and preprocessing | [Columnar readers](https://docs.rs/rusttorch-tabular/latest/rusttorch_tabular/) |

Direct crate names are `rusttorch-vision`, `rusttorch-codec`, `rusttorch-audio`,
`rusttorch-text` and `rusttorch-tabular`. Direct columnar consumers choose
`arrow` or `parquet`. The `full` facade feature enables all domains, including
system FFmpeg linking; install its native prerequisites before enabling it.

## Train an image classifier

Use one subdirectory per class, such as `train/cats/a.jpg` and
`train/dogs/b.png`. The reader sorts class names and file paths, so label IDs
stay stable. A transform prepares each sample before the loader stacks a batch.

```no_run
# #[cfg(feature = "vision")]
# fn run() -> Result<(), Box<dyn std::error::Error>> {
use rusttorch::data::{DataLoader, FnTransform, ResourceLimits, TaskContext};
use rusttorch::vision::{ImageFolder, VisionSample};
use std::num::NonZeroUsize;

let limits = ResourceLimits::default();
let images = ImageFolder::open("train", limits)?;
let mut loader = DataLoader::builder(images)
    .shuffle(42)?
    .batch_size(16)
    .workers(2)
    .prefetch_bytes(NonZeroUsize::new(32 * 1024 * 1024).unwrap())
    .transform(FnTransform::new(move |sample: VisionSample, _: &TaskContext| {
        sample.resize(224, 224, limits)?.to_float()
    }))
    .build()?;
for batch in loader.iter() {
    let batch = batch?;
    assert_eq!(&batch.images.size()[1..], &[3, 224, 224]);
    // Feed batch.images into your classifier and use batch.labels as targets.
}
# Ok(()) }
```

For detection, `CocoDetection` returns boxes and keypoints alongside each image.
Resize, crop and flip keep those targets aligned. Integer masks use nearest
indices so category IDs are preserved exactly. For variable-size images, select
`VecCollate` and let your model process the samples separately. The vision guide
also covers local MNIST and CIFAR binary records without implicit downloads.

## Preserve sequence lengths and fitted statistics

Audio and text batches often contain different sequence lengths. `PadAudio`
returns padded waveforms and original frame counts; `TextCollator` returns token
IDs and explicit attention masks. Use these lengths or masks in your model's
loss so padding does not contribute. Audio sample rate remains explicit;
resample recordings deliberately before collating different rates.

For text, measure token lengths with the same tokenizer and truncation policy
that you use while loading. `TokenBudgetSampler` limits padded work using
`max_sequence_length * batch_size`. Supply your model's actual padding token
ID. Truncation occurs after special-token insertion and can remove a closing
special token, so choose the tokenizer/model policy deliberately. Distributed
sampling may produce unequal batch counts across ranks even when sample counts
match; the training loop must handle that case.

For tables, fit `FittedPreprocessor` once using the training split. Save it with
`to_json()` and restore it with `from_json()` for validation or inference.
Numeric columns use training means and population standard deviations;
categorical IDs are sorted training values with an optional unknown ID of zero.
Transforming validation rows never changes that fitted state. CSV quoting and
newlines, JSONL scalar cells, Arrow nulls and supported Parquet columns all feed
the same typed row interface.

## Bound inputs and worker queues separately

```rust
use rusttorch::data::ResourceLimits;
let limits = ResourceLimits {
    max_encoded_bytes: 8 * 1024 * 1024,
    max_decoded_bytes: 32 * 1024 * 1024,
    max_image_dimension: 4096,
    max_tokens: 8192,
    ..ResourceLimits::default()
};
assert!(limits.tensor_elements(&[usize::MAX, 2]).is_err());
```

The defaults are finite: encoded input 64 MiB, decoded sample payload 256 MiB,
64 million tensor elements, image dimensions 16,384, one million records or
tokens, 4,096 columns, 1 MiB strings, 256 MiB columnar batches/row groups,
one hour of media and 32 streams. Explicit `unlimited()` disables those logical
ceilings while preserving integer-overflow checks.

Metadata checks happen before large output allocation wherever the format
allows. Compressed decoders and columnar readers still own internal scratch
allocations; these limits do not guarantee a process memory ceiling. Domain
samples implement `MemoryFootprint` for the loader's separate `prefetch_bytes`
budget. Set both an input/output ceiling and a queue budget when loading large
samples. Text and audio collators also check padded output and length metadata;
even an empty sequence consumes a record and length entry. Image resize checks
the combined image and mask output, and fitted vocabularies have an aggregate
byte ceiling. Keep local input files unchanged throughout an epoch.

## Enable timestamped native video

`codec` selects system FFmpeg 8; `codec-vcpkg` selects a vcpkg installation,
including shared-library Windows triplets. Install development headers and
libraries plus Clang. System discovery uses pkg-config, or explicit
`FFMPEG_LIBS_DIR`, `FFMPEG_INCLUDE_DIR` and `FFMPEG_LINK_MODE=dynamic`. Add the
library directory to `PATH` on Windows or the platform's shared-library path.
No native binaries are embedded in RustTorch crate archives.

```no_run
# #[cfg(any(feature = "codec", feature = "codec-vcpkg"))]
# fn run() -> Result<(), Box<dyn std::error::Error>> {
use rusttorch::codec::{MediaDecoder, MediaFrame, StreamKind};
use rusttorch::data::{ResourceLimits, batches};
let mut video = MediaDecoder::open("clip.mkv", StreamKind::Video, ResourceLimits::default())?;
video.seek(2.5)?;
for batch in batches(video, 8, false)? {
    for frame in batch? {
        if let MediaFrame::Video(frame) = frame {
            println!("{:?} at {:?}", frame.pixels.size(), frame.timestamp);
        }
    }
}
# Ok(()) }
```

The baseline decodes one selected CPU stream. Seeking removes decoded frames
before the requested presentation time. Audio seeks yield whole blocks whose
start time reaches the target. A missing timestamp cannot satisfy a time seek.
Native errors are yielded once, then the iterator ends. Cancellation is checked
between native calls; an in-progress native call is not forcibly interrupted.

The reproducible native test profile is built by
[`scripts/build-ffmpeg.py`](https://github.com/newpoluton-alt/RustTorch/blob/main/scripts/build-ffmpeg.py)
from checksummed FFmpeg **8.1.2** source. It uses shared libraries, disables GPL,
nonfree, network and automatic external-library detection, and enables only
FFV1/PCM decoding, Matroska/WAV demuxing and local files. Run it with
`python3 scripts/build-ffmpeg.py --prefix /absolute/new/ffmpeg-prefix` on Linux or
macOS; the resulting `rusttorch-build-audit.json` records every linked library's
version, license and configure flags. This minimal test profile does not claim
that other codecs are available. `capabilities()` describes the actual selected
native build. Applications may select a broader build and are responsible for
that build's native licensing obligations.

For documentation without installed tensor or FFmpeg libraries, set `DOCS_RS=1`
and use `cargo doc -p rusttorch-codec --no-default-features --features doc-only,system`.
This checks the public native API using upstream bundled declarations and does
not execute decoding. The docs.rs service uses this documentation-only path.

## What the examples cover

Each package includes runnable loader examples and checked-in tests using tiny
original fixtures. The tests cover real image formats and dataset headers,
audio PCM/FLAC values, FFV1 frame times and seeks, token masks and sampler resume,
and CSV/JSONL/Arrow/Parquet equivalence. Audio spectral values and gradients are
checked against a pinned numerical reference. The package compatibility pages
state the tested scope and intentional limits. They do not claim every API in
external vision, audio, tokenizer or dataframe ecosystems.
