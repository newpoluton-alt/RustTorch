# RustTorch Codec

Read timestamped video frames and audio blocks from local media, seek by
presentation time, and feed ordinary typed RustTorch batches. CPU decoding is
the supported baseline. The package does not download or bundle FFmpeg.

## Installation

Use `rusttorch = { version = "0.4", features = ["codec"] }` for system linking
and `rusttorch::codec`, or use the direct package:

```toml
[dependencies]
rusttorch-codec = { version = "0.4", features = ["system", "download-libtorch"] }
rusttorch-data = "0.4"
```

## Features

`system` selects an existing FFmpeg installation; `vcpkg` selects a vcpkg
installation. Choose one linking provider. Without either, the timestamp and
owned frame types remain available. `download-libtorch` obtains the tensor
runtime; `doc-only` permits documentation generation without tensor linking.

## Native runtime

All tensor operations require **LibTorch 2.13.0**. Use `download-libtorch`, or
set `LIBTORCH` to an extracted distribution and add its library directory to
`PATH` on Windows or the platform's shared-library search path.

Install FFmpeg 8 development headers/shared libraries and a C/Clang toolchain.
The `system` feature probes libraries with pkg-config. Alternatively set
`FFMPEG_LIBS_DIR`, `FFMPEG_INCLUDE_DIR`, and `FFMPEG_LINK_MODE=dynamic` to your
installation's library/include directories. Windows users may choose `vcpkg`
(and the facade's `codec-vcpkg`) with an appropriate shared-library triplet.
The runtime loader must also be able to find the FFmpeg DLLs/shared libraries.

`capabilities()` reports the linked version, configure flags and native license.
An arbitrary system build is not automatically an approved LGPL build. Official
artifact evidence requires a source-built, dynamically linked FFmpeg profile
with GPL/nonfree disabled. Native binaries never enter the crate archive.

## Example: batch video frames

```no_run
# #[cfg(any(feature="system",feature="vcpkg"))]
# fn run() -> Result<(), Box<dyn std::error::Error>> {
use rusttorch_codec::{MediaDecoder, MediaFrame, StreamKind};
use rusttorch_data::{ResourceLimits, batches};

let decoder = MediaDecoder::open("clip.mkv", StreamKind::Video, ResourceLimits::default())?;
println!("decoder: {}", decoder.stream_info().decoder);
for group in batches(decoder, 8, false)? {
    for frame in group? {
        if let MediaFrame::Video(frame) = frame {
            // RGB bytes in [3, height, width], ready for image preprocessing.
            println!("{:?} at {:?}", frame.pixels.size(), frame.timestamp);
        }
    }
}
# Ok(()) }
```

Frames are decoded lazily; batches retain only their owned frame values.
Different resolutions remain separate until you explicitly resize/stack them.
Native decoder errors are returned once, then the iterator ends. The decoder
selects one stream kind; it does not silently combine audio and video timelines.

## Seek and inspect time

```no_run
# #[cfg(any(feature="system",feature="vcpkg"))]
# fn run() -> Result<(), Box<dyn std::error::Error>> {
use rusttorch_codec::{MediaDecoder, StreamKind};
let mut decoder = MediaDecoder::open("clip.mkv", StreamKind::Video, Default::default())?;
decoder.seek(2.5)?;
if let Some(frame) = decoder.next() {
    let frame = frame?;
    assert!(frame.timestamp().unwrap().seconds() >= 2.5);
}
# Ok(()) }
```

`TimeBase` stores a positive rational number of seconds per tick. `Timestamp`
keeps signed presentation ticks and their time base; missing source timestamps
remain `None`. Integer rescaling checks overflow. Seeking first goes to a
keyframe and drops decoded frames whose presentation time precedes the target.
Audio seeking returns whole blocks beginning at or after the target. A source
without timestamps cannot satisfy a time seek and returns an error.

## Decode audio blocks

```no_run
# #[cfg(any(feature="system",feature="vcpkg"))]
# fn run() -> Result<(), Box<dyn std::error::Error>> {
use rusttorch_codec::{MediaDecoder, MediaFrame, StreamKind};
for block in MediaDecoder::open("speech.wav", StreamKind::Audio, Default::default())? {
    if let MediaFrame::Audio(block) = block? {
        // Normalized Float [channels, frames], preserving the source rate.
        println!("{} Hz: {:?}", block.sample_rate, block.samples.size());
    }
}
# Ok(()) }
```

The native audio baseline supports mono/stereo and converts decoded samples to
Float. Unspecified mono/stereo speaker layouts use the conventional default;
other channel counts fail explicitly. For waveform resampling and spectral
features, use `rusttorch-audio` after constructing a waveform from each block.

## Safety and support

Accepted demuxers are Matroska/WebM, WAV, AVI, MOV/MP4, MP3 and FLAC; actual codec
availability depends on the linked FFmpeg build. Network protocols and playlist
demuxers are disabled. Input must be a regular local file. Keep it unchanged
while decoding. Encoded size, stream counts, dimensions, decoded payload,
frame counts and duration have finite `ResourceLimits` ceilings.

A supplied `WorkerContext` is checked between native operations. A native call
cannot be force-cancelled, and decoder internal memory is not a process-RSS
promise. The adapter does not advertise hardware decoding or exact native-codec
checkpoint restoration. Requesting a nonexistent stream returns an error.

The CPU tests decode an original tiny FFV1/PCM fixture, check RGB values and
presentation times, seek, batch frames, compare normalized audio and reject
oversize inputs. `tests/pipelines.rs` also checks rational overflow without any
native feature. docs.rs builds the public native API with upstream documentation
bindings and does not execute native decoding.

## Tested format profile

| Capability | Executable evidence |
|---|---|
| CPU FFV1 video in Matroska, RGB conversion and presentation-time seek | `cpu_video_streams_batches_and_seeks_with_timestamps` |
| Mono PCM WAV, normalized Float output | `native_audio_copies_checked_float_buffers` |
| Rational time conversion and overflow rejection | `rational_time_is_checked_and_exact` |
| Missing stream and configured ceilings | `limits_and_missing_streams_fail_explicitly` |

The reproducible project profile uses `scripts/build-ffmpeg.py` to build
checksummed FFmpeg 8.1.2 with shared libraries, GPL/nonfree disabled and only
FFV1/PCM decoders plus Matroska/WAV demuxers. Its audit JSON records the native
configuration and license. Other permitted demuxers require a broader
user-selected FFmpeg build and do not acquire test evidence automatically.
