//! Stream CPU-decoded video and audio from local media with explicit timestamps.
//!
//! Enable `system` for an existing FFmpeg 8 installation, or `vcpkg` for vcpkg
//! linking. No native libraries are downloaded or embedded in this package.
//! Without either feature, the time/frame types remain available for docs.rs.
//!
//! ```no_run
//! # #[cfg(any(feature="system",feature="vcpkg"))]
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! use rusttorch_codec::{MediaDecoder, StreamKind};
//! use rusttorch_data::{batches, ResourceLimits};
//! let decoder = MediaDecoder::open("clip.mkv", StreamKind::Video, ResourceLimits::default())?;
//! for frames in batches(decoder, 8, false)? {
//!     for frame in frames? { println!("{:?}", frame.timestamp()); }
//! }
//! # Ok(()) }
//! ```
#![deny(missing_docs)]
#![doc = include_str!("../README.md")]
use rusttorch_core::{Device, Tensor};
use rusttorch_data::{MemoryFootprint, PinMemory};
use serde::{Deserialize, Serialize};
/// Codec, resource-limit or local I/O failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CodecError {
    /// Local media file failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Invalid time, dimensions, format or requested capability.
    #[error("{0}")]
    Invalid(String),
    /// Configured input/output ceiling exceeded.
    #[error(transparent)]
    Limit(#[from] rusttorch_core::RustTorchError),
    /// LibTorch tensor construction failed.
    #[error(transparent)]
    Tensor(#[from] tch::TchError),
    /// Native decoder error, with original source retained.
    #[cfg(any(feature = "system", feature = "vcpkg"))]
    #[error(transparent)]
    Native(#[from] rsmpeg::error::RsmpegError),
}
/// Result returned by media operations.
pub type Result<T> = std::result::Result<T, CodecError>;
/// Positive rational seconds per tick, kept exact until display conversion.
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub struct TimeBase {
    numerator: u32,
    denominator: u32,
}
impl TimeBase {
    /// Validates positive numerator and denominator.
    pub fn new(numerator: u32, denominator: u32) -> Result<Self> {
        if numerator == 0 || denominator == 0 {
            return Err(CodecError::Invalid(
                "time base requires positive numerator and denominator".into(),
            ));
        }
        Ok(Self {
            numerator,
            denominator,
        })
    }
    /// Rational numerator.
    pub const fn numerator(self) -> u32 {
        self.numerator
    }
    /// Rational denominator.
    pub const fn denominator(self) -> u32 {
        self.denominator
    }
    /// Converts ticks to approximate floating-point seconds.
    pub fn seconds(self, ticks: i64) -> f64 {
        ticks as f64 * self.numerator as f64 / self.denominator as f64
    }
    /// Rescales timestamps with checked i128 arithmetic and truncation toward zero.
    pub fn rescale(self, ticks: i64, target: Self) -> Result<i64> {
        let value = (ticks as i128) * (self.numerator as i128) * (target.denominator as i128)
            / ((self.denominator as i128) * (target.numerator as i128));
        i64::try_from(value)
            .map_err(|_| CodecError::Invalid("rescaled timestamp overflows i64".into()))
    }
}
impl<'de> Deserialize<'de> for TimeBase {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Wire {
            numerator: u32,
            denominator: u32,
        }
        let w = Wire::deserialize(d)?;
        Self::new(w.numerator, w.denominator).map_err(serde::de::Error::custom)
    }
}
/// Timestamp carrying its exact rational tick unit.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Timestamp {
    /// Presentation ticks, possibly negative.
    pub ticks: i64,
    /// Seconds per tick.
    pub time_base: TimeBase,
}
impl Timestamp {
    /// Approximate presentation time in seconds.
    pub fn seconds(self) -> f64 {
        self.time_base.seconds(self.ticks)
    }
}
/// Kind of stream selected for decoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamKind {
    /// CPU RGB video.
    Video,
    /// CPU interleaved source audio, normalized to planar Float mono/stereo.
    Audio,
}
/// One decoded RGB frame.
#[derive(Debug)]
pub struct VideoFrame {
    /// RGB Uint8 CHW pixels.
    pub pixels: Tensor,
    /// Presentation time, absent when the source provides none.
    pub timestamp: Option<Timestamp>,
}
/// One decoded Float mono/stereo audio block.
#[derive(Debug)]
pub struct AudioFrame {
    /// `[channels,frames]` normalized Float samples.
    pub samples: Tensor,
    /// Frames per second.
    pub sample_rate: u32,
    /// Presentation time, absent when unknown.
    pub timestamp: Option<Timestamp>,
}
/// A typed decoded item. MediaDecoder selects exactly one stream kind.
#[derive(Debug)]
pub enum MediaFrame {
    /// RGB image.
    Video(VideoFrame),
    /// Normalized audio block.
    Audio(AudioFrame),
}
impl MediaFrame {
    /// Returns the source presentation timestamp without inventing missing values.
    pub fn timestamp(&self) -> Option<Timestamp> {
        match self {
            Self::Video(v) => v.timestamp,
            Self::Audio(a) => a.timestamp,
        }
    }
}
impl MemoryFootprint for VideoFrame {
    fn resident_bytes(&self) -> usize {
        self.pixels.resident_bytes()
    }
}
impl MemoryFootprint for AudioFrame {
    fn resident_bytes(&self) -> usize {
        self.samples.resident_bytes()
    }
}
impl MemoryFootprint for MediaFrame {
    fn resident_bytes(&self) -> usize {
        match self {
            Self::Video(v) => v.resident_bytes(),
            Self::Audio(a) => a.resident_bytes(),
        }
    }
}
impl PinMemory for VideoFrame {
    fn pin_memory(mut self, d: Device) -> rusttorch_core::Result<Self> {
        self.pixels = self.pixels.pin_memory(d)?;
        Ok(self)
    }
}
impl PinMemory for AudioFrame {
    fn pin_memory(mut self, d: Device) -> rusttorch_core::Result<Self> {
        self.samples = self.samples.pin_memory(d)?;
        Ok(self)
    }
}
impl PinMemory for MediaFrame {
    fn pin_memory(self, d: Device) -> rusttorch_core::Result<Self> {
        match self {
            Self::Video(v) => Ok(Self::Video(v.pin_memory(d)?)),
            Self::Audio(a) => Ok(Self::Audio(a.pin_memory(d)?)),
        }
    }
}
/// Selected stream metadata, without assuming a constant frame rate.
#[derive(Clone, Debug)]
pub struct StreamInfo {
    /// Zero-based container stream index.
    pub index: usize,
    /// Selected media kind.
    pub kind: StreamKind,
    /// Stream tick unit.
    pub time_base: TimeBase,
    /// Duration in ticks when available.
    pub duration: Option<i64>,
    /// Native decoder name.
    pub decoder: String,
}
/// Actual linked FFmpeg build diagnostics; configuration determines licensing.
#[derive(Clone, Debug)]
pub struct CodecCapabilities {
    /// Runtime version string.
    pub version: String,
    /// FFmpeg configure flags, without labeling arbitrary builds approved.
    pub configuration: String,
    /// Reported native license.
    pub license: String,
    /// CPU decoding baseline is available.
    pub cpu_decode: bool,
}
#[cfg(any(feature = "system", feature = "vcpkg"))]
mod native;
#[cfg(any(feature = "system", feature = "vcpkg"))]
pub use native::{MediaDecoder, capabilities};
