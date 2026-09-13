//! Decode local audio, resample on CPU, and compute differentiable spectral
//! features with LibTorch. Samples use floating-point `[channels, frames]`.
//!
//! ```
//! use rusttorch_audio::{Waveform, SpectrogramConfig};
//! use rusttorch_core::Tensor;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let wave = Waveform::new(Tensor::from_slice(&[0f32; 16]).reshape([1,16]), 8000, 0.)?;
//! let power = wave.spectrogram(SpectrogramConfig { n_fft: 8, hop_length: 4, power: 2. }, Default::default())?;
//! assert_eq!(power.size(), [1,5,3]);
//! # Ok(()) }
//! ```
#![deny(missing_docs)]
#![doc = include_str!("../README.md")]
use rand::Rng;
use rubato::Resampler;
/// Optional native FFmpeg adapter for formats outside the Rust decoder profile.
#[cfg(feature = "codec")]
pub use rusttorch_codec as codec;
use rusttorch_core::{Device, Kind, Tensor};
use rusttorch_data::{Collate, Dataset, MemoryFootprint, PinMemory, ResourceLimits, TaskContext};
use std::{
    fs::File,
    io::Cursor,
    path::{Path, PathBuf},
};
use symphonia::core::{
    codecs::audio::AudioDecoderOptions,
    formats::{FormatOptions, TrackType, probe::Hint},
    io::MediaSourceStream,
    meta::MetadataOptions,
};
/// Audio decoding, configuration, resampling or tensor error.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AudioError {
    /// Local input failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Preserved Symphonia error.
    #[error(transparent)]
    Decode(#[from] symphonia::core::errors::Error),
    /// Invalid format, dimensions or transform parameters.
    #[error("{0}")]
    Invalid(String),
    /// Resource ceiling or shared runtime error.
    #[error(transparent)]
    Limit(#[from] rusttorch_core::RustTorchError),
    /// LibTorch operation failed.
    #[error(transparent)]
    Tensor(#[from] tch::TchError),
    /// Rubato rejected resampler construction.
    #[error(transparent)]
    Resampler(#[from] rubato::ResamplerConstructionError),
    /// Rubato rejected buffer processing.
    #[error(transparent)]
    Resample(#[from] rubato::ResampleError),
}
/// Result returned by audio operations.
pub type Result<T> = std::result::Result<T, AudioError>;
/// Channel ordering carried with decoded samples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChannelLayout {
    /// One channel.
    Mono,
    /// Conventional left/right channels.
    Stereo,
    /// Ordered channels without a claimed speaker-position mapping.
    Discrete(usize),
}
/// Owned waveform metadata and floating-point samples.
#[derive(Debug)]
pub struct Waveform {
    /// Float/Double tensor in `[channels,frames]`, usually normalized to `[-1,1]`.
    pub samples: Tensor,
    /// Frames per second; nonzero.
    pub sample_rate: u32,
    /// Channel ordering. Multi-channel decoding uses `Discrete` unless known.
    pub channel_layout: ChannelLayout,
    /// Time of frame zero in seconds, finite and nonnegative.
    pub time_origin: f64,
}
impl Waveform {
    /// Validates a floating-point waveform and infers mono/stereo channel layout.
    pub fn new(samples: Tensor, sample_rate: u32, time_origin: f64) -> Result<Self> {
        let c = samples.size().first().copied().unwrap_or(0);
        let channel_layout = match c {
            1 => ChannelLayout::Mono,
            2 => ChannelLayout::Stereo,
            _ => ChannelLayout::Discrete(c.max(0) as usize),
        };
        let wave = Self {
            samples,
            sample_rate,
            channel_layout,
            time_origin,
        };
        wave.validate()?;
        Ok(wave)
    }
    /// Validates shape, rate, channel count and finite time origin.
    pub fn validate(&self) -> Result<()> {
        let s = self.samples.size();
        let c = match self.channel_layout {
            ChannelLayout::Mono => 1,
            ChannelLayout::Stereo => 2,
            ChannelLayout::Discrete(c) => c,
        };
        if s.len() != 2
            || s[0] <= 0
            || s[1] < 0
            || s[0] as usize != c
            || !matches!(self.samples.kind(), Kind::Float | Kind::Double)
            || self.sample_rate == 0
            || !self.time_origin.is_finite()
            || self.time_origin < 0.
        {
            return Err(AudioError::Invalid("expected finite-origin Float/Double [channels,frames], positive rate and matching channel layout".into()));
        }
        Ok(())
    }
    /// Number of sample frames per channel.
    pub fn frames(&self) -> usize {
        self.samples.size().get(1).copied().unwrap_or(0).max(0) as usize
    }
    /// Duration in seconds, excluding the time origin.
    pub fn duration(&self) -> f64 {
        self.frames() as f64 / self.sample_rate as f64
    }
    /// Applies finite decibel gain without mutating the original tensor storage.
    pub fn gain(mut self, decibels: f64) -> Result<Self> {
        self.validate()?;
        let factor = 10f64.powf(decibels / 20.);
        if !decibels.is_finite() || !factor.is_finite() || (factor as f32).is_infinite() {
            return Err(AudioError::Invalid(
                "gain must have finite representable amplitude".into(),
            ));
        }
        self.samples = self.samples.f_mul_scalar(factor)?;
        Ok(self)
    }
    /// Adds deterministic uniform noise in `[-amplitude, amplitude]` using task
    /// randomness. Does not read or reseed LibTorch's global random generator.
    pub fn add_noise(
        mut self,
        amplitude: f32,
        context: &TaskContext,
        limits: ResourceLimits,
    ) -> Result<Self> {
        self.validate()?;
        if !amplitude.is_finite() || amplitude < 0. {
            return Err(AudioError::Invalid(
                "noise amplitude must be finite and nonnegative".into(),
            ));
        }
        context
            .check()
            .map_err(|_| AudioError::Invalid("audio transform cancelled".into()))?;
        check_wave(
            self.samples.size()[0] as usize,
            self.frames(),
            self.sample_rate,
            limits,
        )?;
        let n = limits.tensor_elements(&[self.samples.size()[0] as usize, self.frames()])?;
        let mut rng = context.rng();
        let noise = (0..n)
            .map(|_| {
                if amplitude == 0. {
                    0.
                } else {
                    rng.gen_range(-1f32..=1f32) * amplitude
                }
            })
            .collect::<Vec<_>>();
        self.samples = self.samples.f_add(
            &Tensor::f_from_slice(&noise)?
                .f_reshape(self.samples.size())?
                .f_to_kind(self.samples.kind())?
                .f_to_device(self.samples.device())?,
        )?;
        Ok(self)
    }
    /// Shifts by signed frames with zero fill, preserving duration and time origin.
    pub fn time_shift(mut self, frames: i64) -> Result<Self> {
        self.validate()?;
        let n = self.frames() as i64;
        let shift = frames.unsigned_abs().min(n as u64) as i64;
        if shift == n {
            self.samples = self.samples.f_zeros_like()?;
        } else if frames >= 0 {
            self.samples = self
                .samples
                .f_narrow(1, 0, n - shift)?
                .f_constant_pad_nd([shift, 0])?;
        } else {
            self.samples = self
                .samples
                .f_narrow(1, shift, n - shift)?
                .f_constant_pad_nd([0, shift])?;
        }
        Ok(self)
    }
    /// Mixes aligned waveforms with a finite scale. Rate, shape, layout, origin,
    /// dtype and device must match; there is no implicit broadcast or resampling.
    pub fn mix(mut self, other: &Self, scale: f64) -> Result<Self> {
        self.validate()?;
        other.validate()?;
        if !scale.is_finite()
            || self.sample_rate != other.sample_rate
            || self.channel_layout != other.channel_layout
            || self.time_origin != other.time_origin
            || self.samples.size() != other.samples.size()
            || self.samples.kind() != other.samples.kind()
            || self.samples.device() != other.samples.device()
        {
            return Err(AudioError::Invalid(
                "mix requires aligned waveforms and finite scale".into(),
            ));
        }
        self.samples = self.samples.f_add(&other.samples.f_mul_scalar(scale)?)?;
        Ok(self)
    }
    /// Resamples on CPU using Rubato's windowed-sinc interpolation, removing its
    /// startup delay. This host conversion is not differentiable. Output has
    /// `ceil(input_frames * new_rate / old_rate)` frames and preserves origin.
    pub fn resample(self, new_rate: u32, limits: ResourceLimits) -> Result<Self> {
        self.validate()?;
        if new_rate == 0 {
            return Err(AudioError::Invalid("resample rate must be positive".into()));
        }
        if self.samples.device() != Device::Cpu {
            return Err(AudioError::Invalid(
                "Rubato resampling requires a CPU waveform".into(),
            ));
        }
        if self.samples.requires_grad() {
            return Err(AudioError::Invalid(
                "Rubato resampling cannot preserve autograd; detach explicitly".into(),
            ));
        }
        if new_rate == self.sample_rate {
            return Ok(self);
        }
        let channels = self.samples.size()[0] as usize;
        let frames = self.frames();
        check_wave(channels, frames, self.sample_rate, limits)?;
        let expected = ((frames as u128) * (new_rate as u128)).div_ceil(self.sample_rate as u128);
        let expected = usize::try_from(expected)
            .map_err(|_| AudioError::Invalid("resampled length overflow".into()))?;
        check_wave(channels, expected, new_rate, limits)?;
        if frames == 0 {
            return Self::new(self.samples, new_rate, self.time_origin);
        }
        let mut resampler = rubato::Async::<f64>::new_sinc(
            new_rate as f64 / self.sample_rate as f64,
            1.,
            &rubato::SincInterpolationParameters::default(),
            1024,
            channels,
            rubato::FixedAsync::Input,
        )?;
        let needed = resampler.process_all_needed_output_len(frames);
        check_wave(channels, needed, new_rate, limits)?;
        let data = Vec::<f64>::try_from(
            &self
                .samples
                .f_to_kind(Kind::Double)?
                .f_transpose(0, 1)?
                .f_contiguous()?
                .f_view([-1])?,
        )?;
        let input = audioadapter_buffers::owned::InterleavedOwned::new_from(data, channels, frames)
            .map_err(|e| AudioError::Invalid(e.to_string()))?;
        let output = resampler.process_all(&input, frames, None)?.take_data();
        if output.len() != expected * channels {
            return Err(AudioError::Invalid(
                "Rubato returned unexpected output length".into(),
            ));
        }
        let samples = Tensor::f_from_slice(&output)?
            .f_reshape([expected as i64, channels as i64])?
            .f_transpose(0, 1)?
            .f_to_kind(self.samples.kind())?;
        let mut value = Self::new(samples, new_rate, self.time_origin)?;
        value.channel_layout = self.channel_layout;
        Ok(value)
    }
    /// Computes one-sided periodic-Hann STFT magnitudes raised to `power`.
    /// Frames are uncentered with no padding: only complete windows are emitted.
    /// The result is `[channels, n_fft/2+1, windows]` and retains autograd.
    pub fn spectrogram(&self, config: SpectrogramConfig, limits: ResourceLimits) -> Result<Tensor> {
        self.validate()?;
        config.validate()?;
        let frames = self.frames();
        if frames < config.n_fft {
            return Err(AudioError::Invalid(
                "waveform shorter than FFT window".into(),
            ));
        }
        let windows = 1 + (frames - config.n_fft) / config.hop_length;
        let n = limits.tensor_elements(&[
            self.samples.size()[0] as usize,
            config.n_fft / 2 + 1,
            windows,
        ])?;
        limits.check(
            "spectrogram bytes",
            n.checked_mul(16)
                .ok_or_else(|| AudioError::Invalid("spectrogram size overflow".into()))?,
            limits.max_decoded_bytes,
        )?;
        let window = Tensor::f_hann_window(
            config.n_fft as i64,
            (self.samples.kind(), self.samples.device()),
        )?;
        Ok(self
            .samples
            .f_stft_center(
                config.n_fft as i64,
                config.hop_length as i64,
                config.n_fft as i64,
                Some(&window),
                false,
                "reflect",
                false,
                true,
                true,
                false,
            )?
            .f_abs()?
            .f_pow_tensor_scalar(config.power)?)
    }
    /// Projects the spectrogram through HTK triangular Mel filters. Frequencies
    /// span `[0, sample_rate/2]`, with no Slaney area normalization.
    pub fn mel_spectrogram(
        &self,
        config: SpectrogramConfig,
        n_mels: usize,
        limits: ResourceLimits,
    ) -> Result<Tensor> {
        let spectrum = self.spectrogram(config, limits)?;
        let output = limits.tensor_elements(&[
            self.samples.size()[0] as usize,
            n_mels,
            spectrum.size()[2] as usize,
        ])?;
        limits.check(
            "Mel output bytes",
            output
                .checked_mul(8)
                .ok_or_else(|| AudioError::Invalid("Mel output size overflow".into()))?,
            limits.max_decoded_bytes,
        )?;
        let filter = mel_filter_bank(config.n_fft, self.sample_rate, n_mels, limits)?
            .f_to_kind(self.samples.kind())?
            .f_to_device(self.samples.device())?;
        Ok(filter.f_matmul(&spectrum)?)
    }
    /// Computes orthonormal DCT-II of natural-log Mel energies with floor `1e-10`.
    /// This is the explicitly named log-Mel convention, without dB/top-db scaling.
    pub fn mfcc(
        &self,
        config: SpectrogramConfig,
        n_mels: usize,
        n_mfcc: usize,
        limits: ResourceLimits,
    ) -> Result<Tensor> {
        if n_mfcc == 0 || n_mfcc > n_mels {
            return Err(AudioError::Invalid(
                "MFCC count must be in 1..=n_mels".into(),
            ));
        }
        let coefficients = limits.tensor_elements(&[n_mfcc, n_mels])?;
        limits.check(
            "DCT bytes",
            coefficients
                .checked_mul(8)
                .ok_or_else(|| AudioError::Invalid("DCT size overflow".into()))?,
            limits.max_decoded_bytes,
        )?;
        let mel = self
            .mel_spectrogram(config, n_mels, limits)?
            .f_clamp_min(1e-10)?
            .f_log()?;
        let mut dct = Vec::with_capacity(n_mfcc * n_mels);
        for k in 0..n_mfcc {
            for n in 0..n_mels {
                let scale = if k == 0 {
                    (1. / n_mels as f64).sqrt()
                } else {
                    (2. / n_mels as f64).sqrt()
                };
                dct.push(
                    scale
                        * (std::f64::consts::PI * (n as f64 + 0.5) * k as f64 / n_mels as f64)
                            .cos(),
                );
            }
        }
        Ok(Tensor::f_from_slice(&dct)?
            .f_reshape([n_mfcc as i64, n_mels as i64])?
            .f_to_kind(self.samples.kind())?
            .f_to_device(self.samples.device())?
            .f_matmul(&mel)?)
    }
}
fn check_wave(channels: usize, frames: usize, rate: u32, limits: ResourceLimits) -> Result<()> {
    if channels == 0 || channels > 64 || rate == 0 {
        return Err(AudioError::Invalid(
            "expected 1..=64 channels and positive sample rate".into(),
        ));
    }
    let n = limits.tensor_elements(&[channels, frames])?;
    limits.check(
        "decoded waveform bytes",
        n.checked_mul(8)
            .ok_or_else(|| AudioError::Invalid("waveform byte overflow".into()))?,
        limits.max_decoded_bytes,
    )?;
    if frames as u128 > (rate as u128) * (limits.max_media_seconds as u128) {
        return Err(AudioError::Invalid(
            "audio duration exceeds resource limit".into(),
        ));
    }
    Ok(())
}
/// Uncentered STFT configuration, with periodic Hann window and no normalization.
#[derive(Clone, Copy, Debug)]
pub struct SpectrogramConfig {
    /// FFT/window length, at least two.
    pub n_fft: usize,
    /// Hop length, positive.
    pub hop_length: usize,
    /// Positive finite magnitude exponent, typically 1 or 2.
    pub power: f64,
}
impl Default for SpectrogramConfig {
    fn default() -> Self {
        Self {
            n_fft: 400,
            hop_length: 200,
            power: 2.,
        }
    }
}
impl SpectrogramConfig {
    fn validate(&self) -> Result<()> {
        if self.n_fft < 2
            || self.n_fft > i64::MAX as usize
            || self.hop_length == 0
            || self.hop_length > i64::MAX as usize
            || !self.power.is_finite()
            || self.power <= 0.
        {
            return Err(AudioError::Invalid("invalid FFT size, hop or power".into()));
        }
        Ok(())
    }
}
/// Builds Double `[n_mels,n_fft/2+1]` HTK triangular filters on CPU.
pub fn mel_filter_bank(
    n_fft: usize,
    sample_rate: u32,
    n_mels: usize,
    limits: ResourceLimits,
) -> Result<Tensor> {
    if n_fft < 2 || sample_rate == 0 || n_mels == 0 {
        return Err(AudioError::Invalid(
            "positive mel dimensions and rate required".into(),
        ));
    }
    let bins = n_fft / 2 + 1;
    let n = limits.tensor_elements(&[n_mels, bins])?;
    limits.check(
        "Mel filter bytes",
        n.checked_mul(8)
            .ok_or_else(|| AudioError::Invalid("Mel filter byte overflow".into()))?,
        limits.max_decoded_bytes,
    )?;
    let max_mel = 2595. * (1. + (sample_rate as f64 / 2.) / 700.).log10();
    let points = (0..n_mels + 2)
        .map(|i| 700. * (10f64.powf((i as f64 * max_mel / (n_mels + 1) as f64) / 2595.) - 1.))
        .collect::<Vec<_>>();
    let mut values = Vec::with_capacity(n);
    for m in 0..n_mels {
        for k in 0..bins {
            let hz = k as f64 * sample_rate as f64 / n_fft as f64;
            let up = (hz - points[m]) / (points[m + 1] - points[m]);
            let down = (points[m + 2] - hz) / (points[m + 2] - points[m + 1]);
            values.push(up.min(down).max(0.));
        }
    }
    Ok(Tensor::f_from_slice(&values)?.f_reshape([n_mels as i64, bins as i64])?)
}
/// Masks a contiguous frequency/time axis with a finite constant, without
/// changing the input's storage. Axis and interval must be in bounds.
pub fn mask_axis(
    input: &Tensor,
    axis: usize,
    start: usize,
    width: usize,
    value: f64,
) -> Result<Tensor> {
    let shape = input.size();
    if axis >= shape.len()
        || start > shape[axis] as usize
        || width > shape[axis] as usize - start
        || !value.is_finite()
    {
        return Err(AudioError::Invalid(
            "invalid mask axis, interval or value".into(),
        ));
    }
    let mut output = input.f_empty_like()?;
    output.f_copy_(input)?;
    let _ = output
        .f_narrow(axis as i64, start as i64, width as i64)?
        .f_fill_(value)?;
    Ok(output)
}
/// Decodes bounded local WAV/FLAC through Symphonia into CPU Float planar audio.
/// Decoder errors are returned, never skipped; sample-rate/layout changes fail.
pub fn decode_audio(path: impl AsRef<Path>, limits: ResourceLimits) -> Result<Waveform> {
    let bytes = limits.read_encoded(File::open(path.as_ref())?)?;
    let mss = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let mut format = symphonia::default::get_probe().probe(
        &Hint::new(),
        mss,
        FormatOptions::default(),
        MetadataOptions::default(),
    )?;
    limits.check("audio streams", format.tracks().len(), limits.max_streams)?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| AudioError::Invalid("no audio track".into()))?;
    let id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| AudioError::Invalid("missing audio codec parameters".into()))?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())?;
    let mut data = Vec::<f32>::new();
    let mut spec = None;
    while let Some(packet) = format.next_packet()? {
        if packet.track_id != id {
            continue;
        }
        let decoded = decoder.decode(&packet)?;
        let rate = decoded.spec().rate();
        let channels = decoded.spec().channels().count();
        check_wave(channels, 0, rate, limits)?;
        if let Some(previous) = spec {
            if previous != (rate, channels) {
                return Err(AudioError::Invalid(
                    "audio rate or channel count changes within stream".into(),
                ));
            }
        } else {
            spec = Some((rate, channels));
        }
        let next = data
            .len()
            .checked_add(decoded.samples_interleaved())
            .ok_or_else(|| AudioError::Invalid("decoded sample count overflow".into()))?;
        check_wave(channels, next / channels, rate, limits)?;
        let old = data.len();
        data.resize(next, 0.);
        decoded.copy_to_slice_interleaved(&mut data[old..]);
    }
    let (rate, channels) =
        spec.ok_or_else(|| AudioError::Invalid("audio contains no decoded frames".into()))?;
    let frames = data.len() / channels;
    Waveform::new(
        Tensor::f_from_slice(&data)?
            .f_reshape([frames as i64, channels as i64])?
            .f_transpose(0, 1)?,
        rate,
        0.,
    )
}
/// Local audio files decoded independently by loader workers.
pub struct AudioFiles {
    paths: Vec<PathBuf>,
    limits: ResourceLimits,
}
impl AudioFiles {
    /// Creates a bounded file list. Files are opened lazily in `get`.
    pub fn new(paths: Vec<PathBuf>, limits: ResourceLimits) -> Result<Self> {
        limits.check("audio records", paths.len(), limits.max_records)?;
        Ok(Self { paths, limits })
    }
}
impl Dataset for AudioFiles {
    fn get_batch_with_context(
        &self,
        indices: &[usize],
        context: &rusttorch_data::WorkerContext,
    ) -> Result<Vec<Self::Sample>> {
        indices
            .iter()
            .map(|&index| {
                context.check().map_err(|_| {
                    AudioError::Invalid("sample loading cancelled or timed out".into())
                })?;
                self.get(index)
            })
            .collect()
    }

    type Sample = Waveform;
    type Error = AudioError;
    fn len(&self) -> usize {
        self.paths.len()
    }
    fn get(&self, index: usize) -> Result<Waveform> {
        decode_audio(
            self.paths
                .get(index)
                .ok_or_else(|| AudioError::Invalid("audio index out of range".into()))?,
            self.limits,
        )
    }
}
impl MemoryFootprint for Waveform {
    fn resident_bytes(&self) -> usize {
        self.samples
            .resident_bytes()
            .saturating_add(std::mem::size_of::<Self>())
    }
}
impl PinMemory for Waveform {
    fn pin_memory(mut self, d: Device) -> rusttorch_core::Result<Self> {
        self.samples = self.samples.pin_memory(d)?;
        Ok(self)
    }
}
/// Zero-padded equal-rate waveform batch.
#[derive(Debug)]
pub struct AudioBatch {
    /// `[batch,channels,max_frames]` Float/Double audio.
    pub samples: Tensor,
    /// True for real frames, shape `[batch,max_frames]`.
    pub mask: Tensor,
    /// Original frames per sample.
    pub lengths: Vec<usize>,
    /// Common frame rate.
    pub sample_rate: u32,
}
impl MemoryFootprint for AudioBatch {
    fn resident_bytes(&self) -> usize {
        self.samples
            .resident_bytes()
            .saturating_add(self.mask.resident_bytes())
            .saturating_add(
                self.lengths
                    .capacity()
                    .saturating_mul(std::mem::size_of::<usize>()),
            )
    }
}
impl PinMemory for AudioBatch {
    fn pin_memory(mut self, d: Device) -> rusttorch_core::Result<Self> {
        self.samples = self.samples.pin_memory(d)?;
        self.mask = self.mask.pin_memory(d)?;
        Ok(self)
    }
}
/// Pads waveforms after validating equal rate/layout/dtype/device and byte limits.
#[derive(Clone, Copy, Debug, Default)]
pub struct PadAudio {
    /// Limits for the resulting padded allocation.
    pub limits: ResourceLimits,
}
impl Collate<Waveform> for PadAudio {
    type Batch = AudioBatch;
    type Error = AudioError;
    fn collate(&mut self, samples: Vec<Waveform>) -> Result<AudioBatch> {
        self.limits.check(
            "audio batch records",
            samples.len(),
            self.limits.max_records,
        )?;
        let first = samples
            .first()
            .ok_or_else(|| AudioError::Invalid("empty audio batch".into()))?;
        first.validate()?;
        let (rate, c, kind, device, layout) = (
            first.sample_rate,
            first.samples.size()[0],
            first.samples.kind(),
            first.samples.device(),
            first.channel_layout.clone(),
        );
        let width = samples.iter().map(Waveform::frames).max().unwrap_or(0);
        check_wave(c as usize, width, rate, self.limits)?;
        let n = self
            .limits
            .tensor_elements(&[samples.len(), c as usize, width])?;
        let mask_elements = self.limits.tensor_elements(&[samples.len(), width])?;
        self.limits.check(
            "padded audio bytes",
            n.checked_mul(kind.elt_size_in_bytes())
                .and_then(|bytes| bytes.checked_add(mask_elements))
                .and_then(|bytes| {
                    samples
                        .len()
                        .checked_mul(std::mem::size_of::<usize>())
                        .and_then(|lengths| bytes.checked_add(lengths))
                })
                .ok_or_else(|| AudioError::Invalid("audio batch overflow".into()))?,
            self.limits.max_decoded_bytes,
        )?;
        let mut audio = Vec::new();
        let mut masks = Vec::new();
        let mut lengths = Vec::new();
        for s in samples {
            s.validate()?;
            if s.sample_rate != rate
                || s.channel_layout != layout
                || s.samples.kind() != kind
                || s.samples.device() != device
            {
                return Err(AudioError::Invalid(
                    "audio batch rates/layouts/dtypes/devices differ".into(),
                ));
            }
            let len = s.frames();
            lengths.push(len);
            audio.push(s.samples.f_constant_pad_nd([0, (width - len) as i64])?);
            masks.push(Tensor::f_arange(width as i64, (Kind::Int64, device))?.f_lt(len as i64)?);
        }
        Ok(AudioBatch {
            samples: Tensor::f_stack(&audio, 0)?,
            mask: Tensor::f_stack(&masks, 0)?,
            lengths,
            sample_rate: rate,
        })
    }
}
