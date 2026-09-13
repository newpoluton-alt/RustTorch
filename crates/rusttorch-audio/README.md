# RustTorch Audio

Load waveforms, prepare variable-length audio batches, resample recordings and
compute spectral features for speech or sound models. `Waveform.samples` is a
floating-point **[channels, frames]** tensor accompanied by sample rate, channel
layout and a finite time origin.

## Installation

Enable `rusttorch = { version = "0.4", features = ["audio"] }` for
`rusttorch::audio`, or install `rusttorch-audio` directly with its
`download-libtorch` feature. The default decoder profile is Rust-native WAV/PCM
and FLAC; it does not require FFmpeg. The optional `codec` dependency exposes the
separate FFmpeg adapter when your application explicitly configures native
linking.

## Features

The `audio` facade feature enables WAV/PCM and FLAC decoding, resampling and
spectral features. The direct package has no default features. Its optional
`codec` feature re-exports the separate codec package; applications additionally
select that package's `system` or `vcpkg` linking feature.
`download-libtorch` obtains the tensor runtime; `doc-only` builds documentation
without native linking and cannot execute tensor operations.

## Native runtime

Tensor operations require **LibTorch 2.13.0**. Enable `download-libtorch`, or set
`LIBTORCH` to an extracted distribution and add its library directory to `PATH`
on Windows or the platform's shared-library search path. All RustTorch packages
in one application must share the same runtime. Documentation can be checked
with `cargo doc -p rusttorch-audio --no-default-features --features doc-only`.

## Example: Load and pad audio files

```no_run
use rusttorch_audio::{AudioFiles, PadAudio};
use rusttorch_data::{DataLoader, ResourceLimits};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let limits = ResourceLimits::default();
    let files = AudioFiles::new(vec!["first.wav".into(), "second.flac".into()], limits)?;
    let mut loader = DataLoader::builder(files)
        .batch_size(8)
        .workers(2)
        .collate(PadAudio { limits })
        .build()?;
    for batch in loader.iter() {
        let batch = batch?;
        // [batch, channels, longest_frames], plus a [batch, longest_frames] mask.
        println!("{:?}, rate={}", batch.samples.size(), batch.sample_rate);
        // Use batch.mask to exclude padding from losses or attention.
    }
    Ok(())
}
```

A batch requires matching rates, channel layouts, tensor dtypes and devices.
Resample or select channels deliberately before batching; the collator never
silently changes the recording. `lengths` records original durations in frames,
and padding is zero. Pinning delegates to LibTorch through the common loader.

## Resample and augment a recording

```no_run
use rusttorch_audio::decode_audio;
use rusttorch_data::ResourceLimits;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let limits = ResourceLimits::default();
    let wave = decode_audio("recording.wav", limits)?
        .resample(16_000, limits)?
        .gain(-3.)?
        .time_shift(160)?;
    println!("{} seconds at {} Hz", wave.duration(), wave.sample_rate);
    Ok(())
}
```

Rubato resampling uses windowed-sinc interpolation on CPU, removes startup delay
and returns `ceil(input_frames * new_rate / old_rate)` frames. It does not
preserve autograd; a tensor requiring gradients must be explicitly detached.
Gain, zero-filled time shift, mixing and tensor spectral transforms retain the
normal LibTorch graph. `mix` requires aligned origins, lengths and formats.

For repeatable augmentation inside a loader transform, pass the supplied
`TaskContext` to `add_noise(amplitude, context, limits)`. Noise is uniform in
`[-amplitude, amplitude]` and derives from logical sample identity, seed and
epoch. It does not change the global tensor RNG or depend on worker scheduling.

## Compute spectrograms, Mel features and MFCCs

```no_run
use rusttorch_audio::{decode_audio, SpectrogramConfig, mask_axis};
use rusttorch_data::ResourceLimits;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let limits = ResourceLimits::default();
    let wave = decode_audio("recording.wav", limits)?.resample(16_000, limits)?;
    let stft = SpectrogramConfig { n_fft: 400, hop_length: 160, power: 2. };
    let power = wave.spectrogram(stft, limits)?; // [channels, 201, windows]
    let mel = wave.mel_spectrogram(stft, 40, limits)?;
    let mfcc = wave.mfcc(stft, 40, 13, limits)?;
    let masked = mask_axis(&mel, 1, 3, 4, 0.)?; // Zero four Mel bins.
    assert_eq!(mfcc.size()[1], 13);
    assert_eq!(masked.size(), mel.size());
    println!("spectrogram: {:?}", power.size());
    Ok(())
}
```

The spectral convention is explicit: periodic Hann window, **uncentered**
complete windows, no signal padding, no FFT normalization, one-sided spectrum.
`power=1` produces magnitude and `power=2` power. Inputs shorter than one window
return an error.

Mel filters use HTK frequency spacing from 0 to Nyquist and triangular weights,
without area normalization. MFCC uses natural-log Mel energy with floor `1e-10`,
then orthonormal DCT-II; it does not apply dB or top-dB normalization. Match these
choices to how your model was trained. `mask_axis` copies before masking, so
other tensor aliases do not change.

## Formats, safety and evidence

| Capability | Verified scope | Evidence |
|---|---|---|
| WAV/FLAC | Local bounded files, CPU Float conversion | `tests/pipelines.rs` |
| Worker loading/padding | Equal-format samples, explicit masks | `wav_decode_workers_and_padding_preserve_rates_and_lengths` |
| Resampling | CPU sinc interpolation and exact frame count | `resampling_gain_shift_and_masks_have_explicit_behavior` |
| Spectral features | Values, shape and gradients | `spectral_features_retain_gradients_and_known_sine_peak` |
| Input rejection | Bad rate/shape and size ceilings | `audio_boundaries_reject_rate_shape_and_oversize` |

Limits cover encoded bytes, decoded samples, output tensors and media duration.
Decoders preserve errors rather than skipping corrupt packets. Native/library
scratch memory is not a process-RSS promise. No audio datasets or codecs are
downloaded at runtime.

| Operation | Executable evidence |
|---|---|
| WAV/PCM and FLAC decode to identical samples | `flac_decodes_the_same_samples_as_pcm_wav` |
| Worker loading, padding and original lengths | `wav_decode_workers_and_padding_preserve_rates_and_lengths` |
| Sinc resampling, gain, shifts and masking | `resampling_gain_shift_and_masks_have_explicit_behavior` |
| STFT/Mel/MFCC values and input gradients | `spectral_values_and_gradients_match_pinned_python` |
| Projected Mel output limits | `mel_expansion_checks_output_shape_before_projection` |

Padded batch limits include waveform bytes, Boolean masks and original lengths,
including empty recordings. Record and media-duration limits still apply.
Noise generation and resampling check copied host input bytes before allocating.

## Process an FFmpeg audio block

With the direct package's `codec` feature, the codec types are available as
`rusttorch_audio::codec`. Select `system` or `vcpkg` on `rusttorch-codec` to
create its native decoder. Convert each owned audio block into a waveform and
reuse the same preprocessing operations:

```rust
# #[cfg(feature = "codec")]
fn prepare(block: rusttorch_audio::codec::AudioFrame)
    -> rusttorch_audio::Result<rusttorch_audio::Waveform>
{
    let origin = block.timestamp.map_or(0.0, |time| time.seconds());
    let waveform = rusttorch_audio::Waveform::new(block.samples, block.sample_rate, origin)?;
    waveform.gain(-3.0)
}
```

The conversion moves the tensor and preserves the source rate and presentation
time. A negative time origin is rejected by `Waveform`; applications retaining
negative preroll timestamps must choose an explicit rebasing policy first.
