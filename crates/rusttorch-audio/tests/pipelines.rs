use rusttorch_audio::*;
use rusttorch_core::{Device, Kind, Tensor};
use rusttorch_data::{Collate, DataLoader, ResourceLimits};
use std::path::{Path, PathBuf};
fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mono.wav")
}
#[test]
fn wav_decode_workers_and_padding_preserve_rates_and_lengths() {
    let wave = decode_audio(fixture(), Default::default()).unwrap();
    assert_eq!(wave.sample_rate, 8000);
    assert_eq!(wave.frames(), 16);
    assert_eq!(
        Vec::<Vec<f32>>::try_from(&wave.samples).unwrap()[0][..4],
        [0., 0.25, -0.25, 0.5]
    );
    let data = AudioFiles::new(vec![fixture(), fixture()], Default::default()).unwrap();
    let mut loader = DataLoader::builder(data)
        .batch_size(2)
        .collate(PadAudio::default())
        .workers(2)
        .build()
        .unwrap();
    let batch = loader.iter().next().unwrap().unwrap();
    assert_eq!(batch.samples.size(), [2, 1, 16]);
    assert_eq!(batch.lengths, [16, 16]);
}
#[test]
fn spectral_features_retain_gradients_and_known_sine_peak() {
    let x = (0..32)
        .map(|i| (2. * std::f64::consts::PI * i as f64 / 8.).sin())
        .collect::<Vec<_>>();
    let x = Tensor::from_slice(&x)
        .reshape([1, 32])
        .set_requires_grad(true);
    let wave = Waveform::new(x, 8000, 0.).unwrap();
    let config = SpectrogramConfig {
        n_fft: 8,
        hop_length: 4,
        power: 2.,
    };
    let power = wave.spectrogram(config, Default::default()).unwrap();
    assert_eq!(power.size(), [1, 5, 7]);
    assert_eq!(power.get(0).get(1).double_value(&[0]), 4.);
    let mel = wave.mel_spectrogram(config, 3, Default::default()).unwrap();
    assert_eq!(mel.size(), [1, 3, 7]);
    let mfcc = wave.mfcc(config, 3, 2, Default::default()).unwrap();
    assert_eq!(mfcc.size(), [1, 2, 7]);
    mfcc.sum(Kind::Double).backward();
    assert!(wave.samples.grad().isfinite().all().int64_value(&[]) != 0);
}
#[test]
fn resampling_gain_shift_and_masks_have_explicit_behavior() {
    let wave = Waveform::new(Tensor::ones([1, 800], (Kind::Float, Device::Cpu)), 8000, 0.).unwrap();
    let wave = wave.resample(16000, Default::default()).unwrap();
    assert_eq!(wave.frames(), 1600);
    assert_eq!(wave.sample_rate, 16000);
    let wave = wave
        .gain(-6.020599913279624)
        .unwrap()
        .time_shift(2)
        .unwrap();
    assert_eq!(wave.samples.double_value(&[0, 0]), 0.);
    assert!((wave.samples.double_value(&[0, 800]) - 0.5).abs() < 1e-3);
    let input = Tensor::ones([2, 4], (Kind::Float, Device::Cpu));
    let masked = mask_axis(&input, 1, 1, 2, 0.).unwrap();
    assert_eq!(
        Vec::<Vec<f32>>::try_from(&masked).unwrap(),
        [[1., 0., 0., 1.], [1., 0., 0., 1.]]
    );
    assert_eq!(input.sum(Kind::Float).double_value(&[]), 8.);
}
#[test]
fn audio_boundaries_reject_rate_shape_and_oversize() {
    assert!(Waveform::new(Tensor::zeros([1, 4], (Kind::Float, Device::Cpu)), 0, 0.).is_err());
    assert!(
        decode_audio(
            fixture(),
            ResourceLimits {
                max_encoded_bytes: 8,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        decode_audio(
            fixture(),
            ResourceLimits {
                max_tensor_elements: 2,
                ..Default::default()
            }
        )
        .is_err()
    );
    let a = Waveform::new(Tensor::zeros([1, 2], (Kind::Float, Device::Cpu)), 8000, 0.).unwrap();
    let b = Waveform::new(Tensor::zeros([1, 3], (Kind::Float, Device::Cpu)), 16000, 0.).unwrap();
    assert!(PadAudio::default().collate(vec![a, b]).is_err());
}

#[test]
fn spectral_values_and_gradients_match_pinned_python() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/numerical.json")).unwrap();
    assert_eq!(data["pytorch_version"], "2.13.0");
    let values = |key: &str| {
        data[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>()
    };
    let x = Tensor::from_slice(&values("input"))
        .reshape([1, 16])
        .set_requires_grad(true);
    let wave = Waveform::new(x, 8000, 0.).unwrap();
    let config = SpectrogramConfig {
        n_fft: 8,
        hop_length: 4,
        power: 2.,
    };
    let power = wave.spectrogram(config, Default::default()).unwrap();
    let mel = wave.mel_spectrogram(config, 3, Default::default()).unwrap();
    let mfcc = wave.mfcc(config, 3, 2, Default::default()).unwrap();
    mfcc.sum(Kind::Double).backward();
    let filter = mel_filter_bank(8, 8000, 3, Default::default()).unwrap();
    for (key, tensor) in [
        ("power", power),
        ("mel", mel),
        ("mfcc", mfcc),
        ("gradient", wave.samples.grad()),
        ("filter", filter),
    ] {
        let got = Vec::<f64>::try_from(&tensor.reshape([-1])).unwrap();
        let expected = values(key);
        assert_eq!(got.len(), expected.len());
        for (a, b) in got.iter().zip(expected) {
            assert!((a - b).abs() < 1e-8 * (1. + b.abs()), "{key}: {a} != {b}");
        }
    }
}
#[test]
fn flac_decodes_the_same_samples_as_pcm_wav() {
    let wav = decode_audio(fixture(), Default::default()).unwrap();
    let flac = decode_audio(fixture().with_extension("flac"), Default::default()).unwrap();
    assert_eq!(flac.sample_rate, wav.sample_rate);
    assert_eq!(
        Vec::<Vec<f32>>::try_from(&flac.samples).unwrap(),
        Vec::<Vec<f32>>::try_from(&wav.samples).unwrap()
    );
}

#[test]
fn mel_expansion_checks_output_shape_before_projection() {
    let wave = Waveform::new(Tensor::ones([2, 64], (Kind::Float, Device::Cpu)), 8000, 0.).unwrap();
    let cfg = SpectrogramConfig {
        n_fft: 2,
        hop_length: 1,
        power: 2.,
    };
    // Input spectrum 252 and filter 8 elements fit, but Mel output 504 does not.
    let limits = ResourceLimits {
        max_tensor_elements: 300,
        ..Default::default()
    };
    assert!(wave.mel_spectrogram(cfg, 4, limits).is_err());
}

#[test]
fn padded_audio_limits_cover_records_duration_masks_and_lengths() {
    let wave = |frames, kind| {
        Waveform::new(Tensor::zeros([1, frames], (kind, Device::Cpu)), 1, 0.).unwrap()
    };
    for (limits, frames, kind) in [
        (
            ResourceLimits {
                max_records: 1,
                ..Default::default()
            },
            0,
            Kind::Float,
        ),
        (
            ResourceLimits {
                max_media_seconds: 1,
                ..Default::default()
            },
            2,
            Kind::Float,
        ),
        (
            ResourceLimits {
                max_decoded_bytes: 32,
                ..Default::default()
            },
            2,
            Kind::Double,
        ),
        (
            ResourceLimits {
                max_decoded_bytes: std::mem::size_of::<usize>(),
                ..Default::default()
            },
            0,
            Kind::Float,
        ),
    ] {
        assert!(
            PadAudio { limits }
                .collate(vec![wave(frames, kind), wave(frames, kind)])
                .is_err()
        );
    }
}

#[test]
fn audio_noise_and_downsampling_bound_host_payload_before_allocation() {
    let context = rusttorch_data::TaskContext {
        loader_seed: 1,
        epoch: 0,
        rank: 0,
        logical_sample: 0,
        stage: 0,
        cancellation: rusttorch_data::CancellationToken::new(),
        deadline: rusttorch_data::Deadline::none(),
    };
    let wave = Waveform::new(Tensor::zeros([1, 4], (Kind::Float, Device::Cpu)), 8000, 0.).unwrap();
    let limits = ResourceLimits {
        max_decoded_bytes: 1,
        ..Default::default()
    };
    assert!(wave.add_noise(0.1, &context, limits).is_err());
    let wave = Waveform::new(
        Tensor::zeros([1, 10_000], (Kind::Float, Device::Cpu)),
        8000,
        0.,
    )
    .unwrap();
    let limits = ResourceLimits {
        max_decoded_bytes: 16_000,
        ..Default::default()
    };
    assert!(wave.resample(800, limits).is_err());
}
