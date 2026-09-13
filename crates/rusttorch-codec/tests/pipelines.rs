use rusttorch_codec::TimeBase;
#[test]
fn rational_time_is_checked_and_exact() {
    let ms = TimeBase::new(1, 1000).unwrap();
    let samples = TimeBase::new(1, 48000).unwrap();
    assert_eq!(ms.rescale(1500, samples).unwrap(), 72000);
    assert_eq!(ms.seconds(-1000), -1.);
    assert!(TimeBase::new(1, 0).is_err());
    assert!(
        TimeBase::new(u32::MAX, 1)
            .unwrap()
            .rescale(i64::MAX, TimeBase::new(1, u32::MAX).unwrap())
            .is_err()
    );
}
#[cfg(any(feature = "system", feature = "vcpkg"))]
mod native {
    use rusttorch_codec::{MediaDecoder, MediaFrame, StreamKind, capabilities};
    use rusttorch_data::{ResourceLimits, batches};
    fn fixture(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }
    #[test]
    fn cpu_video_streams_batches_and_seeks_with_timestamps() {
        let d =
            MediaDecoder::open(fixture("tiny.mkv"), StreamKind::Video, Default::default()).unwrap();
        let groups = batches(d, 2, false)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(groups.iter().map(Vec::len).collect::<Vec<_>>(), [2, 1]);
        let MediaFrame::Video(first) = &groups[0][0] else {
            panic!("video selected")
        };
        assert_eq!(first.pixels.size(), [3, 8, 8]);
        assert!(first.pixels.double_value(&[0, 0, 0]) > 240.);
        assert_eq!(first.timestamp.unwrap().ticks, 0);
        let mut d =
            MediaDecoder::open(fixture("tiny.mkv"), StreamKind::Video, Default::default()).unwrap();
        d.seek(0.3).unwrap();
        assert!(d.next().unwrap().unwrap().timestamp().unwrap().seconds() >= 0.3);
        assert!(capabilities().cpu_decode);
    }
    #[test]
    fn native_audio_copies_checked_float_buffers() {
        let mut d =
            MediaDecoder::open(fixture("mono.wav"), StreamKind::Audio, Default::default()).unwrap();
        let MediaFrame::Audio(a) = d.next().unwrap().unwrap() else {
            panic!("audio selected")
        };
        assert_eq!(a.sample_rate, 8000);
        assert_eq!(a.samples.size(), [1, 16]);
        assert_eq!(
            Vec::<Vec<f32>>::try_from(&a.samples).unwrap()[0][..4],
            [0., 0.25, -0.25, 0.5]
        );
        assert!(d.next().is_none());
    }
    #[test]
    fn limits_and_missing_streams_fail_explicitly() {
        assert!(
            MediaDecoder::open(
                fixture("tiny.mkv"),
                StreamKind::Video,
                ResourceLimits {
                    max_image_dimension: 4,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            MediaDecoder::open(fixture("mono.wav"), StreamKind::Video, Default::default()).is_err()
        );
        let mut d = MediaDecoder::open(
            fixture("tiny.mkv"),
            StreamKind::Video,
            ResourceLimits {
                max_records: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(d.next().unwrap().is_ok());
        assert!(d.next().unwrap().is_err());
        assert!(d.next().is_none());
    }
}
