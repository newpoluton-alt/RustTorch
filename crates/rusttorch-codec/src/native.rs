use super::*;
use rsmpeg::{
    avcodec::AVCodecContext,
    avformat::AVFormatContextInput,
    avutil::{AVChannelLayout, AVDictionary, AVFrame},
    error::RsmpegError,
    ffi,
    swresample::SwrContext,
    swscale::SwsContext,
};
use rusttorch_core::Tensor;
use rusttorch_data::{ResourceLimits, WorkerContext};
use std::{ffi::CString, path::Path};
/// Reports the actually linked FFmpeg build, including GPL/nonfree configuration.
pub fn capabilities() -> CodecCapabilities {
    CodecCapabilities {
        version: rsmpeg::avutil::version_info()
            .to_string_lossy()
            .into_owned(),
        configuration: rsmpeg::avcodec::configuration()
            .to_string_lossy()
            .into_owned(),
        license: rsmpeg::avcodec::license().to_string_lossy().into_owned(),
        cpu_decode: true,
    }
}
/// Owned FFmpeg decoder over one bounded, regular local file and selected stream.
/// Decode is lazy, one frame at a time; errors are yielded once then iteration
/// terminates. This adapter does not promise exact checkpoint/resume of codecs.
/// Native calls cannot be force-cancelled; cooperative context is checked between
/// packet/frame operations. Network protocols and playlist demuxers are disabled.
pub struct MediaDecoder {
    input: AVFormatContextInput,
    decoder: AVCodecContext,
    info: StreamInfo,
    limits: ResourceLimits,
    draining: bool,
    done: bool,
    count: usize,
    seek_target: Option<f64>,
    context: Option<WorkerContext>,
    audio_frames: u64,
}
impl MediaDecoder {
    /// Opens a regular local file and chooses FFmpeg's best stream of this kind.
    /// Supported demuxers: Matroska/WebM, WAV, AVI, MOV/MP4, MP3 and FLAC. CPU
    /// video and mono/stereo audio are the explicit baseline; no hardware fallback.
    pub fn open(path: impl AsRef<Path>, kind: StreamKind, limits: ResourceLimits) -> Result<Self> {
        let path = std::fs::canonicalize(path)?;
        let metadata = path.metadata()?;
        if !metadata.is_file() {
            return Err(CodecError::Invalid(
                "media input must be a regular local file".into(),
            ));
        }
        limits.check(
            "media encoded bytes",
            usize::try_from(metadata.len()).unwrap_or(usize::MAX),
            limits.max_encoded_bytes,
        )?;
        let path = CString::new(
            path.to_str()
                .ok_or_else(|| CodecError::Invalid("media path must be UTF-8".into()))?,
        )
        .map_err(|_| CodecError::Invalid("media path contains NUL".into()))?;
        let opts = AVDictionary::new(c"protocol_whitelist", c"file", 0)
            .set(
                c"format_whitelist",
                c"matroska,webm,wav,avi,mov,mp3,flac",
                0,
            )
            .set_int(
                c"probesize",
                limits.max_encoded_bytes.clamp(32, 5_000_000) as i64,
                0,
            )
            .set_int(c"analyzeduration", 5_000_000, 0)
            .set_int(c"rw_timeout", 5_000_000, 0)
            .set_int(
                c"max_streams",
                limits.max_streams.min(i32::MAX as usize) as i64,
                0,
            );
        let mut options = Some(opts);
        let input = AVFormatContextInput::builder()
            .url(&path)
            .options(&mut options)
            .open()?;
        limits.check("media streams", input.streams().len(), limits.max_streams)?;
        let media = match kind {
            StreamKind::Video => ffi::AVMEDIA_TYPE_VIDEO,
            StreamKind::Audio => ffi::AVMEDIA_TYPE_AUDIO,
        };
        let (index, codec) = input
            .find_best_stream(media)?
            .ok_or_else(|| CodecError::Invalid("requested media stream not found".into()))?;
        let stream = &input.streams()[index];
        let time_base = TimeBase::new(
            u32::try_from(stream.time_base.num)
                .map_err(|_| CodecError::Invalid("invalid stream time numerator".into()))?,
            u32::try_from(stream.time_base.den)
                .map_err(|_| CodecError::Invalid("invalid stream time denominator".into()))?,
        )?;
        let duration = if stream.duration == ffi::AV_NOPTS_VALUE {
            None
        } else {
            Some(stream.duration)
        };
        if duration.is_some_and(|d| d < 0 || time_base.seconds(d) > limits.max_media_seconds as f64)
        {
            return Err(CodecError::Invalid(
                "declared media duration exceeds limit".into(),
            ));
        }
        let params = stream.codecpar();
        if kind == StreamKind::Video {
            check_video(params.height, params.width, limits)?;
        } else if params.ch_layout.nb_channels < 1 || params.ch_layout.nb_channels > 2 {
            return Err(CodecError::Invalid(
                "native audio baseline supports mono/stereo only".into(),
            ));
        }
        let mut decoder = AVCodecContext::new(&codec);
        decoder.apply_codecpar(&params)?;
        decoder.open(Some(AVDictionary::new_int(
            c"max_pixels",
            limits.max_tensor_elements.min(i64::MAX as usize) as i64,
            0,
        )))?;
        drop(params);
        let info = StreamInfo {
            index,
            kind,
            time_base,
            duration,
            decoder: codec.name().to_string_lossy().into_owned(),
        };
        Ok(Self {
            input,
            decoder,
            info,
            limits,
            draining: false,
            done: false,
            count: 0,
            seek_target: None,
            context: None,
            audio_frames: 0,
        })
    }
    /// Metadata for the selected stream.
    pub fn stream_info(&self) -> &StreamInfo {
        &self.info
    }
    /// Attaches cooperative loader cancellation/deadline checks between native calls.
    pub fn with_worker_context(mut self, context: WorkerContext) -> Self {
        self.context = Some(context);
        self
    }
    /// Seeks to a keyframe then discards frames before the requested presentation
    /// time. Audio returns the first whole block starting at or after that time.
    /// Unknown timestamps after seeking are rejected rather than guessed.
    pub fn seek(&mut self, seconds: f64) -> Result<()> {
        if !seconds.is_finite() || seconds < 0. || seconds > self.limits.max_media_seconds as f64 {
            return Err(CodecError::Invalid("invalid seek time".into()));
        }
        let ticks = seconds * self.info.time_base.denominator() as f64
            / self.info.time_base.numerator() as f64;
        if ticks >= i64::MAX as f64 {
            return Err(CodecError::Invalid("seek timestamp overflows".into()));
        }
        self.input.seek(
            self.info.index as i32,
            ticks.floor() as i64,
            ffi::AVSEEK_FLAG_BACKWARD as i32,
        )?;
        self.decoder.flush_buffers();
        self.draining = false;
        self.done = false;
        self.count = 0;
        self.audio_frames = 0;
        self.seek_target = Some(seconds);
        Ok(())
    }
    fn next_frame(&mut self) -> Result<Option<MediaFrame>> {
        loop {
            if let Some(context) = &self.context
                && (context.cancellation.is_cancelled() || context.deadline.is_expired())
            {
                return Err(CodecError::Invalid(
                    "media decoding cancelled or timed out".into(),
                ));
            }
            match self.decoder.receive_frame() {
                Ok(frame) => {
                    self.count = self
                        .count
                        .checked_add(1)
                        .ok_or_else(|| CodecError::Invalid("frame count overflow".into()))?;
                    self.limits.check(
                        "decoded media frames",
                        self.count,
                        self.limits.max_records,
                    )?;
                    let pts = frame.best_effort_timestamp;
                    let timestamp = if pts == ffi::AV_NOPTS_VALUE {
                        None
                    } else {
                        Some(Timestamp {
                            ticks: pts,
                            time_base: self.info.time_base,
                        })
                    };
                    if timestamp
                        .is_some_and(|t| t.seconds().abs() > self.limits.max_media_seconds as f64)
                    {
                        return Err(CodecError::Invalid(
                            "decoded timestamp exceeds duration ceiling".into(),
                        ));
                    }
                    if let Some(target) = self.seek_target {
                        match timestamp {
                            Some(t) if t.seconds() < target => continue,
                            Some(_) => self.seek_target = None,
                            None => {
                                return Err(CodecError::Invalid(
                                    "cannot perform time seek on frames without timestamps".into(),
                                ));
                            }
                        }
                    }
                    return Ok(Some(match self.info.kind {
                        StreamKind::Video => {
                            MediaFrame::Video(video(frame, timestamp, self.limits)?)
                        }
                        StreamKind::Audio => {
                            let audio = audio(frame, timestamp, self.limits)?;
                            self.audio_frames = self
                                .audio_frames
                                .checked_add(audio.samples.size()[1] as u64)
                                .ok_or_else(|| {
                                    CodecError::Invalid("audio length overflow".into())
                                })?;
                            if self.audio_frames as u128
                                > audio.sample_rate as u128 * self.limits.max_media_seconds as u128
                            {
                                return Err(CodecError::Invalid(
                                    "decoded audio duration exceeds limit".into(),
                                ));
                            }
                            MediaFrame::Audio(audio)
                        }
                    }));
                }
                Err(RsmpegError::DecoderFlushedError) => return Ok(None),
                Err(RsmpegError::DecoderDrainError) => {
                    if self.draining {
                        return Err(CodecError::Invalid(
                            "decoder requested packets after end-of-stream flush".into(),
                        ));
                    }
                }
                Err(e) => return Err(e.into()),
            }
            loop {
                match self.input.read_packet()? {
                    Some(packet) => {
                        if packet.size < 0 || packet.size as usize > self.limits.max_encoded_bytes {
                            return Err(CodecError::Invalid(
                                "invalid or oversized media packet".into(),
                            ));
                        }
                        if packet.stream_index != self.info.index as i32 {
                            continue;
                        }
                        self.decoder.send_packet(Some(&packet))?;
                        break;
                    }
                    None => {
                        self.decoder.send_packet(None)?;
                        self.draining = true;
                        break;
                    }
                }
            }
        }
    }
}
impl Iterator for MediaDecoder {
    type Item = Result<MediaFrame>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.next_frame() {
            Ok(Some(f)) => Some(Ok(f)),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}
fn check_video(h: i32, w: i32, limits: ResourceLimits) -> Result<usize> {
    if h <= 0 || w <= 0 {
        return Err(CodecError::Invalid(
            "video dimensions must be positive".into(),
        ));
    }
    limits.check("video height", h as usize, limits.max_image_dimension)?;
    limits.check("video width", w as usize, limits.max_image_dimension)?;
    let n = limits.tensor_elements(&[3, h as usize, w as usize])?;
    limits.check("RGB frame bytes", n, limits.max_decoded_bytes)?;
    Ok(n)
}
fn video(
    frame: AVFrame,
    timestamp: Option<Timestamp>,
    limits: ResourceLimits,
) -> Result<VideoFrame> {
    let size = check_video(frame.height, frame.width, limits)?;
    let mut scaler = SwsContext::get_context(
        frame.width,
        frame.height,
        frame.format,
        frame.width,
        frame.height,
        ffi::AV_PIX_FMT_RGB24,
        ffi::SWS_BILINEAR,
        None,
        None,
        None,
    )
    .ok_or_else(|| CodecError::Invalid("unsupported video pixel conversion".into()))?;
    let mut output = AVFrame::new();
    output.set_width(frame.width);
    output.set_height(frame.height);
    output.set_format(ffi::AV_PIX_FMT_RGB24);
    output.alloc_buffer()?;
    scaler.scale_frame(&frame, 0, frame.height, &mut output)?;
    let expected = output.image_get_buffer_size(1)?;
    if expected != size {
        return Err(CodecError::Invalid(
            "RGB buffer size differs from checked dimensions".into(),
        ));
    }
    let mut bytes = vec![0u8; size];
    output.image_copy_to_buffer(&mut bytes, 1)?;
    Ok(VideoFrame {
        pixels: Tensor::f_from_slice(&bytes)?
            .f_reshape([frame.height as i64, frame.width as i64, 3])?
            .f_permute([2, 0, 1])?,
        timestamp,
    })
}
fn audio(
    mut frame: AVFrame,
    timestamp: Option<Timestamp>,
    limits: ResourceLimits,
) -> Result<AudioFrame> {
    let channels = frame.ch_layout.nb_channels;
    let samples = frame.nb_samples;
    if !(1..=2).contains(&channels) || samples < 0 || frame.sample_rate <= 0 {
        return Err(CodecError::Invalid(
            "invalid mono/stereo audio frame".into(),
        ));
    }
    let n = limits.tensor_elements(&[channels as usize, samples as usize])?;
    let bytes = n
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| CodecError::Invalid("audio frame size overflow".into()))?;
    limits.check("audio frame bytes", bytes, limits.max_decoded_bytes)?;
    let layout = AVChannelLayout::from_nb_channels(channels);
    if frame.ch_layout.order == ffi::AV_CHANNEL_ORDER_UNSPEC {
        frame.set_ch_layout(*layout);
    }
    let mut converter = SwrContext::new(
        &layout,
        ffi::AV_SAMPLE_FMT_FLT,
        frame.sample_rate,
        &frame.ch_layout,
        frame.format,
        frame.sample_rate,
    )?;
    converter.init()?;
    let mut output = AVFrame::new();
    output.set_ch_layout(*layout);
    output.set_sample_rate(frame.sample_rate);
    output.set_format(ffi::AV_SAMPLE_FMT_FLT);
    output.set_nb_samples(samples);
    output.alloc_buffer()?;
    converter.convert_frame(Some(&frame), &mut output)?;
    if output.nb_samples != samples
        || output.linesize[0] < 0
        || (output.linesize[0] as usize) < bytes
        || output.data[0].is_null()
        || !(output.data[0] as usize).is_multiple_of(std::mem::align_of::<f32>())
    {
        return Err(CodecError::Invalid(
            "native audio output does not match its allocated layout".into(),
        ));
    }
    // SAFETY: `output` is an owned FFmpeg AVFrame allocated above in packed FLT
    // mono/stereo format. Same-rate swresample initialized all `samples` frames;
    // the result count, byte length, pointer and f32 alignment were checked.
    // The immutable slice exists only while output owns that live allocation.
    // Tensor::f_from_slice copies the values before output can be dropped. No
    // raw pointer or borrowed native storage escapes this private boundary.
    let data = unsafe { std::slice::from_raw_parts(output.data[0].cast::<f32>(), n) };
    Ok(AudioFrame {
        samples: Tensor::f_from_slice(data)?
            .f_reshape([samples as i64, channels as i64])?
            .f_transpose(0, 1)?,
        sample_rate: frame.sample_rate as u32,
        timestamp,
    })
}
