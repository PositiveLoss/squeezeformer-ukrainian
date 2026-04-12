use std::fs::File;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{anyhow, bail, Context, Result};
use ffmpeg_next as ffmpeg;
use log::{debug, trace, warn};
use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};

#[derive(Debug, Clone)]
pub(crate) enum AudioSource {
    Path(PathBuf, Option<String>),
    Bytes(Vec<u8>, Option<String>),
}

impl AudioSource {
    pub(crate) fn log_label(&self) -> String {
        match self {
            Self::Path(path, Some(path_hint)) => {
                format!("path={} path_hint={path_hint}", path.display())
            }
            Self::Path(path, None) => format!("path={}", path.display()),
            Self::Bytes(bytes, Some(path_hint)) => {
                format!("bytes={} path_hint={path_hint}", bytes.len())
            }
            Self::Bytes(bytes, None) => format!("bytes={}", bytes.len()),
        }
    }
}

pub(crate) fn decode_audio(
    source: AudioSource,
    fallback_sample_rate: u32,
    ffmpeg_fallback: bool,
) -> Result<(Vec<f32>, u32)> {
    let source_label = source.log_label();
    debug!(
        "decoding audio source={} fallback_sample_rate={} ffmpeg_fallback={}",
        source_label, fallback_sample_rate, ffmpeg_fallback
    );
    match decode_audio_symphonia(source.clone()) {
        Ok(decoded) => {
            debug!(
                "decoded audio with symphonia source={} samples={} sample_rate={}",
                source_label,
                decoded.0.len(),
                decoded.1
            );
            Ok(decoded)
        }
        Err(symphonia_error) if ffmpeg_fallback => {
            warn!(
                "symphonia decode failed for {}; falling back to FFmpeg libraries: {symphonia_error:#}",
                source_label
            );
            decode_audio_ffmpeg(source, fallback_sample_rate).with_context(|| {
                format!("symphonia decode failed: {symphonia_error:#}; FFmpeg fallback failed")
            })
        }
        Err(error) => Err(error),
    }
}

fn decode_audio_symphonia(source: AudioSource) -> Result<(Vec<f32>, u32)> {
    let (mss, extension) = match source {
        AudioSource::Path(path, path_hint) => {
            trace!("opening audio file with symphonia path={}", path.display());
            let extension = path_hint
                .as_deref()
                .and_then(|path| Path::new(path).extension())
                .or_else(|| path.extension())
                .and_then(|value| value.to_str())
                .map(str::to_owned);
            let file = File::open(&path)
                .with_context(|| format!("failed to open audio file {}", path.display()))?;
            (
                MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default()),
                extension,
            )
        }
        AudioSource::Bytes(bytes, path_hint) => {
            let extension = path_hint
                .as_deref()
                .and_then(|path| Path::new(path).extension())
                .and_then(|value| value.to_str())
                .map(str::to_owned);
            (
                MediaSourceStream::new(
                    Box::new(Cursor::new(bytes)),
                    MediaSourceStreamOptions::default(),
                ),
                extension,
            )
        }
    };

    let mut hint = Hint::new();
    if let Some(extension) = extension.as_deref() {
        hint.with_extension(extension);
    }
    let probed = get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| anyhow!("audio container has no default track"))?;
    if track.codec_params.codec == CODEC_TYPE_NULL {
        bail!("unsupported null audio codec");
    }
    let track_id = track.id;
    debug!(
        "symphonia selected track id={} codec={:?} sample_rate={:?}",
        track_id, track.codec_params.codec, track.codec_params.sample_rate
    );
    let mut decoder = get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let mut mono = Vec::new();
    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(16_000);

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(SymphoniaError::ResetRequired) => {
                bail!("decoder reset is not supported for this audio stream");
            }
            Err(error) => return Err(error.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(message)) => {
                trace!("symphonia skipped packet decode error: {message}");
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        append_mono_samples(decoded, &mut mono, &mut sample_rate);
    }

    if mono.is_empty() {
        bail!("decoded audio stream is empty");
    }
    Ok((mono, sample_rate))
}

fn decode_audio_ffmpeg(source: AudioSource, sample_rate: u32) -> Result<(Vec<f32>, u32)> {
    debug!(
        "decoding audio with FFmpeg libraries source={} output_sample_rate={}",
        source.log_label(),
        sample_rate
    );
    match source {
        AudioSource::Path(path, _) => decode_audio_ffmpeg_path(&path, sample_rate),
        AudioSource::Bytes(bytes, path_hint) => {
            decode_audio_ffmpeg_bytes(bytes, path_hint, sample_rate)
        }
    }
}

fn decode_audio_ffmpeg_bytes(
    bytes: Vec<u8>,
    path_hint: Option<String>,
    sample_rate: u32,
) -> Result<(Vec<f32>, u32)> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("asr_features_audio_");
    let suffix;
    if let Some(extension) = path_hint
        .as_deref()
        .and_then(|path| Path::new(path).extension())
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
    {
        suffix = format!(".{extension}");
        builder.suffix(&suffix);
    }
    let mut file = builder
        .tempfile()
        .context("failed to create temporary audio file for FFmpeg decode")?;
    file.write_all(&bytes)
        .context("failed to write audio bytes to temporary file for FFmpeg decode")?;
    file.flush()
        .context("failed to flush temporary audio file for FFmpeg decode")?;
    decode_audio_ffmpeg_path(file.path(), sample_rate)
}

fn init_ffmpeg() -> Result<()> {
    static INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    match INIT.get_or_init(|| ffmpeg::init().map_err(|error| error.to_string())) {
        Ok(()) => Ok(()),
        Err(error) => bail!("failed to initialize FFmpeg libraries: {error}"),
    }
}

fn decode_audio_ffmpeg_path(path: &Path, sample_rate: u32) -> Result<(Vec<f32>, u32)> {
    init_ffmpeg()?;
    let mut input = ffmpeg::format::input(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let stream = input
        .streams()
        .best(ffmpeg::media::Type::Audio)
        .ok_or_else(|| anyhow!("FFmpeg found no audio stream in {}", path.display()))?;
    let stream_index = stream.index();
    let context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
        .context("failed to create FFmpeg decoder context")?;
    let mut decoder = context
        .decoder()
        .audio()
        .context("failed to open FFmpeg audio decoder")?;
    let mut samples = Vec::new();
    let mut resampler = None;

    for (packet_stream, packet) in input.packets() {
        if packet_stream.index() == stream_index {
            decoder
                .send_packet(&packet)
                .context("failed to send packet to FFmpeg decoder")?;
            receive_ffmpeg_frames(&mut decoder, &mut resampler, sample_rate, &mut samples)
                .context("failed to receive FFmpeg decoded frame")?;
        }
    }
    decoder
        .send_eof()
        .context("failed to flush FFmpeg decoder")?;
    receive_ffmpeg_frames(&mut decoder, &mut resampler, sample_rate, &mut samples)
        .context("failed to receive final FFmpeg decoded frame")?;
    flush_ffmpeg_resampler(&mut resampler, &mut samples)
        .context("failed to flush FFmpeg resampler")?;

    if samples.is_empty() {
        bail!("FFmpeg decoded audio stream is empty");
    }
    debug!(
        "decoded audio with FFmpeg libraries samples={} sample_rate={}",
        samples.len(),
        sample_rate
    );
    Ok((samples, sample_rate))
}

struct ActiveFfmpegResampler {
    context: ffmpeg::software::resampling::Context,
    format: ffmpeg::format::Sample,
    channel_layout: ffmpeg::ChannelLayout,
    rate: u32,
}

fn receive_ffmpeg_frames(
    decoder: &mut ffmpeg::decoder::Audio,
    resampler: &mut Option<ActiveFfmpegResampler>,
    output_sample_rate: u32,
    samples: &mut Vec<f32>,
) -> Result<()> {
    let mut decoded = ffmpeg::frame::Audio::empty();
    while decoder.receive_frame(&mut decoded).is_ok() {
        if decoded.channel_layout().is_empty() {
            decoded.set_channel_layout(ffmpeg::ChannelLayout::default(i32::from(
                decoded.channels(),
            )));
        }
        ensure_ffmpeg_resampler(resampler, &decoded, output_sample_rate, samples)?;
        let mut resampled = ffmpeg::frame::Audio::empty();
        resampler
            .as_mut()
            .expect("resampler is initialized")
            .context
            .run(&decoded, &mut resampled)
            .context("FFmpeg resampling failed")?;
        append_ffmpeg_samples(&resampled, samples)?;
    }
    Ok(())
}

fn ensure_ffmpeg_resampler(
    resampler: &mut Option<ActiveFfmpegResampler>,
    frame: &ffmpeg::frame::Audio,
    output_sample_rate: u32,
    samples: &mut Vec<f32>,
) -> Result<()> {
    let input_layout = frame_channel_layout(frame);
    let input_format = frame.format();
    let input_rate = frame.rate();
    let needs_rebuild = resampler.as_ref().is_none_or(|active| {
        active.format != input_format
            || active.channel_layout != input_layout
            || active.rate != input_rate
    });
    if !needs_rebuild {
        return Ok(());
    }
    flush_ffmpeg_resampler(resampler, samples)?;
    let context = ffmpeg::software::resampling::Context::get(
        input_format,
        input_layout,
        input_rate,
        ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
        ffmpeg::ChannelLayout::MONO,
        output_sample_rate,
    )
    .context("failed to create FFmpeg audio resampler")?;
    *resampler = Some(ActiveFfmpegResampler {
        context,
        format: input_format,
        channel_layout: input_layout,
        rate: input_rate,
    });
    Ok(())
}

fn frame_channel_layout(frame: &ffmpeg::frame::Audio) -> ffmpeg::ChannelLayout {
    if frame.channel_layout().is_empty() {
        ffmpeg::ChannelLayout::default(i32::from(frame.channels()))
    } else {
        frame.channel_layout()
    }
}

fn flush_ffmpeg_resampler(
    resampler: &mut Option<ActiveFfmpegResampler>,
    samples: &mut Vec<f32>,
) -> Result<()> {
    let Some(active) = resampler.as_mut() else {
        return Ok(());
    };
    loop {
        let output = *active.context.output();
        let mut resampled = ffmpeg::frame::Audio::new(output.format, 8192, output.channel_layout);
        if active
            .context
            .flush(&mut resampled)
            .context("FFmpeg resampler flush failed")?
            .is_none()
        {
            break;
        }
        append_ffmpeg_samples(&resampled, samples)?;
    }
    Ok(())
}

fn append_ffmpeg_samples(frame: &ffmpeg::frame::Audio, output: &mut Vec<f32>) -> Result<()> {
    if frame.samples() == 0 {
        return Ok(());
    }
    if !frame.is_packed()
        || frame.format() != ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed)
    {
        bail!(
            "FFmpeg resampler produced unsupported sample format {:?}",
            frame.format()
        );
    }
    output.extend_from_slice(frame.plane::<f32>(0));
    Ok(())
}

fn append_mono_samples(decoded: AudioBufferRef<'_>, output: &mut Vec<f32>, sample_rate: &mut u32) {
    let spec = *decoded.spec();
    *sample_rate = spec.rate;
    let channels = spec.channels.count().max(1);
    let mut sample_buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
    sample_buffer.copy_interleaved_ref(decoded);
    for frame in sample_buffer.samples().chunks(channels) {
        let sum: f32 = frame.iter().copied().sum();
        output.push(sum / channels as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffmpeg_library_decoder_reads_wav_file() {
        let mut file = tempfile::Builder::new().suffix(".wav").tempfile().unwrap();
        write_test_wav(&mut file, 16_000, 1_600).unwrap();

        let (samples, sample_rate) = decode_audio_ffmpeg_path(file.path(), 16_000).unwrap();

        assert_eq!(sample_rate, 16_000);
        assert_eq!(samples.len(), 1_600);
        assert!(samples.iter().any(|sample| sample.abs() > 1e-4));
    }

    fn write_test_wav(
        file: &mut tempfile::NamedTempFile,
        sample_rate: u32,
        frames: u32,
    ) -> Result<()> {
        let channels = 1u16;
        let bits_per_sample = 16u16;
        let bytes_per_sample = bits_per_sample / 8;
        let block_align = channels * bytes_per_sample;
        let byte_rate = sample_rate * u32::from(block_align);
        let data_len = frames * u32::from(block_align);
        file.write_all(b"RIFF")?;
        file.write_all(&(36 + data_len).to_le_bytes())?;
        file.write_all(b"WAVE")?;
        file.write_all(b"fmt ")?;
        file.write_all(&16u32.to_le_bytes())?;
        file.write_all(&1u16.to_le_bytes())?;
        file.write_all(&channels.to_le_bytes())?;
        file.write_all(&sample_rate.to_le_bytes())?;
        file.write_all(&byte_rate.to_le_bytes())?;
        file.write_all(&block_align.to_le_bytes())?;
        file.write_all(&bits_per_sample.to_le_bytes())?;
        file.write_all(b"data")?;
        file.write_all(&data_len.to_le_bytes())?;
        for index in 0..frames {
            let sample = (12_000.0 * (index as f32 * 0.03).sin()) as i16;
            file.write_all(&sample.to_le_bytes())?;
        }
        file.flush()?;
        Ok(())
    }
}
