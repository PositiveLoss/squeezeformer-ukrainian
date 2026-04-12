use std::fs::File;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{anyhow, bail, Context, Result};
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
    Path(PathBuf),
    Bytes(Vec<u8>, Option<String>),
}

impl AudioSource {
    pub(crate) fn log_label(&self) -> String {
        match self {
            Self::Path(path) => format!("path={}", path.display()),
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
                "symphonia decode failed for {}; falling back to ffmpeg: {symphonia_error:#}",
                source_label
            );
            decode_audio_ffmpeg(source, fallback_sample_rate).with_context(|| {
                format!("symphonia decode failed: {symphonia_error:#}; ffmpeg fallback failed")
            })
        }
        Err(error) => Err(error),
    }
}

fn decode_audio_symphonia(source: AudioSource) -> Result<(Vec<f32>, u32)> {
    let (mss, extension) = match source {
        AudioSource::Path(path) => {
            trace!("opening audio file with symphonia path={}", path.display());
            let extension = path
                .extension()
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
        "decoding audio with ffmpeg source={} output_sample_rate={}",
        source.log_label(),
        sample_rate
    );
    let sample_rate_arg = sample_rate.to_string();
    let mut command = Command::new("ffmpeg");
    command.args(["-v", "error"]);
    match source {
        AudioSource::Path(path) => {
            command.arg("-i").arg(path);
            let output = command
                .args([
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    &sample_rate_arg,
                    "-f",
                    "f32le",
                    "pipe:1",
                ])
                .output()
                .context("failed to execute ffmpeg")?;
            decode_ffmpeg_output(output, sample_rate)
        }
        AudioSource::Bytes(bytes, _) => {
            let mut child = command
                .args([
                    "-i",
                    "pipe:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    &sample_rate_arg,
                    "-f",
                    "f32le",
                    "pipe:1",
                ])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .context("failed to execute ffmpeg")?;
            {
                let mut stdin = child
                    .stdin
                    .take()
                    .ok_or_else(|| anyhow!("failed to open ffmpeg stdin"))?;
                stdin
                    .write_all(&bytes)
                    .context("failed to write audio bytes to ffmpeg stdin")?;
            }
            let output = child
                .wait_with_output()
                .context("failed to wait for ffmpeg")?;
            decode_ffmpeg_output(output, sample_rate)
        }
    }
}

fn decode_ffmpeg_output(output: std::process::Output, sample_rate: u32) -> Result<(Vec<f32>, u32)> {
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "ffmpeg exited with status {}: {}",
            output.status,
            stderr.trim()
        );
    }
    if !output.stdout.len().is_multiple_of(4) {
        bail!(
            "ffmpeg produced {} bytes, which is not divisible by f32 size",
            output.stdout.len()
        );
    }
    let samples = output
        .stdout
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk size is exact")))
        .collect::<Vec<_>>();
    if samples.is_empty() {
        bail!("ffmpeg decoded audio stream is empty");
    }
    debug!(
        "decoded audio with ffmpeg samples={} sample_rate={}",
        samples.len(),
        sample_rate
    );
    Ok((samples, sample_rate))
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
