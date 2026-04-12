use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Seek, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use arrow::array::{Array, StructArray};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use clap::{ArgAction, Parser, ValueEnum};
use log::{info, warn};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::arrow_utils::{
    column_by_name, scalar_as_bytes, scalar_as_f64, scalar_as_string, struct_child,
};
use crate::util::{hex_full, resolve_path};

const TRANSCRIPT_COLUMNS: &[&str] = &[
    "sentence",
    "transcript",
    "transcription",
    "text",
    "normalized_text",
];
const SPEAKER_COLUMNS: &[&str] = &["client_id", "speaker_id", "speaker"];
const DURATION_COLUMNS: &[&str] = &["duration", "duration_seconds", "audio_duration"];
const DEFAULT_FEATURE_SAMPLE_RATE: u32 = 16_000;
const DEFAULT_FEATURE_N_FFT: usize = 400;
const DEFAULT_FEATURE_WIN_LENGTH: usize = 400;
const DEFAULT_FEATURE_HOP_LENGTH: usize = 160;

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(crate) enum TokenizerKind {
    Character,
    Sentencepiece,
}

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Build Python-compatible disk-backed record cache files"
)]
pub(crate) struct RecordCacheCli {
    /// Dataset source to scan. Repeat to combine multiple sources.
    #[arg(long = "dataset-source", action = ArgAction::Append)]
    pub(crate) dataset_sources: Vec<PathBuf>,

    /// Validation-only dataset source. Repeat to combine multiple sources.
    #[arg(long = "validation-dataset-source", action = ArgAction::Append)]
    pub(crate) validation_dataset_sources: Vec<PathBuf>,

    /// Destination directory for train/validation JSONL and binary sidecar indexes.
    #[arg(long)]
    pub(crate) record_cache_dir: PathBuf,

    /// Split seed matching the Python training loader.
    #[arg(long, default_value_t = 13)]
    pub(crate) seed: u64,

    #[arg(long, default_value_t = 0.1)]
    pub(crate) val_fraction: f64,
    #[arg(long, default_value_t = 0.1)]
    pub(crate) test_fraction: f64,
    #[arg(long)]
    pub(crate) max_train_samples: Option<usize>,
    #[arg(long)]
    pub(crate) max_val_samples: Option<usize>,

    #[arg(long, default_value_t = 1)]
    pub(crate) min_transcript_chars: usize,
    #[arg(long, default_value_t = 400)]
    pub(crate) max_transcript_chars: usize,
    #[arg(long, default_value_t = 0.5)]
    pub(crate) max_symbol_ratio: f64,
    #[arg(long, default_value_t = 0.01)]
    pub(crate) min_audio_duration_sec: f64,
    #[arg(long, default_value_t = 30.0)]
    pub(crate) max_audio_duration_sec: f64,
    #[arg(long, default_value_t = 0.0)]
    pub(crate) min_chars_per_second: f64,
    #[arg(long, default_value_t = f64::INFINITY)]
    pub(crate) max_chars_per_second: f64,
    #[arg(long, default_value_t = 0.0)]
    pub(crate) min_words_per_second: f64,
    #[arg(long, default_value_t = f64::INFINITY)]
    pub(crate) max_words_per_second: f64,
    #[arg(long, default_value_t = 0.0)]
    pub(crate) min_duration_per_char: f64,
    #[arg(long, default_value_t = f64::INFINITY)]
    pub(crate) max_duration_per_char: f64,
    #[arg(long, default_value_t = 0.0)]
    pub(crate) min_duration_per_word: f64,
    #[arg(long, default_value_t = f64::INFINITY)]
    pub(crate) max_duration_per_word: f64,

    /// Tokenizer choice used only to mirror Python's transcript lowercasing default.
    #[arg(long, value_enum, default_value_t = TokenizerKind::Sentencepiece)]
    pub(crate) tokenizer: TokenizerKind,

    /// Preserve only records with a readable local path or embedded audio bytes.
    #[arg(long, default_value_t = false)]
    pub(crate) require_readable_audio: bool,

    /// Preserve only records with embedded audio bytes and store those bytes as cache blobs.
    #[arg(long, default_value_t = false)]
    pub(crate) require_audio_bytes: bool,

    /// Emit progress every N scanned rows per split/source. Set 0 to disable.
    #[arg(long, default_value_t = 1000)]
    pub(crate) progress_interval: usize,
}

#[derive(Debug, Default)]
struct RecordCacheCounters {
    scanned: usize,
    selected: usize,
    skipped_missing_transcript: usize,
    skipped_missing_audio: usize,
    skipped_missing_duration: usize,
    skipped_too_short: usize,
    skipped_too_long: usize,
    skipped_symbol_ratio: usize,
    skipped_no_alnum: usize,
    skipped_audio_too_short: usize,
    skipped_audio_too_long: usize,
    skipped_chars_per_second_too_low: usize,
    skipped_chars_per_second_too_high: usize,
    skipped_words_per_second_too_low: usize,
    skipped_words_per_second_too_high: usize,
    skipped_duration_per_char_too_low: usize,
    skipped_duration_per_char_too_high: usize,
    skipped_duration_per_word_too_low: usize,
    skipped_duration_per_word_too_high: usize,
    skipped_split: usize,
    skipped_unreadable_audio: usize,
}

#[derive(Debug)]
struct RecordCacheOptions {
    split: String,
    seed: u64,
    val_fraction: f64,
    test_fraction: f64,
    max_samples: Option<usize>,
    min_transcript_chars: usize,
    max_transcript_chars: usize,
    max_symbol_ratio: f64,
    min_audio_duration_sec: f64,
    max_audio_duration_sec: f64,
    min_chars_per_second: f64,
    max_chars_per_second: f64,
    min_words_per_second: f64,
    max_words_per_second: f64,
    min_duration_per_char: f64,
    max_duration_per_char: f64,
    min_duration_per_word: f64,
    max_duration_per_word: f64,
    lowercase_transcripts: bool,
    require_readable_audio: bool,
    require_audio_bytes: bool,
    progress_interval: usize,
}

#[derive(Debug)]
struct RawManifestRow {
    id: Option<String>,
    path: Option<String>,
    audio_path: Option<String>,
    audio_bytes: Option<Vec<u8>>,
    transcript: Option<String>,
    duration_seconds: Option<f64>,
    speaker_id: Option<String>,
}

#[derive(Debug)]
struct RecordCacheRecord {
    audio_path: Option<String>,
    audio_bytes: Option<Vec<u8>>,
    transcript: String,
    utterance_id: String,
    speaker_id: Option<String>,
    has_speaker_id: bool,
    estimated_frames: u32,
    num_samples: u64,
    sample_rate: u32,
}

struct RecordCacheWriter {
    records_path: PathBuf,
    temp_records_path: PathBuf,
    records: BufWriter<File>,
    offsets: BufWriter<File>,
    estimated_frames: BufWriter<File>,
    num_samples: BufWriter<File>,
    sample_rates: BufWriter<File>,
    transcript_lengths: BufWriter<File>,
    token_lengths: BufWriter<File>,
}

impl RecordCacheWriter {
    fn new(records_path: &Path) -> Result<Self> {
        if let Some(parent) = records_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create record cache dir {}", parent.display())
            })?;
        }
        let temp_records_path = temporary_record_cache_path(records_path);
        Ok(Self {
            records_path: records_path.to_path_buf(),
            temp_records_path: temp_records_path.clone(),
            records: BufWriter::new(File::create(&temp_records_path).with_context(|| {
                format!(
                    "failed to create temporary record cache {}",
                    temp_records_path.display()
                )
            })?),
            offsets: BufWriter::new(File::create(record_index_path(
                &temp_records_path,
                ".offsets.u64",
            ))?),
            estimated_frames: BufWriter::new(File::create(record_index_path(
                &temp_records_path,
                ".estimated_frames.u32",
            ))?),
            num_samples: BufWriter::new(File::create(record_index_path(
                &temp_records_path,
                ".num_samples.u64",
            ))?),
            sample_rates: BufWriter::new(File::create(record_index_path(
                &temp_records_path,
                ".sample_rates.u32",
            ))?),
            transcript_lengths: BufWriter::new(File::create(record_index_path(
                &temp_records_path,
                ".transcript_lengths.u32",
            ))?),
            token_lengths: BufWriter::new(File::create(record_index_path(
                &temp_records_path,
                ".token_lengths.u32",
            ))?),
        })
    }

    fn push(&mut self, mut record: RecordCacheRecord) -> Result<()> {
        let audio_blob_path = if let Some(audio_bytes) = record.audio_bytes.take() {
            let audio_blob_dir = self
                .records_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(format!(
                    "{}_audio_blobs",
                    self.records_path
                        .file_stem()
                        .and_then(|value| value.to_str())
                        .unwrap_or("records")
                ));
            fs::create_dir_all(&audio_blob_dir).with_context(|| {
                format!(
                    "failed to create audio blob dir {}",
                    audio_blob_dir.display()
                )
            })?;
            let blob_name = format!("{}.bin", hex_full(&Sha256::digest(&audio_bytes)));
            let blob_path = audio_blob_dir.join(blob_name);
            if !blob_path.exists() {
                fs::write(&blob_path, &audio_bytes).with_context(|| {
                    format!("failed to write audio blob {}", blob_path.display())
                })?;
            }
            Some(relative_blob_path(&blob_path, self.records_path.parent()))
        } else {
            None
        };

        let offset = self.records.stream_position()?;
        self.offsets.write_all(&offset.to_le_bytes())?;
        let payload = json!({
            "audio_path": record.audio_path,
            "audio_blob_path": audio_blob_path,
            "transcript": record.transcript,
            "utterance_id": record.utterance_id,
            "speaker_id": record.speaker_id,
            "has_speaker_id": record.has_speaker_id,
        });
        serde_json::to_writer(&mut self.records, &payload)?;
        self.records.write_all(b"\n")?;
        self.estimated_frames
            .write_all(&record.estimated_frames.to_le_bytes())?;
        self.num_samples
            .write_all(&record.num_samples.to_le_bytes())?;
        self.sample_rates
            .write_all(&record.sample_rate.to_le_bytes())?;
        self.transcript_lengths
            .write_all(&(record.transcript.chars().count() as u32).to_le_bytes())?;
        self.token_lengths.write_all(&0u32.to_le_bytes())?;
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        self.records.flush()?;
        self.offsets.flush()?;
        self.estimated_frames.flush()?;
        self.num_samples.flush()?;
        self.sample_rates.flush()?;
        self.transcript_lengths.flush()?;
        self.token_lengths.flush()?;
        self.records.get_ref().sync_all()?;
        self.offsets.get_ref().sync_all()?;
        self.estimated_frames.get_ref().sync_all()?;
        self.num_samples.get_ref().sync_all()?;
        self.sample_rates.get_ref().sync_all()?;
        self.transcript_lengths.get_ref().sync_all()?;
        self.token_lengths.get_ref().sync_all()?;

        for suffix in [
            ".offsets.u64",
            ".estimated_frames.u32",
            ".num_samples.u64",
            ".sample_rates.u32",
            ".transcript_lengths.u32",
            ".token_lengths.u32",
        ] {
            publish_temp_file(
                &record_index_path(&self.temp_records_path, suffix),
                &record_index_path(&self.records_path, suffix),
            )?;
        }
        publish_temp_file(&self.temp_records_path, &self.records_path)?;
        Ok(())
    }
}

pub(crate) fn run_record_cache_cli(cli: RecordCacheCli) -> Result<()> {
    validate_record_cache_cli(&cli)?;
    let lowercase_transcripts = cli.tokenizer != TokenizerKind::Sentencepiece;
    let train_sources = dedupe_existing_sources(&cli.dataset_sources)?;
    if train_sources.is_empty() {
        bail!("record-cache requires at least one --dataset-source");
    }
    let validation_sources = dedupe_existing_sources(&cli.validation_dataset_sources)?;
    let use_external_validation = !validation_sources.is_empty();
    let train_val_fraction = if use_external_validation {
        0.0
    } else {
        cli.val_fraction
    };
    let train_test_fraction = if use_external_validation {
        0.0
    } else {
        cli.test_fraction
    };
    let validation_split = if use_external_validation {
        "train"
    } else {
        "validation"
    };
    let validation_val_fraction = if use_external_validation {
        0.0
    } else {
        cli.val_fraction
    };
    let validation_test_fraction = if use_external_validation {
        0.0
    } else {
        cli.test_fraction
    };
    fs::create_dir_all(&cli.record_cache_dir).with_context(|| {
        format!(
            "failed to create record cache dir {}",
            cli.record_cache_dir.display()
        )
    })?;

    info!(
        "building record cache dir={} train_sources={} validation_sources={} external_validation={} lowercase_transcripts={}",
        cli.record_cache_dir.display(),
        train_sources.len(),
        validation_sources.len(),
        use_external_validation,
        lowercase_transcripts
    );
    build_record_store(
        &train_sources,
        &cli.record_cache_dir.join("train.jsonl"),
        &RecordCacheOptions {
            split: "train".to_string(),
            seed: cli.seed,
            val_fraction: train_val_fraction,
            test_fraction: train_test_fraction,
            max_samples: cli.max_train_samples,
            min_transcript_chars: cli.min_transcript_chars,
            max_transcript_chars: cli.max_transcript_chars,
            max_symbol_ratio: cli.max_symbol_ratio,
            min_audio_duration_sec: cli.min_audio_duration_sec,
            max_audio_duration_sec: cli.max_audio_duration_sec,
            min_chars_per_second: cli.min_chars_per_second,
            max_chars_per_second: cli.max_chars_per_second,
            min_words_per_second: cli.min_words_per_second,
            max_words_per_second: cli.max_words_per_second,
            min_duration_per_char: cli.min_duration_per_char,
            max_duration_per_char: cli.max_duration_per_char,
            min_duration_per_word: cli.min_duration_per_word,
            max_duration_per_word: cli.max_duration_per_word,
            lowercase_transcripts,
            require_readable_audio: cli.require_readable_audio,
            require_audio_bytes: cli.require_audio_bytes,
            progress_interval: cli.progress_interval,
        },
    )?;
    build_record_store(
        if use_external_validation {
            &validation_sources
        } else {
            &train_sources
        },
        &cli.record_cache_dir.join("validation.jsonl"),
        &RecordCacheOptions {
            split: validation_split.to_string(),
            seed: cli.seed,
            val_fraction: validation_val_fraction,
            test_fraction: validation_test_fraction,
            max_samples: cli.max_val_samples,
            min_transcript_chars: cli.min_transcript_chars,
            max_transcript_chars: cli.max_transcript_chars,
            max_symbol_ratio: cli.max_symbol_ratio,
            min_audio_duration_sec: cli.min_audio_duration_sec,
            max_audio_duration_sec: cli.max_audio_duration_sec,
            min_chars_per_second: cli.min_chars_per_second,
            max_chars_per_second: cli.max_chars_per_second,
            min_words_per_second: cli.min_words_per_second,
            max_words_per_second: cli.max_words_per_second,
            min_duration_per_char: cli.min_duration_per_char,
            max_duration_per_char: cli.max_duration_per_char,
            min_duration_per_word: cli.min_duration_per_word,
            max_duration_per_word: cli.max_duration_per_word,
            lowercase_transcripts,
            require_readable_audio: cli.require_readable_audio,
            require_audio_bytes: cli.require_audio_bytes,
            progress_interval: cli.progress_interval,
        },
    )?;
    info!(
        "record cache complete dir={}",
        cli.record_cache_dir.display()
    );
    Ok(())
}

fn validate_record_cache_cli(cli: &RecordCacheCli) -> Result<()> {
    if cli.val_fraction < 0.0 || cli.test_fraction < 0.0 {
        bail!("--val-fraction and --test-fraction must be non-negative");
    }
    if cli.val_fraction + cli.test_fraction >= 1.0 {
        bail!("--val-fraction + --test-fraction must be < 1");
    }
    if cli.min_transcript_chars < 1 {
        bail!("--min-transcript-chars must be >= 1");
    }
    if cli.max_transcript_chars < cli.min_transcript_chars {
        bail!("--max-transcript-chars must be >= --min-transcript-chars");
    }
    if cli.min_audio_duration_sec <= 0.0 {
        bail!("--min-audio-duration-sec must be > 0");
    }
    if cli.max_audio_duration_sec < cli.min_audio_duration_sec {
        bail!("--max-audio-duration-sec must be >= --min-audio-duration-sec");
    }
    if cli.max_chars_per_second < cli.min_chars_per_second {
        bail!("--max-chars-per-second must be >= --min-chars-per-second");
    }
    if cli.max_words_per_second < cli.min_words_per_second {
        bail!("--max-words-per-second must be >= --min-words-per-second");
    }
    if cli.max_duration_per_char < cli.min_duration_per_char {
        bail!("--max-duration-per-char must be >= --min-duration-per-char");
    }
    if cli.max_duration_per_word < cli.min_duration_per_word {
        bail!("--max-duration-per-word must be >= --min-duration-per-word");
    }
    Ok(())
}

fn dedupe_existing_sources(sources: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut resolved = Vec::new();
    let mut seen = HashMap::new();
    for source in sources {
        let canonical = source
            .canonicalize()
            .with_context(|| format!("dataset source does not exist: {}", source.display()))?;
        if seen.insert(canonical.clone(), ()).is_none() {
            resolved.push(canonical);
        }
    }
    Ok(resolved)
}

fn build_record_store(
    sources: &[PathBuf],
    records_path: &Path,
    options: &RecordCacheOptions,
) -> Result<()> {
    let mut writer = RecordCacheWriter::new(records_path)?;
    let mut written = 0usize;
    let mut aggregate = RecordCacheCounters::default();
    for source in sources {
        if let Some(max_samples) = options.max_samples {
            if written >= max_samples {
                break;
            }
        }
        let remaining = options.max_samples.map(|max_samples| max_samples - written);
        let counters = build_record_store_from_source(source, &mut writer, options, remaining)?;
        written += counters.selected;
        merge_record_counters(&mut aggregate, &counters);
    }
    writer.finish()?;
    if aggregate.skipped_unreadable_audio > 0 {
        warn!(
            "record cache skipped unreadable audio records path={} skipped={}",
            records_path.display(),
            aggregate.skipped_unreadable_audio
        );
    }
    if written == 0 {
        bail!(
            "Split '{}' is empty after applying the current split fractions across all dataset sources.",
            options.split
        );
    }
    info!(
        "record cache split complete path={} split={} scanned={} selected={} skipped_split={} skipped_missing_transcript={} skipped_missing_audio={} skipped_missing_duration={}",
        records_path.display(),
        options.split,
        aggregate.scanned,
        aggregate.selected,
        aggregate.skipped_split,
        aggregate.skipped_missing_transcript,
        aggregate.skipped_missing_audio,
        aggregate.skipped_missing_duration
    );
    Ok(())
}

fn build_record_store_from_source(
    source: &Path,
    writer: &mut RecordCacheWriter,
    options: &RecordCacheOptions,
    remaining: Option<usize>,
) -> Result<RecordCacheCounters> {
    let manifest_paths = collect_manifest_paths_for_records(source)?;
    let source_base = if source.is_dir() {
        source.to_path_buf()
    } else {
        source.parent().map(Path::to_path_buf).unwrap_or_default()
    };
    let mut counters = RecordCacheCounters::default();
    for manifest_path in manifest_paths {
        if remaining.is_some_and(|remaining| counters.selected >= remaining) {
            break;
        }
        match manifest_path
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase())
            .as_deref()
        {
            Some("parquet") => read_record_cache_parquet(
                &manifest_path,
                &source_base,
                writer,
                options,
                remaining,
                &mut counters,
            )?,
            Some("tsv") => read_record_cache_tsv(
                &manifest_path,
                &source_base,
                writer,
                options,
                remaining,
                &mut counters,
            )?,
            _ => bail!("unsupported manifest file: {}", manifest_path.display()),
        }
    }
    info!(
        "record loader summary source={} split={} scanned={} selected={} skipped_missing_transcript={} skipped_missing_audio={} skipped_missing_duration={} skipped_too_short={} skipped_too_long={} skipped_symbol_ratio={} skipped_no_alnum={} skipped_audio_too_short={} skipped_audio_too_long={} skipped_chars_per_second_too_low={} skipped_chars_per_second_too_high={} skipped_words_per_second_too_low={} skipped_words_per_second_too_high={} skipped_duration_per_char_too_low={} skipped_duration_per_char_too_high={} skipped_duration_per_word_too_low={} skipped_duration_per_word_too_high={} skipped_split={} max_samples={}",
        source.display(),
        options.split,
        counters.scanned,
        counters.selected,
        counters.skipped_missing_transcript,
        counters.skipped_missing_audio,
        counters.skipped_missing_duration,
        counters.skipped_too_short,
        counters.skipped_too_long,
        counters.skipped_symbol_ratio,
        counters.skipped_no_alnum,
        counters.skipped_audio_too_short,
        counters.skipped_audio_too_long,
        counters.skipped_chars_per_second_too_low,
        counters.skipped_chars_per_second_too_high,
        counters.skipped_words_per_second_too_low,
        counters.skipped_words_per_second_too_high,
        counters.skipped_duration_per_char_too_low,
        counters.skipped_duration_per_char_too_high,
        counters.skipped_duration_per_word_too_low,
        counters.skipped_duration_per_word_too_high,
        counters.skipped_split,
        remaining
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string())
    );
    Ok(counters)
}

fn read_record_cache_parquet(
    manifest_path: &Path,
    source_base: &Path,
    writer: &mut RecordCacheWriter,
    options: &RecordCacheOptions,
    remaining: Option<usize>,
    counters: &mut RecordCacheCounters,
) -> Result<()> {
    info!("loading parquet manifest {}", manifest_path.display());
    let input = File::open(manifest_path).with_context(|| {
        format!(
            "failed to open parquet manifest {}",
            manifest_path.display()
        )
    })?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(input)?
        .with_batch_size(8192)
        .build()?;
    for batch_result in reader {
        let batch = batch_result?;
        for row_index in 0..batch.num_rows() {
            if remaining.is_some_and(|remaining| counters.selected >= remaining) {
                return Ok(());
            }
            counters.scanned += 1;
            let row = RawManifestRow::from_batch(&batch, row_index);
            maybe_write_record(row, source_base, writer, options, counters)?;
        }
    }
    Ok(())
}

fn read_record_cache_tsv(
    manifest_path: &Path,
    source_base: &Path,
    writer: &mut RecordCacheWriter,
    options: &RecordCacheOptions,
    remaining: Option<usize>,
    counters: &mut RecordCacheCounters,
) -> Result<()> {
    info!("loading TSV manifest {}", manifest_path.display());
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(manifest_path)
        .with_context(|| format!("failed to open TSV manifest {}", manifest_path.display()))?;
    let headers = reader.headers()?.clone();
    for row_result in reader.records() {
        if remaining.is_some_and(|remaining| counters.selected >= remaining) {
            return Ok(());
        }
        counters.scanned += 1;
        let row = row_result?;
        let get = |name: &str| {
            headers
                .iter()
                .position(|candidate| candidate == name)
                .and_then(|index| row.get(index))
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        };
        let transcript = TRANSCRIPT_COLUMNS.iter().find_map(|column| get(column));
        let duration_seconds = DURATION_COLUMNS
            .iter()
            .find_map(|column| get(column).and_then(|value| value.parse::<f64>().ok()));
        let speaker_id = SPEAKER_COLUMNS.iter().find_map(|column| get(column));
        maybe_write_record(
            RawManifestRow {
                id: get("id"),
                path: get("path"),
                audio_path: None,
                audio_bytes: None,
                transcript,
                duration_seconds,
                speaker_id,
            },
            source_base,
            writer,
            options,
            counters,
        )?;
    }
    Ok(())
}

impl RawManifestRow {
    fn from_batch(batch: &RecordBatch, row_index: usize) -> Self {
        let transcript = TRANSCRIPT_COLUMNS.iter().find_map(|column| {
            column_by_name(batch, &[*column])
                .and_then(|array| scalar_as_string(array.as_ref(), row_index))
                .filter(|value| !value.trim().is_empty())
        });
        let duration_seconds = DURATION_COLUMNS.iter().find_map(|column| {
            column_by_name(batch, &[*column])
                .and_then(|array| scalar_as_f64(array.as_ref(), row_index))
        });
        let speaker_id = SPEAKER_COLUMNS.iter().find_map(|column| {
            column_by_name(batch, &[*column])
                .and_then(|array| scalar_as_string(array.as_ref(), row_index))
                .filter(|value| !value.is_empty())
        });
        let mut audio_path = None;
        let mut audio_bytes = None;
        if let Some(audio_array) = column_by_name(batch, &["audio"]) {
            match audio_array.data_type() {
                DataType::Struct(_) => {
                    if let Some(struct_array) = audio_array.as_any().downcast_ref::<StructArray>() {
                        if !struct_array.is_null(row_index) {
                            audio_bytes = struct_child(struct_array, "bytes")
                                .and_then(|array| scalar_as_bytes(array.as_ref(), row_index));
                            audio_path = struct_child(struct_array, "path")
                                .and_then(|array| scalar_as_string(array.as_ref(), row_index))
                                .filter(|value| !value.is_empty());
                        }
                    }
                }
                DataType::Binary | DataType::LargeBinary => {
                    audio_bytes = scalar_as_bytes(audio_array.as_ref(), row_index);
                }
                _ => {}
            }
        }
        Self {
            id: column_by_name(batch, &["id", "utterance_id"])
                .and_then(|array| scalar_as_string(array.as_ref(), row_index))
                .filter(|value| !value.is_empty()),
            path: column_by_name(batch, &["path"])
                .and_then(|array| scalar_as_string(array.as_ref(), row_index))
                .filter(|value| !value.is_empty()),
            audio_path,
            audio_bytes,
            transcript,
            duration_seconds,
            speaker_id,
        }
    }
}

fn maybe_write_record(
    row: RawManifestRow,
    source_base: &Path,
    writer: &mut RecordCacheWriter,
    options: &RecordCacheOptions,
    counters: &mut RecordCacheCounters,
) -> Result<()> {
    let Some(transcript) = row
        .transcript
        .as_deref()
        .map(|text| normalize_transcript(text, options.lowercase_transcripts))
        .filter(|text| !text.is_empty())
    else {
        counters.skipped_missing_transcript += 1;
        return Ok(());
    };
    let (audio_path, audio_bytes) =
        match resolve_record_audio(row.path, row.audio_path, row.audio_bytes, source_base) {
            Some(source) => source,
            None => {
                counters.skipped_missing_audio += 1;
                return Ok(());
            }
        };
    let Some(duration_seconds) = row.duration_seconds else {
        counters.skipped_missing_duration += 1;
        return Ok(());
    };
    if duration_seconds < options.min_audio_duration_sec {
        counters.skipped_audio_too_short += 1;
        return Ok(());
    }
    if duration_seconds > options.max_audio_duration_sec {
        counters.skipped_audio_too_long += 1;
        return Ok(());
    }
    if let Some(reason) = transcript_rejection_reason(
        &transcript,
        options.min_transcript_chars,
        options.max_transcript_chars,
        options.max_symbol_ratio,
    ) {
        match reason {
            "too_short" => counters.skipped_too_short += 1,
            "too_long" => counters.skipped_too_long += 1,
            "symbol_ratio" => counters.skipped_symbol_ratio += 1,
            "no_alnum" => counters.skipped_no_alnum += 1,
            _ => {}
        }
        return Ok(());
    }
    if let Some(reason) = alignment_rejection_reason(&transcript, duration_seconds, options) {
        match reason {
            "chars_per_second_too_low" => counters.skipped_chars_per_second_too_low += 1,
            "chars_per_second_too_high" => counters.skipped_chars_per_second_too_high += 1,
            "words_per_second_too_low" => counters.skipped_words_per_second_too_low += 1,
            "words_per_second_too_high" => counters.skipped_words_per_second_too_high += 1,
            "duration_per_char_too_low" => counters.skipped_duration_per_char_too_low += 1,
            "duration_per_char_too_high" => counters.skipped_duration_per_char_too_high += 1,
            "duration_per_word_too_low" => counters.skipped_duration_per_word_too_low += 1,
            "duration_per_word_too_high" => counters.skipped_duration_per_word_too_high += 1,
            _ => {}
        }
        return Ok(());
    }
    let utterance_id = row
        .id
        .filter(|value| !value.is_empty())
        .or_else(|| audio_path.clone())
        .unwrap_or_else(|| counters.scanned.to_string());
    let speaker_id = row.speaker_id.filter(|value| !value.is_empty());
    let split_key = speaker_id.as_deref().unwrap_or(&utterance_id);
    if !record_split_matches(
        split_key,
        &options.split,
        options.seed,
        options.val_fraction,
        options.test_fraction,
    )? {
        counters.skipped_split += 1;
        return Ok(());
    }
    if options.require_audio_bytes && audio_bytes.is_none() {
        counters.skipped_unreadable_audio += 1;
        return Ok(());
    }
    if options.require_readable_audio && audio_bytes.is_none() {
        match audio_path.as_deref() {
            Some(path) if path.starts_with("http://") || path.starts_with("https://") => {}
            Some(path) if Path::new(path).exists() => {}
            _ => {
                counters.skipped_unreadable_audio += 1;
                return Ok(());
            }
        }
    }
    let preserve_audio_bytes = audio_bytes.is_some()
        && (options.require_audio_bytes
            || !audio_path.as_deref().is_some_and(|path| {
                !path.starts_with("http://")
                    && !path.starts_with("https://")
                    && Path::new(path).exists()
            }));
    let record_audio_bytes = if preserve_audio_bytes {
        audio_bytes
    } else {
        None
    };
    let num_samples =
        python_round_half_even(duration_seconds * DEFAULT_FEATURE_SAMPLE_RATE as f64).max(1) as u64;
    let estimated_frames = estimate_default_feature_frames(num_samples as usize);
    writer.push(RecordCacheRecord {
        audio_path,
        audio_bytes: record_audio_bytes,
        transcript,
        utterance_id,
        speaker_id: speaker_id.clone(),
        has_speaker_id: speaker_id.is_some(),
        estimated_frames,
        num_samples,
        sample_rate: DEFAULT_FEATURE_SAMPLE_RATE,
    })?;
    counters.selected += 1;
    if options.progress_interval > 0 && counters.scanned.is_multiple_of(options.progress_interval) {
        info!(
            "record cache progress split={} scanned={} selected={} skipped_split={}",
            options.split, counters.scanned, counters.selected, counters.skipped_split
        );
    }
    Ok(())
}

fn collect_manifest_paths_for_records(source: &Path) -> Result<Vec<PathBuf>> {
    if source.is_file() {
        return Ok(vec![source.to_path_buf()]);
    }
    if !source.is_dir() {
        bail!(
            "dataset source must be a local file or directory: {}",
            source.display()
        );
    }
    let mut tsv_paths = Vec::new();
    collect_paths_with_extension(source, "tsv", &mut tsv_paths)?;
    tsv_paths.sort();
    if !tsv_paths.is_empty() {
        return Ok(tsv_paths);
    }
    let mut parquet_paths = Vec::new();
    collect_paths_with_extension(source, "parquet", &mut parquet_paths)?;
    parquet_paths.sort();
    if parquet_paths.is_empty() {
        bail!(
            "no TSV or Parquet manifest files found under {}",
            source.display()
        );
    }
    Ok(parquet_paths)
}

fn collect_paths_with_extension(
    directory: &Path,
    extension: &str,
    paths: &mut Vec<PathBuf>,
) -> Result<()> {
    for entry in fs::read_dir(directory)
        .with_context(|| format!("failed to read dataset source {}", directory.display()))?
    {
        let path = entry?.path();
        if path.is_dir() {
            collect_paths_with_extension(&path, extension, paths)?;
        } else if path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case(extension))
        {
            paths.push(path);
        }
    }
    Ok(())
}

fn resolve_record_audio(
    top_level_path: Option<String>,
    audio_path: Option<String>,
    audio_bytes: Option<Vec<u8>>,
    source_base: &Path,
) -> Option<(Option<String>, Option<Vec<u8>>)> {
    if let Some(path) = top_level_path.filter(|value| !value.is_empty()) {
        return Some((Some(resolve_path_or_url(source_base, &path)), None));
    }
    let resolved_audio_path = audio_path
        .filter(|value| !value.is_empty())
        .map(|path| resolve_path_or_url(source_base, &path));
    if audio_bytes.is_some() || resolved_audio_path.is_some() {
        return Some((resolved_audio_path, audio_bytes));
    }
    None
}

fn resolve_path_or_url(source_base: &Path, path: &str) -> String {
    if path.starts_with("http://") || path.starts_with("https://") {
        return path.to_string();
    }
    resolve_path(source_base, path)
        .to_string_lossy()
        .to_string()
}

fn normalize_transcript(text: &str, lowercase: bool) -> String {
    let mut normalized = text.trim().to_string();
    if lowercase {
        normalized = normalized
            .chars()
            .flat_map(|character| character.to_lowercase())
            .collect();
    }
    normalized = normalized
        .replace(['’', '`', 'ʼ'], "'")
        .replace(['“', '”', '«', '»'], "\"");
    let collapsed = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
    strip_space_before_punctuation(&collapsed)
}

fn strip_space_before_punctuation(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    for character in text.chars() {
        if matches!(character, ',' | '.' | ';' | ':' | '!' | '?') && output.ends_with(' ') {
            output.pop();
        }
        output.push(character);
    }
    output
}

fn transcript_symbol_ratio(text: &str) -> f64 {
    if text.is_empty() {
        return 1.0;
    }
    let total = text.chars().count();
    let noisy = text
        .chars()
        .filter(|character| {
            !(character.is_alphanumeric()
                || character.is_whitespace()
                || matches!(character, '\'' | '-'))
        })
        .count();
    noisy as f64 / total as f64
}

fn transcript_rejection_reason(
    text: &str,
    min_chars: usize,
    max_chars: usize,
    max_symbol_ratio: f64,
) -> Option<&'static str> {
    let char_count = text.chars().count();
    if char_count < min_chars {
        return Some("too_short");
    }
    if char_count > max_chars {
        return Some("too_long");
    }
    if transcript_symbol_ratio(text) > max_symbol_ratio {
        return Some("symbol_ratio");
    }
    if !text.chars().any(|character| character.is_alphanumeric()) {
        return Some("no_alnum");
    }
    None
}

fn alignment_rejection_reason(
    text: &str,
    duration_seconds: f64,
    options: &RecordCacheOptions,
) -> Option<&'static str> {
    if duration_seconds <= 0.0 {
        return None;
    }
    let char_count = text
        .chars()
        .filter(|character| !character.is_whitespace())
        .count();
    let word_count = text.split_whitespace().count();
    if char_count == 0 || word_count == 0 {
        return None;
    }
    let chars_per_second = char_count as f64 / duration_seconds;
    if chars_per_second < options.min_chars_per_second {
        return Some("chars_per_second_too_low");
    }
    if chars_per_second > options.max_chars_per_second {
        return Some("chars_per_second_too_high");
    }
    let words_per_second = word_count as f64 / duration_seconds;
    if words_per_second < options.min_words_per_second {
        return Some("words_per_second_too_low");
    }
    if words_per_second > options.max_words_per_second {
        return Some("words_per_second_too_high");
    }
    let duration_per_char = duration_seconds / char_count as f64;
    if duration_per_char < options.min_duration_per_char {
        return Some("duration_per_char_too_low");
    }
    if duration_per_char > options.max_duration_per_char {
        return Some("duration_per_char_too_high");
    }
    let duration_per_word = duration_seconds / word_count as f64;
    if duration_per_word < options.min_duration_per_word {
        return Some("duration_per_word_too_low");
    }
    if duration_per_word > options.max_duration_per_word {
        return Some("duration_per_word_too_high");
    }
    None
}

fn record_split_matches(
    split_key: &str,
    split: &str,
    seed: u64,
    val_fraction: f64,
    test_fraction: f64,
) -> Result<bool> {
    let train_cutoff = (1.0 - val_fraction - test_fraction).max(0.0);
    let digest = Sha256::digest(format!("{seed}:{split_key}").as_bytes());
    let mut prefix = [0u8; 8];
    prefix.copy_from_slice(&digest[..8]);
    let score = u64::from_be_bytes(prefix) as f64 / 16_f64.powi(16);
    match split {
        "train" => Ok(score < train_cutoff),
        "validation" => Ok(score >= train_cutoff && score < train_cutoff + val_fraction),
        "test" => Ok(score >= train_cutoff + val_fraction),
        _ => bail!("unsupported split: {split}"),
    }
}

fn python_round_half_even(value: f64) -> i64 {
    let floor = value.floor();
    let fraction = value - floor;
    if (fraction - 0.5).abs() < f64::EPSILON {
        let floor_i = floor as i64;
        if floor_i % 2 == 0 {
            floor_i
        } else {
            floor_i + 1
        }
    } else {
        value.round() as i64
    }
}

fn estimate_default_feature_frames(num_samples: usize) -> u32 {
    let effective_samples = num_samples.max(DEFAULT_FEATURE_N_FFT.max(DEFAULT_FEATURE_WIN_LENGTH));
    ((effective_samples / DEFAULT_FEATURE_HOP_LENGTH) + 1) as u32
}

fn record_index_path(records_path: &Path, suffix: &str) -> PathBuf {
    let mut path = records_path.as_os_str().to_os_string();
    path.push(suffix);
    PathBuf::from(path)
}

fn temporary_record_cache_path(records_path: &Path) -> PathBuf {
    let temp_name = format!(
        "{}.tmp.{}",
        records_path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("records.jsonl"),
        std::process::id()
    );
    records_path
        .parent()
        .map(|parent| parent.join(&temp_name))
        .unwrap_or_else(|| PathBuf::from(temp_name))
}

fn publish_temp_file(temp_path: &Path, final_path: &Path) -> Result<()> {
    if final_path.exists() {
        fs::remove_file(final_path).with_context(|| {
            format!(
                "failed to remove previous record cache file {}",
                final_path.display()
            )
        })?;
    }
    fs::rename(temp_path, final_path).with_context(|| {
        format!(
            "failed to publish record cache file {} to {}",
            temp_path.display(),
            final_path.display()
        )
    })?;
    Ok(())
}

fn relative_blob_path(blob_path: &Path, base: Option<&Path>) -> String {
    if let Some(base) = base {
        if let Ok(relative) = blob_path.strip_prefix(base) {
            return relative.to_string_lossy().to_string();
        }
    }
    blob_path.to_string_lossy().to_string()
}

fn merge_record_counters(target: &mut RecordCacheCounters, source: &RecordCacheCounters) {
    target.scanned += source.scanned;
    target.selected += source.selected;
    target.skipped_missing_transcript += source.skipped_missing_transcript;
    target.skipped_missing_audio += source.skipped_missing_audio;
    target.skipped_missing_duration += source.skipped_missing_duration;
    target.skipped_too_short += source.skipped_too_short;
    target.skipped_too_long += source.skipped_too_long;
    target.skipped_symbol_ratio += source.skipped_symbol_ratio;
    target.skipped_no_alnum += source.skipped_no_alnum;
    target.skipped_audio_too_short += source.skipped_audio_too_short;
    target.skipped_audio_too_long += source.skipped_audio_too_long;
    target.skipped_chars_per_second_too_low += source.skipped_chars_per_second_too_low;
    target.skipped_chars_per_second_too_high += source.skipped_chars_per_second_too_high;
    target.skipped_words_per_second_too_low += source.skipped_words_per_second_too_low;
    target.skipped_words_per_second_too_high += source.skipped_words_per_second_too_high;
    target.skipped_duration_per_char_too_low += source.skipped_duration_per_char_too_low;
    target.skipped_duration_per_char_too_high += source.skipped_duration_per_char_too_high;
    target.skipped_duration_per_word_too_low += source.skipped_duration_per_word_too_low;
    target.skipped_duration_per_word_too_high += source.skipped_duration_per_word_too_high;
    target.skipped_split += source.skipped_split;
    target.skipped_unreadable_audio += source.skipped_unreadable_audio;
}
