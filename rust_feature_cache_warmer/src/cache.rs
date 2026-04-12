use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use arrow::array::{ArrayRef, BinaryBuilder, BooleanBuilder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use log::debug;
use parquet::arrow::ArrowWriter;
use sha2::{Digest, Sha256};

use crate::frontend::FeatureMatrix;
use crate::util::hex_full;

pub(crate) const RUST_PAYLOAD_MAGIC: &[u8; 8] = b"SFCF32L1";
#[derive(Debug)]
pub(crate) struct CacheRow {
    pub(crate) key: String,
    pub(crate) payload: Vec<u8>,
}

pub(crate) fn encode_feature_payload(features: &FeatureMatrix) -> Result<Vec<u8>> {
    let rows: u32 = features
        .rows
        .try_into()
        .context("feature row count does not fit into u32")?;
    let cols: u32 = features
        .cols
        .try_into()
        .context("feature column count does not fit into u32")?;
    let mut payload = Vec::with_capacity(RUST_PAYLOAD_MAGIC.len() + 8 + features.values.len() * 4);
    payload.extend_from_slice(RUST_PAYLOAD_MAGIC);
    payload.extend_from_slice(&rows.to_le_bytes());
    payload.extend_from_slice(&cols.to_le_bytes());
    for value in &features.values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    Ok(payload)
}

pub(crate) fn cache_key(utterance_id: &str, frontend_hash: &str) -> String {
    let digest = Sha256::digest(format!("{utterance_id}:{frontend_hash}").as_bytes());
    hex_full(&digest)
}

pub(crate) struct ShardedCacheWriter {
    shard_dir: PathBuf,
    num_shards: usize,
    rows_per_part: usize,
    pending: HashMap<usize, Vec<CacheRow>>,
    counters: HashMap<usize, usize>,
    schema: Arc<Schema>,
}

impl ShardedCacheWriter {
    pub(crate) fn new(root: &Path, num_shards: usize, rows_per_part: usize) -> Result<Self> {
        let shard_dir = root.join("feature_shards");
        fs::create_dir_all(&shard_dir)
            .with_context(|| format!("failed to create {}", shard_dir.display()))?;
        debug!(
            "initialized sharded cache writer shard_dir={} num_shards={} rows_per_part={}",
            shard_dir.display(),
            num_shards,
            rows_per_part
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("payload", DataType::Binary, true),
            Field::new("deleted", DataType::Boolean, false),
        ]));
        Ok(Self {
            shard_dir,
            num_shards,
            rows_per_part,
            pending: HashMap::new(),
            counters: HashMap::new(),
            schema,
        })
    }

    pub(crate) fn push(&mut self, row: CacheRow) -> Result<()> {
        let shard_index = shard_index(&row.key, self.num_shards)?;
        let rows = self.pending.entry(shard_index).or_default();
        rows.push(row);
        if rows.len() >= self.rows_per_part {
            self.flush_shard(shard_index)?;
        }
        Ok(())
    }

    pub(crate) fn finish(&mut self) -> Result<()> {
        let shard_indices: Vec<usize> = self.pending.keys().copied().collect();
        debug!("flushing pending shards count={}", shard_indices.len());
        for shard_index in shard_indices {
            self.flush_shard(shard_index)?;
        }
        Ok(())
    }

    fn flush_shard(&mut self, shard_index: usize) -> Result<()> {
        let rows = self.pending.remove(&shard_index).unwrap_or_default();
        if rows.is_empty() {
            return Ok(());
        }
        let row_count = rows.len();
        let output_dir = self.shard_dir.join(format!("features_{shard_index:02}"));
        fs::create_dir_all(&output_dir)
            .with_context(|| format!("failed to create {}", output_dir.display()))?;
        let part_counter = self.counters.entry(shard_index).or_insert(0);
        *part_counter += 1;
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let output_path = output_dir.join(format!(
            "part_rust_{}_{}_{:06}.parquet",
            std::process::id(),
            now,
            *part_counter
        ));

        let mut key_builder = StringBuilder::new();
        let mut payload_builder = BinaryBuilder::new();
        let mut deleted_builder = BooleanBuilder::new();
        for row in rows {
            key_builder.append_value(row.key);
            payload_builder.append_value(row.payload);
            deleted_builder.append_value(false);
        }
        let batch = RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(key_builder.finish()) as ArrayRef,
                Arc::new(payload_builder.finish()) as ArrayRef,
                Arc::new(deleted_builder.finish()) as ArrayRef,
            ],
        )?;
        let file = File::create(&output_path)
            .with_context(|| format!("failed to create {}", output_path.display()))?;
        let mut writer = ArrowWriter::try_new(file, self.schema.clone(), None)?;
        writer.write(&batch)?;
        writer.close()?;
        debug!(
            "flushed cache shard={} rows={} path={}",
            shard_index,
            row_count,
            output_path.display()
        );
        Ok(())
    }
}

pub(crate) fn shard_index(key: &str, num_shards: usize) -> Result<usize> {
    if key.len() < 8 {
        bail!("cache key is shorter than 8 hex characters: {key}");
    }
    let prefix = u32::from_str_radix(&key[..8], 16)?;
    Ok(prefix as usize % num_shards)
}
