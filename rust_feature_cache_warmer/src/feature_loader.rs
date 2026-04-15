use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use anyhow::{anyhow, bail, Context, Result};
use arrow::array::{Array, ArrayRef, Int64Builder, StringBuilder, UInt64Builder, UInt8Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use numpy::ndarray::Array2;
use numpy::IntoPyArray;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

use crate::arrow_utils::{column_by_name, scalar_as_bytes, scalar_as_string};
use crate::cache::RUST_PAYLOAD_MAGIC;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FeatureLocation {
    pub(crate) path: PathBuf,
    pub(crate) row_index: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PartMetadata {
    path: PathBuf,
    mtime_ns: i64,
    size: u64,
}

#[derive(Clone, Debug)]
struct PartUpdate {
    order: usize,
    path: PathBuf,
    rows: Vec<PartRowUpdate>,
}

#[derive(Clone, Debug)]
struct PartRowUpdate {
    key: String,
    row_index: usize,
    deleted: bool,
}

#[pyclass]
pub(crate) struct RustParquetFeatureCacheReader {
    index: HashMap<String, FeatureLocation>,
}

#[pymethods]
impl RustParquetFeatureCacheReader {
    #[new]
    #[pyo3(signature = (cache_root, *, num_shards=64))]
    fn new(cache_root: PathBuf, num_shards: usize) -> PyResult<Self> {
        let index = build_feature_index(&cache_root, num_shards).map_err(py_error)?;
        Ok(Self { index })
    }

    fn len(&self) -> usize {
        self.index.len()
    }

    fn fetch_many<'py>(&self, py: Python<'py>, keys: Vec<String>) -> PyResult<Bound<'py, PyList>> {
        let mut positions_by_path: HashMap<PathBuf, Vec<(usize, usize)>> = HashMap::new();
        for (position, key) in keys.iter().enumerate() {
            if let Some(location) = self.index.get(key) {
                positions_by_path
                    .entry(location.path.clone())
                    .or_default()
                    .push((position, location.row_index));
            }
        }
        let output = py
            .detach(|| -> Result<Vec<Option<FeatureMatrixPayload>>> {
                let mut output: Vec<Option<FeatureMatrixPayload>> = vec![None; keys.len()];
                for (path, positions) in positions_by_path {
                    let requested: HashMap<usize, Vec<usize>> = positions.into_iter().fold(
                        HashMap::new(),
                        |mut acc, (position, row_index)| {
                            acc.entry(row_index).or_default().push(position);
                            acc
                        },
                    );
                    let payloads = read_payloads_from_part(&path, requested.keys().copied())?;
                    for (row_index, positions) in requested {
                        let Some(payload) = payloads.get(&row_index) else {
                            continue;
                        };
                        let Ok(decoded) = decode_feature_payload(payload) else {
                            continue;
                        };
                        for position in positions {
                            output[position] = Some(decoded.clone());
                        }
                    }
                }
                Ok(output)
            })
            .map_err(py_error)?;

        let py_items = PyList::empty(py);
        for item in output {
            match item {
                Some(features) => {
                    let array =
                        Array2::from_shape_vec((features.rows, features.cols), features.values)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?;
                    py_items.append(array.into_pyarray(py))?;
                }
                None => py_items.append(py.None())?,
            }
        }
        Ok(py_items)
    }
}

#[derive(Clone)]
struct FeatureMatrixPayload {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

fn py_error(error: anyhow::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(crate) fn build_feature_index(
    cache_root: &Path,
    num_shards: usize,
) -> Result<HashMap<String, FeatureLocation>> {
    let part_metadata = collect_part_metadata(cache_root, num_shards)?;
    if let Some(index) = load_persisted_index(cache_root, &part_metadata)? {
        return Ok(index);
    }
    let index = build_feature_index_from_parts(&part_metadata)?;
    write_persisted_index(cache_root, &part_metadata, &index)?;
    Ok(index)
}

fn build_feature_index_from_parts(
    part_metadata: &[PartMetadata],
) -> Result<HashMap<String, FeatureLocation>> {
    let mut updates = part_metadata
        .par_iter()
        .enumerate()
        .map(|(order, metadata)| read_part_updates(order, &metadata.path))
        .collect::<Result<Vec<_>>>()?;
    updates.sort_by_key(|update| update.order);
    let mut index = HashMap::new();
    for update in updates {
        for row in update.rows {
            if row.deleted {
                index.remove(&row.key);
            } else {
                index.insert(
                    row.key,
                    FeatureLocation {
                        path: update.path.clone(),
                        row_index: row.row_index,
                    },
                );
            }
        }
    }
    Ok(index)
}

fn collect_part_metadata(cache_root: &Path, num_shards: usize) -> Result<Vec<PartMetadata>> {
    let shard_root = cache_root.join("feature_shards");
    let mut parts = Vec::new();
    for shard_index in 0..num_shards {
        let shard_path = shard_root.join(format!("features_{shard_index:02}"));
        if !shard_path.is_dir() {
            continue;
        }
        let mut part_paths = fs::read_dir(&shard_path)
            .with_context(|| format!("failed to read {}", shard_path.display()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("part_") && name.ends_with(".parquet"))
            })
            .collect::<Vec<_>>();
        part_paths.sort_by_key(|path| {
            path.metadata()
                .and_then(|metadata| metadata.modified())
                .ok()
        });
        for part_path in part_paths {
            let metadata = part_path
                .metadata()
                .with_context(|| format!("failed to stat {}", part_path.display()))?;
            let modified = metadata
                .modified()
                .with_context(|| format!("failed to read mtime for {}", part_path.display()))?;
            let mtime_ns = modified
                .duration_since(UNIX_EPOCH)
                .with_context(|| format!("mtime before unix epoch for {}", part_path.display()))?
                .as_nanos()
                .try_into()
                .context("part mtime nanoseconds does not fit into i64")?;
            parts.push(PartMetadata {
                path: part_path,
                mtime_ns,
                size: metadata.len(),
            });
        }
    }
    Ok(parts)
}

fn read_part_updates(order: usize, path: &Path) -> Result<PartUpdate> {
    let input = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(input)?
        .with_batch_size(8192)
        .build()?;
    let mut global_row_index = 0usize;
    let mut rows = Vec::new();
    for batch_result in reader {
        let batch = batch_result?;
        let key_column = column_by_name(&batch, &["key"])
            .ok_or_else(|| anyhow!("{} has no key column", path.display()))?;
        let deleted_column = column_by_name(&batch, &["deleted"])
            .ok_or_else(|| anyhow!("{} has no deleted column", path.display()))?;
        for row_index in 0..batch.num_rows() {
            let Some(key) = scalar_as_string(key_column.as_ref(), row_index) else {
                global_row_index += 1;
                continue;
            };
            let deleted = scalar_as_bool(deleted_column.as_ref(), row_index).unwrap_or(false);
            rows.push(PartRowUpdate {
                key,
                row_index: global_row_index,
                deleted,
            });
            global_row_index += 1;
        }
    }
    Ok(PartUpdate {
        order,
        path: path.to_path_buf(),
        rows,
    })
}

fn read_payloads_from_part(
    path: &Path,
    requested_rows: impl Iterator<Item = usize>,
) -> Result<HashMap<usize, Vec<u8>>> {
    let requested = requested_rows.collect::<HashSet<_>>();
    let input = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(input)?
        .with_batch_size(8192)
        .build()?;
    let mut payloads = HashMap::new();
    let mut global_row_index = 0usize;
    for batch_result in reader {
        let batch = batch_result?;
        let payload_column = column_by_name(&batch, &["payload"])
            .ok_or_else(|| anyhow!("{} has no payload column", path.display()))?;
        for row_index in 0..batch.num_rows() {
            if !requested.contains(&global_row_index) || payloads.contains_key(&global_row_index) {
                global_row_index += 1;
                continue;
            }
            if let Some(payload) = scalar_as_bytes(payload_column.as_ref(), row_index) {
                payloads.insert(global_row_index, payload);
            }
            global_row_index += 1;
        }
        if payloads.len() == requested.len() {
            break;
        }
    }
    Ok(payloads)
}

pub(crate) fn persisted_index_path(cache_root: &Path) -> PathBuf {
    cache_root.join("rust_feature_index.parquet")
}

fn load_persisted_index(
    cache_root: &Path,
    part_metadata: &[PartMetadata],
) -> Result<Option<HashMap<String, FeatureLocation>>> {
    let index_path = persisted_index_path(cache_root);
    if !index_path.is_file() {
        return Ok(None);
    }
    let input = File::open(&index_path)
        .with_context(|| format!("failed to open {}", index_path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(input)?
        .with_batch_size(8192)
        .build()?;
    let mut source_parts = Vec::new();
    let mut index = HashMap::new();
    for batch_result in reader {
        let batch = batch_result?;
        let kind_column = column_by_name(&batch, &["kind"])
            .ok_or_else(|| anyhow!("{} has no kind column", index_path.display()))?;
        let path_column = column_by_name(&batch, &["path"])
            .ok_or_else(|| anyhow!("{} has no path column", index_path.display()))?;
        let mtime_column = column_by_name(&batch, &["mtime_ns"])
            .ok_or_else(|| anyhow!("{} has no mtime_ns column", index_path.display()))?;
        let size_column = column_by_name(&batch, &["size"])
            .ok_or_else(|| anyhow!("{} has no size column", index_path.display()))?;
        let key_column = column_by_name(&batch, &["key"])
            .ok_or_else(|| anyhow!("{} has no key column", index_path.display()))?;
        let row_index_column = column_by_name(&batch, &["row_index"])
            .ok_or_else(|| anyhow!("{} has no row_index column", index_path.display()))?;
        for row_index in 0..batch.num_rows() {
            let kind = scalar_as_u8(kind_column.as_ref(), row_index).unwrap_or(0);
            let path = scalar_as_string(path_column.as_ref(), row_index)
                .map(PathBuf::from)
                .unwrap_or_default();
            if kind == 0 {
                source_parts.push(PartMetadata {
                    path,
                    mtime_ns: scalar_as_i64(mtime_column.as_ref(), row_index).unwrap_or(0),
                    size: scalar_as_u64(size_column.as_ref(), row_index).unwrap_or(0),
                });
                continue;
            }
            let Some(key) = scalar_as_string(key_column.as_ref(), row_index) else {
                continue;
            };
            let Some(cached_row_index) = scalar_as_u64(row_index_column.as_ref(), row_index) else {
                continue;
            };
            index.insert(
                key,
                FeatureLocation {
                    path,
                    row_index: cached_row_index
                        .try_into()
                        .context("cached row_index does not fit into usize")?,
                },
            );
        }
    }
    if source_parts == part_metadata {
        Ok(Some(index))
    } else {
        Ok(None)
    }
}

fn write_persisted_index(
    cache_root: &Path,
    part_metadata: &[PartMetadata],
    index: &HashMap<String, FeatureLocation>,
) -> Result<()> {
    let index_path = persisted_index_path(cache_root);
    let tmp_path = index_path.with_extension("parquet.tmp");
    let mut kind_builder = UInt8Builder::new();
    let mut path_builder = StringBuilder::new();
    let mut mtime_builder = Int64Builder::new();
    let mut size_builder = UInt64Builder::new();
    let mut key_builder = StringBuilder::new();
    let mut row_index_builder = UInt64Builder::new();

    for part in part_metadata {
        kind_builder.append_value(0);
        path_builder.append_value(part.path.to_string_lossy());
        mtime_builder.append_value(part.mtime_ns);
        size_builder.append_value(part.size);
        key_builder.append_null();
        row_index_builder.append_null();
    }
    let mut entries = index.iter().collect::<Vec<_>>();
    entries.sort_by(|left, right| left.0.cmp(right.0));
    for (key, location) in entries {
        kind_builder.append_value(1);
        path_builder.append_value(location.path.to_string_lossy());
        mtime_builder.append_null();
        size_builder.append_null();
        key_builder.append_value(key);
        row_index_builder.append_value(
            location
                .row_index
                .try_into()
                .context("row_index does not fit into u64")?,
        );
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("kind", DataType::UInt8, false),
        Field::new("path", DataType::Utf8, false),
        Field::new("mtime_ns", DataType::Int64, true),
        Field::new("size", DataType::UInt64, true),
        Field::new("key", DataType::Utf8, true),
        Field::new("row_index", DataType::UInt64, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(kind_builder.finish()) as ArrayRef,
            Arc::new(path_builder.finish()) as ArrayRef,
            Arc::new(mtime_builder.finish()) as ArrayRef,
            Arc::new(size_builder.finish()) as ArrayRef,
            Arc::new(key_builder.finish()) as ArrayRef,
            Arc::new(row_index_builder.finish()) as ArrayRef,
        ],
    )?;
    let file = File::create(&tmp_path)
        .with_context(|| format!("failed to create {}", tmp_path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    fs::rename(&tmp_path, &index_path).with_context(|| {
        format!(
            "failed to replace {} with {}",
            index_path.display(),
            tmp_path.display()
        )
    })?;
    Ok(())
}

fn scalar_as_bool(array: &dyn Array, row_index: usize) -> Option<bool> {
    if array.is_null(row_index) {
        return None;
    }
    array
        .as_any()
        .downcast_ref::<arrow::array::BooleanArray>()
        .map(|values| values.value(row_index))
}

fn scalar_as_i64(array: &dyn Array, row_index: usize) -> Option<i64> {
    if array.is_null(row_index) {
        return None;
    }
    array
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .map(|values| values.value(row_index))
}

fn scalar_as_u64(array: &dyn Array, row_index: usize) -> Option<u64> {
    if array.is_null(row_index) {
        return None;
    }
    array
        .as_any()
        .downcast_ref::<arrow::array::UInt64Array>()
        .map(|values| values.value(row_index))
}

fn scalar_as_u8(array: &dyn Array, row_index: usize) -> Option<u8> {
    if array.is_null(row_index) {
        return None;
    }
    array
        .as_any()
        .downcast_ref::<arrow::array::UInt8Array>()
        .map(|values| values.value(row_index))
}

fn decode_feature_payload(payload: &[u8]) -> Result<FeatureMatrixPayload> {
    if !payload.starts_with(RUST_PAYLOAD_MAGIC) {
        bail!("feature payload is not in Rust SFCF32L1 format");
    }
    if payload.len() < RUST_PAYLOAD_MAGIC.len() + 8 {
        bail!("feature payload is shorter than its header");
    }
    let rows = u32::from_le_bytes(
        payload[RUST_PAYLOAD_MAGIC.len()..RUST_PAYLOAD_MAGIC.len() + 4]
            .try_into()
            .expect("slice length checked"),
    ) as usize;
    let cols = u32::from_le_bytes(
        payload[RUST_PAYLOAD_MAGIC.len() + 4..RUST_PAYLOAD_MAGIC.len() + 8]
            .try_into()
            .expect("slice length checked"),
    ) as usize;
    let data = &payload[RUST_PAYLOAD_MAGIC.len() + 8..];
    let expected_len = rows
        .checked_mul(cols)
        .and_then(|values| values.checked_mul(4))
        .ok_or_else(|| anyhow!("feature payload dimensions overflow"))?;
    if data.len() != expected_len {
        bail!(
            "feature payload has invalid size: got {} bytes, expected {}",
            data.len(),
            expected_len
        );
    }
    let values = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk size is 4")))
        .collect();
    Ok(FeatureMatrixPayload { rows, cols, values })
}
