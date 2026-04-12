use arrow::array::{
    Array, ArrayRef, BinaryArray, Float64Array, Int32Array, Int64Array, LargeBinaryArray,
    LargeStringArray, StringArray, StructArray, UInt32Array, UInt64Array,
};
use arrow::record_batch::RecordBatch;

pub(crate) fn column_by_name(batch: &RecordBatch, names: &[&str]) -> Option<ArrayRef> {
    for name in names {
        if let Ok(index) = batch.schema().index_of(name) {
            return Some(batch.column(index).clone());
        }
    }
    None
}

pub(crate) fn struct_child(struct_array: &StructArray, name: &str) -> Option<ArrayRef> {
    struct_array
        .column_names()
        .iter()
        .position(|candidate| *candidate == name)
        .map(|index| struct_array.column(index).clone())
}

pub(crate) fn scalar_as_string(array: &dyn Array, row_index: usize) -> Option<String> {
    if array.is_null(row_index) {
        return None;
    }
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return Some(values.value(row_index).to_string());
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Some(values.value(row_index).to_string());
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        return Some(values.value(row_index).to_string());
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return Some(values.value(row_index).to_string());
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt32Array>() {
        return Some(values.value(row_index).to_string());
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt64Array>() {
        return Some(values.value(row_index).to_string());
    }
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        return Some(values.value(row_index).to_string());
    }
    None
}

pub(crate) fn scalar_as_f64(array: &dyn Array, row_index: usize) -> Option<f64> {
    if array.is_null(row_index) {
        return None;
    }
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        return Some(values.value(row_index));
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        return Some(values.value(row_index) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return Some(values.value(row_index) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt32Array>() {
        return Some(values.value(row_index) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt64Array>() {
        return Some(values.value(row_index) as f64);
    }
    if let Some(value) = scalar_as_string(array, row_index) {
        return value.parse::<f64>().ok();
    }
    None
}

pub(crate) fn scalar_as_bytes(array: &dyn Array, row_index: usize) -> Option<Vec<u8>> {
    if array.is_null(row_index) {
        return None;
    }
    if let Some(values) = array.as_any().downcast_ref::<BinaryArray>() {
        return Some(values.value(row_index).to_vec());
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeBinaryArray>() {
        return Some(values.value(row_index).to_vec());
    }
    None
}
