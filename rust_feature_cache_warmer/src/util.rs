use std::path::{Path, PathBuf};

pub(crate) fn resolve_path(source_base: &Path, path: &str) -> PathBuf {
    let path_buf = PathBuf::from(path);
    if path_buf.is_absolute() {
        path_buf
    } else {
        source_base.join(path_buf)
    }
}

pub(crate) fn py_bool(value: bool) -> &'static str {
    if value {
        "True"
    } else {
        "False"
    }
}

pub(crate) fn py_float(value: f32) -> String {
    if value.is_finite() {
        let mut rendered = format!("{:?}", value);
        if !rendered.contains('.') && !rendered.contains('e') && !rendered.contains('E') {
            rendered.push_str(".0");
        }
        rendered
    } else if value.is_nan() {
        "nan".to_string()
    } else if value.is_sign_positive() {
        "inf".to_string()
    } else {
        "-inf".to_string()
    }
}

pub(crate) fn hex_prefix(bytes: &[u8], chars: usize) -> String {
    hex_full(bytes).chars().take(chars).collect()
}

pub(crate) fn hex_full(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push_str(&format!("{byte:02x}"));
    }
    output
}
