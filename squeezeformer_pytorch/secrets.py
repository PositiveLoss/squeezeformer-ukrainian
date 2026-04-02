from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

HF_TOKEN_PATTERN = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")
REDACTED = "[REDACTED]"
_SECRET_KEY_NAMES = {
    "api_key",
    "auth_token",
    "authorization",
    "hf_token",
    "huggingface_token",
    "password",
    "secret",
    "secret_key",
    "token",
}
_SECRET_KEY_SUBSTRINGS = (
    "api_key",
    "auth_token",
    "authorization",
    "hf_token",
    "huggingface_token",
    "password",
    "secret",
)


def redact_text_secrets(text: str, replacement: str = REDACTED) -> str:
    return HF_TOKEN_PATTERN.sub(replacement, text)


def sanitize_for_serialization(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            subkey: sanitize_for_serialization(subvalue, key=str(subkey))
            for subkey, subvalue in value.items()
        }
    if isinstance(value, list):
        return [sanitize_for_serialization(item, key=key) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_serialization(item, key=key) for item in value]
    if isinstance(value, Path):
        return redact_text_secrets(str(value))
    if isinstance(value, str):
        if _is_secret_key(key):
            return REDACTED
        return redact_text_secrets(value)
    return value


def sanitize_json_text(text: str) -> str:
    payload = json.loads(text)
    sanitized = sanitize_for_serialization(payload)
    return json.dumps(sanitized, indent=2, ensure_ascii=False)


def _is_secret_key(key: str | None) -> bool:
    if key is None:
        return False
    normalized = key.strip().lower()
    if normalized in _SECRET_KEY_NAMES:
        return True
    return any(part in normalized for part in _SECRET_KEY_SUBSTRINGS)
