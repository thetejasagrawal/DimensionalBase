"""
Input validation and sanitization for DimensionalBase.

Rejects unsafe paths, invalid owners, oversized payloads, and malformed
metadata. The core library and security wrapper both rely on these helpers.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from dimensionalbase.exceptions import DimensionalBaseError


class ValidationError(DimensionalBaseError):
    """Bad input — invalid path, value, or metadata."""


# Allowed characters in paths
_PATH_PATTERN = re.compile(r"^[a-zA-Z0-9/_.\-]+$")
_OWNER_PATTERN = re.compile(r"^[a-zA-Z0-9_.\-]+$")
_MAX_PATH_LENGTH = 512
_MAX_OWNER_LENGTH = 128
_MAX_VALUE_LENGTH = 1_048_576  # 1 MB
_MAX_METADATA_KEY_LENGTH = 64
_MAX_METADATA_VALUE_LENGTH = 4096


def validate_path(path: str) -> str:
    """Validate and sanitize a path. Returns the path if valid, raises ValidationError otherwise."""
    if not path:
        raise ValidationError("Path cannot be empty")
    if len(path) > _MAX_PATH_LENGTH:
        raise ValidationError(f"Path too long ({len(path)} > {_MAX_PATH_LENGTH})")
    if "\x00" in path:
        raise ValidationError("Path contains null bytes")
    if ".." in path:
        raise ValidationError("Path contains directory traversal (..) ")
    if path.startswith("/"):
        raise ValidationError("Path cannot start with '/'")
    if path.endswith("/"):
        raise ValidationError("Path cannot end with '/'")
    if "//" in path:
        raise ValidationError("Path cannot contain empty segments ('//')")
    if not _PATH_PATTERN.match(path):
        raise ValidationError(
            f"Path contains invalid characters. Allowed: [a-zA-Z0-9/_.-]"
        )
    return path


def validate_value(value: str) -> str:
    """Validate entry value. Returns value if valid."""
    if not value:
        raise ValidationError("Value cannot be empty")
    if len(value) > _MAX_VALUE_LENGTH:
        raise ValidationError(f"Value too long ({len(value)} > {_MAX_VALUE_LENGTH})")
    if "\x00" in value:
        raise ValidationError("Value contains null bytes")
    return value


def validate_owner(owner: str) -> str:
    """Validate the writing agent identifier."""
    if not owner:
        raise ValidationError("Owner cannot be empty")
    if len(owner) > _MAX_OWNER_LENGTH:
        raise ValidationError(f"Owner too long ({len(owner)} > {_MAX_OWNER_LENGTH})")
    if "\x00" in owner:
        raise ValidationError("Owner contains null bytes")
    if not _OWNER_PATTERN.match(owner):
        raise ValidationError(
            "Owner contains invalid characters. Allowed: [a-zA-Z0-9_.-]"
        )
    return owner


def validate_confidence(confidence: float) -> float:
    """Validate confidence bounds."""
    if not (0.0 <= confidence <= 1.0):
        raise ValidationError(f"Confidence must be 0.0-1.0, got {confidence}")
    return confidence


def validate_metadata(metadata: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Validate metadata keys and values."""
    if metadata is None:
        return {}
    for key, val in metadata.items():
        if len(key) > _MAX_METADATA_KEY_LENGTH:
            raise ValidationError(f"Metadata key too long: {key[:20]}...")
        if len(str(val)) > _MAX_METADATA_VALUE_LENGTH:
            raise ValidationError(f"Metadata value too long for key: {key}")
        if key.startswith("_"):
            raise ValidationError(f"Metadata keys starting with '_' are reserved: {key}")
    return metadata
