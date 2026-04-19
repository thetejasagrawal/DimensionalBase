"""
Encryption at rest for DimensionalBase.

Uses Fernet (AES-128-CBC + HMAC-SHA256) for symmetric encryption of
entry values before SQLite storage.
"""

from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod
from typing import Optional

from dimensionalbase.exceptions import DimensionalBaseError


class EncryptionError(DimensionalBaseError):
    """Encryption or decryption failure."""


class EncryptionProvider(ABC):
    """Abstract base for encryption providers."""

    @abstractmethod
    def encrypt(self, data: str) -> str:
        """Encrypt a string. Returns an encrypted string."""

    @abstractmethod
    def decrypt(self, data: str) -> str:
        """Decrypt an encrypted string. Returns the original string."""


class NullEncryptionProvider(EncryptionProvider):
    """No encryption — passthrough for unencrypted mode."""

    def encrypt(self, data: str) -> str:
        return data

    def decrypt(self, data: str) -> str:
        return data


class FernetEncryptionProvider(EncryptionProvider):
    """Fernet-based encryption (AES-128-CBC + HMAC-SHA256).

    Requires the ``cryptography`` package.
    """

    _FORMAT_PREFIX = "dmb$1$"
    _PBKDF2_ITERATIONS = 600_000
    _LEGACY_PBKDF2_ITERATIONS = 100_000
    _LEGACY_SALT = b"dimensionalbase_salt_v1"
    _SALT_BYTES = 16

    def __init__(self, key: Optional[str] = None, passphrase: Optional[str] = None) -> None:
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        except ImportError:
            raise ImportError(
                "The 'cryptography' package is required for encryption. "
                "Install it with: pip install dimensionalbase[security]"
            )

        self._fernet_cls = Fernet
        self._hashes = hashes
        self._pbkdf2_cls = PBKDF2HMAC
        self._passphrase = passphrase.encode() if passphrase is not None else None

        if key:
            self._fernet = Fernet(key.encode() if isinstance(key, str) else key)
        elif self._passphrase:
            self._fernet = None
        else:
            raise EncryptionError(
                "Encryption requires an explicit key or passphrase. "
                "Refusing to generate an ephemeral key that would make stored data unreadable."
            )

    def _derive_fernet(self, salt: bytes, *, iterations: int = _PBKDF2_ITERATIONS):
        kdf = self._pbkdf2_cls(
            algorithm=self._hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
        )
        derived = base64.urlsafe_b64encode(kdf.derive(self._passphrase))
        return self._fernet_cls(derived)

    def encrypt(self, data: str) -> str:
        try:
            if self._fernet is not None:
                return self._fernet.encrypt(data.encode()).decode()

            salt = os.urandom(self._SALT_BYTES)
            token = self._derive_fernet(salt).encrypt(data.encode()).decode()
            encoded_salt = base64.urlsafe_b64encode(salt).decode()
            return f"{self._FORMAT_PREFIX}{encoded_salt}${token}"
        except Exception as exc:
            raise EncryptionError(f"Encryption failed: {exc}") from exc

    def decrypt(self, data: str) -> str:
        try:
            if self._fernet is not None:
                return self._fernet.decrypt(data.encode()).decode()

            if data.startswith(self._FORMAT_PREFIX):
                encoded_salt, token = data[len(self._FORMAT_PREFIX):].split("$", 1)
                salt = base64.urlsafe_b64decode(encoded_salt.encode())
                return self._derive_fernet(salt).decrypt(token.encode()).decode()

            legacy = self._derive_fernet(
                self._LEGACY_SALT,
                iterations=self._LEGACY_PBKDF2_ITERATIONS,
            )
            return legacy.decrypt(data.encode()).decode()
        except Exception as exc:
            raise EncryptionError(f"Decryption failed: {exc}") from exc
