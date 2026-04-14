"""
Encryption at rest for DimensionalBase.

Uses Fernet (AES-128-CBC + HMAC-SHA256) for symmetric encryption of
entry values before SQLite storage.
"""

from __future__ import annotations

import base64
import hashlib
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

    def __init__(self, key: Optional[str] = None, passphrase: Optional[str] = None) -> None:
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.primitives import hashes
        except ImportError:
            raise ImportError(
                "The 'cryptography' package is required for encryption. "
                "Install it with: pip install dimensionalbase[security]"
            )

        if key:
            self._fernet = Fernet(key.encode() if isinstance(key, str) else key)
        elif passphrase:
            # Derive a key from the passphrase
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"dimensionalbase_salt_v1",
                iterations=100_000,
            )
            derived = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
            self._fernet = Fernet(derived)
        else:
            # Generate a random key
            self._fernet = Fernet(Fernet.generate_key())

    def encrypt(self, data: str) -> str:
        try:
            return self._fernet.encrypt(data.encode()).decode()
        except Exception as exc:
            raise EncryptionError(f"Encryption failed: {exc}") from exc

    def decrypt(self, data: str) -> str:
        try:
            return self._fernet.decrypt(data.encode()).decode()
        except Exception as exc:
            raise EncryptionError(f"Decryption failed: {exc}") from exc
