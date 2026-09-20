"""
Encryption at rest  (Data Security)

AES-256-GCM (authenticated encryption): a tampered or swapped file fails to decrypt.
  * one random 96-bit nonce per file (never reused)
  * the file's pseudonymous ID is bound as associated data, so an encrypted blob cannot
    be silently moved under a different ID
  * the key comes from the environment (ASL_DATA_KEY, base64) - never from the repo.
    In production fetch it from a secrets manager / KMS instead of a shell variable.

File format:  b"ASL1" | 12-byte nonce | ciphertext+tag
"""
from __future__ import annotations
import base64
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"ASL1"
KEY_ENV = "ASL_DATA_KEY"


def generate_key() -> str:
    """Return a fresh 256-bit key, base64-encoded. Store it in a secrets manager."""
    return base64.b64encode(AESGCM.generate_key(bit_length=256)).decode()


def _key(key_b64: str | None = None) -> bytes:
    raw = key_b64 or os.environ.get(KEY_ENV)
    if not raw:
        raise RuntimeError(f"Set {KEY_ENV} (see `python scripts/run_pipeline.py keygen`).")
    key = base64.b64decode(raw)
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes (AES-256).")
    return key


def encrypt_bytes(data: bytes, aad: str, key_b64: str | None = None) -> bytes:
    nonce = os.urandom(12)
    return MAGIC + nonce + AESGCM(_key(key_b64)).encrypt(nonce, data, aad.encode())


def decrypt_bytes(blob: bytes, aad: str, key_b64: str | None = None) -> bytes:
    if blob[:4] != MAGIC:
        raise ValueError("Not an ASL encrypted blob")
    nonce, ct = blob[4:16], blob[16:]
    return AESGCM(_key(key_b64)).decrypt(nonce, ct, aad.encode())   # raises InvalidTag if tampered


class EncryptedImageStore:
    """Directory of <image_id>.enc files. Reads decrypt into memory only (no plaintext on disk)."""

    def __init__(self, root: str | Path, key_b64: str | None = None):
        self.root, self.key = Path(root), key_b64
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, image_id: str, data: bytes) -> Path:
        p = self.root / f"{image_id}.enc"
        p.write_bytes(encrypt_bytes(data, image_id, self.key))
        os.chmod(p, 0o600)
        return p

    def get(self, image_id: str) -> bytes:
        return decrypt_bytes((self.root / f"{image_id}.enc").read_bytes(), image_id, self.key)
