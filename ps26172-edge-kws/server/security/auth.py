"""
HMAC-SHA256 session token authentication.

Token format:
  base64( session_id (16 bytes) | timestamp_unix (8 bytes) | hmac_tag (8 bytes) )
  where hmac_tag = HMAC-SHA256(session_id + timestamp, secret)[0:8]
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from threading import Lock


# ---------------------------------------------------------------------------
# Used token replay cache (thread-safe)
# ---------------------------------------------------------------------------

_used_tokens: dict[str, float] = {}  # token_hex -> expiry_time
_cache_lock = Lock()


class TokenError(Exception):
    """Raised when token verification fails."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code  # Maps to E10x error codes
        self.message = message


# ---------------------------------------------------------------------------
# Token verification
# ---------------------------------------------------------------------------


def verify_token(
    token_b64: str,
    secret: bytes,
    window_s: int = 30,
) -> str:
    """Verify an HMAC-SHA256 session token.

    Args:
        token_b64: Base64-encoded token string from X-KWS-Token header.
        secret: Shared secret bytes (must match firmware config).
        window_s: Allowed timestamp skew in seconds (replay window).

    Returns:
        Session ID (hex string) extracted from the token.

    Raises:
        TokenError: If the token is missing, expired, replayed, or has an
            invalid HMAC signature.
    """
    # Decode base64
    try:
        raw = base64.b64decode(token_b64 + "==")  # Pad for safety
    except Exception:
        raise TokenError(2, "Token is not valid base64.")

    if len(raw) < 32:
        raise TokenError(2, f"Token too short: {len(raw)} bytes, expected ≥ 32.")

    session_id_bytes = raw[0:16]
    ts_bytes = raw[16:24]
    tag = raw[24:32]

    # Check timestamp
    try:
        ts = int.from_bytes(ts_bytes, "big")
    except Exception:
        raise TokenError(2, "Invalid timestamp in token.")

    now = time.time()
    if abs(now - ts) > window_s:
        raise TokenError(3, f"Token expired: age={int(abs(now - ts))}s, window={window_s}s.")

    # Verify HMAC
    expected_tag = hmac.new(
        secret,
        session_id_bytes + ts_bytes,
        digestmod=hashlib.sha256,
    ).digest()[0:8]

    if not hmac.compare_digest(tag, expected_tag):
        raise TokenError(2, "HMAC signature verification failed.")

    # Replay detection
    token_key = raw.hex()
    with _cache_lock:
        _evict_expired_tokens()
        if token_key in _used_tokens:
            raise TokenError(4, "Token has already been used (replay detected).")
        _used_tokens[token_key] = now + window_s * 2  # Cache for 2× window

    session_id = session_id_bytes.hex()
    return session_id


# ---------------------------------------------------------------------------
# Token generation (for testing / examples)
# ---------------------------------------------------------------------------


def generate_token(secret: bytes, window_s: int = 30) -> str:
    """Generate a valid HMAC-SHA256 session token.

    Used in tests, examples, and the Python ESP32 simulator.

    Args:
        secret: Shared secret bytes.
        window_s: Not used in generation; included for API symmetry.

    Returns:
        Base64-encoded token string.
    """
    session_id = os.urandom(16)
    ts = int(time.time()).to_bytes(8, "big")
    tag = hmac.new(secret, session_id + ts, digestmod=hashlib.sha256).digest()[0:8]
    raw = session_id + ts + tag
    return base64.b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _evict_expired_tokens() -> None:
    """Remove expired tokens from the replay cache (call with lock held)."""
    now = time.time()
    expired = [k for k, exp in _used_tokens.items() if exp < now]
    for k in expired:
        del _used_tokens[k]
