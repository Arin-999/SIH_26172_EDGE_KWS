"""
Security unit tests for HMAC-SHA256 session token auth.
"""

import time

import pytest

from server.security.auth import (
    verify_token,
    generate_token,
    TokenError,
)

SECRET = b"test-secret-key-32-bytes-padding-x"


# ---------------------------------------------------------------------------
# Tests: generate_token + verify_token round-trip
# ---------------------------------------------------------------------------


class TestTokenRoundTrip:
    def test_valid_token_passes(self) -> None:
        token = generate_token(SECRET)
        session_id = verify_token(token, SECRET)
        assert isinstance(session_id, str)
        assert len(session_id) == 32  # 16 bytes → 32 hex chars

    def test_different_tokens_per_call(self) -> None:
        t1 = generate_token(SECRET)
        t2 = generate_token(SECRET)
        assert t1 != t2  # Random session_id each time


# ---------------------------------------------------------------------------
# Tests: TokenError cases
# ---------------------------------------------------------------------------


class TestTokenErrors:
    def test_wrong_secret_fails(self) -> None:
        token = generate_token(SECRET)
        with pytest.raises(TokenError) as exc:
            verify_token(token, b"wrong-secret")
        assert exc.value.code == 2  # AUTH_INVALID

    def test_invalid_base64_fails(self) -> None:
        with pytest.raises(TokenError):
            verify_token("not-valid-base64!!!", SECRET)

    def test_too_short_token_fails(self) -> None:
        import base64
        short = base64.b64encode(b"\x00" * 10).decode()
        with pytest.raises(TokenError) as exc:
            verify_token(short, SECRET)
        assert exc.value.code == 2

    def test_replay_detection(self) -> None:
        token = generate_token(SECRET)
        verify_token(token, SECRET)  # First use: OK
        with pytest.raises(TokenError) as exc:
            verify_token(token, SECRET)  # Second use: replay
        assert exc.value.code == 4  # AUTH_REPLAY

    def test_expired_token_fails(self, monkeypatch) -> None:
        """Token with timestamp 60 seconds in the past should fail."""
        import server.security.auth as auth_module

        # Generate a token, then shift time by 60 seconds
        token = generate_token(SECRET)

        original_time = time.time

        def future_time():
            return original_time() + 60

        monkeypatch.setattr(auth_module, "time", type("MockTime", (), {"time": staticmethod(future_time)})())

        with pytest.raises(TokenError) as exc:
            # Use window_s=30, so 60s offset exceeds window
            verify_token(token, SECRET, window_s=30)
        assert exc.value.code == 3  # AUTH_EXPIRED


# ---------------------------------------------------------------------------
# Tests: token format
# ---------------------------------------------------------------------------


class TestTokenFormat:
    def test_token_is_base64_string(self) -> None:
        import base64
        token = generate_token(SECRET)
        decoded = base64.b64decode(token + "==")
        assert len(decoded) >= 32

    def test_session_id_is_hex(self) -> None:
        token = generate_token(SECRET)
        session_id = verify_token(token, SECRET)
        assert all(c in "0123456789abcdef" for c in session_id)
