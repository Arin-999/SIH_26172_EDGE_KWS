# Security Design

---

## 1. Threat Model

| Threat | Vector | Mitigation |
|--------|--------|-----------|
| Eavesdropping on audio stream | Network passive intercept | TLS (`wss://`) encrypts all traffic |
| Man-in-the-middle injection | Forged WebSocket messages | TLS certificate pinning (production) |
| Replay attack | Capture and re-send valid session token | HMAC includes timestamp, 30 s replay window |
| Unauthorized session creation | Connect without credentials | HMAC-SHA256 session token required |
| Wake-word profile exfiltration | Read NVS flash over JTAG | NVS partition encryption (AES-256) |
| Firmware tampering | Malicious firmware flash | Secure Boot v2 (ESP32-S3 eFuse) |
| Denial of service | Flood WebSocket connections | Max 10 concurrent sessions, IP rate limiting |

---

## 2. Transport Security

### 2.1 TLS

All production WebSocket connections use `wss://` (WebSocket over TLS 1.2+).

ESP-IDF configuration:

```
CONFIG_ESP_TLS_USING_MBEDTLS=y
CONFIG_MBEDTLS_TLS_CLIENT=y
CONFIG_MBEDTLS_TLS_SERVER=y
CONFIG_WEBSOCKET_URI_FROM_STDIN=n
```

Server certificate is bundled into firmware via `idf.py`'s certificate embedding mechanism (`CONFIG_WEBSOCKET_SERVER_CERT_PEM`). Certificate pinning is performed by comparing the server's DER-encoded certificate hash against the embedded value.

For development/hackathon: self-signed certificate acceptable. For production: use a CA-signed certificate.

---

## 3. Session Authentication

### 3.1 Token Format

Each WebSocket session requires an `X-KWS-Token` HTTP header during the upgrade handshake.

Token format:

```
token = base64( session_id (16 bytes) | timestamp_unix (8 bytes) | hmac_sha256(session_id | timestamp, secret)[0:8] )
```

- `session_id`: 16 random bytes generated at device boot (stored in PSRAM)
- `timestamp_unix`: Unix timestamp (seconds) at token generation
- `hmac_sha256`: Full HMAC computed over `session_id + timestamp`, truncated to first 8 bytes for compactness
- `secret`: 32-byte shared secret, burned into firmware via `menuconfig` → `KWS Configuration` → `KWS Shared Secret`

### 3.2 Server Verification

```python
def verify_token(token_b64: str, secret: bytes) -> bool:
    raw = base64.b64decode(token_b64)
    session_id = raw[0:16]
    ts = int.from_bytes(raw[16:24], 'big')
    tag = raw[24:32]

    # Replay window: 30 seconds
    if abs(time.time() - ts) > 30:
        return False

    # Verify HMAC
    expected = hmac.new(secret, session_id + raw[16:24], 'sha256').digest()[0:8]
    return hmac.compare_digest(tag, expected)
```

Tokens are invalidated after first use per session (stored in a server-side set, evicted after 60 s).

---

## 4. NVS Partition Encryption

Wake-word prototype embeddings (256 bytes per profile) are stored in the `kws_proto` NVS partition.

ESP32-S3 NVS encryption uses an AES-256 XTS key stored in a dedicated `nvs_keys` partition which is protected by eFuse `BLOCK_KEY0`. The `nvs_keys` partition is written once at provisioning time and cannot be read back after eFuse programming.

Enable in `sdkconfig`:

```
CONFIG_NVS_ENCRYPTION=y
CONFIG_NVS_SEC_KEY_PROTECTION_SCHEME_EFUSE_BASED=y
```

---

## 5. Secure Boot

ESP32-S3 supports Secure Boot v2 (RSA-3072 signature over firmware image). When enabled, the ROM bootloader verifies the firmware signature before loading. An unsigned or tampered firmware image is rejected.

Enable in `sdkconfig`:

```
CONFIG_SECURE_BOOT=y
CONFIG_SECURE_BOOT_V2_ENABLED=y
```

> **Note:** Secure Boot permanently burns eFuse bits. Enable only when ready for production. Incorrect use bricks the device.

---

## 6. Rate Limiting

Server-side rate limits:

| Limit | Value |
|-------|-------|
| Max concurrent WebSocket sessions | 10 |
| Max WebSocket connections per IP per minute | 20 |
| Max audio bytes per session per minute | 10 MB |

Implemented via a connection counter per IP in the FastAPI server using `starlette` middleware.

---

## 7. Key Management

| Secret | Location | Rotation |
|--------|----------|---------|
| HMAC shared secret | Firmware `menuconfig` + server env var `KWS_SECRET` | Requires reflash + server restart |
| NVS encryption key | ESP32-S3 eFuse `BLOCK_KEY0` | Non-rotatable (eFuse is OTP) |
| TLS private key | Server filesystem (protect with file permissions) | Annual rotation recommended |
| Secure Boot RSA key | Offline secure storage (HSM or encrypted USB) | Non-rotatable per device |

---

## 8. Security Checklist

- [ ] Use `wss://` in all production deployments
- [ ] Generate unique HMAC secret per device fleet (use device serial as salt)
- [ ] Enable NVS encryption before first provisioning
- [ ] Enable Secure Boot before production flash
- [ ] Store TLS private key with 600 permissions on the server
- [ ] Set `KWS_SECRET` via environment variable, never hardcode in source
- [ ] Run `pytest tests/security/` before each server deployment
