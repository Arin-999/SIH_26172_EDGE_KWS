# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| `main` branch | Yes |
| `develop` branch | Best-effort |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email the maintainer directly at the contact listed in the project.  
Include:

1. Affected component (`server/security/auth.py`, firmware, etc.)
2. Reproduction steps
3. Impact assessment (authentication bypass, data exposure, etc.)
4. Suggested fix if available

We aim to acknowledge reports within 48 hours and provide a fix within 14 days.

---

## Known Security Properties

### Authentication

- All WebSocket connections require an `X-KWS-Token` header.
- Tokens use **HMAC-SHA256** with a shared secret (32-byte minimum recommended).
- Tokens expire after 30 seconds (configurable via `KWS_AUTH_WINDOW_S`).
- Each token is usable **exactly once** (replay cache, in-memory per process).
- Token comparison uses `hmac.compare_digest()` — constant-time, no timing oracle.

### Transport

- Audio is transmitted over **WebSocket** (TCP). For production use, deploy behind **TLS** (e.g., `wss://` via nginx/Caddy reverse proxy).
- Each audio chunk includes a **4-byte HMAC-SHA256 tag** for integrity verification.

### Known Limitations

| Area | Limitation | Mitigation |
|---|---|---|
| Replay cache | In-memory only — cleared on server restart | Use Redis with TTL for multi-process or persistent replay protection |
| HMAC chunk tag | 4 bytes (32-bit) — adequate for LAN, not cryptographically strong | Increase to 8 bytes for untrusted networks |
| Shared secret | Hardcoded default `dev-secret-do-not-use-in-production` | Always set `KWS_SECRET` from a secrets manager in production |
| No TLS in default config | Plain WebSocket on port 8765 | Deploy behind nginx with SSL termination |
| Max sessions | Hard cap of 10 (configurable) — no per-IP rate limiting beyond this | Add IP-based rate limiting via nginx or FastAPI middleware |

---

## Production Deployment Security Checklist

- [ ] `KWS_SECRET` set to a cryptographically random 32-byte value
- [ ] Server running behind TLS (`wss://`)
- [ ] Firewall restricts port 8765 to authorised clients only
- [ ] `KWS_MAX_SESSIONS` set appropriately for expected load
- [ ] Log output reviewed and sensitive values not logged
- [ ] Firmware flashed with device-specific secret (not the default)
