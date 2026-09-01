"""
FastAPI WebSocket server — main entry point.

Endpoints:
  GET  /health         Liveness check
  WS   /v1/stream      Audio streaming endpoint

Usage:
    uvicorn server.receiver.main:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from server.asr.whisper import WhisperASR
from server.packet_reassembly.assembler import SessionAssembler
from server.security.auth import verify_token, TokenError
from server.agent.handler import IntentHandler
from server.fec.decoder import GapDetector

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kws.server")

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

KWS_SECRET: bytes = os.environ.get("KWS_SECRET", "dev-secret-do-not-use-in-production").encode()
ASR_MODEL: str = os.environ.get("KWS_ASR_MODEL", "base.en")
ASR_DEVICE: str = os.environ.get("KWS_ASR_DEVICE", "cpu")
MAX_SESSIONS: int = int(os.environ.get("KWS_MAX_SESSIONS", "10"))
AUTH_WINDOW_S: int = int(os.environ.get("KWS_AUTH_WINDOW_S", "30"))

VERSION = "1.0"

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

asr: WhisperASR | None = None
agent: IntentHandler = IntentHandler()
active_sessions: dict[str, "SessionState"] = {}


class SessionState:
    """Per-WebSocket session state."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.assembler = SessionAssembler(session_id)
        self.gap_detector = GapDetector(session_id)
        self.created_at = time.time()
        self.utterance_count = 0


# ---------------------------------------------------------------------------
# Lifespan: load ASR model on startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    global asr
    logger.info(f"Loading ASR model '{ASR_MODEL}' on {ASR_DEVICE} ...")
    asr = WhisperASR(model_size=ASR_MODEL, device=ASR_DEVICE)
    logger.info("ASR model ready.")
    yield
    logger.info("Server shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="KWS Edge ASR Server",
    version=VERSION,
    description="WebSocket ASR server for SIH PS26172 Edge KWS system.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Status"])
async def health() -> JSONResponse:
    """Liveness check."""
    return JSONResponse({
        "status": "ok",
        "version": VERSION,
        "active_sessions": len(active_sessions),
        "asr_ready": asr is not None,
    })


@app.get("/sessions", tags=["Status"])
async def list_sessions() -> JSONResponse:
    """List active streaming sessions (debug)."""
    return JSONResponse({
        "sessions": [
            {
                "id": sid,
                "age_s": round(time.time() - s.created_at, 1),
                "utterances": s.utterance_count,
            }
            for sid, s in active_sessions.items()
        ]
    })


# ---------------------------------------------------------------------------
# WebSocket streaming endpoint
# ---------------------------------------------------------------------------

# End-of-utterance marker byte
EOT_MARKER: bytes = b"\xff"
# Maximum audio bytes per utterance (~20 seconds @ 16 kHz int16)
MAX_UTTERANCE_BYTES: int = 16_000 * 2 * 20


@app.websocket("/v1/stream")
async def stream_endpoint(ws: WebSocket) -> None:
    """Main WebSocket audio streaming endpoint.

    Authentication: X-KWS-Token header required.
    Binary frames: PCM audio chunks per protocol/packet-format.md
    Text frames: JSON transcript/error responses
    """
    # --- Authenticate ---
    token = ws.headers.get("x-kws-token") or ws.headers.get("X-KWS-Token")
    if not token:
        await ws.close(code=1008, reason="E101: Missing X-KWS-Token header")
        return

    try:
        session_id = verify_token(token, KWS_SECRET, window_s=AUTH_WINDOW_S)
    except TokenError as e:
        await ws.close(code=1008, reason=f"E10{e.code}: {e.message}")
        return

    # --- Session limit ---
    if len(active_sessions) >= MAX_SESSIONS:
        await ws.close(code=1008, reason="E105: Maximum concurrent sessions reached")
        return

    # --- Accept connection ---
    await ws.accept()
    session = SessionState(session_id)
    active_sessions[session_id] = session
    logger.info(f"[{session_id[:8]}] Session opened. Active: {len(active_sessions)}")

    # Send ack
    await ws.send_json({
        "type": "ack",
        "session_id": session_id,
        "server_time": int(time.time()),
    })

    try:
        await _handle_session(ws, session)
    except WebSocketDisconnect:
        logger.info(f"[{session_id[:8]}] Client disconnected.")
    except Exception as exc:
        logger.exception(f"[{session_id[:8]}] Unhandled error: {exc}")
        try:
            await ws.send_json({
                "type": "error",
                "code": "E501",
                "message": "Internal server error",
                "fatal": True,
            })
        except Exception:
            pass
    finally:
        active_sessions.pop(session_id, None)
        logger.info(f"[{session_id[:8]}] Session closed. Active: {len(active_sessions)}")


async def _handle_session(ws: WebSocket, session: SessionState) -> None:
    """Handle the streaming session loop for one connected client."""
    assert asr is not None, "ASR not initialized"

    while True:
        message = await ws.receive()

        # Handle disconnect
        if message["type"] == "websocket.disconnect":
            break

        # Handle text (unexpected from client, but log it)
        if "text" in message:
            logger.debug(f"[{session.session_id[:8]}] Text message ignored: {message['text'][:80]}")
            continue

        data: bytes = message.get("bytes", b"")
        if not data:
            continue

        # --- End-of-utterance marker ---
        if data == EOT_MARKER:
            audio_bytes = session.assembler.flush()
            if not audio_bytes:
                await ws.send_json({
                    "type": "error",
                    "code": "E205",
                    "message": "Audio too short — no audio received before end marker.",
                    "fatal": False,
                })
                continue

            # Validate length
            if len(audio_bytes) < 16_000 * 2 * 0.2:  # < 200 ms
                await ws.send_json({
                    "type": "error",
                    "code": "E205",
                    "message": "Audio too short (< 200 ms).",
                    "fatal": False,
                })
                continue

            logger.info(
                f"[{session.session_id[:8]}] Utterance {session.utterance_count+1}: "
                f"{len(audio_bytes)} bytes ({len(audio_bytes)/32000:.2f}s)"
            )

            # --- Transcribe ---
            t0 = time.perf_counter()
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, asr.transcribe, audio_bytes
                )
            except Exception as exc:
                logger.error(f"[{session.session_id[:8]}] ASR error: {exc}")
                await ws.send_json({
                    "type": "error",
                    "code": "E302",
                    "message": f"ASR decode failed: {exc}",
                    "fatal": False,
                })
                continue

            latency_ms = int((time.perf_counter() - t0) * 1000)
            session.utterance_count += 1

            # Empty transcript
            if not result.get("text", "").strip():
                await ws.send_json({
                    "type": "error",
                    "code": "E303",
                    "message": "Empty transcript returned.",
                    "fatal": False,
                })
                continue

            # Send transcript
            transcript_frame = {
                "type": "transcript",
                "text": result["text"].strip(),
                "language": result.get("language", "en"),
                "latency_ms": latency_ms,
                "session_id": session.session_id,
            }
            await ws.send_json(transcript_frame)
            logger.info(
                f"[{session.session_id[:8]}] Transcript: '{result['text'].strip()}' "
                f"({latency_ms} ms)"
            )

            # --- Agent ---
            agent_result = agent.process(result["text"].strip())
            if agent_result:
                await ws.send_json({"type": "agent_response", **agent_result})

        else:
            # --- Audio chunk ---
            if len(data) > 32 * 1024:
                await ws.send_json({
                    "type": "error",
                    "code": "E201",
                    "message": "Frame too large (> 32 KB).",
                    "fatal": False,
                })
                continue

            # Check version byte (first byte of header if using packet format)
            if len(data) >= 12:
                version = data[0]
                if version != 0x01:
                    await ws.send_json({
                        "type": "error",
                        "code": "E202",
                        "message": f"Protocol version mismatch: got {version}, expected 1.",
                        "fatal": False,
                    })
                    continue

                # Parse sequence number from header (bytes 2-3, big-endian)
                seq_num = int.from_bytes(data[2:4], "big")
                flags = data[1]
                payload_len = int.from_bytes(data[4:6], "big")
                payload = data[12:12 + payload_len]

                # Gap detection
                gaps = session.gap_detector.update(seq_num)
                if gaps:
                    await ws.send_json({
                        "type": "retransmit_request",
                        "missing_seq": gaps,
                        "session_id": session.session_id,
                    })

                session.assembler.add_chunk(payload)
            else:
                # Raw PCM (development mode — no header)
                session.assembler.add_chunk(data)
