/**
 * @file app_state.h
 * @brief Edge KWS application state machine definitions.
 *
 * Defines the finite state machine (FSM) states, events, and transition
 * API for the ESP32-S3 keyword spotting firmware.
 *
 * State transitions:
 *   BOOT → IDLE → KWS_RUNNING → STREAMING → WAIT_RESPONSE → IDLE
 *                      ↑___________________________|
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/* ──────────────────────────────────────────────────────────────────────────
 * State enumeration
 * ────────────────────────────────────────────────────────────────────────── */

typedef enum {
    APP_STATE_BOOT          = 0,  /**< Power-on / initialisation          */
    APP_STATE_IDLE          = 1,  /**< Listening, KWS inference running   */
    APP_STATE_KWS_RUNNING   = 2,  /**< KWS hit — debounce window active   */
    APP_STATE_STREAMING     = 3,  /**< Streaming PCM audio over WebSocket */
    APP_STATE_WAIT_RESPONSE = 4,  /**< Awaiting ASR transcript from server*/
    APP_STATE_ERROR         = 5,  /**< Unrecoverable error — reboot       */
} app_state_t;

/* ──────────────────────────────────────────────────────────────────────────
 * Event enumeration
 * ────────────────────────────────────────────────────────────────────────── */

typedef enum {
    APP_EVENT_INIT_DONE     = 0,  /**< Hardware init complete              */
    APP_EVENT_KWS_HIT       = 1,  /**< Single frame above threshold        */
    APP_EVENT_WAKE_DETECTED = 2,  /**< Debounce fired — confirmed wake     */
    APP_EVENT_STREAM_DONE   = 3,  /**< End-of-utterance marker sent        */
    APP_EVENT_TRANSCRIPT_RX = 4,  /**< Server returned transcript JSON     */
    APP_EVENT_TIMEOUT       = 5,  /**< No transcript within ASR timeout    */
    APP_EVENT_NET_ERROR     = 6,  /**< WebSocket / WiFi error              */
    APP_EVENT_RESET         = 7,  /**< Manual soft-reset to IDLE           */
} app_event_t;

/* ──────────────────────────────────────────────────────────────────────────
 * Public API
 * ────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Initialise the application state machine.
 *        Must be called once from app_main() before posting events.
 */
void app_state_init(void);

/**
 * @brief Post an event to the state machine (thread-safe).
 * @param event  The event to dispatch.
 */
void app_state_post(app_event_t event);

/**
 * @brief Return the current application state (thread-safe read).
 */
app_state_t app_state_get(void);

/**
 * @brief Return a human-readable name for a state (for logging).
 */
const char *app_state_name(app_state_t state);

#ifdef __cplusplus
}
#endif
