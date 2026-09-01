/**
 * @file app_state.cpp
 * @brief Application state machine implementation.
 *
 * Thread-safe FSM using a FreeRTOS queue for event dispatch and a mutex
 * for state reads. Run app_state_task() in a dedicated FreeRTOS task.
 */

#include "app_state.h"

#include <string.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"

static const char *TAG = "app_state";

/* ──────────────────────────────────────────────────────────────────────────
 * Internal state
 * ────────────────────────────────────────────────────────────────────────── */

static app_state_t      s_state  = APP_STATE_BOOT;
static QueueHandle_t    s_queue  = NULL;
static SemaphoreHandle_t s_mutex = NULL;

/* ──────────────────────────────────────────────────────────────────────────
 * Transition table
 * ────────────────────────────────────────────────────────────────────────── */

static app_state_t _transition(app_state_t current, app_event_t event)
{
    switch (current) {
    case APP_STATE_BOOT:
        if (event == APP_EVENT_INIT_DONE)  return APP_STATE_IDLE;
        break;

    case APP_STATE_IDLE:
        if (event == APP_EVENT_KWS_HIT)       return APP_STATE_KWS_RUNNING;
        if (event == APP_EVENT_NET_ERROR)      return APP_STATE_IDLE;  // stay idle
        break;

    case APP_STATE_KWS_RUNNING:
        if (event == APP_EVENT_WAKE_DETECTED)  return APP_STATE_STREAMING;
        if (event == APP_EVENT_RESET)          return APP_STATE_IDLE;
        if (event == APP_EVENT_TIMEOUT)        return APP_STATE_IDLE;
        break;

    case APP_STATE_STREAMING:
        if (event == APP_EVENT_STREAM_DONE)    return APP_STATE_WAIT_RESPONSE;
        if (event == APP_EVENT_NET_ERROR)      return APP_STATE_IDLE;
        break;

    case APP_STATE_WAIT_RESPONSE:
        if (event == APP_EVENT_TRANSCRIPT_RX)  return APP_STATE_IDLE;
        if (event == APP_EVENT_TIMEOUT)        return APP_STATE_IDLE;
        if (event == APP_EVENT_NET_ERROR)      return APP_STATE_IDLE;
        break;

    case APP_STATE_ERROR:
        break;  // no transitions out — watchdog will reboot

    default:
        break;
    }
    return current;  // no-op
}

/* ──────────────────────────────────────────────────────────────────────────
 * Public API
 * ────────────────────────────────────────────────────────────────────────── */

void app_state_init(void)
{
    s_queue = xQueueCreate(16, sizeof(app_event_t));
    s_mutex = xSemaphoreCreateMutex();
    configASSERT(s_queue);
    configASSERT(s_mutex);
    ESP_LOGI(TAG, "FSM initialised. State: BOOT");
}

void app_state_post(app_event_t event)
{
    if (s_queue) {
        xQueueSend(s_queue, &event, portMAX_DELAY);
    }
}

app_state_t app_state_get(void)
{
    app_state_t st;
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    st = s_state;
    xSemaphoreGive(s_mutex);
    return st;
}

const char *app_state_name(app_state_t state)
{
    switch (state) {
    case APP_STATE_BOOT:          return "BOOT";
    case APP_STATE_IDLE:          return "IDLE";
    case APP_STATE_KWS_RUNNING:   return "KWS_RUNNING";
    case APP_STATE_STREAMING:     return "STREAMING";
    case APP_STATE_WAIT_RESPONSE: return "WAIT_RESPONSE";
    case APP_STATE_ERROR:         return "ERROR";
    default:                      return "UNKNOWN";
    }
}

/**
 * @brief FreeRTOS task: drain the event queue and drive the FSM.
 *        Create this task from app_main() with stack ≥ 2048 words.
 */
void app_state_task(void *arg)
{
    app_event_t event;
    for (;;) {
        if (xQueueReceive(s_queue, &event, portMAX_DELAY) == pdTRUE) {
            app_state_t next = _transition(s_state, event);
            if (next != s_state) {
                ESP_LOGI(TAG, "%s → %s (event %d)",
                         app_state_name(s_state), app_state_name(next), (int)event);
                xSemaphoreTake(s_mutex, portMAX_DELAY);
                s_state = next;
                xSemaphoreGive(s_mutex);
            }
        }
    }
}
