/**
 * @file main.cpp
 * @brief ESP32-S3 Edge KWS firmware entry point.
 *
 * Task layout:
 *   app_state_task   — FSM event dispatcher          (priority 5, 2 KB stack)
 *   audio_task       — I2S DMA read + pre-roll ring  (priority 6, 4 KB stack)
 *   kws_task         — MFCC + TFLite + matcher       (priority 4, 8 KB stack)
 *   stream_task      — WebSocket TX + EOT            (priority 5, 6 KB stack)
 *
 * All inter-task communication uses FreeRTOS queues.
 * KWS inference and audio capture run concurrently via ping-pong buffers.
 */

#include <stdio.h>
#include <string.h>

#include "esp_log.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "app_state.h"

static const char *TAG = "main";

/* ──────────────────────────────────────────────────────────────────────────
 * Forward declarations (implemented in separate .cpp files per subsystem)
 * ────────────────────────────────────────────────────────────────────────── */

// void audio_task(void *arg);   // firmware/esp32/main/audio.cpp
// void kws_task(void *arg);     // firmware/esp32/main/kws.cpp
// void stream_task(void *arg);  // firmware/esp32/main/stream.cpp
// void wifi_init(void);         // firmware/esp32/main/wifi.cpp

/* ──────────────────────────────────────────────────────────────────────────
 * app_main
 * ────────────────────────────────────────────────────────────────────────── */

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "=== KWS Edge Node v1.0 booting ===");

    /* NVS (needed by WiFi) */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /* Initialise FSM */
    app_state_init();
    xTaskCreate(app_state_task, "app_state", 2048, NULL, 5, NULL);

    /* TODO: Initialise WiFi → post APP_EVENT_INIT_DONE on connect */
    /* wifi_init(); */

    /* TODO: Launch audio, KWS, and stream tasks */
    /* xTaskCreate(audio_task,  "audio",  4096, NULL, 6, NULL); */
    /* xTaskCreate(kws_task,    "kws",    8192, NULL, 4, NULL); */
    /* xTaskCreate(stream_task, "stream", 6144, NULL, 5, NULL); */

    /* Temporary: signal INIT_DONE so state machine enters IDLE */
    app_state_post(APP_EVENT_INIT_DONE);

    ESP_LOGI(TAG, "Tasks started. Current state: %s",
             app_state_name(app_state_get()));

    /* app_main may return — FreeRTOS scheduler continues running tasks */
}
