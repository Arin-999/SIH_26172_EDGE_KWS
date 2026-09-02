#include <WiFi.h>
#include <WebSocketsClient.h>
#include <driver/i2s.h>

// #include "model_data.h" // Uncomment this when you generate the ML model

// --- Configuration ---
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* websocket_server = "192.168.31.74"; // Your server's local IP
const uint16_t websocket_port = 8765;

// --- I2S Pins (INMP441) ---
#define I2S_WS 5
#define I2S_SD 4
#define I2S_SCK 6
#define I2S_PORT I2S_NUM_0
#define BUFFER_SIZE 4000

WebSocketsClient webSocket;
bool streaming = false;
unsigned long last_stream_time = 0;

void i2s_init() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 64,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  
  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };
  
  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
}

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.println("[WS] Disconnected!");
      break;
    case WStype_CONNECTED:
      Serial.println("[WS] Connected to server");
      break;
    case WStype_TEXT:
      Serial.printf("[WS] Server sent: %s\n", payload);
      break;
    case WStype_BIN:
    case WStype_ERROR:
    case WStype_FRAGMENT_TEXT_START:
    case WStype_FRAGMENT_BIN_START:
    case WStype_FRAGMENT:
    case WStype_FRAGMENT_FIN:
      break;
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("Starting Edge KWS...");
  
  i2s_init();
  
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected.");
  
  webSocket.begin(websocket_server, websocket_port, "/v1/stream");
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000);
}

void loop() {
  webSocket.loop();
  
  int16_t i2s_read_buff[BUFFER_SIZE];
  size_t bytes_read;
  
  i2s_read(I2S_PORT, (void *)i2s_read_buff, BUFFER_SIZE * sizeof(int16_t), &bytes_read, portMAX_DELAY);
  
  if (!streaming) {
    // [TODO] Calculate MFCC from i2s_read_buff
    // [TODO] Run TFLite Micro inference using g_model from model_data.h
    
    // For testing: randomly trigger a fake wake word detection
    if (random(0, 200) == 0) {
      Serial.println("Wake word detected! Starting stream...");
      streaming = true;
      last_stream_time = millis();
    }
  }
  
  if (streaming) {
    if (webSocket.isConnected()) {
      webSocket.sendBIN((uint8_t*)i2s_read_buff, bytes_read);
      
      // Stop streaming after 3 seconds
      if (millis() - last_stream_time > 3000) {
        Serial.println("End of utterance. Sending marker.");
        uint8_t eot = 0xFF;
        webSocket.sendBIN(&eot, 1);
        streaming = false;
      }
    } else {
      streaming = false; // abort if disconnected
    }
  }
}
