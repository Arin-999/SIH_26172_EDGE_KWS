# ESP32-S3 Pin Configuration

## GPIO Assignment Table

| GPIO | Signal | Direction | Function |
|---|---|---|---|
| 4 | I2S_DIN | Input | INMP441 serial data (SD pin) |
| 5 | I2S_WS | Output | Word Select / L-R clock |
| 6 | I2S_CLK | Output | Bit clock (SCK) |
| 48 | STATUS_LED | Output | Onboard RGB LED (NeoPixel) |
| 0 | BOOT_BTN | Input (pull-up) | Boot mode / user button |

## I2S Peripheral Configuration

```c
i2s_config_t i2s_config = {
    .mode                   = I2S_MODE_MASTER | I2S_MODE_RX,
    .sample_rate            = 16000,
    .bits_per_sample        = I2S_BITS_PER_SAMPLE_32BIT,  // INMP441 outputs 24-bit in 32-bit frame
    .channel_format         = I2S_CHANNEL_FMT_ONLY_LEFT,  // L/R = GND → left channel
    .communication_format   = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags       = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count          = 4,
    .dma_buf_len            = 512,                          // 512 × 4 B = 2 KB per buffer
    .use_apll               = false,
    .tx_desc_auto_clear     = false,
    .fixed_mclk             = 0,
};

i2s_pin_config_t pin_config = {
    .bck_io_num   = GPIO_NUM_6,
    .ws_io_num    = GPIO_NUM_5,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = GPIO_NUM_4,
};
```

## DMA Buffer Sizing

| Parameter | Value | Notes |
|---|---|---|
| Sample rate | 16 000 Hz | 16 kHz mono |
| Bits per sample (wire) | 32 | INMP441 uses MSB-justified 24-bit in 32-bit frame |
| Effective bits | 24 | Right-shift by 8 after read |
| DMA buffer length | 512 samples | 32 ms per buffer |
| DMA buffer count | 4 | 128 ms total DMA depth |
| Bytes per buffer | 512 × 4 = 2 048 | |
| CPU copy interval | Every 512 samples (32 ms) | |

## INMP441 Data Format Conversion

```c
// After DMA read (32-bit raw I2S words):
int32_t raw;
i2s_read(I2S_NUM_0, &raw, sizeof(raw), &bytes_read, portMAX_DELAY);

// Convert to int16 PCM (drop low 8 bits, keep 24-bit MSB portion):
int16_t pcm = (int16_t)(raw >> 14);
```
