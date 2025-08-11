#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <ArduinoWebsockets.h>
#include "driver/i2s.h"

using namespace websockets;

const char* WIFI_SSID = "iPhone von Mahir";
const char* WIFI_PASS = "12345678";
const char* WS_HOST   = "172.20.10.2";
const uint16_t WS_PORT= 8765;
const char* WS_URL    = "ws://172.20.10.2:8765";

// ===== (Seeed XIAO ESP32-C3) =====
// BCLK -> D6 (GPIO21), WS/LRCL -> D4 (GPIO6), DOUT -> D5 (GPIO7), SEL -> 3V3 (правый канал)
#define I2S_SCK_PIN 21
#define I2S_WS_PIN   6
#define I2S_SD_PIN   7

#define SAMPLE_RATE    16000
#define FRAME_SAMPLES  1024
#define DMA_LEN        256

WebsocketsClient ws;

void setupI2S() {
  pinMode(I2S_SD_PIN, INPUT_PULLDOWN);

  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 4,
    .dma_buf_len = DMA_LEN,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pins = {
    .bck_io_num   = I2S_SCK_PIN,
    .ws_io_num    = I2S_WS_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = I2S_SD_PIN
  };

  i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pins);
  i2s_set_clk(I2S_NUM_0, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_MONO);
}

void connectWiFiOnce() {
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.printf("[WiFi] Connecting to %s ...\n", WIFI_SSID);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) {
    delay(200);
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("[WiFi] IP: "); Serial.println(WiFi.localIP());
    Serial.print("[WiFi] GW: "); Serial.println(WiFi.gatewayIP());
    Serial.print("[WiFi] RSSI: "); Serial.println(WiFi.RSSI());
  } else {
    Serial.println("[WiFi] Not connected yet. ESP will keep trying in background.");
  }
}

bool tcpChecked = false;

void tcpReachabilityTest() {
  if (tcpChecked) return;
  if (WiFi.status() != WL_CONNECTED) return;

  WiFiClient cli;
  Serial.printf("[TCP] connect %s:%u ... ", WS_HOST, WS_PORT);
  if (cli.connect(WS_HOST, WS_PORT, 3000)) {
    Serial.println("OK");

    cli.printf("GET / HTTP/1.1\r\nHost: %s:%u\r\nConnection: close\r\n\r\n", WS_HOST, WS_PORT);
    uint32_t t0 = millis();
    while (cli.connected() && millis() - t0 < 2000) {
      while (cli.available()) {
        char c = cli.read();
        Serial.write(c);
      }
      delay(5);
    }
    cli.stop();
  } else {
    Serial.println("FAIL (no route / firewall / wrong bind)");
  }
  tcpChecked = true;
}

uint32_t lastWsAttemptMs = 0;

void ensureWebSocket() {
  if (WiFi.status() != WL_CONNECTED) return;
  if (ws.available()) return;
  if (millis() - lastWsAttemptMs < 2000) return;
  lastWsAttemptMs = millis();

  Serial.println("[WS] Connecting...");
  if (ws.connect(WS_URL)) {
    Serial.println("[WS] Connected");
  } else {
    Serial.println("[WS] Failed");
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);

  setupI2S();
  connectWiFiOnce();

  // Логи событий WebSocket
  ws.onEvent([](WebsocketsEvent e, String){
    if (e == WebsocketsEvent::ConnectionOpened)  Serial.println("[WS] Opened");
    if (e == WebsocketsEvent::ConnectionClosed)  Serial.println("[WS] Closed");
    if (e == WebsocketsEvent::GotPing)           Serial.println("[WS] Ping");
    if (e == WebsocketsEvent::GotPong)           Serial.println("[WS] Pong");
  });
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) { delay(100); return; }

  // 1) разовая TCP/HTTP диагностика
  tcpReachabilityTest();

  // 2) попытка открыть WS
  ensureWebSocket();

  // 3) если WS открыт — шлём аудио
  if (ws.available()) {
    static int32_t i2sBuf[FRAME_SAMPLES];
    static int16_t pcm16[FRAME_SAMPLES];

    size_t br = 0;
    if (i2s_read(I2S_NUM_0, i2sBuf, sizeof(i2sBuf), &br, portMAX_DELAY) == ESP_OK && br > 0) {
      size_t n = br / sizeof(int32_t);
      for (size_t i = 0; i < n; i++) {
        int32_t s24 = (i2sBuf[i] >> 8);     // 24-бит полезные
        pcm16[i] = (int16_t)(s24 >> 8);     // в 16 бит PCM
      }
      ws.sendBinary((const char*)pcm16, n * sizeof(int16_t));
    }
    ws.poll();
  }

  delay(1);
}
