// Gait identifier -- runs on the UNO Q's microcontroller.
//
// Two jobs:
//  1. Read the MPU-6050 50 times a second and send each reading to
//     python/main.py over Bridge "imu" as (ms, ax, ay, az, gx, gy, gz).
//     Acceleration is in g (1 g = gravity), turning speed in degrees/second.
//  2. Show the gait Python sends over Bridge "gait" ("stand", "walk", "run",
//     "bike" or anything else = unknown) as a stick figure on the LED panel.
//     When it changes, the name scrolls by once first. Pictures are in led_art.h.
//
// Wiring: MPU-6050 VCC -> 3.3V, GND -> GND, SDA -> SDA, SCL -> SCL
// (or the Qwiic socket -- both are checked at startup).

#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>
#include <Wire.h>

#include "led_art.h"

const unsigned long PERIOD_MS = 20;  // 50 readings a second
const float ACCEL_PER_G = 2048.0;    // +-16 g range (an ankle can hit 8+ g when running)
const float GYRO_PER_DPS = 16.4;     // +-2000 deg/s range (legs swing fast)

TwoWire *bus = nullptr;
uint8_t addr = 0;

void write_reg(uint8_t reg, uint8_t val) {
  bus->beginTransmission(addr);
  bus->write(reg);
  bus->write(val);
  bus->endTransmission();
}

// Look for the sensor at both of its possible addresses on both I2C buses.
bool find_sensor() {
  TwoWire *buses[] = {&Wire, &Wire1};
  const char *names[] = {"header SDA/SCL", "Qwiic"};
  for (int b = 0; b < 2; b++) {
    buses[b]->begin();
    buses[b]->setClock(400000);
    for (uint8_t a = 0x68; a <= 0x69; a++) {
      buses[b]->beginTransmission(a);
      if (buses[b]->endTransmission() == 0) {
        bus = buses[b];
        addr = a;
        bus->beginTransmission(addr);
        bus->write(0x75);  // WHO_AM_I
        bus->endTransmission(false);
        bus->requestFrom(addr, (uint8_t)1);
        int who = bus->available() ? bus->read() : -1;
        Bridge.notify("status", String("MPU-6050 found on ") + names[b] + " at 0x" +
                                    String(addr, HEX) + ", WHO_AM_I=0x" + String(who, HEX));
        return true;
      }
    }
  }
  return false;
}

// ---------------- LED panel ----------------

Arduino_LED_Matrix matrix;
uint8_t pixels[W * H];

struct Gait {
  const char *name;   // what Python sends
  const char *label;  // what scrolls by
  Picture **frames;
  int frame_count;
  unsigned long frame_ms;   // how long each picture stays up
  unsigned long ground_ms;  // how fast the ground slides (0 = still)
};
const Gait GAITS[] = {
  {"stand", "STAND", STAND, 1, 500, 0},
  {"walk", "WALK", WALK, 2, 350, 140},
  {"run", "RUN", RUN, 2, 160, 60},
  {"bike", "BIKE", BIKE, 4, 180, 90},
};
const int NO_GAIT = -1;  // "other" or anything unknown: shows "?"
const int IDLE = -2;     // not guessing yet: shows "GAIT"

int gait = NO_GAIT;
int frame = 0, ground = 0;
int scroll_pos = -1;  // >= 0 while the name is scrolling
const unsigned long SCROLL_MS = 70;
unsigned long frame_time = 0, ground_time = 0, scroll_time = 0;

uint8_t level(char c) { return c == '#' ? 7 : c == '+' ? 4 : 0; }
const uint8_t GROUND_LEVEL = 1;

const Glyph *glyph(char c) {
  for (const Glyph &g : FONT)
    if (g.c == c) return &g;
  return &FONT[0];
}

// Total width of a word in columns, with one blank column after each letter.
int text_width(const char *text) {
  int w = 0;
  for (const char *p = text; *p; p++) w += strlen(glyph(*p)->rows[0]) + 1;
  return w;
}

// Draw the word with its first column at x (can be off-screen).
void draw_text(const char *text, int x) {
  memset(pixels, 0, sizeof(pixels));
  for (const char *p = text; *p; p++) {
    const Glyph *g = glyph(*p);
    int gw = strlen(g->rows[0]);
    for (int r = 0; r < 5; r++)
      for (int c = 0; c < gw; c++)
        if (x + c >= 0 && x + c < W) pixels[(r + 1) * W + x + c] = level(g->rows[r][c]);
    x += gw + 1;
  }
  matrix.draw(pixels);
}

void draw_figure() {
  const Gait &g = GAITS[gait];
  Picture &pic = *g.frames[frame];
  for (int r = 0; r < H - 1; r++)
    for (int c = 0; c < W; c++) pixels[r * W + c] = level(pic[r][c]);
  // Ground: a solid dim line when standing, dashes sliding left when moving.
  for (int c = 0; c < W; c++)
    pixels[(H - 1) * W + c] = (g.ground_ms == 0 || (c + ground) % 3 != 0) ? GROUND_LEVEL : 0;
  matrix.draw(pixels);
}

void on_gait(String name) {
  int g = name == "idle" ? IDLE : NO_GAIT;
  for (int i = 0; i < 4; i++)
    if (name == GAITS[i].name) g = i;
  if (g == gait) return;
  gait = g;
  frame = 0;
  scroll_pos = W;  // start the name just off the right edge
  scroll_time = 0;
}

void update_panel() {
  unsigned long now = millis();
  const char *label = gait == IDLE ? "GAIT" : gait == NO_GAIT ? "?" : GAITS[gait].label;
  if (scroll_pos >= 0) {
    if (now - scroll_time < SCROLL_MS) return;
    scroll_time = now;
    draw_text(label, scroll_pos - W);
    // Keep "?" and "GAIT" sitting in the middle; scroll real names all the way out.
    if (gait < 0 && scroll_pos - W <= (W - text_width(label) + 1) / 2) return;
    if (--scroll_pos + text_width(label) < 0) scroll_pos = -1;
    return;
  }
  if (gait < 0) return;
  const Gait &g = GAITS[gait];
  bool changed = false;
  if (now - frame_time >= g.frame_ms) {
    frame_time = now;
    frame = (frame + 1) % g.frame_count;
    changed = true;
  }
  if (g.ground_ms && now - ground_time >= g.ground_ms) {
    ground_time = now;
    ground = (ground + 1) % 3;
    changed = true;
  }
  if (changed) draw_figure();
}

// ---------------- setup / loop ----------------

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);  // brightness 0-7
  on_gait("idle");             // show "GAIT" until Python says otherwise
  Bridge.begin();
  Bridge.provide_safe("gait", on_gait);
  while (!find_sensor()) {
    Bridge.notify("status", "no MPU-6050 found -- check wiring");
    delay(2000);
  }
  write_reg(0x6B, 0x01);  // wake up, use the gyro's clock (steadier)
  write_reg(0x1A, 0x03);  // smooth out jitter above ~44 Hz
  write_reg(0x1B, 0x18);  // gyro +-2000 deg/s
  write_reg(0x1C, 0x18);  // accel +-16 g
  Bridge.notify("status", "MPU-6050 ready");
}

void loop() {
  update_panel();
  static unsigned long last = 0;
  if (millis() - last < PERIOD_MS) return;
  last += PERIOD_MS;
  if (millis() - last > 5 * PERIOD_MS) last = millis();  // fell far behind: don't burst

  // 14 bytes starting at ACCEL_XOUT_H: accel x,y,z, temperature, gyro x,y,z
  bus->beginTransmission(addr);
  bus->write(0x3B);
  if (bus->endTransmission(false) != 0 || bus->requestFrom(addr, (uint8_t)14) != 14) {
    Bridge.notify("status", "sensor read failed");
    return;
  }
  int16_t raw[7];
  for (int i = 0; i < 7; i++) raw[i] = (bus->read() << 8) | bus->read();

  Bridge.notify("imu", millis(),
                raw[0] / ACCEL_PER_G, raw[1] / ACCEL_PER_G, raw[2] / ACCEL_PER_G,
                raw[4] / GYRO_PER_DPS, raw[5] / GYRO_PER_DPS, raw[6] / GYRO_PER_DPS);
}
