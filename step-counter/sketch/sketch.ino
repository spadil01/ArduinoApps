// Step counter -- runs on the UNO Q's microcontroller.
//
// Two sensors in, one LED panel out:
//  1. MPU-6050 read 100 times a second. Each reading goes to python/main.py
//     over Bridge "imu" as whole numbers (smaller than decimals, so 100/s fits):
//     (ms, ax, ay, az, gx, gy, gz, dist_mm). Python turns them back into g,
//     degrees/second and cm.
//  2. HC-SR04 ultrasonic pinged 20 times a second. The newest distance rides
//     along with every IMU reading (-1 = no echo / out of range,
//     -2 = the sensor isn't answering at all: wiring).
//  The LED panel shows whatever text Python sends over Bridge "show"
//  (centered if it fits, scrolling if it doesn't) -- the step count, or the
//  countdown while recording. Letters are in font.h. The board stands on its
//  side, so text is turned to match (ROTATE): 2 digits fit, more scroll.
//
// Wiring:
//   MPU-6050: VCC -> 3.3V, GND -> GND, SDA -> SDA, SCL -> SCL (or the Qwiic socket)
//   HC-SR04:  VCC -> 5V, GND -> GND, Trig -> D7,
//             Echo -> 1k resistor -> D8, plus 2k resistor from D8 to GND
//             (the resistors shrink its 5V answer to a safe ~3.3V)

#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>
#include <Wire.h>

#include "font.h"

const unsigned long PERIOD_MS = 10;  // 100 IMU readings a second
const unsigned long PING_MS = 50;    // 20 distance readings a second
const unsigned long NO_ECHO_MS = 300;  // no echo for this long -> distance unknown
const long MAX_ECHO_US = 4 * 5800;     // 4 m; longer = nothing in range

const int TRIG_PIN = 7, ECHO_PIN = 8;

// ---------------- MPU-6050 ----------------

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
        Bridge.notify("status", String("MPU-6050 found on ") + names[b] + " at 0x" + String(addr, HEX));
        return true;
      }
    }
  }
  return false;
}

// ---------------- HC-SR04 ----------------
// The echo pin stays high for as long as the sound took to come back.
// An interrupt times it, so the IMU never waits for the sound.

volatile unsigned long echo_rise = 0;
volatile long echo_us = 0;
volatile bool echo_new = false;

void on_echo() {
  unsigned long now = micros();
  if (digitalRead(ECHO_PIN)) {
    echo_rise = now;
  } else {
    echo_us = now - echo_rise;
    echo_new = true;
  }
}

int dist_mm = -1;
unsigned long last_echo = 0;   // last good reading
unsigned long last_pulse = 0;  // last answer of any kind (a miss is a very long pulse)
bool warned_no_echo = false;

void update_distance() {
  static unsigned long last_ping = 0;
  unsigned long now = millis();
  noInterrupts();
  bool fresh = echo_new;
  long us = echo_us;
  echo_new = false;
  interrupts();
  if (fresh) {
    last_pulse = now;
    warned_no_echo = false;
    // A single missed bounce is common: keep the last good reading until
    // NO_ECHO_MS passes without a new one.
    if (us < MAX_ECHO_US) {
      dist_mm = us * 10 / 58;  // sound: 58 us per cm, there and back
      last_echo = now;
    }
  }
  if (now - last_echo > NO_ECHO_MS) dist_mm = -1;  // nothing in range
  if (now - last_pulse > NO_ECHO_MS) dist_mm = -2;  // no answer at all
  if (now - last_pulse > NO_ECHO_MS && !warned_no_echo && now > 3000) {
    warned_no_echo = true;  // not even a "missed" answer: the sensor isn't connected
    Bridge.notify("status", "no echo from the HC-SR04 -- check Trig/Echo wiring");
  }
  if (now - last_ping >= PING_MS) {
    last_ping = now;
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);
  }
}

// ---------------- LED panel ----------------

// Which way the board stands: 0 = flat, 1 = turned clockwise, 3 = turned
// counter-clockwise. If the numbers come out upside down, swap 1 and 3.
const int ROTATE = 1;
const int VW = ROTATE % 2 ? H : W, VH = ROTATE % 2 ? W : H;  // the panel as you see it

Arduino_LED_Matrix matrix;
uint8_t pixels[W * H];
String text = "STEPS";
int scroll_x = 0;
unsigned long scroll_time = 0;
const unsigned long SCROLL_MS = 80;

const Glyph *glyph(char c) {
  for (const Glyph &g : FONT)
    if (g.c == c) return &g;
  return &FONT[0];  // unknown letters show as a space
}

int text_width() {
  int w = 0;
  for (unsigned i = 0; i < text.length(); i++) w += strlen(glyph(text[i])->rows[0]) + 1;
  return w - 1;  // no blank column after the last letter
}

// Light pixel (x, y) as you see it, wherever that is on the turned panel.
void set_pixel(int x, int y) {
  int px = x, py = y;
  if (ROTATE == 1) px = y, py = H - 1 - x;
  if (ROTATE == 3) px = W - 1 - y, py = x;
  pixels[py * W + px] = 7;
}

void draw_text(int x) {
  memset(pixels, 0, sizeof(pixels));
  for (unsigned i = 0; i < text.length(); i++) {
    const Glyph *g = glyph(text[i]);
    int gw = strlen(g->rows[0]);
    for (int r = 0; r < 5; r++)
      for (int c = 0; c < gw; c++)
        if (x + c >= 0 && x + c < VW && g->rows[r][c] == '#') set_pixel(x + c, r + (VH - 5) / 2);
    x += gw + 1;
  }
  matrix.draw(pixels);
}

void on_show(String t) {
  t.toUpperCase();
  if (t == text) return;
  text = t;
  scroll_x = VW;  // long text starts just off the right edge
  draw_text(text_width() <= VW ? (VW - text_width()) / 2 : scroll_x);
}

void update_panel() {
  int tw = text_width();
  if (tw <= VW || millis() - scroll_time < SCROLL_MS) return;
  scroll_time = millis();
  if (--scroll_x < -tw) scroll_x = VW;  // loop forever
  draw_text(scroll_x);
}

// ---------------- setup / loop ----------------

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);  // brightness 0-7
  Bridge.begin();
  Bridge.provide_safe("show", on_show);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(ECHO_PIN), on_echo, CHANGE);

  while (!find_sensor()) {
    Bridge.notify("status", "no MPU-6050 found -- check wiring");
    delay(2000);
  }
  write_reg(0x6B, 0x01);  // wake up, use the gyro's clock (steadier)
  write_reg(0x1A, 0x03);  // smooth out jitter above ~44 Hz (half of 100 readings/s)
  write_reg(0x1B, 0x18);  // gyro +-2000 deg/s (legs swing fast)
  write_reg(0x1C, 0x18);  // accel +-16 g (a running heel strike can hit 8+ g)
  Bridge.notify("status", "MPU-6050 ready");
}

void loop() {
  update_panel();
  update_distance();

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

  Bridge.notify("imu", millis(), (int)raw[0], (int)raw[1], (int)raw[2],
                (int)raw[4], (int)raw[5], (int)raw[6], dist_mm);
}
