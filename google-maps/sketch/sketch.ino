// Campus navigator: STM32 side of the UNO Q.
// Finds the MPU-6050 on either I2C bus, reads it at 50 Hz, and sends samples
// to Python over the Bridge. Once a second it also sends a status report
// ("imu_diag") so the phone can show exactly what's wrong if nothing arrives.

#include <Wire.h>
#include "Arduino_RouterBridge.h"

TwoWire *bus = nullptr;      // the I2C bus the IMU was found on
uint8_t mpuAddr = 0;         // 0x68 (AD0 low) or 0x69 (AD0 high)
int busId = -1;              // 0 = Wire (A4/A5 header), 1 = Wire1 (Qwiic)
uint8_t whoami = 0;

const unsigned long SAMPLE_US = 20000;   // 50 Hz
unsigned long lastSample = 0, lastDiag = 0, lastSearch = 0;
unsigned long okCount = 0, failCount = 0;
int failStreak = 0;

const float ACC_LSB_PER_G    = 8192.0;   // +-4 g
const float GYRO_LSB_PER_DPS = 65.5;     // +-500 deg/s

bool readReg(TwoWire &w, uint8_t addr, uint8_t reg, uint8_t &val) {
  w.beginTransmission(addr);
  w.write(reg);
  if (w.endTransmission(false) != 0) return false;
  if (w.requestFrom(addr, (uint8_t)1) != 1) return false;
  val = w.read();
  return true;
}

void writeReg(uint8_t reg, uint8_t val) {
  bus->beginTransmission(mpuAddr);
  bus->write(reg);
  bus->write(val);
  bus->endTransmission();
}

// Try both buses and both addresses; WHO_AM_I (0x75) answers 0x68 on a real MPU-6050
// (clones sometimes answer 0x70, 0x72 or 0x98, so any reply counts as "found").
bool findIMU() {
  TwoWire *buses[2] = { &Wire, &Wire1 };
  uint8_t addrs[2] = { 0x68, 0x69 };
  for (int b = 0; b < 2; b++) {
    buses[b]->begin();
    buses[b]->setClock(400000);
    for (int a = 0; a < 2; a++) {
      uint8_t v;
      if (readReg(*buses[b], addrs[a], 0x75, v)) {
        bus = buses[b]; busId = b; mpuAddr = addrs[a]; whoami = v;
        return true;
      }
    }
  }
  return false;
}

void setupIMU() {
  writeReg(0x6B, 0x00);  // wake up
  writeReg(0x1A, 0x03);  // digital low-pass ~44 Hz
  writeReg(0x1B, 0x08);  // gyro +-500 dps
  writeReg(0x1C, 0x08);  // accel +-4 g
}

bool readIMU(float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  bus->beginTransmission(mpuAddr);
  bus->write(0x3B);
  if (bus->endTransmission(false) != 0) return false;
  if (bus->requestFrom(mpuAddr, (uint8_t)14) != 14) return false;
  int16_t raw[7];
  for (int i = 0; i < 7; i++) raw[i] = (bus->read() << 8) | bus->read();
  ax = raw[0] / ACC_LSB_PER_G;  ay = raw[1] / ACC_LSB_PER_G;  az = raw[2] / ACC_LSB_PER_G;
  gx = raw[4] / GYRO_LSB_PER_DPS; gy = raw[5] / GYRO_LSB_PER_DPS; gz = raw[6] / GYRO_LSB_PER_DPS;
  return true;
}

void setup() {
  Bridge.begin();
  if (findIMU()) setupIMU();
}

void loop() {
  unsigned long nowMs = millis();

  // Not found yet (or lost): keep searching every 2 s, so replugging the cable just works.
  if (!bus && nowMs - lastSearch > 2000) {
    lastSearch = nowMs;
    if (findIMU()) setupIMU();
  }

  // Status report once a second: bus (-1 = not found), address, WHO_AM_I, good and failed reads.
  if (nowMs - lastDiag > 1000) {
    lastDiag = nowMs;
    Bridge.notify("imu_diag", busId, (int)mpuAddr, (int)whoami, (int)okCount, (int)failCount);
  }

  if (!bus) return;
  unsigned long now = micros();
  if (now - lastSample < SAMPLE_US) return;
  lastSample = now;

  float ax, ay, az, gx, gy, gz;
  if (readIMU(ax, ay, az, gx, gy, gz)) {
    okCount++;
    failStreak = 0;
    Bridge.notify("imu", nowMs, ax, ay, az, gx, gy, gz);
  } else {
    failCount++;
    if (++failStreak >= 50) { bus = nullptr; busId = -1; failStreak = 0; }   // 1 s of failures in a row: search again
  }
}