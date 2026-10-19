// Sumo robot -- runs on the UNO Q's microcontroller.
//
// One job: measure the rear HC-SR04 ultrasonic sensor about 16 times a second and
// send the distance (cm; -1 = nothing in range, -2 = sensor didn't answer) to python/main.py over
// Bridge "distance". All the deciding happens in Python.
//
// Wiring:
//   HC-SR04 VCC -> 5V, GND -> GND, Trig -> D4, Echo -> D5
//   The HC-SR04's Echo is a 5 V signal and UNO Q pins are 3.3 V. Put a voltage
//   divider on Echo (e.g. Echo -> 1k -> D5, D5 -> 2k -> GND) or use an HC-SR04P.

#include <Arduino_RouterBridge.h>

const int TRIG_PIN = 4;
const int ECHO_PIN = 5;
const unsigned long ECHO_TIMEOUT_US = 25000;  // ~4 m round trip; longer = nothing there
const unsigned long PERIOD_MS = 60;           // ~16 readings a second (HC-SR04's fastest)

float read_distance_cm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  // Time the Echo pulse by hand (pulseIn() can lock up the board's other work).
  unsigned long start = micros();
  while (digitalRead(ECHO_PIN) == LOW) {
    if (micros() - start > ECHO_TIMEOUT_US) return -2;  // sensor never answered (wiring?)
  }
  unsigned long rise = micros();
  while (digitalRead(ECHO_PIN) == HIGH) {
    if (micros() - rise > ECHO_TIMEOUT_US) return -1;   // no echo came back
  }
  return (micros() - rise) / 58.0;  // sound goes there and back at ~343 m/s
}

void setup() {
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);
  Bridge.begin();
  Bridge.notify("status", "ultrasonic ready");
}

void loop() {
  static unsigned long last = 0;
  if (millis() - last >= PERIOD_MS) {
    last = millis();
    float cm = read_distance_cm();
    Bridge.notify("distance", cm);
  }
}
