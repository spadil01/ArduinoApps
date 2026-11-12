// Impenetrable safe -- runs on the UNO Q's microcontroller.
//
// This sketch is just the safe's "hands": it turns lights on/off, moves the
// servo, and reports button presses. All the deciding happens in python/main.py.
//
//   python -> Bridge "light" (pin, 1/0)   turn one light on or off
//   python -> Bridge "servo" (angle)      move the lock servo (0-180 degrees)
//   sketch -> Bridge "button"             the button was just pressed
//   (after each move the servo rests: no pulses, so it doesn't buzz)
//   sketch -> Bridge "status" (text)      messages for the log
//
// Wiring:
//   D8, D9, D10  yellow LEDs (digit 1, 2, 3)
//   D6           green LED (through a transistor)
//   D2           inside LEDs (through a transistor)
//   D3           servo signal
//   D4           push button

#include <Arduino_RouterBridge.h>
#include <Servo.h>

const int LIGHT_PINS[] = {8, 9, 10, 6, 2};
const int SERVO_PIN = 3;
const int BUTTON_PIN = 4;
const unsigned long DEBOUNCE_MS = 30;  // a press must stay steady this long to count
const unsigned long SERVO_REST_MS = 1000;  // after a move, stop pushing (stops the buzzing)

Servo lock;
int idle_level;       // what the button pin reads when nobody is pressing
int stable_level;
int last_reading;
unsigned long last_change = 0;
unsigned long servo_moved = 0;

bool is_light(int pin) {
  for (int p : LIGHT_PINS) if (p == pin) return true;
  return false;
}

void on_light(int pin, int on) {
  if (is_light(pin)) digitalWrite(pin, on ? HIGH : LOW);
}

void on_servo(int angle) {
  if (!lock.attached()) lock.attach(SERVO_PIN);
  lock.write(constrain(angle, 0, 180));
  servo_moved = millis();
}

void setup() {
  for (int p : LIGHT_PINS) {
    pinMode(p, OUTPUT);
    digitalWrite(p, LOW);
  }
  // Pull-up keeps the pin from "floating". Whatever it reads now (nobody
  // pressing) is "not pressed", so the button works whichever way it's wired.
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  delay(50);
  idle_level = stable_level = last_reading = digitalRead(BUTTON_PIN);

  Bridge.begin();
  Bridge.provide_safe("light", on_light);
  Bridge.provide_safe("servo", on_servo);
  Bridge.notify("status", String("hardware ready, button reads ") +
                (idle_level == HIGH ? "HIGH" : "LOW") + " when not pressed");
}

void loop() {
  // The servo has had time to arrive: stop sending it pulses so it goes quiet.
  // Its gears are stiff enough to hold the lock where it is.
  if (lock.attached() && millis() - servo_moved >= SERVO_REST_MS) {
    lock.detach();
  }

  int reading = digitalRead(BUTTON_PIN);
  if (reading != last_reading) {
    last_reading = reading;
    last_change = millis();
  }
  // Only believe a change once it has stayed the same for DEBOUNCE_MS
  // (buttons "bounce" on and off for a few milliseconds when pressed).
  if (reading != stable_level && millis() - last_change >= DEBOUNCE_MS) {
    stable_level = reading;
    if (stable_level != idle_level) Bridge.notify("button", 1);
  }
}
