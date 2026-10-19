# Sumo robot -- runs on the UNO Q's Linux side.
#
#   sketch.ino  -> Bridge "distance" (rear ultrasonic, cm) -> this file
#   laptop      -> MQTT "sumo/lego"   (light sensor colour + joysticks) -> this file
#   this file   -> behavior tree (behavior_tree.py) -> MQTT "sumo/drive" -> laptop -> LEGO motors
#   this file   -> MQTT "sumo/status" (human-readable messages)
#
# The laptop also runs the MQTT broker (mosquitto). If its IP changes, update
# MQTT_HOST below and redeploy.
import json
import socket
import threading
import time

import paho.mqtt.client as mqtt
from arduino.app_utils import *

import behavior_tree as bt

MQTT_HOST = "10.247.137.113"  # <-- the laptop's IP (on the Mac: ipconfig getifaddr en0)
MQTT_PORT = 1883
LEGO_TOPIC = "sumo/lego"
DRIVE_TOPIC = "sumo/drive"
STATUS_TOPIC = "sumo/status"

LOOP_HZ = 50              # minimum rate; new readings also wake the loop at once
DISTANCE_STALE_S = 0.5    # no ultrasonic reading this long -> assume nobody behind

bb = bt.Blackboard()
tree = bt.build_tree()
distance_time = -1e9
recent_cm = []            # every ultrasonic reading since the last report line
connected = False
lock = threading.Lock()
wake = threading.Event()  # set by new readings so we decide right away, not next tick


def on_distance(cm):
    """Called by the sketch about 16 times a second. cm < 0 means no echo."""
    global distance_time
    with lock:
        now = time.monotonic()
        bt.note_distance(bb, cm, now)
        distance_time = now
        recent_cm.append(round(cm))
    wake.set()


def on_status(msg):
    print(f"SKETCH SAYS: {msg}")
    publish(STATUS_TOPIC, f"sketch: {msg}")


def control_loop():
    last_behavior = None
    last_sent, last_sent_time = None, 0.0
    while True:
        wake.wait(1 / LOOP_HZ)
        wake.clear()
        with lock:
            bb.now = time.monotonic()
            if bb.now - distance_time > DISTANCE_STALE_S:
                bb.distance_cm = -1
            tree.tick(bb)
            left, right, behavior = bb.left, bb.right, bb.behavior
        # Send changes at once; otherwise a steady reminder at LOOP_HZ is plenty.
        now = time.monotonic()
        if (left, right, behavior) != last_sent or now - last_sent_time >= 1 / LOOP_HZ:
            publish(DRIVE_TOPIC, json.dumps({"left": left, "right": right, "behavior": behavior}))
            last_sent, last_sent_time = (left, right, behavior), now
        if behavior != last_behavior:
            print(f"-> {behavior}  ({left}, {right})")
            publish(STATUS_TOPIC, behavior)
            last_behavior = behavior


def publish(topic, payload):
    if connected:
        client.publish(topic, payload)


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload)
    except ValueError:
        return
    with lock:
        bb.color = data.get("color")
        bb.hue, bb.sat = data.get("hue"), data.get("sat")
        bb.stick_left = data.get("left", 0) or 0
        bb.stick_right = data.get("right", 0) or 0
        bb.lego_time = time.monotonic()
    wake.set()


def on_connect(client, userdata, flags, reason_code, properties=None):
    global connected
    connected = True
    try:
        client.socket().setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except (AttributeError, OSError):
        pass
    print(f"LAPTOP CONNECTED ({MQTT_HOST}:{MQTT_PORT})")
    client.subscribe(LEGO_TOPIC)
    client.publish(STATUS_TOPIC, "UNO Q sumo-robot connected")


def on_disconnect(client, userdata, flags, reason_code, properties=None):
    global connected
    connected = False
    print(f"lost the laptop ({reason_code}); retrying")


def report_loop():
    """Print a one-line summary every second so the App Lab log shows what's up."""
    while True:
        time.sleep(1)
        with lock:
            lego_ok = bb.now - bb.lego_time < bt.LEGO_STALE_S
            sonar_ok = time.monotonic() - distance_time < DISTANCE_STALE_S
            readings = " ".join(map(str, recent_cm))
            recent_cm.clear()
            print(f"behavior={bb.behavior:22s} wheels=({bb.left:4d},{bb.right:4d}) "
                  f"color={bb.color if lego_ok else '--'} sticks=({bb.stick_left},{bb.stick_right}) "
                  + ("LAPTOP OK" if connected else "no laptop")
                  + f"  behind(cm)={readings if sonar_ok else 'no sensor'}")


print("Sumo robot starting")
Bridge.provide("distance", on_distance)
Bridge.provide("status", on_status)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="unoq-sumo-robot")
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message
client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=10)
client.loop_start()

threading.Thread(target=control_loop, daemon=True).start()
threading.Thread(target=report_loop, daemon=True).start()
App.run()
