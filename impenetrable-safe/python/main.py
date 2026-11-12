# Impenetrable safe -- runs on the UNO Q's Linux side.
#
# This file does NO thinking. It's just a messenger between the laptop and the
# microcontroller (sketch.ino). The camera, the neural network and the game
# rules all live on the laptop (AI_Robotics/impenetrable_safe/safe_lock/safe_game.py).
#
#   laptop -> port 7100 "light <pin> <0|1>"  -> sketch turns a light on/off
#   laptop -> port 7100 "servo <angle>"      -> sketch moves the lock servo
#   sketch -> button press                   -> port 7100 "button" to every laptop
import socketserver
import threading

from arduino.app_utils import *

COMMAND_PORT = 7100

laptops = []              # open connections to laptops
laptops_lock = threading.Lock()


def on_button(_):
    print("button pressed")
    with laptops_lock:
        for conn in list(laptops):
            try:
                conn.sendall(b"button\n")
            except OSError:
                laptops.remove(conn)


def on_status(msg):
    print(f"SKETCH SAYS: {msg}")


class Laptop(socketserver.StreamRequestHandler):
    """One laptop connection. Each line it sends is one command."""

    def handle(self):
        print("laptop connected")
        with laptops_lock:
            laptops.append(self.connection)
        try:
            for raw in self.rfile:
                words = raw.decode().split()
                try:
                    if len(words) == 3 and words[0] == "light":
                        Bridge.notify("light", int(words[1]), int(words[2]))
                    elif len(words) == 2 and words[0] == "servo":
                        Bridge.notify("servo", int(words[1]))
                    else:
                        raise ValueError
                except ValueError:
                    print(f"unknown command: {raw!r}")
        finally:
            with laptops_lock:
                if self.connection in laptops:
                    laptops.remove(self.connection)
            print("laptop disconnected")


Bridge.provide("button", on_button)
Bridge.provide("status", on_status)
socketserver.ThreadingTCPServer.allow_reuse_address = True  # restart without "port in use"
socketserver.ThreadingTCPServer.daemon_threads = True
server = socketserver.ThreadingTCPServer(("0.0.0.0", COMMAND_PORT), Laptop)
threading.Thread(target=server.serve_forever, daemon=True).start()
print(f"ready: waiting for the laptop on port {COMMAND_PORT}")

App.run()
