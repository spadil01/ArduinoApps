# The sumo robot's brain: a small behavior tree.
#
# A behavior tree is a priority list. Every tick (50 times a second) we start at
# the top and ask each branch "do you want to drive?". The first one that says
# yes picks the wheel speeds and the rest are skipped:
#
#   Selector (first one that succeeds wins)
#   |-- No fresh LEGO data?          -> STOP          (safety)
#   |-- Not started yet?             -> ASLEEP        (wake: both sticks down 1 s, let go)
#   |-- Backed out of the ring?      -> LOST          (stop, back to sleep; laptop plays a song)
#   |-- Blue edge seen / escaping?   -> ESCAPE        (back up, then spin; ignores controller)
#   |-- Opponent close behind?       -> FLEE FORWARD  (+ the controller's turning, never backwards)
#   `-- otherwise                    -> CONTROLLER    (centered sticks = sit still)
#
# This file only does the deciding; it never talks to hardware, so it can be
# tested on a laptop (see test_behavior_tree.py).

SUCCESS, FAILURE, RUNNING = "success", "failure", "running"

# --- Tuning -------------------------------------------------------------------
EDGE_COLORS = {3}          # LEGO colour numbers that mean "ring edge" (3 = Blue)
# The sensor calls pale light-blue tape "no colour", so also check the hue
# (colour wheel angle) ourselves. Measured: floor hue 16 / sat 32, light blue
# tape 188 / 12, dark blue tape 207 / 52.
EDGE_HUE = (170, 250)      # blue-ish hues
EDGE_MIN_SAT = 8           # below this it's basically grey/white, hue means nothing
# Speeds are kept low on purpose: faster made the wheels slip when turning
# and carried the robot past the tape before it could react.
DRIVE_MAX = 50             # controller: top % speed going forward/back
TURN_MAX = 35              # controller: top % speed added for turning (spins slip most)
CURVE_INNER = 0.3          # on a curve, the inside wheel goes this fraction of the outside one
ESCAPE_BACK_SPEED = 40     # % speed while backing away from the edge
ESCAPE_BACK_S = 0.7        # keep backing up this long after blue disappears
ESCAPE_TURN_SPEED = 40     # % speed while spinning to face back into the ring
ESCAPE_TURN_S = 0.6
ESCAPE_CURVE_RIGHT = True  # which way to curve when backing up with an opponent behind
OPPONENT_CM = 20          # something closer than this behind us = opponent
OPPONENT_CONFIRM = 2       # close readings in a row before we believe it (ignores blips)
OPPONENT_HOLD_S = 0.3      # keep fleeing this long after the last close reading
FLEE_SPEED = 50            # forward speed when running away from the opponent
DEADZONE = 5               # joystick % that still counts as "centered"
LEGO_STALE_S = 0.5         # no LEGO data for this long -> stop
WAKE_STICK = -90           # both sticks below this (pulled down) = wake-up signal
WAKE_HOLD_S = 1.0          # ... held this long, then let go
SLEEP_AFTER_LOST_S = 3.0   # laptop gone this long -> back to sleep
BACKWARD_MIN = 20          # wheels' sum below -this = clearly backing up (not spinning)
RECENT_TAPE_S = 1.5        # tape seen this recently -> backing onto it is part of an escape, not 'out'


def clamp(v, lo=-100, hi=100):
    return max(lo, min(hi, int(round(v))))


class Blackboard:
    """Everything the tree knows about the world (filled in by main.py)."""

    def __init__(self):
        self.now = 0.0
        self.color = None          # LEGO colour number from the light sensor
        self.hue = None            # light sensor hue, 0-360
        self.sat = None            # light sensor saturation, 0-100
        self.stick_left = 0        # controller left stick, -100..100 (up = +)
        self.stick_right = 0       # controller right stick
        self.lego_time = -1e9      # when the last LEGO message arrived
        self.distance_cm = -1      # rear ultrasonic, -1 = nothing in range
        self.opponent_time = -1e9  # when the opponent was last seen behind us
        self.close_count = 0       # close ultrasonic readings in a row
        self.awake = False         # False until the wake-up stick signal
        # outputs
        self.left = 0
        self.right = 0
        self.behavior = "stop"

    def drive(self, left, right, behavior):
        self.left, self.right, self.behavior = clamp(left), clamp(right), behavior


def note_distance(bb, cm, now):
    """Record one ultrasonic reading (cm < 0 = nothing). Only after
    OPPONENT_CONFIRM close readings in a row do we count it as an opponent."""
    bb.distance_cm = cm
    bb.close_count = bb.close_count + 1 if 0 < cm < OPPONENT_CM else 0
    if bb.close_count >= OPPONENT_CONFIRM:
        bb.opponent_time = now


# --- Generic tree building blocks --------------------------------------------

class Selector:
    """Try children in order; stop at the first one that doesn't fail."""

    def __init__(self, *children):
        self.children = children

    def tick(self, bb):
        for child in self.children:
            status = child.tick(bb)
            if status != FAILURE:
                return status
        return FAILURE


class Sequence:
    """Run children in order; stop at the first one that doesn't succeed."""

    def __init__(self, *children):
        self.children = children

    def tick(self, bb):
        for child in self.children:
            status = child.tick(bb)
            if status != SUCCESS:
                return status
        return SUCCESS


class Condition:
    def __init__(self, test):
        self.test = test

    def tick(self, bb):
        return SUCCESS if self.test(bb) else FAILURE


class Action:
    def __init__(self, act):
        self.act = act

    def tick(self, bb):
        return self.act(bb) or SUCCESS


# --- Conditions ---------------------------------------------------------------

def lego_data_stale(bb):
    # color None = the laptop lost the light sensor; we can't see the edge.
    return bb.now - bb.lego_time > LEGO_STALE_S or bb.color is None


def on_edge(bb):
    if bb.color in EDGE_COLORS:
        return True
    if bb.hue is None or bb.sat is None:
        return False
    return EDGE_HUE[0] <= bb.hue <= EDGE_HUE[1] and bb.sat >= EDGE_MIN_SAT


def opponent_behind(bb):
    return bb.now - bb.opponent_time < OPPONENT_HOLD_S


def opponent_closeness(bb):
    """0 = no opponent behind, up to 1 = touching. Lost the reading for a
    moment but saw them very recently? Call it halfway."""
    if not opponent_behind(bb):
        return 0.0
    if 0 < bb.distance_cm < OPPONENT_CM:
        return 1 - bb.distance_cm / OPPONENT_CM
    return 0.5


# --- Actions ------------------------------------------------------------------

def stop(bb):
    if bb.now - bb.lego_time > SLEEP_AFTER_LOST_S:
        bb.awake = False  # laptop gone for a while -> needs the wake-up signal again
    laptop_gone = bb.now - bb.lego_time > LEGO_STALE_S
    bb.drive(0, 0, "stop: no LEGO data" if laptop_gone else "stop: no light sensor")


class WaitForStart:
    """Keep the robot completely still until the driver gives the wake-up
    signal: both sticks pulled all the way down for WAKE_HOLD_S, then let go.
    It wakes on the let-go so it doesn't lurch backwards."""

    def __init__(self):
        self.held_since = None  # when both sticks went down
        self.ready = False      # held long enough; waiting for the let-go

    def tick(self, bb):
        if bb.awake:
            return FAILURE  # awake -> let the next branches drive
        left, right = sticks(bb)
        both_down = bb.stick_left <= WAKE_STICK and bb.stick_right <= WAKE_STICK
        if both_down:
            if self.held_since is None:
                self.held_since = bb.now
            self.ready = bb.now - self.held_since >= WAKE_HOLD_S
        elif self.ready and left == 0 and right == 0:
            bb.awake, self.held_since, self.ready = True, None, False
            bb.drive(0, 0, "awake!")
            return RUNNING
        elif not self.ready or left > 0 or right > 0:
            # let go too early, or pushed a stick up -> start over
            self.held_since, self.ready = None, False
        bb.drive(0, 0, "asleep: let go to start" if self.ready else "asleep: hold both sticks down")
        return RUNNING


class OutOfRing:
    """Notice when the robot has gone fully out of the ring.

    The light sensor is at the FRONT. Driving backwards from inside the ring,
    the back crosses the tape first, so by the time the front sensor reaches
    the tape the whole robot is out. Exception: if the sensor saw the tape in
    the last RECENT_TAPE_S (whatever we were doing then), backing onto it again
    is just part of an escape -- fixing an overshoot, or the sensor flickering
    at the tape's edge.
    (Driving out forwards can't be told apart from an overshoot with one
    sensor, so it isn't detected.)
    """

    def __init__(self):
        self.on_tape = False
        self.last_tape_seen = -1e9
        self.was_awake = False

    def tick(self, bb):
        if not bb.awake:
            self.was_awake = False
            return FAILURE
        if not self.was_awake:  # just woke up: start fresh, inside the ring
            self.on_tape, self.last_tape_seen, self.was_awake = False, -1e9, True
        edge = on_edge(bb)
        lost = False
        if edge and not self.on_tape:  # front sensor just reached the tape
            backing_up = bb.left + bb.right < -BACKWARD_MIN  # what we told the wheels last tick
            lost = backing_up and bb.now - self.last_tape_seen > RECENT_TAPE_S
        if edge:
            self.last_tape_seen = bb.now
        self.on_tape = edge
        if lost:
            bb.awake = False
            bb.drive(0, 0, "lost: out of the ring")
            return RUNNING
        return FAILURE


class Escape:
    """Back away from the blue edge, then spin to face into the ring.

    Keeps RUNNING (holding the lower branches off) until the whole move is done,
    even after the sensor has left the blue tape. Seeing blue again restarts it.
    """

    def __init__(self):
        self.backing_until = -1e9

    def tick(self, bb):
        if on_edge(bb):
            self.backing_until = bb.now + ESCAPE_BACK_S
        if bb.now < self.backing_until:
            # Opponent behind? Curve while backing up so we don't drive straight
            # into them: the closer they are, the sharper the curve (0 = straight
            # back, 1 = spin in place).
            bend = opponent_closeness(bb)
            outer, inner = -ESCAPE_BACK_SPEED, -ESCAPE_BACK_SPEED * (1 - 2 * bend)
            left, right = (inner, outer) if ESCAPE_CURVE_RIGHT else (outer, inner)
            bb.drive(left, right, "escape: back up" + (" + curve" if bend else ""))
            return RUNNING
        if bb.now < self.backing_until + ESCAPE_TURN_S:
            bb.drive(ESCAPE_TURN_SPEED, -ESCAPE_TURN_SPEED, "escape: turn")
            return RUNNING
        return FAILURE  # not escaping -> let the next branch decide


def sticks(bb):
    left = bb.stick_left if abs(bb.stick_left) > DEADZONE else 0
    right = bb.stick_right if abs(bb.stick_right) > DEADZONE else 0
    return left, right


def forward_and_turn(bb):
    """Turn the sticks into (forward, turn); wheels = forward +/- turn, and a
    positive turn means clockwise (to the right).
      both sticks up / both down   -> straight forward / backward
      only one stick               -> spin on the spot toward that stick's side
                                      (left up = spin left, right up = spin right;
                                      pushing it down spins the other way)
      one up, the other down       -> curve FORWARD toward the down stick's side
                                      (left up + right down = forward arc to the right)
    Scaled down to DRIVE_MAX / TURN_MAX so the wheels grip."""
    left, right = sticks(bb)
    if left * right > 0:  # both up or both down
        return (left + right) / 2 * DRIVE_MAX / 100, 0
    if left * right < 0:  # one up, one down -> forward curve toward the down side
        outer = (abs(left) + abs(right)) / 2 * DRIVE_MAX / 100
        inner = outer * CURVE_INNER
        toward_right = right < 0
        return (outer + inner) / 2, (outer - inner) / 2 * (1 if toward_right else -1)
    # only one stick (or none)
    spin = right - left  # right stick up -> +, left stick up -> -
    return 0, spin * TURN_MAX / 100


def flee_forward(bb):
    # Keep the controller's turn, but never go slower than FLEE_SPEED forward.
    forward, turn = forward_and_turn(bb)
    forward = max(FLEE_SPEED, forward)
    bb.drive(forward + turn, forward - turn, "flee: opponent behind")


def follow_controller(bb):
    forward, turn = forward_and_turn(bb)
    moving = forward or turn
    bb.drive(forward + turn, forward - turn, "controller" if moving else "idle")


def build_tree():
    return Selector(
        Sequence(Condition(lego_data_stale), Action(stop)),
        WaitForStart(),
        OutOfRing(),
        Escape(),
        Sequence(Condition(opponent_behind), Action(flee_forward)),
        Action(follow_controller),
    )
