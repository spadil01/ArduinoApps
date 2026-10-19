# Run on the laptop:  python3 -m pytest sumo-robot/python/test_behavior_tree.py
import behavior_tree as bt


def make(color=7, left=0, right=0, opponent=False, fresh=True, now=10.0):
    bb = bt.Blackboard()
    bb.now = now
    bb.color, bb.stick_left, bb.stick_right = color, left, right
    bb.lego_time = now if fresh else now - 5
    bb.awake = True
    if opponent:
        bb.opponent_time = now
    return bb


def tick(bb, tree=None):
    (tree or bt.build_tree()).tick(bb)
    return bb.left, bb.right, bb.behavior


def test_centered_sticks_sit_still():
    assert tick(make(left=2, right=-3))[:2] == (0, 0)


def test_follows_controller_at_reduced_speed():
    assert tick(make(left=100, right=100)) == (bt.DRIVE_MAX, bt.DRIVE_MAX, "controller")
    assert tick(make(left=0, right=100)) == (bt.TURN_MAX, -bt.TURN_MAX, "controller")


def test_no_lego_data_stops():
    assert tick(make(left=100, right=100, fresh=False))[:2] == (0, 0)


def test_blue_overrides_controller_and_opponent():
    left, right, behavior = tick(make(color=3, left=100, right=100, opponent=True))
    assert behavior.startswith("escape")
    assert min(left, right) == -bt.ESCAPE_BACK_SPEED  # not fleeing forward


def test_escape_finishes_after_blue_is_gone():
    tree = bt.build_tree()
    tick(make(color=3, now=10.0), tree)
    # still backing up just after leaving the tape
    assert tick(make(color=7, now=10.0 + bt.ESCAPE_BACK_S / 2), tree)[2] == "escape: back up"
    # then turning
    assert tick(make(color=7, now=10.0 + bt.ESCAPE_BACK_S + 0.01), tree)[2] == "escape: turn"
    # then back to the controller
    done = 10.0 + bt.ESCAPE_BACK_S + bt.ESCAPE_TURN_S + 0.01
    assert tick(make(color=7, left=30, right=30, now=done), tree)[2] == "controller"


def test_opponent_behind_drives_forward():
    assert tick(make(opponent=True))[:2] == (bt.FLEE_SPEED, bt.FLEE_SPEED)


def test_opponent_behind_ignores_backwards():
    left, right, _ = tick(make(left=-100, right=-100, opponent=True))
    assert left > 0 and right > 0


def test_opponent_behind_keeps_controller_turn():
    left, right, _ = tick(make(left=40, right=-40, opponent=True))  # turn right
    assert left > right
    left, right, _ = tick(make(left=-40, right=40, opponent=True))  # turn left
    assert right > left


def test_blue_with_nobody_behind_backs_straight():
    assert tick(make(color=3))[:2] == (-bt.ESCAPE_BACK_SPEED, -bt.ESCAPE_BACK_SPEED)


def test_blue_with_opponent_behind_curves_more_when_closer():
    def escape_with_opponent_at(cm):
        bb = make(color=3, opponent=True)
        bb.distance_cm = cm
        left, right, behavior = tick(bb)
        assert behavior == "escape: back up + curve"
        return abs(left - right)  # bigger gap between wheels = sharper curve

    assert escape_with_opponent_at(5) > escape_with_opponent_at(15) > 0


def test_lost_light_sensor_stops():
    assert tick(make(color=None, left=100, right=100)) == (0, 0, "stop: no light sensor")


def surface(color, hue, sat):
    bb = make(color=color, left=40, right=40)
    bb.hue, bb.sat = hue, sat
    return tick(bb)[2]


def test_measured_surfaces():
    assert surface(2, 16, 32) == "controller"            # floor
    assert surface(0, 188, 12).startswith("escape")      # light blue tape
    assert surface(3, 207, 52).startswith("escape")      # dark blue tape


def test_asleep_until_wake_signal():
    tree = bt.build_tree()
    bb = make(left=0, right=0, opponent=True)  # opponent behind, but asleep
    bb.awake = False

    def at(t, left, right):
        bb.now = bb.lego_time = t
        bb.opponent_time = t
        bb.stick_left, bb.stick_right = left, right
        return tick(bb, tree)

    assert at(10.0, 0, 0)[:2] == (0, 0)                  # asleep, ignores opponent
    assert at(10.1, -100, -100)[:2] == (0, 0)            # holding down: still still
    assert at(11.2, -100, -100)[2] == "asleep: let go to start"
    assert at(11.25, -50, -40)[:2] == (0, 0)             # letting go (in between)
    assert at(11.3, 0, 0)[2] == "awake!"
    assert at(11.35, 0, 0)[2] == "flee: opponent behind"  # now the rules apply


def test_letting_go_too_early_does_not_wake():
    tree = bt.build_tree()
    bb = make()
    bb.awake = False
    for t, l, r in ((10.0, -100, -100), (10.5, -100, -100), (10.6, 0, 0), (12.0, 0, 0)):
        bb.now = bb.lego_time = t
        bb.stick_left, bb.stick_right = l, r
        tick(bb, tree)
    assert not bb.awake


def test_one_close_reading_is_ignored():
    bb = make()
    bt.note_distance(bb, 12, 10.0)
    bt.note_distance(bb, -1, 10.07)
    bt.note_distance(bb, 12, 10.14)
    assert tick(bb)[2] == "idle"


def test_three_close_readings_in_a_row_count():
    bb = make()
    for i in range(3):
        bt.note_distance(bb, 12, 10.0 - 0.07 * (2 - i))
    assert tick(bb)[2] == "flee: opponent behind"


def test_laptop_gone_a_while_goes_back_to_sleep():
    bb = make(fresh=False)  # last LEGO data 5 s ago
    tick(bb)
    assert not bb.awake


def test_one_stick_spins_in_place():
    left, right, _ = tick(make(left=100, right=0))   # left stick up -> spin left
    assert (left, right) == (-bt.TURN_MAX, bt.TURN_MAX)
    left, right, _ = tick(make(left=0, right=100))   # right stick up -> spin right
    assert (left, right) == (bt.TURN_MAX, -bt.TURN_MAX)
    left, right, _ = tick(make(left=-100, right=0))  # left stick down alone -> no driving
    assert (left, right) == (0, 0)
    left, right, _ = tick(make(left=0, right=-100))  # right stick down alone -> no driving
    assert (left, right) == (0, 0)
    left, right, _ = tick(make(left=50, right=0))    # half push -> half speed spin
    assert left == -right and 0 < right < bt.TURN_MAX


def test_both_sticks_same_way_drive_straight():
    left, right, _ = tick(make(left=100, right=60))
    assert left == right > 0
    left, right, _ = tick(make(left=-100, right=-100))
    assert left == right == -bt.DRIVE_MAX


def run(tree, bb, steps):
    """steps: list of (seconds, color, sticks). Returns every behavior seen."""
    seen = []
    for t, color, stick in steps:
        bb.now = bb.lego_time = t
        bb.color, bb.stick_left, bb.stick_right = color, stick, stick
        tree.tick(bb)
        seen.append(bb.behavior)
    return seen


def test_backing_out_of_the_ring_is_lost():
    tree, bb = bt.build_tree(), make()
    seen = run(tree, bb, [(10.0, 2, -100), (10.1, 2, -100), (10.2, 3, -100)])
    assert seen[-1] == "lost: out of the ring" and not bb.awake


def test_normal_escape_is_not_lost():
    tree, bb = bt.build_tree(), make()
    steps = [(10.0, 2, 100), (10.05, 3, 100)]                       # hit tape going forward
    steps += [(10.05 + 0.05 * i, 2, 0) for i in range(1, 30)]       # escape backs up + spins
    seen = run(tree, bb, steps)
    assert not any(b.startswith("lost") for b in seen) and bb.awake


def test_overshoot_then_backing_over_tape_is_not_lost():
    tree, bb = bt.build_tree(), make()
    steps = [(10.0, 2, 100), (10.05, 3, 100), (10.1, 2, 0),         # overshoot past the tape
             (10.3, 3, 0), (10.4, 2, 0)]                            # escape backs over it again
    seen = run(tree, bb, steps)
    assert not any(b.startswith("lost") for b in seen) and bb.awake


def test_drifting_onto_tape_then_flicker_while_escaping_is_not_lost():
    # What happened on the real robot: sitting still, rolled onto the tape,
    # then the sensor flickered blue/not-blue at the edge while backing off.
    tree, bb = bt.build_tree(), make()
    steps = [(10.0, 2, 0), (10.05, 3, 0), (10.1, 3, 0),             # idle, reaches tape
             (10.15, 2, 0), (10.2, 3, 0), (10.25, 2, 0), (10.3, 3, 0),  # flicker while backing
             (10.35, 2, 0)]
    steps += [(10.35 + 0.05 * i, 2, 0) for i in range(1, 30)]
    seen = run(tree, bb, steps)
    assert not any(b.startswith("lost") for b in seen) and bb.awake


def test_up_and_down_curves_forward_toward_the_down_side():
    left, right, _ = tick(make(left=100, right=-100))  # right stick down -> curve right
    assert left > right > 0
    left, right, _ = tick(make(left=-100, right=100))  # left stick down -> curve left
    assert right > left > 0


def ramp(left, right, **kw):
    bb = make(left=left, right=right, **kw)
    tick(bb)
    return bb.ramp_up


def test_ramp_up_only_while_left_stick_alone_is_down():
    assert ramp(-100, 0)                 # left down alone -> up
    assert not ramp(0, 0)                # centered -> down
    assert not ramp(-100, -100)          # both down = reverse, ramp stays down
    assert not ramp(-100, 100)           # left down + right up = curve, ramp stays down
    assert not ramp(0, -100)             # right down alone -> down


def test_ramp_stays_down_when_asleep_or_no_data():
    bb = make(left=-100, right=0)
    bb.awake = False
    tick(bb)
    assert not bb.ramp_up
    assert not ramp(-100, 0, fresh=False)
