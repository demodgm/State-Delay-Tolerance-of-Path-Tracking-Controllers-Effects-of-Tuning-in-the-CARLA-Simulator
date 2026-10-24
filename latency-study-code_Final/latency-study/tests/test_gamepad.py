"""Gamepad input shaping, against a fake joystick.

The point of pad support is a smoother recorded route, so the shaping
(deadzone, expo, trigger convention) is what matters and is what is tested.
"""

import types

import pytest

from latency.gamepad import (DS4_DEFAULTS, GamepadInput, apply_deadzone,
                             apply_expo, normalise_trigger)


class FakeJoystick:
    def __init__(self, axes=None, buttons=None, name="Wireless Controller"):
        self.axes = list(axes or [0.0] * 6)
        self.buttons = list(buttons or [0] * 13)
        self._name = name

    def init(self):
        pass

    def get_name(self):
        return self._name

    def get_numaxes(self):
        return len(self.axes)

    def get_numbuttons(self):
        return len(self.buttons)

    def get_axis(self, i):
        return self.axes[i]

    def get_button(self, i):
        return self.buttons[i]


FAKE_PYGAME = types.SimpleNamespace(JOYBUTTONDOWN=1536)


def pad(axes=None, buttons=None, **kw):
    js = FakeJoystick(axes, buttons)
    return GamepadInput(FAKE_PYGAME, joystick=js, **kw), js


# --- shaping --------------------------------------------------------------

def test_deadzone_zeroes_small_input():
    assert apply_deadzone(0.05, 0.1) == 0.0
    assert apply_deadzone(-0.05, 0.1) == 0.0


def test_deadzone_rescales_so_full_lock_is_still_reachable():
    """Without rescaling, a 0.1 deadzone caps steering at 0.9 and the car
    can never reach full lock."""
    assert apply_deadzone(1.0, 0.1) == pytest.approx(1.0)
    assert apply_deadzone(-1.0, 0.2) == pytest.approx(-1.0)
    assert apply_deadzone(0.55, 0.1) == pytest.approx(0.5)


def test_expo_gives_finer_control_near_centre():
    assert apply_expo(0.5, 2.0) == pytest.approx(0.25)
    assert apply_expo(1.0, 2.0) == pytest.approx(1.0)
    assert apply_expo(-0.5, 2.0) == pytest.approx(-0.25)


def test_expo_of_one_is_linear():
    for v in (-1.0, -0.3, 0.0, 0.42, 1.0):
        assert apply_expo(v, 1.0) == pytest.approx(v)


def test_trigger_handles_both_driver_conventions():
    assert normalise_trigger(-1.0, rest=-1.0) == pytest.approx(0.0)
    assert normalise_trigger(1.0, rest=-1.0) == pytest.approx(1.0)
    assert normalise_trigger(0.0, rest=-1.0) == pytest.approx(0.5)
    assert normalise_trigger(0.0, rest=0.0) == pytest.approx(0.0)
    assert normalise_trigger(1.0, rest=0.0) == pytest.approx(1.0)


# --- reading --------------------------------------------------------------

def test_resting_pad_commands_nothing():
    g, _ = pad(axes=[0.0, 0.0, -1.0, 0.0, 0.0, -1.0])
    steer, throttle, brake = g.read()
    assert (steer, throttle, brake) == (0.0, 0.0, 0.0)


def test_stick_produces_proportional_steering_not_full_lock():
    """The whole reason for pad support: keyboard steering is a step input."""
    g, js = pad(axes=[0.0, 0.0, -1.0, 0.0, 0.0, -1.0],
                expo=1.0, smoothing=0.0)
    js.axes[0] = 0.5
    steer, _, _ = g.read()
    assert 0.3 < steer < 0.6
    js.axes[0] = 1.0
    steer, _, _ = g.read()
    assert steer == pytest.approx(1.0, abs=1e-6)


def test_triggers_map_to_throttle_and_brake():
    g, js = pad(axes=[0.0, 0.0, -1.0, 0.0, 0.0, -1.0])
    js.axes[DS4_DEFAULTS["axis_throttle"]] = 1.0
    js.axes[DS4_DEFAULTS["axis_brake"]] = 0.0
    _, throttle, brake = g.read()
    assert throttle == pytest.approx(1.0)
    # half-pressed, after the small trigger deadzone is rescaled out
    assert 0.45 < brake < 0.50


def test_smoothing_approaches_the_target_without_overshoot():
    g, js = pad(axes=[0.0] * 6, expo=1.0, smoothing=0.5)
    js.axes[0] = 1.0
    seen = [g.read()[0] for _ in range(25)]
    assert all(0.0 <= s <= 1.0 for s in seen)
    assert seen[0] < seen[5] < seen[-1]
    assert seen[-1] == pytest.approx(1.0, abs=0.01)


def test_custom_mapping_overrides_defaults_and_ignores_none():
    g, js = pad(axes=[0.0] * 6, expo=1.0, smoothing=0.0,
                mapping=dict(axis_steer=3, axis_throttle=None))
    js.axes[3] = 1.0
    assert g.read()[0] == pytest.approx(1.0)
    assert g.map["axis_throttle"] == DS4_DEFAULTS["axis_throttle"]


def test_buttons_map_to_commands():
    g, _ = pad()
    for cmd in ("record", "camera", "quit"):
        e = types.SimpleNamespace(type=FAKE_PYGAME.JOYBUTTONDOWN,
                                  button=DS4_DEFAULTS["btn_" + cmd])
        assert g.command_for(e) == cmd


def test_unmapped_button_and_other_events_are_ignored():
    g, _ = pad()
    assert g.command_for(types.SimpleNamespace(
        type=FAKE_PYGAME.JOYBUTTONDOWN, button=11)) is None
    assert g.command_for(types.SimpleNamespace(type=999, button=0)) is None


# --- robustness -----------------------------------------------------------

def test_no_pad_is_unavailable_rather_than_an_error():
    g = GamepadInput(FAKE_PYGAME, joystick=None)
    assert not g.available
    assert g.read() == (0.0, 0.0, 0.0)
    assert g.button("btn_record") is False
    assert "no gamepad" in g.describe()


def test_a_pad_unplugged_mid_drive_does_not_crash_the_session():
    class Flaky(FakeJoystick):
        def get_axis(self, i):
            raise OSError("device disconnected")

    g = GamepadInput(FAKE_PYGAME, joystick=Flaky())
    assert g.read() == (0.0, 0.0, 0.0)
    # and it must stay off rather than flickering back to phantom input
    assert not g.available
    assert g.read() == (0.0, 0.0, 0.0)


def test_describe_reports_the_pad_name_and_shape():
    g, _ = pad()
    assert "Wireless Controller" in g.describe()
    assert "6 axes" in g.describe()


def test_a_failed_read_never_reports_phantom_throttle():
    """A trigger resting at -1.0 reads 0.0 as half pressed, so a failed read
    that substitutes 0.0 would make a disconnected pad accelerate the car."""
    class DiesLater(FakeJoystick):
        def __init__(self):
            super().__init__(axes=[0.0, 0.0, -1.0, 0.0, 0.0, -1.0])
            self.alive = True

        def get_axis(self, i):
            if not self.alive:
                raise OSError("device disconnected")
            return self.axes[i]

    js = DiesLater()
    g = GamepadInput(FAKE_PYGAME, joystick=js)
    assert g.read() == (0.0, 0.0, 0.0)
    js.alive = False
    steer, throttle, brake = g.read()
    assert throttle == 0.0 and brake == 0.0 and steer == 0.0


def test_an_out_of_range_axis_index_is_rejected_at_setup():
    g = GamepadInput(FAKE_PYGAME, joystick=FakeJoystick(axes=[0.0, 0.0]),
                     mapping=dict(axis_throttle=5))
    assert not g.available
