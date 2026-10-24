"""Analog gamepad input for trajectory recording.

Keyboard steering is a step input: full lock the instant a key goes down.
Every step change lands in the recorded route, and the route is the target
all four controllers are scored against, so reference jitter shows up as
tracking error in every one of them. An analog stick gives continuous
steering, which is the point of supporting one.

Written against a DualShock 4 on Linux (the kernel's hid-sony driver), but
nothing here is Sony-specific: axis and button indices are all configurable,
because mappings differ between drivers, SDL versions, and USB vs Bluetooth.
Run `record_trajectory.py --gamepad-test` to see live indices for your pad.
"""

import math

# DualShock 4, hid-sony driver, as reported by SDL2/pygame.
# Triggers rest at -1.0 and travel to +1.0.
DS4_DEFAULTS = dict(
    axis_steer=0,       # left stick X
    axis_throttle=5,    # R2
    axis_brake=2,       # L2
    btn_record=0,       # Cross
    btn_camera=3,       # Triangle
    btn_quit=1,         # Circle
    btn_handbrake=5,    # R1
)


def apply_deadzone(value, deadzone):
    """Zero the stick near centre, then rescale so full travel is still 1.0.

    Without the rescale, a 0.1 deadzone caps the usable range at 0.9 and the
    car can't reach full lock.
    """
    if deadzone <= 0.0:
        return value
    if abs(value) <= deadzone:
        return 0.0
    scaled = (abs(value) - deadzone) / (1.0 - deadzone)
    return math.copysign(min(scaled, 1.0), value)


def apply_expo(value, expo):
    """Curve the stick so small movements near centre are finer.

    expo = 1.0 is linear. Higher values give more resolution around straight
    ahead, which is where a recorded route spends most of its time and where
    accidental input does the most damage to the reference.
    """
    if expo <= 1.0:
        return value
    return math.copysign(abs(value) ** expo, value)


def normalise_trigger(raw, rest):
    """Map a trigger's raw reading to 0..1.

    Some drivers rest at -1 and travel to +1; others rest at 0. The resting
    value is sampled at startup rather than assumed.
    """
    if rest < -0.5:                      # rests at -1, full travel is 2.0
        return max(0.0, min(1.0, (raw + 1.0) * 0.5))
    return max(0.0, min(1.0, raw))


class GamepadInput:
    """Reads steering, throttle and brake from a joystick.

    Degrades to unavailable rather than raising if no pad is attached, or if
    one is unplugged mid-drive, so a recording session never dies because a
    cable moved.
    """

    def __init__(self, pygame, index=0, deadzone=0.08, expo=1.6,
                 smoothing=0.25, mapping=None, joystick=None):
        self.pygame = pygame
        self.deadzone = deadzone
        self.expo = expo
        self.smoothing = smoothing
        self.map = dict(DS4_DEFAULTS)
        if mapping:
            self.map.update({k: v for k, v in mapping.items() if v is not None})
        self._steer = 0.0
        self._failed = False
        self.name = None
        self.joystick = joystick
        self._trigger_rest = {}

        if self.joystick is None:
            try:
                pygame.joystick.init()
                if pygame.joystick.get_count() > index:
                    self.joystick = pygame.joystick.Joystick(index)
                    self.joystick.init()
            except Exception:
                self.joystick = None

        if self.joystick is not None:
            try:
                self.name = self.joystick.get_name()
                self._sample_trigger_rest()
            except ValueError as exc:
                print("gamepad: {}".format(exc))
                self.joystick = None
            except Exception:
                self.joystick = None

    # -- setup -------------------------------------------------------------

    def _sample_trigger_rest(self):
        """Record where the triggers sit untouched, so either convention works.

        An index that doesn't exist on this device is a mapping error, not
        something to paper over with a default - it would silently produce
        phantom throttle.
        """
        n = self.joystick.get_numaxes()
        for key in ("axis_throttle", "axis_brake"):
            idx = self.map[key]
            if idx >= n:
                raise ValueError(
                    "{} is axis {} but this pad has only {} axes. Run "
                    "--gamepad-test and pass the right index."
                    .format(key, idx, n))
            self._trigger_rest[key] = self.joystick.get_axis(idx)

    @property
    def available(self):
        return self.joystick is not None and not self._failed

    def describe(self):
        if not self.available:
            return "no gamepad detected - using keyboard"
        try:
            n_ax, n_btn = self.joystick.get_numaxes(), self.joystick.get_numbuttons()
        except Exception:
            n_ax = n_btn = 0
        return "gamepad: {} ({} axes, {} buttons)".format(self.name, n_ax, n_btn)

    # -- per-frame ---------------------------------------------------------

    def _axis(self, idx):
        """Return the axis value, or None if it can't be read.

        None matters: a trigger that rests at -1.0 reads 0.0 as HALF PRESSED,
        so substituting 0.0 for a failed read makes a disconnected pad - or a
        wrong axis index - command half throttle. A failed read disables the
        pad instead, handing control back to the keyboard.
        """
        try:
            return self.joystick.get_axis(idx)
        except Exception:
            self._failed = True
            return None

    def read(self):
        """Return (steer, throttle, brake), each already shaped and clamped.

        steer is -1..1 (negative = left), throttle and brake are 0..1.
        """
        if not self.available:
            return 0.0, 0.0, 0.0

        raw = self._axis(self.map["axis_steer"])
        thr_raw = self._axis(self.map["axis_throttle"])
        brk_raw = self._axis(self.map["axis_brake"])
        if raw is None or thr_raw is None or brk_raw is None:
            # pad gone, or an index that doesn't exist on this device
            self._failed = True
            self._steer = 0.0
            return 0.0, 0.0, 0.0

        target = apply_expo(apply_deadzone(raw, self.deadzone), self.expo)
        # light low-pass: removes stick tremor without adding noticeable lag
        a = max(0.0, min(1.0, 1.0 - self.smoothing))
        self._steer += (target - self._steer) * a
        if abs(self._steer) < 1e-4:
            self._steer = 0.0

        throttle = normalise_trigger(
            thr_raw, self._trigger_rest.get("axis_throttle", -1.0))
        brake = normalise_trigger(
            brk_raw, self._trigger_rest.get("axis_brake", -1.0))
        return (max(-1.0, min(1.0, self._steer)),
                apply_deadzone(throttle, 0.04),
                apply_deadzone(brake, 0.04))

    def button(self, name):
        if not self.available:
            return False
        try:
            return bool(self.joystick.get_button(self.map[name]))
        except Exception:
            return False

    def command_for(self, event):
        """Map a JOYBUTTONDOWN event to a command name, or None."""
        if getattr(event, "type", None) != self.pygame.JOYBUTTONDOWN:
            return None
        for cmd in ("record", "camera", "quit"):
            if event.button == self.map["btn_" + cmd]:
                return cmd
        return None
