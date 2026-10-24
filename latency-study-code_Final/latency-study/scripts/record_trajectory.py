"""Record the reference trajectory by driving CARLA manually.

Controls (keyboard)
  arrow keys / WASD   drive (steering and throttle ramp up while held)
  space               handbrake
  R                   start / stop recording  (saves on every stop)
  C                   cycle camera: chase, bonnet, top-down
  Q or Esc            quit

Controls (gamepad, used automatically when one is attached)
  left stick          steering, analog
  R2 / L2             throttle / brake
  R1                  handbrake
  Cross               start / stop recording
  Triangle            cycle camera
  Circle              quit
  The keyboard keys keep working at the same time, so a wrong button mapping
  never traps you. Run with --gamepad-test to see live axis/button indices.

Writes 'x,y,v' per simulation step, the same format the course framework used.
Include both sustained straights and tight turns: a route of straights alone
reports a flatteringly large budget for every controller. Drive smoothly -
the recorded route is the target every controller is scored against, so
steering jitter in the reference shows up as error in all four of them.

Saving is deliberately paranoid, because a lost drive means driving again:
  - a relative --out resolves against the repo root, not the current directory
  - the parent directory is created, and writability is checked BEFORE driving
  - the file is written every time you STOP recording, not only on quit
  - if the chosen path can't be written, the points go to a fallback in your
    home directory rather than being lost
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from latency.gamepad import GamepadInput  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# (x, y, z, pitch) relative to the vehicle
CAMERA_VIEWS = [
    ("chase",     -5.5, 0.0, 2.8, -15.0),
    ("bonnet",     1.6, 0.0, 1.3,   0.0),
    ("top-down",   0.0, 0.0, 18.0, -90.0),
]


def resolve_out(path):
    """Relative paths are relative to the repo root, not the shell's cwd.

    Running this from scripts/ used to send output to scripts/data/, which
    doesn't exist, so the write failed after the drive was already done.
    """
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def route_length(rows):
    return sum(math.hypot(rows[i + 1][0] - rows[i][0], rows[i + 1][1] - rows[i][1])
               for i in range(len(rows) - 1))


def save_rows(rows, path, quiet=False):
    """Write x,y,v per line. Falls back to the home directory on any failure.

    Returns the path actually written, or None if nothing was written.
    """
    if not rows:
        if not quiet:
            print("nothing to save (no points recorded)", file=sys.stderr)
        return None

    def _write(target):
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target, "w") as fh:
            for x, y, v in rows:
                fh.write("{:.6f},{:.6f},{:.3f}\n".format(x, y, v))
        return target

    try:
        written = _write(path)
    except OSError as exc:
        fallback = os.path.join(
            os.path.expanduser("~"),
            "reference_trajectory_recovered_{}.txt".format(int(time.time())))
        print("couldn't write {} ({})".format(path, exc), file=sys.stderr)
        try:
            written = _write(fallback)
            print("points saved to {} instead - move it where you want it"
                  .format(written), file=sys.stderr)
        except OSError as exc2:
            print("fallback write also failed ({}); {} points LOST"
                  .format(exc2, len(rows)), file=sys.stderr)
            return None

    if not quiet:
        print("wrote {}: {} points, {:.0f} m".format(
            written, len(rows), route_length(rows)))
    return written


def preflight(out):
    """Prove the destination is writable before a single metre is driven."""
    parent = os.path.dirname(out) or "."
    try:
        os.makedirs(parent, exist_ok=True)
        probe = out + ".writetest"
        open(probe, "w").close()
        os.remove(probe)
    except OSError as exc:
        raise SystemExit("can't write to {}: {}".format(out, exc))
    return out


def to_surface(image, pygame):
    """CARLA BGRA buffer -> pygame surface."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
    return pygame.surfarray.make_surface(arr.swapaxes(0, 1))


class SteerSmoother:
    """Progressive steering and throttle, as CARLA's own manual_control does.

    Binary full-lock steering produces a reference full of step changes, and
    every controller is then scored against that jitter. Ramping makes a
    route a human can actually drive smoothly.
    """

    def __init__(self, dt):
        self.dt = dt
        self.steer = 0.0
        self.throttle = 0.0

    def update(self, left, right, up, down):
        rate = 1.6 * self.dt
        if left and not right:
            self.steer = max(-0.7, self.steer - rate)
        elif right and not left:
            self.steer = min(0.7, self.steer + rate)
        else:                                   # centre when released
            if abs(self.steer) <= rate:
                self.steer = 0.0
            else:
                self.steer -= math.copysign(rate, self.steer)

        if up:
            self.throttle = min(0.75, self.throttle + 1.2 * self.dt)
        else:
            self.throttle = max(0.0, self.throttle - 2.5 * self.dt)
        return self.steer, self.throttle, (1.0 if down else 0.0)


def gamepad_test():
    """Show live axis and button indices so a pad can be remapped.

    Mappings differ between drivers, SDL versions, and USB vs Bluetooth, so
    guessing isn't reliable. Needs no CARLA server.
    """
    try:
        import pygame
    except ImportError:
        raise SystemExit("pygame isn't installed.  pip install pygame")
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No gamepad detected.")
        print("  USB: plug it in; the kernel's hid-sony driver handles a DS4.")
        print("  Bluetooth: pair it first (hold PS + Share until it flashes).")
        print("  Check with:  ls /dev/input/js*")
        return
    js = pygame.joystick.Joystick(0)
    js.init()
    print("{}: {} axes, {} buttons, {} hats".format(
        js.get_name(), js.get_numaxes(), js.get_numbuttons(), js.get_numhats()))
    print("Move the sticks and press buttons. Ctrl-C to stop.")
    print("Pass the indices you see as --axis-steer / --btn-record etc.\n")
    screen = pygame.display.set_mode((640, 200))
    pygame.display.set_caption("gamepad test - Ctrl-C in the terminal to stop")
    font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()
    try:
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if e.type == pygame.JOYBUTTONDOWN:
                    print("button {} pressed".format(e.button))
            axes = ["{}:{:+.2f}".format(i, js.get_axis(i))
                    for i in range(js.get_numaxes())]
            held = [str(i) for i in range(js.get_numbuttons())
                    if js.get_button(i)]
            screen.fill((18, 18, 18))
            screen.blit(font.render("axes   " + "  ".join(axes[:6]), True,
                                    (220, 220, 220)), (14, 24))
            if len(axes) > 6:
                screen.blit(font.render("       " + "  ".join(axes[6:]), True,
                                        (220, 220, 220)), (14, 48))
            screen.blit(font.render("held   " + (" ".join(held) or "-"), True,
                                    (150, 220, 150)), (14, 84))
            screen.blit(font.render(
                "left stick X is usually axis 0; R2/L2 rest at -1.00",
                True, (140, 140, 140)), (14, 130))
            pygame.display.flip()
            clock.tick(30)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/reference_trajectory.txt",
                    help="relative paths resolve against the repo root")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town03")
    ap.add_argument("--dt", type=float, default=0.02,
                    help="simulation rate while recording; match "
                         "configs/sweep.yaml params.dt for a denser reference")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--gamepad-test", action="store_true",
                    help="print live axis/button indices and exit; needs no "
                         "CARLA server")
    ap.add_argument("--no-gamepad", action="store_true",
                    help="ignore any attached controller")
    ap.add_argument("--pad-deadzone", type=float, default=0.08)
    ap.add_argument("--pad-expo", type=float, default=1.6,
                    help="1.0 is linear; higher gives finer control near centre")
    ap.add_argument("--pad-smoothing", type=float, default=0.25)
    for name, default in (("axis-steer", None), ("axis-throttle", None),
                          ("axis-brake", None), ("btn-record", None),
                          ("btn-camera", None), ("btn-quit", None),
                          ("btn-handbrake", None)):
        ap.add_argument("--" + name, type=int, default=default,
                        help="override the default mapping for this control")
    ap.add_argument("--no-camera", action="store_true",
                    help="status window only; much lighter, but you're "
                         "driving blind unless you watch the server window")
    a = ap.parse_args()

    if a.gamepad_test:
        return gamepad_test()

    out = preflight(resolve_out(a.out))
    print("will save to {}".format(out))

    try:
        import carla
    except ImportError:
        raise SystemExit(
            "The 'carla' package isn't installed.\n"
            "  Install the wheel shipped with your server, e.g.:\n"
            "      pip install <CARLA>/PythonAPI/carla/dist/carla-0.9.15-cp37*.whl")
    try:
        import pygame
    except ImportError:
        raise SystemExit("pygame isn't installed.  pip install pygame")

    client = carla.Client(a.host, a.port)
    client.set_timeout(20.0)
    try:
        world = client.load_world(a.town)
    except RuntimeError as exc:
        raise SystemExit(
            "Couldn't reach a CARLA server at {}:{} ({}).\n"
            "  Start one with:  ./CarlaUE4.sh".format(a.host, a.port, exc))

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = a.dt
    # a camera can't produce images if the server isn't rendering
    settings.no_rendering_mode = bool(a.no_camera)
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    car = world.spawn_actor(bp_lib.find("vehicle.tesla.model3"),
                            world.get_map().get_spawn_points()[0])

    pygame.init()
    size = (460, 150) if a.no_camera else (a.width, a.height)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("R record/stop   C camera   Q quit")
    font = pygame.font.Font(None, 28)
    small = pygame.font.Font(None, 22)

    camera = None
    frame = {"surface": None}
    view_idx = 0

    def attach_camera(idx):
        nonlocal camera
        if a.no_camera:
            return
        if camera is not None:
            camera.stop()
            camera.destroy()
        name, x, y, z, pitch = CAMERA_VIEWS[idx]
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(a.width))
        cam_bp.set_attribute("image_size_y", str(a.height))
        cam_bp.set_attribute("fov", "90")
        tf = carla.Transform(carla.Location(x=x, y=y, z=z),
                             carla.Rotation(pitch=pitch))
        camera = world.spawn_actor(cam_bp, tf, attach_to=car)
        camera.listen(lambda img: frame.__setitem__("surface",
                                                    to_surface(img, pygame)))
        print("camera: {}".format(name))

    attach_camera(view_idx)
    spectator = world.get_spectator()
    pad = None
    if not a.no_gamepad:
        pad = GamepadInput(
            pygame, deadzone=a.pad_deadzone, expo=a.pad_expo,
            smoothing=a.pad_smoothing,
            mapping=dict(axis_steer=a.axis_steer, axis_throttle=a.axis_throttle,
                         axis_brake=a.axis_brake, btn_record=a.btn_record,
                         btn_camera=a.btn_camera, btn_quit=a.btn_quit,
                         btn_handbrake=a.btn_handbrake))
        print(pad.describe())
        if not pad.available:
            print("  (--gamepad-test lists indices if one is attached but "
                  "not picked up)")

    smoother = SteerSmoother(a.dt)
    rows, recording, saved_to = [], False, None

    try:
        while True:
            world.tick()

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    raise KeyboardInterrupt
                cmd = pad.command_for(e) if pad is not None else None
                if cmd == "record":
                    recording = not recording
                    if recording:
                        print("recording")
                    else:
                        saved_to = save_rows(rows, out)
                elif cmd == "camera":
                    view_idx = (view_idx + 1) % len(CAMERA_VIEWS)
                    attach_camera(view_idx)
                elif cmd == "quit":
                    raise KeyboardInterrupt
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_r:
                        recording = not recording
                        if recording:
                            print("recording")
                        else:
                            # save the moment recording stops, so the drive is
                            # safe before anything else can go wrong
                            saved_to = save_rows(rows, out)
                    elif e.key == pygame.K_c:
                        view_idx = (view_idx + 1) % len(CAMERA_VIEWS)
                        attach_camera(view_idx)
                    elif e.key in (pygame.K_q, pygame.K_ESCAPE):
                        raise KeyboardInterrupt

            k = pygame.key.get_pressed()
            steer, throttle, brake = smoother.update(
                k[pygame.K_LEFT] or k[pygame.K_a],
                k[pygame.K_RIGHT] or k[pygame.K_d],
                k[pygame.K_UP] or k[pygame.K_w],
                k[pygame.K_DOWN] or k[pygame.K_s])
            hand_brake = bool(k[pygame.K_SPACE])
            using_pad = False
            if pad is not None and pad.available:
                p_steer, p_throttle, p_brake = pad.read()
                # the pad takes over only while it's actually being moved, so
                # the keyboard stays usable without unplugging anything
                if abs(p_steer) > 0.0 or p_throttle > 0.0 or p_brake > 0.0:
                    using_pad = True
                    steer, throttle, brake = p_steer, p_throttle, p_brake
                    smoother.steer, smoother.throttle = p_steer, p_throttle
                hand_brake = hand_brake or pad.button("btn_handbrake")
            car.apply_control(carla.VehicleControl(
                throttle=throttle, brake=brake, steer=steer,
                hand_brake=hand_brake))

            tr, vel = car.get_transform(), car.get_velocity()
            v = math.hypot(vel.x, vel.y)
            if recording:
                rows.append((tr.location.x, tr.location.y, v))

            # keep the server window following the car too
            spectator.set_transform(carla.Transform(
                tr.location + carla.Location(z=30),
                carla.Rotation(pitch=-90, yaw=tr.rotation.yaw)))

            # ---- draw ----
            if frame["surface"] is not None:
                screen.blit(frame["surface"], (0, 0))
            else:
                screen.fill((18, 18, 18))
                if not a.no_camera:
                    screen.blit(small.render("waiting for camera...", True,
                                             (160, 160, 160)), (16, 16))

            hud = pygame.Surface((440, 112))
            hud.set_alpha(190)
            hud.fill((12, 12, 12))
            screen.blit(hud, (10, 10))
            screen.blit(font.render(
                "{}   {} pts   {:5.1f} km/h   {:.0f} m".format(
                    "REC" if recording else "idle", len(rows), v * 3.6,
                    route_length(rows) if len(rows) > 1 else 0.0),
                True, (255, 95, 95) if recording else (215, 215, 215)), (22, 20))
            screen.blit(small.render(
                "R record/stop (saves on stop)   C camera   Q quit",
                True, (150, 150, 150)), (22, 54))
            screen.blit(small.render(
                "saved: {}".format(os.path.basename(saved_to)) if saved_to
                else "not saved yet", True,
                (125, 205, 125) if saved_to else (180, 130, 130)), (22, 82))
            pygame.display.flip()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if camera is not None:
                camera.stop()
                camera.destroy()
            car.destroy()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            world.apply_settings(settings)
        except Exception:
            pass
        pygame.quit()
        # save again on exit, in case points were added after the last stop
        save_rows(rows, out)


if __name__ == "__main__":
    main()
