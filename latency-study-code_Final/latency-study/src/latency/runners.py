"""Two ways to execute one run.

`CarlaRunner` is the real one and produces the data the paper reports.

`KinematicRunner` is a bicycle-model stand-in whose only purpose is to let
the sweep, the tuner and the analysis be debugged without a CARLA server
attached. It's NOT a physics simulation of a car and its output must never
appear in the paper. Every file it writes is stamped SELFTEST and
analyze.py refuses to build paper tables from stamped data.
"""

import math
import numpy as np

from .delay import DelayBuffer
from .controllers import build

SELFTEST_STAMP = "SELFTEST_NOT_REAL_DATA"


def _progress(log, ref_xy):
    """Fraction of the reference route the car actually reached.

    A run that fails at 2% and a run that fails at 95% need different fixes,
    and 'didn't complete' doesn't distinguish them.
    """
    if not log["x"]:
        return 0.0
    x, y = log["x"][-1], log["y"][-1]
    idx = int(np.argmin(np.hypot(ref_xy[:, 0] - x, ref_xy[:, 1] - y)))
    return float(idx) / max(1, len(ref_xy) - 1)


# --------------------------------------------------------------------------


class CarlaRunner:
    """One run in CARLA: synchronous, fixed timestep, ground-truth state."""

    def __init__(self, params, host="127.0.0.1", port=2000, render=False):
        try:
            import carla  # lazy, so the repo stays importable without it
        except ImportError as exc:
            raise SystemExit(
                "The 'carla' Python package isn't installed.\n"
                "  Install the client matching your server, e.g.:\n"
                "      pip install carla==0.9.15\n"
                "  The version must match the running CARLA server.\n"
                "  To exercise the pipeline with no server attached, add "
                "--selftest\n"
                "  (kinematic stand-in - debugging only, never results)."
            ) from exc

        self.carla = carla
        self.p = params
        self.dt = params["dt"]
        self.client = carla.Client(host, port)
        self.client.set_timeout(20.0)
        try:
            self.world = self.client.load_world(params["town"])
        except RuntimeError as exc:
            raise SystemExit(
                f"Could not reach a CARLA server at {host}:{port} ({exc}).\n"
                "  Start one with:  ./CarlaUE4.sh\n"
                "  then re-run. Add --selftest to run without a server."
            ) from exc
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.dt
        settings.no_rendering_mode = not render
        self.world.apply_settings(settings)
        self.bp = self.world.get_blueprint_library().find(params["vehicle_bp"])
        self._actor = None
        # a previous run that died mid-sweep can leave its car parked on the
        # start point, which then blocks every subsequent spawn
        for leftover in self.world.get_actors().filter("vehicle.*"):
            leftover.destroy()
        self.world.tick()

    def validate_reference(self, ref_xy, sample=40, tol_m=6.0):
        """Fail fast if the route doesn't lie on this town's roads.

        A route recorded in one town and run in another produces a spawn
        failure or a car that drives into scenery, several minutes in.
        """
        carla = self.carla
        idx = np.linspace(0, len(ref_xy) - 1, min(sample, len(ref_xy)))
        far = 0
        for i in idx.astype(int):
            loc = carla.Location(x=float(ref_xy[i, 0]), y=float(ref_xy[i, 1]),
                                 z=0.0)
            wp = self.world.get_map().get_waypoint(loc, project_to_road=True)
            if wp is None:
                far += 1
            else:
                d = math.hypot(wp.transform.location.x - loc.x,
                               wp.transform.location.y - loc.y)
                if d > tol_m:
                    far += 1
        if far > len(idx) * 0.25:
            raise SystemExit(
                "The reference route doesn't match the loaded town.\n"
                "  {}/{} sampled points are more than {:.0f} m from any road "
                "in {}.\n"
                "  Re-record the route in {}, or set params.town to the town "
                "it was recorded in.".format(far, len(idx), tol_m,
                                             self.p["town"], self.p["town"]))

    def _start_transform(self, ref_xy, ref_yaw0):
        """Place the car on the road surface at the route's first point.

        The z of the recorded route isn't stored, and Town03 has elevated
        highway and an underpass, so a fixed z puts the car inside the
        geometry on any raised section and the spawn silently fails. Ask the
        map for the road height at that x,y instead.
        """
        carla = self.carla
        loc = carla.Location(x=float(ref_xy[0, 0]), y=float(ref_xy[0, 1]), z=0.0)
        wp = self.world.get_map().get_waypoint(
            loc, project_to_road=True,
            lane_type=carla.LaneType.Driving | carla.LaneType.Shoulder)
        z = (wp.transform.location.z if wp is not None else 0.0) + 0.3
        return carla.Transform(
            carla.Location(x=loc.x, y=loc.y, z=z),
            carla.Rotation(yaw=math.degrees(ref_yaw0)),
        ), wp

    def _spawn(self, ref_xy, ref_yaw0):
        carla = self.carla
        base, wp = self._start_transform(ref_xy, ref_yaw0)
        # try progressively higher; a few cm of clearance is often all it needs
        for dz in (0.0, 0.3, 0.7, 1.2, 2.0):
            tf = carla.Transform(
                carla.Location(x=base.location.x, y=base.location.y,
                               z=base.location.z + dz),
                base.rotation)
            actor = self.world.try_spawn_actor(self.bp, tf)
            if actor is not None:
                return actor

        blockers = [a.type_id for a in self.world.get_actors()
                    if a.get_location().distance(base.location) < 6.0]
        wp_note = ("no drivable waypoint within reach of that point - is the "
                   "route from a different town?"
                   if wp is None else
                   "nearest road surface is z={:.2f}".format(
                       wp.transform.location.z))
        raise RuntimeError(
            "Couldn't spawn the vehicle at the start of the reference route.\n"
            "  route starts at x={:.1f} y={:.1f}; {}\n"
            "  tried z from {:.2f} to {:.2f}\n"
            "  actors within 6 m: {}\n"
            "  Most likely: the route was recorded in a different town than "
            "params.town ({}), or something is parked on the start point. "
            "Re-record, or trim the first few points off the route file."
            .format(base.location.x, base.location.y, wp_note,
                    base.location.z, base.location.z + 2.0,
                    blockers or "none", self.p["town"]))

    def _reset_actor(self, actor, ref_xy, ref_yaw0):
        """Return an existing car to the start, stopped.

        Re-used across runs: spawning and destroying 300+ times is slow and
        is a reliable way to provoke CARLA into refusing a spawn partway
        through an overnight sweep.
        """
        carla = self.carla
        base, _ = self._start_transform(ref_xy, ref_yaw0)
        actor.set_simulate_physics(False)
        actor.set_transform(base)
        actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0,
                                                 steer=0.0, hand_brake=True))
        self.world.tick()
        actor.set_simulate_physics(True)
        zero = carla.Vector3D(0.0, 0.0, 0.0)
        for name in ("set_target_velocity", "set_velocity"):
            if hasattr(actor, name):
                getattr(actor, name)(zero)
                break
        for name in ("set_target_angular_velocity", "set_angular_velocity"):
            if hasattr(actor, name):
                getattr(actor, name)(zero)
                break
        for _ in range(3):                      # let the suspension settle
            self.world.tick()
        actor.apply_control(carla.VehicleControl(hand_brake=False))
        return actor

    def _get_actor(self, ref_xy, ref_yaw0):
        if self._actor is None:
            self._actor = self._spawn(ref_xy, ref_yaw0)
        else:
            self._reset_actor(self._actor, ref_xy, ref_yaw0)
        return self._actor

    @staticmethod
    def _state(actor):
        tr = actor.get_transform()
        vel = actor.get_velocity()
        return (
            tr.location.x,
            tr.location.y,
            math.radians(tr.rotation.yaw),
            math.hypot(vel.x, vel.y),
        )

    def run(self, controller_name, gains, ref_xy, ref_v, delay_s, seed,
            max_steps=None, off_road_m=None, abort_cte=None,
            abort_after_frac=0.3):
        carla = self.carla
        off_road_m = off_road_m or self.p["off_road_m"]
        max_steps = max_steps or self.p["max_steps"]

        ref_yaw0 = math.atan2(ref_xy[1, 1] - ref_xy[0, 1],
                              ref_xy[1, 0] - ref_xy[0, 0])
        actor = self._get_actor(ref_xy, ref_yaw0)
        ctrl = build(controller_name, gains, self.dt, self.p)
        buf = DelayBuffer.from_delay(delay_s, self.dt)
        log = {k: [] for k in ("t", "x", "y", "yaw", "v", "v_ref", "steer",
                               "throttle", "solve_ms")}
        completed = False
        hint = 0
        err_sum, err_n = 0.0, 0
        aborted = False
        abort_at = int(max_steps * abort_after_frac)
        try:
            for k in range(max_steps):
                self.world.tick()
                true_state = self._state(actor)
                buf.push(true_state)
                steer, accel = ctrl.step(buf.stale(), ref_xy, ref_v)

                throttle = float(np.clip(accel / self.p["max_accel"], 0.0, 1.0))
                brake = float(np.clip(-accel / self.p["max_accel"], 0.0, 1.0))
                actor.apply_control(carla.VehicleControl(
                    throttle=throttle, brake=brake,
                    steer=float(np.clip(steer / self.p["max_steer"], -1.0, 1.0)),
                ))

                x, y, yaw, v = true_state
                # one windowed scan per tick, reused for the speed reference
                # and the off-road check; a full scan of a 1300-point route
                # twice per tick costs more than the controller does
                lo = max(0, hint - 10)
                hi = min(len(ref_xy), hint + 250)
                d = np.hypot(ref_xy[lo:hi, 0] - x, ref_xy[lo:hi, 1] - y)
                local = int(np.argmin(d))
                idx = lo + local
                hint = idx
                nearest = float(d[local])
                log["t"].append(k * self.dt)
                log["x"].append(x); log["y"].append(y)
                log["yaw"].append(yaw); log["v"].append(v)
                log["v_ref"].append(float(ref_v[idx]))
                log["steer"].append(steer); log["throttle"].append(throttle)
                log["solve_ms"].append(getattr(ctrl, "last_solve_ms", 0.0))

                if float(np.hypot(ref_xy[-1, 0] - x, ref_xy[-1, 1] - y)) < 5.0:
                    completed = True
                    break
                # a tuning candidate already tracking an order of magnitude
                # worse than the target can never be selected, and running it
                # to the end of the route is the bulk of the tuning cost
                err_sum += nearest
                err_n += 1
                if (abort_cte is not None and k > abort_at
                        and err_sum / err_n > abort_cte):
                    aborted = True
                    break
                if nearest > off_road_m:
                    break  # departed: completed stays False
        finally:
            actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0,
                                                     steer=0.0))
        log["completed"] = completed
        log["aborted"] = aborted
        log["progress"] = _progress(log, ref_xy)
        return log

    def close(self):
        if self._actor is not None:
            try:
                self._actor.destroy()
            except Exception:
                pass
            self._actor = None
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)


# --------------------------------------------------------------------------


class KinematicRunner:
    """Bicycle-model stand-in. FOR PIPELINE TESTING ONLY - never for results."""

    stamp = SELFTEST_STAMP

    def __init__(self, params):
        self.p = params
        self.dt = params["dt"]

    def run(self, controller_name, gains, ref_xy, ref_v, delay_s, seed,
            max_steps=None, off_road_m=None, abort_cte=None,
            abort_after_frac=0.3):
        rng = np.random.default_rng(seed)
        off_road_m = off_road_m or self.p["off_road_m"]
        max_steps = max_steps or self.p["max_steps"]
        L = self.p["wheelbase"]

        yaw = math.atan2(ref_xy[1, 1] - ref_xy[0, 1], ref_xy[1, 0] - ref_xy[0, 0])
        x, y, v = float(ref_xy[0, 0]), float(ref_xy[0, 1]), 0.0

        ctrl = build(controller_name, gains, self.dt, self.p)
        buf = DelayBuffer.from_delay(delay_s, self.dt)
        log = {k: [] for k in ("t", "x", "y", "yaw", "v", "v_ref", "steer",
                               "throttle", "solve_ms")}
        completed = False
        hint = 0
        err_sum, err_n = 0.0, 0
        aborted = False
        abort_at = int(max_steps * abort_after_frac)
        for k in range(max_steps):
            buf.push((x, y, yaw, v))
            steer, accel = ctrl.step(buf.stale(), ref_xy, ref_v)
            steer = float(np.clip(steer, -self.p["max_steer"], self.p["max_steer"]))
            accel = float(np.clip(accel, -self.p["max_accel"], self.p["max_accel"]))

            lo = max(0, hint - 10)
            hi = min(len(ref_xy), hint + 250)
            d = np.hypot(ref_xy[lo:hi, 0] - x, ref_xy[lo:hi, 1] - y)
            local = int(np.argmin(d))
            idx = lo + local
            hint = idx
            nearest = float(d[local])
            log["t"].append(k * self.dt)
            log["x"].append(x); log["y"].append(y)
            log["yaw"].append(yaw); log["v"].append(v)
            log["v_ref"].append(float(ref_v[idx]))
            log["steer"].append(steer)
            log["throttle"].append(accel / self.p["max_accel"])
            log["solve_ms"].append(getattr(ctrl, "last_solve_ms", 0.0))

            # integrate; small process noise so repeated seeds differ
            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
            yaw += (v / L) * math.tan(steer) * self.dt
            v = max(0.0, v + accel * self.dt + rng.normal(0.0, 0.01))

            if float(np.hypot(ref_xy[-1, 0] - x, ref_xy[-1, 1] - y)) < 5.0:
                completed = True
                break
            # a tuning candidate already tracking an order of magnitude
            # worse than the target can never be selected, and running it
            # to the end of the route is the bulk of the tuning cost
            err_sum += nearest
            err_n += 1
            if (abort_cte is not None and k > abort_at
                    and err_sum / err_n > abort_cte):
                aborted = True
                break
            if nearest > off_road_m:
                break
        log["completed"] = completed
        log["aborted"] = aborted
        log["progress"] = _progress(log, ref_xy)
        return log
