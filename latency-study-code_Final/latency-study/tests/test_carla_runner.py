"""Tests for CarlaRunner's spawn and reset logic, against a fake carla module.

This path can't be exercised without a simulator, and it's exactly where an
overnight sweep dies: a bad spawn height, a leftover actor from a crashed run,
or a route recorded in the wrong town. Mocking `carla` is cheap insurance.
"""

import math
import sys
import types

import numpy as np
import pytest


# --------------------------------------------------------------------------
# minimal fake carla


class Location:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2
                         + (self.z - other.z) ** 2)


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or Location()
        self.rotation = rotation or Rotation()


class Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class VehicleControl:
    def __init__(self, throttle=0.0, brake=0.0, steer=0.0, hand_brake=False):
        self.throttle, self.brake = throttle, brake
        self.steer, self.hand_brake = steer, hand_brake


class LaneType:
    Driving = 1
    Shoulder = 2


class Waypoint:
    def __init__(self, x, y, z):
        self.transform = Transform(Location(x, y, z))


class FakeActor:
    def __init__(self, tf, type_id="vehicle.tesla.model3"):
        self.type_id = type_id
        self.transform = tf
        self.destroyed = False
        self.set_transform_calls = 0
        self.physics = True

    def get_transform(self):
        return self.transform

    def get_location(self):
        return self.transform.location

    def get_velocity(self):
        return Vector3D()

    def set_transform(self, tf):
        self.transform = tf
        self.set_transform_calls += 1

    def set_simulate_physics(self, on):
        self.physics = on

    def set_target_velocity(self, v):
        pass

    def set_target_angular_velocity(self, v):
        pass

    def apply_control(self, c):
        pass

    def destroy(self):
        self.destroyed = True


class FakeActorList(list):
    def filter(self, pattern):
        prefix = pattern.replace("*", "")
        return [a for a in self if a.type_id.startswith(prefix)]


class FakeMap:
    def __init__(self, road_z=10.0, roads_everywhere=True):
        self.road_z = road_z
        self.roads_everywhere = roads_everywhere

    def get_waypoint(self, loc, project_to_road=True, lane_type=None):
        if not self.roads_everywhere and abs(loc.x) > 50:
            return None
        return Waypoint(loc.x, loc.y, self.road_z)


class FakeWorld:
    def __init__(self, blocked_below=None, road_z=10.0, roads_everywhere=True,
                 existing=()):
        self.blocked_below = blocked_below   # spawns below this z fail
        self._map = FakeMap(road_z, roads_everywhere)
        self.actors = FakeActorList(existing)
        self.ticks = 0
        self.settings = types.SimpleNamespace(
            synchronous_mode=False, fixed_delta_seconds=None,
            no_rendering_mode=False)

    def get_settings(self):
        return self.settings

    def apply_settings(self, s):
        self.settings = s

    def get_blueprint_library(self):
        return types.SimpleNamespace(find=lambda name: name)

    def get_map(self):
        return self._map

    def get_actors(self):
        return self.actors

    def try_spawn_actor(self, bp, tf):
        if self.blocked_below is not None and tf.location.z < self.blocked_below:
            return None
        a = FakeActor(tf)
        self.actors.append(a)
        return a

    def tick(self):
        self.ticks += 1

    def get_spectator(self):
        return FakeActor(Transform())


class FakeClient:
    def __init__(self, world):
        self._world = world

    def set_timeout(self, t):
        pass

    def load_world(self, town):
        return self._world


def install_fake_carla(monkeypatch, world):
    mod = types.ModuleType("carla")
    mod.Location, mod.Rotation, mod.Transform = Location, Rotation, Transform
    mod.Vector3D, mod.VehicleControl = Vector3D, VehicleControl
    mod.LaneType = LaneType
    mod.Client = lambda host, port: FakeClient(world)
    monkeypatch.setitem(sys.modules, "carla", mod)
    return mod


PARAMS = dict(town="Town03", vehicle_bp="vehicle.tesla.model3", dt=0.05,
              wheelbase=2.875, max_steer=0.6, max_accel=3.0, max_yaw_rate=1.2,
              mpc_horizon=10, ilqr_iters=4, warmup_s=1.0, off_road_m=6.0,
              max_steps=50)

REF = np.stack([np.arange(60, dtype=float), np.zeros(60)], axis=1)


# --------------------------------------------------------------------------


def test_spawn_height_comes_from_the_map_not_a_constant(monkeypatch):
    """z=0.5 was hardcoded, which puts the car inside elevated roadway."""
    from latency.runners import CarlaRunner
    world = FakeWorld(road_z=12.5)
    install_fake_carla(monkeypatch, world)
    r = CarlaRunner(PARAMS)
    actor = r._spawn(REF, 0.0)
    assert actor.transform.location.z == pytest.approx(12.8, abs=1e-6)


def test_spawn_retries_higher_when_the_first_attempt_is_blocked(monkeypatch):
    from latency.runners import CarlaRunner
    world = FakeWorld(road_z=0.0, blocked_below=0.9)
    install_fake_carla(monkeypatch, world)
    r = CarlaRunner(PARAMS)
    actor = r._spawn(REF, 0.0)
    assert actor.transform.location.z >= 0.9


def test_spawn_failure_message_names_the_town_and_blockers(monkeypatch):
    from latency.runners import CarlaRunner
    world = FakeWorld(road_z=0.0, blocked_below=99.0)
    install_fake_carla(monkeypatch, world)
    r = CarlaRunner(PARAMS)
    with pytest.raises(RuntimeError) as exc:
        r._spawn(REF, 0.0)
    msg = str(exc.value)
    assert "Town03" in msg and "route starts at" in msg


def test_leftover_vehicles_are_cleared_on_startup(monkeypatch):
    """A sweep that died mid-run leaves its car parked on the start point."""
    from latency.runners import CarlaRunner
    stale = FakeActor(Transform(Location(0, 0, 0)))
    world = FakeWorld(existing=[stale])
    install_fake_carla(monkeypatch, world)
    CarlaRunner(PARAMS)
    assert stale.destroyed


def test_actor_is_reused_across_runs_not_respawned(monkeypatch):
    """300+ spawn/destroy cycles is slow and makes CARLA refuse spawns."""
    from latency.runners import CarlaRunner
    world = FakeWorld(road_z=0.0)
    install_fake_carla(monkeypatch, world)
    r = CarlaRunner(PARAMS)
    a1 = r._get_actor(REF, 0.0)
    a2 = r._get_actor(REF, 0.0)
    a3 = r._get_actor(REF, 0.0)
    assert a1 is a2 is a3
    assert len(world.actors.filter("vehicle.")) == 1
    assert a1.set_transform_calls == 2       # reset twice, spawned once


def test_reset_returns_the_car_to_the_start_with_physics_on(monkeypatch):
    from latency.runners import CarlaRunner
    world = FakeWorld(road_z=4.0)
    install_fake_carla(monkeypatch, world)
    r = CarlaRunner(PARAMS)
    actor = r._get_actor(REF, 0.0)
    actor.set_transform(Transform(Location(500, 500, 4.0)))   # drive away
    r._get_actor(REF, 0.0)
    assert actor.transform.location.x == pytest.approx(REF[0, 0])
    assert actor.transform.location.y == pytest.approx(REF[0, 1])
    assert actor.physics is True


def test_route_from_another_town_is_rejected_before_driving(monkeypatch):
    from latency.runners import CarlaRunner
    world = FakeWorld(roads_everywhere=False)
    install_fake_carla(monkeypatch, world)
    r = CarlaRunner(PARAMS)
    far_route = np.stack([np.arange(100, 400, dtype=float),
                          np.zeros(300)], axis=1)
    with pytest.raises(SystemExit) as exc:
        r.validate_reference(far_route)
    assert "doesn't match the loaded town" in str(exc.value)


def test_a_route_on_the_roads_passes_validation(monkeypatch):
    from latency.runners import CarlaRunner
    world = FakeWorld()
    install_fake_carla(monkeypatch, world)
    CarlaRunner(PARAMS).validate_reference(REF)     # must not raise
