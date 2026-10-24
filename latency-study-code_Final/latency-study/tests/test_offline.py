"""Tests that run without CARLA. These cover the parts that can silently
produce wrong numbers: the delay buffer, the cross-track metric, the
oscillation detrending, and the budget rule."""

import math
import numpy as np
import pytest

from latency.delay import DelayBuffer
from latency.metrics import (cross_track_series, detrended_steering_std,
                             distinguishable, dominant_frequency,
                             latency_budget)
from latency import controllers as C


def test_buffer_returns_n_step_old_state():
    b = DelayBuffer(3)
    for i in range(10):
        b.push(i)
        if i >= 3:
            assert b.stale() == i - 3


def test_buffer_before_warm_returns_oldest():
    b = DelayBuffer(5)
    b.push("a"); b.push("b")
    assert b.stale() == "a"
    assert not b.is_warm


def test_zero_delay_is_passthrough():
    b = DelayBuffer.from_delay(0.0, 0.05)
    b.push(42)
    assert b.stale() == 42


def test_non_multiple_delay_is_rejected():
    with pytest.raises(ValueError):
        DelayBuffer.from_delay(0.075, 0.05)   # 1.5 steps
    assert DelayBuffer.from_delay(0.150, 0.05).n == 3


def test_cross_track_is_distance_to_segment_not_vertex():
    # sparse straight line; a point mid-segment is 1.0 from the line, and a
    # nearest-vertex metric would wrongly report ~5.1
    ref = np.array([[0.0, 0.0], [10.0, 0.0]])
    d = cross_track_series(ref, [5.0], [1.0])
    assert abs(d[0] - 1.0) < 1e-9


def test_cross_track_zero_on_path():
    ref = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], float)
    d = cross_track_series(ref, [0.5, 2.5], [0.0, 0.0])
    assert np.allclose(d, 0.0, atol=1e-9)


def test_detrending_ignores_a_slow_turn():
    dt = 0.05
    t = np.arange(0, 20, dt)
    slow = 0.3 * np.sin(2 * np.pi * 0.05 * t)      # a long corner
    assert detrended_steering_std(slow, dt) < 0.05


def test_detrending_catches_a_fast_weave():
    dt = 0.05
    t = np.arange(0, 20, dt)
    weave = 0.3 * np.sin(2 * np.pi * 2.0 * t)      # 2 Hz oscillation
    assert detrended_steering_std(weave, dt) > 0.15


def test_dominant_frequency_recovers_a_known_tone():
    dt = 0.05
    t = np.arange(0, 30, dt)
    sig = np.sin(2 * np.pi * 1.5 * t)
    assert abs(dominant_frequency(sig, dt) - 1.5) < 0.15


def test_budget_fires_on_accuracy():
    d = [0, 50, 100, 150]
    cte = [0.15, 0.20, 0.60, 1.2]
    osc = [0.01, 0.011, 0.012, 0.013]
    b = latency_budget(d, cte, osc, cte_threshold_m=0.5)
    assert b.criterion == "cte"
    assert b.bin_ms == 100                      # first failing sweep value
    assert 50 < b.ms < 100                      # crossing is between the two
    assert abs(b.ms - 87.5) < 0.1               # 0.5 is 75% of the way 0.20->0.60


def test_interpolation_can_be_switched_off():
    d = [0, 50, 100]
    cte = [0.15, 0.20, 0.60]
    osc = [0.01, 0.011, 0.012]
    b = latency_budget(d, cte, osc, 0.5, interpolate=False)
    assert b.ms == 100 and b.method == "bin"


def test_interpolation_separates_controllers_one_bin_apart():
    """The whole point: two controllers whose bin budgets are identical can
    still be distinguished, because they cross the threshold at different
    points inside that bin."""
    d = [0, 50, 100]
    osc = [0.01, 0.011, 0.012]
    a = latency_budget(d, [0.15, 0.30, 0.90], osc, 0.5)   # crosses early
    c = latency_budget(d, [0.15, 0.20, 0.55], osc, 0.5)   # crosses late
    assert a.bin_ms == c.bin_ms == 100
    assert a.ms < c.ms


def test_budget_fires_on_oscillation_before_accuracy():
    d = [0, 50, 100]
    cte = [0.15, 0.16, 0.17]
    osc = [0.01, 0.015, 0.05]
    b = latency_budget(d, cte, osc, cte_threshold_m=0.5)
    assert b.criterion == "osc" and b.bin_ms == 100


def test_budget_reports_none_when_never_failing():
    b = latency_budget([0, 50], [0.1, 0.1], [0.01, 0.01], 0.5)
    assert b.ms is None and b.criterion == "none" and b.method == "none"


def test_nan_counts_as_failure_and_is_not_interpolated():
    b = latency_budget([0, 50, 100], [0.15, 0.2, float("nan")],
                       [0.01, 0.011, 0.012], 0.5)
    assert b.ms == 100


def test_seed_scatter_gives_an_interval_containing_the_estimate():
    d = [0, 50, 100]
    cte = [0.15, 0.20, 0.60]
    sd = [0.01, 0.03, 0.08]
    b = latency_budget(d, cte, osc_std=[0.01, 0.011, 0.012],
                       cte_threshold_m=0.5, cte_sd=sd)
    assert b.lo_ms < b.ms < b.hi_ms


def test_overlapping_intervals_are_not_distinguishable():
    d = [0, 50, 100]
    osc = [0.01, 0.011, 0.012]
    sd = [0.02, 0.05, 0.12]
    a = latency_budget(d, [0.15, 0.22, 0.58], osc, 0.5, cte_sd=sd)
    c = latency_budget(d, [0.15, 0.20, 0.56], osc, 0.5, cte_sd=sd)
    assert a.ms != c.ms                 # point estimates differ
    assert not distinguishable(a, c)    # but not beyond seed scatter


def test_well_separated_budgets_are_distinguishable():
    d = [0, 50, 100, 150, 200]
    osc = [0.01] * 5
    tiny = [0.002] * 5
    a = latency_budget(d, [0.15, 0.60, 0.9, 1.1, 1.3], osc, 0.5, cte_sd=tiny)
    c = latency_budget(d, [0.15, 0.18, 0.20, 0.25, 0.70], osc, 0.5, cte_sd=tiny)
    assert distinguishable(a, c)


def _straight_ref(n=400, spacing=1.0, v=8.0):
    xy = np.stack([np.arange(n) * spacing, np.zeros(n)], axis=1)
    return xy, np.full(n, v)


@pytest.mark.parametrize("name,gains", [
    ("pid", dict(lat_kp=0.5, lat_ki=0.0, lat_kd=0.1,
                 lon_kp=1.0, lon_ki=0.1, lon_kd=0.0)),
    ("pure_pursuit", dict(kv=0.5, l0=4.0, lon_kp=1.0, lon_ki=0.1, lon_kd=0.0)),
    ("stanley", dict(k=1.0, ks=1.0, lon_kp=1.0, lon_ki=0.1, lon_kd=0.0)),
    ("mpc", dict(w_pos=1.0, w_vel=0.5, w_ctrl=0.1)),
])
def test_controller_steers_back_toward_the_path(name, gains):
    """Offset the car to the left of a straight path; steering must be negative
    (to the right). This is the wiring check - a sign error here would make
    every result in the study meaningless."""
    params = dict(wheelbase=2.875, max_steer=0.6, max_accel=3.0,
                  max_yaw_rate=1.2, mpc_horizon=10, ilqr_iters=5)
    ref_xy, ref_v = _straight_ref()
    ctrl = C.build(name, gains, 0.05, params)
    steer, accel = ctrl.step((20.0, 2.0, 0.0, 8.0), ref_xy, ref_v)
    assert steer < 0, f"{name} steered away from the path"
    assert abs(steer) <= params["max_steer"] + 1e-9


@pytest.mark.parametrize("name,gains", [
    ("pid", dict(lat_kp=0.5, lat_ki=0.0, lat_kd=0.1,
                 lon_kp=1.0, lon_ki=0.1, lon_kd=0.0)),
    ("stanley", dict(k=1.0, ks=1.0, lon_kp=1.0, lon_ki=0.1, lon_kd=0.0)),
])
def test_controller_accelerates_when_below_reference_speed(name, gains):
    params = dict(wheelbase=2.875, max_steer=0.6, max_accel=3.0,
                  max_yaw_rate=1.2, mpc_horizon=10, ilqr_iters=5)
    ref_xy, ref_v = _straight_ref(v=10.0)
    ctrl = C.build(name, gains, 0.05, params)
    _, accel = ctrl.step((20.0, 0.0, 0.0, 2.0), ref_xy, ref_v)
    assert accel > 0


def test_ilqr_reduces_cost_on_a_lateral_offset():
    soc = C.UnicycleILQR(dt=0.05, horizon=15, w_pos=1.0, w_vel=0.5,
                         w_ctrl=0.05, u_lim=[1.2, 3.0], iters=8)
    x0 = np.array([0.0, 2.0, 0.0, 8.0])
    targets = [(8.0 * 0.05 * t, 0.0, 8.0) for t in range(16)]
    U0 = np.zeros((15, 2))

    def rollout_cost(U):
        x = x0.copy(); c = 0.0
        for t in range(15):
            c += ((x[0] - targets[t][0]) ** 2 + (x[1] - targets[t][1]) ** 2)
            x = soc._f(x, U[t])
        return c

    U = soc.solve(x0, targets, U0)
    assert rollout_cost(U) < rollout_cost(U0) * 0.99


def test_never_failing_ranks_ahead_of_everything():
    """A controller that survived the whole sweep has the largest budget, not
    a missing one. Getting this backwards inverted the ranking."""
    from latency.metrics import budget_sort_key
    osc = [0.01, 0.011, 0.012]
    survived = latency_budget([0, 50, 100], [0.1, 0.1, 0.1], osc, 0.5)
    failed = latency_budget([0, 50, 100], [0.1, 0.2, 0.9], osc, 0.5)
    assert survived.ms is None
    assert budget_sort_key(survived) < budget_sort_key(failed)
    assert sorted(["f", "s"], key=lambda k: budget_sort_key(
        {"f": failed, "s": survived}[k])) == ["s", "f"]


def test_nan_band_does_not_silently_mean_indistinguishable():
    """None becomes NaN through pandas; NaN comparisons are always False, so
    an unguarded interval test called everything unresolvable."""
    import numpy as np
    from latency.metrics import Budget
    a = Budget(120.0, "osc", 150.0, np.nan, np.nan, "interpolated")
    b = Budget(60.0, "osc", 100.0, np.nan, np.nan, "interpolated")
    assert distinguishable(a, b)


def test_failure_at_zero_delay_is_flagged_not_reported_as_a_budget():
    b = latency_budget([0, 50, 100], [0.9, 1.1, 1.4], [0.01] * 3, 0.5)
    assert b.method == "at_first_sample" and b.ms == 0


# --- reference-trajectory recorder: path handling -------------------------
# These exist because a path bug here costs a re-drive, not a re-run.

import importlib.util as _ilu
import os as _os

_spec = _ilu.spec_from_file_location(
    "record_trajectory",
    _os.path.join(_os.path.dirname(__file__), "..", "scripts",
                  "record_trajectory.py"))
rec = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(rec)


def test_relative_out_resolves_against_repo_root_not_cwd(tmp_path, monkeypatch):
    """Running from scripts/ must not write to scripts/data/."""
    monkeypatch.chdir(tmp_path)
    resolved = rec.resolve_out("data/reference_trajectory.txt")
    assert _os.path.isabs(resolved)
    assert resolved.endswith(_os.path.join("data", "reference_trajectory.txt"))
    assert str(tmp_path) not in resolved


def test_absolute_out_is_left_alone(tmp_path):
    p = str(tmp_path / "route.txt")
    assert rec.resolve_out(p) == p


def test_save_creates_missing_parent_directory(tmp_path):
    target = tmp_path / "deep" / "nested" / "route.txt"
    written = rec.save_rows([(1.0, 2.0, 3.0), (4.0, 6.0, 3.5)], str(target),
                            quiet=True)
    assert written == str(target) and target.exists()
    assert target.read_text().splitlines()[0] == "1.000000,2.000000,3.000"


def test_save_falls_back_rather_than_losing_points(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(_os.path, "expanduser", lambda p: str(tmp_path))
    written = rec.save_rows([(0.0, 0.0, 1.0)], "/proc/can't/write/here.txt",
                            quiet=True)
    assert written is not None and _os.path.exists(written)


def test_preflight_rejects_an_unwritable_destination():
    with pytest.raises(SystemExit):
        rec.preflight("/proc/definitely/not/writable/route.txt")


def test_route_length_matches_simple_geometry():
    assert abs(rec.route_length([(0, 0, 0), (3, 4, 0), (3, 4, 0)]) - 5.0) < 1e-9


def test_steering_ramps_instead_of_snapping_to_full_lock():
    """Binary steering puts step changes into the reference every controller
    is then scored against."""
    sm = rec.SteerSmoother(dt=0.05)
    first, _, _ = sm.update(left=True, right=False, up=False, down=False)
    assert 0 < abs(first) < 0.3
    for _ in range(50):
        steer, _, _ = sm.update(True, False, False, False)
    assert abs(steer) <= 0.7


def test_steering_returns_to_centre_when_released():
    sm = rec.SteerSmoother(dt=0.05)
    for _ in range(20):
        sm.update(False, True, False, False)
    assert sm.steer > 0
    for _ in range(100):
        sm.update(False, False, False, False)
    assert sm.steer == 0.0


def test_throttle_ramps_up_and_decays():
    sm = rec.SteerSmoother(dt=0.05)
    for _ in range(40):
        _, throttle, _ = sm.update(False, False, True, False)
    assert 0.5 < throttle <= 0.75
    for _ in range(80):
        _, throttle, _ = sm.update(False, False, False, False)
    assert throttle == 0.0


def test_brake_is_reported_when_down_is_held():
    sm = rec.SteerSmoother(dt=0.05)
    _, _, brake = sm.update(False, False, False, True)
    assert brake == 1.0


def test_every_camera_view_is_well_formed():
    assert len(rec.CAMERA_VIEWS) >= 2
    for name, x, y, z, pitch in rec.CAMERA_VIEWS:
        assert isinstance(name, str) and name
        assert -90.0 <= pitch <= 90.0 and z > 0


# --- reference loading: the standstill-start trap --------------------------

def _write_route(tmp_path, xy, v):
    import numpy as _np
    p = tmp_path / "route.txt"
    _np.savetxt(str(p), _np.column_stack([xy, v]), delimiter=",", fmt="%.6f")
    return str(p)


def test_stationary_start_is_trimmed(tmp_path, capsys):
    """Recording started while parked gives v=0 at the front; the speed loop
    then commands zero throttle and the car never moves, so every candidate
    tuning is rejected for a reason that has nothing to do with tuning."""
    import numpy as _np
    from latency.sweep import load_reference
    xy = _np.stack([_np.arange(100, dtype=float), _np.zeros(100)], axis=1)
    v = _np.concatenate([_np.zeros(15), _np.full(80, 10.0), _np.zeros(5)])
    out_xy, out_v = load_reference(_write_route(tmp_path, xy, v))
    assert len(out_xy) == 80
    assert out_v.min() > 0
    assert "trimmed 15 stationary points" in capsys.readouterr().out


def test_interior_stop_is_floored_not_left_at_zero(tmp_path):
    import numpy as _np
    from latency.sweep import load_reference
    xy = _np.stack([_np.arange(100, dtype=float), _np.zeros(100)], axis=1)
    v = _np.full(100, 10.0)
    v[40:50] = 0.0                      # waited at a light
    _, out_v = load_reference(_write_route(tmp_path, xy, v), v_min=0.5)
    assert out_v.min() >= 0.5


def test_route_that_never_moved_is_rejected_with_advice(tmp_path):
    import numpy as _np
    from latency.sweep import load_reference
    xy = _np.stack([_np.arange(50, dtype=float), _np.zeros(50)], axis=1)
    with pytest.raises(SystemExit) as exc:
        load_reference(_write_route(tmp_path, xy, _np.zeros(50)))
    assert "never moved" in str(exc.value)


def test_too_short_a_route_is_rejected(tmp_path):
    import numpy as _np
    from latency.sweep import load_reference
    xy = _np.stack([_np.arange(10, dtype=float), _np.zeros(10)], axis=1)
    with pytest.raises(SystemExit) as exc:
        load_reference(_write_route(tmp_path, xy, _np.full(10, 10.0)))
    assert "longer route" in str(exc.value)


def test_a_standstill_start_would_have_stalled_the_car():
    """Guards the behaviour the trimming exists to prevent."""
    import numpy as _np, yaml
    from latency.runners import KinematicRunner
    p = yaml.safe_load(open("configs/selftest.yaml"))["params"]
    xy = _np.stack([_np.arange(300, dtype=float), _np.zeros(300)], axis=1)
    gains = dict(lat_kp=0.5, lat_ki=0.0, lat_kd=0.1,
                 lon_kp=1.0, lon_ki=0.1, lon_kd=0.0)
    stalled = KinematicRunner(p).run(
        "pid", gains, xy,
        _np.concatenate([_np.zeros(12), _np.full(288, 10.0)]), 0.0, seed=0)
    assert stalled["progress"] < 0.05 and not stalled["completed"]
    moving = KinematicRunner(p).run(
        "pid", gains, xy, _np.full(300, 10.0), 0.0, seed=0)
    assert moving["progress"] > 0.9 and moving["completed"]


# --- oscillation cap calibration ------------------------------------------

def _tuning_cfg(cap="auto", factor=1.5):
    import yaml
    cfg = yaml.safe_load(open("configs/selftest.yaml"))
    cfg["tuning"]["osc_cap_rad"] = cap
    cfg["tuning"]["osc_cap_factor"] = factor
    cfg["tuning"]["grid"]["pid"] = {"lat_kp": [0.3, 0.9, 3], "lat_kd": [0.0, 0.2, 2]}
    return cfg


def _good_route(n=320, v=9.0):
    import numpy as _np
    t = _np.linspace(0, 1, n)
    xy = _np.stack([300 * t, 14 * _np.sin(2 * _np.pi * t)], axis=1)
    return xy, _np.full(n, v)


def test_auto_cap_accepts_a_route_that_a_fixed_cap_would_reject():
    """The 0.05 rad constant was a guess about the route, not about control.
    A tight twisty route needs more steering activity than an open one."""
    from latency.sweep import tune_matched
    from latency.runners import KinematicRunner
    cfg = _tuning_cfg(cap="auto")
    xy, v = _good_route()
    chosen, directions = tune_matched(KinematicRunner(cfg["params"]), cfg,
                                      "pid", xy, v, verbose=False)
    gains, s = chosen["matched"]
    assert gains and s["completed"]
    assert set(directions) == {"lat_kp", "lat_kd"}

    impossible = _tuning_cfg(cap=1e-9)
    with pytest.raises(RuntimeError) as exc:
        tune_matched(KinematicRunner(impossible["params"]), impossible, "pid",
                     xy, v, verbose=False)
    assert "fixed in your config" in str(exc.value)


def test_auto_cap_message_reports_what_it_calibrated_to():
    from latency.sweep import tune_matched
    from latency.runners import KinematicRunner
    cfg = _tuning_cfg(cap="auto", factor=1.0001)
    xy, v = _good_route()
    try:
        tune_matched(KinematicRunner(cfg["params"]), cfg, "pid", xy, v,
                     verbose=False)
    except RuntimeError as exc:
        assert "osc_cap_factor" in str(exc)


def test_route_description_flags_a_short_straight_fast_route(capsys):
    import numpy as _np
    from latency.sweep import describe_reference
    xy = _np.stack([_np.arange(0, 200, 1.0), _np.zeros(200)], axis=1)
    stats = describe_reference(xy, _np.full(200, 20.0), "fake")
    out = capsys.readouterr().out
    assert stats["length_m"] < 500
    assert "800-1200 m" in out          # too short
    assert "deg of total turning" in out  # too straight
    assert "72 km/h" in out             # too fast


def test_route_description_is_quiet_about_a_good_route(capsys):
    import numpy as _np
    from latency.sweep import describe_reference
    t = _np.linspace(0, 1, 1200)
    xy = _np.stack([900 * t, 120 * _np.sin(4 * _np.pi * t)], axis=1)
    describe_reference(xy, _np.full(1200, 10.0), "fake")
    assert "note:" not in capsys.readouterr().out


# --- matched-tuning quality ------------------------------------------------

def test_grid_edge_in_the_needed_direction_is_reported():
    """Three of four controllers pinned against a grid bound, and the search
    still returned its best corner without saying so."""
    from latency.sweep import check_matched_quality
    notes = check_matched_quality(
        "stanley", {"k": 3.0, "ks": 0.5}, {"mean_cte": 0.302},
        {"k": [0.3, 3.0, 8], "ks": [0.5, 2.0, 3]}, target=0.15, tol=0.05)
    joined = " ".join(notes)
    assert "missed the 0.150 m target" in joined
    assert "k is pinned at the top" in joined and "widen to 6" in joined


def test_pure_pursuit_edge_points_downward_not_upward():
    """A shorter lookahead tracks tighter, so the bound to extend is the
    bottom one - the opposite of every other controller here."""
    from latency.sweep import check_matched_quality
    notes = check_matched_quality(
        "pure_pursuit", {"kv": 0.8, "l0": 2.0}, {"mean_cte": 0.461},
        {"kv": [0.2, 1.2, 6], "l0": [2.0, 8.0, 4]}, target=0.15, tol=0.05)
    joined = " ".join(notes)
    assert "l0 is pinned at the bottom" in joined and "lower" in joined


def test_a_tuning_that_hit_the_target_from_inside_the_grid_is_quiet():
    from latency.sweep import check_matched_quality
    assert check_matched_quality(
        "pid", {"lat_kp": 0.8, "lat_kd": 0.2}, {"mean_cte": 0.152},
        {"lat_kp": [0.2, 1.6, 8], "lat_kd": [0.0, 0.4, 3]},
        target=0.15, tol=0.05) == []


def test_missing_the_target_is_flagged_even_from_inside_the_grid():
    from latency.sweep import check_matched_quality
    notes = check_matched_quality(
        "pid", {"lat_kp": 0.8, "lat_kd": 0.2}, {"mean_cte": 0.40},
        {"lat_kp": [0.2, 1.6, 8], "lat_kd": [0.0, 0.4, 3]},
        target=0.15, tol=0.05)
    assert any("missed" in n for n in notes)


# --- iLQR optimisation must not change what it computes --------------------

def test_early_exit_matches_a_fully_converged_solve():
    """The solver stops when cost stops improving. That's only legitimate if
    it lands in the same place as grinding through every iteration."""
    import numpy as _np
    from latency.controllers import UnicycleILQR
    x0 = _np.array([0.0, 1.5, 0.05, 9.0])
    targets = _np.array([[9.0 * 0.05 * t, 0.0, 9.0] for t in range(21)])
    U0 = _np.zeros((20, 2))

    fast = UnicycleILQR(0.05, 20, 1.0, 0.5, 0.1, [1.2, 3.0], iters=8, tol=1e-4)
    thorough = UnicycleILQR(0.05, 20, 1.0, 0.5, 0.1, [1.2, 3.0], iters=60,
                            tol=0.0)
    Uf = fast.solve(x0, targets, U0)
    Ut = thorough.solve(x0, targets, U0)

    cf = fast._cost(fast._rollout(x0, Uf), Uf, targets)
    ct = thorough._cost(thorough._rollout(x0, Ut), Ut, targets)
    assert cf <= ct * 1.02           # within 2% of the fully converged cost
    assert _np.max(_np.abs(Uf[0] - Ut[0])) < 0.05   # same first command


def test_closed_form_inverse_matches_numpy():
    import numpy as _np
    from latency.controllers import UnicycleILQR
    for M in (_np.array([[2.0, 0.3], [0.1, 1.5]]),
              _np.array([[1e3, -2.0], [-2.0, 4e-2]])):
        assert _np.allclose(UnicycleILQR._inv2(M), _np.linalg.inv(M))


def test_singular_control_hessian_does_not_blow_up():
    import numpy as _np
    from latency.controllers import UnicycleILQR
    out = UnicycleILQR._inv2(_np.zeros((2, 2)))
    assert _np.all(_np.isfinite(out))


def test_windowed_reference_scan_finds_the_same_point_as_a_full_scan():
    """The runner used to scan the whole route twice per tick. The windowed
    version must not latch onto a different part of a route that doubles
    back."""
    import numpy as _np, yaml
    from latency.runners import KinematicRunner
    p = yaml.safe_load(open("configs/sweep.yaml"))["params"]
    p = dict(p); p["max_steps"] = 400
    t = _np.linspace(0, 1, 600)
    xy = _np.stack([200 * t, 30 * _np.sin(4 * _np.pi * t)], axis=1)
    v = _np.full(600, 8.0)
    log = KinematicRunner(p).run(
        "pid", dict(lat_kp=0.5, lat_ki=0.0, lat_kd=0.1,
                    lon_kp=1.0, lon_ki=0.1, lon_kd=0.0),
        xy, v, 0.0, seed=0)
    for x, y, vr in zip(log["x"][::25], log["y"][::25], log["v_ref"][::25]):
        full = int(_np.argmin(_np.hypot(xy[:, 0] - x, xy[:, 1] - y)))
        assert abs(vr - v[full]) < 1e-9


# --- measured gain direction, and the resolution floor ---------------------

def test_gain_direction_is_measured_not_assumed():
    """The hardcoded table said raising w_pos tracks tighter. On a real route
    it did the opposite, which flipped the sign of MPC's exchange rate."""
    from latency.sweep import measure_gain_directions
    tighter = [({"g": x}, {"mean_cte": 1.0 - 0.1 * x}) for x in range(8)]
    looser = [({"g": x}, {"mean_cte": 0.2 + 0.1 * x}) for x in range(8)]
    assert measure_gain_directions(tighter, ["g"])["g"] == +1
    assert measure_gain_directions(looser, ["g"])["g"] == -1


def test_gain_direction_falls_back_when_the_data_says_nothing():
    from latency.sweep import measure_gain_directions
    flat = [({"g": x}, {"mean_cte": 0.5}) for x in range(8)]
    assert measure_gain_directions(flat, ["g"], {"g": -1})["g"] == -1


def test_aggressive_always_means_tighter_whatever_the_direction():
    """Otherwise Table V compares a looser tuning against a tighter one with
    the sign flipped, which is how MPC got a negative exchange rate."""
    from latency.sweep import _scale_dominant
    up = _scale_dominant({"w_pos": 2.0}, "mpc", 2.0, {"w_pos": +1})
    down = _scale_dominant({"w_pos": 2.0}, "mpc", 2.0, {"w_pos": -1})
    assert up["w_pos"] == 4.0        # raising it tracks tighter
    assert down["w_pos"] == 1.0      # lowering it does


def test_budget_below_one_sweep_step_is_not_reported_as_a_number():
    """A crossing interpolated inside the first bracket is extrapolation from
    two points, not a measurement."""
    d = [0, 50, 100]
    cte = [0.48, 2.0, 3.0]          # crosses 0.50 almost immediately
    b = latency_budget(d, cte, [0.01] * 3, 0.5, resolution_ms=50)
    assert b.method == "below_resolution"
    assert b.ms < 50


def test_a_budget_above_the_step_is_still_reported_normally():
    d = [0, 50, 100]
    cte = [0.15, 0.20, 0.90]
    b = latency_budget(d, cte, [0.01] * 3, 0.5, resolution_ms=50)
    assert b.method == "interpolated" and 50 < b.ms < 100


def test_band_follows_the_criterion_that_decided_the_budget():
    """When both criteria fire in the same interval the earlier crossing wins,
    and the band must describe that curve - not the other one."""
    d = [0, 50]
    cte = [0.45, 0.60]              # crosses 0.50 late in the interval
    osc = [0.01, 0.30]              # doubles almost at once -> earlier
    b = latency_budget(d, cte, osc, 0.5, cte_sd=[0.01, 0.01],
                       osc_sd=[0.001, 0.02])
    assert b.criterion == "both"
    if b.lo_ms is not None:
        assert b.lo_ms <= b.ms <= b.hi_ms


def test_tracking_improving_with_delay_is_flagged():
    """Delay can't improve tracking. When it looks like it does, the tuning
    is lag-dominated and the delay is cancelling that lag."""
    import pandas as _pd
    from latency.analyze import improves_with_delay
    better = _pd.DataFrame({"delay_ms": [0, 50, 100, 150, 200, 250],
                            "mean_cte": [0.223, .219, .215, .212, .209, .206]})
    normal = _pd.DataFrame({"delay_ms": [0, 50, 100, 150, 200, 250],
                            "mean_cte": [0.15, .16, .19, .35, .90, 2.1]})
    assert improves_with_delay(better)
    assert not improves_with_delay(normal)


def test_flat_then_failing_curve_is_not_flagged():
    import pandas as _pd
    from latency.analyze import improves_with_delay
    flat = _pd.DataFrame({"delay_ms": [0, 50, 100, 150, 200, 250],
                          "mean_cte": [0.20, .20, .20, .21, .60, 2.0]})
    assert not improves_with_delay(flat)


def test_three_tunings_come_from_measured_points_and_are_ordered():
    """Scaling a gain away from a matched point that sits on a grid edge
    walks outside the searched range, and 'aggressive' can land looser than
    'conservative'."""
    import numpy as _np, yaml
    from latency.sweep import tune_matched
    from latency.runners import KinematicRunner
    cfg = yaml.safe_load(open("configs/selftest.yaml"))
    cfg["tuning"]["grid"]["pid"] = {"lat_kp": [0.1, 1.2, 6],
                                    "lat_kd": [0.0, 0.3, 2]}
    t = _np.linspace(0, 1, 320)
    xy = _np.stack([300 * t, 14 * _np.sin(2 * _np.pi * t)], axis=1)
    chosen, _ = tune_matched(KinematicRunner(cfg["params"]), cfg, "pid",
                             xy, _np.full(320, 9.0), verbose=False)
    assert set(chosen) == {"conservative", "matched", "aggressive"}
    ctes = {n: chosen[n][1]["mean_cte"] for n in chosen}
    assert ctes["aggressive"] <= ctes["matched"] <= ctes["conservative"]


# --- tuning cost ----------------------------------------------------------

def test_max_steps_is_sized_from_the_route(capsys):
    import numpy as _np
    from latency.sweep import resolve_max_steps
    ref = _np.zeros((1045, 2))
    cfg = {"params": {"max_steps": "auto", "dt": 0.05}}
    steps = resolve_max_steps(cfg, ref)          # no speeds: point fallback
    assert 1045 < steps < 1045 * 4
    assert "auto ->" in capsys.readouterr().out


def test_an_explicit_max_steps_is_respected():
    import numpy as _np
    from latency.sweep import resolve_max_steps
    cfg = {"params": {"max_steps": 777, "dt": 0.05}}
    assert resolve_max_steps(cfg, _np.zeros((100, 2)), verbose=False) == 777


def test_candidate_tracking_worse_than_the_threshold_is_aborted():
    """Most grid candidates can never be selected; running them to the end of
    the route is the bulk of the tuning cost. A candidate that stays on the
    road but tracks badly would otherwise run to completion."""
    import numpy as _np, yaml
    from latency.runners import KinematicRunner
    p = dict(yaml.safe_load(open("configs/sweep.yaml"))["params"])
    p["dt"] = 0.05
    p["max_steps"] = 1500
    t = _np.linspace(0, 1, 600)
    xy = _np.stack([400 * t, 20 * _np.sin(2 * _np.pi * t)], axis=1)
    v = _np.full(600, 9.0)
    good = dict(lat_kp=0.5, lat_ki=0.0, lat_kd=0.1,
                lon_kp=1.0, lon_ki=0.1, lon_kd=0.0)
    full = KinematicRunner(p).run("pid", good, xy, v, 0.0, seed=0)
    # an abort threshold below what this tuning achieves must cut it short
    cut = KinematicRunner(p).run("pid", good, xy, v, 0.0, seed=0,
                                 abort_cte=0.001)
    assert len(cut["x"]) < len(full["x"])
    assert not cut["completed"]


def test_a_good_candidate_is_not_aborted():
    import numpy as _np, yaml
    from latency.runners import KinematicRunner
    p = dict(yaml.safe_load(open("configs/sweep.yaml"))["params"])
    p["dt"] = 0.05
    p["max_steps"] = 1200
    t = _np.linspace(0, 1, 500)
    xy = _np.stack([400 * t, 20 * _np.sin(2 * _np.pi * t)], axis=1)
    v = _np.full(500, 9.0)
    good = dict(lat_kp=0.5, lat_ki=0.0, lat_kd=0.1,
                lon_kp=1.0, lon_ki=0.1, lon_kd=0.0)
    log = KinematicRunner(p).run("pid", good, xy, v, 0.0, seed=0, abort_cte=1.5)
    assert log["completed"]


def test_every_candidate_is_classified_not_just_the_last():
    """A progress-print insertion once moved the classification block outside
    the candidate loop, so only the final grid point was ever evaluated and
    'matched' was whatever the grid happened to end on."""
    import numpy as _np, yaml
    from latency.runners import KinematicRunner
    from latency import sweep as sw

    cfg = yaml.safe_load(open("configs/selftest.yaml"))
    cfg["tuning"]["grid"]["pid"] = {"lat_kp": [0.2, 1.0, 5],
                                    "lat_kd": [0.0, 0.2, 2]}
    t = _np.linspace(0, 1, 320)
    xy = _np.stack([300 * t, 14 * _np.sin(2 * _np.pi * t)], axis=1)
    v = _np.full(320, 9.0)

    seen = []
    real_run = KinematicRunner.run

    def spy(self, controller, gains, *a, **kw):
        seen.append(dict(gains))
        return real_run(self, controller, gains, *a, **kw)

    KinematicRunner.run = spy
    try:
        chosen, _ = sw.tune_matched(KinematicRunner(cfg["params"]), cfg,
                                    "pid", xy, v, verbose=False)
    finally:
        KinematicRunner.run = real_run

    assert len(seen) == 10                      # 5 x 2 grid, all run
    # the selected tuning must be a real search result, not simply the last
    assert len({tuple(sorted(g.items())) for g in seen}) == 10
    picked = chosen["matched"][0]
    assert any(abs(picked["lat_kp"] - g["lat_kp"]) < 1e-9 for g in seen)


def test_tuning_is_searched_with_verbose_off_as_well():
    """The bug only bit when verbose was True, so quiet runs looked fine."""
    import numpy as _np, yaml
    from latency.runners import KinematicRunner
    from latency.sweep import tune_matched
    cfg = yaml.safe_load(open("configs/selftest.yaml"))
    cfg["tuning"]["grid"]["pid"] = {"lat_kp": [0.2, 1.0, 5],
                                    "lat_kd": [0.0, 0.2, 2]}
    t = _np.linspace(0, 1, 320)
    xy = _np.stack([300 * t, 14 * _np.sin(2 * _np.pi * t)], axis=1)
    v = _np.full(320, 9.0)
    quiet, _ = tune_matched(KinematicRunner(cfg["params"]), cfg, "pid", xy, v,
                            verbose=False)
    loud, _ = tune_matched(KinematicRunner(cfg["params"]), cfg, "pid", xy, v,
                           verbose=True)
    assert quiet["matched"][0] == loud["matched"][0]


def test_conservative_tuning_must_stay_inside_the_failure_threshold():
    """A conservative tuning whose nominal error already exceeds the failure
    threshold has no room to degrade, so its budget is zero by construction
    and Table V gets a negative exchange rate."""
    import numpy as _np, yaml
    from latency.sweep import tune_matched
    from latency.runners import KinematicRunner
    cfg = yaml.safe_load(open("configs/selftest.yaml"))
    cfg["tuning"]["failure_cte_m"] = 0.50
    cfg["tuning"]["conservative_headroom"] = 0.6
    cfg["tuning"]["grid"]["pid"] = {"lat_kp": [0.03, 1.2, 8],
                                    "lat_kd": [0.0, 0.3, 2]}
    t = _np.linspace(0, 1, 320)
    xy = _np.stack([300 * t, 16 * _np.sin(2 * _np.pi * t)], axis=1)
    chosen, _ = tune_matched(KinematicRunner(cfg["params"]), cfg, "pid",
                             xy, _np.full(320, 9.0), verbose=False)
    cons = chosen["conservative"][1]["mean_cte"]
    assert cons <= 0.50 * 0.6 + 1e-9, cons


def test_nan_metrics_at_zero_delay_are_not_reported_as_already_failing():
    """Missing data and an already-failed tuning need different fixes."""
    import numpy as _np
    b = latency_budget([0, 50, 100], [float("nan"), 0.2, 0.3],
                       [0.01, 0.01, 0.01], 0.5)
    assert b.method == "no_metrics"
    bad = latency_budget([0, 50, 100], [0.9, 1.0, 1.1], [0.01] * 3, 0.5)
    assert bad.method == "at_first_sample"


def test_max_steps_follows_driving_time_not_point_count():
    """A route recorded at 20 Hz and replayed at 50 Hz needs 2.5x the ticks.
    Counting reference points gave every candidate too little time to finish."""
    import numpy as _np
    from latency.sweep import resolve_max_steps, route_duration_s
    n = 1146
    t = _np.linspace(0, 1, n)
    xy = _np.stack([560 * t, _np.zeros(n)], axis=1)
    v = _np.full(n, 560.0 / 57.0)            # ~57 s of driving
    assert abs(route_duration_s(xy, v) - 57.0) < 1.0

    slow = resolve_max_steps({"params": {"max_steps": "auto", "dt": 0.05}},
                             xy, v, verbose=False)
    fast = resolve_max_steps({"params": {"max_steps": "auto", "dt": 0.02}},
                             xy, v, verbose=False)
    assert fast > slow * 2                    # scales with the rate
    for dt, steps in ((0.05, slow), (0.02, fast)):
        assert steps > (57.0 / dt) * 1.5      # real headroom at both rates


def test_duration_is_independent_of_how_densely_the_route_was_sampled():
    import numpy as _np
    from latency.sweep import route_duration_s
    for n in (300, 1200, 4000):
        t = _np.linspace(0, 1, n)
        xy = _np.stack([500 * t, _np.zeros(n)], axis=1)
        assert abs(route_duration_s(xy, _np.full(n, 10.0)) - 50.0) < 0.5


def test_every_command_sizes_its_step_limit():
    """tune, sweep and doctor must all size max_steps, or one of them silently
    runs with a limit meant for a different simulation rate."""
    import re
    src = open("src/latency/sweep.py").read()
    for fn in ("cmd_tune", "cmd_sweep", "cmd_doctor"):
        body = src[src.index("def {}(".format(fn)):]
        body = body[:body.index("\ndef ", 1)] if "\ndef " in body[1:] else body
        assert "resolve_max_steps" in body, fn


def test_quiet_tuning_is_not_failed_by_a_trivial_absolute_wobble():
    """A doubling criterion alone penalises quiet tunings: 0.005 -> 0.010 rad
    is 0.6 degrees of steering. That made conservative budgets come out BELOW
    aggressive ones and flipped the sign of Table V."""
    d = [0, 50, 100, 150]
    cte = [0.20, 0.21, 0.22, 0.60]
    quiet = [0.005, 0.011, 0.012, 0.013]      # doubles, but stays tiny
    b = latency_budget(d, cte, quiet, 0.5, osc_floor_rad=0.02)
    assert b.criterion == "cte"                # not tripped by the wobble
    no_floor = latency_budget(d, cte, quiet, 0.5)   # default: floor off
    assert no_floor.criterion == "osc"         # the old behaviour
    assert no_floor.ms < b.ms


def test_real_oscillation_still_trips_the_criterion():
    d = [0, 50, 100]
    b = latency_budget(d, [0.20, 0.21, 0.22], [0.03, 0.04, 0.12], 0.5,
                       osc_floor_rad=0.02)
    assert b.criterion == "osc" and b.ms is not None


def test_floor_does_not_fire_below_itself_even_on_a_big_relative_jump():
    d = [0, 50, 100]
    b = latency_budget(d, [0.2, 0.2, 0.2], [0.001, 0.008, 0.015], 0.5,
                       osc_floor_rad=0.02)
    assert b.criterion == "none"
