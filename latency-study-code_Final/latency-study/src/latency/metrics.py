"""Metrics, and the latency-budget definition used in the paper.

The budget definition here is the single most contestable choice in the
study, so it lives in one function with its thresholds passed in, rather
than being scattered through the analysis.
"""

from collections import namedtuple

import numpy as np


def cross_track_series(ref_xy, xs, ys):
    """Perpendicular distance to the reference polyline at every logged pose.

    Distance is to the nearest *segment*, not the nearest vertex, so the
    result doesn't depend on how densely the reference was sampled.
    """
    ref = np.asarray(ref_xy, float)
    p0 = ref[:-1]
    p1 = ref[1:]
    seg = p1 - p0
    L2 = np.sum(seg ** 2, axis=1)
    L2[L2 < 1e-12] = 1e-12
    out = np.empty(len(xs))
    for i, (x, y) in enumerate(zip(xs, ys)):
        w = np.stack([x - p0[:, 0], y - p0[:, 1]], axis=1)
        t = np.clip(np.sum(w * seg, axis=1) / L2, 0.0, 1.0)
        proj = p0 + seg * t[:, None]
        out[i] = np.min(np.hypot(x - proj[:, 0], y - proj[:, 1]))
    return out


def detrended_steering_std(steer, dt, cutoff_hz=0.5):
    """Std. dev. of the steering command after removing the low-frequency part.

    Cornering is a legitimate low-frequency steering input. Without removing
    it, a route with more corners would register as a controller that
    oscillates more, which would make the oscillation criterion measure the
    road rather than the controller.
    """
    steer = np.asarray(steer, float)
    if len(steer) < 8:
        return 0.0
    n = max(3, int(round(1.0 / (cutoff_hz * dt))))
    if n % 2 == 0:
        n += 1
    kern = np.ones(n) / n
    pad = n // 2
    padded = np.pad(steer, pad, mode="edge")
    trend = np.convolve(padded, kern, mode="valid")
    return float(np.std(steer - trend))


def dominant_frequency(steer, dt, fmin=0.2):
    """Peak frequency of the steering signal above fmin, in Hz."""
    steer = np.asarray(steer, float)
    if len(steer) < 16:
        return float("nan")
    s = steer - np.mean(steer)
    spec = np.abs(np.fft.rfft(s * np.hanning(len(s))))
    freqs = np.fft.rfftfreq(len(s), dt)
    mask = freqs >= fmin
    if not mask.any():
        return float("nan")
    return float(freqs[mask][int(np.argmax(spec[mask]))])


def summarise_run(log, ref_xy, dt, warmup_s=3.0):
    """Reduce one run's time series to the row the sweep writes out."""
    k0 = int(round(warmup_s / dt))
    xs = np.asarray(log["x"][k0:], float)
    ys = np.asarray(log["y"][k0:], float)
    vs = np.asarray(log["v"][k0:], float)
    vref = np.asarray(log["v_ref"][k0:], float)
    steer = np.asarray(log["steer"][k0:], float)
    if len(xs) < 10:
        return dict(completed=False, mean_cte=float("nan"), max_cte=float("nan"),
                    mean_v_err=float("nan"), osc_std=float("nan"),
                    dom_freq=float("nan"), n_samples=len(xs))
    cte = cross_track_series(ref_xy, xs, ys)
    return dict(
        completed=bool(log.get("completed", True)),
        mean_cte=float(np.mean(cte)),
        max_cte=float(np.max(cte)),
        mean_v_err=float(np.mean(np.abs(vs - vref))),
        osc_std=detrended_steering_std(steer, dt),
        dom_freq=dominant_frequency(steer, dt),
        n_samples=int(len(xs)),
    )


Budget = namedtuple(
    "Budget",
    "ms criterion bin_ms lo_ms hi_ms method",
)
"""ms      interpolated threshold crossing, the number to report
   criterion  which test fired first: cte | osc | both | none
   bin_ms     first failing sweep value (what a bin-only method would give)
   lo_ms/hi_ms  crossing of the mean +/- 1 SD curves; None if no SD supplied
   method     interpolated | bin | at_first_sample | none
"""


def _first_crossing(d, y, thr):
    """First index where y crosses above thr, and the interpolated delay.

    Returns (index, delay) or (None, None). NaN counts as a failure but
    can't be interpolated through, so the bin value is returned instead.
    """
    for i in range(len(d)):
        if np.isnan(y[i]):
            return i, float(d[i])
        if y[i] > thr:
            if i == 0:
                return 0, float(d[0])
            y0, y1 = y[i - 1], y[i]
            if not np.isfinite(y0) or abs(y1 - y0) < 1e-12:
                return i, float(d[i])
            frac = (thr - y0) / (y1 - y0)
            frac = min(max(frac, 0.0), 1.0)
            return i, float(d[i - 1] + frac * (d[i] - d[i - 1]))
    return None, None


def latency_budget(delays_ms, mean_cte, osc_std, cte_threshold_m,
                   osc_multiple=2.0, osc_nominal=None, cte_sd=None,
                   osc_sd=None, interpolate=True, resolution_ms=None,
                   osc_floor_rad=0.0):
    """Delay at which the controller first fails either criterion.

    The sweep is discrete, so taking the first failing sweep value quantises
    every budget to the step size. With a 50 ms step and controllers 50 ms
    apart, no ranking is resolvable. Interpolating the threshold crossing
    between the two bracketing sweep points recovers sub-step resolution
    from data already collected.

    Pass cte_sd (per-delay standard deviation over seeds) to get lo_ms and
    hi_ms, the crossings of the mean +/- 1 SD curves. Two controllers whose
    [lo, hi] intervals overlap aren't distinguishable by this experiment
    and shouldn't be ranked against each other.
    """
    order = np.argsort(delays_ms)
    d = np.asarray(delays_ms, float)[order]
    c = np.asarray(mean_cte, float)[order]
    o = np.asarray(osc_std, float)[order]
    if osc_nominal is None:
        osc_nominal = o[0]
    # A doubling criterion alone is unfair between tunings: a quiet tuning
    # trips on a tiny absolute increase (0.005 -> 0.010 rad is 0.6 degrees of
    # steering wobble) while a busy one is allowed four degrees. That made
    # conservative tunings fail EARLIER than aggressive ones and flipped the
    # sign of the exchange rate. The floor says what counts as oscillation at
    # all, regardless of how quiet the controller started.
    if osc_nominal > 0:
        osc_thr = max(osc_multiple * osc_nominal, osc_floor_rad)
    else:
        osc_thr = np.inf

    i_c, d_c = _first_crossing(d, c, cte_threshold_m)
    i_o, d_o = _first_crossing(d, o, osc_thr)

    if i_c is None and i_o is None:
        return Budget(None, "none", None, None, None, "none")

    # `decided_by` is which curve produced the reported number. It can differ
    # from `crit`: when both criteria fire in the same interval, the earlier
    # interpolated crossing wins, and the uncertainty band has to be computed
    # on THAT curve or it describes a different quantity than the estimate.
    if i_c is None:
        crit = decided_by = "osc"
        idx, cross = i_o, d_o
    elif i_o is None:
        crit = decided_by = "cte"
        idx, cross = i_c, d_c
    elif i_c == i_o:
        crit = "both"
        decided_by = "cte" if d_c <= d_o else "osc"
        idx, cross = i_c, min(d_c, d_o)
    elif i_c < i_o:
        crit = decided_by = "cte"
        idx, cross = i_c, d_c
    else:
        crit = decided_by = "osc"
        idx, cross = i_o, d_o

    bin_ms = float(d[idx])
    if not interpolate:
        return Budget(bin_ms, crit, bin_ms, None, None, "bin")
    if idx == 0:
        first_nan = (np.isnan(c[0]) or np.isnan(o[0]))
        method = "no_metrics" if first_nan else "at_first_sample"
    else:
        method = "interpolated"

    # seed-scatter band, on whichever curve actually decided the budget
    # a crossing interpolated below one sweep step isn't a measurement: the
    # bracket is [0, step] and everything inside it is extrapolated from two
    # points. Report the step as an upper bound instead of false precision.
    if (method == "interpolated" and resolution_ms is not None
            and cross < resolution_ms):
        return Budget(round(cross, 1), crit, bin_ms, None, None,
                      "below_resolution")

    lo = hi = None
    if method == "interpolated":
        if decided_by == "cte" and cte_sd is not None:
            y, thr, raw = c, cte_threshold_m, cte_sd
        elif decided_by == "osc" and osc_sd is not None and np.isfinite(osc_thr):
            y, thr, raw = o, osc_thr, osc_sd
        else:
            y = None
        if y is not None:
            sd = np.nan_to_num(np.asarray(raw, float)[order], nan=0.0)
            _, lo = _first_crossing(d, y + sd, thr)   # fails earlier
            _, hi = _first_crossing(d, y - sd, thr)   # fails later
            if hi is None:
                hi = float(d[-1])

    return Budget(round(cross, 1), crit, bin_ms,
                  None if lo is None else round(lo, 1),
                  None if hi is None else round(hi, 1),
                  method)


def _missing(v):
    """None survives a round trip through pandas as NaN; treat both alike."""
    return v is None or (isinstance(v, float) and np.isnan(v))


def budget_sort_key(b: Budget):
    """Sort key for ranking, largest budget first.

    A controller that never failed inside the sweep has the LARGEST budget,
    not a missing one, so it must sort ahead of every measured value.
    """
    return -np.inf if _missing(b.ms) else -float(b.ms)


def distinguishable(a: Budget, b: Budget) -> bool:
    """True if two budgets differ by more than their seed-level uncertainty.

    Falls back to a strict inequality when no band is available, which is
    optimistic - prefer to pass cte_sd and osc_sd.
    """
    if _missing(a.ms) or _missing(b.ms):
        return _missing(a.ms) != _missing(b.ms)
    if any(_missing(v) for v in (a.lo_ms, a.hi_ms, b.lo_ms, b.hi_ms)):
        return float(a.ms) != float(b.ms)
    return a.hi_ms < b.lo_ms or b.hi_ms < a.lo_ms
