"""Accuracy-matched tuning, then the frozen-gain delay sweep.

Order matters and is enforced here: gains are found at zero delay, written
to disk, and only then read back for the sweep. The sweep has no access to
the tuner, so it can't silently re-tune.
"""

import argparse
import csv
import json
import math
import os
import time

import numpy as np
import yaml

from .metrics import summarise_run
from .runners import CarlaRunner, KinematicRunner, SELFTEST_STAMP


# --------------------------------------------------------------------------


def load_reference(path, v_min=0.5, verbose=True):
    """Read the manually recorded trajectory: x,y,v per line.

    A route recorded by pressing R while parked begins with v = 0. The speed
    loop then sees zero error, commands zero throttle, and the car never
    moves - so every candidate tuning is rejected as "didn't complete" and
    the real cause is invisible. Leading and trailing stationary points are
    therefore trimmed, and any interior near-stop is floored to v_min so a
    red light you waited at can't deadlock the run either.
    """
    if not os.path.exists(path):
        raise SystemExit(
            "Reference trajectory not found: {}\n"
            "  Record one by driving Town03 manually:\n"
            "      python scripts/record_trajectory.py --out {}\n"
            "  or point 'reference:' in your config at an existing file.\n"
            "  For a synthetic route to test the pipeline:\n"
            "      python scripts/make_selftest_reference.py  (then --selftest)"
            .format(path, path))
    rows = np.loadtxt(path, delimiter=",")
    if rows.ndim != 2 or rows.shape[1] < 3:
        raise ValueError("{}: expected 'x,y,v' per line".format(path))
    xy, v = rows[:, :2].astype(float), rows[:, 2].astype(float)

    moving = np.flatnonzero(v > v_min)
    if len(moving) == 0:
        raise SystemExit(
            "Every point in {} has speed <= {} m/s - the car never moved.\n"
            "  Press R to start recording, then drive, then press R again."
            .format(path, v_min))

    lo, hi = int(moving[0]), int(moving[-1]) + 1
    trimmed_head, trimmed_tail = lo, len(v) - hi
    xy, v = xy[lo:hi], v[lo:hi]
    if len(xy) < 20:
        raise SystemExit(
            "Only {} moving points in {}. Record a longer route - aim for "
            "800-1200 m with several turns.".format(len(xy), path))

    interior_stops = int(np.sum(v < v_min))
    v = np.maximum(v, v_min)

    if verbose:
        if trimmed_head or trimmed_tail:
            print("[reference] trimmed {} stationary points from the start "
                  "and {} from the end".format(trimmed_head, trimmed_tail))
        if interior_stops:
            print("[reference] {} interior points were below {} m/s and were "
                  "floored to it (the car can't be asked to stop mid-route)"
                  .format(interior_stops, v_min))
        describe_reference(xy, v, path)
    return xy, v


def describe_reference(xy, v, path=""):
    """Print route stats, and warn about shapes that distort the study.

    The recorded route is the target all four controllers are scored against,
    so its character sets the baseline every budget is measured from.
    """
    length = float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))
    duration = route_duration_s(xy, v)
    heading = np.unwrap(np.arctan2(*np.diff(xy, axis=0).T[::-1]))
    turning = float(np.sum(np.abs(np.diff(heading))))

    print("[reference] {}: {} points, {:.0f} m, {:.0f} s, {:.1f}-{:.1f} km/h, "
          "{:.0f} deg of turning".format(
              path, len(xy), length, duration,
              v.min() * 3.6, v.max() * 3.6, math.degrees(turning)))

    warn = []
    if length < 500:
        warn.append("route is {:.0f} m; 800-1200 m gives the sweep more to "
                    "work with, and short routes make every budget noisier"
                    .format(length))
    if turning < math.radians(180):
        warn.append("only {:.0f} deg of total turning; delay hurts most in "
                    "corners, so a mostly straight route reports a "
                    "flatteringly large budget for everything"
                    .format(math.degrees(turning)))
    if v.max() > 16.7:
        warn.append("top speed {:.0f} km/h; fast sections are hard to track "
                    "and will compress all four budgets"
                    .format(v.max() * 3.6))
    if v.max() / max(v.min(), 0.1) > 8:
        warn.append("speed ranges {:.0f}x between slowest and fastest; a "
                    "steadier profile makes the matched tuning easier to hit"
                    .format(v.max() / max(v.min(), 0.1)))
    for w in warn:
        print("[reference] note: {}".format(w))
    return dict(length_m=length, duration_s=duration, turning_rad=turning)


def route_duration_s(xy, v, v_floor=0.5):
    """How long the route takes to drive, from its own geometry and speeds.

    Independent of both the recording rate and the simulation rate, which a
    point count isn't: one reference point equals one tick only when the
    two rates happen to match.
    """
    seg = np.hypot(*np.diff(np.asarray(xy, float), axis=0).T)
    vmid = np.maximum((np.asarray(v, float)[:-1] + np.asarray(v, float)[1:])
                      / 2.0, v_floor)
    return float(np.sum(seg / vmid))


def resolve_max_steps(cfg, ref_xy, ref_v=None, verbose=True):
    """Size the step limit to the route instead of guessing a constant.

    A fixed 4000 against a 1045-tick route means any candidate that never
    finishes burns ~4x a normal run, and most grid candidates never finish.
    That, not the grid size, is what makes tuning slow.
    """
    want = cfg["params"].get("max_steps", "auto")
    if want == "auto" or want is None:
        dt = float(cfg["params"]["dt"])
        if ref_v is not None:
            seconds = route_duration_s(ref_xy, ref_v)
            steps = int(seconds * 2.5 / dt) + 200
            note = "route takes {:.0f} s to drive, {:.0f} ticks at {:.0f} Hz".format(
                seconds, seconds / dt, 1.0 / dt)
        else:                       # no speeds available: fall back to points
            steps = int(len(ref_xy) * 2.5) + 200
            note = "{} reference points".format(len(ref_xy))
        cfg["params"]["max_steps"] = steps
        if verbose:
            print("[config] max_steps auto -> {} ({})".format(steps, note))
    return cfg["params"]["max_steps"]


def make_runner(cfg, selftest, ref_xy=None):
    if selftest:
        return KinematicRunner(cfg["params"]), True
    runner = CarlaRunner(cfg["params"], host=cfg.get("host", "127.0.0.1"),
                         port=cfg.get("port", 2000),
                         render=cfg.get("render", False))
    if ref_xy is not None:
        runner.validate_reference(ref_xy)
    return runner, False


# --------------------------------------------------------------------------
# Tuning


# Which way each tunable moves nominal accuracy: +1 means raising it tracks
# more tightly, -1 means lowering it does. Used to tell you which end of a
# grid to extend when the search pins against a boundary.
GAIN_DIRECTION = {
    "pid":          {"lat_kp": +1, "lat_kd": +1},
    "stanley":      {"k": +1, "ks": -1},
    "pure_pursuit": {"kv": -1, "l0": -1},   # shorter lookahead tracks tighter
    "mpc":          {"w_pos": +1, "w_ctrl": -1},
}


def measure_gain_directions(results, keys, fallback=None):
    """Work out empirically whether raising each gain tracks tighter.

    The hardcoded table below is a guess that was wrong for MPC: raising
    w_pos made tracking WORSE on this route, because the optimum sat below
    the grid. Since every grid candidate has already been run, the direction
    can simply be measured: +1 if raising the gain lowered cross-track
    error, -1 if it raised it.

    This matters beyond bookkeeping. "aggressive" has to mean tighter
    tracking by definition, or the exchange rate in Table V compares a
    looser tuning against a tighter one with the sign flipped.
    """
    out = {}
    for key in keys:
        xs = [g[key] for g, _ in results if key in g]
        ys = [su["mean_cte"] for g, su in results if key in g]
        direction = 0
        if len(set(xs)) > 1:
            cx, cy = np.asarray(xs, float), np.asarray(ys, float)
            if cx.std() > 1e-12 and cy.std() > 1e-12:
                corr = float(np.corrcoef(cx, cy)[0, 1])
                if abs(corr) > 0.15:
                    direction = -1 if corr > 0 else +1   # cte down = tighter
        if direction == 0 and fallback:
            direction = fallback.get(key, 0)
        out[key] = direction
    return out


def check_matched_quality(controller, gains, summary, grid, target, tol,
                          directions=None):
    """Warn when the matched tuning didn't actually match.

    The accuracy-matched tuning is the whole basis of the cross-controller
    comparison: if the four controllers don't start from equal nominal
    error, Table III compares controllers that were never on equal terms and
    the ranking means nothing. A grid whose optimum lies outside its own
    bounds fails silently, because the search still returns its best corner.
    """
    notes = []
    miss = summary["mean_cte"] - target
    if abs(miss) > tol:
        notes.append(
            "missed the {:.3f} m target by {:+.3f} m ({:+.0f}%)".format(
                target, miss, 100.0 * miss / target))

    # improve = +1 when the controller must track more tightly to reach the
    # target, -1 when it must loosen off
    improve = +1 if miss > 0 else -1
    for key in grid:
        if key not in gains:
            continue
        lo, hi = grid[key][0], grid[key][1]
        direction = (directions or {}).get(key)
        if direction is None:
            direction = GAIN_DIRECTION.get(controller, {}).get(key, 0)
        if direction == 0:
            continue
        at_lo, at_hi = abs(gains[key] - lo) < 1e-9, abs(gains[key] - hi) < 1e-9
        if not (at_lo or at_hi):
            continue
        # which way this particular gain has to move. For pure pursuit a
        # SHORTER lookahead tracks tighter, so its blocked bound is the
        # bottom one - the opposite of every other controller here.
        move = improve * direction
        blocked = (move > 0 and at_hi) or (move < 0 and at_lo)
        if blocked:
            new = hi * 2.0 if move > 0 else lo / 2.0
            notes.append(
                "{} is pinned at the {} of its grid ({}) and still needs to "
                "go {}; widen to {:.3g}".format(
                    key, "top" if at_hi else "bottom", gains[key],
                    "higher" if move > 0 else "lower", new))
        else:
            notes.append("{}={} sits on a grid edge".format(key, gains[key]))
    return notes


DOMINANT_GAINS = {
    "pid": ["lat_kp"],
    "stanley": ["k"],
    "pure_pursuit": ["kv", "l0"],
    "mpc": ["w_pos"],
}


def _scale_dominant(gains, controller, factor, directions=None):
    """Scale the dominant gain so that factor > 1 always means TIGHTER.

    Which arithmetic achieves that depends on the gain: a shorter pure
    pursuit lookahead tracks tighter, so its parameters must be divided
    where PID's proportional gain is multiplied. The direction is taken
    from the tuning run's own measurements where available.
    """
    g = dict(gains)
    for key in DOMINANT_GAINS.get(controller, []):
        if key not in g:
            continue
        d = (directions or {}).get(key)
        if d is None or d == 0:
            d = GAIN_DIRECTION.get(controller, {}).get(key, +1)
        g[key] = g[key] * factor if d > 0 else g[key] / factor
    return g


def tune_matched(runner, cfg, controller, ref_xy, ref_v, verbose=True):
    """Coarse grid search at zero delay for the target nominal accuracy.

    Every candidate is run first and filtered afterwards, because the
    oscillation cap is calibrated from the results rather than fixed in
    advance. An absolute cap is a guess about the route: a short twisty route
    needs more steering activity than a long open one, and the detrending
    can't fully separate a two-second corner from a one-hertz weave. A cap
    set relative to the quietest tuning that works asks the right question -
    "does this tuning oscillate much more than this route requires?" - rather
    than comparing against a number carried over from somewhere else.
    """
    p = cfg["params"]
    target = cfg["tuning"]["target_cte_m"]
    cap_cfg = cfg["tuning"].get("osc_cap_rad", "auto")
    cap_factor = float(cfg["tuning"].get("osc_cap_factor", 1.5))
    grid = cfg["tuning"]["grid"][controller]
    base = cfg["tuning"]["base_gains"][controller]

    keys = list(grid.keys())
    axes = [np.linspace(*grid[k]) for k in keys]
    combos = np.array(np.meshgrid(*axes)).T.reshape(-1, len(keys))

    results = []
    reasons = {"incomplete": 0, "nan": 0, "usable": 0}
    abort_cte = target * float(cfg["tuning"].get("abort_cte_multiple", 10.0))
    t_start = time.time()
    n_aborted, n_ticks = 0, 0
    for n_done, combo in enumerate(combos, 1):
        gains = dict(base)
        gains.update({k: float(val) for k, val in zip(keys, combo)})
        log = runner.run(controller, gains, ref_xy, ref_v, 0.0, seed=0,
                         abort_cte=abort_cte)
        n_aborted += int(log.get("aborted", False))
        n_ticks += len(log["x"])

        s = summarise_run(log, ref_xy, p["dt"], p["warmup_s"])
        s["progress"] = log.get("progress", float("nan"))
        if not s["completed"]:
            reasons["incomplete"] += 1
        elif math.isnan(s["mean_cte"]) or math.isnan(s["osc_std"]):
            reasons["nan"] += 1
        else:
            reasons["usable"] += 1
            results.append((gains, s))

        if verbose and (n_done % 10 == 0 or n_done == len(combos)):
            el = time.time() - t_start
            print("      {}/{} candidates, {} usable, {:.0f}s elapsed, "
                  "~{:.0f}s left".format(
                      n_done, len(combos), reasons["usable"], el,
                      el / n_done * (len(combos) - n_done)), flush=True)

    if verbose:
        el = time.time() - t_start
        print("      {} candidates in {:.0f}s; {} usable, {} cut short early"
              .format(len(combos), el, reasons["usable"], n_aborted))

    if results:
        oscs = np.array([r[1]["osc_std"] for r in results], float)
        quietest = float(oscs.min())
        typical = float(np.median(oscs))
        if cap_cfg == "auto":
            # anchored to the MEDIAN completing candidate, not the minimum.
            # The quietest candidate is the sluggiest one, so a small multiple
            # of it admits only near-sluggish tunings - which collapsed all
            # three tuning levels onto the same point on a real route. The
            # median is what driving this route normally costs in steering.
            cap = max(typical * cap_factor, 1e-6)
            cap_note = ("auto: {:.4f} = {:.1f}x the median completing tuning "
                        "({:.4f}; quietest {:.4f})".format(
                            cap, cap_factor, typical, quietest))
        else:
            cap = float(cap_cfg)
            cap_note = "fixed at {:.4f} from config".format(cap)
        passing = [(g, s) for g, s in results if s["osc_std"] <= cap]
    else:
        quietest, cap, cap_note, passing = float("nan"), 0.0, "n/a", []

    if passing:
        # All three tunings are chosen from candidates that were actually
        # run, by the nominal accuracy they achieved. The previous approach
        # scaled the dominant gain away from the matched point, which walks
        # outside the searched range whenever the matched tuning sits on a
        # grid edge - and then "aggressive" can be the LOOSER setting, which
        # is how Table V ended up with negative exchange rates.
        f = float(cfg["tuning"].get("scale_factor", 2.0))
        # The conservative tuning must still be INSIDE the failure threshold
        # at zero delay, or its budget is zero by construction and Table V
        # gets a negative exchange rate. Leave it clear room to degrade.
        fail_cte = float(cfg["tuning"].get("failure_cte_m", 0.50))
        cons_cap = fail_cte * float(cfg["tuning"].get("conservative_headroom",
                                                      0.6))
        levels = {"conservative": min(target * f, cons_cap),
                  "matched": target,
                  "aggressive": target / f}
        usable = [gs for gs in passing if gs[1]["mean_cte"] <= cons_cap]
        if usable:
            passing = usable
        elif verbose:
            print("       ! no tuning tracks within {:.2f} m (the conservative "
                  "headroom); every budget will read as an immediate "
                  "failure".format(cons_cap))
        chosen = {}
        for name, want in levels.items():
            g, su = min(passing, key=lambda gs: abs(gs[1]["mean_cte"] - want))
            chosen[name] = (g, su)

        # If the target levels land on the same candidate, the grid simply has
        # nothing at those accuracies. Fall back to the extremes of what DID
        # pass, so the three tunings at least span the measured range instead
        # of being one tuning under three labels.
        picked = [id(chosen[n][0]) for n in ("conservative", "matched",
                                             "aggressive")]
        if len(set(picked)) < 3:
            by_cte = sorted(passing, key=lambda gs: gs[1]["mean_cte"])
            if len(by_cte) >= 3:
                chosen["aggressive"] = by_cte[0]
                # the loosest tuning that's still inside the headroom, not
                # simply the loosest that completed the route
                chosen["conservative"] = by_cte[-1]
                mid = min(by_cte[1:-1],
                          key=lambda gs: abs(gs[1]["mean_cte"] - target))
                chosen["matched"] = mid
                if verbose:
                    print("       (target accuracy levels collapsed onto one "
                          "candidate; using the spread of what passed)")
        directions = measure_gain_directions(
            results, keys, GAIN_DIRECTION.get(controller))
        if verbose:
            su = chosen["matched"][1]
            print("    {}: cte={:.3f} osc={:.4f} ({}) {}".format(
                controller, su["mean_cte"], su["osc_std"], cap_note,
                chosen["matched"][0]))
            print("       tunings by measured nominal cte: " + ", ".join(
                "{} {:.3f}".format(n, chosen[n][1]["mean_cte"])
                for n in ("conservative", "matched", "aggressive")))
        return chosen, directions

    n = len(combos)
    lines = [
        "No tuning for {} passed, out of {} candidates.".format(controller, n),
        "  didn't finish the route : {}".format(reasons["incomplete"]),
        "  produced no valid metrics: {}".format(reasons["nan"]),
        "  completed but too noisy  : {}".format(
            reasons["usable"] - len(passing)),
        "  oscillation cap          : {}".format(cap_note),
    ]
    if results:
        best_prog = max(r[1]["progress"] for r in results)
        lines += [
            "  best progress along the route   : {:.0%}".format(best_prog),
            "  best mean cross-track error     : {:.3f} m (target {})".format(
                min(r[1]["mean_cte"] for r in results), target),
            "  quietest completing tuning      : {:.4f} rad".format(quietest),
            "",
        ]
        if cap_cfg != "auto":
            lines.append("  The cap is fixed in your config and nothing met it. "
                         "Set tuning.osc_cap_rad to 'auto' to calibrate it from "
                         "this route, or raise it above {:.4f}.".format(quietest))
        else:
            lines.append("  Raise tuning.osc_cap_factor above {:.1f}, or shift "
                         "the grid toward lower gains.".format(cap_factor))
    else:
        lines += ["", "  Nothing completed the route. Run the diagnostic:",
                  "      python -m latency.sweep doctor"]
    raise RuntimeError("\n".join(lines))


def cmd_tune(cfg, args):
    ref_xy, ref_v = load_reference(cfg["reference"])
    resolve_max_steps(cfg, ref_xy, ref_v)
    runner, is_selftest = make_runner(cfg, args.selftest, ref_xy)
    out = {"_selftest": is_selftest,
           "_stamp": SELFTEST_STAMP if is_selftest else None,
           "target_cte_m": cfg["tuning"]["target_cte_m"], "controllers": {}}
    tol = float(cfg["tuning"].get("target_tolerance_m", 0.05))
    quality = {}

    for c in cfg["controllers"]:
        print("[tune] {}".format(c))
        chosen, directions = tune_matched(runner, cfg, c, ref_xy, ref_v)
        matched, summary = chosen["matched"]
        entry = {name: chosen[name][0] for name in chosen}
        entry["matched_nominal_cte"] = summary["mean_cte"]
        entry["matched_nominal_osc"] = summary["osc_std"]
        entry["measured_gain_directions"] = directions
        entry["nominal_cte_by_tuning"] = {
            n: chosen[n][1]["mean_cte"] for n in chosen}
        out["controllers"][c] = entry

        # all three must be distinct and correctly ordered, or the three
        # "tunings" are one tuning wearing three labels
        by_cte = entry["nominal_cte_by_tuning"]
        if not (by_cte["aggressive"] < by_cte["matched"] < by_cte["conservative"]):
            print("  ! the three tunings aren't ordered by accuracy "
                  "({:.3f} / {:.3f} / {:.3f}) - the grid doesn't span enough "
                  "range to separate them".format(
                      by_cte["aggressive"], by_cte["matched"],
                      by_cte["conservative"]))
        elif by_cte["conservative"] / max(by_cte["aggressive"], 1e-9) < 1.5:
            print("  ! conservative and aggressive differ by only {:.2f}x in "
                  "nominal error; the tuning manipulation barely took".format(
                      by_cte["conservative"] / max(by_cte["aggressive"], 1e-9)))
        notes = check_matched_quality(c, matched, summary,
                                      cfg["tuning"]["grid"][c],
                                      cfg["tuning"]["target_cte_m"], tol,
                                      directions)
        quality[c] = (summary["mean_cte"], notes)
        for n in notes:
            print("  ! {}".format(n))
        print("  -> matched cte {:.3f} m".format(summary["mean_cte"]))

    os.makedirs(os.path.dirname(args.gains) or ".", exist_ok=True)
    with open(args.gains, "w") as fh:
        json.dump(out, fh, indent=2)
    print("[tune] wrote {}".format(args.gains))

    ctes = [q[0] for q in quality.values()]
    spread = max(ctes) / max(min(ctes), 1e-9)
    print("\n[tune] matched nominal error: " + ", ".join(
        "{} {:.3f}".format(c, q[0]) for c, q in quality.items()))
    if spread > 1.5:
        print("[tune] WARNING: the matched tuning didn't match. Nominal error "
              "spans {:.2f}x across controllers (target 1.00x).".format(spread))
        print("        Table III then compares controllers that didn't start "
              "equal, so its ranking isn't the fair")
        print("        comparison the study is built on. Widen the grids as "
              "noted above and re-run tuning.")
    elif any(q[1] for q in quality.values()):
        print("[tune] matched to within {:.2f}x across controllers - good "
              "enough for Table III.".format(spread))
        print("        Some gains still sat on a grid edge (noted above). That "
              "limits the tuning RANGE, so Table IV's")
        print("        invariance test is weaker than it could be, but the "
              "matched comparison itself is sound.")
    else:
        print("[tune] matched to within {:.2f}x across controllers, no gain on "
              "a grid edge.".format(spread))


# --------------------------------------------------------------------------
# Sweep

FIELDS = ["controller", "tuning", "delay_ms", "seed", "completed", "mean_cte",
          "max_cte", "mean_v_err", "osc_std", "dom_freq", "n_samples",
          "mean_solve_ms", "selftest"]


def cmd_sweep(cfg, args):
    ref_xy, ref_v = load_reference(cfg["reference"])
    resolve_max_steps(cfg, ref_xy, ref_v)
    runner, is_selftest = make_runner(cfg, args.selftest, ref_xy)
    with open(args.gains) as fh:
        gainfile = json.load(fh)
    if gainfile.get("_selftest") and not args.selftest:
        raise SystemExit("gains were tuned in selftest mode; re-tune against CARLA")

    p = cfg["params"]
    delays = cfg["sweep"]["delays_ms"]
    seeds = list(range(cfg["sweep"]["seeds"]))
    tunings = cfg["sweep"]["tunings"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new = not os.path.exists(args.out)
    if not new:
        # appending selftest rows to real ones (or the reverse) silently
        # poisons the file; analyze then refuses and the real runs look lost
        with open(args.out) as fh:
            existing = [ln.rstrip("\n").rsplit(",", 1)[-1]
                        for ln in fh.readlines()[1:] if ln.strip()]
        if existing and any(e != str(int(is_selftest)) for e in existing):
            raise SystemExit(
                "{} already holds {} rows and this run writes {} rows.\n"
                "  These must not share a file. Write somewhere else:\n"
                "      --out results/{}_runs.csv\n"
                "  or move the existing file aside."
                .format(args.out,
                        "selftest" if is_selftest else "CARLA",
                        "selftest" if is_selftest else "CARLA",
                        "selftest" if is_selftest else "carla"))
    per_ctrl = len(tunings) * len(delays) * len(seeds)
    total = len(cfg["controllers"]) * per_ctrl
    done = 0
    t_start = time.time()
    # MPC costs roughly 40x a geometric controller, so a single global
    # average produces an ETA that's wrong by an order of magnitude until
    # the sweep is nearly over. Track each controller separately.
    spent = {c: 0.0 for c in cfg["controllers"]}
    ran = {c: 0 for c in cfg["controllers"]}
    left = {c: per_ctrl for c in cfg["controllers"]}

    def _eta():
        known = [c for c in ran if ran[c]]
        if not known:
            return float("nan")
        fallback = sum(spent[c] / ran[c] for c in known) / len(known)
        return sum(left[c] * (spent[c] / ran[c] if ran[c] else fallback)
                   for c in left)

    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for c in cfg["controllers"]:
            for tname in tunings:
                gains = gainfile["controllers"][c][tname]
                for d_ms in delays:
                    for seed in seeds:
                        t_run = time.time()
                        log = runner.run(c, gains, ref_xy, ref_v,
                                         d_ms / 1000.0, seed=seed)
                        spent[c] += time.time() - t_run
                        ran[c] += 1
                        left[c] -= 1
                        s = summarise_run(log, ref_xy, p["dt"], p["warmup_s"])
                        w.writerow({
                            "controller": c, "tuning": tname, "delay_ms": d_ms,
                            "seed": seed, **{k: s[k] for k in
                                             ("completed", "mean_cte", "max_cte",
                                              "mean_v_err", "osc_std", "dom_freq",
                                              "n_samples")},
                            "mean_solve_ms": float(np.mean(log["solve_ms"])),
                            "selftest": int(is_selftest),
                        })
                        fh.flush()
                        done += 1
                        print("  [{}/{}] {}/{} d={}ms seed={} cte={:.3f} "
                              "eta={:.1f}min".format(
                                  done, total, c, tname, d_ms, seed,
                                  s["mean_cte"], _eta() / 60.0), flush=True)
    if hasattr(runner, "close"):
        runner.close()
    print("[sweep] wrote {}".format(args.out))
    print("[sweep] took {:.1f} min total. Cost per run by controller:".format(
        (time.time() - t_start) / 60.0))
    for c in sorted(ran, key=lambda k: -(spent[k] / max(ran[k], 1))):
        if ran[c]:
            print("    {:<14}{:6.2f} s x {:3d} runs = {:5.1f} min".format(
                c, spent[c] / ran[c], ran[c], spent[c] / 60.0))


# --------------------------------------------------------------------------


def cmd_doctor(cfg, args):
    """One run per controller at base gains, zero delay, with a verdict.

    Answers the question the tuner can't: is the route usable at all, and
    does each controller move the car?
    """
    ref_xy, ref_v = load_reference(cfg["reference"])
    resolve_max_steps(cfg, ref_xy, ref_v)
    runner, is_selftest = make_runner(cfg, args.selftest, ref_xy)
    p = cfg["params"]
    print()
    print("{:<14}{:>10}{:>11}{:>10}{:>10}{:>11}  {}".format(
        "controller", "finished", "progress", "cte(m)", "osc(rad)",
        "mean v", "verdict"))
    for c in cfg["controllers"]:
        gains = cfg["tuning"]["base_gains"][c]
        log = runner.run(c, gains, ref_xy, ref_v, 0.0, seed=0)
        s = summarise_run(log, ref_xy, p["dt"], p["warmup_s"])
        prog = log.get("progress", 0.0)
        mean_v = float(np.mean(log["v"])) if log["v"] else 0.0
        if prog < 0.05:
            verdict = "never moved - check the speed profile"
        elif not s["completed"]:
            verdict = "left the route or ran out of steps"
        elif s["osc_std"] > cfg["tuning"]["osc_cap_rad"]:
            verdict = "completes but oscillates at base gains"
        else:
            verdict = "ok"
        print("{:<14}{:>10}{:>10.0%}{:>10.3f}{:>10.4f}{:>9.1f} km/h  {}".format(
            c, str(s["completed"]), prog, s["mean_cte"], s["osc_std"],
            mean_v * 3.6, verdict))
    print()
    print("Reference speed: {:.1f}-{:.1f} km/h, {} points".format(
        ref_v.min() * 3.6, ref_v.max() * 3.6, len(ref_v)))
    if hasattr(runner, "close"):
        runner.close()


def main():
    ap = argparse.ArgumentParser(description="latency tolerance experiment")
    ap.add_argument("command", choices=["tune", "sweep", "doctor"])
    ap.add_argument("--config", default="configs/sweep.yaml")
    ap.add_argument("--gains", default="results/gains.json")
    ap.add_argument("--out", default="results/runs.csv")
    ap.add_argument("--selftest", action="store_true",
                    help="use the kinematic stand-in; output is NOT real data")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.selftest:
        print(f"*** {SELFTEST_STAMP}: kinematic stand-in, not CARLA ***")
    {"tune": cmd_tune, "sweep": cmd_sweep,
     "doctor": cmd_doctor}[args.command](cfg, args)


if __name__ == "__main__":
    main()
