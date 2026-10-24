# State Delay Tolerance of Path Tracking Controllers: Effects of Tuning in the CARLA Simulator

Code for the paper by Efe Demir Degim, *The Ember Review*, October 2026:
https://emberreview.org/articles/state-delay-tolerance-path-tracking-controllers/

Does the ranking of path-tracking controllers by delay tolerance
survive a change of tuning?

Four controllers (PID, pure pursuit, Stanley, MPC) track a recorded
trajectory in CARLA while a controlled delay is injected
into the state they read. Each is run at three tunings, one of which matches
all four to equal nominal accuracy at zero delay, so the comparison starts
from a common point.

## Results

From the reported runs in `results/` (Table III of the paper). Matched tuning:
all four controllers set to the same zero-delay accuracy.

| Controller | CTE at d=0 (m) | Latency budget (ms) | Failure criterion |
|---|---|---|---|
| PID | 0.154 | 128 | oscillation |
| Pure pursuit | 0.147 | 122 | oscillation |
| Stanley | 0.150 | 36 | oscillation |
| MPC | 0.134 | diagnostic only | — |

Ordering tuning-invariant across the swept range: **no**. At conservative
tuning Stanley (89 ms) beats PID (44 ms); pure pursuit is first or tied first at
all three tunings. MPC is not ranked: its tracking improved with delay, which
the analysis flags as a lag-dominated tuning rather than a latency budget. All
three tunings are in `results/table4.txt`.

## What you need before running

- A CARLA server, with a client wheel that matches it. Prefer the wheel in
  `PythonAPI/carla/dist/` over the one on PyPI — the versions can't drift
  that way. The `cpXX` in the filename is the Python version to build your
  env around.
- A reference route you recorded yourself (see below). Nothing ships with
  one; the study is defined against your route.
- About 45 minutes for a full run at the shipped settings. The tuning grid
  and the MPC solver are most of it, and both print where the time went.

## Reproduce

Requires a CARLA server (tested against 0.9.15) listening on `:2000`.

Python 3.7 or newer. 3.7 is supported because CARLA 0.9.15 ships a `cp37`
client wheel, and pairing the env to that wheel is the least painful way to
match client to server; dependency floors are set so pip resolves to
3.7-compatible versions there and to current ones on a newer interpreter.

```bash
pip install -e .                   # or: pip install -r requirements.txt

# CARLA client: prefer the wheel shipped with your server over PyPI, so the
# client version cannot drift from the server version
pip install <CARLA>/PythonAPI/carla/dist/carla-0.9.15-cp37*.whl

# 1. record the reference route by driving Town03 manually
pip install pygame
python scripts/record_trajectory.py

# 2. everything else
bash scripts/run_experiment.sh
```

`run_experiment.sh` resolves the repo root from its own location and puts
`src/` on `PYTHONPATH` itself, so it works from any directory and whether or
not you ran `pip install -e .`. If you skip the editable install and want to
call the modules directly, `export PYTHONPATH=src` first.

To exercise the whole pipeline with no CARLA server attached:

```bash
python scripts/make_selftest_reference.py
bash scripts/run_experiment.sh --selftest
```

`run_experiment.sh` runs the accuracy-matched tuning search, then the frozen-gain
delay sweep, then the analysis. The sweep is 4 controllers x 3 tunings x 10
delays x 3 seeds = 360 runs, plus the tuning grid on top.

MPC dominates the wall clock: its iLQR solve runs once per simulation tick,
so one MPC run costs roughly 40x a geometric one. The sweep prints a cost
breakdown per controller when it finishes, and its ETA is tracked per
controller for that reason — a single global average is wrong by an order of
magnitude until the run is nearly over. Shrink `mpc_horizon` or raise
`ilqr_iters`' tolerance if you need it faster; `--no-camera` on the server and
`render: false` also help.

Outputs:

- `results/gains.json` — the frozen gains at all three tunings
- `results/runs.csv` — one row per run
- `results/per_condition.csv`, `results/budgets.csv`
- `results/table3.txt`, `table4.txt`, `table5.txt` — paste straight into the paper
- `figures/fig1_degradation.png`, `fig2_tradeoff.png`, `fig3_oscillation.png`

## Recording the route

`scripts/record_trajectory.py` opens a chase-camera view of the car.

| keyboard | gamepad | |
|---|---|---|
| arrows / WASD | left stick, R2, L2 | drive |
| space | R1 | handbrake |
| `R` | Cross | start / stop recording — **saves every time you stop** |
| `C` | Triangle | cycle camera: chase, bonnet, top-down |
| `Q` / Esc | Circle | quit |

A controller is picked up automatically if one is attached — a DualShock 4
works over USB with no setup, since the kernel's `hid-sony` driver handles it.
The keyboard keeps working at the same time, so a wrong button mapping never
traps you, and the pad only takes over while it's actually being moved.

Use one if you have one. Keyboard steering is a step input: full lock the
instant a key goes down. Those step changes land in the recorded route, and
the route is the target all four controllers are scored against, so reference
jitter appears as tracking error in every one of them. An analog stick gives
continuous steering, which is the entire reason for supporting it.

Mappings differ between drivers, SDL versions and USB vs Bluetooth. To check
yours:

```bash
python scripts/record_trajectory.py --gamepad-test     # no CARLA needed
```

That prints live axis and button indices; pass what you see as
`--axis-steer`, `--btn-record` and so on. `--pad-expo` controls how much finer
the response is near centre (1.0 is linear, default 1.6), `--pad-deadzone`
the dead region around it, and `--no-gamepad` ignores the pad entirely.

The HUD shows point count, speed, accumulated route length, and whether the
file has been written yet. Aim for roughly 800-1200 m with turns of several
different radii and at least a couple of sustained straights.

Drive smoothly. The recorded route is the target all four controllers are
scored against, so steering jitter in the reference appears as tracking error
in every one of them — which inflates the matched-tuning baseline and shrinks
every budget. Steering ramps rather than snapping to full lock for this
reason; take corners in one continuous input rather than a series of stabs.

`--no-camera` falls back to the small status window if rendering is too slow,
but then you're driving by the server window alone.

## Check the tuning before you read any table

The accuracy-matched tuning is the basis of every cross-controller claim in
the study. If the four controllers don't start from equal nominal error,
Table III compares controllers that were never on equal terms and its ranking
means nothing.

The tuner now says so. It reports how far each controller missed the target
and, when the search pins against a grid bound, which way to widen it:

```
[tune] stanley
  ! missed the 0.150 m target by +0.152 m (+101%)
  ! k is pinned at the top of its grid (3.0) and still needs to go higher; widen to 6
...
[tune] WARNING: nominal error spans 3.4x across controllers (target 1.0x)
```

A search whose optimum lies outside its own grid fails silently otherwise —
it still returns its best corner. Widen the bounds it names and re-run tuning
before trusting any cross-controller result.

All three tunings are picked from candidates that were actually run, by the
nominal error they achieved (`target x 2`, `target`, `target / 2`). The
earlier version scaled the dominant gain away from the matched point, which
walks outside the searched range whenever the matched tuning sits on a grid
edge — and then "aggressive" can land looser than "conservative", which is
how Table V produced negative exchange rates.

Tracking that IMPROVES with delay is flagged and must not be reported as a
budget. Delay can't improve tracking; when it appears to, the tuning is
lag-dominated and the injected delay is partly cancelling that lag.

Which direction to widen is measured, not assumed. Every grid candidate has
already been run, so whether raising a gain tracks tighter or looser is read
off the results. This matters: a hardcoded table had MPC backwards, which
made its "aggressive" tuning the looser one and flipped the sign of its
exchange rate in Table V.

Budgets that interpolate to less than one sweep step print as `<50` rather
than a number. Inside the first bracket there are only two points, so any
figure there is extrapolation, not measurement — if you need resolution that
fine, shorten the sweep step (which means raising the simulation rate, since
delay is quantised to whole ticks).

## When something won't run

```bash
python -m latency.sweep doctor
```

One run per controller at base gains with no delay, reporting whether each
finished, how far along the route it got, its error and oscillation, and a
verdict. This answers the question the tuner can't: is the route usable at
all, and does each controller actually move the car?

The route is trimmed and checked when it loads. Points before the car starts
moving and after it stops are dropped, and any interior near-stop is floored
to a crawl — a route recorded by pressing R while parked begins at `v = 0`,
which makes the speed loop command zero throttle forever, and every candidate
tuning then fails for a reason that has nothing to do with tuning.

## How the delay is injected

`src/latency/delay.py`. Every state the simulator produces goes into a FIFO;
the controller reads the entry `n = round(d / dt)` steps old. No controller
source is touched, so all four see identically-constructed inputs.

`DelayBuffer.from_delay` refuses any delay that isn't a whole multiple of the
timestep rather than rounding silently — a silent round would put a systematic
error on the x-axis of every figure.

## Testing without CARLA

`--selftest` swaps the CARLA runner for a kinematic bicycle stand-in so the
tuner, sweep and analysis can be debugged offline:

```bash
python scripts/make_selftest_reference.py
python -m latency.sweep tune  --config configs/selftest.yaml --gains results/selftest_gains.json --selftest
python -m latency.sweep sweep --config configs/selftest.yaml --gains results/selftest_gains.json --out results/selftest_runs.csv --selftest
python -m latency.analyze --runs results/selftest_runs.csv --allow-selftest
```

**Self-test output isn't data.** It's a bicycle model with no tyre model, no
suspension, no actuator dynamics. Every row it writes carries `selftest=1`,
`analyze.py` refuses to build tables from it without `--allow-selftest`, and
any figure it produces is watermarked. Nothing from a self-test run belongs in
the paper.

Unit tests (no CARLA needed):

```bash
pytest -q          # pythonpath is set in pyproject.toml
```

These cover the parts that can silently produce wrong numbers: the delay
buffer indexing, distance-to-segment (not distance-to-vertex) cross-track
error, the oscillation detrending, the budget rule, and a steering-sign check
on each controller.

## Layout

```
src/latency/
  delay.py        DelayBuffer, JitterBuffer   <- the experimental manipulation
  controllers.py  PID, pure pursuit, Stanley, MPC (self-contained iLQR)
  metrics.py      cross-track error, oscillation, the latency-budget rule
  runners.py      CarlaRunner, KinematicRunner
  sweep.py        accuracy-matched tuning, then the frozen-gain sweep
  analyze.py      runs.csv -> tables 3-5 and figures 1-3
configs/sweep.yaml   every experimental parameter
```

## Choices worth arguing with

- **The oscillation cap during tuning is calibrated, not fixed.** Its job is
  to reject tunings that are already marginally stable before any delay is
  injected. An absolute number is really a guess about the route — a short
  twisty route needs more steering activity than a long open one, and the
  detrending can't fully separate a two-second corner from a one-hertz
  weave. `osc_cap_rad: auto` sets the cap at `osc_cap_factor` times the
  quietest tuning that completed the route, which asks the right question:
  does this tuning oscillate much more than this route requires? Set a number
  instead if you want a cap that's comparable across routes.
- **The failure thresholds.** The budget fires at 0.50 m mean cross-track
  error or at a doubling of steering oscillation. Both are defensible, neither
  is unique. Figure 1 reports the full degradation curves so a reader who
  prefers a different threshold can read off their own budget.
- **Budgets are interpolated, not binned.** Delay is quantised to whole
  simulation steps, so at the 50 Hz used here the resolution is 20 ms. Reporting the
  first failing sweep value would quantise every budget to that step, and
  controllers one step apart couldn't be ranked at all. `latency_budget`
  instead interpolates where the degradation curve crosses the threshold
  between the two bracketing sweep points. `--no-interpolate` restores the
  binned value, and both are reported side by side in Table III.
- **Rankings carry a resolvability check.** Seed-level scatter is propagated
  into a `+/-1 SD` band on each budget. Any two controllers whose bands
  overlap are printed as unresolved and must not be ranked against each other
  — a rank change between an unresolved pair isn't evidence of a swap. If
  everything comes out unresolved, narrow the sweep step, add seeds, or report
  the pair as tied.
- **Gains are frozen at zero delay.** Re-tuning at every delay level would
  measure tuning effort rather than tolerance. See `sweep.py`: the sweep has no
  access to the tuner.
- **Ground-truth state.** Real stale information is also noisy. Using ground
  truth isolates the delay variable, which makes these budgets upper bounds.
- **One car, reused for every run.** The vehicle is spawned once and reset to
  the start between runs rather than destroyed and respawned 300+ times, which
  is slow and eventually makes CARLA refuse a spawn partway through a sweep.
  Spawn height comes from the map's road surface at the route's first point,
  not a constant — Town03 has elevated sections where a fixed z lands inside
  the geometry.

## License

MIT.

## Acknowledgement

Controller implementations build on the framework from the
'Techniques for Self-Driving Cars' course assignments.


## Known limitations of this implementation

Worth reading before interpreting output, and worth repeating in the paper's
limitations section.

- **The MPC can select a degenerate tuning.** On routes with a wide speed
  range its position weight tends to settle at the bottom of the grid, and in
  that regime the loop is lag-dominated: tracking error *falls* as delay
  grows, out to a couple of hundred ms. That isn't a latency tolerance, and
  the analysis flags it rather than reporting it. The cause is the unicycle
  model's `steer = atan(L*omega/v)` conversion saturating at low speed, which
  is a property of the formulation, not a tuning mistake.
- **Stanley has the same low-speed problem.** Its cross-track term scales as
  `1/v`, so below roughly 15 km/h it demands more lock than the car has. PID
  and pure pursuit have no `1/v` term and don't suffer from it. A route with
  a narrow speed band is the fix; no gain value is.
- **Delay resolution is tied to the simulation rate.** Delay is quantised to
  whole ticks, so `dt` sets the finest budget you can measure. At 50 Hz
  that's 20 ms. Anything finer needs a higher rate and proportionally more
  wall clock.
- **Ground-truth state.** Real stale information is also noisy and biased, so
  these budgets are upper bounds on what the same controllers would tolerate
  with a real estimator in the loop.
- **One route, one town, one vehicle.** Twelve controller-tuning pairs on a
  single route is twelve data points about a large space.
