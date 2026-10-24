"""Turn results/runs.csv into the paper's Tables III-V and Figures 1-3.

Emits Word-pasteable text and LaTeX side by side, so the numbers reaching
the paper are never retyped by hand.

Refuses to run on selftest data unless --allow-selftest is passed, and
watermarks every figure it produces from such data.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import latency_budget, distinguishable, budget_sort_key

TUNINGS = ["conservative", "matched", "aggressive"]
NICE = {"pid": "PID", "pure_pursuit": "Pure pursuit",
        "stanley": "Stanley", "mpc": "MPC"}


def load(path, allow_selftest):
    df = pd.read_csv(path)
    if "selftest" in df.columns and df["selftest"].any():
        if not allow_selftest:
            raise SystemExit(
                "runs.csv contains selftest rows (kinematic stand-in, not "
                "CARLA). These are for debugging the pipeline only and must "
                "not reach the paper. Re-run the sweep against CARLA, or "
                "pass --allow-selftest to inspect the pipeline."
            )
        print("*** WARNING: analysing SELFTEST data. Not real results. ***")
    return df


def per_condition(df):
    """Mean over seeds for each (controller, tuning, delay) cell."""
    g = df.groupby(["controller", "tuning", "delay_ms"], as_index=False).agg(
        mean_cte=("mean_cte", "mean"),
        cte_sd=("mean_cte", "std"),
        max_cte=("max_cte", "mean"),
        mean_v_err=("mean_v_err", "mean"),
        osc_std=("osc_std", "mean"),
        osc_sd=("osc_std", "std"),
        dom_freq=("dom_freq", "mean"),
        completion=("completed", "mean"),
        solve_ms=("mean_solve_ms", "mean"),
        n=("seed", "count"),
    )
    return g.sort_values(["controller", "tuning", "delay_ms"])


def budgets(cond, cte_threshold, osc_multiple, interpolate=True,
            osc_floor=0.0):
    rows = []
    for (c, t), sub in cond.groupby(["controller", "tuning"]):
        sub = sub.sort_values("delay_ms")
        step = (float(np.min(np.diff(sub["delay_ms"].values)))
                if len(sub) > 1 else None)
        b = latency_budget(sub["delay_ms"].values, sub["mean_cte"].values,
                           sub["osc_std"].values, cte_threshold, osc_multiple,
                           cte_sd=sub["cte_sd"].values,
                           osc_sd=sub["osc_sd"].values, interpolate=interpolate,
                           resolution_ms=step, osc_floor_rad=osc_floor)
        nominal = sub.iloc[0]
        rows.append(dict(controller=c, tuning=t, budget_ms=b.ms,
                         budget_bin_ms=b.bin_ms, budget_lo=b.lo_ms,
                         budget_hi=b.hi_ms, method=b.method,
                         criterion=b.criterion, sweep_step_ms=(step if step is not None else np.nan),
                         nominal_cte=nominal["mean_cte"],
                         completion=nominal["completion"],
                         solve_ms=nominal["solve_ms"]))
    return pd.DataFrame(rows)


def _budget_objs(bud, tuning):
    """Rebuild Budget-like tuples for the resolvability check."""
    from .metrics import Budget
    out = {}
    for _, r in bud[bud.tuning == tuning].iterrows():
        out[r.controller] = Budget(r.budget_ms, r.criterion, r.budget_bin_ms,
                                   r.budget_lo, r.budget_hi, r.method)
    return out


def resolvability(bud, tuning):
    """Which pairs this experiment can actually tell apart, and which it can't."""
    objs = _budget_objs(bud, tuning)
    names = sorted(objs, key=lambda c: budget_sort_key(objs[c]))
    unresolved = []
    for i in range(len(names) - 1):
        a, b = names[i], names[i + 1]
        if not distinguishable(objs[a], objs[b]):
            unresolved.append((a, b))
    return names, unresolved


def improves_with_delay(sub, frac=0.97):
    """True if tracking got BETTER as delay grew, before any failure.

    Delay can't improve tracking. When it appears to, the tuning is
    lag-dominated and the injected delay is partly cancelling that lag, so
    the resulting "budget" measures a mistuned controller rather than a
    latency tolerance. Seen on an MPC whose position weight was pinned at
    the bottom of its grid.
    """
    sub = sub.sort_values("delay_ms")
    c = sub["mean_cte"].values
    if len(c) < 4 or np.isnan(c[0]):
        return False
    pre = c[:max(3, int(len(c) * 0.6))]
    if np.isnan(pre).any():
        return False
    # the minimum must sit away from zero delay, and the trend across the
    # pre-failure region must be downward rather than one lucky point
    trend = float(np.polyfit(np.arange(len(pre)), pre, 1)[0])
    return bool(pre.min() < c[0] * frac and int(np.argmin(pre)) > 0
                and trend < 0)


def fmt(v, nd=3, dash=">sweep"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return dash
    return f"{v:.{nd}f}"


def table3(bud, cond, out):
    m = bud[bud.tuning == "matched"].set_index("controller")
    lines = ["Table III - latency budget at matched tuning", ""]
    lines.append(f"{'Controller':<14}{'CTE d=0 (m)':>14}{'Budget (ms)':>14}"
                 f"{'+/-1SD (ms)':>18}{'Bin (ms)':>11}{'Criterion':>12}"
                 f"{'Completion':>12}")
    for c in ["pid", "pure_pursuit", "stanley", "mpc"]:
        if c not in m.index:
            continue
        r = m.loc[c]
        band = ("n/a" if (r.budget_lo is None or pd.isna(r.budget_lo))
                else f"{r.budget_lo:.0f}-{r.budget_hi:.0f}")
        if r.method == "no_metrics":
            shown = "NO METRICS"
        elif r.method == "at_first_sample":
            shown = "FAILS AT d=0"
        elif r.method == "below_resolution":
            shown = "<{:.0f}".format(r.sweep_step_ms)
        else:
            shown = fmt(r.budget_ms, 1)
        lines.append(f"{NICE[c]:<14}{fmt(r.nominal_cte):>14}"
                     f"{shown:>14}{band:>18}"
                     f"{fmt(r.budget_bin_ms, 0):>11}{r.criterion:>12}"
                     f"{fmt(r.completion, 2):>12}")
    suspicious = [c for c, sub in cond[cond.tuning == "matched"].groupby(
        "controller") if improves_with_delay(sub)]
    if suspicious:
        lines.append("")
        lines.append("NOT A LATENCY BUDGET - tracking IMPROVED with delay for: "
                     + ", ".join(NICE.get(c, c) for c in suspicious))
        lines.append("  Delay can't improve tracking. A lag-dominated tuning "
                     "is being partly cancelled by the")
        lines.append("  injected delay, so this number measures a mistuned "
                     "controller, not its latency tolerance.")
        lines.append("  Re-tune (check for a gain pinned at a grid edge) "
                     "before reporting these.")

    order, unresolved = resolvability(bud, "matched")
    step = bud.sweep_step_ms.dropna().min()
    lines += ["", "Ordering: " + " > ".join(NICE.get(c, c) for c in order)]
    if unresolved:
        lines.append("NOT RESOLVABLE - these pairs overlap within seed scatter "
                     "and must not be ranked against each other:")
        for a, b in unresolved:
            lines.append(f"  {NICE.get(a,a)} vs {NICE.get(b,b)}")
        lines.append(f"  (sweep step {step:.0f} ms; narrow it, add seeds, or "
                     f"report these as tied)")
    else:
        lines.append("All adjacent pairs separated by more than seed scatter.")
    txt = "\n".join(lines)
    open(os.path.join(out, "table3.txt"), "w").write(txt + "\n")
    return txt


def table4(bud, out):
    piv = bud.pivot(index="controller", columns="tuning", values="budget_ms")
    piv = piv.reindex(columns=TUNINGS)
    # never-failed reads as NaN here but means "larger than any measured value"
    never = bud.pivot(index="controller", columns="tuning",
                      values="criterion").reindex(columns=TUNINGS) == "none"
    ranked_vals = piv.where(~never, np.inf)
    ranks = ranked_vals.rank(ascending=False, method="min")
    lines = ["Table IV - budget at three tunings, and rank change", ""]
    lines.append(f"{'Controller':<14}" + "".join(f"{t:>15}" for t in TUNINGS)
                 + f"{'Rank changes':>14}")
    for c in ["pid", "pure_pursuit", "stanley", "mpc"]:
        if c not in piv.index:
            continue
        sub_c = {t: bud[(bud.controller == c) & (bud.tuning == t)]
                 for t in TUNINGS}
        def _cell(t):
            row = sub_c[t]
            if row.empty:
                return "n/a"
            m, st = row.method.iloc[0], row.sweep_step_ms.iloc[0]
            if m == "no_metrics":
                return "NO METRICS"
            if m == "at_first_sample":
                return "FAILS AT d=0"
            if m == "below_resolution":
                return "<{:.0f}".format(st)
            return fmt(piv.loc[c, t], 0)
        vals = "".join("{:>15}".format(_cell(t)) for t in TUNINGS)
        rk = ranks.loc[c]
        changed = int((rk != rk["matched"]).sum())
        lines.append(f"{NICE[c]:<14}{vals}{changed:>14}")
    stable = bool((ranks.nunique(axis=1) == 1).all())
    lines += ["", f"Ordering tuning-invariant across the swept range: "
                  f"{'YES' if stable else 'NO'}"]
    for t in TUNINGS:
        _, unres = resolvability(bud, t)
        if unres:
            pairs = "; ".join(f"{NICE.get(a,a)}~{NICE.get(b,b)}" for a, b in unres)
            lines.append(f"  {t:<14}: unresolved pairs -> {pairs}")
    if any(resolvability(bud, t)[1] for t in TUNINGS):
        lines.append("  A rank change between unresolved pairs isn't evidence "
                     "of a real swap.")
    if not stable:
        for t in TUNINGS:
            order = ranks[t].sort_values().index.tolist()
            lines.append(f"  {t:<14}: " + " > ".join(NICE.get(c, c) for c in order))
    txt = "\n".join(lines)
    open(os.path.join(out, "table4.txt"), "w").write(txt + "\n")
    return txt


def table5(bud, out):
    lines = ["Table V - accuracy-for-tolerance exchange rate", "",
             f"{'Controller':<14}{'d CTE (m)':>12}{'d budget (ms)':>15}"
             f"{'ms per cm':>12}"]
    for c in ["pid", "pure_pursuit", "stanley", "mpc"]:
        sub = bud[bud.controller == c].set_index("tuning")
        if not {"conservative", "aggressive"} <= set(sub.index):
            continue
        dc = sub.loc["conservative", "nominal_cte"] - sub.loc["aggressive", "nominal_cte"]
        ba = sub.loc["aggressive", "budget_ms"]
        bc = sub.loc["conservative", "budget_ms"]
        if ba is None or bc is None or pd.isna(ba) or pd.isna(bc):
            rate = float("nan"); db = float("nan")
        else:
            db = bc - ba
            rate = db / (dc * 100.0) if abs(dc) > 1e-9 else float("nan")
        lines.append(f"{NICE[c]:<14}{fmt(dc):>12}{fmt(db, 0):>15}{fmt(rate, 1):>12}")
    txt = "\n".join(lines)
    open(os.path.join(out, "table5.txt"), "w").write(txt + "\n")
    return txt


def _wm(fig, selftest):
    if selftest:
        fig.text(0.5, 0.5, "SELFTEST\nNOT REAL DATA", fontsize=34, color="red",
                 alpha=0.25, ha="center", va="center", rotation=30, weight="bold")


def fig1(cond, cte_threshold, out, selftest, bud=None):
    sub = cond[cond.tuning == "matched"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for c, s in sub.groupby("controller"):
        s = s.sort_values("delay_ms")
        ax.plot(s.delay_ms, s.mean_cte, marker="o", ms=3.5, label=NICE.get(c, c))
        if s.cte_sd.notna().any():
            ax.fill_between(s.delay_ms, s.mean_cte - s.cte_sd.fillna(0),
                            s.mean_cte + s.cte_sd.fillna(0), alpha=0.15)
    ax.axhline(cte_threshold, ls="--", c="k", lw=1)
    if bud is not None:
        # mark each interpolated budget on the axis, coloured to its curve, so
        # the sub-bin crossings are visible against the sweep points
        colours = {ln.get_label(): ln.get_color() for ln in ax.get_lines()}
        for _, r in bud[bud.tuning == "matched"].iterrows():
            if r.budget_ms is None or pd.isna(r.budget_ms) or r.criterion == "none":
                continue
            col = colours.get(NICE.get(r.controller, r.controller), "k")
            ax.axvline(r.budget_ms, color=col, lw=0.8, ls=":", alpha=0.7, zorder=1)
            ax.plot([r.budget_ms], [0], marker="^", ms=7, color=col,
                    transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)
        ax.set_xlabel(ax.get_xlabel() + "\ntriangles = interpolated budgets")
    ax.annotate("failure threshold", (ax.get_xlim()[0], cte_threshold),
                xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Injected sensing-to-actuation delay (ms)")
    ax.set_ylabel("Mean cross-track error (m)")
    ax.set_title("Degradation at matched tuning")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    _wm(fig, selftest)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_degradation.png"), dpi=200)
    plt.close(fig)


def fig2(bud, out, selftest):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for c, s in bud.groupby("controller"):
        s = s.set_index("tuning").reindex(TUNINGS).dropna(subset=["budget_ms"])
        if s.empty:
            continue
        ax.plot(s.budget_ms, s.nominal_cte, marker="o", ms=5,
                label=NICE.get(c, c))
        for t, r in s.iterrows():
            ax.annotate(t[:4], (r.budget_ms, r.nominal_cte), fontsize=7,
                        xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Latency budget (ms)  -  further right is better")
    ax.set_ylabel("Nominal cross-track error at d=0 (m)")
    ax.set_title("The tuning trade-off: crossing curves = tuning-dependent ranking")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    _wm(fig, selftest)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_tradeoff.png"), dpi=200)
    plt.close(fig)


def fig3(cond, out, selftest):
    sub = cond[cond.tuning == "matched"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for c, s in sub.groupby("controller"):
        s = s.sort_values("delay_ms")
        ax.plot(s.delay_ms, s.osc_std, marker="s", ms=3.5, label=NICE.get(c, c))
    ax.set_xlabel("Injected delay (ms)")
    ax.set_ylabel("Detrended steering std. dev. (rad)")
    ax.set_title("Onset of oscillation")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    _wm(fig, selftest)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_oscillation.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--cte-threshold", type=float, default=0.50,
                    help="metres; state the justification in the paper")
    ap.add_argument("--osc-multiple", type=float, default=2.0)
    ap.add_argument("--osc-floor", type=float, default=0.0,
                    help="rad; below this, steering variation doesn't count as "
                         "oscillation however much it grew relatively. Off by "
                         "default so numbers match the reported run; 0.02 stops "
                         "quiet tunings failing on a 0.6 degree wobble")
    ap.add_argument("--no-interpolate", action="store_true",
                    help="report the first failing sweep value instead of "
                         "the interpolated threshold crossing")
    ap.add_argument("--allow-selftest", action="store_true")
    args = ap.parse_args()

    df = load(args.runs, args.allow_selftest)
    selftest = bool(df.get("selftest", pd.Series([0])).any())
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)

    cond = per_condition(df)
    cond.to_csv(os.path.join(args.out, "per_condition.csv"), index=False)
    bud = budgets(cond, args.cte_threshold, args.osc_multiple,
                  interpolate=not args.no_interpolate,
                  osc_floor=args.osc_floor)
    bud.to_csv(os.path.join(args.out, "budgets.csv"), index=False)

    print(table3(bud, cond, args.out), "\n")
    print(table4(bud, args.out), "\n")
    print(table5(bud, args.out))

    fig1(cond, args.cte_threshold, args.figures, selftest, bud)
    fig2(bud, args.figures, selftest)
    fig3(cond, args.figures, selftest)
    print(f"\n[analyze] tables -> {args.out}/  figures -> {args.figures}/")


if __name__ == "__main__":
    main()
