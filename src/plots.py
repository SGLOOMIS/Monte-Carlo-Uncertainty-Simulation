"""Plots for the voltage-divider Monte Carlo study.

Every figure is built from synthetic simulation output and is labelled as such.
Each function saves one PNG and returns its path.
"""

import matplotlib

matplotlib.use("Agg")  # write files only; no window needed
import matplotlib.pyplot as plt
import numpy as np

# Colours: a colour-blind-checked categorical order plus neutral chart ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LIMIT_COLOR = "#d03b3b"
SERIES = {"v_in": "#2a78d6", "r1": "#eb6834", "r2": "#1baf7a"}
MARKERS = {"v_in": "o", "r1": "s", "r2": "^"}  # second cue besides colour
MAIN_COLOR = SERIES["v_in"]
MEAN_COLOR = SERIES["r1"]

INPUT_LABELS = {"v_in": "Supply voltage V_in", "r1": "Resistor R1", "r2": "Resistor R2"}

SYNTHETIC_NOTE = "Synthetic data from a hypothetical circuit - not a measurement"


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK)


def _new_figure(ncols, width):
    fig, axes = plt.subplots(1, ncols, figsize=(width, 4.6), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax in axes:
        _style_axes(ax)
    return fig, axes


def _finish(fig, title, path):
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")
    fig.text(0.01, 0.01, SYNTHETIC_NOTE, color=INK_MUTED, fontsize=8, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_output_distribution(v_out, nominal, summary, path):
    """Histogram of simulated V_out against the nominal value and the limit."""
    fig, (ax,) = _new_figure(1, 9)
    limit = summary["limit"]

    counts, edges, patches = ax.hist(v_out, bins=80, density=True, color=MAIN_COLOR,
                                     edgecolor=SURFACE, linewidth=0.4, label="Simulated V_out")
    # Highlight the bins above the engineering limit.
    for left, patch in zip(edges[:-1], patches):
        if left >= limit:
            patch.set_facecolor(LIMIT_COLOR)

    top = counts.max()
    p5 = summary["percentiles"][5]
    p95 = summary["percentiles"][95]
    ax.axvspan(p5, p95, color=MAIN_COLOR, alpha=0.08, label="5th-95th percentile")
    ax.axvline(nominal, color=INK, linestyle="--", linewidth=2, label=f"Nominal = {nominal:.3f} V")
    ax.axvline(summary["mean"], color=MEAN_COLOR, linewidth=2,
               label=f"Simulated mean = {summary['mean']:.3f} V")
    ax.axvline(limit, color=LIMIT_COLOR, linewidth=2, label=f"Limit = {limit:.2f} V")
    ax.text(limit, top * 1.02, f"  P(V_out > limit) = {summary['p_exceed']:.2%}",
            color=INK, fontsize=9, va="bottom")

    ax.set_ylim(0, top * 1.12)
    ax.set_xlabel("Output voltage V_out [V]")
    ax.set_ylabel("Probability density [1/V]")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper left")
    return _finish(fig, f"Distribution of V_out ({summary['n']:,} synthetic samples)", path)


def plot_convergence(rows, nominal, limit, path):
    """Mean and P(exceed) estimates versus sample count, plus their spread."""
    fig, (ax_mean, ax_prob, ax_err) = _new_figure(3, 15)
    ns = np.array([row["n"] for row in rows])

    # Every individual repeat as a faint dot, the average of repeats as a line.
    for row in rows:
        x = np.full(len(row["means"]), row["n"])
        ax_mean.scatter(x, row["means"], s=10, color=MAIN_COLOR, alpha=0.35, linewidths=0)
        ax_prob.scatter(x, row["p_exceeds"] * 100, s=10, color=MAIN_COLOR, alpha=0.35, linewidths=0)
    ax_mean.plot(ns, [row["means"].mean() for row in rows], color=MAIN_COLOR, linewidth=2,
                 marker="o", markersize=5, label="Average of repeats")
    ax_mean.axhline(nominal, color=INK, linestyle="--", linewidth=1.5, label="Nominal")
    ax_prob.plot(ns, [row["p_exceeds"].mean() * 100 for row in rows], color=MAIN_COLOR,
                 linewidth=2, marker="o", markersize=5, label="Average of repeats")

    ax_mean.set_title("Estimated mean of V_out", fontsize=11)
    ax_mean.set_ylabel("V_out mean [V]")
    ax_prob.set_title(f"Estimated P(V_out > {limit:.2f} V)", fontsize=11)
    ax_prob.set_ylabel("Probability [%]")

    # Sampling error: spread across repeats, with the 1/sqrt(n) reference.
    mean_spread = np.array([row["mean_spread"] for row in rows])
    prob_spread = np.array([row["p_exceed_spread"] for row in rows])
    reference = mean_spread[0] * np.sqrt(ns[0] / ns)
    ax_err.plot(ns, mean_spread, color=SERIES["v_in"], linewidth=2, marker="o", markersize=6,
                label="Spread of mean [V]")
    ax_err.plot(ns, prob_spread, color=SERIES["r1"], linewidth=2, marker="s", markersize=6,
                label="Spread of P(exceed) [fraction]")
    ax_err.plot(ns, reference, color=INK_MUTED, linestyle=":", linewidth=1.5,
                label="1 / sqrt(n) reference")
    ax_err.set_yscale("log")
    ax_err.set_title("Numerical sampling error", fontsize=11)
    ax_err.set_ylabel("Std. dev. across repeats")

    for ax in (ax_mean, ax_prob, ax_err):
        ax.set_xscale("log")
        ax.set_xlabel("Number of simulations n")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    n_repeats = len(rows[0]["means"])
    return _finish(fig, f"Convergence with sample size ({n_repeats} repeats per n, each dot = one repeat)", path)


def plot_sensitivity(results, baseline_std, baseline_p, path):
    """Output spread and P(exceed) as each input's uncertainty is scaled alone."""
    fig, (ax_std, ax_prob) = _new_figure(2, 12)

    for name, rows in results.items():
        factors = [row["factor"] for row in rows]
        stds = [row["std"] * 1000 for row in rows]
        probs = [row["p_exceed"] * 100 for row in rows]
        style = dict(color=SERIES[name], marker=MARKERS[name], markersize=7, linewidth=2,
                     label=INPUT_LABELS[name])
        ax_std.plot(factors, stds, **style)
        ax_prob.plot(factors, probs, **style)

    ax_std.axhline(baseline_std * 1000, color=INK_MUTED, linestyle=":", linewidth=1.2, label="Baseline")
    ax_prob.axhline(baseline_p * 100, color=INK_MUTED, linestyle=":", linewidth=1.2, label="Baseline")

    ax_std.set_title("Std. dev. of V_out", fontsize=11)
    ax_std.set_ylabel("Std. dev. [mV]")
    ax_prob.set_title("Probability of exceeding the limit", fontsize=11)
    ax_prob.set_ylabel("Probability [%]")
    for ax in (ax_std, ax_prob):
        ax.set_xlabel("Scale factor on one input's uncertainty (others at baseline)")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper left")

    return _finish(fig, "Sensitivity: scaling one input's uncertainty at a time", path)
