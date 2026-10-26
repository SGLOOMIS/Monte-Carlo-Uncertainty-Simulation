"""Monte Carlo uncertainty analysis of a hypothetical voltage divider.

Everything in this module works on synthetic (randomly generated) data.
The circuit is a textbook example, not a measured physical circuit.

Circuit:

    V_in ──[ R1 ]──┬──[ R2 ]── GND
                   │
                 V_out   (voltage across R2)

    V_out = V_in * R2 / (R1 + R2)
"""

import numpy as np


# ---------------------------------------------------------------------------
# System equation
# ---------------------------------------------------------------------------

def divider_output(v_in, r1, r2):
    """Return the voltage across R2 in an unloaded two-resistor divider.

    Works on plain floats or NumPy arrays (element by element).

    Parameters
    ----------
    v_in : supply voltage [V]
    r1   : top resistor, between supply and output node [ohm]
    r2   : bottom resistor, between output node and ground [ohm]
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    if np.any(r1 <= 0) or np.any(r2 <= 0):
        raise ValueError("Resistor values must be positive.")

    v_out = np.asarray(v_in, dtype=float) * r2 / (r1 + r2)

    # Return a plain float when every input was a scalar.
    if v_out.ndim == 0:
        return float(v_out)
    return v_out


# ---------------------------------------------------------------------------
# Random sampling
# ---------------------------------------------------------------------------
#
# Each uncertain input is described by a small dictionary, for example
#
#     {"nominal": 12.0, "distribution": "normal",  "std": 0.2}
#     {"nominal": 7000, "distribution": "uniform", "half_width": 350}
#
# "normal"  -> Gaussian with the given standard deviation.
# "uniform" -> flat between nominal - half_width and nominal + half_width.
#
# The full set of inputs is a dict {"v_in": spec, "r1": spec, "r2": spec}.
# See the CONFIGURATION section of main.py for the values actually used.

INPUT_NAMES = ("v_in", "r1", "r2")


def sample_parameter(rng, spec, n):
    """Draw n random values for one input described by `spec`."""
    nominal = spec["nominal"]
    if spec["distribution"] == "normal":
        return rng.normal(loc=nominal, scale=spec["std"], size=n)
    if spec["distribution"] == "uniform":
        half_width = spec["half_width"]
        return rng.uniform(low=nominal - half_width, high=nominal + half_width, size=n)
    raise ValueError(f"Unknown distribution: {spec['distribution']!r}")


def draw_samples(params, n, seed):
    """Draw n samples of every input. The same seed always gives the same samples.

    Inputs are drawn in a fixed order (v_in, r1, r2) so results are reproducible.
    """
    rng = np.random.default_rng(seed)
    samples = {}
    for name in INPUT_NAMES:
        samples[name] = sample_parameter(rng, params[name], n)
    return samples


def nominal_output(params):
    """Output voltage with every input at its nominal value (no uncertainty)."""
    return divider_output(params["v_in"]["nominal"], params["r1"]["nominal"], params["r2"]["nominal"])


def run_simulation(params, n, seed):
    """Draw n input samples and compute V_out for every one of them.

    Returns (samples, v_out) where samples is a dict of input arrays.
    """
    samples = draw_samples(params, n, seed)
    v_out = divider_output(samples["v_in"], samples["r1"], samples["r2"])
    return samples, v_out


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize(v_out, limit, percentiles=(1, 5, 50, 95, 99)):
    """Summary statistics of the simulated outputs.

    `limit` is the engineering limit; p_exceed is the fraction of samples
    with V_out strictly greater than it.

    The two *_std_error entries estimate numerical sampling error: how much
    the mean and p_exceed would move if the simulation were rerun with a
    different seed. They shrink like 1 / sqrt(n).
    """
    n = len(v_out)
    mean = np.mean(v_out)
    std = np.std(v_out, ddof=1)
    p_exceed = np.mean(v_out > limit)

    percentile_values = {}
    for p in percentiles:
        percentile_values[p] = float(np.percentile(v_out, p))

    return {
        "n": n,
        "mean": float(mean),
        "median": float(np.median(v_out)),
        "std": float(std),
        "min": float(np.min(v_out)),
        "max": float(np.max(v_out)),
        "percentiles": percentile_values,
        "limit": limit,
        "p_exceed": float(p_exceed),
        "mean_std_error": float(std / np.sqrt(n)),
        "p_exceed_std_error": float(np.sqrt(p_exceed * (1 - p_exceed) / n)),
    }


# ---------------------------------------------------------------------------
# Convergence: how results change as the number of simulations grows
# ---------------------------------------------------------------------------

def convergence_study(params, sample_sizes, limit, seed, n_repeats=20):
    """Rerun the simulation at several sample sizes.

    For each sample size n the simulation is repeated n_repeats times with
    seeds seed, seed + 1, ... The spread of the estimates across repeats is
    a direct measurement of the numerical sampling error at that n.
    """
    rows = []
    for n in sample_sizes:
        means = []
        stds = []
        p_exceeds = []
        for repeat in range(n_repeats):
            _, v_out = run_simulation(params, n, seed + repeat)
            means.append(np.mean(v_out))
            stds.append(np.std(v_out, ddof=1))
            p_exceeds.append(np.mean(v_out > limit))

        rows.append({
            "n": n,
            "means": np.array(means),
            "stds": np.array(stds),
            "p_exceeds": np.array(p_exceeds),
            "mean_spread": float(np.std(means, ddof=1)),
            "std_spread": float(np.std(stds, ddof=1)),
            "p_exceed_spread": float(np.std(p_exceeds, ddof=1)),
        })
    return rows


# ---------------------------------------------------------------------------
# Sensitivity: vary one input's uncertainty at a time
# ---------------------------------------------------------------------------

def scale_uncertainty(spec, factor):
    """Copy of `spec` with its spread (std or half_width) multiplied by factor.

    factor = 0 removes the uncertainty (the input is fixed at nominal),
    factor = 1 leaves it unchanged, factor = 2 doubles it.
    """
    scaled = dict(spec)
    if spec["distribution"] == "normal":
        scaled["std"] = spec["std"] * factor
    elif spec["distribution"] == "uniform":
        scaled["half_width"] = spec["half_width"] * factor
    else:
        raise ValueError(f"Unknown distribution: {spec['distribution']!r}")
    return scaled


def sensitivity_study(params, factors, n, limit, seed):
    """Scale one input's uncertainty at a time, leaving the others at baseline.

    Every run uses the same seed, so differences between runs come from the
    change in uncertainty rather than from different random draws.

    Returns {input_name: [ {"factor", "std", "p_exceed", "mean"}, ... ]}.
    """
    results = {}
    for name in INPUT_NAMES:
        rows = []
        for factor in factors:
            modified = dict(params)
            modified[name] = scale_uncertainty(params[name], factor)
            _, v_out = run_simulation(modified, n, seed)
            rows.append({
                "factor": factor,
                "mean": float(np.mean(v_out)),
                "std": float(np.std(v_out, ddof=1)),
                "p_exceed": float(np.mean(v_out > limit)),
            })
        results[name] = rows
    return results
