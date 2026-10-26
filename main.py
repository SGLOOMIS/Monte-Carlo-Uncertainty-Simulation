"""Monte Carlo uncertainty study of a hypothetical voltage divider.

Run with:   python main.py

All numbers are synthetic. The circuit is a made-up teaching example, not a
measured device. Figures are written to the outputs/ folder.
"""

from pathlib import Path

from src.plots import plot_convergence, plot_output_distribution, plot_sensitivity
from src.simulation import (
    INPUT_NAMES,
    convergence_study,
    nominal_output,
    run_simulation,
    sensitivity_study,
    summarize,
)

# ===========================================================================
# CONFIGURATION  -  edit this section to change the study
# ===========================================================================
#
# Hypothetical circuit: a 12 V supply divided down to 5 V by R1 = 7.0 kohm
# and R2 = 5.0 kohm.  Nominal V_out = 12 * 5000 / (7000 + 5000) = 5.000 V.
#
# Uncertainty in each input (all values invented for illustration):
#   v_in : normal,  std 0.20 V           (supply regulation / ripple)
#   r1   : uniform, +/-5 % of nominal     (resistor tolerance band, flat
#                                          because nothing is known inside it)
#   r2   : normal,  std = 5 % / 3         (a 5 % part whose tolerance is
#                                          treated as a 3-sigma bound)

PARAMETERS = {
    "v_in": {"nominal": 12.0, "distribution": "normal", "std": 0.20, "units": "V"},
    "r1": {"nominal": 7000.0, "distribution": "uniform", "half_width": 0.05 * 7000.0, "units": "ohm"},
    "r2": {"nominal": 5000.0, "distribution": "normal", "std": 0.05 * 5000.0 / 3, "units": "ohm"},
}

# Engineering limit: a hypothetical downstream input rated for at most 5.2 V.
V_OUT_LIMIT = 5.2

N_SIMULATIONS = 100_000
RANDOM_SEED = 2026           # same seed -> identical results on every run
PERCENTILES = (1, 5, 50, 95, 99)

# Convergence study: sample sizes to try, and repeats (different seeds) at each.
CONVERGENCE_SIZES = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
CONVERGENCE_REPEATS = 20

# Sensitivity study: multiply ONE input's std / half-width by each factor
# while the other inputs keep their baseline uncertainty.
SENSITIVITY_FACTORS = [0.0, 0.5, 1.0, 1.5, 2.0]
SENSITIVITY_N = 50_000

OUTPUT_DIR = Path("outputs")

# ===========================================================================


def describe(spec):
    if spec["distribution"] == "normal":
        return f"normal, std {spec['std']:g} {spec['units']}"
    return f"uniform, +/- {spec['half_width']:g} {spec['units']}"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("SYNTHETIC DATA - hypothetical voltage divider, not a real measurement\n")

    # --- Inputs and nominal result ------------------------------------------
    print("Inputs")
    for name in INPUT_NAMES:
        spec = PARAMETERS[name]
        print(f"  {name:5s} nominal {spec['nominal']:>8g} {spec['units']:3s}  ({describe(spec)})")
    nominal = nominal_output(PARAMETERS)
    print(f"\nNominal V_out (no uncertainty): {nominal:.4f} V")

    # --- Main Monte Carlo run -----------------------------------------------
    _, v_out = run_simulation(PARAMETERS, N_SIMULATIONS, RANDOM_SEED)
    s = summarize(v_out, V_OUT_LIMIT, PERCENTILES)

    print(f"\nMonte Carlo results  (n = {s['n']:,}, seed = {RANDOM_SEED})")
    print(f"  mean                {s['mean']:.4f} V   (+/- {s['mean_std_error']:.4f} V sampling error)")
    print(f"  median              {s['median']:.4f} V")
    print(f"  standard deviation  {s['std']:.4f} V")
    print(f"  min / max           {s['min']:.4f} / {s['max']:.4f} V")
    for p, value in s["percentiles"].items():
        print(f"  percentile P{p:<2d}      {value:.4f} V")
    print(f"  P(V_out > {V_OUT_LIMIT} V)    {s['p_exceed']:.4%}   "
          f"(+/- {s['p_exceed_std_error']:.4%} sampling error)")
    print(f"  mean - nominal      {s['mean'] - nominal:+.5f} V")

    # --- Convergence --------------------------------------------------------
    rows = convergence_study(PARAMETERS, CONVERGENCE_SIZES, V_OUT_LIMIT, RANDOM_SEED, CONVERGENCE_REPEATS)
    print(f"\nConvergence  ({CONVERGENCE_REPEATS} repeats per n; 'spread' = std. dev. across repeats)")
    print(f"  {'n':>8s}  {'mean [V]':>9s}  {'spread':>8s}  {'std [V]':>8s}  {'P(exceed)':>9s}  {'spread':>8s}")
    for row in rows:
        print(f"  {row['n']:>8,d}  {row['means'].mean():9.4f}  {row['mean_spread']:8.5f}  "
              f"{row['stds'].mean():8.4f}  {row['p_exceeds'].mean():9.3%}  {row['p_exceed_spread']:8.3%}")

    # --- Sensitivity --------------------------------------------------------
    sensitivity = sensitivity_study(PARAMETERS, SENSITIVITY_FACTORS, SENSITIVITY_N, V_OUT_LIMIT, RANDOM_SEED)
    _, v_base = run_simulation(PARAMETERS, SENSITIVITY_N, RANDOM_SEED)
    base = summarize(v_base, V_OUT_LIMIT)
    print(f"\nSensitivity  (n = {SENSITIVITY_N:,} per run; one input's uncertainty scaled, others at baseline)")
    header = "".join(f"  x{f:<4g}" for f in SENSITIVITY_FACTORS)
    print(f"  output std [mV]  {header}")
    for name in INPUT_NAMES:
        values = "".join(f"  {row['std'] * 1000:5.1f}" for row in sensitivity[name])
        print(f"    {name:13s}{values}")
    print(f"  P(exceed) [%]    {header}")
    for name in INPUT_NAMES:
        values = "".join(f"  {row['p_exceed'] * 100:5.2f}" for row in sensitivity[name])
        print(f"    {name:13s}{values}")

    # --- Figures ------------------------------------------------------------
    paths = [
        plot_output_distribution(v_out, nominal, s, OUTPUT_DIR / "output_distribution.png"),
        plot_convergence(rows, nominal, V_OUT_LIMIT, OUTPUT_DIR / "convergence.png"),
        plot_sensitivity(sensitivity, base["std"], base["p_exceed"], OUTPUT_DIR / "sensitivity.png"),
    ]
    print("\nFigures written:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
