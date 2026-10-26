"""Tests for sampling, statistics, convergence, and sensitivity.

All data here is synthetic.
"""

import numpy as np
import pytest

from src.simulation import (
    convergence_study,
    draw_samples,
    nominal_output,
    run_simulation,
    sample_parameter,
    scale_uncertainty,
    sensitivity_study,
    summarize,
)

PARAMS = {
    "v_in": {"nominal": 12.0, "distribution": "normal", "std": 0.2},
    "r1": {"nominal": 7000.0, "distribution": "uniform", "half_width": 350.0},
    "r2": {"nominal": 5000.0, "distribution": "normal", "std": 5000.0 * 0.05 / 3},
}
LIMIT = 5.2


# --- reproducibility -------------------------------------------------------

def test_same_seed_gives_identical_samples():
    a = draw_samples(PARAMS, 1000, seed=42)
    b = draw_samples(PARAMS, 1000, seed=42)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])


def test_different_seed_gives_different_samples():
    a = draw_samples(PARAMS, 1000, seed=42)
    b = draw_samples(PARAMS, 1000, seed=43)
    assert not np.array_equal(a["v_in"], b["v_in"])


# --- individual distributions ----------------------------------------------

def test_uniform_samples_stay_inside_bounds():
    rng = np.random.default_rng(0)
    values = sample_parameter(rng, PARAMS["r1"], 100_000)
    assert values.min() >= 7000.0 - 350.0
    assert values.max() <= 7000.0 + 350.0
    # Std of a uniform distribution of half-width a is a / sqrt(3).
    assert np.std(values) == pytest.approx(350.0 / np.sqrt(3), rel=0.01)


def test_normal_samples_have_requested_mean_and_std():
    rng = np.random.default_rng(0)
    values = sample_parameter(rng, PARAMS["v_in"], 100_000)
    assert np.mean(values) == pytest.approx(12.0, abs=0.005)
    assert np.std(values) == pytest.approx(0.2, rel=0.01)


def test_unknown_distribution_is_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_parameter(rng, {"nominal": 1.0, "distribution": "lognormal"}, 10)


# --- simulation -------------------------------------------------------------

def test_nominal_output_matches_hand_calculation():
    # 12 * 5000 / 12000 = 5.0 V
    assert nominal_output(PARAMS) == pytest.approx(5.0)


def test_zero_uncertainty_reproduces_nominal_exactly():
    fixed = {name: scale_uncertainty(spec, 0.0) for name, spec in PARAMS.items()}
    _, v_out = run_simulation(fixed, 500, seed=1)
    np.testing.assert_allclose(v_out, 5.0)


def test_one_output_per_sample():
    samples, v_out = run_simulation(PARAMS, 1234, seed=1)
    assert v_out.shape == (1234,)
    assert samples["r2"].shape == (1234,)


def test_monte_carlo_std_matches_first_order_propagation():
    # Hand derivation (linearised error propagation) at the nominal point:
    #   dV/dVin = R2 / (R1 + R2)              = 5000 / 12000      = 0.41667
    #   dV/dR1  = -Vin * R2 / (R1 + R2)^2     = -12*5000/12000^2  = -4.1667e-4 V/ohm
    #   dV/dR2  =  Vin * R1 / (R1 + R2)^2     =  12*7000/12000^2  =  5.8333e-4 V/ohm
    #   sigma_Vin = 0.2 V, sigma_R1 = 350 / sqrt(3) ohm, sigma_R2 = 83.333 ohm
    #   sigma_V^2 = sum of (derivative * sigma)^2
    sens_vin = 5000 / 12000
    sens_r1 = -12 * 5000 / 12000**2
    sens_r2 = 12 * 7000 / 12000**2
    sigma_v = np.sqrt((sens_vin * 0.2) ** 2
                      + (sens_r1 * 350 / np.sqrt(3)) ** 2
                      + (sens_r2 * 5000 * 0.05 / 3) ** 2)

    _, v_out = run_simulation(PARAMS, 200_000, seed=7)
    # The divider is only mildly nonlinear at these tolerances, so the
    # linear estimate should agree to within a couple of percent.
    assert np.std(v_out) == pytest.approx(sigma_v, rel=0.02)
    assert np.mean(v_out) == pytest.approx(5.0, abs=0.005)


# --- summary statistics -----------------------------------------------------

def test_summary_of_known_synthetic_values():
    values = np.arange(1, 101, dtype=float)  # 1, 2, ..., 100
    s = summarize(values, limit=90.0, percentiles=(50,))
    assert s["n"] == 100
    assert s["mean"] == pytest.approx(50.5)
    assert s["median"] == pytest.approx(50.5)
    assert s["percentiles"][50] == pytest.approx(50.5)
    # 91..100 are strictly above 90 -> 10 of 100 samples.
    assert s["p_exceed"] == pytest.approx(0.10)
    # Sample std of 1..100 is sqrt(100 * 101 / 12) = 29.0115
    assert s["std"] == pytest.approx(29.0115, rel=1e-4)
    assert s["mean_std_error"] == pytest.approx(2.90115, rel=1e-4)
    # sqrt(0.1 * 0.9 / 100) = 0.03
    assert s["p_exceed_std_error"] == pytest.approx(0.03)


def test_value_equal_to_limit_does_not_count_as_exceeding():
    s = summarize(np.array([1.0, 2.0, 3.0]), limit=3.0)
    assert s["p_exceed"] == 0.0


# --- convergence ------------------------------------------------------------

def test_sampling_error_shrinks_as_n_grows():
    rows = convergence_study(PARAMS, [100, 10_000], LIMIT, seed=0, n_repeats=20)
    small, large = rows
    assert large["mean_spread"] < small["mean_spread"]
    # 100x more samples should give roughly 10x less spread (1/sqrt(n)).
    ratio = small["mean_spread"] / large["mean_spread"]
    assert 5 < ratio < 20


# --- sensitivity ------------------------------------------------------------

def test_scale_uncertainty_does_not_modify_original():
    original = dict(PARAMS["v_in"])
    scaled = scale_uncertainty(PARAMS["v_in"], 2.0)
    assert scaled["std"] == pytest.approx(0.4)
    assert PARAMS["v_in"] == original


def test_output_spread_grows_with_each_input_uncertainty():
    factors = [0.0, 0.5, 1.0, 2.0]
    results = sensitivity_study(PARAMS, factors, n=20_000, limit=LIMIT, seed=3)
    for name, rows in results.items():
        stds = [row["std"] for row in rows]
        assert stds == sorted(stds), f"{name}: output std should rise with input std"


def test_baseline_factor_matches_plain_simulation():
    results = sensitivity_study(PARAMS, [1.0], n=5000, limit=LIMIT, seed=11)
    _, v_out = run_simulation(PARAMS, 5000, seed=11)
    for rows in results.values():
        assert rows[0]["std"] == pytest.approx(np.std(v_out, ddof=1))
