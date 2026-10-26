"""Check the voltage-divider equation against hand-calculated examples.

Each expected value below was worked out by hand from
V_out = V_in * R2 / (R1 + R2).
"""

import numpy as np
import pytest

from src.simulation import divider_output


def test_equal_resistors_halve_the_supply():
    # 10 V * 1000 / (1000 + 1000) = 10 * 0.5 = 5.0 V
    assert divider_output(10.0, 1000.0, 1000.0) == pytest.approx(5.0)


def test_two_to_one_ratio():
    # 9 V * 1000 / (2000 + 1000) = 9 / 3 = 3.0 V
    assert divider_output(9.0, 2000.0, 1000.0) == pytest.approx(3.0)


def test_nominal_example_used_in_main():
    # 12 V * 5000 / (7000 + 5000) = 12 * 5/12 = 5.0 V
    assert divider_output(12.0, 7000.0, 5000.0) == pytest.approx(5.0)


def test_non_round_values():
    # 5 V * 1700 / (3300 + 1700) = 5 * 0.34 = 1.7 V
    assert divider_output(5.0, 3300.0, 1700.0) == pytest.approx(1.7)

    # 24 V * 4700 / (10000 + 4700) = 112800 / 14700 = 7.673469... V
    assert divider_output(24.0, 10_000.0, 4700.0) == pytest.approx(7.6734694, rel=1e-7)


def test_zero_supply_gives_zero_output():
    assert divider_output(0.0, 1000.0, 2000.0) == 0.0


def test_scaling_both_resistors_does_not_change_output():
    # Only the ratio R2 / (R1 + R2) matters.
    assert divider_output(12.0, 7.0, 5.0) == pytest.approx(divider_output(12.0, 7000.0, 5000.0))


def test_output_never_exceeds_supply():
    assert divider_output(12.0, 1.0, 1e9) < 12.0
    assert divider_output(12.0, 1e9, 1.0) > 0.0


def test_scalar_inputs_return_float():
    assert isinstance(divider_output(12.0, 7000.0, 5000.0), float)


def test_array_inputs_match_scalar_loop():
    v_in = np.array([10.0, 9.0, 12.0, 5.0])
    r1 = np.array([1000.0, 2000.0, 7000.0, 3300.0])
    r2 = np.array([1000.0, 1000.0, 5000.0, 1700.0])

    result = divider_output(v_in, r1, r2)

    expected = []
    for i in range(len(v_in)):
        expected.append(divider_output(v_in[i], r1[i], r2[i]))
    np.testing.assert_allclose(result, [5.0, 3.0, 5.0, 1.7])
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("r1, r2", [(0.0, 1000.0), (1000.0, 0.0), (-10.0, 1000.0), (1000.0, -5.0)])
def test_non_positive_resistors_are_rejected(r1, r2):
    with pytest.raises(ValueError):
        divider_output(12.0, r1, r2)


def test_one_bad_resistor_in_an_array_is_rejected():
    with pytest.raises(ValueError):
        divider_output(12.0, np.array([1000.0, -1.0]), np.array([1000.0, 1000.0]))
