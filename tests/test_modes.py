import mpmath as mp
import numpy as np

from axion_ic.fourier import wavevector_grid
from axion_ic.helicity import generate_helicity_field
from axion_ic.modes import (
    BarnabyPelosoModeEvaluator,
    barnaby_peloso_mode_factors,
    coulomb_g0_and_derivative,
    gauge_field_at_time,
    gauge_mode_amplitudes,
    tachyonic_mask,
)


def test_barnaby_peloso_mode_pair_matches_appendix_a_formulas():
    """The optional backend implements both BP asymptotic mode formulas."""

    xi = 2.5
    x = np.array([0.1, 0.7, 2.0, 4.5])
    mode, effective_derivative = barnaby_peloso_mode_factors(xi, x)
    expected_mode = (x / (2.0 * xi)) ** 0.25 * np.exp(
        np.pi * xi - 2.0 * np.sqrt(2.0 * xi * x)
    )
    np.testing.assert_allclose(mode, expected_mode, rtol=2e-15)
    np.testing.assert_allclose(
        effective_derivative,
        -np.sqrt(2.0 * xi / x) * expected_mode,
        rtol=2e-15,
    )


def test_barnaby_peloso_evaluator_produces_paper_A_and_A_prime():
    """The generic mode path applies the BP factors with the correct k powers."""

    xi = 2.5
    tau = -0.6
    k = np.array([0.0, 0.5, 2.0, 7.0, 9.0])
    mode, derivative = gauge_mode_amplitudes(
        k,
        tau,
        xi,
        mode_evaluator=BarnabyPelosoModeEvaluator(),
    )
    active = tachyonic_mask(k, tau, xi)
    x = -k[active] * tau
    g = (x / (2.0 * xi)) ** 0.25 * np.exp(
        np.pi * xi - 2.0 * np.sqrt(2.0 * xi * x)
    )
    np.testing.assert_allclose(mode[active], g / np.sqrt(2.0 * k[active]))
    np.testing.assert_allclose(
        derivative[active],
        np.sqrt(k[active] * xi / x) * g,
    )
    np.testing.assert_array_equal(mode[~active], 0.0)
    np.testing.assert_array_equal(derivative[~active], 0.0)


def test_kummer_representation_matches_coulomb_g0():
    """The Kummer-U formula agrees with an independent Coulomb evaluator."""

    xi = 2.3
    x = np.array([0.15, 0.8, 2.1, 4.4])
    g0, derivative = coulomb_g0_and_derivative(xi, x)

    with mp.workdps(40):
        expected_g0 = np.array([float(mp.coulombg(0, xi, value)) for value in x])
        expected_derivative = np.array(
            [
                float(mp.diff(lambda y: mp.coulombg(0, xi, y), value))
                for value in x
            ]
        )

    np.testing.assert_allclose(g0, expected_g0, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        derivative,
        expected_derivative,
        rtol=2e-11,
        atol=2e-11,
    )


def test_tachyonic_mask_selects_only_zero_to_two_xi():
    """The mask implements 0 < -k tau < 2 xi and excludes k=0."""

    k = np.array([0.0, 0.25, 1.0, 1.999, 2.0, 3.0])
    np.testing.assert_array_equal(
        tachyonic_mask(k, tau=-2.0, xi=2.0),
        [False, True, True, True, False, False],
    )


def test_gauge_mode_amplitudes_apply_exact_normalization_and_mask():
    """Mode amplitudes use G0/sqrt(2k), its chain-rule derivative, and zeros."""

    xi = 1.7
    tau = -0.8
    k = np.array([0.0, 0.5, 2.0, 5.0])
    mode, derivative = gauge_mode_amplitudes(k, tau, xi)

    active = tachyonic_mask(k, tau, xi)
    x = -k[active] * tau
    g0, dg0_dx = coulomb_g0_and_derivative(xi, x)
    np.testing.assert_allclose(mode[active], g0 / np.sqrt(2.0 * k[active]))
    np.testing.assert_allclose(
        derivative[active],
        -np.sqrt(k[active] / 2.0) * dg0_dx,
    )
    np.testing.assert_array_equal(mode[~active], 0.0)
    np.testing.assert_array_equal(derivative[~active], 0.0)


def test_gauge_field_reuses_and_only_rescales_helicity_realization():
    """The vector field is the fixed random helicity field times a real shell factor."""

    grid_size = 5
    box_size = 8.0
    helicity = generate_helicity_field(
        grid_size,
        box_size,
        rng=np.random.default_rng(31415),
    )
    _, _, _, magnitude = wavevector_grid(grid_size, box_size)
    mode, derivative = gauge_mode_amplitudes(magnitude, tau=-0.7, xi=2.0)
    field, field_derivative = gauge_field_at_time(
        helicity,
        magnitude,
        tau=-0.7,
        xi=2.0,
    )

    np.testing.assert_allclose(field, helicity * mode)
    np.testing.assert_allclose(field_derivative, helicity * derivative)
