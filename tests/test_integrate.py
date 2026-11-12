import numpy as np
import pytest
from scipy.special import spherical_jn

import axion_ic.integrate as integrate_module
from axion_ic.fourier import box_inverse, wavevector_grid
from axion_ic.helicity import generate_helicity_field
from axion_ic.integrate import (
    curvature_from_response,
    dimensionless_time_grid,
    direct_time_grid,
    integrate_inverse_decay_direct,
    integrate_inverse_decay_response_direct,
    late_time_response_kernel,
    log_gauss_legendre_grid,
    trapezoid_weights,
)


def _negative_index_view(field):
    """Return the scalar grid evaluated at each periodic negative index."""

    negative = (-np.arange(field.shape[0])) % field.shape[0]
    result = np.take(field, negative, axis=0)
    result = np.take(result, negative, axis=1)
    return np.take(result, negative, axis=2)


def test_response_kernel_uses_stable_spherical_bessel_identity():
    """K equals x*j1(x)/k^2 and approaches tau^2/3 at very small k."""

    tau = -2.3
    k = np.array([0.0, 1.0e-10, 0.4, 2.1])
    x = -k * tau
    kernel = late_time_response_kernel(k, tau)

    expected = np.empty_like(k)
    expected[0] = tau**2 / 3.0
    expected[1:] = x[1:] * spherical_jn(1, x[1:]) / k[1:] ** 2
    np.testing.assert_allclose(kernel, expected, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(kernel[1], tau**2 / 3.0, rtol=2e-15)
    direct_bracket = np.sin(x[2:]) / x[2:] - np.cos(x[2:])
    np.testing.assert_allclose(
        kernel[2:],
        direct_bracket / k[2:] ** 2,
        rtol=2e-15,
        atol=2e-15,
    )


def test_trapezoid_weights_integrate_linear_function_exactly():
    """Nonuniform trapezoid weights reproduce the exact linear integral."""

    coordinate = np.array([0.2, 0.35, 1.1, 2.8])
    weights = trapezoid_weights(coordinate)
    numerical = np.sum(weights * (2.0 * coordinate - 0.7))
    exact = (coordinate[-1] ** 2 - 0.7 * coordinate[-1]) - (
        coordinate[0] ** 2 - 0.7 * coordinate[0]
    )
    np.testing.assert_allclose(numerical, exact, rtol=2e-15, atol=2e-15)


def test_log_gauss_legendre_grid_integrates_smooth_log_function():
    """The transformed Gauss rule returns positive nodes and du weights."""

    lower = 0.03
    upper = 5.0
    coordinate, weights = log_gauss_legendre_grid(lower, upper, 16)
    assert np.all(np.diff(coordinate) > 0.0)
    assert lower < coordinate[0] < coordinate[-1] < upper
    assert np.all(weights > 0.0)

    numerical = np.sum(weights * coordinate**2)
    exact = (upper**3 - lower**3) / 3.0
    np.testing.assert_allclose(numerical, exact, rtol=2e-10, atol=0.0)


def test_time_grid_has_physical_early_bound_and_shell_entries():
    """The y grid ends at 2 xi/kmin and explicitly contains every entry time."""

    _, _, _, magnitude = wavevector_grid(grid_size=3, box_size=6.0)
    xi = 1.8
    y, weights = direct_time_grid(
        magnitude,
        xi,
        y_min=0.08,
        number_of_log_points=7,
    )
    shells = np.unique(magnitude[magnitude > 0.0])
    entries = 2.0 * xi / shells
    entries = entries[entries >= 0.08]

    np.testing.assert_allclose(y[-1], 2.0 * xi / shells[0])
    for entry in entries:
        assert np.any(np.isclose(y, entry, rtol=2e-15, atol=0.0))
    np.testing.assert_allclose(np.sum(weights), y[-1] - y[0])


def test_time_grid_can_bracket_each_shell_entry_from_both_sides():
    """Optional offset nodes resolve the left and right limits of every step."""

    _, _, _, magnitude = wavevector_grid(grid_size=3, box_size=6.0)
    xi = 1.8
    offset = 1.0e-5
    y_min = 0.08
    y, _ = direct_time_grid(
        magnitude,
        xi,
        y_min=y_min,
        number_of_log_points=7,
        shell_entry_offset_fraction=offset,
    )
    entries = 2.0 * xi / np.unique(magnitude[magnitude > 0.0])
    entries = entries[(entries >= y_min) & (entries <= y[-1])]

    for entry in entries:
        for point in (entry * (1.0 - offset), entry, entry * (1.0 + offset)):
            if y_min <= point <= y[-1]:
                assert np.any(np.isclose(y, point, rtol=2e-15, atol=0.0))


def test_dimensionless_time_bounds_use_only_retained_gauge_modes():
    """u_min and u_max come from x_min/kappa_max and 2*xi/kappa_min."""

    kappa = np.array([0.0, 1.0, 2.0, 4.0])
    retained = np.array([False, True, True, False])
    u, _ = dimensionless_time_grid(
        kappa,
        xi=1.5,
        late_time_x_min=0.2,
        number_of_log_points=5,
        retained_gauge_mask=retained,
        include_shell_entries=False,
    )

    np.testing.assert_allclose(u[0], 0.2 / 2.0)
    np.testing.assert_allclose(u[-1], 2.0 * 1.5 / 1.0)


def test_dimensionless_time_grid_can_use_global_log_gauss_legendre():
    """Global log-u Gaussian quadrature uses interior nodes and du weights."""

    kappa = np.array([0.0, 1.0, 2.0, 4.0])
    retained = np.array([False, True, True, False])
    u, weights = dimensionless_time_grid(
        kappa,
        xi=1.5,
        late_time_x_min=0.2,
        number_of_log_points=12,
        retained_gauge_mask=retained,
        include_shell_entries=False,
        quadrature="log_gauss_legendre",
    )

    lower = 0.2 / 2.0
    upper = 2.0 * 1.5 / 1.0
    assert lower < u[0] < u[-1] < upper
    np.testing.assert_allclose(np.sum(weights), upper - lower, rtol=2e-8, atol=0.0)


def test_global_log_gauss_legendre_rejects_shell_entry_insertion():
    """Shell-entry nodes would break the global Gaussian quadrature rule."""

    kappa = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="global quadrature"):
        dimensionless_time_grid(
            kappa,
            xi=1.5,
            late_time_x_min=0.2,
            number_of_log_points=8,
            quadrature="log_gauss_legendre",
        )


def test_response_and_prefactor_refactor_matches_compatibility_integrator():
    """The dimensionless two-step API reproduces the previous scalar result."""

    grid_size = 3
    box_size = 6.0
    helicity = generate_helicity_field(
        grid_size,
        box_size,
        rng=np.random.default_rng(723),
    )
    _, _, _, magnitude = wavevector_grid(grid_size, box_size)
    retained = np.any(helicity != 0.0, axis=0)
    kappa_max = np.max(magnitude[retained])
    response = integrate_inverse_decay_response_direct(
        helicity,
        magnitude,
        box_size,
        xi=1.5,
        late_time_x_min=0.12 * kappa_max,
        number_of_log_points=5,
        decimal_precision=30,
    )
    curvature = curvature_from_response(response, 1.5, 2.1e-9)
    compatibility = integrate_inverse_decay_direct(
        helicity,
        magnitude,
        box_size,
        xi=1.5,
        vacuum_curvature_power=2.1e-9,
        y_min=0.12,
        number_of_log_points=5,
        decimal_precision=30,
    )

    np.testing.assert_allclose(curvature, compatibility, rtol=2e-15, atol=2e-20)
    assert curvature[0, 0, 0] == 0.0
    assert np.linalg.norm(curvature) > 0.0
    np.testing.assert_allclose(
        _negative_index_view(curvature),
        curvature.conj(),
        rtol=3e-12,
        atol=1e-20,
    )
    curvature_x = box_inverse(curvature, box_size)
    assert np.max(np.abs(curvature_x.imag)) < 1e-18


def test_curvature_prefactor_accepts_scale_dependent_field_amplitude():
    """An external-k spectrum is applied only after the response integral."""

    response = np.ones((2, 2, 2), dtype=complex) * (1.0 + 0.5j)
    field_amplitude = np.arange(8, dtype=float).reshape(2, 2, 2) * 1.0e-10
    xi = 2.3
    curvature = curvature_from_response(response, xi, field_amplitude)
    expected = 8.0 * np.pi**2 * xi * field_amplitude * response
    expected[0, 0, 0] = 0.0

    np.testing.assert_allclose(curvature, expected, rtol=2e-15, atol=2e-20)


def test_response_integrator_forwards_padding_rule(monkeypatch):
    """The time integrator uses the requested source-padding backend."""

    grid_size = 3
    box_size = 2.0 * np.pi
    helicity = generate_helicity_field(
        grid_size,
        box_size,
        rng=np.random.default_rng(42),
    )
    _, _, _, kappa = wavevector_grid(grid_size, box_size)
    captured = []

    def fake_source(*args, padding_rule, **kwargs):
        captured.append(padding_rule)
        return np.zeros_like(kappa, dtype=complex)

    monkeypatch.setattr(integrate_module, "conformal_source_at_time", fake_source)
    integrate_inverse_decay_response_direct(
        helicity,
        kappa,
        box_size,
        xi=1.5,
        late_time_x_min=0.2,
        number_of_log_points=2,
        include_shell_entries=False,
        padding_rule="five_halves",
    )

    assert captured
    assert set(captured) == {"five_halves"}
