import numpy as np
import pytest

from axion_ic.fourier import box_inverse, wavevector_grid
from axion_ic.helicity import generate_helicity_field
from axion_ic.modes import gauge_field_at_time
from axion_ic.source import (
    conformal_source_at_time,
    conformal_source_k,
    direct_dealiased_conformal_source_k,
    direct_conformal_source_k,
    padded_conformal_source_k,
)


def _negative_index_view(field):
    """Return the array evaluated at the periodic negative wavevector index."""

    negative = (-np.arange(field.shape[-1])) % field.shape[-1]
    result = np.take(field, negative, axis=-3)
    result = np.take(result, negative, axis=-2)
    return np.take(result, negative, axis=-1)


def test_fft_source_matches_explicit_circular_convolution():
    """The FFT backend has the required 1/V factor and k-q index ordering."""

    grid_size = 3
    box_size = 2.7
    rng = np.random.default_rng(827)
    gauge = rng.normal(size=(3,) + (grid_size,) * 3) + 1j * rng.normal(
        size=(3,) + (grid_size,) * 3
    )
    derivative = rng.normal(size=gauge.shape) + 1j * rng.normal(size=gauge.shape)
    _, _, _, magnitude = wavevector_grid(grid_size, box_size)

    fft_source = conformal_source_k(
        gauge,
        derivative,
        magnitude,
        box_size,
        remove_zero_mode=False,
    )
    direct_source = direct_conformal_source_k(
        gauge,
        derivative,
        magnitude,
        box_size,
        remove_zero_mode=False,
    )
    np.testing.assert_allclose(fft_source, direct_source, rtol=3e-15, atol=3e-15)


def test_source_momentum_multiplier_belongs_to_undifferentiated_field():
    """A single mode pair is weighted by the q carried by A_i(q)."""

    grid_size = 5
    box_size = 3.4
    volume = box_size**3
    gauge = np.zeros((3,) + (grid_size,) * 3, dtype=complex)
    derivative = np.zeros_like(gauge)
    gauge_mode = (1, 0, 0)
    derivative_mode = (0, 2, 0)
    target_mode = tuple(
        (gauge_mode[axis] + derivative_mode[axis]) % grid_size
        for axis in range(3)
    )
    gauge_amplitude = 1.2 - 0.3j
    derivative_amplitude = -0.4 + 0.8j
    gauge[(0,) + gauge_mode] = gauge_amplitude
    derivative[(0,) + derivative_mode] = derivative_amplitude
    _, _, _, magnitude = wavevector_grid(grid_size, box_size)

    source = conformal_source_k(
        gauge,
        derivative,
        magnitude,
        box_size,
        remove_zero_mode=False,
    )
    expected = (
        magnitude[gauge_mode]
        * derivative_amplitude
        * gauge_amplitude
        / volume
    )

    np.testing.assert_allclose(source[target_mode], expected, rtol=2e-15, atol=2e-15)
    source[target_mode] = 0.0
    np.testing.assert_allclose(source, 0.0, atol=2e-15)


def test_padded_source_matches_explicit_non_circular_convolution():
    """Both padding rules recover every explicitly dealiased retained mode."""

    grid_size = 4
    box_size = 3.1
    rng = np.random.default_rng(9872)
    gauge = rng.normal(size=(3,) + (grid_size,) * 3) + 1j * rng.normal(
        size=(3,) + (grid_size,) * 3
    )
    derivative = rng.normal(size=gauge.shape) + 1j * rng.normal(size=gauge.shape)
    _, _, _, magnitude = wavevector_grid(grid_size, box_size)

    explicit = direct_dealiased_conformal_source_k(
        gauge,
        derivative,
        magnitude,
        box_size,
        remove_zero_mode=False,
    )
    for padding_rule in ("three_halves", "five_halves"):
        padded = padded_conformal_source_k(
            gauge,
            derivative,
            magnitude,
            box_size,
            remove_zero_mode=False,
            padding_rule=padding_rule,
        )
        np.testing.assert_allclose(padded, explicit, rtol=8e-15, atol=8e-15)


def test_padded_source_rejects_unknown_padding_rule():
    """A misspelled rule cannot silently select a different padded grid."""

    grid_size = 3
    box_size = 2.0 * np.pi
    gauge = np.zeros((3,) + (grid_size,) * 3, dtype=complex)
    derivative = np.zeros_like(gauge)
    _, _, _, magnitude = wavevector_grid(grid_size, box_size)

    with pytest.raises(ValueError, match="padding_rule"):
        padded_conformal_source_k(
            gauge,
            derivative,
            magnitude,
            box_size,
            padding_rule="double",
        )


def test_padding_prevents_high_mode_pair_from_aliasing_into_retained_mode():
    """The N=8 product 3+3=6 is not allowed to contaminate mode -2."""

    grid_size = 8
    box_size = 2.0 * np.pi
    gauge = np.zeros((3,) + (grid_size,) * 3, dtype=complex)
    derivative = np.zeros_like(gauge)
    high_mode = (3, 0, 0)
    aliased_mode = (-2 % grid_size, 0, 0)
    gauge[(0,) + high_mode] = 2.0
    derivative[(0,) + high_mode] = 3.0
    _, _, _, magnitude = wavevector_grid(grid_size, box_size)

    circular = conformal_source_k(
        gauge,
        derivative,
        magnitude,
        box_size,
        remove_zero_mode=False,
    )
    padded = padded_conformal_source_k(
        gauge,
        derivative,
        magnitude,
        box_size,
        remove_zero_mode=False,
    )

    assert abs(circular[aliased_mode]) > 0.0
    np.testing.assert_allclose(padded, 0.0, atol=2e-15)


def test_source_zero_mode_is_removed_without_changing_other_modes():
    """Zero-mode subtraction affects only the homogeneous source coefficient."""

    grid_size = 3
    box_size = 4.1
    rng = np.random.default_rng(91)
    gauge = rng.normal(size=(3,) + (grid_size,) * 3)
    derivative = rng.normal(size=gauge.shape)
    _, _, _, magnitude = wavevector_grid(grid_size, box_size)

    retained = conformal_source_k(
        gauge,
        derivative,
        magnitude,
        box_size,
        remove_zero_mode=False,
    )
    subtracted = conformal_source_k(gauge, derivative, magnitude, box_size)

    assert retained[0, 0, 0] != 0.0
    assert subtracted[0, 0, 0] == 0.0
    np.testing.assert_array_equal(subtracted.ravel()[1:], retained.ravel()[1:])


def test_physical_padded_source_is_hermitian_and_real_in_position_space():
    """A real coherent gauge realization produces a real quadratic source."""

    grid_size = 5
    box_size = 8.0
    xi = 2.0
    tau = -0.7
    helicity = generate_helicity_field(
        grid_size,
        box_size,
        rng=np.random.default_rng(451),
    )
    _, _, _, magnitude = wavevector_grid(grid_size, box_size)
    gauge, derivative = gauge_field_at_time(helicity, magnitude, tau, xi)
    source = padded_conformal_source_k(gauge, derivative, magnitude, box_size)

    np.testing.assert_allclose(
        _negative_index_view(source),
        source.conj(),
        rtol=2e-13,
        atol=2e-13,
    )
    source_x = box_inverse(source, box_size)
    assert np.max(np.abs(source_x.imag)) < 2e-13


def test_one_call_source_matches_separately_constructed_gauge_fields():
    """The convenience function reuses the same helicity field for A and A'."""

    grid_size = 3
    box_size = 6.0
    xi = 1.5
    tau = -0.6
    helicity = generate_helicity_field(
        grid_size,
        box_size,
        rng=np.random.default_rng(112),
    )
    _, _, _, magnitude = wavevector_grid(grid_size, box_size)
    gauge, derivative = gauge_field_at_time(helicity, magnitude, tau, xi)
    expected = padded_conformal_source_k(gauge, derivative, magnitude, box_size)
    actual = conformal_source_at_time(
        helicity,
        magnitude,
        tau,
        xi,
        box_size,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


def test_one_call_source_accepts_five_halves_padding():
    """The one-call source forwards the selected padding rule."""

    grid_size = 3
    box_size = 6.0
    xi = 1.5
    tau = -0.6
    helicity = generate_helicity_field(
        grid_size,
        box_size,
        rng=np.random.default_rng(119),
    )
    _, _, _, magnitude = wavevector_grid(grid_size, box_size)
    gauge, derivative = gauge_field_at_time(helicity, magnitude, tau, xi)
    expected = padded_conformal_source_k(
        gauge,
        derivative,
        magnitude,
        box_size,
        padding_rule="five_halves",
    )
    actual = conformal_source_at_time(
        helicity,
        magnitude,
        tau,
        xi,
        box_size,
        padding_rule="five_halves",
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)
