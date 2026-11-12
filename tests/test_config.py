import numpy as np
import pytest

from axion_ic.config import DimensionlessGrid, PowerLawSpectrum, ScaleMapping


def test_default_dimensionless_grid_has_integer_momentum_spacing():
    """A 2*pi box makes the Fourier labels equal to integer kappa values."""

    grid = DimensionlessGrid(grid_size=8)
    kappa_x, kappa_y, kappa_z, magnitude = grid.wavevectors()
    expected = np.array([0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0])

    np.testing.assert_allclose(kappa_x[:, 0, 0], expected, atol=2e-15)
    np.testing.assert_allclose(kappa_y[0, :, 0], expected, atol=2e-15)
    np.testing.assert_allclose(kappa_z[0, 0, :], expected, atol=2e-15)
    np.testing.assert_allclose(grid.momentum_spacing, 1.0, atol=2e-15)
    np.testing.assert_allclose(grid.component_nyquist, 4.0, atol=2e-15)
    assert magnitude.shape == (8, 8, 8)


def test_desi_scale_mapping_and_box_size():
    """Dimensionless modes 10--100 map to 0.02--0.20 h/Mpc."""

    scale = ScaleMapping.from_h_mpc_inverse(
        k_star_h_mpc_inverse=0.002,
        hubble_h=0.68,
    )
    kappa = np.array([10.0, 100.0])

    np.testing.assert_allclose(scale.k_h_mpc_inverse(kappa), [0.02, 0.20])
    np.testing.assert_allclose(
        scale.k_mpc_inverse(kappa),
        [0.0136, 0.136],
    )
    np.testing.assert_allclose(
        scale.kappa_from_mpc_inverse(scale.k_mpc_inverse(kappa)),
        kappa,
    )
    np.testing.assert_allclose(
        scale.box_size_h_inverse_mpc(2.0 * np.pi),
        1000.0 * np.pi,
    )


def test_position_mapping_round_trip():
    """Dimensionless and physical comoving coordinates are inverse mappings."""

    scale = ScaleMapping(k_star_mpc_inverse=0.01, hubble_h=0.7)
    dimensionless = np.array([-2.0, 0.0, 1.5])
    physical = scale.position_mpc(dimensionless)

    np.testing.assert_allclose(physical, [-200.0, 0.0, 150.0])
    np.testing.assert_allclose(scale.dimensionless_position(physical), dimensionless)


def test_power_law_is_normalized_at_physical_pivot():
    """The spectrum equals A_s when k_star*kappa equals the pivot scale."""

    amplitude = 2.1e-9
    scale = ScaleMapping(k_star_mpc_inverse=0.01, hubble_h=0.7)
    spectrum = PowerLawSpectrum(
        amplitude=amplitude,
        spectral_index=0.96,
        pivot_mpc_inverse=0.05,
    )

    power = spectrum.dimensionless_power(np.array([0.0, 5.0]), scale)
    assert power[0] == 0.0
    np.testing.assert_allclose(power[1], amplitude, rtol=2e-15)


def test_power_law_has_expected_tilt_ratio():
    """Power ratios depend only on the physical wavenumber ratio."""

    spectral_index = 0.965
    scale = ScaleMapping(k_star_mpc_inverse=0.004, hubble_h=0.68)
    spectrum = PowerLawSpectrum(
        amplitude=2.1e-9,
        spectral_index=spectral_index,
    )
    kappa = np.array([2.0, 20.0])
    power = spectrum.dimensionless_power(kappa, scale)

    np.testing.assert_allclose(
        power[1] / power[0],
        (kappa[1] / kappa[0]) ** (spectral_index - 1.0),
        rtol=2e-15,
    )


def test_scale_invariant_spectrum_is_independent_of_k_star():
    """For n_s=1 every nonzero mode has the same power for any mapping."""

    amplitude = 2.1e-9
    spectrum = PowerLawSpectrum(amplitude=amplitude, spectral_index=1.0)
    kappa = np.array([0.0, 1.0, 40.0])
    first = spectrum.dimensionless_power(
        kappa,
        ScaleMapping(k_star_mpc_inverse=0.001, hubble_h=0.68),
    )
    second = spectrum.dimensionless_power(
        kappa,
        ScaleMapping(k_star_mpc_inverse=0.05, hubble_h=0.68),
    )

    np.testing.assert_array_equal(first, [0.0, amplitude, amplitude])
    np.testing.assert_array_equal(second, first)


@pytest.mark.parametrize(
    ("constructor", "exception"),
    [
        (lambda: DimensionlessGrid(grid_size=1), ValueError),
        (lambda: DimensionlessGrid(grid_size=True), TypeError),
        (lambda: DimensionlessGrid(grid_size=8, box_size=0.0), ValueError),
        (lambda: ScaleMapping(k_star_mpc_inverse=-1.0, hubble_h=0.7), ValueError),
        (lambda: ScaleMapping(k_star_mpc_inverse=0.01, hubble_h=0.0), ValueError),
        (lambda: PowerLawSpectrum(amplitude=0.0), ValueError),
        (lambda: PowerLawSpectrum(amplitude=2.1e-9, spectral_index=np.nan), ValueError),
        (lambda: PowerLawSpectrum(amplitude=2.1e-9, pivot_mpc_inverse=-0.05), ValueError),
    ],
)
def test_invalid_configuration_values_are_rejected(constructor, exception):
    """Invalid geometry, scale, and spectrum parameters fail immediately."""

    with pytest.raises(exception):
        constructor()
