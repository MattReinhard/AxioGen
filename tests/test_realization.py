import numpy as np
import pytest

from axion_ic.config import DimensionlessGrid, PowerLawSpectrum, ScaleMapping
from axion_ic.realization import apply_physical_spectrum


def _response(grid_size):
    """Return a finite test response with the required vanishing zero mode."""

    response = np.ones((grid_size,) * 3, dtype=complex) * (1.0 + 0.25j)
    response[0, 0, 0] = 0.0
    return response


def test_physical_spectrum_is_applied_at_k_star_times_kappa():
    """The high-level API evaluates the tilt on the physical momentum grid."""

    grid = DimensionlessGrid(5)
    scale = ScaleMapping(k_star_mpc_inverse=0.01, hubble_h=0.7)
    spectrum = PowerLawSpectrum(
        amplitude=2.1e-9,
        spectral_index=0.96,
        pivot_mpc_inverse=0.05,
    )
    response = _response(grid.grid_size)
    result = apply_physical_spectrum(response, grid, 2.0, scale, spectrum)

    np.testing.assert_allclose(result.k_mpc_inverse, 0.01 * result.kappa)
    np.testing.assert_allclose(
        result.k_h_mpc_inverse,
        result.k_mpc_inverse / 0.7,
    )
    sourced_field_amplitude = np.sqrt(spectrum.amplitude * result.vacuum_power)
    expected = 8.0 * np.pi**2 * 2.0 * sourced_field_amplitude * response
    np.testing.assert_allclose(result.dimensionless_curvature_k, expected)
    np.testing.assert_allclose(
        result.physical_curvature_k_mpc3,
        expected / scale.k_star_mpc_inverse**3,
    )
    assert result.dimensionless_curvature_k[0, 0, 0] == 0.0
    assert result.physical_curvature_k_mpc3[0, 0, 0] == 0.0
    assert result.vacuum_power[0, 0, 0] == 0.0


def test_changing_k_star_changes_only_external_scale_dependent_quantities():
    """The reusable response is independent of the chosen physical mapping."""

    grid = DimensionlessGrid(5)
    spectrum = PowerLawSpectrum(amplitude=2.1e-9, spectral_index=0.96)
    response = _response(grid.grid_size)
    first = apply_physical_spectrum(
        response,
        grid,
        2.0,
        ScaleMapping(k_star_mpc_inverse=0.005, hubble_h=0.7),
        spectrum,
    )
    second = apply_physical_spectrum(
        response,
        grid,
        2.0,
        ScaleMapping(k_star_mpc_inverse=0.02, hubble_h=0.7),
        spectrum,
    )

    np.testing.assert_array_equal(first.response_k, response)
    np.testing.assert_array_equal(second.response_k, response)
    np.testing.assert_array_equal(first.kappa, second.kappa)
    np.testing.assert_allclose(second.k_mpc_inverse, 4.0 * first.k_mpc_inverse)
    nonzero = first.kappa > 0.0
    expected_ratio = 4.0 ** ((spectrum.spectral_index - 1.0) / 2.0)
    np.testing.assert_allclose(
        second.dimensionless_curvature_k[nonzero]
        / first.dimensionless_curvature_k[nonzero],
        expected_ratio,
    )
    np.testing.assert_allclose(
        second.physical_curvature_k_mpc3[nonzero]
        / first.physical_curvature_k_mpc3[nonzero],
        expected_ratio / 4.0**3,
    )


def test_scale_invariant_spectrum_removes_k_star_dependence_from_curvature():
    """For n_s=1, remapping the box leaves the curvature coefficients fixed."""

    grid = DimensionlessGrid(4)
    spectrum = PowerLawSpectrum(amplitude=2.1e-9, spectral_index=1.0)
    response = _response(grid.grid_size)
    first = apply_physical_spectrum(
        response,
        grid,
        1.5,
        ScaleMapping(k_star_mpc_inverse=0.001, hubble_h=0.68),
        spectrum,
    )
    second = apply_physical_spectrum(
        response,
        grid,
        1.5,
        ScaleMapping(k_star_mpc_inverse=0.05, hubble_h=0.68),
        spectrum,
    )

    np.testing.assert_array_equal(
        first.dimensionless_curvature_k,
        second.dimensionless_curvature_k,
    )
    np.testing.assert_allclose(
        second.physical_curvature_k_mpc3,
        first.physical_curvature_k_mpc3 / 50.0**3,
    )


def test_result_records_physical_box_and_spectrum_metadata():
    """The result exposes the information needed to interpret saved arrays."""

    grid = DimensionlessGrid(5, box_size=2.0 * np.pi)
    scale = ScaleMapping.from_h_mpc_inverse(0.002, hubble_h=0.68)
    spectrum = PowerLawSpectrum(2.1e-9, 0.965, 0.05)
    result = apply_physical_spectrum(_response(5), grid, 2.5, scale, spectrum)
    metadata = result.metadata

    np.testing.assert_allclose(result.box_size_h_inverse_mpc, 1000.0 * np.pi)
    assert metadata["grid_size"] == 5
    assert metadata["xi"] == 2.5
    assert metadata["vacuum_amplitude"] == 2.1e-9
    assert metadata["vacuum_spectral_index"] == 0.965
    assert metadata["represented_k_min_mpc_inverse"] > 0.0
    assert metadata["represented_k_max_mpc_inverse"] > metadata[
        "represented_k_min_mpc_inverse"
    ]


@pytest.mark.parametrize(
    "bad_response",
    [
        np.zeros((3, 3), dtype=complex),
        np.full((3, 3, 3), np.nan + 0.0j),
        np.pad(np.array([[[1.0 + 0.0j]]]), ((0, 2), (0, 2), (0, 2))),
    ],
)
def test_invalid_response_fields_are_rejected(bad_response):
    """Shape, finiteness, and zero-mode mistakes fail before physical mapping."""

    with pytest.raises(ValueError):
        apply_physical_spectrum(
            bad_response,
            DimensionlessGrid(3),
            2.0,
            ScaleMapping(0.01, 0.7),
            PowerLawSpectrum(2.1e-9),
        )
