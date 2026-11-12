import json

import numpy as np
import pytest

from axion_ic.config import DimensionlessGrid, ScaleMapping
from axion_ic.fourier import box_inverse
from vacuum.generate import (
    draw_scalar_white_noise_k,
    generate_vacuum_realization,
    main,
    ordinary_power_from_dimensionless,
)
from vacuum.normalization import (
    ScalarPowerNormalization,
    F2_FIT_AMPLITUDE,
    F2_FIT_EXPONENT,
    fitted_f2,
)


def _negative_index_view(field):
    negative = (-np.arange(field.shape[0])) % field.shape[0]
    result = np.take(field, negative, axis=0)
    result = np.take(result, negative, axis=1)
    return np.take(result, negative, axis=2)


def _configuration(grid_size=7):
    normalization = ScalarPowerNormalization.from_current_fit(2.5)
    return (
        DimensionlessGrid(grid_size, 2.0 * np.pi),
        ScaleMapping.from_h_mpc_inverse(2.0 * np.pi / 2000.0, 0.68),
        normalization,
        normalization.vacuum_spectrum(),
    )


def test_current_f2_fit_and_quadratic_normalization():
    normalization = ScalarPowerNormalization.from_current_fit(2.5)

    np.testing.assert_allclose(
        normalization.f2,
        F2_FIT_AMPLITUDE / 2.5**F2_FIT_EXPONENT,
    )
    reconstructed_total = (
        normalization.vacuum_amplitude + normalization.sourced_amplitude
    )
    np.testing.assert_allclose(
        reconstructed_total,
        normalization.target_total_amplitude,
        rtol=2.0e-15,
    )
    np.testing.assert_allclose(
        normalization.vacuum_fraction + normalization.sourced_fraction,
        1.0,
        rtol=2.0e-15,
    )
    assert normalization.vacuum_amplitude < normalization.target_total_amplitude


@pytest.mark.parametrize("xi", [0.5, 1.8, 3.0, 4.5])
def test_current_f2_fit_accepts_any_positive_xi(xi):
    np.testing.assert_allclose(
        fitted_f2(xi),
        F2_FIT_AMPLITUDE / xi**F2_FIT_EXPONENT,
    )


def test_zero_f2_recovers_vacuum_only_amplitude():
    normalization = ScalarPowerNormalization(
        xi=2.5,
        target_total_amplitude=2.1e-9,
        f2=0.0,
    )

    assert normalization.vacuum_amplitude == normalization.target_total_amplitude
    assert normalization.sourced_amplitude == 0.0
    assert normalization.vacuum_fraction == 1.0
    assert normalization.sourced_fraction == 0.0


@pytest.mark.parametrize("xi", [0.0, -1.0, np.inf, np.nan])
def test_current_f2_fit_rejects_nonpositive_or_nonfinite_xi(xi):
    with pytest.raises(ValueError, match="finite and positive"):
        fitted_f2(xi)


def test_real_space_draw_and_real_filter_preserve_reality():
    grid, scale, normalization, spectrum = _configuration()
    realization = generate_vacuum_realization(
        grid,
        scale,
        normalization,
        spectrum,
        rng=np.random.default_rng(12345),
    )
    field = realization.dimensionless_curvature_k

    np.testing.assert_allclose(
        _negative_index_view(field),
        field.conj(),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    values_x = box_inverse(field, grid.box_size)
    assert np.max(np.abs(values_x.imag)) < 2.0e-13
    np.testing.assert_allclose(realization.real_space(), values_x.real)
    assert field[0, 0, 0] == 0.0


def test_generator_applies_square_root_of_target_ordinary_power():
    grid, scale, normalization, spectrum = _configuration(grid_size=5)
    seed = 7721
    white_noise = draw_scalar_white_noise_k(np.random.default_rng(seed), grid)
    kappa = grid.wavevectors()[3]
    target = normalization.vacuum_spectrum().dimensionless_power(kappa, scale)
    ordinary_power = ordinary_power_from_dimensionless(kappa, target)

    realization = generate_vacuum_realization(
        grid,
        scale,
        normalization,
        spectrum,
        rng=np.random.default_rng(seed),
    )
    expected = white_noise * np.sqrt(ordinary_power)
    expected[0, 0, 0] = 0.0
    np.testing.assert_allclose(realization.dimensionless_curvature_k, expected)


def test_ensemble_mode_variance_matches_finite_box_power():
    grid, scale, normalization, spectrum = _configuration(grid_size=5)
    mode = (1, 1, 0)
    kappa = grid.wavevectors()[3]
    target_dimensionless = normalization.vacuum_spectrum().dimensionless_power(
        kappa,
        scale,
    )
    target_ordinary = ordinary_power_from_dimensionless(kappa, target_dimensionless)
    expected_variance = grid.volume * target_ordinary[mode]

    rng = np.random.default_rng(9982)
    samples = np.empty(1600, dtype=complex)
    for index in range(samples.size):
        realization = generate_vacuum_realization(
            grid,
            scale,
            normalization,
            spectrum,
            rng=rng,
        )
        samples[index] = realization.dimensionless_curvature_k[mode]

    assert abs(np.mean(samples)) < 0.08 * np.sqrt(expected_variance)
    np.testing.assert_allclose(
        np.mean(np.abs(samples) ** 2),
        expected_variance,
        rtol=0.06,
    )


@pytest.mark.parametrize(
    ("kappa", "power"),
    [
        ([-1.0, 1.0], [1.0, 1.0]),
        ([0.0, 1.0], [0.0, np.nan]),
        ([1.0, 2.0], [1.0, 2.0, 3.0]),
    ],
)
def test_invalid_spectrum_inputs_are_rejected(kappa, power):
    with pytest.raises(ValueError):
        ordinary_power_from_dimensionless(kappa, power)


def test_command_line_generator_saves_field_and_metadata(tmp_path):
    parameter_file = tmp_path / "parameters.toml"
    parameter_file.write_text(
        """
[grid]
grid_size = 5
dimensionless_box_size = 6.283185307179586

[physics]
xi = 2.5
target_total_amplitude = 2.1e-9
spectral_index = 0.965
pivot_mpc_inverse = 0.05
k_star_h_mpc_inverse = 0.003141592653589793
hubble_h = 0.68

[integration]
late_time_x_min = 0.09
number_of_log_points = 8
include_shell_entries = true
shell_entry_offset_fraction = 0.0
decimal_precision = 25
dealias = true

[realization]
inverse_decay_seed = 1234
vacuum_seed = 42
output = "unused_sourced_output.npz"
save_helicity = false
""",
        encoding="utf-8",
    )
    output = tmp_path / "vacuum.npz"
    assert (
        main(
            [
                str(parameter_file),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    with np.load(output, allow_pickle=False) as archive:
        field = archive["dimensionless_curvature_k"]
        metadata = json.loads(str(archive["metadata_json"]))
        assert archive.files == ["dimensionless_curvature_k", "metadata_json"]
    assert field.shape == (5, 5, 5)
    assert metadata["vacuum_seed"] == 42
    assert metadata["grid_size"] == 5
    assert metadata["dimensionless_box_size"] == 2.0 * np.pi
    np.testing.assert_allclose(
        metadata["k_star_h_mpc_inverse"],
        2.0 * np.pi / 2000.0,
    )
    assert metadata["vacuum_spectral_index"] == 0.965
    assert metadata["xi"] == 2.5
    assert metadata["target_total_amplitude"] == 2.1e-9
    assert metadata["f2_source"] == "monomial_1p59e-4_xi_minus_6p42"
    np.testing.assert_allclose(
        metadata["vacuum_amplitude"] + metadata["sourced_amplitude"],
        metadata["target_total_amplitude"],
    )
