import numpy as np
import pytest

from axion_ic.coulomb_interpolation import (
    CoulombG0Interpolator,
    build_coulomb_interpolator,
    default_coulomb_table_path,
    load_or_build_coulomb_interpolator,
    pointwise_interpolation_errors,
)
from axion_ic.modes import gauge_mode_amplitudes


@pytest.fixture(scope="module")
def interpolator():
    """Build one compact exact table shared by the interpolation tests."""

    return build_coulomb_interpolator(
        xi=2.5,
        x_min=1.0e-6,
        x_max=5.0,
        number_of_points=513,
        decimal_precision=25,
    )


def test_cubic_splines_match_direct_kummer_at_off_grid_points(interpolator):
    """Both tabulated functions agree pointwise between logarithmic nodes."""

    midpoints = np.sqrt(interpolator.x[:-1] * interpolator.x[1:])
    validation_x = np.concatenate((midpoints[::8], midpoints[-1:]))
    errors = pointwise_interpolation_errors(
        interpolator,
        validation_x,
        decimal_precision=25,
    )

    assert errors.maximum_relative_g0 < 1.0e-5
    assert errors.maximum_relative_derivative < 1.0e-5


def test_table_round_trip_preserves_spline_values(interpolator, tmp_path):
    """A saved table reloads without pickle or loss of numerical values."""

    path = tmp_path / "coulomb_xi2p5.npz"
    interpolator.save(path)
    loaded = CoulombG0Interpolator.load(path)
    x = np.geomspace(2.0e-5, 4.9, 23)

    assert loaded.xi == interpolator.xi
    assert loaded.decimal_precision == interpolator.decimal_precision
    for candidate, reference in zip(
        loaded.evaluate(2.5, x),
        interpolator.evaluate(2.5, x),
    ):
        np.testing.assert_array_equal(candidate, reference)


def test_interpolator_rejects_wrong_xi_and_extrapolation(interpolator):
    """A table cannot silently serve another xi or values outside its range."""

    with pytest.raises(ValueError, match="not xi"):
        interpolator.evaluate(2.4, np.array([1.0]))
    with pytest.raises(ValueError, match="outside"):
        interpolator.evaluate(2.5, np.array([0.5 * interpolator.x_min]))
    with pytest.raises(ValueError, match="outside"):
        interpolator.evaluate(2.5, np.array([1.01 * interpolator.x_max]))


def test_gauge_mode_backend_uses_interpolated_values(interpolator):
    """The mode layer can replace direct Kummer calls without changing modes."""

    kappa = np.array([0.0, 0.2, 0.7, 1.3, 2.0])
    tau = -1.7
    direct = gauge_mode_amplitudes(
        kappa,
        tau,
        2.5,
        decimal_precision=25,
    )
    interpolated = gauge_mode_amplitudes(
        kappa,
        tau,
        2.5,
        mode_evaluator=interpolator,
    )

    for candidate, reference in zip(interpolated, direct):
        np.testing.assert_allclose(candidate, reference, rtol=1.0e-5, atol=1e-10)


def test_default_table_path_encodes_xi(tmp_path):
    assert default_coulomb_table_path(2.5, tmp_path) == (
        tmp_path / "coulomb_xi2p5.npz"
    )


def test_missing_table_is_built_saved_and_reused(tmp_path, monkeypatch):
    path = tmp_path / "coulomb_xi1p7.npz"
    builds = 0
    original_builder = build_coulomb_interpolator

    def compact_builder(xi, x_min, x_max, number_of_points, **options):
        nonlocal builds
        builds += 1
        return original_builder(
            xi,
            x_min,
            x_max,
            17,
            decimal_precision=options["decimal_precision"],
        )

    monkeypatch.setattr(
        "axion_ic.coulomb_interpolation.build_coulomb_interpolator",
        compact_builder,
    )
    built = load_or_build_coulomb_interpolator(1.7, path, decimal_precision=25)
    loaded = load_or_build_coulomb_interpolator(1.7, path, decimal_precision=25)

    assert path.is_file()
    assert builds == 1
    assert built.xi == loaded.xi == 1.7
    assert built.x_max == 2.0 * built.xi
    np.testing.assert_array_equal(built.x, loaded.x)


def test_existing_table_with_wrong_xi_is_not_replaced(interpolator, tmp_path):
    path = tmp_path / "wrong_xi.npz"
    interpolator.save(path)

    with pytest.raises(ValueError, match="does not match"):
        load_or_build_coulomb_interpolator(1.5, path)

    assert CoulombG0Interpolator.load(path).xi == 2.5
