"""Logarithmic tables and cubic-spline interpolation of exact Coulomb modes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline

from .modes import coulomb_g0_and_derivative


TABLE_FORMAT_VERSION = 1
DEFAULT_TABLE_X_MIN = 1.0e-6
DEFAULT_TABLE_POINTS = 513
DEFAULT_TABLE_DECIMAL_PRECISION = 40


def default_coulomb_table_path(
    xi: float,
    directory: str | Path = "tables",
) -> Path:
    """Return the standard cache path for a fixed-``xi`` Coulomb table."""

    xi = float(xi)
    if not np.isfinite(xi) or xi <= 0.0:
        raise ValueError("xi must be finite and positive")
    label = format(xi, ".15g").replace("-", "m").replace("+", "")
    label = label.replace(".", "p")
    return Path(directory) / f"coulomb_xi{label}.npz"


@dataclass(frozen=True)
class CoulombInterpolationErrors:
    """Pointwise interpolation errors against direct Kummer evaluations."""

    maximum_absolute_g0: float
    maximum_relative_g0: float
    maximum_absolute_derivative: float
    maximum_relative_derivative: float


class CoulombG0Interpolator:
    """Cubic splines for ``G_0(xi,x)`` and its derivative at fixed ``xi``."""

    def __init__(
        self,
        xi: float,
        x: ArrayLike,
        g0: ArrayLike,
        derivative: ArrayLike,
        *,
        decimal_precision: int,
    ) -> None:
        self.xi = float(xi)
        self.x = np.asarray(x, dtype=float)
        self.g0 = np.asarray(g0, dtype=float)
        self.derivative = np.asarray(derivative, dtype=float)
        self.decimal_precision = int(decimal_precision)
        if not np.isfinite(self.xi) or self.xi <= 0.0:
            raise ValueError("xi must be finite and positive")
        if self.x.ndim != 1 or self.x.size < 4:
            raise ValueError("x must be a one-dimensional grid with at least 4 points")
        if self.g0.shape != self.x.shape or self.derivative.shape != self.x.shape:
            raise ValueError("g0 and derivative must match the x grid")
        if (
            np.any(~np.isfinite(self.x))
            or np.any(self.x <= 0.0)
            or np.any(np.diff(self.x) <= 0.0)
        ):
            raise ValueError("x must be finite, positive, and strictly increasing")
        if np.any(~np.isfinite(self.g0)) or np.any(~np.isfinite(self.derivative)):
            raise ValueError("table values must be finite")
        if self.decimal_precision < 15:
            raise ValueError("decimal_precision must be at least 15")
        self._g0_spline = CubicSpline(self.x, self.g0, extrapolate=False)
        self._derivative_spline = CubicSpline(
            self.x,
            self.derivative,
            extrapolate=False,
        )

    @property
    def x_min(self) -> float:
        return float(self.x[0])

    @property
    def x_max(self) -> float:
        return float(self.x[-1])

    def evaluate(
        self,
        xi: float,
        x: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate both splines without permitting extrapolation."""

        requested_xi = float(xi)
        if not np.isclose(requested_xi, self.xi, rtol=1.0e-13, atol=0.0):
            raise ValueError(
                f"Coulomb table has xi={self.xi:g}, not xi={requested_xi:g}"
            )
        values = np.asarray(x, dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError("x must contain finite values")
        if np.any(values < self.x_min) or np.any(values > self.x_max):
            raise ValueError(
                f"x lies outside Coulomb table range [{self.x_min:g}, "
                f"{self.x_max:g}]"
            )
        return (
            np.asarray(self._g0_spline(values), dtype=float),
            np.asarray(self._derivative_spline(values), dtype=float),
        )

    def save(self, path: str | Path) -> None:
        """Save the table and construction metadata to a NumPy archive."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as stream:
            np.savez_compressed(
                stream,
                format_version=np.asarray(TABLE_FORMAT_VERSION),
                xi=np.asarray(self.xi),
                x=self.x,
                g0=self.g0,
                derivative=self.derivative,
                decimal_precision=np.asarray(self.decimal_precision),
                spacing=np.asarray("logarithmic"),
                interpolation=np.asarray("scipy.interpolate.CubicSpline"),
            )

    @classmethod
    def load(cls, path: str | Path) -> "CoulombG0Interpolator":
        """Load and validate a saved table without enabling pickle."""

        with np.load(Path(path), allow_pickle=False) as archive:
            version = int(archive["format_version"])
            if version != TABLE_FORMAT_VERSION:
                raise ValueError(
                    f"unsupported Coulomb table format version {version}"
                )
            if str(archive["spacing"]) != "logarithmic":
                raise ValueError("Coulomb table is not logarithmically spaced")
            return cls(
                xi=float(archive["xi"]),
                x=archive["x"],
                g0=archive["g0"],
                derivative=archive["derivative"],
                decimal_precision=int(archive["decimal_precision"]),
            )


def build_coulomb_interpolator(
    xi: float,
    x_min: float,
    x_max: float,
    number_of_points: int,
    *,
    decimal_precision: int = 40,
) -> CoulombG0Interpolator:
    """Evaluate exact Kummer modes on a logarithmic grid and build splines."""

    xi = float(xi)
    x_min = float(x_min)
    x_max = float(x_max)
    if not np.isfinite(xi) or xi <= 0.0:
        raise ValueError("xi must be finite and positive")
    if not np.isfinite(x_min) or x_min <= 0.0:
        raise ValueError("x_min must be finite and positive")
    if not np.isfinite(x_max) or x_max <= x_min:
        raise ValueError("x_max must be finite and greater than x_min")
    if (
        isinstance(number_of_points, bool)
        or not isinstance(number_of_points, (int, np.integer))
        or number_of_points < 4
    ):
        raise ValueError("number_of_points must be an integer of at least 4")
    x = np.geomspace(x_min, x_max, int(number_of_points))
    g0, derivative = coulomb_g0_and_derivative(
        xi,
        x,
        decimal_precision=decimal_precision,
    )
    return CoulombG0Interpolator(
        xi,
        x,
        g0,
        derivative,
        decimal_precision=decimal_precision,
    )


def load_or_build_coulomb_interpolator(
    xi: float,
    path: str | Path,
    *,
    x_min: float = DEFAULT_TABLE_X_MIN,
    x_max: float | None = None,
    number_of_points: int = DEFAULT_TABLE_POINTS,
    decimal_precision: int = DEFAULT_TABLE_DECIMAL_PRECISION,
) -> CoulombG0Interpolator:
    """Load a fixed-``xi`` table, building and saving it when absent.

    The default upper bound is ``2*xi``, covering the complete tachyonic
    interval without spending table points outside it. Existing archives are
    never replaced; their normal load-time and ``xi`` validation applies.
    """

    xi = float(xi)
    output = Path(path).expanduser().resolve()
    if output.exists():
        interpolator = CoulombG0Interpolator.load(output)
        if not np.isclose(interpolator.xi, xi, rtol=1.0e-13, atol=0.0):
            raise ValueError(
                f"Coulomb table xi={interpolator.xi:g} does not match "
                f"requested xi={xi:g}: {output}"
            )
        return interpolator

    upper_bound = 2.0 * xi if x_max is None else x_max
    interpolator = build_coulomb_interpolator(
        xi,
        x_min,
        upper_bound,
        number_of_points,
        decimal_precision=decimal_precision,
    )
    interpolator.save(output)
    return interpolator


def pointwise_interpolation_errors(
    interpolator: CoulombG0Interpolator,
    x: ArrayLike,
    *,
    decimal_precision: int | None = None,
) -> CoulombInterpolationErrors:
    """Compare spline values with direct Kummer evaluations at supplied points."""

    values = np.asarray(x, dtype=float)
    direct_g0, direct_derivative = coulomb_g0_and_derivative(
        interpolator.xi,
        values,
        decimal_precision=(
            interpolator.decimal_precision
            if decimal_precision is None
            else decimal_precision
        ),
    )
    spline_g0, spline_derivative = interpolator.evaluate(interpolator.xi, values)
    absolute_g0 = np.abs(spline_g0 - direct_g0)
    absolute_derivative = np.abs(spline_derivative - direct_derivative)
    g0_floor = np.finfo(float).eps * max(1.0, float(np.max(np.abs(direct_g0))))
    derivative_floor = np.finfo(float).eps * max(
        1.0,
        float(np.max(np.abs(direct_derivative))),
    )
    return CoulombInterpolationErrors(
        maximum_absolute_g0=float(np.max(absolute_g0)),
        maximum_relative_g0=float(
            np.max(absolute_g0 / np.maximum(np.abs(direct_g0), g0_floor))
        ),
        maximum_absolute_derivative=float(np.max(absolute_derivative)),
        maximum_relative_derivative=float(
            np.max(
                absolute_derivative
                / np.maximum(np.abs(direct_derivative), derivative_floor)
            )
        ),
    )
