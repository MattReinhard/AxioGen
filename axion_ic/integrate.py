"""Late-time response kernel and direct-Kummer conformal-time integration."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .source import conformal_source_at_time

try:
    from scipy.special import spherical_jn
except ImportError:  # Keep non-integration package imports usable without SciPy.
    spherical_jn = None


def _require_scipy() -> None:
    if spherical_jn is None:  # pragma: no cover - exercised only without dependency
        raise ImportError(
            "the response kernel requires scipy; install the project requirements"
        )


def late_time_response_kernel(
    wavenumber: ArrayLike,
    tau: float,
) -> NDArray[np.float64]:
    """Return ``K(k,tau) = [sin(x)/x - cos(x)]/k**2`` for ``x=-k tau``.

    The bracket is evaluated as ``x * spherical_jn(1, x)``.  This avoids the
    cancellation of two nearly equal numbers at small ``x``.  At ``k=0`` the
    continuous limit ``tau**2/3`` is returned, although the curvature zero
    mode is removed from every realization.
    """

    _require_scipy()
    tau = float(tau)
    if not np.isfinite(tau) or tau >= 0.0:
        raise ValueError("tau must be finite and negative")
    k = np.asarray(wavenumber, dtype=float)
    if np.any(~np.isfinite(k)) or np.any(k < 0.0):
        raise ValueError("wavenumber must contain finite nonnegative values")

    x = -k * tau
    kernel = np.empty_like(k)
    nonzero = k > 0.0
    kernel[nonzero] = (
        x[nonzero] * spherical_jn(1, x[nonzero]) / k[nonzero] ** 2
    )
    kernel[~nonzero] = tau**2 / 3.0
    return kernel


def trapezoid_weights(coordinate: ArrayLike) -> NDArray[np.float64]:
    """Return nonuniform trapezoidal weights for a strictly increasing grid."""

    values = np.asarray(coordinate, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("coordinate must be a one-dimensional array of length >= 2")
    if np.any(~np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
        raise ValueError("coordinate must be finite and strictly increasing")

    weights = np.empty_like(values)
    weights[0] = 0.5 * (values[1] - values[0])
    weights[-1] = 0.5 * (values[-1] - values[-2])
    weights[1:-1] = 0.5 * (values[2:] - values[:-2])
    return weights


def log_gauss_legendre_grid(
    lower_bound: float,
    upper_bound: float,
    number_of_points: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return nodes and ``du`` weights for global Gauss-Legendre in ``log(u)``."""

    lower_bound = float(lower_bound)
    upper_bound = float(upper_bound)
    if (
        not np.isfinite(lower_bound)
        or not np.isfinite(upper_bound)
        or lower_bound <= 0.0
        or lower_bound >= upper_bound
    ):
        raise ValueError("bounds must satisfy 0 < lower_bound < upper_bound")
    if (
        isinstance(number_of_points, bool)
        or not isinstance(number_of_points, (int, np.integer))
        or number_of_points < 2
    ):
        raise ValueError("number_of_points must be an integer of at least 2")

    nodes, weights = np.polynomial.legendre.leggauss(int(number_of_points))
    log_lower = np.log(lower_bound)
    log_upper = np.log(upper_bound)
    center = 0.5 * (log_upper + log_lower)
    half_width = 0.5 * (log_upper - log_lower)
    log_coordinate = center + half_width * nodes
    coordinate = np.exp(log_coordinate)
    return coordinate, half_width * weights * coordinate


def _positive_retained_shells(
    wavenumber: ArrayLike,
    retained_gauge_mask: ArrayLike | None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    kappa = np.asarray(wavenumber, dtype=float)
    if np.any(~np.isfinite(kappa)) or np.any(kappa < 0.0):
        raise ValueError("wavenumber must contain finite nonnegative values")

    if retained_gauge_mask is None:
        retained = np.ones(kappa.shape, dtype=bool)
    else:
        retained = np.asarray(retained_gauge_mask, dtype=bool)
        if retained.shape != kappa.shape:
            raise ValueError("retained_gauge_mask must match the wavenumber grid")

    positive_shells = np.unique(kappa[retained & (kappa > 0.0)])
    if positive_shells.size == 0:
        raise ValueError("the retained gauge grid must contain a positive mode")
    return positive_shells, retained


def _time_grid_from_bounds(
    positive_shells: NDArray[np.float64],
    xi: float,
    lower_bound: float,
    number_of_log_points: int,
    include_shell_entries: bool,
    shell_entry_offset_fraction: float,
    quadrature: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    xi = float(xi)
    lower_bound = float(lower_bound)
    if not np.isfinite(xi) or xi <= 0.0:
        raise ValueError("xi must be finite and positive")
    if not np.isfinite(lower_bound) or lower_bound <= 0.0:
        raise ValueError("the late-time lower bound must be finite and positive")
    if (
        isinstance(number_of_log_points, bool)
        or not isinstance(number_of_log_points, (int, np.integer))
        or number_of_log_points < 2
    ):
        raise ValueError("number_of_log_points must be an integer of at least 2")
    shell_entry_offset_fraction = float(shell_entry_offset_fraction)
    if (
        not np.isfinite(shell_entry_offset_fraction)
        or shell_entry_offset_fraction < 0.0
        or shell_entry_offset_fraction >= 1.0
    ):
        raise ValueError("shell_entry_offset_fraction must satisfy 0 <= value < 1")
    if shell_entry_offset_fraction > 0.0 and not include_shell_entries:
        raise ValueError(
            "shell_entry_offset_fraction requires include_shell_entries=True"
        )
    if quadrature not in {"log_trapezoid", "log_gauss_legendre"}:
        raise ValueError(
            "quadrature must be 'log_trapezoid' or 'log_gauss_legendre'"
        )

    upper_bound = 2.0 * xi / positive_shells[0]
    if lower_bound >= upper_bound:
        raise ValueError("the late-time bound must be smaller than 2 xi/kappa_min")

    if quadrature == "log_gauss_legendre":
        if include_shell_entries or shell_entry_offset_fraction > 0.0:
            raise ValueError(
                "log_gauss_legendre is a global quadrature and cannot insert "
                "shell-entry nodes"
            )
        return log_gauss_legendre_grid(
            lower_bound,
            upper_bound,
            int(number_of_log_points),
        )

    coordinate = np.geomspace(
        lower_bound,
        upper_bound,
        int(number_of_log_points),
    )
    if include_shell_entries:
        entries = 2.0 * xi / positive_shells
        entries = entries[(entries >= lower_bound) & (entries <= upper_bound)]
        inserted = [entries]
        if shell_entry_offset_fraction > 0.0:
            inserted.extend(
                (
                    entries * (1.0 - shell_entry_offset_fraction),
                    entries * (1.0 + shell_entry_offset_fraction),
                )
            )
        inserted_points = np.concatenate(inserted)
        inserted_points = inserted_points[
            (inserted_points >= lower_bound) & (inserted_points <= upper_bound)
        ]
        coordinate = np.unique(np.concatenate((coordinate, inserted_points)))

    return coordinate, trapezoid_weights(coordinate)


def dimensionless_time_grid(
    wavenumber: ArrayLike,
    xi: float,
    *,
    late_time_x_min: float,
    number_of_log_points: int,
    retained_gauge_mask: ArrayLike | None = None,
    include_shell_entries: bool = True,
    shell_entry_offset_fraction: float = 0.0,
    quadrature: str = "log_trapezoid",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build the increasing dimensionless grid ``u=-k_star*tau``.

    The late-time bound is derived from
    ``u_min=late_time_x_min/kappa_max`` using only retained gauge modes.  The
    early bound is ``u_max=2*xi/kappa_min``.  Shell-entry times are likewise
    constructed only from the retained gauge momentum shells.
    """

    positive_shells, _ = _positive_retained_shells(
        wavenumber,
        retained_gauge_mask,
    )
    late_time_x_min = float(late_time_x_min)
    if not np.isfinite(late_time_x_min) or late_time_x_min <= 0.0:
        raise ValueError("late_time_x_min must be finite and positive")
    u_min = late_time_x_min / positive_shells[-1]
    return _time_grid_from_bounds(
        positive_shells,
        xi,
        u_min,
        number_of_log_points,
        include_shell_entries,
        shell_entry_offset_fraction,
        quadrature,
    )


def direct_time_grid(
    wavenumber: ArrayLike,
    xi: float,
    *,
    y_min: float,
    number_of_log_points: int,
    include_shell_entries: bool = True,
    shell_entry_offset_fraction: float = 0.0,
    quadrature: str = "log_trapezoid",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build the increasing ``y=-tau`` grid for the direct reference integral.

    The early-time endpoint is fixed by ``y_max=2 xi/k_min``.  A logarithmic
    base grid resolves the long time range.  By default, all distinct lattice
    shell-entry times ``2 xi/q`` are inserted so the sharp tachyonic boundary
    is never silently crossed inside a coarse quadrature interval.  A positive
    ``shell_entry_offset_fraction`` additionally inserts points immediately
    below and above each entry for a one-sided resolution check.
    """

    positive_shells, _ = _positive_retained_shells(wavenumber, None)
    return _time_grid_from_bounds(
        positive_shells,
        xi,
        y_min,
        number_of_log_points,
        include_shell_entries,
        shell_entry_offset_fraction,
        quadrature,
    )


def integrate_inverse_decay_response_direct(
    helicity_coefficients: ArrayLike,
    wavenumber: ArrayLike,
    box_size: float,
    xi: float,
    *,
    late_time_x_min: float,
    number_of_log_points: int,
    retained_gauge_mask: ArrayLike | None = None,
    include_shell_entries: bool = True,
    shell_entry_offset_fraction: float = 0.0,
    quadrature: str = "log_trapezoid",
    decimal_precision: int = 40,
    dealias: bool = True,
    padding_rule: str = "three_halves",
    mode_evaluator: object | None = None,
) -> NDArray[np.complex128]:
    """Return the dimensionless response before its primordial prefactor.

    The returned field is

        R_kappa = integral du u K(kappa,-u) S(-u,kappa).

    It depends on the dimensionless gauge realization and ``xi``, but not on
    ``A_s``, ``n_s``, a pivot scale, or ``k_star``.  The same helicity field is
    reused at every time to preserve unequal-time stochastic coherence.
    Direct Kummer modes are used unless ``mode_evaluator`` is supplied.
    """

    helicity = np.asarray(helicity_coefficients, dtype=np.complex128)
    kappa = np.asarray(wavenumber, dtype=float)
    if helicity.ndim != 4 or helicity.shape[0] != 3:
        raise ValueError("helicity_coefficients must have shape (3, N, N, N)")
    if helicity.shape[1:] != kappa.shape:
        raise ValueError("wavenumber must match the helicity Fourier grid")

    structural_mask = np.any(helicity != 0.0, axis=0)
    if retained_gauge_mask is None:
        retained = structural_mask
    else:
        supplied = np.asarray(retained_gauge_mask, dtype=bool)
        if supplied.shape != kappa.shape:
            raise ValueError("retained_gauge_mask must match the Fourier grid")
        retained = structural_mask & supplied
    retained_helicity = np.where(retained[np.newaxis, ...], helicity, 0.0)

    u, weights = dimensionless_time_grid(
        kappa,
        xi,
        late_time_x_min=late_time_x_min,
        number_of_log_points=number_of_log_points,
        retained_gauge_mask=retained,
        include_shell_entries=include_shell_entries,
        shell_entry_offset_fraction=shell_entry_offset_fraction,
        quadrature=quadrature,
    )
    response = np.zeros(kappa.shape, dtype=np.complex128)

    for value, weight in zip(u, weights):
        dimensionless_tau = -float(value)
        source = conformal_source_at_time(
            retained_helicity,
            kappa,
            dimensionless_tau,
            xi,
            box_size,
            decimal_precision=decimal_precision,
            mode_evaluator=mode_evaluator,
            remove_zero_mode=True,
            dealias=dealias,
            padding_rule=padding_rule,
        )
        response += (
            weight
            * value
            * late_time_response_kernel(kappa, dimensionless_tau)
            * source
        )

    response[0, 0, 0] = 0.0
    return response


def curvature_from_response(
    response: ArrayLike,
    xi: float,
    sourced_field_amplitude: ArrayLike,
) -> NDArray[np.complex128]:
    """Apply the external field-level amplitude to a response realization."""

    integrated = np.asarray(response, dtype=np.complex128)
    if integrated.ndim != 3 or len(set(integrated.shape)) != 1:
        raise ValueError("response must have shape (N, N, N)")
    xi = float(xi)
    if not np.isfinite(xi) or xi <= 0.0:
        raise ValueError("xi must be finite and positive")
    field_amplitude = np.asarray(sourced_field_amplitude, dtype=float)
    try:
        field_amplitude = np.broadcast_to(field_amplitude, integrated.shape)
    except ValueError as error:
        raise ValueError("sourced_field_amplitude must broadcast to response") from error
    if np.any(~np.isfinite(field_amplitude)) or np.any(field_amplitude < 0.0):
        raise ValueError("sourced_field_amplitude must be finite and nonnegative")

    curvature = 8.0 * np.pi**2 * xi * field_amplitude * integrated
    curvature = np.asarray(curvature, dtype=np.complex128)
    curvature[0, 0, 0] = 0.0
    return curvature


def integrate_inverse_decay_direct(
    helicity_coefficients: ArrayLike,
    wavenumber: ArrayLike,
    box_size: float,
    xi: float,
    vacuum_curvature_power: float,
    *,
    y_min: float,
    number_of_log_points: int,
    include_shell_entries: bool = True,
    shell_entry_offset_fraction: float = 0.0,
    quadrature: str = "log_trapezoid",
    decimal_precision: int = 40,
    dealias: bool = True,
    padding_rule: str = "three_halves",
    mode_evaluator: object | None = None,
) -> NDArray[np.complex128]:
    """Integrate one small-grid curvature realization with exact Kummer modes.

    This correctness-first reference evaluates the Coulomb functions directly
    at every time and is therefore intentionally not the production-speed
    implementation.  The same ``helicity_coefficients`` array is reused at
    every time, preserving the unequal-time stochastic coherence.

    The accumulated expression is

        zeta_k = +8 pi**2 xi P_vac integral dy y K(k,-y) S(-y,k).
    """

    helicity = np.asarray(helicity_coefficients, dtype=np.complex128)
    kappa = np.asarray(wavenumber, dtype=float)
    vacuum_curvature_power = float(vacuum_curvature_power)
    if helicity.ndim != 4 or helicity.shape[0] != 3:
        raise ValueError("helicity_coefficients must have shape (3, N, N, N)")
    if helicity.shape[1:] != kappa.shape:
        raise ValueError("wavenumber must match the helicity Fourier grid")
    if not np.isfinite(vacuum_curvature_power) or vacuum_curvature_power <= 0.0:
        raise ValueError("vacuum_curvature_power must be finite and positive")

    structural_mask = np.any(helicity != 0.0, axis=0)
    positive_shells, _ = _positive_retained_shells(kappa, structural_mask)
    late_time_x_min = float(y_min) * positive_shells[-1]
    response = integrate_inverse_decay_response_direct(
        helicity,
        kappa,
        box_size,
        xi,
        late_time_x_min=late_time_x_min,
        number_of_log_points=number_of_log_points,
        retained_gauge_mask=structural_mask,
        include_shell_entries=include_shell_entries,
        shell_entry_offset_fraction=shell_entry_offset_fraction,
        quadrature=quadrature,
        decimal_precision=decimal_precision,
        dealias=dealias,
        padding_rule=padding_rule,
        mode_evaluator=mode_evaluator,
    )
    return curvature_from_response(response, xi, vacuum_curvature_power)
