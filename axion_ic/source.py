"""Quadratic inverse-decay source constructed from amplified gauge fields."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fourier import box_forward, box_inverse
from .modes import gauge_field_at_time


def _validate_source_inputs(
    gauge_field_k: ArrayLike,
    gauge_derivative_k: ArrayLike,
    wavenumber: ArrayLike,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.float64],
]:
    gauge = np.asarray(gauge_field_k, dtype=np.complex128)
    derivative = np.asarray(gauge_derivative_k, dtype=np.complex128)
    k = np.asarray(wavenumber, dtype=float)

    if gauge.ndim != 4 or gauge.shape[0] != 3:
        raise ValueError("gauge_field_k must have shape (3, N, N, N)")
    if derivative.shape != gauge.shape:
        raise ValueError("gauge_derivative_k must match gauge_field_k")
    if len(set(gauge.shape[1:])) != 1:
        raise ValueError("only cubic Fourier grids are supported")
    if k.shape != gauge.shape[1:]:
        raise ValueError("wavenumber must match the spatial Fourier grid")
    if np.any(~np.isfinite(k)) or np.any(k < 0.0):
        raise ValueError("wavenumber must contain finite nonnegative values")
    return gauge, derivative, k


def conformal_source_k(
    gauge_field_k: ArrayLike,
    gauge_derivative_k: ArrayLike,
    wavenumber: ArrayLike,
    box_size: float,
    *,
    remove_zero_mode: bool = True,
) -> NDArray[np.complex128]:
    """Compute the circular pseudospectral convolution ``S_k``.

    For the retained amplified helicity,

        curl A(k) = k A(k),
        S_k = (1/V) sum_q sum_i A'_i(k-q) q A_i(q).

    The convolution is evaluated by transforming ``A'`` and ``k A`` to
    position space, taking their component-wise dot product, and transforming
    the product back.  This is the conformal source ``A' . curl A``; scale
    factors and the overall sign in ``E . B = -S/a^4`` are not applied here.

    This unpadded implementation is the exact circular convolution on the
    supplied grid.  A production calculation will require a separate
    padding/dealiasing strategy.
    """

    gauge, derivative, k = _validate_source_inputs(
        gauge_field_k,
        gauge_derivative_k,
        wavenumber,
    )
    derivative_x = box_inverse(derivative, box_size)
    curl_x = box_inverse(k * gauge, box_size)
    source_x = np.sum(derivative_x * curl_x, axis=0)
    source_k = box_forward(source_x, box_size)

    if remove_zero_mode:
        source_k[0, 0, 0] = 0.0
    return source_k


def direct_conformal_source_k(
    gauge_field_k: ArrayLike,
    gauge_derivative_k: ArrayLike,
    wavenumber: ArrayLike,
    box_size: float,
    *,
    remove_zero_mode: bool = True,
) -> NDArray[np.complex128]:
    """Compute ``S_k`` by an explicit circular sum for tiny validation grids.

    This is mathematically equivalent to ``conformal_source_k`` but scales as
    ``N**6`` rather than ``N**3 log N``.  It is intended as an independent
    normalization and index-ordering reference, not as a production backend.
    """

    gauge, derivative, k = _validate_source_inputs(
        gauge_field_k,
        gauge_derivative_k,
        wavenumber,
    )
    if not np.isfinite(box_size) or box_size <= 0.0:
        raise ValueError("box_size must be finite and positive")

    grid_shape = k.shape
    q_indices = np.indices(grid_shape)
    weighted_gauge = k * gauge
    source = np.empty(grid_shape, dtype=np.complex128)
    volume = float(box_size) ** 3

    for target in np.ndindex(grid_shape):
        partner = tuple(
            (target[axis] - q_indices[axis]) % grid_shape[axis]
            for axis in range(3)
        )
        source[target] = sum(
            np.sum(derivative[component][partner] * weighted_gauge[component])
            for component in range(3)
        ) / volume

    if remove_zero_mode:
        source[0, 0, 0] = 0.0
    return source


def _signed_mode_indices(grid_size: int) -> NDArray[np.int64]:
    """Return integer Fourier labels in NumPy bin ordering."""

    return np.rint(np.fft.fftfreq(grid_size) * grid_size).astype(np.int64)


def _padded_grid_size(grid_size: int, padding_rule: str) -> int:
    """Return the padded side length selected by a named de-aliasing rule."""

    ratios = {
        "three_halves": (3, 2),
        "five_halves": (5, 2),
    }
    try:
        numerator, denominator = ratios[padding_rule]
    except (KeyError, TypeError) as error:
        choices = ", ".join(repr(choice) for choice in ratios)
        raise ValueError(f"padding_rule must be one of {choices}") from error
    return (numerator * grid_size + denominator - 1) // denominator


def _zero_scalar_nyquist_planes(field: NDArray[np.complex128]) -> None:
    """Remove the boundary excluded by the strict retained-mode region."""

    grid_size = field.shape[0]
    if grid_size % 2:
        return
    modes = _signed_mode_indices(grid_size)
    mx, my, mz = np.meshgrid(modes, modes, modes, indexing="ij")
    nyquist = (
        (np.abs(mx) == grid_size // 2)
        | (np.abs(my) == grid_size // 2)
        | (np.abs(mz) == grid_size // 2)
    )
    field[nyquist] = 0.0


def padded_conformal_source_k(
    gauge_field_k: ArrayLike,
    gauge_derivative_k: ArrayLike,
    wavenumber: ArrayLike,
    box_size: float,
    *,
    remove_zero_mode: bool = True,
    padding_rule: str = "three_halves",
) -> NDArray[np.complex128]:
    """Compute ``S_k`` with three-halves or five-halves padding.

    Each vector component is embedded without rescaling in a grid of size
    ``ceil(3N/2)`` by default, or ``ceil(5N/2)`` when
    ``padding_rule="five_halves"``.  Its position-space contribution is
    accumulated into one scalar padded grid before that product is transformed
    back and the original Fourier bins are extracted.  Streaming the components
    keeps the peak memory close to scalar padded work arrays rather than full
    padded vector fields.  The discarded outer padded bins catch aliases that
    would otherwise wrap into the retained original modes.

    For even ``N``, output Nyquist planes are excluded so the advertised
    retained region is strictly ``|m_x|, |m_y|, |m_z| < N/2``.
    """

    gauge, derivative, k = _validate_source_inputs(
        gauge_field_k,
        gauge_derivative_k,
        wavenumber,
    )
    if not np.isfinite(box_size) or box_size <= 0.0:
        raise ValueError("box_size must be finite and positive")

    grid_size = k.shape[0]
    padded_size = _padded_grid_size(grid_size, padding_rule)
    original_modes = _signed_mode_indices(grid_size)
    padded_bins = original_modes % padded_size
    scalar_bins = np.ix_(padded_bins, padded_bins, padded_bins)
    padded_shape = (padded_size, padded_size, padded_size)

    source_x = np.zeros(padded_shape, dtype=np.complex128)
    padded_derivative = np.zeros(padded_shape, dtype=np.complex128)
    padded_curl = np.zeros(padded_shape, dtype=np.complex128)

    for component in range(3):
        padded_derivative.fill(0.0)
        padded_curl.fill(0.0)
        padded_derivative[scalar_bins] = derivative[component]
        padded_curl[scalar_bins] = k * gauge[component]

        derivative_x = box_inverse(padded_derivative, box_size)
        curl_x = box_inverse(padded_curl, box_size)
        derivative_x *= curl_x
        source_x += derivative_x

    padded_source = box_forward(source_x, box_size)

    source = padded_source[np.ix_(padded_bins, padded_bins, padded_bins)].copy()
    _zero_scalar_nyquist_planes(source)
    if remove_zero_mode:
        source[0, 0, 0] = 0.0
    return source


def direct_dealiased_conformal_source_k(
    gauge_field_k: ArrayLike,
    gauge_derivative_k: ArrayLike,
    wavenumber: ArrayLike,
    box_size: float,
    *,
    remove_zero_mode: bool = True,
) -> NDArray[np.complex128]:
    """Evaluate the non-circular convolution for tiny validation grids.

    Only pairs obeying the ordinary integer relation ``p + q = m`` are
    included.  Pairs obeying that relation only modulo ``N`` are aliases and
    are deliberately omitted.  This ``N**6`` implementation is the reference
    for ``padded_conformal_source_k``.
    """

    gauge, derivative, k = _validate_source_inputs(
        gauge_field_k,
        gauge_derivative_k,
        wavenumber,
    )
    if not np.isfinite(box_size) or box_size <= 0.0:
        raise ValueError("box_size must be finite and positive")

    grid_size = k.shape[0]
    modes = _signed_mode_indices(grid_size)
    mode_to_bin = {int(mode): index for index, mode in enumerate(modes)}
    weighted_gauge = k * gauge
    source = np.zeros(k.shape, dtype=np.complex128)
    volume = float(box_size) ** 3

    for target_bin in np.ndindex(k.shape):
        target_mode = tuple(int(modes[index]) for index in target_bin)
        if grid_size % 2 == 0 and any(
            abs(component) == grid_size // 2 for component in target_mode
        ):
            continue

        total = 0.0j
        for q_bin in np.ndindex(k.shape):
            q_mode = tuple(int(modes[index]) for index in q_bin)
            partner_mode = tuple(
                target_mode[axis] - q_mode[axis] for axis in range(3)
            )
            if not all(component in mode_to_bin for component in partner_mode):
                continue
            partner_bin = tuple(mode_to_bin[component] for component in partner_mode)
            total += np.sum(
                derivative[(slice(None),) + partner_bin]
                * weighted_gauge[(slice(None),) + q_bin]
            )
        source[target_bin] = total / volume

    if remove_zero_mode:
        source[0, 0, 0] = 0.0
    return source


def conformal_source_at_time(
    helicity_coefficients: ArrayLike,
    wavenumber: ArrayLike,
    tau: float,
    xi: float,
    box_size: float,
    *,
    decimal_precision: int = 40,
    mode_evaluator: object | None = None,
    remove_zero_mode: bool = True,
    dealias: bool = True,
    padding_rule: str = "three_halves",
) -> NDArray[np.complex128]:
    """Construct the restricted gauge fields and their source at one time.

    Three-halves padding is used by default; ``padding_rule="five_halves"``
    selects the larger padded grid.  Set ``dealias=False`` only for diagnostics
    that intentionally require the circular grid convolution.
    """

    gauge, derivative = gauge_field_at_time(
        helicity_coefficients,
        wavenumber,
        tau,
        xi,
        decimal_precision=decimal_precision,
        mode_evaluator=mode_evaluator,
    )
    if dealias:
        return padded_conformal_source_k(
            gauge,
            derivative,
            wavenumber,
            box_size,
            remove_zero_mode=remove_zero_mode,
            padding_rule=padding_rule,
        )
    return conformal_source_k(
        gauge,
        derivative,
        wavenumber,
        box_size,
        remove_zero_mode=remove_zero_mode,
    )
