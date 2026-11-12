"""Finite-box Fourier transforms in the convention of Reinhard et al.

For a cubic box of volume ``V = L**3`` and spacing ``dx = L / N``,

    f(x_n) = (1 / V) sum_k f_k exp(-i k.x_n),
    f_k = dx**3 sum_x f(x_n) exp(+i k.x_n).

NumPy assigns the negative exponential to ``fftn`` and the positive
exponential (with a factor ``1 / N**3``) to ``ifftn``.  The apparently
reversed use of those routines below is therefore intentional.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


SPATIAL_AXES = (-3, -2, -1)


def _validate_geometry(grid_size: int, box_size: float) -> tuple[int, float]:
    if isinstance(grid_size, bool) or not isinstance(grid_size, (int, np.integer)):
        raise TypeError("grid_size must be an integer")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if not np.isfinite(box_size) or box_size <= 0.0:
        raise ValueError("box_size must be finite and positive")
    return int(grid_size), float(box_size)


def _validate_transform_input(values: ArrayLike) -> NDArray[np.generic]:
    array = np.asarray(values)
    if array.ndim < 3:
        raise ValueError("the input must have at least three spatial axes")
    return array


def box_forward(values_x: ArrayLike, box_size: float) -> NDArray[np.complex128]:
    """Transform sampled position-space values to box Fourier coefficients.

    Leading axes, such as a vector-component axis, are left untouched.
    """

    array = _validate_transform_input(values_x)
    if not np.isfinite(box_size) or box_size <= 0.0:
        raise ValueError("box_size must be finite and positive")
    volume = float(box_size) ** 3
    return volume * np.fft.ifftn(array, axes=SPATIAL_AXES)


def box_inverse(coefficients_k: ArrayLike, box_size: float) -> NDArray[np.complex128]:
    """Transform box Fourier coefficients to sampled position-space values.

    Leading axes, such as a vector-component axis, are left untouched.
    """

    array = _validate_transform_input(coefficients_k)
    if not np.isfinite(box_size) or box_size <= 0.0:
        raise ValueError("box_size must be finite and positive")
    volume = float(box_size) ** 3
    return np.fft.fftn(array, axes=SPATIAL_AXES) / volume


def wavevector_grid(
    grid_size: int, box_size: float
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return ``kx``, ``ky``, ``kz``, and ``|k|`` in NumPy FFT ordering."""

    grid_size, box_size = _validate_geometry(grid_size, box_size)
    spacing = box_size / grid_size
    one_dimensional = 2.0 * np.pi * np.fft.fftfreq(grid_size, d=spacing)
    kx, ky, kz = np.meshgrid(
        one_dimensional,
        one_dimensional,
        one_dimensional,
        indexing="ij",
    )
    magnitude = np.sqrt(kx * kx + ky * ky + kz * kz)
    return kx, ky, kz, magnitude
