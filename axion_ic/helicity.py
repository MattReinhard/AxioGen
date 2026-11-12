"""Gaussian white noise projected onto Reinhard et al.'s amplified helicity."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fourier import box_forward, wavevector_grid


def minus_helicity_projector(k_vector: ArrayLike) -> NDArray[np.complex128]:
    """Return the rank-one projector onto the paper's amplified ``-`` mode.

    This constructs the explicit ``(3, 3)`` matrix for one wavevector and is
    intended mainly for algebra tests, covariance checks, and diagnostics.
    ``project_minus_helicity`` applies the same operator to a full grid
    without allocating a ``(3, 3, N, N, N)`` projector array.

    The Fourier convention is ``exp(-i k.x)``.  A projected vector ``h``
    consequently obeys

        k.h = 0,
        -i k cross h = |k| h.
    """

    vector = np.asarray(k_vector, dtype=float)
    if vector.shape != (3,):
        raise ValueError("k_vector must have shape (3,)")
    magnitude = np.linalg.norm(vector)
    if magnitude == 0.0:
        return np.zeros((3, 3), dtype=np.complex128)

    khat = vector / magnitude
    transverse = np.eye(3) - np.outer(khat, khat)
    cross = np.array(
        [
            [0.0, -khat[2], khat[1]],
            [khat[2], 0.0, -khat[0]],
            [-khat[1], khat[0], 0.0],
        ]
    )
    return 0.5 * (transverse - 1j * cross)


def _validate_vector_grid(
    coefficients: ArrayLike,
    kx: ArrayLike,
    ky: ArrayLike,
    kz: ArrayLike,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    vector = np.asarray(coefficients, dtype=np.complex128)
    grids = tuple(np.asarray(component, dtype=float) for component in (kx, ky, kz))
    if vector.ndim != 4 or vector.shape[0] != 3:
        raise ValueError("coefficients must have shape (3, N, N, N)")
    if any(grid.shape != vector.shape[1:] for grid in grids):
        raise ValueError("each wavevector grid must match the spatial shape")
    if len(set(vector.shape[1:])) != 1:
        raise ValueError("only cubic grids are supported")
    return vector, grids[0], grids[1], grids[2]


def _zero_nyquist_planes(field: NDArray[np.complex128]) -> None:
    grid_size = field.shape[1]
    if grid_size % 2:
        return
    indices = np.rint(np.fft.fftfreq(grid_size) * grid_size).astype(int)
    nx, ny, nz = np.meshgrid(indices, indices, indices, indexing="ij")
    nyquist = (
        (np.abs(nx) == grid_size // 2)
        | (np.abs(ny) == grid_size // 2)
        | (np.abs(nz) == grid_size // 2)
    )
    field[:, nyquist] = 0.0


def project_minus_helicity(
    coefficients: ArrayLike,
    kx: ArrayLike,
    ky: ArrayLike,
    kz: ArrayLike,
    *,
    zero_nyquist: bool = True,
) -> NDArray[np.complex128]:
    """Project a full vector Fourier grid onto the amplified helicity.

    This is the memory-efficient grid-wide counterpart of
    ``minus_helicity_projector``.  It evaluates the projector's transverse
    and cross-product pieces directly instead of constructing a separate
    ``(3, 3)`` matrix at every wavevector.
    """

    vector, kx, ky, kz = _validate_vector_grid(coefficients, kx, ky, kz)
    magnitude = np.sqrt(kx * kx + ky * ky + kz * kz)
    nonzero = magnitude > 0.0

    khat = np.zeros_like(vector.real)
    for index, component in enumerate((kx, ky, kz)):
        khat[index, nonzero] = component[nonzero] / magnitude[nonzero]

    longitudinal_amplitude = np.sum(khat * vector, axis=0)
    transverse = vector - khat * longitudinal_amplitude

    cross = np.empty_like(vector)
    cross[0] = khat[1] * vector[2] - khat[2] * vector[1]
    cross[1] = khat[2] * vector[0] - khat[0] * vector[2]
    cross[2] = khat[0] * vector[1] - khat[1] * vector[0]

    projected = 0.5 * (transverse - 1j * cross)
    projected[:, ~nonzero] = 0.0
    if zero_nyquist:
        _zero_nyquist_planes(projected)
    return projected


def draw_vector_white_noise_k(
    rng: np.random.Generator,
    grid_size: int,
    box_size: float,
) -> NDArray[np.complex128]:
    """Draw vector white noise with ``<w_i(k) w_j*(k')> = V delta_ij delta_kk'``.

    Drawing real position-space noise automatically imposes Hermitian
    symmetry on its Fourier coefficients.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    # wavevector_grid provides the shared geometry validation.
    wavevector_grid(grid_size, box_size)
    spacing = float(box_size) / int(grid_size)
    standard_deviation = spacing ** (-1.5)
    noise_x = rng.normal(
        loc=0.0,
        scale=standard_deviation,
        size=(3, grid_size, grid_size, grid_size),
    )
    return box_forward(noise_x, box_size)


def generate_helicity_field(
    grid_size: int,
    box_size: float,
    *,
    rng: np.random.Generator,
    zero_nyquist: bool = True,
) -> NDArray[np.complex128]:
    """Generate one coherent realization ``h_i(k) = P^-_ij(k) w_j(k)``."""

    kx, ky, kz, _ = wavevector_grid(grid_size, box_size)
    white_noise = draw_vector_white_noise_k(rng, grid_size, box_size)
    return project_minus_helicity(
        white_noise,
        kx,
        ky,
        kz,
        zero_nyquist=zero_nyquist,
    )
