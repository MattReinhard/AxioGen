"""Apply physical scale information to a dimensionless response field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import DimensionlessGrid, PowerLawSpectrum, ScaleMapping
from .integrate import curvature_from_response


@dataclass(frozen=True)
class PhysicalCurvatureRealization:
    """A sourced curvature realization and its physical interpretation.

    ``response_k`` is the dimensionless response produced by the time
    integral.  ``dimensionless_curvature_k`` applies half the power-spectrum
    tilt at field level, so its squared amplitude contains one tilt factor.
    The physical-convention coefficient additionally satisfies
    ``physical_curvature_k_mpc3=dimensionless_curvature_k/k_star**3``.
    """

    response_k: NDArray[np.complex128]
    dimensionless_curvature_k: NDArray[np.complex128]
    physical_curvature_k_mpc3: NDArray[np.complex128]
    vacuum_power: NDArray[np.float64]
    kappa: NDArray[np.float64]
    k_mpc_inverse: NDArray[np.float64]
    k_h_mpc_inverse: NDArray[np.float64]
    grid: DimensionlessGrid
    scale: ScaleMapping
    spectrum: PowerLawSpectrum
    xi: float

    @property
    def box_size_mpc(self) -> float:
        """Physical comoving box length in Mpc."""

        return self.scale.box_size_mpc(self.grid.box_size)

    @property
    def box_size_h_inverse_mpc(self) -> float:
        """Numerical box length when quoted in ``h^-1 Mpc``."""

        return self.scale.box_size_h_inverse_mpc(self.grid.box_size)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return scalar metadata suitable for a future output file header."""

        positive = self.kappa > 0.0
        return {
            "grid_size": self.grid.grid_size,
            "dimensionless_box_size": self.grid.box_size,
            "dimensionless_momentum_spacing": self.grid.momentum_spacing,
            "k_star_mpc_inverse": self.scale.k_star_mpc_inverse,
            "hubble_h": self.scale.hubble_h,
            "box_size_mpc": self.box_size_mpc,
            "box_size_h_inverse_mpc": self.box_size_h_inverse_mpc,
            "represented_k_min_mpc_inverse": float(
                np.min(self.k_mpc_inverse[positive])
            ),
            "represented_k_max_mpc_inverse": float(
                np.max(self.k_mpc_inverse[positive])
            ),
            "vacuum_amplitude": self.spectrum.amplitude,
            "vacuum_spectral_index": self.spectrum.spectral_index,
            "vacuum_pivot_mpc_inverse": self.spectrum.pivot_mpc_inverse,
            "xi": self.xi,
            "dimensionless_fourier_coefficient": "zeta_tilde(kappa)",
            "physical_fourier_coefficient_units": "Mpc^3",
        }


def apply_physical_spectrum(
    response: ArrayLike,
    grid: DimensionlessGrid,
    xi: float,
    scale: ScaleMapping,
    spectrum: PowerLawSpectrum,
) -> PhysicalCurvatureRealization:
    """Map a dimensionless response to physical scales and curvature.

    This is deliberately a post-processing operation.  Changing ``k_star``
    or the vacuum spectrum cannot alter ``response_k``; it only changes the
    physical wavenumber arrays and the external spectrum multiplying the
    response.
    """

    if not isinstance(grid, DimensionlessGrid):
        raise TypeError("grid must be a DimensionlessGrid")
    if not isinstance(scale, ScaleMapping):
        raise TypeError("scale must be a ScaleMapping")
    if not isinstance(spectrum, PowerLawSpectrum):
        raise TypeError("spectrum must be a PowerLawSpectrum")

    response_k = np.asarray(response, dtype=np.complex128)
    expected_shape = (grid.grid_size,) * 3
    if response_k.shape != expected_shape:
        raise ValueError(f"response must have shape {expected_shape}")
    if np.any(~np.isfinite(response_k)):
        raise ValueError("response must contain only finite values")
    if response_k[0, 0, 0] != 0.0:
        raise ValueError("the response Fourier zero mode must be zero")

    xi = float(xi)
    if not np.isfinite(xi) or xi <= 0.0:
        raise ValueError("xi must be finite and positive")

    _, _, _, kappa = grid.wavevectors()
    vacuum_power = spectrum.dimensionless_power(kappa, scale)
    # Half the tilt at field level gives one full tilt factor in the power.
    sourced_field_amplitude = np.sqrt(spectrum.amplitude * vacuum_power)
    dimensionless_curvature_k = curvature_from_response(
        response_k,
        xi,
        sourced_field_amplitude,
    )
    physical_curvature_k_mpc3 = (
        dimensionless_curvature_k / scale.k_star_mpc_inverse**3
    )

    return PhysicalCurvatureRealization(
        response_k=response_k.copy(),
        dimensionless_curvature_k=dimensionless_curvature_k,
        physical_curvature_k_mpc3=physical_curvature_k_mpc3,
        vacuum_power=vacuum_power,
        kappa=kappa,
        k_mpc_inverse=scale.k_mpc_inverse(kappa),
        k_h_mpc_inverse=scale.k_h_mpc_inverse(kappa),
        grid=grid,
        scale=scale,
        spectrum=spectrum,
        xi=xi,
    )
