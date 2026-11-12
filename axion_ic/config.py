"""Validated dimensionless geometry and physical scale interpretation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fourier import wavevector_grid


def _finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _finite_nonnegative_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    return array


@dataclass(frozen=True)
class DimensionlessGrid:
    """Dimensionless periodic-box geometry used by the numerical core.

    The default ``box_size=2*pi`` gives unit Fourier spacing
    ``Delta kappa=1``.  The returned wavevector arrays are therefore
    interpreted as the dimensionless momenta ``kappa = k/k_star``.
    """

    grid_size: int
    box_size: float = 2.0 * np.pi

    def __post_init__(self) -> None:
        if isinstance(self.grid_size, bool) or not isinstance(
            self.grid_size, (int, np.integer)
        ):
            raise TypeError("grid_size must be an integer")
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        object.__setattr__(self, "grid_size", int(self.grid_size))
        object.__setattr__(
            self,
            "box_size",
            _finite_positive(self.box_size, "box_size"),
        )

    @property
    def spacing(self) -> float:
        """Dimensionless position-space lattice spacing."""

        return self.box_size / self.grid_size

    @property
    def volume(self) -> float:
        """Dimensionless box volume."""

        return self.box_size**3

    @property
    def momentum_spacing(self) -> float:
        """Dimensionless fundamental Fourier spacing ``Delta kappa``."""

        return 2.0 * np.pi / self.box_size

    @property
    def component_nyquist(self) -> float:
        """Formal component Nyquist magnitude before boundary removal."""

        return np.pi * self.grid_size / self.box_size

    def wavevectors(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Return ``kappa_x``, ``kappa_y``, ``kappa_z``, and ``|kappa|``."""

        return wavevector_grid(self.grid_size, self.box_size)


@dataclass(frozen=True)
class ScaleMapping:
    """Map dimensionless coordinates to physical comoving units.

    ``k_star_mpc_inverse`` is stored in ``Mpc^-1`` without an ``h`` factor.
    Values conventionally quoted in ``h Mpc^-1`` can be supplied through
    ``from_h_mpc_inverse``.
    """

    k_star_mpc_inverse: float
    hubble_h: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "k_star_mpc_inverse",
            _finite_positive(self.k_star_mpc_inverse, "k_star_mpc_inverse"),
        )
        object.__setattr__(
            self,
            "hubble_h",
            _finite_positive(self.hubble_h, "hubble_h"),
        )

    @classmethod
    def from_h_mpc_inverse(
        cls,
        k_star_h_mpc_inverse: float,
        hubble_h: float,
    ) -> "ScaleMapping":
        """Construct from the numerical coefficient in ``h Mpc^-1``."""

        k_star_h_mpc_inverse = _finite_positive(
            k_star_h_mpc_inverse,
            "k_star_h_mpc_inverse",
        )
        hubble_h = _finite_positive(hubble_h, "hubble_h")
        return cls(
            k_star_mpc_inverse=k_star_h_mpc_inverse * hubble_h,
            hubble_h=hubble_h,
        )

    def k_mpc_inverse(self, kappa: ArrayLike) -> NDArray[np.float64]:
        """Convert ``kappa`` to a wavenumber in ``Mpc^-1``."""

        dimensionless = _finite_nonnegative_array(kappa, "kappa")
        return self.k_star_mpc_inverse * dimensionless

    def k_h_mpc_inverse(self, kappa: ArrayLike) -> NDArray[np.float64]:
        """Return the numerical coefficient of ``k`` in ``h Mpc^-1``."""

        return self.k_mpc_inverse(kappa) / self.hubble_h

    def kappa_from_mpc_inverse(self, wavenumber: ArrayLike) -> NDArray[np.float64]:
        """Convert a wavenumber in ``Mpc^-1`` to ``kappa``."""

        physical = _finite_nonnegative_array(wavenumber, "wavenumber")
        return physical / self.k_star_mpc_inverse

    def position_mpc(self, dimensionless_position: ArrayLike) -> NDArray[np.float64]:
        """Convert ``X=k_star*x`` to a comoving position in Mpc."""

        coordinate = np.asarray(dimensionless_position, dtype=float)
        if np.any(~np.isfinite(coordinate)):
            raise ValueError("dimensionless_position must contain finite values")
        return coordinate / self.k_star_mpc_inverse

    def dimensionless_position(self, position_mpc: ArrayLike) -> NDArray[np.float64]:
        """Convert a comoving position in Mpc to ``X=k_star*x``."""

        coordinate = np.asarray(position_mpc, dtype=float)
        if np.any(~np.isfinite(coordinate)):
            raise ValueError("position_mpc must contain finite values")
        return self.k_star_mpc_inverse * coordinate

    def box_size_mpc(self, dimensionless_box_size: float) -> float:
        """Convert a dimensionless box length to Mpc."""

        size = _finite_positive(dimensionless_box_size, "dimensionless_box_size")
        return size / self.k_star_mpc_inverse

    def box_size_h_inverse_mpc(self, dimensionless_box_size: float) -> float:
        """Return the numerical coefficient of the box size in ``h^-1 Mpc``."""

        return self.hubble_h * self.box_size_mpc(dimensionless_box_size)


@dataclass(frozen=True)
class PowerLawSpectrum:
    """Power-law vacuum curvature spectrum tied to a physical pivot scale."""

    amplitude: float
    spectral_index: float = 1.0
    pivot_mpc_inverse: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "amplitude",
            _finite_positive(self.amplitude, "amplitude"),
        )
        spectral_index = float(self.spectral_index)
        if not np.isfinite(spectral_index):
            raise ValueError("spectral_index must be finite")
        object.__setattr__(self, "spectral_index", spectral_index)
        object.__setattr__(
            self,
            "pivot_mpc_inverse",
            _finite_positive(self.pivot_mpc_inverse, "pivot_mpc_inverse"),
        )

    def dimensionless_power(
        self,
        kappa: ArrayLike,
        scale: ScaleMapping,
    ) -> NDArray[np.float64]:
        """Evaluate ``Delta_vac^2(k_star*kappa)`` with a zero Fourier mode."""

        if not isinstance(scale, ScaleMapping):
            raise TypeError("scale must be a ScaleMapping")
        dimensionless = _finite_nonnegative_array(kappa, "kappa")
        result = np.zeros_like(dimensionless)
        nonzero = dimensionless > 0.0
        physical_k = scale.k_mpc_inverse(dimensionless[nonzero])
        result[nonzero] = self.amplitude * (
            physical_k / self.pivot_mpc_inverse
        ) ** (self.spectral_index - 1.0)
        return result

    def power(
        self,
        kappa: ArrayLike,
        scale: ScaleMapping,
    ) -> NDArray[np.float64]:
        """Backward-compatible alias for ``dimensionless_power``."""

        return self.dimensionless_power(kappa, scale)
