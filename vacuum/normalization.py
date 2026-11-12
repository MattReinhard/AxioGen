"""Vacuum-plus-source normalization of scalar curvature power."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axion_ic.config import PowerLawSpectrum


# Current monomial fit for the sourced scalar amplitude.
F2_FIT_AMPLITUDE = 1.59e-4
F2_FIT_EXPONENT = 6.42
F2_FIT_NAME = "monomial_1p59e-4_xi_minus_6p42"


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def fitted_f2(xi: float) -> float:
    """Evaluate the current monomial fit for any finite positive ``xi``."""

    xi = _finite_positive(xi, "xi")
    return F2_FIT_AMPLITUDE / xi**F2_FIT_EXPONENT


@dataclass(frozen=True)
class ScalarPowerNormalization:
    """Scalar amplitudes satisfying the vacuum-plus-source relation.

    At the pivot scale, the amplitudes obey

    ``target = vacuum + f2 * exp(4*pi*xi) * vacuum**2``.
    """

    xi: float
    target_total_amplitude: float
    f2: float
    f2_source: str = "explicit"

    def __post_init__(self) -> None:
        object.__setattr__(self, "xi", _finite_positive(self.xi, "xi"))
        object.__setattr__(
            self,
            "target_total_amplitude",
            _finite_positive(
                self.target_total_amplitude,
                "target_total_amplitude",
            ),
        )
        object.__setattr__(self, "f2", _finite_nonnegative(self.f2, "f2"))
        if not isinstance(self.f2_source, str) or not self.f2_source.strip():
            raise ValueError("f2_source must be a nonempty string")

    @classmethod
    def from_current_fit(
        cls,
        xi: float,
        target_total_amplitude: float = 2.1e-9,
    ) -> "ScalarPowerNormalization":
        """Construct the normalization from the current monomial fit."""

        return cls(
            xi=xi,
            target_total_amplitude=target_total_amplitude,
            f2=fitted_f2(xi),
            f2_source=F2_FIT_NAME,
        )

    @property
    def source_coefficient(self) -> float:
        """Return ``f2*exp(4*pi*xi)``."""

        return self.f2 * float(np.exp(4.0 * np.pi * self.xi))

    @property
    def vacuum_amplitude(self) -> float:
        """Return the positive, numerically stable quadratic solution."""

        discriminant_root = np.sqrt(
            1.0
            + 4.0
            * self.source_coefficient
            * self.target_total_amplitude
        )
        return 2.0 * self.target_total_amplitude / (1.0 + discriminant_root)

    @property
    def sourced_amplitude(self) -> float:
        """Return the inverse-decay contribution at the pivot scale."""

        return self.source_coefficient * self.vacuum_amplitude**2

    @property
    def vacuum_fraction(self) -> float:
        return self.vacuum_amplitude / self.target_total_amplitude

    @property
    def sourced_fraction(self) -> float:
        return self.sourced_amplitude / self.target_total_amplitude

    def vacuum_spectrum(
        self,
        spectral_index: float = 0.965,
        pivot_mpc_inverse: float = 0.05,
    ) -> PowerLawSpectrum:
        """Return the power-law vacuum spectrum implied by this normalization."""

        return PowerLawSpectrum(
            amplitude=self.vacuum_amplitude,
            spectral_index=spectral_index,
            pivot_mpc_inverse=pivot_mpc_inverse,
        )

    def as_dict(self) -> dict[str, float | str]:
        """Return JSON-serializable normalization metadata."""

        return {
            "xi": self.xi,
            "target_total_amplitude": self.target_total_amplitude,
            "f2": self.f2,
            "f2_source": self.f2_source,
            "source_coefficient_f2_exp_4pi_xi": self.source_coefficient,
            "vacuum_amplitude": self.vacuum_amplitude,
            "sourced_amplitude": self.sourced_amplitude,
            "vacuum_fraction": self.vacuum_fraction,
            "sourced_fraction": self.sourced_fraction,
        }
