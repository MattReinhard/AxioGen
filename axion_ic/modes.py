"""Amplified gauge modes and their action on a helicity realization."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    import mpmath as mp
except ImportError:  # Keep Fourier-only imports usable without this optional backend.
    mp = None


def _require_mpmath() -> None:
    if mp is None:  # pragma: no cover - exercised only without dependency
        raise ImportError(
            "exact Coulomb modes require mpmath; install the project requirements"
        )


def _validate_xi(xi: float) -> float:
    xi = float(xi)
    if not np.isfinite(xi) or xi <= 0.0:
        raise ValueError("xi must be finite and positive")
    return xi


def _kummer_coulomb_h0_and_derivative(
    xi: float,
    x: float,
) -> tuple[complex, complex]:
    """Evaluate ``H_0^+ = G_0 + i F_0`` and its x derivative.

    The representation is

        H_0^+(xi, x) = C exp(i x) z U(a, 2, z),
        a = 1 + i xi,  z = -2 i x,

    where ``C = exp(pi xi/2) Gamma(1+i xi)/|Gamma(1+i xi)|``.
    Its derivative is evaluated analytically using
    ``dU(a,b,z)/dz = -a U(a+1,b+1,z)``.
    """

    a = 1.0 + 1j * xi
    z = -2j * x
    gamma = mp.gamma(1.0 + 1j * xi)
    normalization = mp.exp(mp.pi * xi / 2.0) * gamma / abs(gamma)

    hyperu = mp.hyperu(a, 2.0, z)
    shifted_hyperu = mp.hyperu(a + 1.0, 3.0, z)
    common = normalization * mp.exp(1j * x)

    h0 = common * z * hyperu
    dh0_dx = common * (
        (1j * z - 2j) * hyperu + 2j * a * z * shifted_hyperu
    )
    return complex(h0), complex(dh0_dx)


def coulomb_g0_and_derivative(
    xi: float,
    x: ArrayLike,
    *,
    decimal_precision: int = 40,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the exact real Coulomb mode ``G_0(xi,x)`` and ``dG_0/dx``.

    ``x`` must be strictly positive.  Array entries with repeated values are
    evaluated only once, which is useful because a Fourier lattice contains
    many wavevectors with the same magnitude.
    """

    _require_mpmath()
    xi = _validate_xi(xi)
    values = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("x must contain only finite, strictly positive values")
    if not isinstance(decimal_precision, int) or decimal_precision < 15:
        raise ValueError("decimal_precision must be an integer of at least 15")

    unique_values, inverse = np.unique(values, return_inverse=True)
    unique_g0 = np.empty(unique_values.size, dtype=float)
    unique_derivative = np.empty(unique_values.size, dtype=float)

    with mp.workdps(decimal_precision):
        for index, value in enumerate(unique_values):
            h0, dh0_dx = _kummer_coulomb_h0_and_derivative(xi, float(value))
            unique_g0[index] = h0.real
            unique_derivative[index] = dh0_dx.real

    shape = values.shape
    g0 = unique_g0[inverse].reshape(shape)
    derivative = unique_derivative[inverse].reshape(shape)
    return g0, derivative


def barnaby_peloso_mode_factors(
    xi: float,
    x: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the asymptotic mode factors used by Barnaby--Namba--Peloso.

    Their Appendix-A approximation is

        g_BP = (x / (2 xi))**(1/4) exp(pi xi - 2 sqrt(2 xi x)),
        A    = g_BP / sqrt(2 k),
        A'   = sqrt(k xi / x) g_BP.

    The second returned array is the effective factor ``d_BP`` satisfying
    ``A' = -sqrt(k/2) d_BP``.  It is therefore
    ``d_BP = -sqrt(2 xi/x) g_BP``.  It is intentionally the paper's
    separately approximated derivative, not the exact derivative of
    ``g_BP``.
    """

    xi = _validate_xi(xi)
    values = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("x must contain only finite, strictly positive values")
    mode = (values / (2.0 * xi)) ** 0.25 * np.exp(
        np.pi * xi - 2.0 * np.sqrt(2.0 * xi * values)
    )
    effective_derivative = -np.sqrt(2.0 * xi / values) * mode
    return (
        np.asarray(mode, dtype=float),
        np.asarray(effective_derivative, dtype=float),
    )


class BarnabyPelosoModeEvaluator:
    """Evaluator implementing the paper's asymptotic ``A`` and ``A'`` pair."""

    def evaluate(
        self,
        xi: float,
        x: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return factors compatible with ``gauge_mode_amplitudes``."""

        return barnaby_peloso_mode_factors(xi, x)


def tachyonic_mask(
    wavenumber: ArrayLike,
    tau: float,
    xi: float,
) -> NDArray[np.bool_]:
    """Select Fourier modes in the Reinhard et al. region ``0 < -k tau < 2 xi``."""

    xi = _validate_xi(xi)
    tau = float(tau)
    if not np.isfinite(tau) or tau >= 0.0:
        raise ValueError("tau must be finite and negative")
    k = np.asarray(wavenumber, dtype=float)
    if np.any(~np.isfinite(k)) or np.any(k < 0.0):
        raise ValueError("wavenumber must contain only finite nonnegative values")

    x = -k * tau
    return (x > 0.0) & (x < 2.0 * xi)


def gauge_mode_amplitudes(
    wavenumber: ArrayLike,
    tau: float,
    xi: float,
    *,
    decimal_precision: int = 40,
    mode_evaluator: object | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return tachyonically restricted ``A_-(tau,k)`` and ``A_-'(tau,k)``.

    With the default exact-Coulomb evaluator and inside
    ``0 < -k tau < 2 xi``,

        A_-  = G_0(xi,-k tau) / sqrt(2k),
        A_-' = -sqrt(k/2) dG_0(xi,x)/dx.

    A supplied evaluator can replace the two dimensionless factors, including
    the Barnaby--Peloso asymptotic ``A,A'`` pair.  Both arrays are zero outside
    the tachyonic region.  The sharp mask itself is not differentiated when
    constructing ``A_-'``.
    """

    k = np.asarray(wavenumber, dtype=float)
    active = tachyonic_mask(k, tau, xi)
    mode = np.zeros_like(k)
    derivative = np.zeros_like(k)

    if np.any(active):
        active_k = k[active]
        x = -active_k * float(tau)
        if mode_evaluator is None:
            g0, dg0_dx = coulomb_g0_and_derivative(
                xi,
                x,
                decimal_precision=decimal_precision,
            )
        else:
            evaluate = getattr(mode_evaluator, "evaluate", None)
            if evaluate is None:
                raise TypeError("mode_evaluator must provide evaluate(xi, x)")
            g0, dg0_dx = evaluate(xi, x)
        mode[active] = g0 / np.sqrt(2.0 * active_k)
        derivative[active] = -np.sqrt(active_k / 2.0) * dg0_dx

    return mode, derivative


def gauge_field_at_time(
    helicity_coefficients: ArrayLike,
    wavenumber: ArrayLike,
    tau: float,
    xi: float,
    *,
    decimal_precision: int = 40,
    mode_evaluator: object | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Multiply one fixed ``h_i(k)`` realization by the selected gauge modes.

    Reusing the same ``helicity_coefficients`` at every requested time keeps
    the unequal-time stochastic history coherent.  Direct Kummer evaluation
    is the default; a validated exact interpolator or diagnostic asymptotic
    evaluator can be supplied.
    """

    helicity = np.asarray(helicity_coefficients, dtype=np.complex128)
    k = np.asarray(wavenumber, dtype=float)
    if helicity.ndim < 2 or helicity.shape[0] != 3:
        raise ValueError("helicity_coefficients must have shape (3, ...)")
    if helicity.shape[1:] != k.shape:
        raise ValueError("wavenumber shape must match the Fourier grid")

    mode, derivative = gauge_mode_amplitudes(
        k,
        tau,
        xi,
        decimal_precision=decimal_precision,
        mode_evaluator=mode_evaluator,
    )
    return helicity * mode, helicity * derivative
