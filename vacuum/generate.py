"""Generate a Gaussian vacuum curvature realization from its power spectrum."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from axion_ic.config import DimensionlessGrid, PowerLawSpectrum, ScaleMapping
from axion_ic.fourier import box_forward, box_inverse
from axion_ic.generate import load_run_parameters

from .normalization import ScalarPowerNormalization


def ordinary_power_from_dimensionless(
    kappa: ArrayLike,
    dimensionless_power: ArrayLike,
) -> NDArray[np.float64]:
    """Return ``P(kappa)=2*pi**2*calP(kappa)/kappa**3`` with zero mean mode."""

    wavenumber = np.asarray(kappa, dtype=float)
    spectrum = np.asarray(dimensionless_power, dtype=float)
    try:
        wavenumber, spectrum = np.broadcast_arrays(wavenumber, spectrum)
    except ValueError as error:
        raise ValueError("kappa and dimensionless_power must broadcast") from error
    if np.any(~np.isfinite(wavenumber)) or np.any(wavenumber < 0.0):
        raise ValueError("kappa must contain finite nonnegative values")
    if np.any(~np.isfinite(spectrum)) or np.any(spectrum < 0.0):
        raise ValueError("dimensionless_power must be finite and nonnegative")

    ordinary_power = np.zeros(wavenumber.shape, dtype=float)
    nonzero = wavenumber > 0.0
    ordinary_power[nonzero] = (
        2.0 * np.pi**2 * spectrum[nonzero] / wavenumber[nonzero] ** 3
    )
    return ordinary_power


def draw_scalar_white_noise_k(
    rng: np.random.Generator,
    grid: DimensionlessGrid,
) -> NDArray[np.complex128]:
    """Draw scalar noise satisfying ``<|w_k|**2> = V`` mode by mode."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if not isinstance(grid, DimensionlessGrid):
        raise TypeError("grid must be a DimensionlessGrid")

    noise_x = rng.normal(
        loc=0.0,
        scale=grid.spacing ** (-1.5),
        size=(grid.grid_size,) * 3,
    )
    return box_forward(noise_x, grid.box_size)


@dataclass(frozen=True)
class VacuumCurvatureRealization:
    """One late-time Gaussian vacuum curvature realization."""

    dimensionless_curvature_k: NDArray[np.complex128]
    grid: DimensionlessGrid
    scale: ScaleMapping
    spectrum: PowerLawSpectrum
    normalization: ScalarPowerNormalization

    @property
    def kappa(self) -> NDArray[np.float64]:
        """Return the dimensionless wavenumber magnitude grid."""

        return self.grid.wavevectors()[3]

    @property
    def vacuum_power(self) -> NDArray[np.float64]:
        """Return the target dimensionless spectrum ``calP_zeta(k)``."""

        return self.spectrum.dimensionless_power(self.kappa, self.scale)

    @property
    def physical_curvature_k_mpc3(self) -> NDArray[np.complex128]:
        """Return Fourier coefficients in the physical ``Mpc**3`` convention."""

        return self.dimensionless_curvature_k / self.scale.k_star_mpc_inverse**3

    def real_space(self) -> NDArray[np.float64]:
        """Return the real dimensionless curvature field on the position grid."""

        values = box_inverse(self.dimensionless_curvature_k, self.grid.box_size)
        scale = max(1.0, float(np.max(np.abs(values.real))))
        residual = float(np.max(np.abs(values.imag)))
        if residual > 1.0e-10 * scale:
            raise ValueError(f"unexpected imaginary residual {residual:.3e}")
        return values.real.copy()

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the scalar information needed to interpret the saved field."""

        metadata = {
            "field": "Gaussian vacuum curvature perturbation",
            "grid_size": self.grid.grid_size,
            "dimensionless_box_size": self.grid.box_size,
            "dimensionless_momentum_spacing": self.grid.momentum_spacing,
            "k_star_mpc_inverse": self.scale.k_star_mpc_inverse,
            "k_star_h_mpc_inverse": (
                self.scale.k_star_mpc_inverse / self.scale.hubble_h
            ),
            "hubble_h": self.scale.hubble_h,
            "box_size_mpc": self.scale.box_size_mpc(self.grid.box_size),
            "box_size_h_inverse_mpc": self.scale.box_size_h_inverse_mpc(
                self.grid.box_size
            ),
            "vacuum_amplitude": self.spectrum.amplitude,
            "vacuum_spectral_index": self.spectrum.spectral_index,
            "vacuum_pivot_mpc_inverse": self.spectrum.pivot_mpc_inverse,
            "dimensionless_fourier_coefficient": "zeta_tilde_vac(kappa)",
            "physical_fourier_coefficient_units": "Mpc^3",
        }
        metadata.update(self.normalization.as_dict())
        return metadata


def generate_vacuum_realization(
    grid: DimensionlessGrid,
    scale: ScaleMapping,
    normalization: ScalarPowerNormalization,
    spectrum: PowerLawSpectrum,
    *,
    rng: np.random.Generator,
) -> VacuumCurvatureRealization:
    """Draw a Gaussian vacuum realization with normalized scalar power."""

    if not isinstance(grid, DimensionlessGrid):
        raise TypeError("grid must be a DimensionlessGrid")
    if not isinstance(scale, ScaleMapping):
        raise TypeError("scale must be a ScaleMapping")
    if not isinstance(normalization, ScalarPowerNormalization):
        raise TypeError("normalization must be a ScalarPowerNormalization")
    if not isinstance(spectrum, PowerLawSpectrum):
        raise TypeError("spectrum must be a PowerLawSpectrum")
    if not np.isclose(
        spectrum.amplitude,
        normalization.vacuum_amplitude,
        rtol=1.0e-14,
        atol=0.0,
    ):
        raise ValueError(
            "spectrum amplitude must equal the normalized vacuum amplitude"
        )

    kappa = grid.wavevectors()[3]
    dimensionless_power = spectrum.dimensionless_power(kappa, scale)
    ordinary_power = ordinary_power_from_dimensionless(
        kappa,
        dimensionless_power,
    )
    curvature_k = draw_scalar_white_noise_k(rng, grid)
    curvature_k *= np.sqrt(ordinary_power)
    curvature_k[0, 0, 0] = 0.0
    return VacuumCurvatureRealization(
        dimensionless_curvature_k=curvature_k,
        grid=grid,
        scale=scale,
        spectrum=spectrum,
        normalization=normalization,
    )


def save_vacuum_realization(
    realization: VacuumCurvatureRealization,
    output: Path,
    *,
    vacuum_seed: int,
    save_real_space: bool,
    overwrite: bool,
) -> None:
    """Atomically save one realization without duplicating deterministic grids."""

    output = output.expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output}; pass --overwrite to replace it"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(realization.metadata)
    metadata["vacuum_seed"] = int(vacuum_seed)
    arrays: dict[str, Any] = {
        "dimensionless_curvature_k": realization.dimensionless_curvature_k,
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if save_real_space:
        arrays["dimensionless_curvature_x"] = realization.real_space()

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.stem}.",
            suffix=".npz",
            dir=output.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            np.savez(temporary, **arrays)
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "parameter_file",
        type=Path,
        help=(
            "shared TOML configuration; grid size, dimensionless box size, "
            "xi, target amplitude, optional f2, spectral tilt, k_star, and h "
            "are read from this file"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-real-space", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line generator."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        shared = load_run_parameters(arguments.parameter_file)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    realization = generate_vacuum_realization(
        shared.grid,
        shared.scale,
        shared.normalization,
        shared.spectrum,
        rng=np.random.default_rng(shared.vacuum_seed),
    )
    save_vacuum_realization(
        realization,
        arguments.output,
        vacuum_seed=shared.vacuum_seed,
        save_real_space=arguments.save_real_space,
        overwrite=arguments.overwrite,
    )
    print(f"Saved vacuum realization: {arguments.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
