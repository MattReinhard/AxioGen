"""Command-line generation of one inverse-decay curvature realization."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
import time
import tomllib
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .config import DimensionlessGrid, PowerLawSpectrum, ScaleMapping
from .coulomb_interpolation import (
    default_coulomb_table_path,
    load_or_build_coulomb_interpolator,
)
from .fourier import box_inverse
from .helicity import generate_helicity_field
from .integrate import integrate_inverse_decay_response_direct
from .modes import BarnabyPelosoModeEvaluator
from .realization import PhysicalCurvatureRealization, apply_physical_spectrum
from vacuum.normalization import ScalarPowerNormalization


@dataclass(frozen=True)
class IntegrationParameters:
    """Validated settings for the inverse-decay time integral."""

    late_time_x_min: float
    number_of_log_points: int
    include_shell_entries: bool = True
    shell_entry_offset_fraction: float = 0.0
    quadrature: str = "log_trapezoid"
    decimal_precision: int = 40
    dealias: bool = True
    padding_rule: str = "three_halves"
    mode_backend: str = "direct"
    coulomb_table: Path | None = None


@dataclass(frozen=True)
class RunParameters:
    """All validated inputs required to generate and save one realization."""

    grid: DimensionlessGrid
    xi: float
    scale: ScaleMapping
    normalization: ScalarPowerNormalization
    spectrum: PowerLawSpectrum
    integration: IntegrationParameters
    inverse_decay_seed: int
    vacuum_seed: int
    output: Path
    save_helicity: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable resolved configuration."""

        return {
            "grid": {
                "grid_size": self.grid.grid_size,
                "dimensionless_box_size": self.grid.box_size,
            },
            "physics": {
                "xi": self.xi,
                "target_total_amplitude": (
                    self.normalization.target_total_amplitude
                ),
                "f2": self.normalization.f2,
                "f2_source": self.normalization.f2_source,
                "vacuum_amplitude": self.spectrum.amplitude,
                "spectral_index": self.spectrum.spectral_index,
                "pivot_mpc_inverse": self.spectrum.pivot_mpc_inverse,
                "k_star_mpc_inverse": self.scale.k_star_mpc_inverse,
                "k_star_h_mpc_inverse": (
                    self.scale.k_star_mpc_inverse / self.scale.hubble_h
                ),
                "hubble_h": self.scale.hubble_h,
            },
            "integration": {
                "late_time_x_min": self.integration.late_time_x_min,
                "number_of_log_points": self.integration.number_of_log_points,
                "include_shell_entries": self.integration.include_shell_entries,
                "shell_entry_offset_fraction": (
                    self.integration.shell_entry_offset_fraction
                ),
                "quadrature": self.integration.quadrature,
                "decimal_precision": self.integration.decimal_precision,
                "dealias": self.integration.dealias,
                "padding_rule": self.integration.padding_rule,
                "mode_backend": self.integration.mode_backend,
                "coulomb_table": (
                    None
                    if self.integration.coulomb_table is None
                    else str(self.integration.coulomb_table)
                ),
            },
            "realization": {
                "inverse_decay_seed": self.inverse_decay_seed,
                "vacuum_seed": self.vacuum_seed,
                "output": str(self.output),
                "save_helicity": self.save_helicity,
            },
        }


@dataclass(frozen=True)
class GeneratedRun:
    """In-memory products and diagnostics from one completed run."""

    realization: PhysicalCurvatureRealization
    retained_gauge_mask: NDArray[np.bool_]
    helicity_coefficients: NDArray[np.complex128]
    elapsed_seconds: float
    maximum_real_space_imaginary_residual: float


def _table(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing required [{name}] table")
    return value


def _reject_unknown(table: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"unknown key(s) in [{name}]: {', '.join(unknown)}")


def _required(table: dict[str, Any], key: str, section: str) -> Any:
    if key not in table:
        raise ValueError(f"missing required parameter [{section}].{key}")
    return table[key]


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be true or false")
    return value


def load_run_parameters(path: str | Path) -> RunParameters:
    """Load and strictly validate one TOML run configuration."""

    parameter_path = Path(path).expanduser().resolve()
    with parameter_path.open("rb") as stream:
        document = tomllib.load(stream)
    if not isinstance(document, dict):  # pragma: no cover - guaranteed by tomllib
        raise ValueError("the TOML document must contain tables")
    _reject_unknown(document, {"grid", "physics", "integration", "realization"}, "root")

    grid_table = _table(document, "grid")
    physics = _table(document, "physics")
    integration = _table(document, "integration")
    realization = _table(document, "realization")

    _reject_unknown(
        grid_table,
        {"grid_size", "dimensionless_box_size"},
        "grid",
    )
    _reject_unknown(
        physics,
        {
            "xi",
            "target_total_amplitude",
            "f2",
            "spectral_index",
            "pivot_mpc_inverse",
            "k_star_h_mpc_inverse",
            "hubble_h",
        },
        "physics",
    )
    _reject_unknown(
        integration,
        {
            "late_time_x_min",
            "number_of_log_points",
            "include_shell_entries",
            "shell_entry_offset_fraction",
            "quadrature",
            "decimal_precision",
            "dealias",
            "padding_rule",
            "mode_backend",
            "coulomb_table",
        },
        "integration",
    )
    _reject_unknown(
        realization,
        {"inverse_decay_seed", "vacuum_seed", "output", "save_helicity"},
        "realization",
    )

    grid = DimensionlessGrid(
        grid_size=_integer(
            _required(grid_table, "grid_size", "grid"),
            "grid.grid_size",
            2,
        ),
        box_size=_positive_float(
            _required(grid_table, "dimensionless_box_size", "grid"),
            "grid.dimensionless_box_size",
        ),
    )
    xi = _positive_float(_required(physics, "xi", "physics"), "physics.xi")
    hubble_h = _positive_float(
        _required(physics, "hubble_h", "physics"),
        "physics.hubble_h",
    )
    scale = ScaleMapping.from_h_mpc_inverse(
        _positive_float(
            _required(physics, "k_star_h_mpc_inverse", "physics"),
            "physics.k_star_h_mpc_inverse",
        ),
        hubble_h,
    )
    target_total_amplitude = _positive_float(
        _required(physics, "target_total_amplitude", "physics"),
        "physics.target_total_amplitude",
    )
    if "f2" in physics:
        normalization = ScalarPowerNormalization(
            xi=xi,
            target_total_amplitude=target_total_amplitude,
            f2=_nonnegative_float(physics["f2"], "physics.f2"),
        )
    else:
        normalization = ScalarPowerNormalization.from_current_fit(
            xi,
            target_total_amplitude,
        )
    spectrum = normalization.vacuum_spectrum(
        spectral_index=float(_required(physics, "spectral_index", "physics")),
        pivot_mpc_inverse=_positive_float(
            _required(physics, "pivot_mpc_inverse", "physics"),
            "physics.pivot_mpc_inverse",
        ),
    )

    offset = float(integration.get("shell_entry_offset_fraction", 0.0))
    if not np.isfinite(offset) or not 0.0 <= offset < 1.0:
        raise ValueError(
            "integration.shell_entry_offset_fraction must satisfy 0 <= value < 1"
        )
    include_entries = _boolean(
        integration.get("include_shell_entries", True),
        "integration.include_shell_entries",
    )
    if offset > 0.0 and not include_entries:
        raise ValueError(
            "a positive shell_entry_offset_fraction requires shell entries"
        )
    quadrature = integration.get("quadrature", "log_trapezoid")
    if quadrature not in {"log_trapezoid", "log_gauss_legendre"}:
        raise ValueError(
            "integration.quadrature must be 'log_trapezoid' or "
            "'log_gauss_legendre'"
        )
    if quadrature == "log_gauss_legendre" and (
        include_entries or offset > 0.0
    ):
        raise ValueError(
            "integration.quadrature='log_gauss_legendre' requires "
            "include_shell_entries=false and shell_entry_offset_fraction=0"
        )
    padding_rule = integration.get("padding_rule", "three_halves")
    if padding_rule not in {"three_halves", "five_halves"}:
        raise ValueError(
            "integration.padding_rule must be 'three_halves' or 'five_halves'"
        )
    mode_backend = integration.get("mode_backend", "direct")
    if mode_backend not in {"direct", "interpolated", "approximate"}:
        raise ValueError(
            "integration.mode_backend must be 'direct', 'interpolated', "
            "or 'approximate'"
        )
    table_value = integration.get("coulomb_table")
    coulomb_table = None
    if table_value is not None:
        if not isinstance(table_value, str) or not table_value.strip():
            raise TypeError("integration.coulomb_table must be a nonempty path string")
        coulomb_table = Path(table_value).expanduser()
        if not coulomb_table.is_absolute():
            coulomb_table = parameter_path.parent / coulomb_table
        coulomb_table = coulomb_table.resolve()
    if mode_backend == "interpolated" and coulomb_table is None:
        coulomb_table = default_coulomb_table_path(
            xi,
            parameter_path.parent / "tables",
        ).resolve()
    if mode_backend != "interpolated" and coulomb_table is not None:
        raise ValueError(
            "integration.coulomb_table is only valid for the interpolated backend"
        )

    integration_parameters = IntegrationParameters(
        late_time_x_min=_positive_float(
            _required(integration, "late_time_x_min", "integration"),
            "integration.late_time_x_min",
        ),
        number_of_log_points=_integer(
            _required(integration, "number_of_log_points", "integration"),
            "integration.number_of_log_points",
            2,
        ),
        include_shell_entries=include_entries,
        shell_entry_offset_fraction=offset,
        quadrature=quadrature,
        decimal_precision=_integer(
            integration.get("decimal_precision", 40),
            "integration.decimal_precision",
            15,
        ),
        dealias=_boolean(integration.get("dealias", True), "integration.dealias"),
        padding_rule=padding_rule,
        mode_backend=mode_backend,
        coulomb_table=coulomb_table,
    )

    inverse_decay_seed = _integer(
        _required(realization, "inverse_decay_seed", "realization"),
        "realization.inverse_decay_seed",
        0,
    )
    vacuum_seed = _integer(
        _required(realization, "vacuum_seed", "realization"),
        "realization.vacuum_seed",
        0,
    )
    if inverse_decay_seed == vacuum_seed:
        raise ValueError(
            "realization.inverse_decay_seed and realization.vacuum_seed "
            "must be different"
        )
    output_value = _required(realization, "output", "realization")
    if not isinstance(output_value, str) or not output_value.strip():
        raise TypeError("realization.output must be a nonempty path string")
    output = Path(output_value).expanduser()
    if not output.is_absolute():
        output = parameter_path.parent / output
    output = output.resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("realization.output must end in .npz")

    return RunParameters(
        grid=grid,
        xi=xi,
        scale=scale,
        normalization=normalization,
        spectrum=spectrum,
        integration=integration_parameters,
        inverse_decay_seed=inverse_decay_seed,
        vacuum_seed=vacuum_seed,
        output=output,
        save_helicity=_boolean(
            realization.get("save_helicity", False),
            "realization.save_helicity",
        ),
    )


def generate_from_parameters(parameters: RunParameters) -> GeneratedRun:
    """Generate one complete realization without writing it to disk."""

    start = time.perf_counter()
    _, _, _, kappa = parameters.grid.wavevectors()
    helicity = generate_helicity_field(
        parameters.grid.grid_size,
        parameters.grid.box_size,
        rng=np.random.default_rng(parameters.inverse_decay_seed),
    )
    retained = np.any(helicity != 0.0, axis=0)
    settings = parameters.integration
    mode_evaluator = None
    if settings.mode_backend == "interpolated":
        if settings.coulomb_table is None:  # Defensive; resolved by the loader.
            raise ValueError("interpolated mode backend requires a Coulomb table")
        mode_evaluator = load_or_build_coulomb_interpolator(
            parameters.xi,
            settings.coulomb_table,
        )
    elif settings.mode_backend == "approximate":
        mode_evaluator = BarnabyPelosoModeEvaluator()
    response = integrate_inverse_decay_response_direct(
        helicity,
        kappa,
        parameters.grid.box_size,
        parameters.xi,
        late_time_x_min=settings.late_time_x_min,
        number_of_log_points=settings.number_of_log_points,
        retained_gauge_mask=retained,
        include_shell_entries=settings.include_shell_entries,
        shell_entry_offset_fraction=settings.shell_entry_offset_fraction,
        quadrature=settings.quadrature,
        decimal_precision=settings.decimal_precision,
        dealias=settings.dealias,
        padding_rule=settings.padding_rule,
        mode_evaluator=mode_evaluator,
    )
    physical = apply_physical_spectrum(
        response,
        parameters.grid,
        parameters.xi,
        parameters.scale,
        parameters.spectrum,
    )
    values_x = box_inverse(
        physical.dimensionless_curvature_k,
        parameters.grid.box_size,
    )
    imaginary_residual = float(np.max(np.abs(values_x.imag)))
    elapsed = time.perf_counter() - start
    return GeneratedRun(
        realization=physical,
        retained_gauge_mask=retained,
        helicity_coefficients=helicity,
        elapsed_seconds=elapsed,
        maximum_real_space_imaginary_residual=imaginary_residual,
    )


def save_generated_run(
    run: GeneratedRun,
    parameters: RunParameters,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically save a generated run as a compressed NumPy archive."""

    output = parameters.output
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output}; pass --overwrite to replace it"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run.realization
    metadata = dict(result.metadata)
    metadata.update(parameters.normalization.as_dict())
    metadata.update(
        {
            "inverse_decay_seed": parameters.inverse_decay_seed,
            "elapsed_seconds": run.elapsed_seconds,
            "maximum_real_space_imaginary_residual": (
                run.maximum_real_space_imaginary_residual
            ),
            "mode_backend": parameters.integration.mode_backend,
            "quadrature": parameters.integration.quadrature,
            "dealias": parameters.integration.dealias,
            "padding_rule": parameters.integration.padding_rule,
            "coulomb_table": (
                None
                if parameters.integration.coulomb_table is None
                else str(parameters.integration.coulomb_table)
            ),
        }
    )
    arrays: dict[str, Any] = {
        "response_k": result.response_k,
        "dimensionless_curvature_k": result.dimensionless_curvature_k,
        "physical_curvature_k_mpc3": result.physical_curvature_k_mpc3,
        "vacuum_power": result.vacuum_power,
        "kappa": result.kappa,
        "k_mpc_inverse": result.k_mpc_inverse,
        "k_h_mpc_inverse": result.k_h_mpc_inverse,
        "retained_gauge_mask": run.retained_gauge_mask,
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "resolved_parameters_json": np.asarray(
            json.dumps(parameters.as_dict(), sort_keys=True)
        ),
    }
    if parameters.save_helicity:
        arrays["helicity_coefficients"] = run.helicity_coefficients

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
            np.savez_compressed(temporary, **arrays)
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one axion-inflation inverse-decay realization."
    )
    parser.add_argument("parameter_file", type=Path, help="TOML parameter file")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the configured output file if it already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the resolved configuration without integrating",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        parameters = load_run_parameters(arguments.parameter_file)
        print(json.dumps(parameters.as_dict(), indent=2, sort_keys=True))
        if arguments.dry_run:
            print("Configuration is valid; no realization was generated.")
            return 0
        if parameters.output.exists() and not arguments.overwrite:
            raise FileExistsError(
                f"output already exists: {parameters.output}; "
                "pass --overwrite to replace it"
            )
        print("Generating coherent gauge realization and time integral...")
        run = generate_from_parameters(parameters)
        save_generated_run(run, parameters, overwrite=arguments.overwrite)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(f"Saved realization: {parameters.output}")
    print(f"Elapsed time: {run.elapsed_seconds:.3f} s")
    print(
        "Maximum real-space imaginary residual: "
        f"{run.maximum_real_space_imaginary_residual:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
