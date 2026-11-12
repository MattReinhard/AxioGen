import json

import numpy as np
import pytest

import axion_ic.generate as generate


def _write_parameters(
    path,
    *,
    output="result.npz",
    integration_extra="",
    extra="",
):
    """Write a minimal valid TOML configuration for CLI tests."""

    path.write_text(
        f"""
[grid]
grid_size = 3
dimensionless_box_size = 6.283185307179586

[physics]
xi = 2.5
target_total_amplitude = 2.1e-9
spectral_index = 0.965
pivot_mpc_inverse = 0.05
k_star_h_mpc_inverse = 0.002
hubble_h = 0.68

[integration]
late_time_x_min = 0.09
number_of_log_points = 8
include_shell_entries = true
shell_entry_offset_fraction = 1e-5
decimal_precision = 25
dealias = true
{integration_extra}

[realization]
inverse_decay_seed = 1234
vacuum_seed = 5678
output = "{output}"
save_helicity = false
{extra}
""",
        encoding="utf-8",
    )


def test_load_run_parameters_resolves_and_converts_units(tmp_path):
    """TOML values become validated objects and a config-relative output."""

    parameter_file = tmp_path / "parameters.toml"
    _write_parameters(parameter_file, output="outputs/test.npz")
    parameters = generate.load_run_parameters(parameter_file)

    assert parameters.grid.grid_size == 3
    assert parameters.inverse_decay_seed == 1234
    assert parameters.vacuum_seed == 5678
    assert parameters.output == (tmp_path / "outputs/test.npz").resolve()
    np.testing.assert_allclose(parameters.scale.k_star_mpc_inverse, 0.002 * 0.68)
    assert parameters.integration.number_of_log_points == 8
    assert parameters.integration.shell_entry_offset_fraction == 1.0e-5
    assert parameters.integration.quadrature == "log_trapezoid"
    assert parameters.integration.padding_rule == "three_halves"
    assert parameters.normalization.target_total_amplitude == 2.1e-9
    assert (
        parameters.normalization.f2_source
        == "monomial_1p59e-4_xi_minus_6p42"
    )
    assert (
        parameters.spectrum.amplitude
        == parameters.normalization.vacuum_amplitude
    )


def test_unknown_parameter_is_rejected(tmp_path):
    """Misspelled or unsupported keys cannot be silently ignored."""

    parameter_file = tmp_path / "parameters.toml"
    _write_parameters(parameter_file, extra="mystery_option = 4")

    with pytest.raises(ValueError, match="unknown key"):
        generate.load_run_parameters(parameter_file)


def test_equal_component_seeds_are_rejected(tmp_path):
    parameter_file = tmp_path / "parameters.toml"
    _write_parameters(parameter_file)
    contents = parameter_file.read_text(encoding="utf-8").replace(
        "vacuum_seed = 5678",
        "vacuum_seed = 1234",
    )
    parameter_file.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="must be different"):
        generate.load_run_parameters(parameter_file)


def test_precomputed_vacuum_amplitude_is_rejected(tmp_path):
    """The TOML cannot bypass the shared scalar-power normalization."""

    parameter_file = tmp_path / "parameters.toml"
    _write_parameters(parameter_file)
    contents = parameter_file.read_text(encoding="utf-8").replace(
        "target_total_amplitude = 2.1e-9",
        "vacuum_amplitude = 2.020332015066137e-9",
    )
    parameter_file.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="unknown key.*vacuum_amplitude"):
        generate.load_run_parameters(parameter_file)


def test_explicit_f2_in_toml_overrides_builtin_fit(tmp_path):
    parameter_file = tmp_path / "parameters.toml"
    _write_parameters(parameter_file)
    contents = parameter_file.read_text(encoding="utf-8")
    contents = contents.replace("xi = 2.5", "xi = 1.5")
    contents = contents.replace(
        "target_total_amplitude = 2.1e-9",
        "target_total_amplitude = 2.1e-9\nf2 = 1.25e-7",
    )
    parameter_file.write_text(contents, encoding="utf-8")

    parameters = generate.load_run_parameters(parameter_file)

    assert parameters.normalization.xi == 1.5
    assert parameters.normalization.f2 == 1.25e-7
    assert parameters.normalization.f2_source == "explicit"
    assert parameters.spectrum.amplitude == parameters.normalization.vacuum_amplitude


def test_builtin_f2_fit_accepts_arbitrary_positive_xi(tmp_path):
    parameter_file = tmp_path / "parameters.toml"
    _write_parameters(parameter_file)
    contents = parameter_file.read_text(encoding="utf-8").replace(
        "xi = 2.5",
        "xi = 1.0",
    )
    parameter_file.write_text(contents, encoding="utf-8")

    parameters = generate.load_run_parameters(parameter_file)

    assert parameters.normalization.xi == 1.0
    assert parameters.normalization.f2 == pytest.approx(1.59e-4)
    assert parameters.normalization.f2_source == "monomial_1p59e-4_xi_minus_6p42"


def test_generate_and_save_archive_without_expensive_integral(tmp_path, monkeypatch):
    """The runner connects all stages and writes documented archive entries."""

    parameter_file = tmp_path / "parameters.toml"
    _write_parameters(parameter_file)
    parameters = generate.load_run_parameters(parameter_file)

    def fake_integrator(helicity, kappa, box_size, xi, **options):
        response = np.ones_like(kappa, dtype=complex) * (2.0 - 0.5j)
        response[0, 0, 0] = 0.0
        return response

    monkeypatch.setattr(
        generate,
        "integrate_inverse_decay_response_direct",
        fake_integrator,
    )
    run = generate.generate_from_parameters(parameters)
    generate.save_generated_run(run, parameters)

    with np.load(parameters.output, allow_pickle=False) as archive:
        expected = {
            "response_k",
            "dimensionless_curvature_k",
            "physical_curvature_k_mpc3",
            "vacuum_power",
            "kappa",
            "k_mpc_inverse",
            "k_h_mpc_inverse",
            "retained_gauge_mask",
            "metadata_json",
            "resolved_parameters_json",
        }
        assert set(archive.files) == expected
        metadata = json.loads(str(archive["metadata_json"]))
        resolved = json.loads(str(archive["resolved_parameters_json"]))
        assert metadata["inverse_decay_seed"] == 1234
        assert metadata["mode_backend"] == "direct"
        assert metadata["quadrature"] == "log_trapezoid"
        assert metadata["dealias"] is True
        assert metadata["padding_rule"] == "three_halves"
        assert metadata["target_total_amplitude"] == 2.1e-9
        assert metadata["f2_source"] == "monomial_1p59e-4_xi_minus_6p42"
        assert metadata["vacuum_amplitude"] < metadata["target_total_amplitude"]
        assert resolved["integration"]["quadrature"] == "log_trapezoid"
        assert resolved["integration"]["padding_rule"] == "three_halves"
        assert resolved["physics"]["xi"] == 2.5
        assert resolved["physics"]["target_total_amplitude"] == 2.1e-9
        assert resolved["physics"]["vacuum_amplitude"] < 2.1e-9
        assert archive["response_k"][0, 0, 0] == 0.0

    with pytest.raises(FileExistsError):
        generate.save_generated_run(run, parameters)


def test_dry_run_validates_without_creating_output(tmp_path, capsys):
    """The CLI can inspect its resolved configuration without integrating."""

    parameter_file = tmp_path / "parameters.toml"
    _write_parameters(parameter_file)

    assert generate.main([str(parameter_file), "--dry-run"]) == 0
    assert not (tmp_path / "result.npz").exists()
    assert "Configuration is valid" in capsys.readouterr().out


def test_barnaby_peloso_backend_reaches_integrator_and_metadata(
    tmp_path,
    monkeypatch,
):
    """The BP configuration selects its evaluator without a Coulomb table."""

    parameter_file = tmp_path / "parameters.toml"
    _write_parameters(
        parameter_file,
        integration_extra='mode_backend = "approximate"',
    )
    parameters = generate.load_run_parameters(parameter_file)
    captured = {}

    def fake_integrator(helicity, kappa, box_size, xi, **options):
        captured.update(options)
        response = np.zeros_like(kappa, dtype=complex)
        return response

    monkeypatch.setattr(
        generate,
        "integrate_inverse_decay_response_direct",
        fake_integrator,
    )
    run = generate.generate_from_parameters(parameters)
    generate.save_generated_run(run, parameters)

    evaluator = captured["mode_evaluator"]
    assert evaluator.__class__.__name__ == "BarnabyPelosoModeEvaluator"
    with np.load(parameters.output, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        resolved = json.loads(str(archive["resolved_parameters_json"]))
    assert metadata["mode_backend"] == "approximate"
    assert metadata["coulomb_table"] is None
    assert resolved["integration"]["mode_backend"] == "approximate"


def test_barnaby_peloso_backend_rejects_coulomb_table(tmp_path):
    """A BP run cannot silently carry an unused Coulomb-table setting."""

    parameter_file = tmp_path / "parameters.toml"
    _write_parameters(
        parameter_file,
        integration_extra=(
            'mode_backend = "approximate"\n'
            'coulomb_table = "table.npz"'
        ),
    )
    with pytest.raises(ValueError, match="only valid for the interpolated"):
        generate.load_run_parameters(parameter_file)


def test_log_gauss_legendre_quadrature_requires_no_shell_entries(tmp_path):
    """Global Gaussian quadrature cannot be combined with inserted shell nodes."""

    parameter_file = tmp_path / "parameters.toml"
    _write_parameters(
        parameter_file,
        integration_extra='quadrature = "log_gauss_legendre"',
    )

    with pytest.raises(ValueError, match="include_shell_entries=false"):
        generate.load_run_parameters(parameter_file)


def test_log_gauss_legendre_quadrature_reaches_integrator(tmp_path, monkeypatch):
    """The TOML quadrature setting is passed to the response integrator."""

    parameter_file = tmp_path / "parameters.toml"
    _write_parameters(
        parameter_file,
        integration_extra='quadrature = "log_gauss_legendre"',
    )
    text = parameter_file.read_text(encoding="utf-8")
    text = text.replace("include_shell_entries = true", "include_shell_entries = false")
    text = text.replace(
        "shell_entry_offset_fraction = 1e-5",
        "shell_entry_offset_fraction = 0.0",
    )
    parameter_file.write_text(text, encoding="utf-8")
    parameters = generate.load_run_parameters(parameter_file)
    captured = {}

    def fake_integrator(helicity, kappa, box_size, xi, **options):
        captured.update(options)
        response = np.zeros_like(kappa, dtype=complex)
        return response

    monkeypatch.setattr(
        generate,
        "integrate_inverse_decay_response_direct",
        fake_integrator,
    )
    generate.generate_from_parameters(parameters)

    assert captured["quadrature"] == "log_gauss_legendre"


def test_five_halves_padding_rule_reaches_integrator(tmp_path, monkeypatch):
    """The selected padding rule is passed through the production runner."""

    parameter_file = tmp_path / "parameters.toml"
    _write_parameters(
        parameter_file,
        integration_extra='padding_rule = "five_halves"',
    )
    parameters = generate.load_run_parameters(parameter_file)
    captured = {}

    def fake_integrator(helicity, kappa, box_size, xi, **options):
        captured.update(options)
        return np.zeros_like(kappa, dtype=complex)

    monkeypatch.setattr(
        generate,
        "integrate_inverse_decay_response_direct",
        fake_integrator,
    )
    generate.generate_from_parameters(parameters)

    assert parameters.integration.padding_rule == "five_halves"
    assert captured["padding_rule"] == "five_halves"


def test_invalid_padding_rule_is_rejected(tmp_path):
    parameter_file = tmp_path / "parameters.toml"
    _write_parameters(
        parameter_file,
        integration_extra='padding_rule = "double"',
    )

    with pytest.raises(ValueError, match="padding_rule"):
        generate.load_run_parameters(parameter_file)


def test_interpolated_backend_derives_table_path_and_builds_if_missing(
    tmp_path,
    monkeypatch,
):
    parameter_file = tmp_path / "parameters.toml"
    _write_parameters(
        parameter_file,
        integration_extra='mode_backend = "interpolated"',
    )
    parameters = generate.load_run_parameters(parameter_file)
    expected = tmp_path / "tables/coulomb_xi2p5.npz"
    assert parameters.integration.coulomb_table == expected.resolve()

    evaluator = object()
    captured = {}

    def fake_load_or_build(xi, path):
        captured["xi"] = xi
        captured["path"] = path
        return evaluator

    def fake_integrator(helicity, kappa, box_size, xi, **options):
        captured.update(options)
        return np.zeros_like(kappa, dtype=complex)

    monkeypatch.setattr(generate, "load_or_build_coulomb_interpolator", fake_load_or_build)
    monkeypatch.setattr(
        generate,
        "integrate_inverse_decay_response_direct",
        fake_integrator,
    )
    generate.generate_from_parameters(parameters)

    assert captured["xi"] == 2.5
    assert captured["path"] == expected.resolve()
    assert captured["mode_evaluator"] is evaluator
