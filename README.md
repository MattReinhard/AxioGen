# Curvature Perturbations Generator

This repository generates Fourier-space realizations of primordial curvature
perturbations on a periodic cubic grid. It contains two matched components:

- `axion_ic`: inverse-decay sourced curvature perturbations;
- `vacuum`: independent Gaussian vacuum curvature perturbations, including
  shared normalization of the vacuum and sourced amplitudes.

The code uses dimensionless simulation coordinates

$$
\mathbf X=k_\star\mathbf x,\qquad
\boldsymbol\kappa=\mathbf k/k_\star.
$$

The adopted grid and physical scale are defined in `parameters.toml`. They
obey

$$
L_{\mathrm{physical}}=\frac{\widetilde L}{k_\star}.
$$

For the checked-in configuration, the box represents `2000 h^-1 Mpc` and
its fundamental mode is `k_star`.

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

## Generate a vacuum realization

```bash
generate-vacuum-curvature \
    parameters.toml \
    --output outputs/vacuum_seed_20260805.npz
```

Equivalently, run `python -m vacuum.generate`. The command computes the
vacuum pivot amplitude from

$$
A_s=\mathcal P+f_2(\xi)e^{4\pi\xi}\mathcal P^2
$$

and draws a Gaussian field with dimensionless vacuum spectrum

$$
\mathcal P(k)=\mathcal P(k_0)(k/k_0)^{n_s-1}.
$$

The built-in fit is `f2(xi) = 1.59e-4 / xi^6.42` and was obtained for
`1.8 < xi < 3`. The code does not enforce this fit range. To use an explicit
value instead, set `f2` under `[physics]` in the TOML file.

The vacuum draw uses `realization.vacuum_seed` from the TOML file. The sourced
draw independently uses `realization.inverse_decay_seed`.

Both generators read the grid size, dimensionless box size, `xi`, target total
amplitude, optional explicit `f2`, spectral index, pivot, `k_star`, and `h`
from the same TOML file. The shared loader derives the vacuum amplitude once,
and both generators use that result. There are no separate vacuum CLI
overrides for these shared quantities. In particular, the adopted `L` and
`k_star` are defined once in `parameters.toml` under `[grid]` and `[physics]`,
respectively.

## Generate a sourced realization

Validate the included example configuration without running the integral:

```bash
generate-inverse-decay parameters.toml --dry-run
```

Run it with:

```bash
generate-inverse-decay parameters.toml
```

Equivalently, run `python -m axion_ic.generate parameters.toml`.
The checked-in production geometry uses `N=256` and
`dimensionless_box_size=2*pi`.

The dimensionless response is converted to the sourced curvature field using

$$
\widetilde\zeta_{\mathrm{inv}}(\boldsymbol\kappa)
=8\pi^2\xi\,\mathcal P(\xi)
\left(\frac{k_\star\kappa}{k_0}\right)^{(n_s-1)/2}
\mathscr R(\boldsymbol\kappa).
$$

The half exponent appears at field level so that the inverse-decay power,
like the vacuum power, contains exactly one factor of
$(k/k_0)^{n_s-1}$.

The example specifies `target_total_amplitude`; its `vacuum_amplitude` is a
derived quantity. The sourced and vacuum generators therefore cannot acquire
different amplitudes from the same TOML configuration.

## Output conventions

The stored dimensionless Fourier coefficient is
`dimensionless_curvature_k`. Its physical-convention counterpart is

$$
\mathcal R_{\mathbf k}
=\widetilde{\mathcal R}_{\boldsymbol\kappa}/k_\star^3.
$$

The corresponding power spectra satisfy

$$
P_{\mathcal R}(k)
=\widetilde P_{\mathcal R}(\kappa)/k_\star^3,
\qquad
\Delta_{\mathcal R}^2(k)
=\frac{\kappa^3}{2\pi^2}\widetilde P_{\mathcal R}(\kappa).
$$

The two commands currently save the vacuum and sourced components separately.
A total realization is their mode-by-mode sum, using matched grid and scale
settings and statistically independent random draws. A combined orchestration
command is not yet included.

The `[realization]` table requires separate `inverse_decay_seed` and
`vacuum_seed` values and rejects configurations in which they are equal.

## Tests

```bash
python -m pytest -q
```

The test suite contains focused generator and numerical-unit tests only. It
does not include notebooks, convergence campaigns, plotting scripts, or
production outputs.
