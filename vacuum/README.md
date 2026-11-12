# Normalized vacuum curvature realizations

For a specified value of `xi`, the generator first determines the vacuum
amplitude from the vacuum-plus-source relation

$$
A_s=\mathcal P_{\mathrm{vac}}
  + f_2(\xi)e^{4\pi\xi}\mathcal P_{\mathrm{vac}}^2.
$$

It uses the stable positive solution

$$
\mathcal P_{\mathrm{vac}}
=\frac{2A_s}{1+\sqrt{1+4A_s f_2(\xi)e^{4\pi\xi}}}.
$$

The current default is the measured fit used by the project,

$$
f_2(\xi)=\frac{1.59\times10^{-4}}{\xi^{6.42}},
\qquad 1.8<\xi<3.
$$

This is the range over which the fit was obtained; the code does not enforce
it. An explicit value can instead be supplied as `f2` under `[physics]` in
the TOML file.

The generator draws real Gaussian white noise with variance
`1 / DeltaX**3`, transforms it with the project's finite-box convention, and
filters each nonzero Fourier mode by

$$
\sqrt{P_\zeta(\kappa)}
=\sqrt{\frac{2\pi^2}{\kappa^3}\mathcal P_\zeta(k_\star\kappa)}.
$$

The real-space draw automatically gives Hermitian Fourier coefficients.  The
isotropic real filter preserves this condition, so the final curvature field
is real without manually pairing Fourier modes.

Generate one realization with

```bash
python -m vacuum.generate \
    parameters.toml \
    --output outputs/vacuum/seed_20260805.npz
```

The grid size, `Ltilde`, `xi`, target total amplitude, optional explicit `f2`,
`k_star`, `h`, `n_s`, and `k_pivot` are read from the same TOML file used by
the sourced generator. The shared loader derives the vacuum amplitude before
either generator runs. Pass `--save-real-space` to include the final
position-space field in the archive.
The vacuum draw uses `realization.vacuum_seed`; the sourced draw uses the
distinct `realization.inverse_decay_seed` value.
The saved metadata records the target total, vacuum, and sourced pivot
amplitudes and their fractions.
The physical Fourier coefficient is recovered from the stored dimensionless
coefficient through

$$
\zeta_{\mathbf k}^{\mathrm{physical}}
=\widetilde\zeta_{\boldsymbol\kappa}/k_\star^3.
$$
