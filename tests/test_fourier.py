import numpy as np

from axion_ic.fourier import box_forward, box_inverse, wavevector_grid


def test_box_transform_round_trip_for_vector_field():
    """The box forward and inverse transforms recover every vector component."""

    rng = np.random.default_rng(4312)
    values_x = rng.normal(size=(3, 6, 6, 6)) + 1j * rng.normal(
        size=(3, 6, 6, 6)
    )
    recovered_x = box_inverse(box_forward(values_x, box_size=4.3), box_size=4.3)
    np.testing.assert_allclose(recovered_x, values_x, rtol=2e-15, atol=2e-15)


def test_inverse_transform_has_paper_phase_sign():
    """A single Fourier coefficient produces the expected box plane wave.

    For the paper convention, a coefficient at integer mode ``m`` must give
    ``amplitude * exp(-2*pi*i*m.n/N) / V`` in position space.  This test
    catches both an incorrect exponential sign and a missing ``1 / V``
    normalization in ``box_inverse``.
    """

    grid_size = 7
    box_size = 3.2
    volume = box_size**3
    amplitude = 1.7 - 0.4j
    mode = (1, 2, -1)

    coefficients_k = np.zeros((grid_size,) * 3, dtype=complex)
    coefficients_k[mode[0], mode[1], mode[2] % grid_size] = amplitude
    values_x = box_inverse(coefficients_k, box_size)

    n0, n1, n2 = np.meshgrid(
        np.arange(grid_size),
        np.arange(grid_size),
        np.arange(grid_size),
        indexing="ij",
    )
    phase = -2j * np.pi * (
        mode[0] * n0 + mode[1] * n1 + mode[2] * n2
    ) / grid_size
    expected = amplitude * np.exp(phase) / volume
    np.testing.assert_allclose(values_x, expected, rtol=2e-15, atol=2e-15)


def test_parseval_normalization():
    """Position- and Fourier-space norms contain the expected box factors."""

    rng = np.random.default_rng(772)
    grid_size = 8
    box_size = 5.1
    spacing = box_size / grid_size
    volume = box_size**3
    values_x = rng.normal(size=(grid_size,) * 3)
    coefficients_k = box_forward(values_x, box_size)

    position_norm = spacing**3 * np.sum(np.abs(values_x) ** 2)
    fourier_norm = np.sum(np.abs(coefficients_k) ** 2) / volume
    np.testing.assert_allclose(position_norm, fourier_norm, rtol=2e-15)


def test_wavevector_grid_uses_periodic_box_spacing():
    """Wavevectors use 2*pi/L spacing and NumPy's signed mode ordering."""

    grid_size = 6
    box_size = 3.0
    kx, ky, kz, magnitude = wavevector_grid(grid_size, box_size)
    expected_one_dimensional = (2.0 * np.pi / box_size) * np.array(
        [0, 1, 2, -3, -2, -1]
    )

    np.testing.assert_allclose(kx[:, 0, 0], expected_one_dimensional)
    np.testing.assert_allclose(ky[0, :, 0], expected_one_dimensional)
    np.testing.assert_allclose(kz[0, 0, :], expected_one_dimensional)
    np.testing.assert_allclose(magnitude, np.sqrt(kx * kx + ky * ky + kz * kz))
