import numpy as np

from axion_ic.fourier import box_inverse, wavevector_grid
from axion_ic.helicity import (
    generate_helicity_field,
    minus_helicity_projector,
)


def _negative_index_view(field):
    negative = (-np.arange(field.shape[1])) % field.shape[1]
    result = np.take(field, negative, axis=1)
    result = np.take(result, negative, axis=2)
    return np.take(result, negative, axis=3)


def _cross_with_k(field, kx, ky, kz):
    cross = np.empty_like(field)
    cross[0] = ky * field[2] - kz * field[1]
    cross[1] = kz * field[0] - kx * field[2]
    cross[2] = kx * field[1] - ky * field[0]
    return cross


def test_minus_helicity_projector_algebra():
    k_vector = np.array([1.3, -0.7, 2.1])
    magnitude = np.linalg.norm(k_vector)
    projector = minus_helicity_projector(k_vector)

    np.testing.assert_allclose(projector.conj().T, projector, atol=2e-15)
    np.testing.assert_allclose(projector @ projector, projector, atol=2e-15)
    np.testing.assert_allclose(np.trace(projector), 1.0, atol=2e-15)
    np.testing.assert_allclose(k_vector @ projector, 0.0, atol=2e-15)

    for column in range(3):
        projected_basis = projector[:, column]
        np.testing.assert_allclose(
            -1j * np.cross(k_vector, projected_basis),
            magnitude * projected_basis,
            atol=2e-15,
        )


def test_generated_field_is_real_transverse_and_minus_helicity():
    grid_size = 7
    box_size = 4.6
    field = generate_helicity_field(
        grid_size,
        box_size,
        rng=np.random.default_rng(9831),
    )
    kx, ky, kz, magnitude = wavevector_grid(grid_size, box_size)

    np.testing.assert_allclose(
        _negative_index_view(field),
        field.conj(),
        rtol=2e-14,
        atol=2e-14,
    )
    values_x = box_inverse(field, box_size)
    assert np.max(np.abs(values_x.imag)) < 2e-13

    field_norm = np.linalg.norm(magnitude * field)
    divergence = kx * field[0] + ky * field[1] + kz * field[2]
    assert np.linalg.norm(divergence) / field_norm < 2e-15

    curl = -1j * _cross_with_k(field, kx, ky, kz)
    assert np.linalg.norm(curl - magnitude * field) / field_norm < 2e-15
    np.testing.assert_array_equal(field[:, 0, 0, 0], 0.0)


def test_even_grid_nyquist_planes_are_zero():
    grid_size = 8
    field = generate_helicity_field(
        grid_size,
        box_size=5.0,
        rng=np.random.default_rng(123),
    )
    indices = np.rint(np.fft.fftfreq(grid_size) * grid_size).astype(int)
    nx, ny, nz = np.meshgrid(indices, indices, indices, indexing="ij")
    nyquist = (
        (np.abs(nx) == grid_size // 2)
        | (np.abs(ny) == grid_size // 2)
        | (np.abs(nz) == grid_size // 2)
    )
    np.testing.assert_array_equal(field[:, nyquist], 0.0)


def test_projected_noise_has_expected_covariance():
    grid_size = 5
    box_size = 2.3
    volume = box_size**3
    mode_index = (1, 1, 2)
    kx, ky, kz, _ = wavevector_grid(grid_size, box_size)
    k_vector = np.array(
        [
            kx[mode_index],
            ky[mode_index],
            kz[mode_index],
        ]
    )
    expected_covariance = volume * minus_helicity_projector(k_vector)

    rng = np.random.default_rng(24680)
    samples = np.empty((1600, 3), dtype=complex)
    for sample_index in range(samples.shape[0]):
        field = generate_helicity_field(
            grid_size,
            box_size,
            rng=rng,
        )
        samples[sample_index] = field[(slice(None),) + mode_index]

    sample_mean = np.mean(samples, axis=0)
    sample_covariance = np.einsum(
        "ni,nj->ij",
        samples,
        samples.conj(),
    ) / samples.shape[0]

    assert np.max(np.abs(sample_mean)) < 0.08 * np.sqrt(volume)
    np.testing.assert_allclose(
        sample_covariance,
        expected_covariance,
        rtol=0.08,
        atol=0.04 * volume,
    )
