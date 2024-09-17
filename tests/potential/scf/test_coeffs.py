"""Test the Coefficient Calculations."""

import gala.potential as gp
import numpy as np

import quaxed.numpy as jnp

import galax.potential as gpx


def test_compute_coeffs_discrete():
    """Test the ``normalization_Knl`` function.

    .. todo::

        This test is not very good. It should be improved.
    """
    # Setup
    rng = np.random.default_rng(42)
    particle_xyz = rng.normal(0.0, 5.0, size=(3, 10_000))
    particle_xyz[2] = np.abs(particle_xyz[2])
    particle_xyz = jnp.array(particle_xyz)

    particle_mass = jnp.ones(particle_xyz.shape[1])
    particle_mass = 1e12 * particle_mass / particle_mass.sum()

    nmax = 2
    lmax = 3
    r_s = 10

    # Gala
    gala_Snlm, gala_Tnlm = gp.scf.compute_coeffs_discrete(
        np.array(particle_xyz), np.array(particle_mass), nmax=nmax, lmax=lmax, r_s=r_s
    )

    # Galdynamix
    Snlm, Tnlm = gpx.scf.compute_coeffs_discrete(
        particle_xyz, particle_mass, nmax=nmax, lmax=lmax, r_s=r_s
    )

    np.testing.assert_allclose(Snlm, gala_Snlm, rtol=1e-7)
    np.testing.assert_allclose(Tnlm, gala_Tnlm, rtol=1e-7)
