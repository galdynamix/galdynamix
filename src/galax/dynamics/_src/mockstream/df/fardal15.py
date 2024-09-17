"""galax: Galactic Dynamix in Jax."""

__all__ = ["FardalStreamDF"]


from functools import partial
from typing import final

import jax
import jax.random as jr
from jaxtyping import PRNGKeyArray

import coordinax as cx
import quaxed.numpy as jnp

import galax.potential as gp
import galax.typing as gt
from .base import AbstractStreamDF
from galax.dynamics._src.funcs import _orbital_angular_frequency, tidal_radius

# ============================================================
# Constants

kr_bar = 2.0
kvphi_bar = 0.3

kz_bar = 0.0
kvz_bar = 0.0

sigma_kr = 0.5  # TODO: use actual Fardal values
sigma_kvphi = 0.5  # TODO: use actual Fardal values
sigma_kz = 0.5
sigma_kvz = 0.5

# ============================================================


@final
class FardalStreamDF(AbstractStreamDF):
    """Fardal Stream Distribution Function.

    A class for representing the Fardal+2015 distribution function for
    generating stellar streams based on Fardal et al. 2015
    https://ui.adsabs.harvard.edu/abs/2015MNRAS.452..301F/abstract
    """

    @partial(jax.jit, inline=True)
    def _sample(
        self,
        key: PRNGKeyArray,
        potential: gp.AbstractPotentialBase,
        x: gt.LengthBatchableVec3,
        v: gt.SpeedBatchableVec3,
        prog_mass: gt.BatchableFloatQScalar,
        t: gt.BatchableFloatQScalar,
    ) -> tuple[
        gt.LengthBatchVec3, gt.SpeedBatchVec3, gt.LengthBatchVec3, gt.SpeedBatchVec3
    ]:
        """Generate stream particle initial conditions."""
        # Random number generation
        key1, key2, key3, key4 = jr.split(key, 4)

        omega_val = _orbital_angular_frequency(x, v)[..., None]

        # r-hat
        r_hat = cx.normalize_vector(x)

        r_tidal = tidal_radius(potential, x, v, prog_mass, t)[..., None]
        v_circ = omega_val * r_tidal  # relative velocity

        # z-hat
        L_vec = jnp.linalg.cross(x, v)
        z_hat = cx.normalize_vector(L_vec)

        # phi-hat
        phi_vec = v - jnp.sum(v * r_hat, axis=-1, keepdims=True) * r_hat
        phi_hat = cx.normalize_vector(phi_vec)

        # k vals
        shape = r_tidal.shape
        kr_samp = kr_bar + jr.normal(key1, shape) * sigma_kr
        kvphi_samp = kr_samp * (kvphi_bar + jr.normal(key2, shape) * sigma_kvphi)
        kz_samp = kz_bar + jr.normal(key3, shape) * sigma_kz
        kvz_samp = kvz_bar + jr.normal(key4, shape) * sigma_kvz

        # Trailing arm
        x_trail = x + r_tidal * (kr_samp * r_hat + kz_samp * z_hat)
        v_trail = v + v_circ * (kvphi_samp * phi_hat + kvz_samp * z_hat)

        # Leading arm
        x_lead = x - r_tidal * (kr_samp * r_hat - kz_samp * z_hat)
        v_lead = v - v_circ * (kvphi_samp * phi_hat - kvz_samp * z_hat)

        return x_lead, v_lead, x_trail, v_trail