"""galax: Galactic Dynamix in Jax."""

__all__ = ["ChenStreamDF"]


import warnings
from functools import partial
from typing import final

import jax
import jax.random as jr
from jaxtyping import PRNGKeyArray

import quaxed.array_api as xp
import quaxed.numpy as qnp

import galax.potential as gp
import galax.typing as gt
from .base import AbstractStreamDF
from .fardal15 import tidal_radius

# ============================================================
# Constants

mean = qnp.array([1.6, -30, 0, 1, 20, 0])

cov = qnp.array(
    [
        [0.1225, 0, 0, 0, -4.9, 0],
        [0, 529, 0, 0, 0, 0],
        [0, 0, 144, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [-4.9, 0, 0, 0, 400, 0],
        [0, 0, 0, 0, 0, 484],
    ]
)

# ============================================================


@final
class ChenStreamDF(AbstractStreamDF):
    """Chen Stream Distribution Function.

    A class for representing the Chen+2024 distribution function for
    generating stellar streams based on Chen et al. 2024
    https://ui.adsabs.harvard.edu/abs/2024arXiv240801496C/abstract
    """

    def __init__(self) -> None:
        super().__init__()
        warnings.warn(
            'Currently only the "no progenitor" version '
            "of the Chen+24 model is supported!",
            RuntimeWarning,
            stacklevel=1,
        )

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

        # x_new-hat
        r = xp.linalg.vector_norm(x, axis=-1, keepdims=True)
        x_new_hat = x / r

        # z_new-hat
        L_vec = qnp.cross(x, v)
        z_new_hat = L_vec / xp.linalg.vector_norm(L_vec, axis=-1, keepdims=True)

        # y_new-hat
        phi_vec = v - xp.sum(v * x_new_hat, axis=-1, keepdims=True) * x_new_hat
        y_new_hat = phi_vec / xp.linalg.vector_norm(phi_vec, axis=-1, keepdims=True)

        r_tidal = tidal_radius(potential, x, v, prog_mass, t)

        # Bill Chen: method="cholesky" doesn't work here!
        posvel = jr.multivariate_normal(
            key, mean, cov, shape=r_tidal.shape, method="svd"
        )

        Dr = posvel[:, 0] * r_tidal

        v_esc = qnp.sqrt(2 * potential.constants["G"] * prog_mass / Dr)
        Dv = posvel[:, 3] * v_esc

        # convert degrees to radians
        phi = posvel[:, 1] * 0.017453292519943295
        theta = posvel[:, 2] * 0.017453292519943295
        alpha = posvel[:, 4] * 0.017453292519943295
        beta = posvel[:, 5] * 0.017453292519943295

        # Trailing arm
        x_trail = (
            x
            + (Dr * qnp.cos(theta) * qnp.cos(phi))[:, qnp.newaxis] * x_new_hat
            + (Dr * qnp.cos(theta) * qnp.sin(phi))[:, qnp.newaxis] * y_new_hat
            + (Dr * qnp.sin(theta))[:, qnp.newaxis] * z_new_hat
        )
        v_trail = (
            v
            + (Dv * qnp.cos(beta) * qnp.cos(alpha))[:, qnp.newaxis] * x_new_hat
            + (Dv * qnp.cos(beta) * qnp.sin(alpha))[:, qnp.newaxis] * y_new_hat
            + (Dv * qnp.sin(beta))[:, qnp.newaxis] * z_new_hat
        )

        # Leading arm
        x_lead = (
            x
            - (Dr * qnp.cos(theta) * qnp.cos(phi))[:, qnp.newaxis] * x_new_hat
            - (Dr * qnp.cos(theta) * qnp.sin(phi))[:, qnp.newaxis] * y_new_hat
            + (Dr * qnp.sin(theta))[:, qnp.newaxis] * z_new_hat
        )
        v_lead = (
            v
            - (Dv * qnp.cos(beta) * qnp.cos(alpha))[:, qnp.newaxis] * x_new_hat
            - (Dv * qnp.cos(beta) * qnp.sin(alpha))[:, qnp.newaxis] * y_new_hat
            + (Dv * qnp.sin(beta))[:, qnp.newaxis] * z_new_hat
        )

        return x_lead, v_lead, x_trail, v_trail
