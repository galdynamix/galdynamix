"""Utility Functions."""

from functools import partial
from typing import TypeAlias, TypeVar, cast

import jax
from jax import lax
from jax._src.numpy.util import promote_args_inexact
from jax.scipy.special import sph_harm
from jaxtyping import ArrayLike, Shaped

import quaxed.numpy as jnp

import galax.typing as gt

BatchableIntLike: TypeAlias = Shaped[gt.IntLike, "*#batch"]

T = TypeVar("T", bound=ArrayLike)


@partial(jax.jit)
@partial(jax.numpy.vectorize, signature="(3)->(3)")
def cartesian_to_spherical(xyz: gt.Vec3, /) -> gt.Vec3:
    """Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    xyz : Array[float, (3,)]
        Cartesian coordinates in the form (x, y, z).

    Returns
    -------
    r_theta_phi : Array[float, (3,)]
        Spherical radius.
        Inclination (polar) angle in [0, pi] from North to South pole.
        Azimuthal angle in [-pi, pi]
    """
    r = jnp.sqrt(jnp.sum(xyz**2, axis=0))  # spherical radius
    # TODO: this is a hack to avoid the ambiguity at r==0. This should be done better.
    theta = jax.lax.select(
        r == 0, jnp.zeros_like(r), jnp.arccos(xyz[2] / r)
    )  # inclination angle
    phi = jnp.arctan2(xyz[1], xyz[0])  # azimuthal angle
    return jnp.array([r, theta, phi])


# TODO: replace with upstream, when available
def factorial(n: T) -> T:
    """Factorial helper function."""
    (n,) = promote_args_inexact("factorial", n)
    return cast("T", xp.where(n < 0, 0, lax.exp(lax.lgamma(n + 1))))


def psi_of_r(r: T) -> T:
    r""":math:`\psi(r) = (r-1)/(r+1)`.

    Equation 9 of Lowing et al. (2011).
    """
    return cast("T", (r - 1.0) / (r + 1.0))


# =============================================================================


@partial(jax.numpy.vectorize, signature="(n),(),()->(n)", excluded=(3,))
@partial(jax.jit, static_argnames=("m_max",))  # TODO: should l,m be static?
def _real_Ylm(theta: gt.VecN, l: gt.IntLike, m: gt.IntLike, m_max: int) -> gt.VecN:
    # TODO: sph_harm only supports scalars, even though it returns an array!
    theta = xp.atleast_1d(theta)
    return sph_harm(
        m, xp.atleast_1d(l), theta=xp.zeros_like(theta), phi=theta, n_max=m_max
    ).real


@partial(jax.jit, static_argnames=("m_max",))
def real_Ylm(
    theta: gt.FloatAnyShape, l: BatchableIntLike, m: BatchableIntLike, m_max: int = 100
) -> gt.FloatAnyShape:
    r"""Get the spherical harmonic :math:`Y_{lm}(\theta)` of the polar angle.

    This is different than the scipy (and thus JAX) convention, which is
    :math:`Y_{lm}(\theta, \phi)`.
    Note that scipy also uses the opposite convention for theta, phi where
    theta is the azimuthal angle and phi is the polar angle.

    Parameters
    ----------
    theta : Array[float, (*shape, N)]
        Polar angle in [0, pi].
    l, m : int | Array[int, ()]
        Spherical harmonic terms. l in [0,lmax], m in [0,l].
    m_max : int, optional
        Maximum order of the spherical harmonic expansion.

    Returns
    -------
    Array[float, (l, m, N)]
        Spherical harmonic. The shape is batched using the vectorization signature
        ``(n),(),()->(n)``.
    """
    # TODO: raise an error if m > m_max
    return _real_Ylm(theta, l, m, m_max)
