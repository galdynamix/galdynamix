"""Self-Consistent Field Potential."""

__all__ = ["compute_coeffs_discrete"]


from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

import galax.typing as gt
from .bfe_helper import phi_nl_vec
from .coeffs_helper import expansion_coeffs_Anl_discrete
from .gegenbauer import GegenbauerCalculator
from .utils import cartesian_to_spherical, real_Ylm


@partial(jax.jit, static_argnames=("nmax", "lmax", "gegenbauer"))
def compute_coeffs_discrete(
    xyz: Float[Array, "samples 3"],
    mass: Float[Array, "samples"],  # type: ignore[name-defined]
    *,
    nmax: gt.IntLike,
    lmax: gt.IntLike,
    r_s: gt.FloatQScalar,
    gegenbauer: GegenbauerCalculator | None = None,
) -> tuple[
    Float[Array, "{nmax}+1 {lmax}+1 {lmax}+1"],
    Float[Array, "{nmax}+1 {lmax}+1 {lmax}+1"],
]:
    """Compute expansion coefficients for the SCF potential.

    Compute the expansion coefficients for representing the density distribution
    of input points as a basis function expansion. The points, ``xyz``, are
    assumed to be samples from the density distribution.

    This is Equation 15 of Lowing et al. (2011).

    Parameters
    ----------
    xyz : Array[float, (n_samples, 3)]
        Samples from the density distribution.
        :todo:`unit support`
    mass : Array[float, (n_samples,)]
        Mass of each sample.
        :todo:`unit support`
    nmax : int
        Maximum value of ``n`` for the radial expansion.
    lmax : int
        Maximum value of ``l`` for the spherical harmonics.
    r_s : numeric
        Scale radius.
        :todo:`unit support`

    gegenbauer : GegenbauerCalculator, optional
        Gegenbauer calculator. This is used to compute the Gegenbauer
        polynomials efficiently. If not provided, a new calculator will be
        created.

    Returns
    -------
    Snlm : Array[float, (nmax+1, lmax+1, lmax+1)]
        The value of the cosine expansion coefficient.
    Tnlm : Array[float, (nmax+1, lmax+1, lmax+1)]
        The value of the sine expansion coefficient.
    """
    if gegenbauer is None:
        ggncalc = GegenbauerCalculator(nmax=nmax)
    elif gegenbauer.nmax < nmax:
        msg = "gegenbauer.nmax != nmax"
        raise ValueError(msg)
    else:
        ggncalc = gegenbauer

    rthetaphi = cartesian_to_spherical(xyz)
    r = rthetaphi[..., 0]
    theta = rthetaphi[..., 1]
    phi = rthetaphi[..., 2]
    s = r / r_s

    ns = jnp.arange(nmax + 1)[:, None]  # (n, l)
    ls = jnp.arange(lmax + 1)[None, :]  # (n, l)

    Anl_til = expansion_coeffs_Anl_discrete(ns, ls)  # (n, l)
    phinl = phi_nl_vec(s, ns, ls, ggncalc)  # (n, l, N)

    li, mi = jnp.tril_indices(lmax + 1)  # (l*(l+1)//2,)
    lm = jnp.zeros((lmax + 1, lmax + 1), dtype=int).at[li, mi].set(li)  # (l, m)
    ms = jnp.zeros((lmax + 1, lmax + 1), dtype=int).at[li, mi].set(mi)  # (l, m)
    # TODO: this is VERY SLOW. Can we do better?
    Ylm = real_Ylm(theta, lm, ms, m_max=100)  # (l, m, N)

    delta = jax.lax.select(ms == 0, jnp.ones_like(ms), jnp.zeros_like(ms))  # (l, m)
    mvalid = jnp.zeros((lmax + 1, lmax + 1)).at[li, mi].set(1)  # select m <= l

    tmp = (  # (n, l, m, N) using broadcasting
        mvalid[None, :, :, None]
        * (2 - delta[None, :, :, None])
        * Anl_til[:, :, None, None]
        * mass[None, None, None, :]
        * phinl[:, :, None, :]
        * Ylm[None, :, :, :]
    )
    Snlm = jnp.sum(tmp * jnp.cos(ms[None, :, :, None] * phi[None, None, None]), axis=-1)
    Tnlm = jnp.sum(tmp * jnp.sin(ms[None, :, :, None] * phi[None, None, None]), axis=-1)

    return Snlm, Tnlm
