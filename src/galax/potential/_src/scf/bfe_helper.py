"""Self-Consistent Field Potential."""

__all__: list[str] = []

from functools import partial

import jax
from jaxtyping import Array, Float

from .coeffs_helper import normalization_Knl
from .utils import psi_of_r
from galax.potential._src.scf.gegenbauer import GegenbauerCalculator
from galax.typing import IntLike


@partial(jax.jit, static_argnames=("gegenbauer",))
def rho_nl(
    s: Float[Array, "N"], n: int, l: int, *, gegenbauer: GegenbauerCalculator
) -> Float[Array, "N"]:
    r"""Radial density expansion terms.

    Parameters
    ----------
    s : Array[float, (n,)]
        Scaled radius :math:`r/r_s`.
    n : int
        Radial expansion term.
    l : int
        Spherical harmonic term.
    gegenbauer : GegenbauerCalculator, keyword-only
        Gegenbauer calculator. This is used to compute the Gegenbauer
        polynomials efficiently.

    Returns
    -------
    Array[float, (n,)]
    """
    return (
        jnp.sqrt(4 * jnp.pi)
        * (normalization_Knl(n=n, l=l) / (2 * jnp.pi))
        * (s**l / (s * (1 + s) ** (2 * l + 3)))
        * gegenbauer(n, 2 * l + 1.5, psi_of_r(s))
    )


# ======================================================================


@partial(jax.jit, static_argnames=("gegenbauer",))
def phi_nl(
    s: Float[Array, "samples"], n: IntLike, l: IntLike, gegenbauer: GegenbauerCalculator
) -> Float[Array, "samples"]:
    r"""Angular density expansion terms.

    Parameters
    ----------
    s : Array[float, (n_samples,)]
        Scaled radius :math:`r/r_s`.
    n : int
        Radial expansion term.
    l : int
        Spherical harmonic term.
    gegenbauer : GegenbauerCalculator, keyword-only
        Gegenbauer calculator. This is used to compute the Gegenbauer
        polynomials efficiently.

    Returns
    -------
    Array[float, (n_samples,)]

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from galax.potential._src.scf.gegenbauer import GegenbauerCalculator
    >>> gegenbauer = GegenbauerCalculator(100)
    >>> phi_nl(0.5, 1, 1, gegenbauer=gegenbauer)
    Array(0.5, dtype=float32)
    >>> phi_nl(jnp.array([0.5, 0.5]), 1, 1, gegenbauer=gegenbauer)
    Array([0.5, 0.5], dtype=float32)
    """
    return (
        -jnp.sqrt(4 * jnp.pi)
        * (s**l / (1.0 + s) ** (2 * l + 1))
        * gegenbauer(n, 2 * l + 1.5, psi_of_r(s))
    )


phi_nl_vec = jnp.vectorize(phi_nl, signature="(n),(),()->(n)", excluded=(3,))

phi_nl_grad = jax.jit(jnp.vectorize(jax.grad(phi_nl, argnums=0), excluded=(3,)))
r"""Derivative :math:`\frac{d}{dr} \phi_{nl}`."""
