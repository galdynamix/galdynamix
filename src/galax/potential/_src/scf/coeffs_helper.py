"""Self-Consistent Field Potential."""

__all__: list[str] = []

from typing import overload

from jax.scipy.special import gamma
from jaxtyping import Array, Float, Integer

import quaxed.numpy as jnp

from .utils import factorial


@overload
def normalization_Knl(n: int, l: int) -> float: ...


@overload
def normalization_Knl(n: Array, l: Array) -> Array: ...


def normalization_Knl(
    n: Integer[Array, "*#shape"] | int, l: Integer[Array, "*#shape"] | int
) -> Float[Array, "*shape"] | float:
    """SCF normalization factor.

    Parameters
    ----------
    n : int
        Radial expansion term.
    l : int
        Spherical harmonic term.

    Returns
    -------
    float
    """
    return 0.5 * n * (n + 4 * l + 3.0) + (l + 1) * (2 * l + 1)


def expansion_coeffs_Anl_discrete(
    n: Integer[Array, "*#shape"], l: Integer[Array, "*#shape"]
) -> Float[Array, "*shape"]:
    """Return normalization factor for the coefficients.

    Equation 16 of Lowing et al. (2011).

    Parameters
    ----------
    n : int
        Radial expansion term.
    l : int
        spherical harmonic term.

    Returns
    -------
    float
    """
    Knl = normalization_Knl(n=n, l=l)
    prefac = -(2 ** (8.0 * l + 6)) / (4 * jnp.pi * Knl)
    numerator = factorial(n) * (n + 2 * l + 1.5) * gamma(2 * l + 1.5) ** 2
    denominator = gamma(n + 4.0 * l + 3.0)
    return prefac * (numerator / denominator)
