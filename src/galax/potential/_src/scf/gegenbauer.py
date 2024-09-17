"""Gegenbauer polynomials."""

__all__ = ["GegenbauerCalculator"]

from functools import partial
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import sympy as sp
import sympy2jax
from jax.scipy.special import gamma

from .utils import factorial
from galax.typing import FloatLike, IntLike, VecN
from galax.utils._jax import vectorize_method


@runtime_checkable
class AbstractGegenbauerDerivativeTerm(Protocol):
    def __call__(self, x: VecN, alpha: float) -> VecN: ...


def _compute_weight_function_derivatives(
    nmax: int,
) -> tuple[AbstractGegenbauerDerivativeTerm, ...]:
    """Compute the nth derivative of the weight function for the Gegenbauer polynomials.

    .. warning::

        This function is still VERY slow for nmax > 7.

    .. todo::

        This isn't quite Equation 22.2.3 of Abramowitz & Stegun (1972).

    It's too costly to have JAX compute the Gegenbauer polynomials directly, so
    we instead compute the nth derivative of the weight function, and then
    use the recurrence relation to compute the Gegenbauer polynomials.
    """
    # Make Symbols
    x: sp.Symbol = sp.Symbol("x")
    n: sp.Symbol = sp.Symbol("n", integer=True)
    alpha: sp.Symbol = sp.Symbol("alpha", positive=True)

    # weight function for the Gegenbauer polynomials
    weight = (1 - x**2) ** (n + alpha - 0.5)

    # Compute the nth derivative term for the Gegenbauer polynomials
    # start with the 0th derivative
    fn_sympy: sp.Expr = sp.simplify(sp.diff(weight, x, 0))
    func_list = []
    for i in range(nmax + 1):
        # Symbolic computation of the nth derivative
        if i > 0:
            fn_sympy = sp.simplify(sp.diff(fn_sympy, x, 1))
        # Convert to a JAX function
        fn_jax = sympy2jax.SymbolicModule(fn_sympy)
        # Re-arrange the arguments
        fn_jax_save = lambda x, alpha, n=i, fn=fn_jax: fn(  # noqa: E731
            x=x, alpha=alpha, n=n
        )
        func_list.append(fn_jax_save)
    return tuple(func_list)


class GegenbauerCalculator(eqx.Module):  # type: ignore[misc]
    """Gegenbauer (ultraspherical) polynomial."""

    nmax: int
    """Maximum order of the Gegenbauer polynomials."""

    _weights: tuple[AbstractGegenbauerDerivativeTerm, ...] = eqx.field(
        init=False, static=True, repr=False
    )
    """Tuple of weights for nth derivative of the Gegenbauer polynomials.

    This is computed in the __post_init__ method.
    """

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_weights", _compute_weight_function_derivatives(self.nmax)
        )

    # TODO: switch requires integer n, but everything else already vectorized
    @partial(jax.jit, static_argnames=("n", "alpha"))
    @vectorize_method(signature="(),(),()->()")
    def __call__(
        self, n: IntLike, alpha: IntLike | FloatLike, x: FloatLike | VecN
    ) -> FloatLike | VecN:
        r"""Calculate :math:`C_n^\alpha(x)`."""
        # # TODO: the Gegenbauer polynomials have limits on valid inputs
        nth_deriv = jax.lax.switch(n, self._weights, x, alpha)

        # TODO: write out full mathematical derivation
        factor0 = ((-1.0) ** n) / ((2**n) * factorial(n))
        factor1 = (gamma(alpha + 0.5) * gamma(n + 2.0 * alpha)) / (
            gamma(2.0 * alpha) * gamma(alpha + n + 0.5)
        )
        factor2 = (1.0 - x**2) ** (-alpha + 0.5)
        return factor0 * factor1 * factor2 * nth_deriv
