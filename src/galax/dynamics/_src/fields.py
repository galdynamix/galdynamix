"""Fields."""

__all__ = ["AbstractField"]

from abc import abstractmethod
from typing import Any

import diffrax as dfx
import equinox as eqx
from jaxtyping import PyTree
from plum import dispatch

import diffraxtra as dfxtra


class AbstractField(eqx.Module, strict=True):  # type: ignore[misc,call-arg]
    """Abstract base class for fields."""

    @abstractmethod
    def __call__(self, t: Any, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the field at time `t`."""
        raise NotImplementedError  # pragma: no cover

    @dispatch.abstract
    def terms(self, solver: Any, /) -> PyTree[dfx.AbstractTerm]:
        """Return the `diffrax.AbstractTerm` `jaxtyping.PyTree` for integration."""
        raise NotImplementedError  # pragma: no cover


# ==================================================================


@AbstractField.terms.dispatch  # type: ignore[misc]
def terms(
    self: AbstractField, wrapper: dfxtra.DiffEqSolver, /
) -> PyTree[dfx.AbstractTerm]:
    """Return diffeq terms, redispatching to the solver.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> solver = gd.solve.DiffEqSolver(dfx.Dopri8())

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> field.terms(solver)
    ODETerm(vector_field=<wrapped function __call__>)

    """
    return self.terms(wrapper.solver)
