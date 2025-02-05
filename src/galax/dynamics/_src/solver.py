"""Dynamics Solvers.

This is private API.

"""

__all__ = ["AbstractSolver"]


import abc
from collections.abc import Mapping
from typing import Any

import equinox as eqx
from plum import dispatch


class AbstractSolver(eqx.Module, strict=True):  # type: ignore[call-arg,misc]
    """ABC for solvers.

    Notes
    -----
    The ``init``, ``step``, and ``solve`` methods are abstract and should be
    implemented by subclasses.

    """

    @abc.abstractmethod
    def init(self, *args: Any, **kwargs: Any) -> Any:
        """Initialize the solver."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, *args: Any, **kwargs: Any) -> Any:
        """Step the solver."""
        raise NotImplementedError

    @abc.abstractmethod
    def solve(self, *args: Any, **kwargs: Any) -> Any:
        """Solve, initializing and stepping to the solution."""
        raise NotImplementedError

    # ==================================================================
    # Convenience methods

    @classmethod
    @dispatch.abstract
    def from_(
        cls: "type[AbstractSolver]", *args: Any, **kwargs: Any
    ) -> "AbstractSolver":
        """Create a new solver from the argument."""
        raise NotImplementedError  # pragma: no cover


# =========================================================
# Constructors


@AbstractSolver.from_.dispatch
def from_(cls: type[AbstractSolver], solver: AbstractSolver) -> AbstractSolver:
    """Create a new solver from the argument.

    Examples
    --------
    >>> import galax.dynamics as gd
    >>> solver = gd.DynamicsSolver()
    >>> new_solver = gd.DynamicsSolver.from_(solver)
    >>> new_solver is solver
    True

    >>> class MySolver(AbstractSolver):
    ...     def init(self, *args, **kwargs): pass
    ...     def step(self, *args, **kwargs): pass
    ...     def solve(self, *args, **kwargs): pass
    >>> try: new_solver = MySolver.from_(solver)
    ... except TypeError as e: print(e)
    Cannot convert <class 'galax.dynamics...DynamicsSolver'> to <class '...MySolver'>

    """
    if not isinstance(solver, cls):
        msg = f"Cannot convert {type(solver)} to {cls}"
        raise TypeError(msg)

    return solver


@AbstractSolver.from_.dispatch(precedence=-1)
def from_(cls: type[AbstractSolver], obj: Any) -> AbstractSolver:
    """Pass argument to solver's init.

    Examples
    --------
    >>> import diffrax as dfx
    >>> from galax.dynamics.solve import DynamicsSolver, DiffEqSolver

    >>> DynamicsSolver.from_( DiffEqSolver(dfx.Dopri5()))
    DynamicsSolver(
      diffeqsolver=DiffEqSolver(
        solver=Dopri5(scan_kind=None),
        stepsize_controller=ConstantStepSize(),
        adjoint=RecursiveCheckpointAdjoint(checkpoints=None) ) )

    >>> DynamicsSolver.from_(dfx.Dopri5())
    DynamicsSolver(
      diffeqsolver=DiffEqSolver(
        solver=Dopri5(scan_kind=None),
        stepsize_controller=ConstantStepSize(),
        adjoint=RecursiveCheckpointAdjoint(checkpoints=None) ) )

    """
    return cls(obj)


@AbstractSolver.from_.dispatch
def from_(cls: type[AbstractSolver], obj: Mapping[str, Any]) -> AbstractSolver:
    """Create a new solver from the argument.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import galax.dynamics as gd

    >>> gd.DynamicsSolver.from_({})
    DynamicsSolver(
      diffeqsolver=DiffEqSolver(
        solver=Dopri8(scan_kind=None),
        stepsize_controller=PIDController( ... ),
        adjoint=RecursiveCheckpointAdjoint(checkpoints=None) ) )

    >>> gd.DynamicsSolver.from_({"diffeqsolver": dfx.Dopri5()})
    DynamicsSolver(
      diffeqsolver=DiffEqSolver(
        solver=Dopri5(scan_kind=None),
        stepsize_controller=ConstantStepSize(),
        adjoint=RecursiveCheckpointAdjoint(checkpoints=None) ) )

    """
    return cls(**obj)
