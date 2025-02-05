""":mod:`galax.dynamics.integrate`."""

__all__ = [
    "evaluate_orbit",
    "Integrator",
    "Interpolant",
    "parse_time_specification",
    "AbstractSolver",
    "DynamicsSolver",
    "InterpolatedPhaseSpacePosition",
    # Diffraxtra external library
    "DiffEqSolver",
    "VectorizedDenseInterpolation",
]

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics.integrate", RUNTIME_TYPECHECKER):
    from diffraxtra import DiffEqSolver, VectorizedDenseInterpolation

    from ._src.dynamics import DynamicsSolver, parse_time_specification
    from ._src.integrate import Integrator, Interpolant, InterpolatedPhaseSpacePosition
    from ._src.orbit import evaluate_orbit
    from ._src.solver import AbstractSolver

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER
