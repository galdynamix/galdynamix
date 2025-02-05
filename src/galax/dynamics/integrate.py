""":mod:`galax.dynamics.integrate`."""

__all__ = [
    "evaluate_orbit",
    "Integrator",
    "Interpolant",
    "parse_time_specification",
    "InterpolatedPhaseSpacePosition",
]

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics.integrate", RUNTIME_TYPECHECKER):
    from ._src.dynamics import parse_time_specification
    from ._src.integrate import (
        Integrator,
        Interpolant,
        InterpolatedPhaseSpacePosition,
        evaluate_orbit,
    )

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER
