""":mod:`galax.dynamics`."""

__all__ = [
    # Modules
    "fields",
    "solve",
    "integrate",  # TODO: deprecate
    "mockstream",
    "plot",
    "cluster",
    # solve
    "evaluate_orbit",
    "Orbit",
    "AbstractSolver",
    "DynamicsSolver",
    # mockstream
    "MockStreamArm",
    "MockStream",
    "MockStreamGenerator",
    # mockstream.df
    "AbstractStreamDF",
    "FardalStreamDF",
    "ChenStreamDF",
    # functions
    "specific_angular_momentum",
    "lagrange_points",
    "tidal_radius",
    "omega",
]


from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics", RUNTIME_TYPECHECKER):
    from . import cluster, fields, integrate, mockstream, plot
    from ._src.api import omega, specific_angular_momentum
    from ._src.cluster.funcs import lagrange_points, tidal_radius
    from ._src.orbit import Orbit
    from .mockstream import (
        AbstractStreamDF,
        ChenStreamDF,
        FardalStreamDF,
        MockStream,
        MockStreamArm,
        MockStreamGenerator,
    )
    from .solve import AbstractSolver, DynamicsSolver, evaluate_orbit

    #
    # isort: split
    from ._src import register_api
    from ._src.dynamics import register_gc


# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER, register_api, register_gc
