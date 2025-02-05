"""Compatibility."""

__all__: list[str] = []

from typing import Literal

from astropy.units import Quantity as APYQuantity
from plum import convert, dispatch

import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
import galax.typing as gt

# =============================================================================
# evaluate_orbit


@dispatch
def evaluate_orbit(
    pot: gp.AbstractPotential,
    w0: gc.PhaseSpacePosition | gt.BtSz6,
    t: APYQuantity,
    /,
    *,
    solver: gd.DynamicsSolver | None = None,
    dense: Literal[True, False] = False,
) -> gd.Orbit:
    """Compute an orbit in a potential.

    This is the Astropy-compatible version of the function.

    Examples
    --------
    We start by integrating a single orbit in the potential of a point mass.  A
    few standard imports are needed:

    >>> import numpy as np
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    We can then create the point-mass' potential, with galactic units:

    >>> potential = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    We can then integrate an initial phase-space position in this potential to
    get an orbit:

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([10., 0., 0.], "kpc"),
    ...                            p=u.Quantity([0., 200, 0.], "km/s"),
    ...                            t=u.Quantity(-100, "Myr"))
    >>> ts = u.Quantity(np.linspace(0., 1., 4), "Gyr")

    >>> orbit = gd.evaluate_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
      q=CartesianPos3D(...), p=CartesianVel3D(...),
      t=Quantity['time'](Array(..., dtype=float64), unit='Myr'),
      frame=SimulationFrame(),
      potential=KeplerPotential(...),
      interpolant=None
    )

    >>> ts = u.Quantity(np.linspace(0., 1., 10), "Gyr")
    >>> orbit = gd.evaluate_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
      q=CartesianPos3D(...), p=CartesianVel3D(...),
      t=Quantity['time'](Array(..., dtype=float64), unit='Myr'),
      frame=SimulationFrame(),
      potential=KeplerPotential(...),
      interpolant=None
    )

    We can also integrate a batch of orbits at once:

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([[10., 0, 0], [10., 0, 0]], "kpc"),
    ...                            p=u.Quantity([[0, 200, 0], [0, 220, 0]], "km/s"),
    ...                            t=u.Quantity([-100, -150], "Myr"))
    >>> orbit = gd.evaluate_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
      q=CartesianPos3D(
        x=Quantity[PhysicalType('length')](value=f64[2,10], unit=Unit("kpc")),
        ...
      ),
      p=CartesianVel3D(...),
      t=Quantity['time'](Array(..., dtype=float64), unit='Myr'),
      frame=SimulationFrame(),
      potential=KeplerPotential(...),
      interpolant=None
    )

    :class:`~galax.dynamics.PhaseSpacePosition` has a ``t`` argument for the
    time at which the position is given. As noted earlier, this can be used to
    integrate from a different time than the initial time of the position:

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([10., 0., 0.], "kpc"),
    ...                            p=u.Quantity([0., 200, 0.], "km/s"),
    ...                            t=u.Quantity(0, "Myr"))
    >>> ts = u.Quantity(np.linspace(0.3, 1.0, 8), "Gyr")
    >>> orbit = gd.evaluate_orbit(potential, w0, ts)
    >>> orbit.q[0]  # doctest: +SKIP
    Array([ 9.779, -0.3102,  0.        ], dtype=float64)

    """
    orbit: gd.Orbit = gd.evaluate_orbit(
        pot, w0, convert(t, u.Quantity), solver=solver, dense=dense
    )
    return orbit
