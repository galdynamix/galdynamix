"""Dynamics Solvers.

This is private API.

"""

__all__: list[str] = []


import diffrax as dfx

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import UncheckedQuantity as FastQ

import galax.coordinates as gc


@gc.AbstractOnePhaseSpacePosition.from_.dispatch  # type: ignore[misc,attr-defined]
def from_(
    cls: type[gc.AbstractOnePhaseSpacePosition],
    soln: dfx.Solution,
    *,
    frame: cx.frames.AbstractReferenceFrame,  # not dispatched on, but required
    units: u.AbstractUnitSystem,  # not dispatched on, but required
    unbatch_time: bool = False,
) -> gc.AbstractOnePhaseSpacePosition:
    """Convert a solution to a phase-space position.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...    r_s=u.Quantity(5, "kpc"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)
    >>> solver = gd.DynamicsSolver()  # defaults to Dopri8
    >>> w0 = gc.PhaseSpacePosition(
    ...     q=u.Quantity([[8, 0, 9], [9, 0, 3]], "kpc"),
    ...     p=u.Quantity([0, 220, 0], "km/s"),
    ...     t=u.Quantity(0, "Gyr"))
    >>> t1 = u.Quantity(1, "Gyr")
    >>> soln = solver.solve(field, w0, t1)

    >>> w = gc.PhaseSpacePosition.from_(soln, units=pot.units, frame=w0.frame)
    >>> print(w)
    PhaseSpacePosition(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [[[-5.151 -6.454 -5.795]]
             [[ 4.277  4.633  1.426]]]>,
        p=<CartesianVel3D (x[kpc / Myr], y[kpc / Myr], z[kpc / Myr])
            [[[ 0.225 -0.068  0.253]]
             [[-0.439 -0.002 -0.146]]]>,
        t=Quantity['time'](Array([1000.], dtype=float64), unit='Myr'),
        frame=SimulationFrame())

    """
    # Reshape (T, *batch) to (*batch, T)
    t = soln.ts  # already in the correct shape
    q = jnp.moveaxis(soln.ys[0], 0, -2)
    p = jnp.moveaxis(soln.ys[1], 0, -2)

    # Reshape (*batch,T=1,6) to (*batch,6) if t is a scalar
    if unbatch_time and t.shape[-1] == 1:
        t = t[..., -1]
        q = q[..., -1, :]
        p = p[..., -1, :]

    # Convert the solution to a phase-space position
    return cls(
        q=cx.CartesianPos3D.from_(q, units["length"]),
        p=cx.CartesianVel3D.from_(p, units["speed"]),
        t=FastQ(t, units["time"]),
        frame=frame,
    )
