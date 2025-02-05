"""galax: Galactic Dynamix in Jax."""

__all__ = ["evaluate_orbit"]

from typing import Any, Literal

from plum import dispatch

import quaxed.numpy as jnp
from unxt.quantity import UncheckedQuantity as FastQ

import galax.coordinates as gc
import galax.dynamics._src.custom_types as gdt
import galax.potential as gp
import galax.typing as gt
from .interp import PhaseSpaceInterpolation
from .orbit import Orbit
from galax.dynamics._src.dynamics import DynamicsSolver, HamiltonianField

# TODO: enable setting the default integrator
_default_solver: DynamicsSolver = DynamicsSolver()


@dispatch
def evaluate_orbit(
    pot: gp.AbstractPotential,
    w0: gc.PhaseSpacePosition | gdt.BtQParr | gt.BtSz6,
    t: Any,
    /,
    *,
    solver: DynamicsSolver | None = None,
    dense: Literal[True, False] = False,
) -> Orbit:
    """Compute an orbit in a potential.

    :class:`~galax.coordinates.PhaseSpacePosition` includes a time in addition
    to the position (and velocity) information, enabling the orbit to be
    evaluated over a time range that is different from the initial time of the
    position.

    Parameters
    ----------
    pot : :class:`~galax.potential.AbstractPotential`
        The potential in which to integrate the orbit.
    w0 : PhaseSpacePosition | Array[number, (*batch, 6)]
        The phase-space position (includes velocity and time) from which to
        integrate. Integration includes the time of the initial position, so be
        sure to set the initial time to the desired value. See the `t` argument
        for more details.

        - :class:`~galax.dynamics.PhaseSpacePosition`[number, (*batch,)]:
            The full phase-space position, including position, velocity, and
            time. `w0` will be integrated from ``w0.t`` to ``t[0]``, then
            integrated from ``t[0]`` to ``t[1]``, returning the orbit calculated
            at `t`. If `w0.t` is `None`, it is assumed to be `t[0]`.
        - tuple[Array[number, (*batch, 3)], Array[number, (*batch, 3)]]:
            A :class:`~galax.coordinates.PhaseSpacePosition` will be
            constructed, interpreting the array as the  'q', 'p', with 't' set
            to ``t[0]``.
        - Array[number, (*batch, 6)]:
            A :class:`~galax.coordinates.PhaseSpacePosition` will be
            constructed, interpreting the array as the  'q', 'p' (each
            Array[number, (*batch, 3)]) arguments, with 't' set to ``t[0]``.
    t: Quantity[number, (time,), 'time']
        Array of times at which to compute the orbit. The first element should
        be the initial time and the last element should be the final time and
        the array should be monotonically moving from the first to final time.
        See the Examples section for options when constructing this argument.

        .. note::

            This is NOT the timesteps to use for integration, which are
            controlled by the `integrator`; the default integrator
            :class:`~galax.integrator.Integrator` uses adaptive timesteps.

    integrator : :class:`~galax.integrate.Integrator`, keyword-only
        Integrator to use.  If `None`, the default integrator
        :class:`~galax.integrator.Integrator` is used.  This integrator is used
        twice: once to integrate from `w0.t` to `t[0]` and then from `t[0]` to
        `t[1]`.

    dense: bool, optional keyword-only
        If `True`, return a dense (interpolated) orbit.  If `False`, return the
        orbit at the requested times.  Default is `False`.

    Returns
    -------
    orbit : :class:`~galax.dynamics.Orbit`
        The integrated orbit evaluated at the given times.

    Examples
    --------
    We start by integrating a single orbit in the potential of a point mass.  A
    few standard imports are needed:

    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    We can then create the point-mass' potential, with galactic units:

    >>> potential = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    We can then integrate an initial phase-space position in this potential to
    get an orbit:

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([10, 0, 0], "kpc"),
    ...                            p=u.Quantity([0, 200, 0], "km/s"),
    ...                            t=u.Quantity(-100, "Myr"))
    >>> ts = u.Quantity(jnp.linspace(0, 1, 4), "Gyr")
    >>> orbit = gd.evaluate_orbit(potential, w0, ts)
    >>> print(orbit)
    Orbit(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [[ 8.953 -1.324  0.   ]
             [ 9.035  1.276  0.   ]
             [ 2.644 -2.021  0.   ]
             [ 9.827 -0.562  0.   ]]>,
        p=<CartesianVel3D (x[kpc / Myr], y[kpc / Myr], z[kpc / Myr])
            [[ 0.322  0.181  0.   ]
             [-0.308  0.183  0.   ]
             [ 1.336 -0.247  0.   ]
             [ 0.126  0.201  0.   ]]>,
        t=UncheckedQuantity(Array([ ... ], dtype=float64), unit='Myr'),
        frame=SimulationFrame(),
        potential=KeplerPotential( ... ),
        interpolant=None)

    Note how there are 4 points in the orbit, corresponding to the 4 requested
    return times. These are the times at which the orbit is evaluated, not the
    times at which the orbit is integrated. The phase-space position `w0` is
    defined at `t=-100`, but the orbit is integrated from `t=0` to `t=1000`.
    Changing the number of times is easy:

    >>> ts = u.Quantity(jnp.linspace(0, 1, 10), "Gyr")
    >>> orbit = gd.evaluate_orbit(potential, w0, ts)
    >>> print(orbit)
    Orbit(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [[ 8.953 -1.324  0.   ]
             ...
             [ 9.827 -0.562  0.   ]]>,
        p=<CartesianVel3D (x[kpc / Myr], y[kpc / Myr], z[kpc / Myr])
            [[ 0.322  0.181  0.   ]
             ...
             [ 0.126  0.201  0.   ]]>,
        t=UncheckedQuantity(Array([ ... ], dtype=float64), unit='Myr'),
        frame=SimulationFrame(),
        potential=KeplerPotential( ... ),
        interpolant=None)

    Or evaluating at a single time:

    >>> orbit = gd.evaluate_orbit(potential, w0, u.Quantity(0.5, "Gyr"))
    >>> print(orbit)
    Orbit(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [[inf inf inf]]>,
        p=<CartesianVel3D (x[kpc / Myr], y[kpc / Myr], z[kpc / Myr])
            [[inf inf inf]]>,
        t=UncheckedQuantity(Array([500.], dtype=float64, weak_type=True), unit='Myr'),
        frame=SimulationFrame(),
        potential=KeplerPotential( ... ),
        interpolant=None)

    We can also integrate a batch of orbits at once:

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([[10, 0, 0], [10., 0, 0]], "kpc"),
    ...                            p=u.Quantity([[0, 200, 0], [0, 220, 0]], "km/s"),
    ...                            t=u.Quantity([-100, -150], "Myr"))
    >>> orbit = gd.evaluate_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
        q=CartesianPos3D( ... ), p=CartesianVel3D( ... ),
        t=UncheckedQuantity(Array([ ... ], dtype=float64), unit='Myr'),
        frame=SimulationFrame(),
        potential=KeplerPotential( ... ),
        interpolant=None
    )

    :class:`~galax.dynamics.PhaseSpacePosition` has a ``t`` argument for the
    time at which the position is given. As noted earlier, this can be used to
    integrate from a different time than the initial time of the position:

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([10, 0, 0], "kpc"),
    ...                            p=u.Quantity([0, 200, 0], "km/s"),
    ...                            t=u.Quantity(0, "Myr"))
    >>> ts = u.Quantity(jnp.linspace(0.3, 1, 8), "Gyr")
    >>> orbit = gd.evaluate_orbit(potential, w0, ts)
    >>> orbit.q[0]  # doctest: +SKIP
    Array([ 9.779, -0.3102,  0.        ], dtype=float64)

    Note that IS NOT the same as ``w0``. ``w0`` is integrated from ``t=0`` to
    ``t=300`` and then from ``t=300`` to ``t=1000``.

    .. note::

        If you want to reproduce :mod:`gala`'s behavior, you can use
        :class:`~galax.dynamics.PhaseSpacePosition` with ``t=None``.
        `evaluate_orbit` will then assume ``w0`` is defined at `t`[0].
    """
    # Setup
    units = pot.units
    solver = _default_solver if solver is None else solver
    t = jnp.atleast_1d(FastQ.from_(t, units["time"]))  # ensure t units

    field = HamiltonianField(pot)

    # Parse t0 for the initial integration
    tw0 = w0.t if (isinstance(w0, gc.PhaseSpacePosition) and w0.t is not None) else t[0]

    # Initial integration `w0.t` to `t[0]`.
    # TODO: diffrax's `solver_state`, `controller_state`
    # TODO: `max_steps` as kwarg.
    fullt0 = jnp.full_like(tw0, fill_value=t[0])
    soln0 = solver.solve(field, w0, tw0, fullt0, dense=False, unbatch_time=True)

    # Orbit integration `t[0]` to `t[-1]`
    # TODO: `max_steps` as kwarg.
    ys = (FastQ(soln0.ys[0], units["length"]), FastQ(soln0.ys[1], units["speed"]))
    soln = solver.solve(
        field, ys, t[0], t[-1], saveat=t, dense=dense, vectorize_interpolation=True
    )

    # Return the orbit object
    return Orbit(
        q=FastQ(soln.ys[0], units["length"]),
        p=FastQ(soln.ys[1], units["speed"]),
        t=t,
        frame=getattr(w0, "frame", gc.frames.SimulationFrame()),
        potential=pot,
        interpolant=(
            PhaseSpaceInterpolation(soln.interpolation, units=units) if dense else None
        ),
    )


@dispatch
def evaluate_orbit(
    pot: gp.AbstractPotential,
    w0: gc.PhaseSpacePosition | gt.BtSz6,
    /,
    *,
    t: Any,
    **kwargs: Any,
) -> Orbit:
    """Compute an orbit in a potential, supporting `t` as a keyword argument.

    Examples
    --------
    First some imports:

    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    We can then create the point-mass' potential, with galactic units:

    >>> potential = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    We can then integrate an initial phase-space position in this potential to
    get an orbit:

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([10, 0, 0], "kpc"),
    ...                            p=u.Quantity([0, 200, 0], "km/s"),
    ...                            t=u.Quantity(-100, "Myr"))
    >>> ts = jnp.linspace(0, 1000, 4)  # (1 Gyr, 4 steps)
    >>> orbit = gd.evaluate_orbit(potential, w0, t=ts)
    >>> orbit
    Orbit(
        q=CartesianPos3D( ... ), p=CartesianVel3D( ... ),
        t=UncheckedQuantity(Array([ ... ], dtype=float64), unit='Myr'),
        frame=SimulationFrame(),
        potential=KeplerPotential( ... ),
        interpolant=None
    )

    """
    return evaluate_orbit(pot, w0, t, **kwargs)
