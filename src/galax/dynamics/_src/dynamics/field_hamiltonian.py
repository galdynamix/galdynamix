"""Hamiltonian field."""

__all__ = ["HamiltonianField"]

from functools import partial
from typing import Any, TypeAlias, final

import diffrax as dfx
import equinox as eqx
import jax
from plum import convert, dispatch

import coordinax as cx
import coordinax.vecs as cxv
import unxt as u
from unxt.quantity import UncheckedQuantity as FastQ

import galax.coordinates as gc
import galax.dynamics._src.custom_types as gdt
import galax.potential as gp
import galax.typing as gt
from .field_base import AbstractDynamicsField


@final
class HamiltonianField(AbstractDynamicsField, strict=True):  # type: ignore[call-arg]
    r"""Dynamics field for Hamiltonian EoM.

    This is for Hamilton's equations for motion for a particle in a potential.

    .. math::

            \\dot{q} = \frac{dH}{dp} \\ \\dot{p} = -\frac{dH}{dq}

    .. note::

        Calling this object in a jit context will provide a significant speedup.

    .. warning::

        The call method currently returns a `tuple[Array[float, (3,)],
        Array[float, (3,)]]`. In future, when `unxt.Quantity` is registered with
        `quax.quaxify` for `diffrax.diffeqsolve` then this will return
        `tuple[Quantity[float, (3,), 'length'], Quantity[float, (3,),
        'speed']]`. Later, when `coordinax.AbstractVector` is registered with
        `quax.quaxify` for `diffrax.diffeqsolve` then this will return
        `tuple[CartesianPos3D, CartesianVel3D]`.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...    r_s=u.Quantity(5, "kpc"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)
    >>> field
    HamiltonianField( potential=HernquistPotential( ... ) )

    The `.terms()` method returns a PyTree of `diffrax.AbstractTerm` objects
    that can be used to integrate the equations of motion with a `diffrax`
    solver (e.g. `diffrax.diffeqsolve`). The `.terms()` method is solver
    dependent, e.g. returning a `ODETerm` for `diffrax.Dopri8` and
    `tuple[ODETerm, ODETerm]` for `diffrax.SemiImplicitEuler`.

    >>> solver = dfx.Dopri8()
    >>> field.terms(solver)
    ODETerm(vector_field=<wrapped function __call__>)

    >>> solver = dfx.SemiImplicitEuler()
    >>> field.terms(solver)
    (ODETerm( ... ), ODETerm( ... ))

    Just to continue the example, we can use this field to integrate the
    equations of motion:

    >>> solver = gd.DynamicsSolver()  # defaults to Dopri8
    >>> w0 = gc.PhaseSpacePosition(
    ...     q=u.Quantity([[8, 0, 9], [9, 0, 3]], "kpc"),
    ...     p=u.Quantity([0, 220, 0], "km/s"),
    ...     t=u.Quantity(0, "Gyr"))
    >>> t1 = u.Quantity(1, "Gyr")
    >>> soln = solver.solve(field, w0, t1)
    >>> soln
    Solution( t0=f64[], t1=f64[], ts=f64[1],
              ys=(f64[1,2,3], f64[1,2,3]),
              ... )

    >>> w = gc.PhaseSpacePosition.from_(soln, frame=w0.frame, units=pot.units)
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

    #: Potential.
    potential: gp.AbstractPotential

    @property
    def units(self) -> u.AbstractUnitSystem:
        return self.potential.units

    @dispatch.abstract
    def __call__(self, *_: Any) -> tuple[Any, Any]:
        """Evaluate the field at a position and time.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import galax.coordinates as gc
        >>> import galax.potential as gp
        >>> import galax.dynamics as gd

        >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
        >>> field = gd.fields.HamiltonianField(pot)

        HamiltonianField can be called using many different combinations of
        arguments. For example, it can be called with a
        `galax.coordinates.PhaseSpacePosition`:

        >>> w = gc.PhaseSpacePosition(t=u.Quantity(0, "Gyr"),
        ...                           q=u.Quantity([8., 0, 0], "kpc"),
        ...                           p=u.Quantity([0, 220, 0], "km/s"))

        >>> field(w)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        Or with a `coordinax.frames.AbstractCoordinate`:

        >>> from plum import convert
        >>> coord = convert(w, cx.Coordinate)

        >>> field(coord)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        Of with a `coordinax.vecs.Space`, so long as it contains a
        `coordinax.vecs.FourVector` for the position, since a time is
        required:

        >>> space = cx.Space(length=cxv.FourVector.from_([0, 8., 0, 0], "kpc"),
        ...                  speed=cxv.CartesianVel3D.from_([0, 220, 0], "km/s"))

        >>> field(space)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        We can break apart the `coordinax.vecs.Space` into its components:

        >>> tq, p = space["length"], space["speed"]

        >>> field(tq, p)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        If the position is not a `coordinax.vecs.FourVector` then a time must
        be passed as a separate argument:

        >>> t, q = tq.t, tq.q

        >>> field(t, q, p)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        The position and velocity can be passed as `unxt.Quantity` arrays,
        in which case they are assumed to be in Cartesian coordinates.
        The position and velocity may be grouped in a tuple:

        >>> q = u.Quantity([8., 0, 0], "kpc")
        >>> p = u.Quantity([0, 220, 0], "km/s")

        >>> field(t, (q, p))
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        Or passed as separate arguments:

        >>> field(t, q, p)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        Finally, the data can be passed as jax arrays, in which case they are
        considered to be Cartesian coordinates and in the unit system of the
        field (``field.units``):

        >>> t, q, p = t.ustrip(pot.units), q.ustrip(pot.units), p.ustrip(pot.units)
        >>> field(t, q, p)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        >>> field(t, (q, p))
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        >>> qp = jnp.concat([q, p])
        >>> field(t, qp)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64),
         Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        >>> tqp = jnp.concat([t[None], q, p])
        >>> field(tqp)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        """
        raise NotImplementedError  # pragma: no cover

    # ---------------------------
    # Private API to support symplectic integrators. It would be good to figure
    # out a way to make these methods part of `__call__`.

    @eqx.filter_jit  # type: ignore[misc]
    def _dqdt(self, t: Any, p: gdt.BBtParr, args: Any, /) -> gdt.BBtParr:  # noqa: ARG002
        """Call with time, position quantity arrays."""
        return p

    @eqx.filter_jit  # type: ignore[misc]
    def _dpdt(self, t: gt.BBtScalarSz0, q: gdt.BBtQarr, args: Any, /) -> gdt.BtAarr:  # noqa: ARG002
        """Call with time, velocity quantity arrays."""
        units = self.units
        a: gdt.BtAarr = -self.potential._gradient(  # noqa: SLF001
            FastQ(q, units["length"]),
            FastQ(t, units["time"]),
        ).ustrip(units["acceleration"])
        return a


# ===============================================
# Terms dispatches


@AbstractDynamicsField.terms.dispatch  # type: ignore[misc]
def terms(
    self: HamiltonianField,
    _: dfx.SemiImplicitEuler,
    /,
) -> tuple[dfx.AbstractTerm, dfx.AbstractTerm]:
    """Return the AbstractTerm terms for the SemiImplicitEuler solver.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> solver = dfx.SemiImplicitEuler()

    >>> field.terms(solver)
    (ODETerm( ... ), ODETerm( ... ))

    For completeness we'll integrate the EoM.

    >>> dynamics_solver = gd.DynamicsSolver(solver)
    >>> w0 = gc.PhaseSpacePosition(
    ...     q=u.Quantity([8., 0, 0], "kpc"),
    ...     p=u.Quantity([0, 220, 0], "km/s"),
    ...     t=u.Quantity(0, "Gyr"))
    >>> t1 = u.Quantity(200, "Myr")

    >>> soln = dynamics_solver.solve(field, w0, t1, dt0=0.001, max_steps=200_000)
    >>> print(gc.PhaseSpacePosition.from_(soln, units=pot.units, frame=w0.frame))
    PhaseSpacePosition(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [[ 7.645 -0.701  0.   ]]>,
        p=<CartesianVel3D (x[kpc / Myr], y[kpc / Myr], z[kpc / Myr])
            [[0.228 0.215 0.   ]]>,
        t=Quantity['time'](Array([200.], dtype=float64), unit='Myr'),
        frame=SimulationFrame())

    """
    return (dfx.ODETerm(self._dqdt), dfx.ODETerm(self._dpdt))


# ===============================================
# Call dispatches


Args: TypeAlias = tuple[Any, ...] | None


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField, t: gt.RealQuSz0, q: gdt.BBtQ, p: gdt.BBtP, _: Args = None, /
) -> gdt.BtPAarr:
    """Call with time, position, velocity quantity arrays."""
    # TODO: not require unit munging
    units = self.units
    a = -self.potential._gradient(q, t).ustrip(units["acceleration"])  # noqa: SLF001
    return p.ustrip(units["speed"]), a


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField, t: gt.RealQuSz0, qp: gdt.BBtQP, args: Args = None, /
) -> gdt.BtPAarr:
    """Call with time, (position, velocity) quantity arrays."""
    return self(t, qp[0], qp[1], args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField,
    t: gt.BBtScalarSz0,
    q: gdt.BBtQarr,
    p: gdt.BBtParr,
    args: Args = None,
    /,
) -> gdt.BtPAarr:
    """Call with time, position, velocity arrays.

    The arrays are considered to be in the unit system of the field
    (`field.units`) Which can be checked with ``unitsystem["X"]`` for X in time,
    position, and speed.

    """
    units = self.units
    return self(
        FastQ(t, units["time"]),
        FastQ(q, units["length"]),
        FastQ(p, units["speed"]),
        args,
    )


# Note: this is the one called by DynamicsSolver, so it's special-cased
@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField, t: gt.BBtScalarSz0, qp: gdt.BBtQParr, _: Args = None, /
) -> gdt.BtPAarr:
    """Call with time, (position, velocity) arrays.

    The arrays are considered to be in the unit system of the field
    (`field.units`) Which can be checked with ``unitsystem["X"]`` for X in time,
    position, and speed.

    """
    # TODO: not require unit munging
    units = self.units
    a = -self.potential._gradient(  # noqa: SLF001
        FastQ(qp[0], units["length"]),
        FastQ(t, units["time"]),
    ).ustrip(units["acceleration"])
    return qp[1], a


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField, t: gt.BBtScalarSz0, qp: gt.BBtSz6, args: Args = None, /
) -> gdt.BtPAarr:
    """Call with time, pos-vel 6 array.

    The arrays are considered to be in the unit system of the field
    (`field.units`) Which can be checked with ``unitsystem["X"]`` for X in time,
    position, and speed.

    """
    return self(t, (qp[..., 0:3], qp[..., 3:6]), args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(self: HamiltonianField, qp: gt.BBtSz7, args: Args = None, /) -> gdt.BtPAarr:
    """Call with time, pos-vel 7 array.

    The arrays are considered to be in the unit system of the field
    (`field.units`) Which can be checked with ``unitsystem["X"]`` for X in time,
    position, and speed.

    """
    return self(qp[..., 0], (qp[..., 1:4], qp[..., 4:7]), args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField,
    t: gt.TimeBBtSz0,
    q: cxv.AbstractPos3D,
    p: cxv.AbstractVel3D,
    args: Args = None,
    /,
) -> gdt.BtPAarr:
    """Call with time and `coordinax` ``AbstractPos3D`` and ``AbstractVel3D``."""
    return self(t, (convert(q, FastQ), convert(p, FastQ)), args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField,
    tq: cxv.FourVector,
    p: cxv.AbstractVel3D,
    args: Args = None,
    /,
) -> gdt.BtPAarr:
    """Call with `coordinax.vecs.FourVector`, `coordinax.vecs.AbstractVel3D`."""
    return self(tq.t, tq.q, p, args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(self: HamiltonianField, space: cxv.Space, args: Args = None, /) -> gdt.BtPAarr:
    """Call with `coordinax.vecs.Space`."""
    # TODO: better packing re 4Vec for "length"
    assert isinstance(space["length"], cxv.FourVector)  # noqa: S101
    return self(space["length"], space["speed"], args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField, w: cx.frames.AbstractCoordinate, args: Args = None, /
) -> gdt.BtPAarr:
    """Call with `coordinax.AbstractCoordinate`."""
    w = w.to_frame(gc.frames.SimulationFrame())  # TODO: enable other frames
    return self(w.data, args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField, w: gc.PhaseSpacePosition, args: Args = None, /
) -> gdt.BtPAarr:
    """Call with `galax.coordinates.PhaseSpacePosition`."""
    assert w.t is not None  # noqa: S101
    w = w.to_frame(gc.frames.SimulationFrame())  # TODO: enable other frames
    return self(w.t, w._qp(units=self.units), args)  # noqa: SLF001
