"""ABC for phase-space positions."""

__all__ = ["AbstractPhaseSpaceCoordinate", "ComponentShapeTuple"]

from abc import abstractmethod
from functools import partial
from typing import Any, NamedTuple
from typing_extensions import override

import equinox as eqx
import jax
from plum import convert

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import BareQuantity as FastQ

import galax.typing as gt
from galax.coordinates._src.base import AbstractPhaseSpaceObject
from galax.coordinates._src.frames import SimulationFrame


class ComponentShapeTuple(NamedTuple):
    """Component shape of the phase-space position."""

    q: int
    """Shape of the position component."""

    p: int
    """Shape of the momentum component."""

    t: int
    """Shape of the time component."""


# =============================================================================


# TODO: make it strict=True
class AbstractPhaseSpaceCoordinate(AbstractPhaseSpaceObject):
    r"""ABC underlying phase-space positions and their composites.

    The phase-space position is a point in the 3+3+1-dimensional phase space
    :math:`\mathbb{R}^7` of a dynamical system. It is composed of the position
    :math:`\boldsymbol{q}\in\mathbb{R}^3`, the conjugate momentum
    :math:`\boldsymbol{p}\in\mathbb{R}^3`, and the time
    :math:`t\in\mathbb{R}^1`.

    Examples
    --------
    With the following imports:

    >>> import unxt as u
    >>> import galax.coordinates as gc

    We can create a phase-space position from a mapping:

    >>> obj = {"q": u.Quantity([1, 2, 3], "kpc"),
    ...        "p": u.Quantity([4, 5, 6], "km/s"),
    ...        "t": u.Quantity(0, "Gyr")}
    >>> gc.PhaseSpaceCoordinate.from_(obj)
    PhaseSpaceCoordinate(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity['time'](Array(0, dtype=int64, ...), unit='Gyr'),
        frame=SimulationFrame()
    )


    """

    q: eqx.AbstractVar[cx.vecs.AbstractPos3D]
    """Positions."""

    p: eqx.AbstractVar[cx.vecs.AbstractVel3D]
    """Conjugate momenta at positions ``q``."""

    t: eqx.AbstractVar[gt.BBtFloatQuSz0]
    """Time corresponding to the positions and momenta."""

    frame: eqx.AbstractVar[SimulationFrame]  # TODO: support frames
    """The reference frame of the phase-space coordinate."""

    # ==========================================================================
    # Coordinate API

    @classmethod
    def _dimensionality(cls) -> int:
        """Return the dimensionality of the phase-space position.

        Examples
        --------
        >>> import galax.coordinates as gc
        >>> gc.PhaseSpaceCoordinate._dimensionality()
        7

        """
        return 7  # TODO: should it be 7? Also make it a Final

    @override
    @property
    def data(self) -> cx.Space:  # type: ignore[misc]
        """Return the data as a space.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.coordinates as gc

        We can create a phase-space position:

        >>> pos = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
        ...                             p=u.Quantity([4, 5, 6], "km/s"),
        ...                             t=u.Quantity(0, "Gyr"))
        >>> pos.data
        Space({ 'length': FourVector( ... ), 'speed': CartesianVel3D( ... ) })

        """
        return cx.Space(length=cx.vecs.FourVector(t=self.t, q=self.q), speed=self.p)

    # ==========================================================================
    # Array API

    @property
    @abstractmethod
    def _shape_tuple(self) -> tuple[gt.Shape, ComponentShapeTuple]:
        """Batch, component shape."""
        raise NotImplementedError

    # ==========================================================================
    # Convenience methods

    def wt(self, *, units: Any) -> gt.BtSz7:
        """Phase-space position as an Array[float, (*batch, 1+Q+P)].

        This is the full phase-space position, including the time.

        Parameters
        ----------
        units : `unxt.AbstractUnitSystem`, keyword-only
            The unit system. :func:`~unxt.unitsystem` is used to
            convert the input to a unit system.

        Returns
        -------
        Array[float, (*batch, (T=1)+Q+P)]
            The full phase-space position, including time on the first axis.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.coordinates as gc

        We can create a phase-space position and convert it to a 7-vector:

        >>> psp = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
        ...                               p=u.Quantity([4, 5, 6], "km/s"),
        ...                               t=u.Quantity(7.0, "Myr"))
        >>> psp.wt(units="galactic")
            Array([7.00000000e+00, 1.00000000e+00, 2.00000000e+00, 3.00000000e+00,
                4.09084866e-03, 5.11356083e-03, 6.13627299e-03], dtype=float64, ...)

        """
        usys = u.unitsystem(units)
        batch, comps = self._shape_tuple
        cart = self.vconvert(cx.CartesianPos3D).uconvert(usys)
        q = jnp.broadcast_to(convert(cart.q, FastQ), (*batch, comps.q))
        p = jnp.broadcast_to(convert(cart.p, FastQ), (*batch, comps.p))
        t = jnp.broadcast_to(self.t.ustrip(usys["time"])[..., None], (*batch, comps.t))
        return jnp.concat((t, q.value, p.value), axis=-1)

    # ==========================================================================
    # Dynamical quantities

    def potential_energy(
        self, potential: "AbstractPotential"
    ) -> u.Quantity["specific energy"]:
        r"""Return the specific potential energy.

        .. math::

            E_\Phi = \Phi(\boldsymbol{q})

        Parameters
        ----------
        potential : `galax.potential.AbstractPotential`
            The potential object to compute the energy from.

        Returns
        -------
        E : Array[float, (*batch,)]
            The specific potential energy.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import galax.coordinates as gc
        >>> import galax.potential as gp

        We can construct a phase-space position:

        >>> q = cx.CartesianPos3D(
        ...     x=u.Quantity(1, "kpc"),
        ...     y=u.Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "kpc"),
        ...     z=u.Quantity(2, "kpc"))
        >>> p = cx.CartesianVel3D(
        ...     x=u.Quantity(0, "km/s"),
        ...     y=u.Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "km/s"),
        ...     z=u.Quantity(0, "km/s"))
        >>> w = gc.PhaseSpaceCoordinate(q, p, t=u.Quantity(0, "Myr"))

        We can compute the kinetic energy:

        >>> pot = gp.MilkyWayPotential()
        >>> w.potential_energy(pot)
        Quantity[...](Array(..., dtype=float64), unit='kpc2 / Myr2')
        """
        return potential.potential(self.q, t=self.t)

    @partial(jax.jit, inline=True)
    def total_energy(self, potential: "AbstractPotential") -> gt.BtFloatQuSz0:
        r"""Return the specific total energy.

        .. math::

            E_K = \frac{1}{2} \\, |\boldsymbol{v}|^2
            E_\Phi = \Phi(\boldsymbol{q})
            E = E_K + E_\Phi

        Parameters
        ----------
        potential : `galax.potential.AbstractPotential`
            The potential object to compute the energy from.

        Returns
        -------
        E : Array[float, (*batch,)]
            The kinetic energy.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import galax.coordinates as gc
        >>> import galax.potential as gp

        We can construct a phase-space position:

        >>> q = cx.CartesianPos3D(
        ...     x=u.Quantity(1, "kpc"),
        ...     y=u.Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "kpc"),
        ...     z=u.Quantity(2, "kpc"))
        >>> p = cx.CartesianVel3D(
        ...     x=u.Quantity(0, "km/s"),
        ...     y=u.Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "km/s"),
        ...     z=u.Quantity(0, "km/s"))
        >>> w = gc.PhaseSpaceCoordinate(q, p, t=u.Quantity(0, "Myr"))

        We can compute the kinetic energy:

        >>> pot = gp.MilkyWayPotential()
        >>> w.total_energy(pot)
        Quantity[...](Array(..., dtype=float64), unit='km2 / s2')
        """
        return self.kinetic_energy() + self.potential_energy(potential)

    @partial(jax.jit, inline=True)
    def angular_momentum(self) -> cx.vecs.CartesianGeneric3D:
        r"""Compute the angular momentum.

        .. math::

            \boldsymbol{{L}} = \boldsymbol{{q}} \times \boldsymbol{{p}}

        See :ref:`shape-conventions` for more information about the shapes of
        input and output objects.

        Returns
        -------
        L : Array[float, (*batch,3)]
            Array of angular momentum vectors in Cartesian coordinates.

        Examples
        --------
        We assume the following imports

        >>> import unxt as u
        >>> import galax.coordinates as gc

        We can compute the angular momentum of a single object

        >>> q = u.Quantity([1., 0, 0], "au")
        >>> p = u.Quantity([0, 2., 0], "au/yr")
        >>> t = u.Quantity(0, "yr")
        >>> w = gc.PhaseSpaceCoordinate(q=q, p=p, t=t)
        >>> h = w.angular_momentum()
        >>> print(h)
        <CartesianGeneric3D (x[AU2 / yr], y[AU2 / yr], z[AU2 / yr])
            [0. 0. 2.]>

        """
        from galax.dynamics import specific_angular_momentum

        return specific_angular_momentum(self)
