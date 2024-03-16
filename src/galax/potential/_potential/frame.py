"""Wrapper to add frame operations to a potential."""

__all__ = ["PotentialFrame"]


from dataclasses import replace
from typing import cast, final

import equinox as eqx

from coordinax.operators import OperatorSequence, simplify_op
from unxt import Quantity, UnitSystem

import galax.typing as gt
from galax.potential._potential.base import AbstractPotentialBase
from galax.utils import ImmutableDict


@final
class PotentialFrame(AbstractPotentialBase):
    """Reference frame of the potential.

    Examples
    --------
    In this example, we create a triaxial Hernquist potential and apply a few
    coordinate transformations.

    First some imports:

    >>> from unxt import Quantity
    >>> import coordinax.operators as cxo
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp

    Now we define a triaxial Hernquist potential with a time-dependent mass:

    >>> mfunc = gp.UserParameter(lambda t: 1e12 * (1 + t.to_value("Gyr") / 10), unit="Msun")
    >>> pot = gp.TriaxialHernquistPotential(m=mfunc, c=Quantity(1, "kpc"),
    ...                                     q1=1, q2=0.5, units="galactic")

    Let's see the triaxiality of the potential:

    >>> t = Quantity(0, "Gyr")
    >>> w1 = gc.PhaseSpacePosition(q=Quantity([1, 0, 0], "kpc"),
    ...                            p=Quantity([0, 1, 0], "km/s"),
    ...                            t=t)

    The triaxiality can be seen in the potential energy of the three positions:

    >>> pot.potential_energy(w1).decompose(pot.units)
    Quantity['specific energy'](Array(-2.24925108, dtype=float64), unit='kpc2 / Myr2')

    >>> q = Quantity([0, 1, 0], "kpc")
    >>> pot.potential_energy(q, t).decompose(pot.units)
    Quantity['specific energy'](Array(-2.24925108, dtype=float64), unit='kpc2 / Myr2')

    >>> q = Quantity([0, 0, 1], "kpc")
    >>> pot.potential_energy(q, t).decompose(pot.units)
    Quantity['specific energy'](Array(-1.49950072, dtype=float64), unit='kpc2 / Myr2')

    Let's apply a spatial translation to the potential:

    >>> op1 = cxo.GalileanSpatialTranslationOperator(Quantity([3, 0, 0], "kpc"))
    >>> op1
    GalileanSpatialTranslationOperator( translation=Cartesian3DVector( ... ) )

    >>> framedpot1 = gp.PotentialFrame(potential=pot, operator=op1)
    >>> framedpot1
    PotentialFrame(
      potential=TriaxialHernquistPotential( ... ),
      operator=OperatorSequence(
        operators=( GalileanSpatialTranslationOperator( ... ), )
      )
    )

    Now the potential energy is different because the potential has been
    translated by 3 kpc in the x-direction:

    >>> framedpot1.potential_energy(w1).decompose(pot.units)
    Quantity['specific energy'](Array(-1.49950072, dtype=float64), unit='kpc2 / Myr2')

    This is the same as evaluating the untranslated potential at [-2, 0, 0] kpc:

    >>> q = Quantity([-2, 0, 0], "kpc")
    >>> pot.potential_energy(q, Quantity(0, "Gyr")).decompose(pot.units)
    Quantity['specific energy'](Array(-1.49950072, dtype=float64), unit='kpc2 / Myr2')

    We can also apply a time translation to the potential:

    >>> op2 = cxo.GalileanTranslationOperator(Quantity([1_000, 0, 0, 0], "kpc"))
    >>> op2.translation.t.to("Myr")
    Quantity['time'](Array(3.26156378, dtype=float64), unit='Myr')

    >>> framedpot2 = gp.PotentialFrame(potential=pot, operator=op2)

    We can see that the potential energy is the same as before, since we have
    been evaluating the potential at ``w1.t=t=0``:

    >>> framedpot2.potential_energy(w1).decompose(pot.units)
    Quantity['specific energy'](Array(-2.24851747, dtype=float64), unit='kpc2 / Myr2')

    But if we evaluate the potential at a different time, the potential energy
    will be different:

    >>> from dataclasses import replace
    >>> w2 = replace(w1, t=Quantity(10, "Myr"))
    >>> framedpot2.potential_energy(w2).decompose(pot.units)
    Quantity['specific energy'](Array(-2.25076672, dtype=float64), unit='kpc2 / Myr2')

    Now let's boost the potential by 200 km/s in the y-direction:

    >>> op3 = cxo.GalileanBoostOperator(Quantity([0, 200, 0], "km/s"))
    >>> op3
    GalileanBoostOperator( velocity=CartesianDifferential3D( ... ) )

    >>> framedpot3 = gp.PotentialFrame(potential=pot, operator=op3)
    >>> framedpot3.potential_energy(w2).decompose(pot.units)
    Quantity['specific energy'](Array(-1.37421204, dtype=float64), unit='kpc2 / Myr2')

    Alternatively we can rotate the potential by 90 degrees about the y-axis:

    >>> import quaxed.array_api as xp
    >>> theta = Quantity(90, "deg")
    >>> Ry = xp.asarray([[xp.cos(theta),  0, xp.sin(theta)],
    ...                  [0,              1, 0            ],
    ...                  [-xp.sin(theta), 0, xp.cos(theta)]])
    >>> op4 = cxo.GalileanRotationOperator(Ry)
    >>> op4
    GalileanRotationOperator(rotation=f64[3,3])

    >>> framedpot4 = gp.PotentialFrame(potential=pot, operator=op4)
    >>> framedpot4.potential_energy(w1).decompose(pot.units)
    Quantity['specific energy'](Array(-1.49950072, dtype=float64), unit='kpc2 / Myr2')

    >>> q = Quantity([0, 0, 1], "kpc")
    >>> framedpot4.potential_energy(q, t).decompose(pot.units)
    Quantity['specific energy'](Array(-2.24925108, dtype=float64), unit='kpc2 / Myr2')

    If you look all the way back to the first examples, you will see that the
    potential energy at [1, 0, 0] and [0, 0, 1] have swapped, as expected for a
    90 degree rotation about the y-axis!

    Lastly, we can apply a sequence of transformations to the potential. There
    are two ways to do this. The first is to create a pre-defined composite
    operator, like a :class:`~galax.coordinates.operators.GalileanOperator`:

    >>> op5 = cxo.GalileanOperator(rotation=op4, translation=op2, velocity=op3)
    >>> op5
    GalileanOperator(
      rotation=GalileanRotationOperator(rotation=f64[3,3]),
      translation=GalileanTranslationOperator( ... ),
      velocity=GalileanBoostOperator( ... )
    )

    >>> framedpot5 = gp.PotentialFrame(potential=pot, operator=op5)
    >>> framedpot5.potential_energy(w2).decompose(pot.units)
    Quantity['specific energy'](Array(-1.16598068, dtype=float64), unit='kpc2 / Myr2')

    The second way is to create a custom sequence of operators. In this case we
    will make a sequence that mimics the previous example:

    >>> op6 = op4 | op2 | op3
    >>> framedpot6 = gp.PotentialFrame(potential=pot, operator=op6)
    >>> framedpot6.potential_energy(w2).decompose(pot.units)
    Quantity['specific energy'](Array(-1.16598068, dtype=float64), unit='kpc2 / Myr2')

    We've seen that the potential can be time-dependent, but so far the
    operators have been Galilean. Let's fix the time-dependent mass of the
    potential but make the frame operator time-dependent in a more interesting
    way. We will also exaggerate the triaxiality of the potential to make the
    effect of the rotation more obvious:

    >>> pot2 = gp.TriaxialHernquistPotential(m=Quantity(1e12, "Msun"),
    ...     c=Quantity(1, "kpc"), q1=0.1, q2=0.1, units="galactic")

    >>> op7 = gc.operators.ConstantRotationZOperator(Omega_z=Quantity(90, "deg/Gyr"))
    >>> framedpot7 = gp.PotentialFrame(potential=pot2, operator=op7)

    The potential energy at a given position will change with time:

    >>> framedpot7.potential_energy(w1).decompose(pot.units).value  # t=0 Gyr
    Array(-2.24925108, dtype=float64)
    >>> framedpot7.potential_energy(w2).decompose(pot.units).value  # t=1 Gyr
    Array(-2.23568166, dtype=float64)
    """  # noqa: E501

    potential: AbstractPotentialBase

    operator: OperatorSequence = eqx.field(default=(), converter=OperatorSequence)
    """Transformation to reference frame of the potential.

    The default is no transformation, ie the coordinates are specified in the
    'simulation' frame.
    """

    @property
    def units(self) -> UnitSystem:
        """The unit system of the potential."""
        return cast(UnitSystem, self.potential.units)

    @property
    def constants(self) -> ImmutableDict[Quantity]:
        """The constants of the potential."""
        return cast("ImmutableDict[Quantity]", self.potential.constants)

    def _potential_energy(
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.BatchFloatQScalar:
        """Compute the potential energy at the given position(s).

        This method applies the frame operators to the coordinates and then
        evaluates the potential energy at the transformed coordinates.

        Parameters
        ----------
        q : Array[float, (3,)]
            The position(s) at which to compute the potential energy.
        t : float
            The time at which to compute the potential energy.

        Returns
        -------
        Array[float, (...)]
            The potential energy at the given position(s).
        """
        # Make inverse operator  # TODO: pre-compute and cache
        inv = self.operator.inverse
        # Transform the position, time.
        qp, tp = inv(q, t)
        # Evaluate the potential energy at the transformed position, time.
        return self.potential._potential_energy(qp, tp)  # noqa: SLF001

    # ruff: noqa: ERA001
    # def _gradient(
    #     self, q: BatchVec3, /, t: BatchableRealScalarLike | RealScalar
    # ) -> BatchVec3:  # TODO: inputs w/ units
    #     """See ``gradient``."""
    #     # Transform the position, time.
    #     qp, tp = self.operator.inverse(
    #         Quantity(q, self.units["length"]), Quantity(t, self.units["time"])
    #     )
    #     # Evaluate the gradient at the transformed position, time.
    #     gradp = self.potential._gradient(qp.value, tp.value)
    #     grad, _ = self.operator(Quantity(gradp, self.units["acceleration"]), tp)
    #     return grad.value


#####################################################################


@simplify_op.register  # type: ignore[misc]
def _simplify_op(frame: PotentialFrame, /) -> PotentialFrame:
    """Simplify the operators in an PotentialFrame."""
    return replace(frame, operator=simplify_op(frame.operator))
