"""galax: Galactic Dynamix in Jax."""

__all__ = ["PhaseSpaceInterpolation"]

from typing import Any, cast
from typing_extensions import override

import diffrax as dfx
import equinox as eqx

import diffraxtra as dfxtra
import unxt as u

import galax.coordinates as gc
import galax.typing as gt


class PhaseSpaceInterpolation(dfxtra.AbstractVectorizedDenseInterpolation):  # type: ignore[misc]
    """Phase-space interpolation for orbit evaluation."""

    #: The vectorized interpolation object.
    interp: dfxtra.VectorizedDenseInterpolation = eqx.field(
        converter=dfxtra.VectorizedDenseInterpolation.from_
    )

    #: The unit system for the interpolation.
    units: u.AbstractUnitSystem = eqx.field(static=True, converter=u.unitsystem)

    @override
    @eqx.filter_jit  # type: ignore[misc]
    def evaluate(self, ts: u.Quantity) -> gc.PhaseSpacePosition:
        # Parse the time
        t = u.Quantity.from_(ts, self.units["time"])

        # Evaluate the interpolation
        ys = self.interp.evaluate(t.ustrip(self.units["time"]))

        # Return as a phase-space position
        return gc.PhaseSpacePosition(
            q=u.Quantity(ys[0], self.units["length"]),
            p=u.Quantity(ys[1], self.units["speed"]),
            t=t,
        )

    def __call__(self, *args: Any, **kwds: Any) -> gc.PhaseSpacePosition:
        return cast(gc.PhaseSpacePosition, self.evaluate(*args, **kwds))

    @property
    def scalar_interpolation(self) -> dfx.DenseInterpolation:
        """Return the scalar interpolation for the phase-space position."""
        return cast(dfx.DenseInterpolation, self.interp.scalar_interpolation)

    @property
    def batch_shape(self) -> gt.Shape:
        """Return the batch shape of the interpolation."""
        return cast(gt.Shape, self.interp.batch_shape)

    @property
    def y0_shape(self) -> gt.Shape:
        """Return the shape of the initial value."""
        return cast(gt.Shape, self.interp.y0_shape)
