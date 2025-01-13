"""Simulatino Reference Frames.

Building off of `coordinax.frames`.

"""

__all__ = ["SimulationFrame"]

from typing import final

import coordinax.frames as cxf


@final
class SimulationFrame(cxf.AbstractReferenceFrame):  # type: ignore[misc]
    """A null reference frame.

    This is a reference frame that cannot be transformed to or from.

    Examples
    --------
    >>> import coordinax.frames as cxf
    >>> import galax.coordinates as gc

    >>> sim = gc.frames.SimulationFrame()
    >>> icrs = cxf.ICRS()

    >>> try:
    ...     cxf.frame_transform_op(sim, icrs)
    ... except Exception as e:
    ...     print(e)
    `frame_transform_op(SimulationFrame(), ICRS())` could not be resolved...

    """
