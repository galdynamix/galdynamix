"""Compatibility.

TODO: make all the ``gala`` compat be in a linked package.

"""

__all__: list[str] = []

import warnings
from typing import Any, cast

import gala.dynamics as gd
from astropy.coordinates import BaseDifferential, BaseRepresentation
from plum import conversion_method, convert

import galax.coordinates as gcx

# --------------------------------------------------
# Gala to Galax


@conversion_method(type_from=gd.PhaseSpacePosition, type_to=gcx.PhaseSpacePosition)  # type: ignore[misc]
def gala_psp_to_galax_psp(
    obj: gd.PhaseSpacePosition, /, t: Any | None = None
) -> gcx.PhaseSpacePosition:
    """Gala to galax ``PhaseSpacePosition``.

    Examples
    --------
    With the following imports:

    >>> import gala.dynamics as gd
    >>> import galax.coordinates as gcx
    >>> import astropy.units as u
    >>> from plum import convert

    We can create a :class:`gala.dynamics.PhaseSpacePosition` and convert it to
    a :class:`galax.coordinates.PhaseSpacePosition`

    >>> gala_w = gd.PhaseSpacePosition(pos=[1, 2, 3] * u.kpc,
    ...                                vel=[4, 5, 6] * u.km / u.s)
    >>> gala_w
    <PhaseSpacePosition cartesian, dim=3, shape=()>

    >>> galax_w = convert(gala_w, gcx.PhaseSpacePosition)
    >>> galax_w
    PhaseSpacePosition(
        q=CartesianPosition3D( ... ),
        p=CartesianVelocity3D( ... ),
        t=None
    )
    """
    return gcx.PhaseSpacePosition(q=obj.pos, p=obj.vel, t=t)


@gcx.PhaseSpacePosition.constructor._f.register  # type: ignore[misc]  # noqa: SLF001
def constructor(
    _: type[gcx.PhaseSpacePosition], obj: gd.PhaseSpacePosition, /, t: Any | None = None
) -> gcx.PhaseSpacePosition:
    """Construct a :mod:`galax` PhaseSpacePosition from a :mod:`gala` one.

    Examples
    --------
    With the following imports:

    >>> import gala.dynamics as gd
    >>> import galax.coordinates as gcx
    >>> import astropy.units as u

    We can create a :class:`gala.dynamics.PhaseSpacePosition` and construct a
    :class:`galax.coordinates.PhaseSpacePosition` from it.

    >>> gala_w = gd.PhaseSpacePosition(pos=[1, 2, 3] * u.kpc,
    ...                                vel=[4, 5, 6] * u.km / u.s)
    >>> galax_w = gcx.PhaseSpacePosition.constructor(gala_w)
    >>> galax_w
    PhaseSpacePosition(
        q=CartesianPosition3D( ... ),
        p=CartesianVelocity3D( ... ),
        t=None
    )

    We can also pass a time:

    >>> galax_w = gcx.PhaseSpacePosition.constructor(gala_w, t=2 * u.Myr)

    """
    return cast(gcx.PhaseSpacePosition, gala_psp_to_galax_psp(obj, t=t))


# --------------------------------------------------
# Galax to Gala


@conversion_method(type_from=gcx.PhaseSpacePosition, type_to=gd.PhaseSpacePosition)  # type: ignore[misc]
def galax_psp_to_gala_psp(obj: gcx.PhaseSpacePosition, /) -> gd.PhaseSpacePosition:
    """Galax to gala ``PhaseSpacePosition``.

    .. warning::

        The time is not preserved in the conversion!

    Examples
    --------
    With the following imports:

    >>> from warnings import catch_warnings, filterwarnings
    >>> import gala.dynamics as gd
    >>> import galax.coordinates as gcx
    >>> import astropy.units as u
    >>> from plum import convert

    We can create a :class:`galax.coordinates.PhaseSpacePosition` and convert it
    to a :class:`gala.dynamics.PhaseSpacePosition`.

    >>> galax_w = gcx.PhaseSpacePosition(
    ...     q=[1, 2, 3] * u.kpc, p=[4, 5, 6] * u.km / u.s, t=2 * u.Myr
    ... )

    >>> with catch_warnings():
    ...     filterwarnings("ignore")
    ...     gala_w = convert(galax_w, gd.PhaseSpacePosition)
    >>> gala_w
    <PhaseSpacePosition cartesian, dim=3, shape=()>

    Note that the time is not preserved in the conversion!

    """
    warnings.warn("The time is not preserved in the conversion!", stacklevel=2)

    return gd.PhaseSpacePosition(
        pos=convert(obj.q, BaseRepresentation), vel=convert(obj.p, BaseDifferential)
    )
