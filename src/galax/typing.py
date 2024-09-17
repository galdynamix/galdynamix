"""Type hints for galax."""

# TODO: Finalize variable names and make everything public.
__all__: list[str] = []

from typing import TypeAlias

import astropy.units as u
from jaxtyping import Array, Float, Integer, Shaped

from unxt import Quantity

# =============================================================================

Shape: TypeAlias = tuple[int, ...]
Dimension: TypeAlias = u.PhysicalType
Unit: TypeAlias = u.Unit | u.UnitBase | u.CompositeUnit
UnitEquivalency: TypeAlias = u.Equivalency

# =============================================================================
# Scalars

# An integer scalar.
IntScalar = Integer[Array, ""]
IntQScalar = Integer[Quantity, ""]
IntLike = IntScalar | int

# A float scalar.
FloatScalar = Float[Array, ""]
FloatQScalar = Float[Quantity, ""]

FloatLike = FloatScalar | float | int
"""A float(/int) or float scalar."""

# A float or integer scalar.
RealScalar = FloatScalar | IntScalar
RealQScalar = FloatQScalar | IntQScalar

# A float or integer or float(/int) scalar.
RealScalarLike = FloatLike | IntLike


# =============================================================================
# Vectors

# -----------------------------------------------------------------------------
# Shaped

# 1-vector
Vec1 = Float[Array, "1"]
QVec1 = Float[Quantity, "1"]

# A 3-vector, e.g. q=(x, y, z) or p=(vx, vy, vz).
Vec3 = Float[Array, "3"]
QVec3 = Float[Quantity, "3"]

Vec4 = Float[Array, "4"]
"""A 4-vector e.g. w=(t, x, y, z)."""

Vec6 = Float[Array, "6"]
"""A 6-vector e.g. w=(x, y, z, vx, vy, vz)."""

Vec7 = Float[Array, "7"]
"""A 7-vector e.g. w=(x, y, z, vx, vy, vz, t)."""

VecN = Float[Array, "N"]
"""An (N,)-vector."""

# -----------------
# Specific

# Time vector
VecTime = Float[Array, "time"]
QVecTime = Float[Quantity, "time"]
VecTime1 = Float[Vec1, "time"]
VecTime3 = Float[Vec3, "time"]
VecTime6 = Float[Vec6, "time"]
VecTime7 = Float[Vec7, "time"]

# -----------------------------------------------------------------------------
# Vector Batches

# -----------------
# Scalars

BatchFloatScalar = Shaped[FloatScalar, "*batch"]

BatchableFloatQScalar = Shaped[FloatQScalar, "*#batch"]

BatchFloatQScalar = Shaped[FloatQScalar, "*batch"]

BatchRealQScalar = Shaped[RealQScalar, "*batch"]
BatchableRealScalar = Shaped[RealScalar, "*#batch"]
BatchableRealQScalar = Shaped[RealQScalar, "*#batch"]

BatchableRealScalarLike = BatchableRealScalar | RealScalarLike

# -----------------
# Batched

# Zero or more batches of 3-vectors.
BatchVec3 = Shaped[Vec3, "*batch"]
BatchQVec3 = Shaped[QVec3, "*batch"]

# Zero or more batches of 6-vectors
BatchVec6 = Shaped[Vec6, "*batch"]

# Zero or more batches of 7-vectors
BatchVec7 = Shaped[Vec7, "*batch"]

# -----------------
# Specific

BatchVecTime = Shaped[VecTime, "*batch"]
BatchVecTime6 = Shaped[VecTime6, "*batch"]
BatchVecTime7 = Shaped[VecTime7, "*batch"]
BatchQVecTime = Shaped[QVecTime, "*batch"]

# -----------------
# Any Shape

FloatAnyShape = Float[Array, "..."]
FloatQAnyShape = Float[Quantity, "..."]


# =============================================================================
# Matrices

QMatrix33 = Float[Quantity, "3 3"]
BatchQMatrix33 = Shaped[QMatrix33, "*batch"]


# =============================================================================

MassScalar = Shaped[Quantity["mass"], ""]
MassBatchScalar = Shaped[Quantity["mass"], "*batch"]
MassBatchableScalar = Shaped[Quantity["mass"], "*#batch"]

TimeScalar = Shaped[Quantity["time"], ""]
TimeBatchScalar = Shaped[Quantity["time"], "*batch"]
TimeBatchableScalar = Shaped[Quantity["time"], "*#batch"]

LengthScalar = Shaped[Quantity["length"], ""]
LengthVec3 = Shaped[Quantity["length"], "3"]
LengthBatchVec3 = Shaped[LengthVec3, "*batch"]
LengthBatchableVec3 = Shaped[LengthVec3, "*#batch"]

SpeedVec3: TypeAlias = Shaped[Quantity["speed"], "3"]
SpeedBatchVec3: TypeAlias = Shaped[SpeedVec3, "*batch"]
SpeedBatchableVec3: TypeAlias = Shaped[SpeedVec3, "*#batch"]

SpecificEnergyScalar = Float[Quantity["specific_energy"], ""]
SpecificEnergyBatchScalar = Float[Quantity["specific_energy"], "*batch"]
