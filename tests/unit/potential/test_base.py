import copy
from functools import partial
from typing import Any

import array_api_jax_compat as xp
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax_quantity import Quantity
from quax import quaxify

import galax.dynamics as gd
from galax.potential import AbstractPotentialBase
from galax.typing import (
    BatchableFloatOrIntScalarLike,
    BatchFloatScalar,
    BatchVec3,
    FloatOrIntScalar,
    FloatScalar,
    Vec3,
    Vec6,
)
from galax.units import UnitSystem, galactic
from galax.utils._jax import vectorize_method

from .io.test_gala import GalaIOMixin

array_equal = quaxify(jnp.array_equal)


class TestAbstractPotentialBase(GalaIOMixin):
    """Test the `galax.potential.AbstractPotentialBase` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[AbstractPotentialBase]:
        class TestPotential(AbstractPotentialBase):
            units: UnitSystem = eqx.field(default=galactic, static=True)
            _G: float = eqx.field(init=False, static=True, repr=False, converter=float)

            def __post_init__(self):
                object.__setattr__(self, "_G", 1.0)

            @partial(jax.jit)
            @vectorize_method(signature="(3),()->()")
            def _potential_energy(
                self, q: BatchVec3, t: BatchableFloatOrIntScalarLike
            ) -> BatchFloatScalar:
                return xp.sum(q, axis=-1)

        return TestPotential

    @pytest.fixture(scope="class")
    def fields_(self) -> dict[str, Any]:
        return {}

    @pytest.fixture()
    def fields(self, fields_) -> dict[str, Any]:
        return copy.copy(fields_)

    @pytest.fixture(scope="class")
    def pot(
        self, pot_cls: type[AbstractPotentialBase], fields_: dict[str, Any]
    ) -> AbstractPotentialBase:
        """Create a concrete potential instance for testing."""
        return pot_cls(**fields_)

    # ---------------------------------

    @pytest.fixture(scope="class")
    def x(self) -> Vec3:
        """Create a position vector for testing."""
        return xp.asarray([1, 2, 3], dtype=float)

    @pytest.fixture(scope="class")
    def v(self) -> Vec3:
        """Create a velocity vector for testing."""
        return xp.asarray([4, 5, 6], dtype=float)

    @pytest.fixture(scope="class")
    def xv(self, x: Vec3, v: Vec3) -> Vec6:
        """Create a phase-space vector for testing."""
        return xp.concat([x, v])

    @pytest.fixture(scope="class")
    def t(self) -> float:
        """Create a time for testing."""
        return 0.0

    ###########################################################################

    def test_init(self) -> None:
        """Test the initialization of `AbstractPotentialBase`."""
        # Test that the abstract class cannot be instantiated
        with pytest.raises(TypeError):
            AbstractPotentialBase()

        # Test that the concrete class can be instantiated
        class TestPotential(AbstractPotentialBase):
            units: UnitSystem = eqx.field(default=galactic, static=True)

            def _potential_energy(self, q: Vec3, /, t: FloatOrIntScalar) -> FloatScalar:
                return xp.sum(q, axis=-1)

        pot = TestPotential()
        assert isinstance(pot, AbstractPotentialBase)

    # =========================================================================

    # ---------------------------------

    def test_potential_energy(self, pot: AbstractPotentialBase, x: Vec3) -> None:
        """Test the `AbstractPotentialBase.potential_energy` method."""
        assert pot.potential_energy(x, t=0) == 6

    # ---------------------------------

    def test_call(self, pot: AbstractPotentialBase, x: Vec3) -> None:
        """Test the `AbstractPotentialBase.__call__` method."""
        assert xp.equal(pot(x, t=0), pot.potential_energy(x, t=0))

    def test_gradient(self, pot: AbstractPotentialBase, x: Vec3) -> None:
        """Test the `AbstractPotentialBase.gradient` method."""
        assert array_equal(pot.gradient(x, t=0), xp.ones_like(x))

    def test_density(self, pot: AbstractPotentialBase, x: Vec3) -> None:
        """Test the `AbstractPotentialBase.density` method."""
        assert pot.density(x, t=0) == 0.0

    def test_hessian(self, pot: AbstractPotentialBase, x: Vec3) -> None:
        """Test the `AbstractPotentialBase.hessian` method."""
        assert array_equal(
            pot.hessian(x, t=0),
            xp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )

    def test_acceleration(self, pot: AbstractPotentialBase, x: Vec3) -> None:
        """Test the `AbstractPotentialBase.acceleration` method."""
        assert array_equal(pot.acceleration(x, t=0), -pot.gradient(x, t=0))

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: Vec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        assert array_equal(pot.tidal_tensor(x, t=0), expect)

    # =========================================================================

    def test_integrate_orbit(self, pot: AbstractPotentialBase, xv: Vec6) -> None:
        """Test the `AbstractPotentialBase.integrate_orbit` method."""
        ts = Quantity(xp.linspace(0.0, 1.0, 100), "Myr")

        orbit = pot.integrate_orbit(xv, ts)
        assert isinstance(orbit, gd.Orbit)
        assert orbit.shape == (len(ts),)
        assert array_equal(orbit.t, ts.value)

    def test_integrate_orbit_batch(self, pot: AbstractPotentialBase, xv: Vec6) -> None:
        """Test the `AbstractPotentialBase.integrate_orbit` method."""
        ts = xp.linspace(0.0, 1.0, 100)

        # Simple batch
        orbits = pot.integrate_orbit(xv[None, :], ts)
        assert isinstance(orbits, gd.Orbit)
        assert orbits.shape == (1, len(ts))
        assert array_equal(orbits.t, ts)

        # More complicated batch
        xv2 = xp.stack([xv, xv], axis=0)
        orbits = pot.integrate_orbit(xv2, ts)
        assert isinstance(orbits, gd.Orbit)
        assert orbits.shape == (2, len(ts))
        assert array_equal(orbits.t, ts)
