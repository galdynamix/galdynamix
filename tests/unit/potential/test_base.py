import copy
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Any, ClassVar, final

import equinox as eqx
import jax
import pytest
from plum import convert

import quaxed.numpy as jnp
import unxt as u
from unxt.unitsystems import galactic
from xmmutablemap import ImmutableMap

import galax.dynamics as gd
import galax.potential as gp
import galax.potential.params as gpp
import galax.typing as gt
from .io.test_gala import GalaIOMixin
from galax.potential._src.base import default_constants


class AbstractPotential_Test(GalaIOMixin, metaclass=ABCMeta):
    """Test the `galax.potential.AbstractPotential` class."""

    @pytest.fixture(scope="class")
    @abstractmethod
    def pot_cls(self) -> type[gp.AbstractPotential]: ...

    @pytest.fixture(scope="class")
    def units(self) -> u.AbstractUnitSystem:
        return galactic

    @pytest.fixture(scope="class")
    def field_units(self, units: u.AbstractUnitSystem) -> u.AbstractUnitSystem:
        return units

    @pytest.fixture(scope="class")
    def fields_(self, field_units: u.AbstractUnitSystem) -> dict[str, Any]:
        return {"units": field_units}

    @pytest.fixture
    def fields(self, fields_) -> dict[str, Any]:
        return copy.copy(fields_)

    @pytest.fixture(scope="class")
    def pot(
        self, pot_cls: type[gp.AbstractPotential], fields_: dict[str, Any]
    ) -> gp.AbstractPotential:
        """Create a concrete potential instance for testing."""
        return pot_cls(**fields_)

    # ---------------------------------

    @pytest.fixture(scope="class")
    def x(self, units: u.AbstractUnitSystem) -> gt.QuSz3:
        """Create a position vector for testing."""
        return u.Quantity(jnp.asarray([1, 2, 3], dtype=float), units["length"])

    @pytest.fixture(scope="class")
    def v(sel, units: u.AbstractUnitSystem) -> gt.QuSz3:
        """Create a velocity vector for testing."""
        return u.Quantity(jnp.asarray([4, 5, 6], dtype=float), units["speed"])

    @pytest.fixture(scope="class")
    def xv(self, x: gt.QuSz3, v: gt.QuSz3) -> gt.Sz6:
        """Create a phase-space vector for testing."""
        return jnp.concat([x.value, v.value])

    # ---------------------------------

    @pytest.fixture(scope="class")
    def batchx(self, units: u.AbstractUnitSystem) -> gt.BtQuSz3:
        """Create a batch of position vectors for testing."""
        return u.Quantity(
            jnp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float), units["length"]
        )

    @pytest.fixture(scope="class")
    def batchv(self, units: u.AbstractUnitSystem) -> gt.BtQuSz3:
        """Create a batch of velocity vectors for testing."""
        return u.Quantity(
            jnp.asarray([[4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=float),
            units["speed"],
        )

    @pytest.fixture(scope="class")
    def batchxv(self, batchx: gt.BtQuSz3, batchv: gt.BtQuSz3) -> gt.BtSz3:
        """Create a batch of phase-space vectors for testing."""
        return jnp.concatenate([batchx.value, batchv.value], axis=-1)

    # ---------------------------------

    @pytest.fixture(scope="class")
    def t(self) -> u.Quantity["time"]:
        """Create a time for testing."""
        return u.Quantity(0.0, "Gyr")

    ###########################################################################

    def test_init(
        self, pot_cls: type[gp.AbstractPotential], fields_: dict[str, Any]
    ) -> gp.AbstractPotential:
        """Create a concrete potential instance for testing."""
        pot = pot_cls(**fields_)
        assert isinstance(pot, pot_cls)

        # TODO: more tests

    # =========================================================================

    # ---------------------------------

    @abstractmethod
    def test_potential(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `potential` method."""
        ...

    def test_potential_batch(
        self, pot: gp.AbstractPotential, batchx: gt.BtQuSz3
    ) -> None:
        """Test the `AbstractPotential.potential` method."""
        # Test that the method works on batches.
        assert pot.potential(batchx, t=0).shape == batchx.shape[:-1]
        # Test that the batched method is equivalent to the scalar method
        assert jnp.allclose(
            pot.potential(batchx, t=0)[0],
            pot.potential(batchx[0], t=0),
            atol=u.Quantity(1e-15, pot.units["specific energy"]),
        )

    # ---------------------------------

    def test_call(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.__call__` method."""
        assert jnp.equal(pot(x, 0), pot.potential(x, 0))

    @abstractmethod
    def test_gradient(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.gradient` method."""
        ...

    @abstractmethod
    def test_density(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.density` method."""
        ...

    @abstractmethod
    def test_hessian(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.hessian` method."""
        ...

    def test_acceleration(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.acceleration` method."""
        acc = convert(pot.acceleration(x, t=0), u.Quantity)
        grad = convert(pot.gradient(x, t=0), u.Quantity)
        assert jnp.array_equal(acc, -grad)

    # ---------------------------------
    # Convenience methods

    @abstractmethod
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        ...

    # =========================================================================

    def test_evaluate_orbit(self, pot: gp.AbstractPotential, xv: gt.Sz6) -> None:
        """Test the `AbstractPotential.evaluate_orbit` method."""
        ts = u.Quantity(jnp.linspace(0.0, 1.0, 100), "Myr")

        orbit = pot.evaluate_orbit(xv, ts)
        assert isinstance(orbit, gd.Orbit)
        assert orbit.shape == (len(ts.value),)  # TODO: don't use .value
        assert jnp.array_equal(orbit.t, ts)

    def test_evaluate_orbit_batch(self, pot: gp.AbstractPotential, xv: gt.Sz6) -> None:
        """Test the `AbstractPotential.evaluate_orbit` method."""
        ts = u.Quantity(jnp.linspace(0.0, 1.0, 100), "Myr")

        # Simple batch
        orbits = pot.evaluate_orbit(xv[None, :], ts)
        assert isinstance(orbits, gd.Orbit)
        assert orbits.shape == (len(ts), 1)
        assert jnp.allclose(orbits.t, ts, atol=u.Quantity(1e-16, "Myr"))

        # More complicated batch
        xv2 = jnp.stack([xv, xv], axis=0)
        orbits = pot.evaluate_orbit(xv2, ts)
        assert isinstance(orbits, gd.Orbit)
        assert orbits.shape == (len(ts), 2)
        assert jnp.allclose(orbits.t, ts, atol=u.Quantity(1e-16, "Myr"))


##############################################################################


@final
class TestAbstractPotential(AbstractPotential_Test):
    """Test the `galax.potential.AbstractPotential` class."""

    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.AbstractPotential]:
        class TestPotential(gp.AbstractPotential):
            m_tot: gpp.AbstractParameter = gpp.ParameterField(
                dimensions="mass", default=u.Quantity(1e12, "Msun")
            )
            units: u.AbstractUnitSystem = eqx.field(default=galactic, static=True)
            constants: ImmutableMap[str, u.Quantity] = eqx.field(
                default=default_constants, converter=ImmutableMap
            )

            @partial(jax.jit, inline=True)
            def _potential(  # TODO: inputs w/ units
                self, q: gt.BtQuSz3, t: gt.BBtRealQuSz0, /
            ) -> gt.SpecificEnergyBtSz0:
                return (
                    self.constants["G"]
                    * self.m_tot(t)
                    / jnp.linalg.vector_norm(q, axis=-1)
                )

        return TestPotential

    ###########################################################################

    def test_init(self, pot_cls) -> None:
        """Test the initialization of `AbstractPotential`."""
        # Test that the abstract class cannot be instantiated
        with pytest.raises(TypeError):
            gp.AbstractPotential()

        # Test that the concrete class can be instantiated
        pot = pot_cls()
        assert isinstance(pot, gp.AbstractPotential)

    # =========================================================================

    # ---------------------------------

    def test_potential(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.potential` method."""
        assert jnp.allclose(
            pot.potential(x, t=0),
            u.Quantity(1.20227527, "kpc2/Myr2"),
            atol=u.Quantity(1e-8, "kpc2/Myr2"),
        )

    # ---------------------------------

    def test_gradient(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.gradient` method."""
        expect = u.Quantity(
            [-0.08587681, -0.17175361, -0.25763042], pot.units["acceleration"]
        )
        got = convert(pot.gradient(x, t=0), u.Quantity)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.density` method."""
        # TODO: fix negative density!!!
        expect = u.Quantity(-2.647e-7, pot.units["mass density"])
        assert jnp.allclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.hessian` method."""
        expected = u.Quantity(
            jnp.asarray(
                [
                    [-0.06747463, 0.03680435, 0.05520652],
                    [0.03680435, -0.01226812, 0.11041304],
                    [0.05520652, 0.11041304, 0.07974275],
                ]
            ),
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expected, atol=u.Quantity(1e-8, "1/Myr2")
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [-0.06747463, 0.03680435, 0.05520652],
                [0.03680435, -0.01226812, 0.11041304],
                [0.05520652, 0.11041304, 0.07974275],
            ],
            pot.units["frequency drift"],
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
