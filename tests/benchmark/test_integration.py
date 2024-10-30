"""Benchmark tests for orbit integration."""
# pylint: disable=redefined-outer-name

from functools import partial
from typing import Any

import diffrax
import equinox as eqx
import jax
import pytest
from jaxtyping import Shaped
from plum import convert

import coordinax as cx
import quaxed.numpy as jnp
from unxt import AbstractUnitSystem, Quantity
from unxt.unitsystems import galactic

import galax.coordinates as gc
import galax.potential as gp
import galax.typing as gt

###############################################################################
# Fixtures


@pytest.fixture
def batch_len() -> int:
    """Length of batch dimension for the `w0` coordinate."""
    return 5_700


@pytest.fixture
def usys() -> AbstractUnitSystem:
    """Return the unit system."""
    return galactic


# ------------------------------------------


@pytest.fixture
def q0_v(batch_len: int) -> cx.CartesianPosition3D:
    """Position Vector."""
    return cx.CartesianPosition3D(
        x=Quantity(jnp.linspace(7.5, 8.5, batch_len), "kpc"),
        y=Quantity(jnp.zeros(batch_len), "kpc"),
        z=Quantity(jnp.zeros(batch_len), "kpc"),
    )


@pytest.fixture
def q0_q(q0_v: cx.CartesianPosition3D) -> Quantity["length"]:
    """Position coordinate."""
    return convert(q0_v, Quantity)


# ------------------------------------------


@pytest.fixture
def p0_v(batch_len: int) -> cx.CartesianVelocity3D:
    """Velocity Vector."""
    return cx.CartesianVelocity3D(
        d_x=Quantity(jnp.zeros(batch_len), "km/s"),
        d_y=Quantity(jnp.linspace(225, 235, batch_len), "km/s"),
        d_z=Quantity(jnp.zeros(batch_len), "km/s"),
    )


@pytest.fixture
def p0_q(p0_v: cx.CartesianVelocity3D) -> Quantity["speed"]:
    """Velocity coordinate."""
    return convert(p0_v, Quantity)


# ------------------------------------------


@pytest.fixture
def t0(batch_len: int) -> Quantity["time"]:
    """Time coordinate."""
    return Quantity(jnp.zeros(batch_len), "Myr")


@pytest.fixture
def t0_scalar(t0: Quantity["time"]) -> Shaped[Quantity["time"], ""]:
    """Time coordinate."""
    return t0[0]


@pytest.fixture
def t0_scalar_value(t0_scalar: Quantity["time"], usys: AbstractUnitSystem) -> float:
    """Time coordinate."""
    return t0_scalar.to_units(usys["time"]).value


# ------------------------------------------


@pytest.fixture
def w0_v(
    q0_v: cx.CartesianPosition3D, p0_v: cx.CartesianVelocity3D, t0: Quantity
) -> gc.PhaseSpacePosition:
    """Phase-space coordinate."""
    return gc.PhaseSpacePosition(q=q0_v, p=p0_v, t=t0)


@pytest.fixture
def w0_a(w0_v: gc.PhaseSpacePosition) -> gt.BatchVec6:
    """Phase-space coordinate."""
    return w0_v.w(units="galactic")


@pytest.fixture
def w0_a_scalar(w0_a: gt.BatchVec6) -> gt.Vec6:
    """Phase-space coordinate."""
    return w0_a[0]


# ------------------------------------------


@pytest.fixture
def pot() -> gp.AbstractPotentialBase:
    """Potential in which to integrate the orbits."""
    return gp.MilkyWayPotential()


# ------------------------------------------


@pytest.fixture
def solver() -> diffrax.AbstractSolver:
    """Solver keyword arguments."""
    return diffrax.Dopri8(scan_kind="bounded")


@pytest.fixture
def stepsize_controller() -> diffrax.AbstractStepSizeController:
    return diffrax.PIDController(rtol=1e-7, atol=1e-7)


@pytest.fixture
def t1_scalar() -> gt.TimeScalar:
    """Time coordinate."""
    return Quantity(300.0, "Myr")


@pytest.fixture
def saveat(t0_scalar: gt.TimeScalar, t1_scalar: gt.TimeScalar) -> gt.QVecTime:
    """Time coordinate."""
    return jnp.linspace(t0_scalar, t1_scalar * 0.99, 100)


@pytest.fixture
def saveat_scalar(saveat: gt.QVecTime) -> Shaped[Quantity["time"], "1"]:
    """Time coordinate."""
    return saveat[-1:]


###############################################################################
# Diffrax:
#
# These are included as baseline benchmarks for comparison with the
# Galax implementation.


@pytest.mark.benchmark(group="diffrax")
def test_diffrax_scalar_inputs(
    pot: gp.AbstractPotentialBase,
    w0_a_scalar: gt.Vec6,
    t0_scalar: Quantity["time"],
    t1_scalar: Quantity["time"],
    saveat_scalar: Shaped[Quantity["time"], "1"],
    stepsize_controller: diffrax.AbstractStepSizeController,
    solver: diffrax.AbstractSolver,
) -> None:
    """Benchmark the Diffrax implementation with scalar inputs."""
    _ = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(pot._dynamics_deriv),
        solver=solver,
        t0=t0_scalar.value,
        t1=t1_scalar.value,
        y0=w0_a_scalar,
        dt0=None,
        args=(),
        saveat=diffrax.SaveAt(t0=False, t1=False, ts=saveat_scalar.value, dense=False),
        stepsize_controller=stepsize_controller,
        max_steps=50_0000,
    )


@pytest.mark.benchmark(group="diffrax")
def test_diffrax_times_inputs(
    pot: gp.AbstractPotentialBase,
    w0_a_scalar: gt.Vec6,
    t0_scalar: Quantity["time"],
    t1_scalar: Quantity["time"],
    saveat: gt.QVecTime,
    stepsize_controller: diffrax.AbstractStepSizeController,
    solver: diffrax.AbstractSolver,
) -> None:
    """Benchmark the Diffrax implementation with saveat times inputs."""
    _ = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(pot._dynamics_deriv),
        solver=solver,
        t0=t0_scalar.value,
        t1=t1_scalar.value,
        y0=w0_a_scalar,
        dt0=None,
        args=(),
        saveat=diffrax.SaveAt(t0=False, t1=False, ts=saveat.value, dense=False),
        stepsize_controller=stepsize_controller,
        max_steps=50_0000,
    )


@pytest.mark.benchmark(group="diffrax")
def test_diffrax_batched_inputs(
    pot: gp.AbstractPotentialBase,
    w0_a: gt.Vec6,
    t0_scalar: Quantity["time"],
    t1_scalar: Quantity["time"],
    saveat_scalar: Shaped[Quantity["time"], "1"],
    stepsize_controller: diffrax.AbstractStepSizeController,
    solver: diffrax.AbstractSolver,
) -> None:
    """Benchmark the Diffrax implementation with saveat times inputs."""
    _ = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(pot._dynamics_deriv),
        solver=solver,
        t0=t0_scalar.value,
        t1=t1_scalar.value,
        y0=w0_a,
        dt0=None,
        args=(),
        saveat=diffrax.SaveAt(t0=False, t1=False, ts=saveat_scalar.value, dense=False),
        stepsize_controller=stepsize_controller,
        max_steps=50_0000,
    )


@pytest.mark.benchmark(group="diffrax")
def test_diffrax_batched_and_times_inputs(
    pot: gp.AbstractPotentialBase,
    w0_a: gt.BatchVec6,
    t0_scalar: Quantity["time"],
    t1_scalar: Quantity["time"],
    saveat: gt.QVecTime,
    stepsize_controller: diffrax.AbstractStepSizeController,
    solver: diffrax.AbstractSolver,
) -> None:
    """Benchmark the Diffrax implementation with saveat times inputs."""
    _ = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(pot._dynamics_deriv),
        solver=solver,
        t0=t0_scalar.value,
        t1=t1_scalar.value,
        y0=w0_a,
        dt0=None,
        args=(),
        saveat=diffrax.SaveAt(t0=False, t1=False, ts=saveat.value, dense=False),
        stepsize_controller=stepsize_controller,
        max_steps=50_0000,
    )


def test_diffrax_batched_t0_inputs(
    benchmark: Any,
    pot: gp.AbstractPotentialBase,
    w0_a_scalar: gt.Vec6,
    t0: gt.QVecTime,
    t1_scalar: gt.TimeScalar,
    saveat_scalar: Shaped[Quantity["time"], "1"],
    stepsize_controller: diffrax.AbstractStepSizeController,
    solver: diffrax.AbstractSolver,
) -> None:
    """Benchmark the Diffrax implementation with scalar inputs."""
    terms = diffrax.ODETerm(pot._dynamics_deriv)

    @eqx.filter_jit
    @partial(jax.numpy.vectorize, signature="()->()")
    def do(t0: gt.FloatScalar) -> diffrax.Solution:
        soln: diffrax.Solution = diffrax.diffeqsolve(
            terms=terms,
            solver=solver,
            t0=t0,
            t1=t1_scalar.value,
            y0=w0_a_scalar,
            dt0=None,
            args=(),
            saveat=diffrax.SaveAt(
                t0=False, t1=False, ts=saveat_scalar.value, dense=False
            ),
            stepsize_controller=stepsize_controller,
            max_steps=50_0000,
        )
        return soln

    result = benchmark(lambda: do(t0.value))
    assert result.ys.shape == (5700, 1, 6)


def test_diffrax_batched_t0_and_times_inputs(
    benchmark: Any,
    pot: gp.AbstractPotentialBase,
    w0_a_scalar: gt.Vec6,
    t0: gt.QVecTime,
    t1_scalar: gt.TimeScalar,
    saveat: Shaped[Quantity["time"], "1"],
    stepsize_controller: diffrax.AbstractStepSizeController,
    solver: diffrax.AbstractSolver,
) -> None:
    """Benchmark the Diffrax implementation with scalar inputs."""
    terms = diffrax.ODETerm(pot._dynamics_deriv)

    @eqx.filter_jit
    @partial(jax.numpy.vectorize, signature="()->()")
    def do(t0: gt.FloatScalar) -> diffrax.Solution:
        soln: diffrax.Solution = diffrax.diffeqsolve(
            terms=terms,
            solver=solver,
            t0=t0,
            t1=t1_scalar.value,
            y0=w0_a_scalar,
            dt0=None,
            args=(),
            saveat=diffrax.SaveAt(t0=False, t1=False, ts=saveat.value, dense=False),
            stepsize_controller=stepsize_controller,
            max_steps=50_0000,
        )
        return soln

    result = benchmark(lambda: do(t0.value))
    assert result.ys.shape == (5700, 100, 6)
