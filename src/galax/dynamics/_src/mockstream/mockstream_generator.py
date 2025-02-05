"""Generator for mock streams."""

__all__ = ["MockStreamGenerator"]

from dataclasses import KW_ONLY, replace
from functools import partial
from typing import TypeAlias, cast, final

import equinox as eqx
import jax
import jax.extend as jex
from jaxtyping import PRNGKeyArray

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import UncheckedQuantity as FastQ

import galax.coordinates as gc
import galax.typing as gt
from .core import MockStream, MockStreamArm
from .df import AbstractStreamDF, ProgenitorMassCallable
from .utils import cond_reverse
from galax.dynamics._src.dynamics import DynamicsSolver
from galax.dynamics._src.orbit import Orbit, evaluate_orbit
from galax.potential import AbstractPotential

Carry: TypeAlias = tuple[gt.IntSz0, gt.SzN, gt.SzN]

_default_solver: DynamicsSolver = DynamicsSolver()


@final
class MockStreamGenerator(eqx.Module):  # type: ignore[misc]
    """Generate a mock stellar stream in the specified external potential."""

    df: AbstractStreamDF
    """Distribution function for generating mock streams.

    E.g. ``galax.dynamics.mockstream.FardalStreamDF``.
    """

    potential: AbstractPotential
    """Potential in which the progenitor orbits and creates a stream."""

    _: KW_ONLY
    progenitor_solver: DynamicsSolver = eqx.field(default=_default_solver, static=True)
    """Solver for the progenitor orbit."""

    stream_solver: DynamicsSolver = eqx.field(default=_default_solver, static=True)
    """Solver for the stream."""

    @property
    def units(self) -> u.AbstractUnitSystem:
        """Units of the potential."""
        return cast(u.AbstractUnitSystem, self.potential.units)

    # ==========================================================================

    def _progenitor_trajectory(
        self, w0: gc.PhaseSpacePosition, ts: gt.QuSzTime
    ) -> Orbit:
        """Integrate the progenitor orbit."""
        return cast(
            Orbit,
            evaluate_orbit(self.potential, w0, ts, solver=self.progenitor_solver),
        )

    # ==========================================================================

    @partial(jax.jit)
    def _run_scan(  # TODO: output shape depends on the input shape
        self,
        ts: gt.QuSzTime,
        mock0_lead: MockStreamArm,
        mock0_trail: MockStreamArm,
    ) -> tuple[gt.BtSz6, gt.BtSz6]:
        """Generate stellar stream by scanning over the release model/integration.

        Better for CPU usage.
        """
        w0_lead = mock0_lead.w(units=self.units)
        w0_trail = mock0_trail.w(units=self.units)
        t_f = ts[-1] + u.Quantity(1e-3, ts.unit)  # TODO: not bump in the final time.

        def one_pt_intg(
            carry: Carry, _: gt.IntSz0
        ) -> tuple[Carry, tuple[gt.SzN, gt.SzN]]:
            """Integrate one point along the stream.

            Parameters
            ----------
            carry : tuple[int, SzN, SzN]
                Initial state of the particle at index `idx`.
            idx : int
                Index of the current point.
            """
            i, w0_l_i, w0_t_i = carry
            tstep = jnp.asarray([ts[i], t_f])

            def integ_ics(ics: gt.Sz6) -> gt.SzN:
                # TODO: only return the final state
                return evaluate_orbit(
                    self.potential, ics, tstep, solver=self.stream_solver
                ).w(units=self.units)[-1]

            # vmap integration over leading and trailing arm
            w0_lt_i = jnp.vstack([w0_l_i, w0_t_i])
            w_l, w_t = jax.vmap(integ_ics, in_axes=(0,))(w0_lt_i)
            # Prepare for next iteration
            carry_out = (i + 1, w0_lead[i + 1, :], w0_trail[i + 1, :])
            return carry_out, (w_l, w_t)

        carry_init = (0, w0_lead[0, :], w0_trail[0, :])
        pt_ids = jnp.arange(len(w0_lead))
        lead_arm_w, trail_arm_w = jax.lax.scan(one_pt_intg, carry_init, pt_ids)[1]

        return lead_arm_w, trail_arm_w

    @partial(jax.jit)
    def _run_vmap(  # TODO: output shape depends on the input shape
        self,
        ts: gt.QuSzTime,
        mock0_lead: MockStreamArm,
        mock0_trail: MockStreamArm,
    ) -> tuple[gt.BtSz6, gt.BtSz6]:
        """Generate stellar stream by vmapping over the release model/integration.

        Better for GPU usage.
        """
        t_f = ts[-1] + u.Quantity(1e-3, ts.unit)  # TODO: not bump in the final time.

        @partial(jax.jit, inline=True)
        def one_pt_intg(
            i: gt.IntSz0, w0_l_i: gt.Sz6, w0_t_i: gt.Sz6
        ) -> tuple[gt.Sz6, gt.Sz6]:
            tstep = jnp.asarray([ts[i], t_f])
            w_lead = evaluate_orbit(
                self.potential, w0_l_i, tstep, solver=self.stream_solver
            ).w(units=self.potential.units)[-1]
            w_trail = evaluate_orbit(
                self.potential, w0_t_i, tstep, solver=self.stream_solver
            ).w(units=self.potential.units)[-1]
            return w_lead, w_trail

        w0_lead = mock0_lead.w(units=self.units)
        w0_trail = mock0_trail.w(units=self.units)
        pt_ids = jnp.arange(len(w0_lead))
        lead_arm_w, trail_arm_w = jax.vmap(one_pt_intg)(pt_ids, w0_lead, w0_trail)
        return lead_arm_w, trail_arm_w

    @partial(jax.jit, static_argnames=("vmapped",))
    def run(
        self,
        rng: PRNGKeyArray,
        ts: gt.QuSzTime,
        prog_w0: gc.PhaseSpacePosition | gt.Sz6,
        prog_mass: gt.FloatQuSz0 | ProgenitorMassCallable,
        *,
        vmapped: bool | None = None,
    ) -> tuple[MockStream, gc.PhaseSpacePosition]:
        """Generate mock stellar stream.

        Parameters
        ----------
        rng : :class:`jaxtyping.PRNGKeyArray`
            Random number generator.
        ts : Quantity[float, (time,), "time"]
            Stripping times.
        prog_w0 : PhaseSpacePosition[float, ()]
            Initial conditions of the progenitor.

            The recommended way to pass in the progenitor's initial conditions
            is as a :class:`~galax.coordinates.PhaseSpacePosition` object with a
            set time. This is the most explicit and is guaranteed to have the
            correct units and no surprises about the progenitor.

            .. note::

                If the time is not set, it is assumed to be the first stripping
                time.

            Alternatively, you can pass in a 6-element array of the Cartesian
            phase-space coordinates (x, y, z, vx, vy, vz) in the same units as
            the potential.

        prog_mass : Quantity[float, (), "mass"] | `ProgenitorMassCallable`
            Mass of the progenitor. May be a Quantity or a callable that returns
            the progenitor mass at the given times.

        vmapped : bool | None, optional keyword-only
            Whether to use `jax.vmap` (`True`) or `jax.lax.scan` (`False`) to
            vectorize the integration. ``vmapped=True`` is recommended for GPU
            usage, while ``vmapped=False`` is recommended for CPU usage.  If
            `None` (default), then `jax.vmap` is used on GPU and `jax.lax.scan`
            otherwise.

        Returns
        -------
        mockstream : :class:`galax.dynamcis.MockStreamArm`
            Leading and/or trailing arms of the mock stream.
        prog_o : :class:`galax.coordinates.PhaseSpacePosition`
            The final phase-space(+time) position of the progenitor.
        """
        # Parse vmapped
        use_vmap = (
            jex.backend.get_backend().platform == "gpu" if vmapped is None else vmapped
        )

        # Ensure w0 is a PhaseSpacePosition
        w0: gc.PhaseSpacePosition
        if isinstance(prog_w0, gc.PhaseSpacePosition):
            w0 = prog_w0 if prog_w0.t is not None else replace(prog_w0, t=ts[0])
        else:
            w0 = gc.PhaseSpacePosition(
                q=u.Quantity(prog_w0[0:3], self.units["length"]),
                p=u.Quantity(prog_w0[3:6], self.units["speed"]),
                t=uconvert(self.potential.units["time"], ts[0]),
            )
        w0 = eqx.error_if(w0, w0.ndim > 0, "prog_w0 must be scalar")
        # TODO: allow for multiple progenitors

        # If the time stepping passed in is negative, assume this means that all
        # of the initial conditions are at *end time*, and we need to reverse
        # them before treating them as initial conditions
        ts = cond_reverse(ts[1] < ts[0], ts)

        # Integrate the progenitor orbit, evaluating at the stripping times
        prog_o = self._progenitor_trajectory(w0, ts)

        # TODO: here sep out lead vs trailing
        # Generate initial conditions from the DF, along the integrated
        # progenitor orbit. The release times are the stripping times.
        mock0 = self.df.sample(rng, self.potential, prog_o, prog_mass)

        if use_vmap:
            lead_arm_w, trail_arm_w = self._run_vmap(ts, mock0["lead"], mock0["trail"])
        else:
            lead_arm_w, trail_arm_w = self._run_scan(ts, mock0["lead"], mock0["trail"])

        t = jnp.ones_like(ts) * ts.value[-1]  # TODO: ensure this time is correct

        frame = (
            prog_w0.frame
            if isinstance(prog_w0, gc.PhaseSpacePosition)
            else gc.frames.SimulationFrame()
        )

        comps = {}
        comps["lead"] = MockStreamArm(
            q=FastQ(lead_arm_w[:, 0:3], self.units["length"]),
            p=FastQ(lead_arm_w[:, 3:6], self.units["speed"]),
            t=t,
            release_time=mock0["lead"].release_time,
            frame=frame,
        )
        comps["trail"] = MockStreamArm(
            q=FastQ(trail_arm_w[:, 0:3], self.units["length"]),
            p=FastQ(trail_arm_w[:, 3:6], self.units["speed"]),
            t=t,
            release_time=mock0["trail"].release_time,
            frame=frame,
        )

        return MockStream(comps), prog_o[-1]
