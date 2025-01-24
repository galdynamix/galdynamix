"""vectorized wrapper around `diffrax.DenseInterpolation`.

This is private API.

"""

__all__ = ["VectorizedDenseInterpolation"]

from collections.abc import Callable
from functools import partial
from typing import Any, TypeAlias, cast, final
from typing_extensions import override

import diffrax as dfx
import equinox as eqx
import jax
from jaxtyping import Array, ArrayLike, Float, Int, PyTree, Real, Shaped

import quaxed.numpy as jnp

BatchedRealTimes: TypeAlias = Real[Array, "{self.batch_shape} times"]
BatchedRealScalar: TypeAlias = Real[Array, "{self.batch_shape}"]
BatchedIntScalar: TypeAlias = Int[Array, "{self.batch_shape}"]
RealScalarLike: TypeAlias = Float[ArrayLike, ""] | Int[ArrayLike, ""]
VecDenseInfos: TypeAlias = dict[
    str, PyTree[Shaped[Array, "{self.batch_shape} times-1 ..."]]
]
Shape: TypeAlias = tuple[int, ...]


class AbstractVectorizedDenseInterpolation(dfx.AbstractPath[Any]):  # type: ignore[misc]
    """ABC for vectorized wrapper around a `diffrax.DenseInterpolation`."""

    #: Dense interpolation with flattened batch dimensions.
    scalar_interpolation: eqx.AbstractVar[dfx.DenseInterpolation]

    #: The batch shape of the interpolation without vectorization over the
    #: solver that produced this interpolation. E.g.
    batch_shape: eqx.AbstractVar[Shape]

    #: The shape of the solution.
    y0_shape: eqx.AbstractVar[PyTree[Shape, "Y"]]

    @property
    def batch_ndim(self) -> int:
        """The number of batch dimensions."""
        return len(self.batch_shape)

    @eqx.filter_jit  # type: ignore[misc]
    def __call__(
        self,
        t0: Real[Array, "time"],
        t1: Real[Array, "time"] | None = None,
        left: bool = True,  # noqa: FBT001, FBT002
    ) -> PyTree[Shaped[Array, "?*shape"], "Y"]:
        return self.evaluate(t0, t1, left)

    # =======================
    # DenseInterpolation API
    # modified to have batch dimensions.

    @override
    @eqx.filter_jit  # type: ignore[misc]
    def evaluate(
        self,
        t0: Real[Array, "*#time"],
        t1: Real[Array, "*#time"] | None = None,
        left: bool = True,
    ) -> PyTree[Shaped[Array, "?*shape"], "Y"]:
        """Evaluate the interpolation at any point in the region of integration.

        Args:
            t0: The point to evaluate the solution at.
            t1: If passed, then the increment from `t0` to `t1` is returned.
                (``=evaluate(t1) - evaluate(t0)``)
            left: When evaluating at a jump in the solution, whether to return
                the left-limit or the right-limit at that point.

        Return:
            The solution at the given time.
            Shape (*batch, *time, *y0.shape)

        """
        # If t1, then return the difference
        if t1 is not None:
            t0, t1 = jnp.broadcast_arrays(t0, t1)
            return self.evaluate(t1, left=left) - self.evaluate(t0, left=left)

        # Prepare t0 for evaluation
        t0 = jnp.asarray(t0)  # ensure array
        t0shape = t0.shape  # store shape for unpacking
        t0 = jnp.atleast_1d(t0).flatten().astype(float)  # ensure >= 1D for vmap

        # Evaluate the scalar interpolation over the batch dimension of the
        # interpolator and an array of times.
        ys = jax.vmap(  # vmap over the batch dimension of the interpolator
            lambda interp: jax.vmap(partial(interp.evaluate, left=left))(t0)
        )(self.scalar_interpolation)

        # Reshape the result to match the input shape in the time axes.
        # Since the interp is flattened, this is always the 1st index.
        ys = jax.tree.map(lambda x: x.reshape(x.shape[0], *t0shape, *x.shape[2:]), ys)

        # Reshape the 0th dimension back to the original batch shape.
        ys = jax.tree.map(lambda x: x.reshape(*self.batch_shape, *x.shape[1:]), ys)

        return ys  # noqa: RET504

    @property
    def t0(self) -> BatchedRealScalar:
        """The start time of the interpolation."""
        flatt0 = jax.vmap(lambda x: x.t0)(self.scalar_interpolation)
        return flatt0.reshape(*self.batch_shape)

    @property
    def t1(self) -> BatchedRealScalar:
        """The end time of the interpolation."""
        flatt1 = jax.vmap(lambda x: x.t1)(self.scalar_interpolation)
        return flatt1.reshape(*self.batch_shape)

    @property
    def ts(self) -> BatchedRealTimes:
        """The times of the interpolation."""
        return self.scalar_interpolation.ts

    @property
    def ts_size(self) -> Int[Array, "..."]:  # TODO: shape
        """The number of times in the interpolation."""
        return self.scalar_interpolation.ts_size

    @property
    def infos(self) -> VecDenseInfos:
        """The infos of the interpolation."""
        return cast(VecDenseInfos, self.scalar_interpolation.infos)

    @property
    def interpolation_cls(self) -> Callable[..., dfx.AbstractLocalInterpolation]:
        """The interpolation class of the interpolation."""
        return cast(
            Callable[..., dfx.AbstractLocalInterpolation],
            self.scalar_interpolation.interpolation_cls,
        )

    @property
    def direction(self) -> BatchedIntScalar:
        """Direction vector."""
        return self.scalar_interpolation.direction

    @property
    def t0_if_trivial(self) -> BatchedRealScalar:
        """The start time of the interpolation if scalar input."""
        return self.scalar_interpolation.t0_if_trivial

    @property
    def y0_if_trivial(self) -> PyTree[RealScalarLike, "Y"]:  # TODO: shape
        """The start value of the interpolation if scalar input."""
        return self.scalar_interpolation.y0_if_trivial


# =============================================================================


@final
class VectorizedDenseInterpolation(AbstractVectorizedDenseInterpolation):
    """Vectorized wrapper around a `diffrax.DenseInterpolation`.

    This also works on non-batched interpolations.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import diffrax as dfx

    We'll start with a non-batched interpolation:

    >>> vector_field = lambda t, y, args: -y
    >>> term = dfx.ODETerm(vector_field)
    >>> solver = dfx.Dopri5()
    >>> ts = jnp.array([0.0, 1, 2, 3])
    >>> saveat = dfx.SaveAt(ts=ts, dense=True)
    >>> stepsize_controller = dfx.PIDController(rtol=1e-5, atol=1e-5)

    >>> sol = dfx.diffeqsolve(
    ...     term, solver, t0=0, t1=3, dt0=0.1, y0=1, saveat=saveat,
    ...     stepsize_controller=stepsize_controller)
    >>> interp = VectorizedDenseInterpolation(sol.interpolation)
    >>> interp
    VectorizedDenseInterpolation(
      scalar_interpolation=DenseInterpolation(
        ts=f64[1,4097],
        ts_size=weak_i64[1],
        infos={'k': f64[1,4096,7], 'y0': f64[1,4096], 'y1': f64[1,4096]},
        interpolation_cls=<class 'diffrax._solver.dopri5._Dopri5Interpolation'>,
        direction=weak_i64[1],
        t0_if_trivial=f64[1],
        y0_if_trivial=f64[1]
      ),
      batch_shape=(),
      y0_shape=()
    )

    This can be evaluated by the normal means:

    >>> interp.evaluate(ts[-1])  # scalar evaluation
    Array(0.04978961, dtype=float64)

    It also works on arrays, without needed to manually apply `jax.vmap`:

    >>> interp.evaluate(ts)  # It works on arrays!
    Array([1. , 0.36788338, 0.13533922, 0.04978961], dtype=float64)

    >>> interp.evaluate(ts, ts[0])  # t1 - t0 mixed scalar and array
    Array([0. , 0.63211662, 0.86466078, 0.95021039], dtype=float64)

    Better yet, the time array may be arbitrarily shaped:

    >>> interp.evaluate(ts.reshape(2, 2)).round(3)
    Array([[1.   , 0.368],
           [0.135, 0.05 ]], dtype=float64)

    As a convenience, we can also apply the `VectorizedDenseInterpolation` to
    the solution to modify the interpolation "in-place" (when in a jitted
    context, otherwise out-of-place, returning a copy):

    >>> sol = VectorizedDenseInterpolation.apply_to_solution(sol)
    >>> isinstance(sol, dfx.Solution)
    True
    >>> isinstance(sol.interpolation, VectorizedDenseInterpolation)
    True

    Now we'll batch the interpolation:

    >>> @jax.vmap
    ... def solve(y0):
    ...     sol = dfx.diffeqsolve(
    ...         term, solver, t0=0, t1=3, dt0=0.1, y0=y0, saveat=saveat,
    ...         stepsize_controller=stepsize_controller)
    ...     return sol
    >>> sol = solve(jnp.array([1, 2, 3]))
    >>> interp = VectorizedDenseInterpolation(sol.interpolation)

    >>> interp.evaluate(ts[-1]).round(3)  # scalar eval of batched interp
    Array([0.05 , 0.1  , 0.149], dtype=float64)

    >>> interp.evaluate(ts).round(3)  # array eval of batched interp
    Array([[1.   , 0.368, 0.135, 0.05 ],
           [2.   , 0.736, 0.271, 0.1  ],
           [3.   , 1.104, 0.406, 0.149]], dtype=float64)

    >>> interp.evaluate(ts, ts[0]).round(3)  # mixed scalar and array eval
    Array([[0.   , 0.632, 0.865, 0.95 ],
           [0.   , 1.264, 1.729, 1.9  ],
           [0.   , 1.896, 2.594, 2.851]], dtype=float64)

    >>> ys = interp.evaluate(ts.reshape(2, 2)).round(3)  # arbitrary shape eval
    >>> ys
    Array([[[1.   , 0.368],
            [0.135, 0.05 ]],
           [[2.   , 0.736],
            [0.271, 0.1  ]],
           [[3.   , 1.104],
            [0.406, 0.149]]], dtype=float64)
    >>> ys.shape  # (batch, *times)
    (3, 2, 2)

    Let's inspect the rest of the API.

    >>> interp.scalar_interpolation  # (flattened) original interpolation
    DenseInterpolation(
      ts=f64[3,4097],
      ts_size=weak_i64[3],
      infos={'k': f64[3,4096,7], 'y0': f64[3,4096], 'y1': f64[3,4096]},
      interpolation_cls=<class 'diffrax._solver.dopri5._Dopri5Interpolation'>,
      direction=weak_i64[3],
      t0_if_trivial=f64[3],
      y0_if_trivial=f64[3]
    )

    >>> interp.batch_shape  # batch shape of the interpolation
    (3,)

    >>> interp.t0  # start time of the interpolation
    Array([0., 0., 0.], dtype=float64)

    >>> interp.t1  # end time of the interpolation
    Array([3., 3., 3.], dtype=float64)

    >>> interp.ts.shape  # times of the interpolation
    (3, 4097)

    >>> interp.ts_size
    Array([8, 9, 9], dtype=int64, weak_type=True)

    >>> jax.tree.map(lambda x: x.shape, interp.infos)
    {'k': (3, 4096, 7), 'y0': (3, 4096), 'y1': (3, 4096)}

    >>> interp.interpolation_cls
    <class 'diffrax..._Dopri5Interpolation'>

    >>> interp.direction
    Array([1, 1, 1], dtype=int64, weak_type=True)

    >>> interp.t0_if_trivial
    Array([0., 0., 0.], dtype=float64)

    >>> interp.y0_if_trivial
    Array([1., 2., 3.], dtype=float64)

    """

    #: Dense interpolation with flattened batch dimensions.
    scalar_interpolation: dfx.DenseInterpolation

    #: The batch shape of the interpolation without vectorization over the
    #: solver that produced this interpolation. E.g.
    batch_shape: Shape

    #: The shape of the solution.
    y0_shape: PyTree[Shape, "Y"]

    def __init__(self, interp: dfx.DenseInterpolation, /) -> None:
        # # Store the batch shape
        bshape = interp.t0_if_trivial.shape
        bshape = eqx.error_if(
            bshape,
            bshape != interp.t0_if_trivial.shape,
            "batch_shape must match the shape of the ts_size of the interpolation",
        )
        self.batch_shape = bshape
        self.y0_shape = jax.tree.map(
            lambda x: x.shape[self.batch_ndim :], interp.y0_if_trivial
        )

        # Flatten the batch shape of the interpolation
        self.scalar_interpolation = jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[self.batch_ndim :]),
            interp,
            is_leaf=eqx.is_array,
        )

    # =======================
    # Convenience methods

    @classmethod
    def apply_to_solution(cls, soln: dfx.Solution, /) -> dfx.Solution:
        """Make a `diffrax.Solution` interpolation vectorized.

        This does an out-of-place transformation, wrapping the interpolation
        in a `VectorizedDenseInterpolation`.

        """
        if soln.interpolation is None:
            return soln

        return eqx.tree_at(
            lambda tree: tree.interpolation, soln, cls(soln.interpolation)
        )
