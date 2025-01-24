"""galax: Galactic Dynamix in Jax."""

__all__: list[str] = []

from typing import Literal, TypeAlias, overload

import quax
from jaxtyping import Array, Shaped

import coordinax as cx
import quaxed.numpy as jnp

import galax.typing as gt
from galax.utils._jax import quaxify

AnyScalar: TypeAlias = Shaped[Array, ""]
ArrayAnyShape: TypeAlias = Shaped[Array | quax.ArrayValue, "..."]


def vector_batched_shape(obj: cx.vecs.AbstractVector) -> tuple[gt.Shape, int]:
    """Return the batch and component shape of a vector."""
    return obj.shape, len(obj.components)


@overload
def batched_shape(
    arr: ArrayAnyShape | AnyScalar, /, *, expect_ndim: Literal[0]
) -> tuple[gt.Shape, gt.Shape]: ...


@overload
def batched_shape(
    arr: ArrayAnyShape | AnyScalar, /, *, expect_ndim: Literal[1]
) -> tuple[gt.Shape, tuple[int]]: ...


@overload
def batched_shape(
    arr: ArrayAnyShape | AnyScalar, /, *, expect_ndim: Literal[2]
) -> tuple[gt.Shape, tuple[int, int]]: ...


@overload
def batched_shape(
    arr: ArrayAnyShape | AnyScalar, /, *, expect_ndim: int
) -> tuple[gt.Shape, gt.Shape]: ...


@quaxify
def batched_shape(
    arr: ArrayAnyShape | AnyScalar | float | int, /, *, expect_ndim: int
) -> tuple[gt.Shape, gt.Shape]:
    """Return the (batch_shape, arr_shape) an array.

    Parameters
    ----------
    arr : array-like
        The array to get the shape of.
    expect_ndim : int
        The expected dimensionality of the array.

    Returns
    -------
    batch_shape : tuple[int, ...]
        The shape of the batch.
    arr_shape : tuple[int, ...]
        The shape of the array.

    Examples
    --------
    Standard imports:

        >>> import quaxed.numpy as jnp
        >>> from galax.utils._shape import batched_shape

    Expecting a scalar:

        >>> batched_shape(0, expect_ndim=0)
        ((), ())
        >>> batched_shape(jnp.asarray([1]), expect_ndim=0)
        ((1,), ())
        >>> batched_shape(jnp.asarray([1, 2, 3]), expect_ndim=0)
        ((3,), ())

    Expecting a 1D vector:

        >>> batched_shape(jnp.asarray(0), expect_ndim=1)
        ((), ())
        >>> batched_shape(jnp.asarray([1]), expect_ndim=1)
        ((), (1,))
        >>> batched_shape(jnp.asarray([1, 2, 3]), expect_ndim=1)
        ((), (3,))
        >>> batched_shape(jnp.asarray([[1, 2, 3]]), expect_ndim=1)
        ((1,), (3,))

    Expecting a 2D matrix:

        >>> batched_shape(jnp.asarray([[1]]), expect_ndim=2)
        ((), (1, 1))
        >>> batched_shape(jnp.asarray([[[1]]]), expect_ndim=2)
        ((1,), (1, 1))
        >>> batched_shape(jnp.asarray([[[1]], [[1]]]), expect_ndim=2)
        ((2,), (1, 1))
    """
    shape: gt.Shape = jnp.asarray(arr).shape
    ndim = len(shape)
    return shape[: ndim - expect_ndim], shape[ndim - expect_ndim :]


@quaxify
def expand_batch_dims(arr: ArrayAnyShape, /, ndim: int) -> ArrayAnyShape:
    """Expand the batch dimensions of an array.

    Parameters
    ----------
    arr : array-like
        The array to expand the batch dimensions of.
    ndim : int
        The number of batch dimensions to expand.

    Returns
    -------
    arr : array-like
        The array with expanded batch dimensions.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from galax.utils._shape import expand_batch_dims

    >>> expand_batch_dims(jnp.array(0), ndim=0).shape
    ()

    >>> expand_batch_dims(jnp.array([0]), ndim=0).shape
    (1,)

    >>> expand_batch_dims(jnp.array(0), ndim=1).shape
    (1,)

    >>> expand_batch_dims(jnp.array([0, 1]), ndim=1).shape
    (1, 2)
    """
    return jnp.expand_dims(arr, axis=tuple(range(ndim)))


@quaxify
def expand_arr_dims(arr: ArrayAnyShape, /, ndim: int) -> ArrayAnyShape:
    """Expand the array dimensions of an array.

    Parameters
    ----------
    arr : array-like
        The array to expand the array dimensions of.
    ndim : int
        The number of array dimensions to expand.

    Returns
    -------
    arr : array-like
        The array with expanded array dimensions.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from galax.utils._shape import expand_arr_dims

    >>> expand_arr_dims(jnp.array(0), ndim=0).shape
    ()

    >>> expand_arr_dims(jnp.array([0]), ndim=0).shape
    (1,)

    >>> expand_arr_dims(jnp.array(0), ndim=1).shape
    (1,)

    >>> expand_arr_dims(jnp.array([0, 0]), ndim=1).shape
    (2, 1)
    """
    nbatch = len(arr.shape)
    return jnp.expand_dims(arr, axis=tuple(nbatch + i for i in range(ndim)))
