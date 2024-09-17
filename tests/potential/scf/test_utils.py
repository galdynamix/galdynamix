"""Test :mod:`galax.potential._src.scf.utils`."""

import hypothesis
import hypothesis.extra.numpy as hnp
import jax
import numpy as np
import numpy.typing as npt
import scipy.special as sp
from hypothesis import assume, given, strategies as st

import quaxed.numpy as jnp

from galax.potential._src.scf.utils import (
    cartesian_to_spherical,
    factorial,
    psi_of_r,
    real_Ylm,
)


# TODO: use hnp.floating_dtypes()
# TODO: test more batch dimensions
def xyz_strategy() -> st.SearchStrategy[np.ndarray]:
    return hnp.arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(1, 100), st.integers(3, 3)),
        elements=st.floats(-10, 10, allow_subnormal=False, allow_nan=False),
    )


@given(xyz_strategy())
def test_cartesian_to_spherical(xyz):
    """Test the ``cartesian_to_spherical`` function."""
    assume(np.all(xyz.sum(axis=1) != 0))

    n = len(xyz)
    xyz = jnp.asarray(xyz)

    rthetaphi = cartesian_to_spherical(xyz)
    r = rthetaphi[..., 0]
    theta = rthetaphi[..., 1]
    phi = rthetaphi[..., 2]

    # Check
    assert r.shape == (n,)
    assert theta.shape == (n,)
    assert phi.shape == (n,)

    assert jnp.all(r >= 0)
    assert jnp.all(theta >= 0) & jnp.all(theta <= jnp.pi)
    assert jnp.all(phi >= -jnp.pi) & jnp.all(phi <= jnp.pi)


def test_cartesian_to_spherical_jac():
    """Test the ``cartesian_to_spherical`` function."""
    # Scalar
    xyz = jnp.asarray([1, 0, 0], dtype=float)
    assert xyz.shape == (3,)

    output = jax.jacfwd(cartesian_to_spherical)(xyz)
    np.testing.assert_array_equal(
        output, [[1.0, 0.0, 0.0], [-0.0, -0.0, -1.0], [0.0, 1.0, 0.0]]
    )

    # Vector
    xyz = jnp.asarray([[1, 0, 0], [0, 1, 0]], dtype=float)
    assert xyz.shape == (2, 3)

    output = jax.jacfwd(cartesian_to_spherical)(xyz)
    assert output.shape == (2, 3, 2, 3)  # WTF?
    np.testing.assert_array_equal(
        output[0, :, 0, :], [[1.0, 0.0, 0.0], [-0.0, -0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    np.testing.assert_array_equal(
        output[1, :, 1, :], [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]
    )


# =============================================================================


@given(
    n=st.integers(0, 100)
    | hnp.arrays(dtype=int, shape=hnp.array_shapes(), elements=st.integers(0, 100))
)
def test_factorial(n: int | npt.NDArray[np.int_]):
    """Test the ``factorial`` function."""
    got = factorial(jnp.asarray(n))
    expected = sp.factorial(n)
    np.testing.assert_allclose(got, expected)


# =============================================================================


@given(
    r=st.floats(0, 100)
    | hnp.arrays(dtype=float, shape=hnp.array_shapes(), elements=st.floats(0, 100)),
)
def test_psi_of_r(r):
    """Test the ``psi_of_r`` function."""
    got = psi_of_r(r)
    expected = (r - 1) / (r + 1)
    np.testing.assert_allclose(got, expected)


# =============================================================================


def test_Ylm_jitting():
    """Test the ``real_Ylm`` function."""
    got = real_Ylm(5, 0, np.pi)
    expected = np.real(sp.sph_harm(0, 5, 0, np.pi))
    np.testing.assert_allclose(got, expected)


@hypothesis.settings(deadline=500)
@given(l=st.integers(1, 25), m=st.integers(1, 25), theta=st.floats(0, np.pi))
def test_real_Ylm(l, m, theta):
    """Test the ``real_Ylm`` function."""
    assume(theta != 0)
    assume(m <= l)
    got = real_Ylm(l, m, theta)
    expected = np.real(sp.sph_harm(m, l, 0, theta))
    np.testing.assert_allclose(got, expected)


# # TODO: mark as slow
# # TODO: test batch dimensions of l, m, theta
# @hypothesis.settings(deadline=500)
# @given(
#     l=st.integers(1, 25),
#     m=st.integers(1, 25),
#     theta=hnp.arrays(
#         dtype=np.float64,
#         shape=st.integers(1, 100),
#         elements=st.floats(1e-5, np.pi, allow_subnormal=False, allow_nan=False),
#     ),
# )
# def test_real_Ylm_vec(l, m, theta):
#     """Test the ``real_Ylm`` function."""
#     assume(m <= l)
#     got = real_Ylm(l, m, theta)
#     expected = np.real(sp.sph_harm(m, l, 0, theta))
#     np.testing.assert_allclose(got, expected)
