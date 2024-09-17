"""Test the Gegenbauer class."""

import jax
import numpy as np
import pytest

import quaxed.numpy as jnp

from galax.potential._src.scf.bfe_helper import rho_nl
from galax.potential._src.scf.coeffs_helper import (
    expansion_coeffs_Anl_discrete,
    normalization_Knl,
)
from galax.potential._src.scf.gegenbauer import GegenbauerCalculator


def test_normalization_Knl():
    """Test the ``normalization_Knl`` function.

    .. todo::

        This test is not very good. It should be improved.
    """
    assert normalization_Knl(0, 0) == 1
    assert normalization_Knl(1, 0) == 3
    assert normalization_Knl(2, 0) == 6
    assert normalization_Knl(0, 1) == 6
    assert normalization_Knl(0, 2) == 15
    assert normalization_Knl(1, 1) == 10


# =============================================================================


def test_expansion_coeffs_Anl_discrete():
    """Test the ``expansion_coeffs_Anl_discrete`` function.

    .. todo::

        This test is not very good. It should be improved.
    """
    np.testing.assert_allclose(expansion_coeffs_Anl_discrete(0, 0), -3)
    np.testing.assert_allclose(
        expansion_coeffs_Anl_discrete(1, 0), -0.555555, rtol=1e-5
    )


# =============================================================================


@jax.jit
def compare_rho_nl(s, n, l):
    """Compare the ``rho_nl`` function."""
    gc = GegenbauerCalculator(10)

    mock = rho_nl(s, n, l, gegenbauer=gc)
    observed = jax.lax.stop_gradient(mock)

    return -jnp.sum((observed - mock) ** 2)


@pytest.mark.skip(reason="TODO")
def test_rho_nl():
    """Test the ``rho_nl`` function."""
    s = jnp.linspace(0, 4, 100, dtype=float)
    n = jnp.array([1.0])
    l = jnp.array([2.0])

    first_deriv = jax.jacfwd(compare_rho_nl)(s, n, l)
    assert first_deriv == 0


@pytest.mark.skip(reason="TODO")
def test_phi_nl():
    """Test the ``phi_nl`` function."""
    raise NotImplementedError
