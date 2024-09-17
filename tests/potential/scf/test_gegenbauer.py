"""Test the Gegenbauer class."""

from types import LambdaType

import numpy as np
import pytest
from scipy.special import gegenbauer as gegenbauer_sp

import quaxed.numpy as jnp

from galax.potential._src.scf.gegenbauer import (
    GegenbauerCalculator,
    _compute_weight_function_derivatives,
)


class TestGegenbauerCalculator:
    """Test the GegenbauerCalculator class."""

    def test_compute_weight_function_derivatives(self):
        """Test the ``_compute_weight_function_derivatives`` function.

        ..todo::

            This test is not very good, since it doesn't actually check the
            correctness of the output. It just checks that the function runs.

        """
        nmax = 5
        terms = _compute_weight_function_derivatives(nmax)
        assert len(terms) == nmax + 1

        x = jnp.linspace(0.02, 0.99, 10000)
        alpha = 2
        got = terms[0](x, alpha)
        assert got.shape == x.shape

    @pytest.mark.parametrize("nmax", [0, 2, 6])
    def test_init(self, nmax):
        """Test initializing the GegenbauerCalculator."""
        gc = GegenbauerCalculator(nmax)
        assert gc.nmax == nmax

        # Check the pre-computed weights
        assert isinstance(gc._weights, tuple)
        assert len(gc._weights) == nmax + 1
        assert all(isinstance(w, LambdaType) for w in gc._weights)

    # @pytest.mark.parametrize("nmax", [0, 1, 2, 3, 4, 5, 6])
    def test_call(self):
        """Test the functor."""
        # Setup
        nmax = 5
        x = jnp.linspace(0.02, 0.99, 10000)
        gc = GegenbauerCalculator(nmax)

        # # n > nmax should raise an error
        # with pytest.raises(ValueError):
        #     gc(6, 0, x)

        # With ints and floats
        got = gc(1, 0, x)
        assert got == pytest.approx(gegenbauer_sp(n=1, alpha=0)(x), rel=1e-8)

    def test_call_vectorized(self):
        """Test the functor with vectorized arguments."""
        # Setup
        nmax = 5
        x = jnp.linspace(0.02, 0.99, 10000)
        gc = GegenbauerCalculator(nmax)

        n = jnp.array([3, 5])
        alpha = 1.5
        got = gc(n, alpha, x[:, None])
        expected = np.c_[
            gegenbauer_sp(n=3, alpha=alpha)(np.array(x)),
            gegenbauer_sp(n=5, alpha=alpha)(np.array(x)),
        ]
        assert got.shape == expected.shape
        assert got == pytest.approx(expected, rel=1e-8)

    @pytest.mark.skip(reason="TODO with `hypothesis`")
    def test_validity(self):
        """Test the regions of validity of the Gegenbauer calculator."""
        raise NotImplementedError
