"""Self-Consistent Field Potential."""

__all__ = ["SCFPotential", "STnlmSnapshotParameter"]

from collections.abc import Callable
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

import galax.typing as gt
from .bfe_helper import phi_nl_vec, rho_nl as calculate_rho_nl
from .coeffs import compute_coeffs_discrete
from .gegenbauer import GegenbauerCalculator
from .utils import cartesian_to_spherical, real_Ylm
from galax.potential._src.core import AbstractPotential
from galax.potential._src.param import AbstractParameter, ParameterField

##############################################################################


class SCFPotential(AbstractPotential):
    r"""Self-Consistent Field (SCF) potential.

    A gravitational potential represented as a basis function expansion.  This
    uses the self-consistent field (SCF) method of Hernquist & Ostriker (1992)
    and Lowing et al. (2011), and represents all coefficients as real
    quantities.

    Parameters
    ----------
    m : numeric
        Scale mass.
    r_s : numeric
        Scale length.
    Snlm : Array[float, (nmax+1, lmax+1, lmax+1)] | Callable
        Array of coefficients for the cos() terms of the expansion.  This should
        be a 3D array with shape `(nmax+1, lmax+1, lmax+1)`, where `nmax` is the
        number of radial expansion terms and `lmax` is the number of spherical
        harmonic `l` terms.  If a callable is provided, it should accept a
        single argument `t` and return the array of coefficients for that time.
    Tnlm : Array[float, (nmax+1, lmax+1, lmax+1)] | Callable
        Array of coefficients for the sin() terms of the expansion.  This should
        be a 3D array with shape `(nmax+1, lmax+1, lmax+1)`, where `nmax` is the
        number of radial expansion terms and `lmax` is the number of spherical
        harmonic `l` terms.  If a callable is provided, it should accept a
        single argument `t` and return the array of coefficients for that time.
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the length,
        mass, time, and angle units.
    """

    m: AbstractParameter = ParameterField(dimensions="mass")
    r_s: AbstractParameter = ParameterField(dimensions="length")
    Snlm: AbstractParameter = ParameterField(dimensions="dimensionless")
    Tnlm: AbstractParameter = ParameterField(dimensions="dimensionless")

    nmax: int = eqx.field(init=False, static=True, repr=False)
    lmax: int = eqx.field(init=False, static=True, repr=False)
    _ultra_sph: GegenbauerCalculator = eqx.field(init=False, static=True, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        # shape parameters
        shape = self.Snlm(0).shape
        object.__setattr__(self, "nmax", shape[0] - 1)
        object.__setattr__(self, "lmax", shape[1] - 1)

        # gegenbauer calculator
        object.__setattr__(self, "_ultra_sph", GegenbauerCalculator(self.nmax))

    # ==========================================================================

    @partial(jax.jit, inline=True)
    def _potential(
        self, xyz: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.VecN | gt.FloatScalar:
        r_s = self.r_s(t)
        r, theta, phi = cartesian_to_spherical(xyz).T

        s = jnp.atleast_1d(r / r_s)  # ([n],[l],[m],[N])
        theta = jnp.atleast_1d(theta)[None, None, None]  # ([n],[l],[m],[N])
        phi = jnp.atleast_1d(phi)[None, None, None]  # ([n],[l],[m],[N])

        ns = jnp.arange(self.nmax + 1)[:, None, None]  # (n, [l], [m])
        ls = jnp.arange(self.lmax + 1)[None, :, None]  # ([n], l, [m])
        phi_nl = phi_nl_vec(s, ns, ls, self._ultra_sph)  # (n, l, [m], N)

        li, mi = jnp.tril_indices(self.lmax + 1)  # (l*(l+1)//2,)
        shape = (1, self.lmax + 1, self.lmax + 1, 1)  # ([n], l, m, [N])
        midx = jnp.zeros(shape, dtype=int).at[:, li, mi, 0].set(mi)  # ([n], l, m, [N])

        Ylm = jnp.zeros(shape[:-1] + (len(s),))
        Ylm = Ylm.at[0, li, mi, :].set(
            real_Ylm(theta[:, 0, 0, :], li[..., None], mi[..., None])
        )

        Snlm = self.Snlm(t, r_s=r_s)[..., None]
        Tnlm = self.Tnlm(t, r_s=r_s)[..., None]

        out = (self._G * self.m(t) / r_s) * jnp.sum(
            Ylm * phi_nl * (Snlm * jnp.cos(midx * phi) + Tnlm * jnp.sin(midx * phi)),
            axis=(0, 1, 2),
        )
        return out[0] if len(xyz.shape) == 1 else out

    @partial(jax.jit, inline=True)
    @eqx.filter_vmap(in_axes=(None, 1, None))  # type: ignore[misc]  # on `q` axis 1
    def _density(self, q: gt.LengthVec3, /, t: gt.RealQScalar) -> Float[Array, "N"]:  # type: ignore[name-defined]
        """Compute the density at the given position(s)."""
        r, theta, phi = cartesian_to_spherical(q)
        r_s = self.r_s(t)
        s = jnp.atleast_1d(r / r_s)[:, None, None, None]
        theta = jnp.atleast_1d(theta)[:, None, None, None]
        phi = jnp.atleast_1d(phi)[:, None, None, None]

        ns = jnp.arange(self.nmax + 1)[:, None, None]  # (n, [l], [m])
        ls = jnp.arange(self.lmax + 1)[None, :, None]  # ([n], l, [m])

        phi_nl = calculate_rho_nl(s, ns[None], ls[None], gegenbauer=self._ultra_sph)

        li, mi = jnp.tril_indices(self.lmax + 1)  # (l*(l+1)//2,)
        shape = (1, 1, self.lmax + 1, self.lmax + 1)
        midx = jnp.zeros(shape, dtype=int).at[:, :, li, mi].set(mi)
        Ylm = jnp.zeros((len(theta), 1, self.lmax + 1, self.lmax + 1))
        Ylm = Ylm.at[:, li, mi, :].set(real_Ylm(li[None], mi[None], theta[:, :, 0, 0]))

        Snlm = self.Snlm(t, r_s=r_s)[None]
        Tnlm = self.Tnlm(t, r_s=r_s)[None]

        out = (self._G * self.m(t) / r_s) * jnp.sum(
            Ylm * phi_nl * (Snlm * jnp.cos(midx * phi) + Tnlm * jnp.sin(midx * phi)),
            axis=(1, 2, 3),
        )
        return out[0] if len(q.shape) == 1 else out


# =============================================================================


class STnlmSnapshotParameter(AbstractParameter):  # type: ignore[misc]
    """Parameter for the STnlm coefficients."""

    snapshot: Callable[  # type: ignore[name-defined]
        [Float[Array, "N"]],
        tuple[Float[Array, "3 N"], Float[Array, "N"]],
    ]
    """Cartesian coordinates of the snapshot.

    This should be a callable that accepts a single argument `t` and returns
    the cartesian coordinates and the masses of the snapshot at that time.
    """

    nmax: int = eqx.field(static=True, converter=int)
    """Radial expansion term."""

    lmax: int = eqx.field(static=True, converter=int)
    """Spherical harmonic term."""

    def __call__(
        self, t: gt.TimeScalar, *, r_s: gt.LengthScalar, **_: Any
    ) -> tuple[
        Float[Array, "{self.nmax}+1 {self.lmax}+1 {self.lmax}+1"],
        Float[Array, "{self.nmax}+1 {self.lmax}+1 {self.lmax}+1"],
    ]:
        """Return the coefficients at the given time(s).

        Parameters
        ----------
        t : float | Array[float, ()]
            Time at which to evaluate the coefficients.
        r_s : float | Array[float, ()]
            Scale length of the potential at the given time(s.
        **kwargs : Any
            Additional keyword arguments are ignored.

        Returns
        -------
        Snlm : Array[float, (nmax+1, lmax+1, lmax+1)]
            The value of the cosine expansion coefficient.
        Tnlm : Array[float, (nmax+1, lmax+1, lmax+1)]
            The value of the sine expansion coefficient.
        """
        xyz, m = self.snapshot(t)
        coeffs: tuple[Array, Array] = compute_coeffs_discrete(
            xyz, m, nmax=self.nmax, lmax=self.lmax, r_s=r_s
        )
        return coeffs
