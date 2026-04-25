"""Unit tests for `dfmm.schemes.cholesky.primitives`.

The primitives() helper inverts the 8-field conserved state into
readable field values. Bugs here would silently corrupt every
diagnostic.
"""
import numpy as np
import pytest

from dfmm.schemes.cholesky import primitives
from dfmm.schemes._common import (IDX_RHO, IDX_MOM, IDX_EXX, IDX_PP,
                                  IDX_L1, IDX_ALPHA, IDX_BETA, IDX_M3)


def _to_conserved(rho, u, Pxx, Pp, L1, alpha, beta, Q):
    N = len(rho) if hasattr(rho, '__len__') else 1
    U = np.zeros((8, N))
    U[IDX_RHO]   = rho
    U[IDX_MOM]   = rho * u
    U[IDX_EXX]   = rho * u * u + Pxx
    U[IDX_PP]    = Pp
    U[IDX_L1]    = rho * L1
    U[IDX_ALPHA] = rho * alpha
    U[IDX_BETA]  = rho * beta
    U[IDX_M3]    = rho * u**3 + 3 * u * Pxx + Q
    return U


def test_primitives_roundtrip_uniform_state():
    """Primitive -> conserved -> primitive should recover the inputs
    bit-identically for a uniform state."""
    N = 16
    rho = np.full(N, 1.5)
    u = np.full(N, 0.3)
    Pxx = np.full(N, 0.1); Pp = np.full(N, 0.12)
    L1 = np.linspace(0, 1, N)
    alpha = np.full(N, 0.02); beta = np.full(N, 0.01)
    Q = np.full(N, 0.0)

    U = _to_conserved(rho, u, Pxx, Pp, L1, alpha, beta, Q)
    (rho_r, u_r, Pxx_r, Pp_r, L1_r, alpha_r, beta_r,
     gamma_r, Sxx_r, Sxv_r, Svv_r, Q_r) = primitives(U)

    assert np.allclose(rho_r, rho)
    assert np.allclose(u_r, u)
    assert np.allclose(Pxx_r, Pxx)
    assert np.allclose(Pp_r, Pp)
    assert np.allclose(L1_r, L1)
    assert np.allclose(alpha_r, alpha)
    assert np.allclose(beta_r, beta)
    assert np.allclose(Q_r, Q)


def test_primitives_gamma_nonnegative():
    """gamma^2 = max(Sigma_vv - beta^2, 0) guarantees gamma >= 0 even when
    the realizability headroom is violated at input."""
    N = 4
    rho = np.full(N, 1.0)
    u = np.full(N, 0.0)
    Pxx = np.full(N, 0.01)  # Svv = 0.01, sqrt(Svv) = 0.1
    Pp = np.full(N, 0.01)
    L1 = np.zeros(N); alpha = np.full(N, 0.02)
    # Inject a beta > sqrt(Svv) to violate realizability
    beta = np.array([0.05, 0.1, 0.2, 0.5])
    Q = np.zeros(N)

    U = _to_conserved(rho, u, Pxx, Pp, L1, alpha, beta, Q)
    _, _, _, _, _, _, _, gamma_r, *_ = primitives(U)
    assert (gamma_r >= 0.0).all(), \
        f"gamma should be non-negative; got {gamma_r}"


def test_primitives_sigma_consistency():
    """Sigma_xx = alpha^2, Sigma_xv = alpha*beta per the Cholesky factorisation."""
    N = 8
    rho = np.full(N, 1.0); u = np.zeros(N)
    Pxx = np.full(N, 0.1); Pp = np.full(N, 0.1)
    L1 = np.zeros(N)
    alpha = np.linspace(0.01, 0.1, N)
    beta = np.linspace(-0.05, 0.05, N)
    Q = np.zeros(N)

    U = _to_conserved(rho, u, Pxx, Pp, L1, alpha, beta, Q)
    (_, _, _, _, _, _, _, _, Sxx_r, Sxv_r, _, _) = primitives(U)
    assert np.allclose(Sxx_r, alpha**2)
    assert np.allclose(Sxv_r, alpha * beta)


def test_primitives_heat_flux_recovery():
    """Q = M3 - rho*u^3 - 3*u*Pxx; given known Q, recovery should match."""
    N = 4
    rho = np.full(N, 1.2); u = np.full(N, 0.4)
    Pxx = np.full(N, 0.15); Pp = np.full(N, 0.1)
    L1 = np.zeros(N); alpha = np.full(N, 0.02); beta = np.full(N, 0.0)
    Q = np.array([0.0, 0.1, -0.05, 0.5])

    U = _to_conserved(rho, u, Pxx, Pp, L1, alpha, beta, Q)
    (_, _, _, _, _, _, _, _, _, _, _, Q_r) = primitives(U)
    assert np.allclose(Q_r, Q)
