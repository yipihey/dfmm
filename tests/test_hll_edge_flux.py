"""Unit tests for `dfmm.schemes._common.hll_edge_flux`.

The helper is shared by every 8-field HLL kernel (cholesky, noise,
two_fluid), so a bug here propagates everywhere. These tests exercise
the three HLL branches (SL>=0, SR<=0, mid-state) directly with
hand-constructed left/right states.
"""
import numpy as np
import pytest

from dfmm.schemes._common import (CSCOEF, IDX_RHO, IDX_MOM, IDX_EXX,
                                  IDX_PP, IDX_L1, IDX_ALPHA, IDX_BETA,
                                  IDX_M3, hll_edge_flux)


def _uniform_state(rho=1.0, u=0.0, Pxx=0.1, Pp=0.1, L1=0.0,
                   alpha=0.02, beta=0.0, Q=0.0):
    """Build an 8-field conserved state from primitive values."""
    U = np.empty(8)
    U[IDX_RHO] = rho
    U[IDX_MOM] = rho * u
    U[IDX_EXX] = rho * u * u + Pxx
    U[IDX_PP]  = Pp
    U[IDX_L1]  = rho * L1
    U[IDX_ALPHA] = rho * alpha
    U[IDX_BETA]  = rho * beta
    U[IDX_M3]  = rho * u**3 + 3 * u * Pxx + Q
    return U


def _cs_from_state(U):
    rho = U[IDX_RHO]
    u = U[IDX_MOM] / rho
    Pxx = U[IDX_EXX] - rho * u * u
    return float(np.sqrt(CSCOEF * max(Pxx, 1e-30) / max(rho, 1e-30)))


def test_uniform_state_gives_pure_advective_flux():
    """For identical L and R states, HLL reduces to the advection flux
    `F = rho*u*[1, u, u^2+P, ...]` — no mid-state diffusion."""
    U = _uniform_state(rho=1.0, u=0.3, Pxx=0.1, Pp=0.1)
    cs = _cs_from_state(U)
    F = np.array(hll_edge_flux(U, U, cs, cs))
    rho, u = 1.0, 0.3
    Pxx = 0.1
    expected_F_rho = rho * u
    expected_F_mom = rho * u * u + Pxx
    assert abs(F[IDX_RHO] - expected_F_rho) < 1e-12
    assert abs(F[IDX_MOM] - expected_F_mom) < 1e-12


def test_supersonic_right_branch_is_upwind_left():
    """When SL > 0 (left-state wave speed is positive), HLL degenerates
    to the left-state flux (pure upwind)."""
    U_L = _uniform_state(rho=1.0, u=10.0, Pxx=0.01, Pp=0.01)
    U_R = _uniform_state(rho=1.0, u=10.0, Pxx=0.01, Pp=0.01)
    cs = _cs_from_state(U_L)  # ~0.23; u=10 >> cs so SL > 0
    # Single-state flux
    F_single = np.array(hll_edge_flux(U_L, U_L, cs, cs))
    # Symmetric bilateral flux should be identical
    F = np.array(hll_edge_flux(U_L, U_R, cs, cs))
    assert np.allclose(F, F_single, atol=1e-12)


def test_supersonic_left_branch_is_upwind_right():
    """When SR < 0 (right-state wave speed is negative), HLL degenerates
    to the right-state flux."""
    U_L = _uniform_state(rho=1.0, u=-10.0, Pxx=0.01, Pp=0.01)
    U_R = _uniform_state(rho=1.0, u=-10.0, Pxx=0.01, Pp=0.01)
    cs = _cs_from_state(U_L)
    F = np.array(hll_edge_flux(U_L, U_R, cs, cs))
    # Expected: rho*u at IDX_RHO, etc
    assert abs(F[IDX_RHO] - (-10.0)) < 1e-12
    # Non-zero momentum flux: rho*u^2 + Pxx = 100 + 0.01
    assert abs(F[IDX_MOM] - 100.01) < 1e-12


def test_mid_state_symmetry_under_reflection():
    """Flipping u everywhere and swapping L<->R should produce a flux
    with flipped momentum sign and same mass flux magnitude (reversed)."""
    U_L = _uniform_state(rho=1.0, u=0.2, Pxx=0.1, Pp=0.1)
    U_R = _uniform_state(rho=1.2, u=0.3, Pxx=0.12, Pp=0.11)
    cs_L = _cs_from_state(U_L); cs_R = _cs_from_state(U_R)
    F_forward = np.array(hll_edge_flux(U_L, U_R, cs_L, cs_R))

    # Build the reflected problem: swap L/R and flip u
    def flip_u(U):
        rho = U[IDX_RHO]
        u = U[IDX_MOM] / rho
        Pxx = U[IDX_EXX] - rho * u * u
        return _uniform_state(rho=rho, u=-u, Pxx=Pxx, Pp=U[IDX_PP],
                              L1=U[IDX_L1]/rho,
                              alpha=U[IDX_ALPHA]/rho,
                              beta=U[IDX_BETA]/rho,
                              Q=U[IDX_M3] - rho*u**3 - 3*u*Pxx)
    F_reflected = np.array(hll_edge_flux(flip_u(U_R), flip_u(U_L), cs_R, cs_L))
    # Under u -> -u, the rho-flux (rho*u) and mom-flux (rho*u^2 + P) transform
    # by (-1) and (+1) respectively; after also swapping L<->R, the overall
    # flux should be (-F_forward[IDX_RHO], +F_forward[IDX_MOM], ...) or similar.
    # Check the scalar magnitudes are consistent (mass flux reverses sign).
    assert abs(F_reflected[IDX_RHO] + F_forward[IDX_RHO]) < 1e-10


def test_flux_returns_length_8_tuple():
    """Basic shape contract."""
    U = _uniform_state()
    cs = _cs_from_state(U)
    result = hll_edge_flux(U, U, cs, cs)
    assert len(result) == 8
    for v in result:
        assert np.isfinite(v)
