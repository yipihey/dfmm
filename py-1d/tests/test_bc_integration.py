"""Integration tests for the ghost-cell BC variants.

Validates that run_to actually honours each BC type end-to-end:
- periodic: net momentum conserved
- transmissive: no reflected waves (wave passes through cleanly)
- reflective: velocity sign-flipped at the wall
- dirichlet: ghost cells follow the specified fixed state

Separate from the existing smoke tests because those exercised only
periodic + transmissive via the shipped scheme dispatch; the new
ghost-cell machinery expands the BC catalogue and the dispatcher.
"""
import numpy as np
import pytest

from dfmm.config import SimulationConfig
from dfmm.integrate import run_to
from dfmm.setups.wavepool import make_wave_pool_ic


def _uniform(N, u0=0.0, rho0=1.0, P0=0.1):
    x = np.linspace(0, 1, N, endpoint=False) + 0.5 / N
    rho = np.full(N, rho0); u = np.full(N, u0)
    Pxx = np.full(N, P0); Pp = np.full(N, P0)
    alpha0 = np.full(N, 0.02); beta0 = np.zeros(N)
    Q0 = np.zeros(N)
    M3 = rho*u**3 + 3*u*Pxx + Q0
    U = np.array([rho, rho*u, rho*u*u + Pxx, Pp,
                  rho*x, rho*alpha0, rho*beta0, M3])
    return U, x


def test_periodic_conserves_mass_and_momentum():
    """Standard periodic integration: mass and momentum conserved to
    machine precision (this is also in test_conservation, repeated
    here in the BC-specific context)."""
    U0, _ = make_wave_pool_ic(64, u0=1.0, P0=0.1, seed=42)
    M0 = U0[0].sum(); P0 = U0[1].sum()
    cfg = SimulationConfig(bc_left='periodic', bc_right='periodic')
    snaps, _ = run_to(U0, t_end=0.05, save_times=[0.05], cfg=cfg)
    U_end = snaps[-1][1]
    assert abs(U_end[0].sum() - M0) < 1e-10
    assert abs(U_end[1].sum() - P0) < 1e-10


def test_transmissive_stable_on_uniform_state():
    """Uniform flow under transmissive BCs stays uniform (no
    boundary-induced waves)."""
    U0, _ = _uniform(64, u0=0.0)  # static
    cfg = SimulationConfig(bc_left='transmissive', bc_right='transmissive')
    snaps, _ = run_to(U0, t_end=0.05, save_times=[0.05], cfg=cfg)
    U_end = snaps[-1][1]
    # All cells should still be near-uniform
    assert np.std(U_end[0]) < 1e-6
    assert np.all(np.isfinite(U_end))


def test_reflective_flips_momentum_of_incoming_wave():
    """Uniform flow moving into a reflective wall: after a short time,
    momentum near the wall reverses sign."""
    U0, _ = _uniform(32, u0=0.3, rho0=1.0, P0=0.1)
    cfg = SimulationConfig(bc_left='reflective', bc_right='reflective')
    # Run long enough for a wave to propagate from the wall
    snaps, _ = run_to(U0, t_end=0.05, save_times=[0.05], cfg=cfg)
    U_end = snaps[-1][1]
    # Near the right wall (last few cells), momentum has been flipped
    # by the wall. At least the cell adjacent to the right boundary
    # should show u < u0 (the reflected wave has slowed / reversed
    # the fluid there).
    u_end = U_end[1] / U_end[0]
    assert u_end[-1] < 0.3, \
        f"expected wall deceleration; got u[-1] = {u_end[-1]}"
    # The interior well away from walls still holds some positive
    # momentum (the reflected wave hasn't reached the middle yet).
    assert u_end[15] > 0, f"middle should be still moving: u[15]={u_end[15]}"


def test_muscl_runs_with_all_bc_types():
    """Basic smoke: every BC variant runs to completion under MUSCL
    without NaN/realizability breakdown."""
    U0, _ = _uniform(32, u0=0.1, P0=0.1)
    for bc_left, bc_right in [('periodic', 'periodic'),
                               ('transmissive', 'transmissive'),
                               ('reflective', 'reflective'),
                               ('transmissive', 'reflective')]:
        cfg = SimulationConfig(
            reconstruction='muscl', limiter='minmod',
            bc_left=bc_left, bc_right=bc_right)
        snaps, _ = run_to(U0.copy(), t_end=0.02,
                           save_times=[0.02], cfg=cfg)
        U_end = snaps[-1][1]
        assert np.all(np.isfinite(U_end)), (
            f"BC ({bc_left}, {bc_right}) produced NaN/Inf")
        # Realizability
        rho_end = U_end[0]
        assert np.all(rho_end > 0)
        u_end = U_end[1] / rho_end
        Pxx_end = U_end[2] - rho_end * u_end**2
        beta_end = U_end[6] / rho_end
        Svv_end = Pxx_end / np.maximum(rho_end, 1e-30)
        assert np.all(Svv_end - beta_end**2 >= -1e-10)


def test_dirichlet_ghosts_hold_fixed_state():
    """With Dirichlet BCs, the interior state near the boundary should
    be driven toward the specified inflow state over time."""
    U0, _ = _uniform(32, u0=0.0, rho0=1.0, P0=0.1)
    # Inflow state on the left with higher density
    rho_L, u_L, P_L = 2.0, 0.2, 0.2
    Pxx_L = P_L; Pp_L = P_L
    alpha_L = 0.02; beta_L = 0.0; Q_L = 0.0
    M3_L = rho_L * u_L**3 + 3 * u_L * Pxx_L + Q_L
    # Field order: (rho, rho u, rho u^2 + Pxx, Pp, rho L1, rho alpha, rho beta, M3)
    state_left = np.array([rho_L, rho_L*u_L, rho_L*u_L*u_L + Pxx_L, Pp_L,
                           rho_L * 0.0, rho_L*alpha_L, rho_L*beta_L, M3_L])
    state_right = U0[:, -1].copy()  # mirror of the last interior cell
    cfg = SimulationConfig(
        bc_left='dirichlet', bc_right='dirichlet',
        bc_state_left=state_left, bc_state_right=state_right)
    snaps, _ = run_to(U0, t_end=0.02, save_times=[0.02], cfg=cfg)
    U_end = snaps[-1][1]
    assert np.all(np.isfinite(U_end))
    # Rho near the left boundary has risen from 1.0 toward 2.0
    assert U_end[0, 0] > 1.01, (
        f"Dirichlet inflow didn't push rho up: rho[0]={U_end[0, 0]}")
