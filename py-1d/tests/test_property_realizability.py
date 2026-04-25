"""Property-based tests for realizability preservation under the
cholesky scheme.

Uses `hypothesis` to generate valid physical initial states and checks
that one integrator step keeps them realizable (rho > 0,
Sigma_vv - beta^2 >= 0 with the configured headroom, no NaN/Inf).
This guards the refactors against regressions that are scenario-
dependent rather than caught by the fixed wave-pool/Sod cases.
"""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from dfmm.config import SimulationConfig
from dfmm.integrate import run_to


def _make_uniform_state(N, rho0, u0, Pxx0, Pp0, alpha0, beta0, L1_offset=0.0):
    """Build a uniform 8-field state (smoke-friendly; no gradients)."""
    x = np.linspace(0, 1, N, endpoint=False) + 0.5 / N
    rho = np.full(N, rho0)
    u   = np.full(N, u0)
    Pxx = np.full(N, Pxx0); Pp = np.full(N, Pp0)
    alpha = np.full(N, alpha0); beta = np.full(N, beta0)
    Q = np.zeros(N)
    M3 = rho * u**3 + 3 * u * Pxx + Q
    U = np.array([rho, rho*u, rho*u*u + Pxx, Pp,
                  rho*(x + L1_offset), rho*alpha, rho*beta, M3])
    return U


@settings(max_examples=50, deadline=None)
@given(
    rho0=st.floats(min_value=0.1, max_value=5.0),
    u0=st.floats(min_value=-0.5, max_value=0.5),
    P0=st.floats(min_value=0.01, max_value=1.0),
    # Draw beta as a fraction of sqrt(Sigma_vv) within the physical range
    beta_frac=st.floats(min_value=-0.85, max_value=0.85),
)
def test_uniform_state_stays_realizable_first_order(rho0, u0, P0, beta_frac):
    """One first-order step on any uniform valid state preserves realizability."""
    Svv = P0 / rho0
    beta0 = beta_frac * np.sqrt(Svv)
    U = _make_uniform_state(N=16, rho0=rho0, u0=u0,
                            Pxx0=P0, Pp0=P0,
                            alpha0=0.02, beta0=beta0)
    cfg = SimulationConfig(reconstruction='first')
    snaps, _ = run_to(U, t_end=1e-3, save_times=[1e-3], cfg=cfg)
    U_end = snaps[-1][1]

    assert np.all(np.isfinite(U_end)), "scheme produced NaN/Inf"
    rho_end = U_end[0]
    assert np.all(rho_end > 0), f"rho went non-positive: min={rho_end.min()}"

    u_end = U_end[1] / rho_end
    Pxx_end = U_end[2] - rho_end * u_end**2
    beta_end = U_end[6] / rho_end
    Svv_end = Pxx_end / np.maximum(rho_end, 1e-30)
    # Realizability (strict up to O(1e-10) float slack)
    assert np.all(Svv_end - beta_end**2 >= -1e-10), (
        f"realizability violated: "
        f"min (Svv - beta^2) = {(Svv_end - beta_end**2).min():.3e}")


@settings(max_examples=30, deadline=None)
@given(
    rho0=st.floats(min_value=0.2, max_value=3.0),
    u0=st.floats(min_value=-0.3, max_value=0.3),
    P0=st.floats(min_value=0.05, max_value=0.5),
    beta_frac=st.floats(min_value=-0.6, max_value=0.6),
)
def test_uniform_state_stays_realizable_muscl(rho0, u0, P0, beta_frac):
    """Same invariant under MUSCL-Hancock reconstruction."""
    Svv = P0 / rho0
    beta0 = beta_frac * np.sqrt(Svv)
    U = _make_uniform_state(N=16, rho0=rho0, u0=u0,
                            Pxx0=P0, Pp0=P0,
                            alpha0=0.02, beta0=beta0)
    cfg = SimulationConfig(reconstruction='muscl', limiter='minmod')
    snaps, _ = run_to(U, t_end=1e-3, save_times=[1e-3], cfg=cfg)
    U_end = snaps[-1][1]

    assert np.all(np.isfinite(U_end))
    rho_end = U_end[0]
    assert np.all(rho_end > 0)

    u_end = U_end[1] / rho_end
    Pxx_end = U_end[2] - rho_end * u_end**2
    beta_end = U_end[6] / rho_end
    Svv_end = Pxx_end / np.maximum(rho_end, 1e-30)
    assert np.all(Svv_end - beta_end**2 >= -1e-10)


@settings(max_examples=30, deadline=None)
@given(
    rho_amp=st.floats(min_value=0.0, max_value=0.3),
    u_amp=st.floats(min_value=-0.3, max_value=0.3),
    P0=st.floats(min_value=0.05, max_value=0.5),
)
def test_sinusoidal_perturbation_stays_realizable(rho_amp, u_amp, P0):
    """Gradients present: sinusoidal rho and u perturbations around
    a uniform background. One first-order step should not violate
    realizability."""
    N = 32
    x = np.linspace(0, 1, N, endpoint=False) + 0.5 / N
    rho = 1.0 + rho_amp * np.sin(2 * np.pi * x)
    u   = u_amp * np.cos(2 * np.pi * x)
    Pxx = np.full(N, P0); Pp = np.full(N, P0)
    alpha = np.full(N, 0.02); beta = np.zeros(N)
    Q = np.zeros(N)
    M3 = rho * u**3 + 3 * u * Pxx + Q
    U = np.array([rho, rho*u, rho*u*u + Pxx, Pp,
                  rho*x, rho*alpha, rho*beta, M3])
    cfg = SimulationConfig(reconstruction='first')
    snaps, _ = run_to(U, t_end=1e-3, save_times=[1e-3], cfg=cfg)
    U_end = snaps[-1][1]

    assert np.all(np.isfinite(U_end))
    rho_end = U_end[0]
    assert np.all(rho_end > 0)
    u_end = U_end[1] / rho_end
    Pxx_end = U_end[2] - rho_end * u_end**2
    beta_end = U_end[6] / rho_end
    Svv_end = Pxx_end / np.maximum(rho_end, 1e-30)
    assert np.all(Svv_end - beta_end**2 >= -1e-10)
