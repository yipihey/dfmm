"""Spatial convergence-order tests for the cholesky hll_step variants.

Setup: a uniform velocity u0 carries a low-amplitude sinusoidal rho
perturbation around a periodic domain for a short time. Since u is
uniform and tau is huge, the Eulerian state's rho field is purely
advected: the exact solution at time t is the initial sinusoid
shifted by u0*t.

We run the scheme at several resolutions N and compute the L2 rho
error against the exact solution. The pairwise error ratio
  ratio(N, 2N) = L2(N) / L2(2N)
approaches 2^p for a p-th-order method.

First-order HLL:         expected ratio ≈ 2   (p = 1)
MUSCL-Hancock + minmod:  expected ratio > 3   (p > 1.5 at resolved N)

The MUSCL bar is loose because at coarse N the method hasn't entered
its asymptotic regime yet. Empirically the ratio climbs from ~3.1 at
N=32/64 to ~3.6 at N=128/256, consistent with p → 2 as dx → 0.
"""
import numpy as np
import pytest

from dfmm.config import SimulationConfig
from dfmm.integrate import run_to


U0_ADVECT = 0.2
AMP = 0.1
T_END = 0.2


def _setup(N, u0=U0_ADVECT, P0=0.01, amp=AMP, sigma_x0=0.02):
    """Uniform-u periodic IC with sinusoidal rho; tau is set huge in
    the run so BGK is a no-op and the scheme is a pure advection test."""
    x = np.linspace(0, 1, N, endpoint=False) + 0.5 / N
    rho = 1.0 + amp * np.sin(2 * np.pi * x)
    u = np.full(N, u0)
    Pxx = np.full(N, P0); Pp = np.full(N, P0)
    alpha0 = np.full(N, sigma_x0); beta0 = np.zeros(N); Q0 = np.zeros(N)
    M3 = rho * u ** 3 + 3 * u * Pxx + Q0
    U = np.array([rho, rho * u, rho * u * u + Pxx, Pp,
                  rho * x, rho * alpha0, rho * beta0, M3])
    return U, x


def _exact(x, u0=U0_ADVECT, t=T_END, amp=AMP):
    return 1.0 + amp * np.sin(2 * np.pi * (x - u0 * t))


def _l2_rho(recon, N, t=T_END):
    cfg = SimulationConfig(reconstruction=recon, tau=1e6)
    U, x = _setup(N)
    snaps, _ = run_to(U, t_end=t, save_times=[t], cfg=cfg)
    rho_sim = snaps[-1][1][0]
    rho_exact = _exact(x, t=t)
    return float(np.sqrt(np.mean((rho_sim - rho_exact) ** 2)))


def test_first_order_slope_near_one():
    """First-order HLL converges at p ≈ 1 (L2 error halves per 2x resolution)."""
    errs = [_l2_rho('first', N) for N in (32, 64, 128)]
    for a, b in zip(errs[:-1], errs[1:]):
        ratio = a / b
        assert 1.7 < ratio < 2.3, \
            f"first-order ratio {ratio:.3f} (expected ~2, i.e. slope ~1)"


def test_muscl_slope_clearly_above_one():
    """MUSCL-Hancock with minmod is distinctly better than first-order.

    We require absolute error at least 2x smaller than first-order at
    matched N, and the ratio between successive resolutions > 3
    (slope > log2(3) ≈ 1.58) — a conservative bar that still
    excludes any regression to first-order behaviour."""
    errs_muscl = [_l2_rho('muscl', N) for N in (64, 128, 256)]
    errs_first = [_l2_rho('first', N) for N in (64, 128, 256)]

    # MUSCL lower error at each resolution
    for e_m, e_f in zip(errs_muscl, errs_first):
        assert e_m < 0.5 * e_f, \
            f"MUSCL error {e_m:.2e} not << first-order {e_f:.2e}"

    # Convergence ratio > 3 per doubling (slope > 1.58)
    for a, b in zip(errs_muscl[:-1], errs_muscl[1:]):
        ratio = a / b
        assert ratio > 3.0, \
            f"MUSCL ratio {ratio:.3f} below 3 (slope < 1.58)"
