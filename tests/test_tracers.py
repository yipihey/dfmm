"""Smoke tests for the Lagrangian tracer system.

Verify initial construction, the identity-update round-trip (update on
t=0 state leaves tracers at their initial positions), tracking through
a short simulation, field sampling, and adaptive refinement.
"""
import numpy as np
import pytest

from dfmm import Tracers
from dfmm.integrate import run_to
from dfmm.schemes._common import IDX_RHO, IDX_L1


def _uniform_periodic_ic(N=64, u0=0.25, rho0=1.0, P0=0.1, sigma_x0=0.02):
    """Uniform rho, uniform u, periodic — cleanest case for tracer checks."""
    x = np.linspace(0, 1, N, endpoint=False) + 0.5 / N
    rho = np.full(N, rho0)
    u = np.full(N, u0)
    Pxx = np.full(N, P0); Pp = np.full(N, P0)
    alpha0 = np.full(N, sigma_x0); beta0 = np.zeros(N)
    Q0 = np.zeros(N)
    M3 = rho * u ** 3 + 3 * u * Pxx + Q0
    U = np.array([rho, rho * u, rho * u * u + Pxx, Pp,
                  rho * x, rho * alpha0, rho * beta0, M3])
    return U, x


def test_init_from_conditions():
    """At t=0 the tracer layout matches the stated conventions."""
    U0, x = _uniform_periodic_ic(N=64, rho0=1.0)
    tr = Tracers.from_initial_conditions(U0, x, periodic=True)

    # One tracer per cell, at the cell center.
    assert tr.n == 64
    assert np.max(np.abs(tr.x - x)) < 1e-15

    # q is monotone non-decreasing.
    assert np.all(np.diff(tr.q) >= -1e-15)

    # For uniform rho, q is uniformly spaced in mass = rho*dx per step.
    expected_dq = 1.0 / 64  # rho0 * dx with rho0=1, dx=1/64
    assert np.max(np.abs(np.diff(tr.q) - expected_dq)) < 1e-14

    # First cell's q is half its mass (mass coordinate at cell center).
    assert abs(tr.q[0] - 0.5 * expected_dq) < 1e-14

    # Labels equal the shipped-setup convention L1(x, 0) = x.
    assert np.max(np.abs(tr.label - tr.x)) < 1e-14


def test_init_transmissive_has_same_count():
    """Both periodic and transmissive produce N cell-center tracers."""
    U0, x = _uniform_periodic_ic(N=32)
    tr = Tracers.from_initial_conditions(U0, x, periodic=False)
    assert tr.n == 32


def test_update_at_t0_is_identity():
    """Updating against the initial U leaves tracer positions unchanged."""
    U0, x = _uniform_periodic_ic(N=128)
    tr = Tracers.from_initial_conditions(U0, x, periodic=True)
    x_before = tr.x.copy()
    tr.update(U0, x)
    assert np.max(np.abs(tr.x - x_before)) < 1e-12


def test_tracks_uniform_advection_per_step_transmissive():
    """When update() runs every timestep in a seam-free setup, tracers
    advect with the flow within O(dx).

    Uses transmissive BCs because the periodic scheme's HLL diffusion
    smears the seam discontinuity in the non-periodic L1 = x label
    field within a few steps (see module docstring). We check only
    the interior tracers; tracers that leave the domain through the
    outflow boundary are excluded.
    """
    from dfmm.schemes.cholesky import hll_step_transmissive, max_signal_speed

    u0 = 0.1
    t_end = 0.2
    U, x = _uniform_periodic_ic(N=128, u0=u0)  # geometry only; BC=transmissive
    dx = x[1] - x[0]
    tr = Tracers.from_initial_conditions(U, x, periodic=False)
    x_before = tr.x.copy()

    t = 0.0
    cfl = 0.3
    while t < t_end:
        smax = max_signal_speed(U)
        dt = min(cfl * dx / smax, t_end - t)
        U = hll_step_transmissive(U, dx, dt, tau=1e-3)
        tr.update(U, x)
        t += dt

    # Check interior tracers only (skip the outermost 5 cells on each side
    # where outflow / boundary effects dominate).
    expected = u0 * t_end
    interior = slice(5, tr.n - 5)
    disp = tr.x[interior] - x_before[interior]
    valid = np.isfinite(disp)
    assert valid.all(), "interior tracers should not be lost"
    assert abs(disp.mean() - expected) < 2 * dx, \
        f"mean interior displacement {disp.mean():.4f} vs expected {expected:.4f}"
    assert np.max(np.abs(disp - expected)) < 10 * dx


def test_sample_reproduces_uniform_field():
    """sample() on a spatially-uniform field returns that constant."""
    U0, x = _uniform_periodic_ic(N=64, rho0=2.5)
    tr = Tracers.from_initial_conditions(U0, x, periodic=True)
    tr.update(U0, x)
    rho_at_tracers = tr.sample(U0[IDX_RHO])
    assert np.max(np.abs(rho_at_tracers - 2.5)) < 1e-12


def test_refine_bisects_large_gaps():
    """refine() inserts at q-midpoints wherever Δx > refine_ratio * dx."""
    U0, x = _uniform_periodic_ic(N=16)
    dx = x[1] - x[0]  # = 1/16

    # Construct a synthetic Tracers with one big gap at the start.
    q = np.array([0.0, 0.2, 0.25, 0.95])
    x_pos = np.array([0.0, 0.5, 0.6, 0.95])  # gap[0] = 0.5 > 1.5*dx ≈ 0.094
    label = x_pos.copy()
    tr = Tracers(q=q, label=label, x=x_pos,
                 domain=(0.0, 1.0), periodic=False)

    n_before = tr.n
    n_new = tr.refine(x, refine_ratio=1.5)
    assert n_new >= 1
    assert tr.n == n_before + n_new

    # Verify q, x, label arrays stay sorted after insertion.
    assert np.all(np.diff(tr.q) >= -1e-15)
    assert np.all(np.diff(tr.x) >= -1e-15)

    # Inserted tracer after gap[0] should be the q-midpoint of (q[0], q[1]).
    assert abs(tr.q[1] - 0.5 * (q[0] + q[1])) < 1e-14


def test_refine_grows_capacity():
    """Refinement over an initially-tight capacity reallocates storage."""
    U0, x = _uniform_periodic_ic(N=8)
    dx = x[1] - x[0]
    q = np.array([0.0, 0.4, 0.9])
    x_pos = np.array([0.0, 0.4, 0.9])  # both gaps are > 1.5*dx = 0.1875
    tr = Tracers(q=q, label=x_pos.copy(), x=x_pos,
                 domain=(0.0, 1.0), periodic=False, capacity=3)
    assert tr._capacity == 3
    n_new = tr.refine(x, refine_ratio=1.5)
    assert n_new >= 1
    assert tr._capacity >= tr.n
