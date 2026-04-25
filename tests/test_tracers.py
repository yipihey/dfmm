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
from dfmm.setups.wavepool import make_wave_pool_ic


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

    # Periodic default drops boundary_buffer=2 cells on each end.
    assert tr.n == 60
    # Tracers live on cells 2..61; x positions match those cell centers.
    assert np.max(np.abs(tr.x - x[2:-2])) < 1e-15

    # q is monotone non-decreasing.
    assert np.all(np.diff(tr.q) >= -1e-15)

    # For uniform rho, q is uniformly spaced in mass = rho*dx per step.
    expected_dq = 1.0 / 64
    assert np.max(np.abs(np.diff(tr.q) - expected_dq)) < 1e-14

    # First retained tracer is on cell index 2; its q is the cumulative
    # mass up to cell 2's center = 2*dq + 0.5*dq = 2.5 * expected_dq.
    assert abs(tr.q[0] - 2.5 * expected_dq) < 1e-14

    # Labels equal the shipped-setup convention L1(x, 0) = x.
    assert np.max(np.abs(tr.label - tr.x)) < 1e-14


def test_init_transmissive_keeps_all_cells():
    """Transmissive keeps the full N tracers (no boundary buffer by default)."""
    U0, x = _uniform_periodic_ic(N=32)
    tr = Tracers.from_initial_conditions(U0, x, periodic=False)
    assert tr.n == 32


def test_init_boundary_buffer_zero_periodic():
    """Periodic with boundary_buffer=0 reproduces the full N tracers."""
    U0, x = _uniform_periodic_ic(N=32)
    tr = Tracers.from_initial_conditions(U0, x, periodic=True,
                                         boundary_buffer=0)
    assert tr.n == 32


def test_update_at_t0_is_identity():
    """Updating against the initial U leaves tracer positions unchanged."""
    U0, x = _uniform_periodic_ic(N=128)
    tr = Tracers.from_initial_conditions(U0, x, periodic=True)
    assert tr.n == 124  # default buffer = 2 on each end
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


def test_wavepool_periodic_tracking():
    """Tracers survive a full wave-pool run with a cs-scaled seam buffer.

    Seam smearing depth scales roughly as `cs * t / dx`, where cs is
    the scheme's max signal speed. For the wave-pool with P0=0.1 the
    sound speed is cs ≈ sqrt((3+√6)*0.1) ≈ 0.74; at N=128 and
    t_end=0.1 this works out to ~9 cells, so we pass
    `boundary_buffer=12` to cover the damaged zone with margin. The
    default-value of 2 is appropriate for shorter or lower-cs runs.

    The wave-pool IC has zero mean momentum by construction, so the
    seam stays stationary and the buffer brackets it permanently.
    """
    from dfmm.schemes.cholesky import hll_step_periodic, max_signal_speed

    N = 128
    t_end = 0.1
    cfl = 0.3
    buf = 12

    U, x = make_wave_pool_ic(N, u0=1.0, P0=0.1, seed=42)
    dx = x[1] - x[0]
    tr = Tracers.from_initial_conditions(U, x, periodic=True,
                                         boundary_buffer=buf)
    assert tr.n == N - 2 * buf
    x_before = tr.x.copy()

    t = 0.0
    while t < t_end:
        smax = max_signal_speed(U)
        dt = min(cfl * dx / smax, t_end - t)
        U = hll_step_periodic(U, dx, dt, tau=1e-3)
        tr.update(U, x)
        t += dt

    # (a) No tracer is lost (neither NaN'd nor drifted into the
    # buffered damaged zone).
    assert np.all(tr.valid), \
        f"{tr.lost.sum()} tracers became lost (NaN or in damaged zone)"

    # (b) Consecutive-tracer gaps stay bounded — no crossings /
    # runaway. The wave-pool IC is smooth so pairs shouldn't bunch
    # or separate by more than a few dx.
    gaps = np.diff(tr.x)
    assert gaps.min() > -0.5, "tracers have wrapped-crossed (negative gap)"
    assert gaps.max() < 5 * dx, \
        f"largest tracer gap = {gaps.max():.4f} exceeds 5*dx = {5*dx:.4f}"

    # (c) Mean displacement is small — wave-pool has zero net momentum.
    disp = tr.x - x_before
    disp -= np.round(disp)  # wrap any periodic roll
    assert abs(disp.mean()) < 2 * dx, \
        f"mean displacement {disp.mean():.4f} exceeds 2*dx = {2*dx:.4f}"


def test_step_stats_before_first_update():
    """step_stats() returns None until update() is called."""
    U0, x = _uniform_periodic_ic(N=32)
    tr = Tracers.from_initial_conditions(U0, x, periodic=True)
    assert tr.step_stats() is None


def test_step_stats_at_t0_identity():
    """First update() against the initial U leaves Δx ≈ 0 everywhere."""
    U0, x = _uniform_periodic_ic(N=128)
    tr = Tracers.from_initial_conditions(U0, x, periodic=True)
    tr.update(U0, x)
    stats = tr.step_stats()
    assert stats['n_valid'] == tr.n
    assert abs(stats['mean']) < 1e-12
    assert abs(stats['median']) < 1e-12
    assert stats['max_abs'] < 1e-12


def test_step_stats_tracks_uniform_advection():
    """Under a uniform-u transmissive flow, Δx per step ≈ u*dt with
    near-zero spread; max_abs stays well under max_scan * dx."""
    from dfmm.schemes.cholesky import hll_step_transmissive, max_signal_speed

    u0 = 0.1
    U, x = _uniform_periodic_ic(N=128, u0=u0)
    dx = x[1] - x[0]
    tr = Tracers.from_initial_conditions(U, x, periodic=False)

    cfl = 0.3
    smax = max_signal_speed(U)
    dt = cfl * dx / smax
    U = hll_step_transmissive(U, dx, dt, tau=1e-3)
    tr.update(U, x)

    stats = tr.step_stats()
    expected = u0 * dt
    # Interior tracers track uniform advection; allow O(dx) error
    # per step due to HLL diffusion of the L1 field.
    assert abs(stats['median'] - expected) < 0.5 * dx
    assert stats['max_abs'] < 5 * dx  # within max_scan's reach


def test_step_stats_wavepool_reasonable_magnitude():
    """Sanity: per-step Δx in a wave-pool run is bounded by a few
    CFL*dx, consistent with the scheme's CFL-limited dt."""
    from dfmm.schemes.cholesky import hll_step_periodic, max_signal_speed

    U, x = make_wave_pool_ic(N=128, u0=1.0, P0=0.1, seed=42)
    dx = x[1] - x[0]
    tr = Tracers.from_initial_conditions(U, x, periodic=True,
                                         boundary_buffer=12)

    cfl = 0.3
    max_observed = 0.0
    for _ in range(10):
        smax = max_signal_speed(U)
        dt = cfl * dx / smax
        U = hll_step_periodic(U, dx, dt, tau=1e-3)
        tr.update(U, x)
        stats = tr.step_stats()
        max_observed = max(max_observed, stats['max_abs'])

    # Each step's max |Δx| should stay well under the scan reach.
    assert max_observed < 5 * dx, \
        f"per-step max_abs = {max_observed:.4f} approaches 5*dx = {5*dx:.4f}"


def test_lost_flag_catches_damaged_zone_entry():
    """A tracer manually placed inside the buffered damaged zone is
    flagged as lost on the next update()."""
    U0, x = _uniform_periodic_ic(N=128)
    dx = x[1] - x[0]
    # Construct a small set: two safely mid-domain tracers + one deep
    # inside the left damaged zone (cell 1, where the HLL-smeared L1
    # makes the lookup unreliable).
    q = np.array([dx, 0.3, 0.5])
    x_pos = np.array([1.5 * dx, 0.3, 0.5])
    label = x_pos.copy()
    tr = Tracers(q=q, label=label, x=x_pos, domain=(0.0, 1.0),
                 periodic=True, boundary_buffer=5)
    tr._idx_hint[:3] = np.array([1, 38, 64], dtype=np.int64)
    tr.update(U0, x)
    # Mid-domain tracers are valid; the near-seam one is lost.
    assert tr.valid.sum() == 2
    assert tr.lost[0] == True
    assert tr.lost[1] == False
    assert tr.lost[2] == False


def test_lost_flag_clears_on_drift_out():
    """If a previously-lost tracer drifts back out of the damaged zone,
    the flag clears on the next update()."""
    U0, x = _uniform_periodic_ic(N=128)
    dx = x[1] - x[0]
    # Start a tracer inside the buffer.
    tr = Tracers(q=np.array([dx]), label=np.array([1.5 * dx]),
                 x=np.array([1.5 * dx]), domain=(0.0, 1.0),
                 periodic=True, boundary_buffer=5)
    tr._idx_hint[:1] = np.array([1], dtype=np.int64)
    tr.update(U0, x)
    assert tr.lost[0] == True

    # Manually move it out of the damaged zone, then update.
    tr._x[0] = 0.5
    tr._label[0] = 0.5
    tr._idx_hint[0] = 64
    tr.update(U0, x)
    assert tr.lost[0] == False


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


def test_add_seam_tracer_periodic_basic():
    """add_seam_tracer in periodic mode places one tracer at the
    parabolic x(q=0) seam estimate (≈ 0 for the L1 = x IC), marks
    it as seam, and the identity update keeps it in place even
    though it sits inside the boundary buffer."""
    U0, x = _uniform_periodic_ic(N=64)
    dx = x[1] - x[0]
    tr = Tracers.from_initial_conditions(U0, x, periodic=True,
                                         boundary_buffer=4)
    n_before = tr.n
    k = tr.add_seam_tracer(U0, x)
    assert k == n_before
    assert tr.n == n_before + 1
    assert tr.seam[k] == True
    assert tr.seam.sum() == 1
    # At t=0 with L1 = x and a uniform-rho ramp, the parabola
    # x(q=0) gives x=0 (machine epsilon).
    assert min(tr.x[k], 1.0 - tr.x[k]) < 1e-10, \
        f"seam tracer t=0 x={tr.x[k]}, expected ~0 or ~1"
    # Identity update at t=0: seam tracer must NOT be flagged lost,
    # despite living inside the buffer.
    tr.update(U0, x)
    assert not tr.lost[k]


def test_add_seam_tracer_transmissive_is_noop():
    """Outside periodic mode there is no seam — calling add_seam_tracer
    is a no-op (returns -1, no tracer added)."""
    U0, x = _uniform_periodic_ic(N=32)
    tr = Tracers.from_initial_conditions(U0, x, periodic=False)
    n_before = tr.n
    rv = tr.add_seam_tracer(U0, x)
    assert rv == -1
    assert tr.n == n_before
    assert tr.seam.sum() == 0


def test_seam_tracer_linear_fit_t0_matches_quadratic():
    """At t=0 the L1 = x ramp is linear, so the linear and quadratic
    fits both return x(q=0) = 0 to machine precision and agree with
    each other. Both seam tracers can coexist in one Tracers object."""
    U0, x = _uniform_periodic_ic(N=128)
    tr = Tracers.from_initial_conditions(U0, x, periodic=True,
                                         boundary_buffer=8)
    k_q = tr.add_seam_tracer(U0, x, fit='quadratic')
    k_l = tr.add_seam_tracer(U0, x, fit='linear')
    assert tr.seam.sum() == 2
    # Both x(q=0) values are within machine epsilon of the wrap.
    for k in (k_q, k_l):
        d = min(tr.x[k], 1.0 - tr.x[k])
        assert d < 1e-10
    # They agree with each other at t=0 to machine precision.
    diff = tr.x[k_q] - tr.x[k_l]
    diff -= round(diff)
    assert abs(diff) < 1e-10


def test_seam_tracer_invalid_fit_raises():
    """A bad `fit=` kwarg should raise immediately rather than silently
    fall back to one of the modes."""
    U0, x = _uniform_periodic_ic(N=32)
    tr = Tracers.from_initial_conditions(U0, x, periodic=True,
                                         boundary_buffer=2)
    with pytest.raises(ValueError):
        tr.add_seam_tracer(U0, x, fit='cubic')


def test_seam_tracer_t0_exact_for_linear_ramp():
    """At t=0 the L1 = x convention gives a perfectly linear
    Lagrangian field; the parabolic x(q) fit through the
    shifted-largest cell and its two right-neighbours collapses
    to a line and x(q=0) lands on the seam to round-off."""
    U0, x = _uniform_periodic_ic(N=128)
    tr = Tracers.from_initial_conditions(U0, x, periodic=True,
                                         boundary_buffer=8)
    k = tr.add_seam_tracer(U0, x)
    # x = 0 (mod 1) within machine epsilon
    distance_to_seam = min(tr.x[k], 1.0 - tr.x[k])
    assert distance_to_seam < 1e-10, \
        f"seam x = {tr.x[k]:.3e}, distance from wrap = {distance_to_seam:.3e}"


def test_seam_tracer_survives_wavepool_run():
    """The seam tracer never NaN's out across a wave-pool run."""
    from dfmm.schemes.cholesky import hll_step_periodic, max_signal_speed

    N = 128
    t_end = 0.3
    cfl = 0.3
    buf = 12

    U, x = make_wave_pool_ic(N, u0=1.0, P0=0.1, seed=42)
    dx = x[1] - x[0]
    tr = Tracers.from_initial_conditions(U, x, periodic=True,
                                         boundary_buffer=buf)
    seam_k = tr.add_seam_tracer(U, x)

    n_lost_steps = 0
    t = 0.0
    while t < t_end:
        smax = max_signal_speed(U)
        dt = min(cfl * dx / smax, t_end - t)
        U = hll_step_periodic(U, dx, dt, tau=1e-3)
        tr.update(U, x)
        if tr.lost[seam_k]:
            n_lost_steps += 1
        assert np.isfinite(tr.x[seam_k])
        assert 0.0 <= tr.x[seam_k] < 1.0
        t += dt
    assert n_lost_steps == 0, \
        f"seam tracer was flagged lost on {n_lost_steps} steps"


def test_lagrangian_period_keeps_L1_near_linear():
    """With `lagrangian_period=L` the periodic BC shifts the L1 ghost
    values by ±period * rho so the linear-ramp Lagrangian field
    stays continuous through the wrap. After many steps the L1 field
    should remain very close to the initial L1 = x ramp (no HLL
    smearing of an artificial seam discontinuity)."""
    from dfmm.config import SimulationConfig
    from dfmm.schemes._common import Workspace
    from dfmm.schemes.cholesky import hll_step, max_signal_speed
    from dfmm.schemes.boundaries import (pad_with_ghosts, unpad_ghosts,
                                          apply_periodic)

    N = 64
    period = 1.0
    U, x = make_wave_pool_ic(N, u0=1.0, P0=0.1, seed=42)
    dx = x[1] - x[0]
    cfg = SimulationConfig(cfl=0.3, tau=1e-3,
                           bc_left='periodic', bc_right='periodic')
    n_ghost = cfg.n_ghost

    def _step(U_in, lagrangian_period, n_steps):
        Ug = pad_with_ghosts(U_in.copy(), n_ghost)
        ws = Workspace.for_padded_state(Ug)
        for _ in range(n_steps):
            apply_periodic(Ug, n_ghost, lagrangian_period=lagrangian_period)
            smax = max_signal_speed(unpad_ghosts(Ug, n_ghost))
            dt = cfg.cfl * dx / smax
            hll_step(Ug, dx, dt, cfg.tau, n_ghost, cfg.rho_floor,
                     cfg.alpha_floor, cfg.realizability_headroom,
                     ws.Unew, ws.Fleft)
            Ug, ws.Unew = ws.Unew, Ug
        return unpad_ghosts(Ug, n_ghost)

    U_old = _step(U, lagrangian_period=0.0, n_steps=20)
    U_new = _step(U, lagrangian_period=period, n_steps=20)

    L1_old = U_old[IDX_L1] / np.maximum(U_old[IDX_RHO], 1e-30)
    L1_new = U_new[IDX_L1] / np.maximum(U_new[IDX_RHO], 1e-30)
    err_old = float(np.max(np.abs(L1_old - x)))
    err_new = float(np.max(np.abs(L1_new - x)))
    # New BC should keep the L1 field dramatically closer to the
    # ideal linear ramp.
    assert err_new < 0.1
    assert err_new < 0.2 * err_old


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
