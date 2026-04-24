"""
Calibrated Kramers-Moyal noise model and energy-conservative stepper.

The stochastic sub-grid correction takes the form

    epsilon(rho u) = C_A * rho * (du/dx) * dt
                   + C_B * rho * sqrt(max(-du/dx, 0) * dt) * eta,

where eta is a unit-variance Laplace random variable, C_A ~ 0.34 is the
compression-induced drift coefficient, and C_B ~ 0.55 is the noise
amplitude coefficient; both are calibrated once per scheme via fine-DNS
comparison on a broadband wave-pool flow (see
`experiments/11_kmles_calibrate.py`).

Per-step injection of delta(rho u) changes kinetic energy by

    Delta_KE = u * delta + delta^2/(2 rho).

To preserve total energy, the stepper in this module debits the same
Delta_KE from the internal energy reservoir, distributed equally across
the three internal degrees of freedom:

    P_xx -> P_xx - (2/3) Delta_KE,  P_perp -> P_perp - (2/3) Delta_KE,

so that (P_xx + 2 P_perp)/2, the internal energy per volume, decreases
by exactly Delta_KE. An amplitude limiter caps the per-cell injection
at 25% of local internal energy to prevent rare Laplace heavy-tail
events from requesting more than is thermodynamically available.

Spatial correlation of the noise is produced by drawing per-cell i.i.d.
Laplace samples, then smoothing with a Gaussian kernel of correlation
length ell_corr cells and renormalizing to preserve variance. The
optimal ell_corr from the calibration is ~2 cells.
"""
from dataclasses import replace
import numpy as np
import numba as nb

from ..config import SimulationConfig
from ..schemes.cholesky import max_signal_speed
from ..schemes._common import (CSCOEF, IDX_RHO, IDX_MOM, IDX_EXX, IDX_PP,
                               IDX_L1, IDX_ALPHA, IDX_BETA, IDX_M3,
                               hll_edge_flux, Workspace)
from ..schemes.boundaries import pad_with_ghosts, unpad_ghosts, apply_mixed


def smooth_gaussian_periodic(eta, ell_corr):
    """Gaussian-smooth a periodic 1D array in k-space, preserve variance.

    If ell_corr <= 0 the input is returned unchanged.
    """
    if ell_corr <= 0:
        return eta
    N = len(eta)
    k = np.fft.fftfreq(N, d=1.0)
    G = np.exp(-2*np.pi**2 * k**2 * ell_corr**2)
    eta_k = np.fft.fft(eta)
    eta_smooth = np.real(np.fft.ifft(eta_k * G))
    std_orig = np.std(eta)
    std_smooth = np.std(eta_smooth)
    if std_smooth > 0:
        eta_smooth *= std_orig/std_smooth
    return eta_smooth


@nb.njit(cache=True, fastmath=False)
def hll_step_noise(U, dx, dt, tau, n_ghost,
                   rho_floor, alpha_floor, realizability_headroom,
                   pressure_floor, noise_ke_budget_fraction,
                   C_A, C_B, eta_draw,
                   Unew, Fleft):
    """Ghost-cell HLL + BGK + energy-conservative noise-injection step.

    Mirrors `dfmm.schemes.cholesky.hll_step` layout and then applies
    the noise-injection block after BGK. `eta_draw` is sized to the
    interior (length N, not N_tot) — only interior cells receive
    noise. Caller fills ghost cells via `boundaries.apply_mixed`
    beforehand so flux stencils at the domain edges work consistently
    with whatever BC the caller requested.

    All tolerances (rho_floor, alpha_floor, realizability_headroom,
    pressure_floor, noise_ke_budget_fraction) are passed explicitly
    as scalars so SimulationConfig fields can be sensitivity-studied
    without recompilation.
    """
    n_fields, N_tot = U.shape
    N = N_tot - 2 * n_ghost
    inv_dx = 1.0 / dx

    rho   = U[IDX_RHO]
    u     = U[IDX_MOM]/rho
    Pxx   = U[IDX_EXX] - rho*u*u
    Pp    = U[IDX_PP]
    M3    = U[IDX_M3]
    Q     = M3 - rho*u*u*u - 3.0*u*Pxx
    cs    = np.sqrt(CSCOEF*np.maximum(Pxx, rho_floor)/np.maximum(rho, rho_floor))

    # Fluxes at interior faces
    for i in range(n_ghost, n_ghost + N + 1):
        F0, F1, F2, F3, F4, F5, F6, F7 = hll_edge_flux(
            U[:, i-1], U[:, i], cs[i-1], cs[i])
        Fleft[0, i] = F0; Fleft[1, i] = F1; Fleft[2, i] = F2; Fleft[3, i] = F3
        Fleft[4, i] = F4; Fleft[5, i] = F5; Fleft[6, i] = F6; Fleft[7, i] = F7

    # Conservative update (interior cells only)
    for j in range(n_ghost, n_ghost + N):
        for k in range(n_fields):
            Unew[k, j] = U[k, j] - dt*inv_dx*(Fleft[k, j+1] - Fleft[k, j])
    # Copy ghost cells through unchanged.
    for g in range(n_ghost):
        for k in range(n_fields):
            Unew[k, g] = U[k, g]
            Unew[k, n_ghost + N + g] = U[k, n_ghost + N + g]

    # Exact-exponential BGK relaxation (interior cells)
    decay = np.exp(-dt/tau)
    for j in range(n_ghost, n_ghost + N):
        rho_n = Unew[IDX_RHO, j]
        u_n   = Unew[IDX_MOM, j]/rho_n
        Pxx_n = Unew[IDX_EXX, j] - rho_n*u_n*u_n
        Pp_n  = Unew[IDX_PP,  j]
        M3_n  = Unew[IDX_M3,  j]
        Q_n   = M3_n - rho_n*u_n*u_n*u_n - 3.0*u_n*Pxx_n
        b_n   = Unew[IDX_BETA, j]/rho_n
        P_iso = (Pxx_n + 2.0*Pp_n)/3.0
        Pxx_new = P_iso + (Pxx_n - P_iso)*decay
        Pp_new  = P_iso + (Pp_n  - P_iso)*decay
        Q_new = Q_n * decay
        b_new = b_n * decay
        Svv_n = max(Pxx_new/rho_n, rho_floor)
        b_max = realizability_headroom * np.sqrt(Svv_n)
        if b_new >  b_max: b_new =  b_max
        elif b_new < -b_max: b_new = -b_max
        Unew[IDX_EXX, j] = rho_n*u_n*u_n + Pxx_new
        Unew[IDX_PP,  j] = Pp_new
        Unew[IDX_BETA,j] = rho_n*b_new
        Unew[IDX_M3,  j] = rho_n*u_n*u_n*u_n + 3.0*u_n*Pxx_new + Q_new

    # Energy-conservative noise injection (interior cells).
    # dudx uses the ghost-filled neighbours, so the stencil is
    # periodic when ghosts came from apply_periodic, one-sided when
    # from apply_transmissive, etc.
    for j in range(n_ghost, n_ghost + N):
        eta_idx = j - n_ghost
        dudx_j = (u[j+1] - u[j-1]) * 0.5 / dx
        rho_i = Unew[IDX_RHO, j]
        u_i   = Unew[IDX_MOM, j]/rho_i
        Exx_old = Unew[IDX_EXX, j]
        Pxx_old = Exx_old - rho_i*u_i*u_i
        Pp_old  = Unew[IDX_PP,  j]
        M3_old  = Unew[IDX_M3,  j]
        Q_old   = M3_old - rho_i*u_i*u_i*u_i - 3.0*u_i*Pxx_old

        drift_term = C_A * rho[j] * dudx_j * dt
        compression = max(-dudx_j, 0.0)
        noise_amp = C_B * rho[j] * np.sqrt(compression * dt)
        delta_rhou = drift_term + noise_amp * eta_draw[eta_idx]

        # Amplitude limiter: KE injection must not exceed the
        # configured fraction of local internal energy.
        IE_local = 0.5*Pxx_old + Pp_old
        KE_budget = noise_ke_budget_fraction * IE_local
        abs_u = abs(u_i)
        disc = abs_u*abs_u + 2.0*KE_budget/rho_i
        delta_max = (-abs_u + np.sqrt(disc))*rho_i
        if delta_rhou > delta_max: delta_rhou = delta_max
        elif delta_rhou < -delta_max: delta_rhou = -delta_max

        rhou_new = Unew[IDX_MOM, j] + delta_rhou
        u_new = rhou_new/rho_i

        Delta_KE_vol = u_i * delta_rhou + 0.5 * delta_rhou**2 / rho_i

        # Debit internal energy to preserve total energy
        Pxx_new = Pxx_old - (2.0/3.0)*Delta_KE_vol
        Pp_new  = Pp_old  - (2.0/3.0)*Delta_KE_vol
        if Pxx_new < pressure_floor: Pxx_new = pressure_floor
        if Pp_new  < pressure_floor: Pp_new  = pressure_floor

        Unew[IDX_MOM, j] = rhou_new
        Unew[IDX_EXX, j] = rho_i*u_new*u_new + Pxx_new
        Unew[IDX_PP,  j] = Pp_new
        Unew[IDX_M3,  j] = rho_i*u_new**3 + 3.0*u_new*Pxx_new + Q_old

    return Unew


@nb.njit(cache=True, fastmath=False)
def hll_step_noise_econsv(U, dx, dt, tau, C_A, C_B, eta_draw):
    """Periodic HLL + BGK step with energy-conservative noise injection.

    Parameters
    ----------
    U : ndarray, shape (8, N)
        Pre-step state.
    dx, dt : float
    tau : float
        BGK relaxation time.
    C_A, C_B : float
        Calibrated drift and noise coefficients.
    eta_draw : ndarray, shape (N,)
        Pre-drawn unit-variance noise values (typically Laplace, possibly
        spatially smoothed by `smooth_gaussian_periodic`).

    Returns
    -------
    Unew : ndarray, shape (8, N)
        Updated state after one HLL+BGK+noise step.
    """
    n_fields, N = U.shape
    Unew = np.empty_like(U)

    rho   = U[IDX_RHO]
    u     = U[IDX_MOM]/rho
    Pxx   = U[IDX_EXX] - rho*u*u
    Pp    = U[IDX_PP]
    L1    = U[IDX_L1]/rho
    alpha = U[IDX_ALPHA]/rho
    beta  = U[IDX_BETA]/rho
    M3    = U[IDX_M3]
    Q     = M3 - rho*u*u*u - 3.0*u*Pxx
    cs    = np.sqrt(CSCOEF*np.maximum(Pxx, 1e-30)/np.maximum(rho, 1e-30))

    # Centered-difference du/dx (periodic)
    dudx = np.empty(N)
    inv_2dx = 0.5/dx
    for i in range(N):
        ip = (i+1) % N; im = (i-1) % N
        dudx[i] = (u[ip] - u[im])*inv_2dx

    # HLL fluxes
    Fleft = np.empty((n_fields, N))
    for i in range(N):
        l = (i-1) % N
        F0, F1, F2, F3, F4, F5, F6, F7 = hll_edge_flux(U[:, l], U[:, i], cs[l], cs[i])
        Fleft[0, i] = F0; Fleft[1, i] = F1; Fleft[2, i] = F2; Fleft[3, i] = F3
        Fleft[4, i] = F4; Fleft[5, i] = F5; Fleft[6, i] = F6; Fleft[7, i] = F7

    # Conservative update
    inv_dx = 1.0/dx
    for i in range(N):
        ip = (i+1) % N
        for k in range(n_fields):
            Unew[k, i] = U[k, i] - dt*inv_dx*(Fleft[k, ip] - Fleft[k, i])

    # Exact-exponential BGK relaxation
    decay = np.exp(-dt/tau)
    for i in range(N):
        rho_n = Unew[IDX_RHO, i]
        u_n   = Unew[IDX_MOM, i]/rho_n
        Pxx_n = Unew[IDX_EXX, i] - rho_n*u_n*u_n
        Pp_n  = Unew[IDX_PP,  i]
        M3_n  = Unew[IDX_M3,  i]
        Q_n   = M3_n - rho_n*u_n*u_n*u_n - 3.0*u_n*Pxx_n
        b_n   = Unew[IDX_BETA, i]/rho_n
        P_iso = (Pxx_n + 2.0*Pp_n)/3.0
        Pxx_new = P_iso + (Pxx_n - P_iso)*decay
        Pp_new  = P_iso + (Pp_n  - P_iso)*decay
        Q_new = Q_n * decay
        b_new = b_n * decay
        Svv_n = max(Pxx_new/rho_n, 1e-30)
        b_max = 0.999*np.sqrt(Svv_n)
        if b_new >  b_max: b_new =  b_max
        elif b_new < -b_max: b_new = -b_max
        Unew[IDX_EXX, i] = rho_n*u_n*u_n + Pxx_new
        Unew[IDX_PP,  i] = Pp_new
        Unew[IDX_BETA,i] = rho_n*b_new
        Unew[IDX_M3,  i] = rho_n*u_n*u_n*u_n + 3.0*u_n*Pxx_new + Q_new

    # Energy-conservative noise injection
    for i in range(N):
        rho_i = Unew[IDX_RHO, i]
        u_i   = Unew[IDX_MOM, i]/rho_i
        Exx_old = Unew[IDX_EXX, i]
        Pxx_old = Exx_old - rho_i*u_i*u_i
        Pp_old  = Unew[IDX_PP,  i]
        M3_old  = Unew[IDX_M3,  i]
        Q_old   = M3_old - rho_i*u_i*u_i*u_i - 3.0*u_i*Pxx_old

        drift_term = C_A * rho[i] * dudx[i] * dt
        compression = max(-dudx[i], 0.0)
        noise_amp = C_B * rho[i] * np.sqrt(compression * dt)
        delta_rhou = drift_term + noise_amp * eta_draw[i]

        # Amplitude limiter so the injected KE cannot exceed 25% of
        # local internal energy (otherwise heavy-tail Laplace draws can
        # occasionally request more energy than locally available).
        IE_local = 0.5*Pxx_old + Pp_old
        KE_budget = 0.25*IE_local
        abs_u = abs(u_i)
        disc = abs_u*abs_u + 2.0*KE_budget/rho_i
        delta_max = (-abs_u + np.sqrt(disc))*rho_i
        if delta_rhou > delta_max: delta_rhou = delta_max
        elif delta_rhou < -delta_max: delta_rhou = -delta_max

        rhou_new = Unew[IDX_MOM, i] + delta_rhou
        u_new = rhou_new/rho_i

        Delta_KE_vol = u_i * delta_rhou + 0.5 * delta_rhou**2 / rho_i

        # Debit internal energy to preserve total energy
        Pxx_new = Pxx_old - (2.0/3.0)*Delta_KE_vol
        Pp_new  = Pp_old  - (2.0/3.0)*Delta_KE_vol
        if Pxx_new < 1e-8: Pxx_new = 1e-8
        if Pp_new  < 1e-8: Pp_new  = 1e-8

        Unew[IDX_MOM, i] = rhou_new
        Unew[IDX_EXX, i] = rho_i*u_new*u_new + Pxx_new
        Unew[IDX_PP,  i] = Pp_new
        Unew[IDX_M3,  i] = rho_i*u_new**3 + 3.0*u_new*Pxx_new + Q_old

    return Unew


def run_noise(U, t_end, save_times, C_A, C_B, ell_corr=2.0, cfl=None,
              tau=None, seed=0, cfg=None):
    """Energy-conservative driver with calibrated noise injection.

    Wraps the ghost-cell `hll_step_noise` kernel with per-step noise-
    draw + smoothing and the SimulationConfig-driven tolerances that
    the rest of the package uses.

    Parameters
    ----------
    U : ndarray, shape (8, N)
        Initial state (interior only; ghost padding is handled
        internally).
    t_end : float
    save_times : array-like
    C_A, C_B : float
        Calibrated noise coefficients (typically loaded from
        `data/noise_model_params.npz`).
    ell_corr : float, default 2.0
        Spatial correlation length of the noise, in cells.
    cfl : float, optional
        CFL coefficient; overrides `cfg.cfl` if both given.
    tau : float, optional
        BGK relaxation time; overrides `cfg.tau`.
    seed : int
        RNG seed for the Laplace draws.
    cfg : SimulationConfig, optional
        Full config. Defaults to `SimulationConfig()` (periodic BCs,
        first-order, standard floors). Pass an explicit cfg to run
        noise with non-default floors, limiter sensitivities, or
        non-periodic BCs.

    Returns
    -------
    snapshots : list of (float, ndarray)
        Matches the convention of `dfmm.integrate.run_to`.
    """
    if cfg is None:
        cfg = SimulationConfig()
    overrides = {}
    if cfl is not None: overrides['cfl'] = cfl
    if tau is not None: overrides['tau'] = tau
    if overrides:
        cfg = replace(cfg, **overrides)

    n_ghost = cfg.n_ghost
    N = U.shape[1]
    dx = 1.0 / N

    # Pad and allocate workspace once.
    U_curr = pad_with_ghosts(U, n_ghost)
    ws = Workspace.for_padded_state(U_curr, reconstruction=cfg.reconstruction)
    U_next = ws.Unew

    rng = np.random.default_rng(seed)
    t = 0.0
    snapshots = [(0.0, U.copy())]
    save_idx = 0
    while t < t_end and save_idx < len(save_times):
        next_save = save_times[save_idx]
        apply_mixed(U_curr, n_ghost, cfg.bc_left, cfg.bc_right,
                    state_left=cfg.bc_state_left,
                    state_right=cfg.bc_state_right)
        smax = max_signal_speed(unpad_ghosts(U_curr, n_ghost))
        dt = min(cfg.cfl * dx / smax, next_save - t, t_end - t)
        if dt <= 1e-14:
            snapshots.append((t, unpad_ghosts(U_curr, n_ghost).copy()))
            save_idx += 1
            continue
        eta_white = rng.laplace(scale=1.0/np.sqrt(2.0), size=N)
        eta = smooth_gaussian_periodic(eta_white, ell_corr) if ell_corr > 0 \
              else eta_white
        hll_step_noise(U_curr, dx, dt, cfg.tau, n_ghost,
                       cfg.rho_floor, cfg.alpha_floor,
                       cfg.realizability_headroom,
                       cfg.pressure_floor, cfg.noise_ke_budget_fraction,
                       C_A, C_B, eta,
                       U_next, ws.Fleft)
        U_curr, U_next = U_next, U_curr
        t += dt
        if t >= next_save - 1e-12:
            snapshots.append((t, unpad_ghosts(U_curr, n_ghost).copy()))
            save_idx += 1
    return snapshots


def total_energy(U):
    """Per-cell total energy (KE + internal E per unit volume).

    Useful for verifying energy conservation of the noise-augmented
    scheme; unaltered compared to the deterministic total-energy formula.
    """
    rho = U[0]; u = U[1]/rho
    Pxx = U[2] - rho*u*u
    Pp = U[3]
    KE = 0.5*rho*u*u
    IE = 0.5*(Pxx + 2*Pp)
    return KE + IE
