"""
Cholesky-form moment scheme — the eight-field production scheme.

The 2x2 position-velocity phase-space covariance Sigma is carried in
factored form  Sigma = L L^T  with

    L = [[alpha, 0      ],
         [beta , gamma  ]]

so that

    Sigma_xx = alpha^2
    Sigma_xv = alpha * beta
    Sigma_vv = beta^2 + gamma^2.

Sigma_vv = P_xx / rho is determined by the hydro variables. We evolve
alpha and beta as independent fields; gamma is reconstructed pointwise
from gamma^2 = max(Sigma_vv - beta^2, 0), which makes realizability
automatic. The vanishing of gamma flags phase-space rank collapse and
serves as a built-in closure-quality diagnostic with several decades of
dynamic range.

Kinematic source equations (derived from the dot-Sigma equations under
phase-space advection):

    D alpha / Dt = beta
    D beta  / Dt = gamma^2 / alpha - (du/dx) * beta

BGK relaxation drives beta toward zero with timescale tau (exact
exponential); alpha is unchanged by relaxation.

Conserved state vector (eight fields):

    U[0] = rho
    U[1] = rho u
    U[2] = E_xx = rho u^2 + P_xx
    U[3] = P_perp
    U[4] = rho L_1               (Lagrangian label)
    U[5] = rho alpha             (Cholesky 11-component)
    U[6] = rho beta              (Cholesky 21-component)
    U[7] = M_3 = rho <v^3>       (third velocity moment)

Fourth-moment closure is Wick-Gaussian by default.
"""
import numpy as np
import numba as nb

from ._common import (CSCOEF, IDX_RHO, IDX_MOM, IDX_EXX, IDX_PP,
                      IDX_L1, IDX_ALPHA, IDX_BETA, IDX_M3,
                      hll_edge_flux)


def primitives(U):
    rho   = U[IDX_RHO]
    u     = U[IDX_MOM]/rho
    Pxx   = U[IDX_EXX] - rho*u*u
    Pp    = U[IDX_PP]
    L1    = U[IDX_L1]/rho
    alpha = U[IDX_ALPHA]/rho
    beta  = U[IDX_BETA]/rho
    M3    = U[IDX_M3]
    Q     = M3 - rho*u*u*u - 3.0*u*Pxx
    Sigma_vv = np.maximum(Pxx, 1e-30)/np.maximum(rho, 1e-30)
    Sigma_xx = alpha*alpha
    Sigma_xv = alpha*beta
    gamma2   = np.maximum(Sigma_vv - beta*beta, 0.0)
    gamma    = np.sqrt(gamma2)
    return rho, u, Pxx, Pp, L1, alpha, beta, gamma, Sigma_xx, Sigma_xv, Sigma_vv, Q


@nb.njit(cache=True, fastmath=False)
def max_signal_speed(U):
    rho = U[IDX_RHO]; u = U[IDX_MOM]/rho
    Pxx = U[IDX_EXX] - rho*u*u
    cs = np.sqrt(CSCOEF*np.maximum(Pxx, 1e-30)/np.maximum(rho, 1e-30))
    return float(np.max(np.abs(u) + cs))


@nb.njit(cache=True, fastmath=False)
def hll_step(U, dx, dt, tau, n_ghost,
             rho_floor, alpha_floor, realizability_headroom):
    """Unified HLL + BGK step on a ghost-padded state.

    Caller is responsible for filling the ghost cells before calling
    (see `dfmm.schemes.boundaries.apply_mixed`). Only the interior
    cells of the returned array hold updated values; the ghost slots
    are copied through unchanged (so a subsequent BC apply overwrites
    them cleanly).

    `U` must have shape `(n_fields, N + 2*n_ghost)`; interior is the
    slice `U[:, n_ghost:n_ghost+N]`. All tolerances (rho_floor,
    alpha_floor, realizability_headroom) are scalar args so Numba
    stays a happy camper.
    """
    n_fields, N_tot = U.shape
    N = N_tot - 2 * n_ghost
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
    cs    = np.sqrt(CSCOEF*np.maximum(Pxx, rho_floor)/np.maximum(rho, rho_floor))

    # Flux at the N+1 interior faces. Face i (for i in [n_ghost,
    # n_ghost+N]) is between cell i-1 and cell i in the padded layout.
    Fleft = np.empty((n_fields, N_tot + 1))
    for i in range(n_ghost, n_ghost + N + 1):
        F0, F1, F2, F3, F4, F5, F6, F7 = hll_edge_flux(
            U[:, i-1], U[:, i], cs[i-1], cs[i])
        Fleft[0, i] = F0; Fleft[1, i] = F1; Fleft[2, i] = F2; Fleft[3, i] = F3
        Fleft[4, i] = F4; Fleft[5, i] = F5; Fleft[6, i] = F6; Fleft[7, i] = F7

    # Conservative update for interior cells
    inv_dx = 1.0/dx
    for j in range(n_ghost, n_ghost + N):
        for k in range(n_fields):
            Unew[k, j] = U[k, j] - dt*inv_dx*(Fleft[k, j+1] - Fleft[k, j])
    # Copy ghost cells through (apply_* will rewrite them on next step)
    for g in range(n_ghost):
        for k in range(n_fields):
            Unew[k, g] = U[k, g]
            Unew[k, n_ghost + N + g] = U[k, n_ghost + N + g]

    # Liouville sources for alpha and beta.
    # Ghost cells provide neighbours for the central du/dx stencil,
    # which is why n_ghost >= 1 is required. For periodic BCs with
    # apply_periodic this reproduces the original modulo-wrapped
    # stencil bit-identically. For transmissive (copy-nearest-interior)
    # it gives du/dx at the boundary as (u_inner_next - u_inner) /
    # (2*dx), i.e. half the original one-sided forward diff — a
    # modest numerical change, consistent with zero-gradient at the
    # boundary.
    for j in range(n_ghost, n_ghost + N):
        Sigma_vv_i = Pxx[j]/max(rho[j], rho_floor)
        a = alpha[j]; b = beta[j]
        gamma2_signed = Sigma_vv_i - b*b
        dudx_i = (u[j+1] - u[j-1])/(2.0*dx)
        Unew[IDX_ALPHA, j] += dt*rho[j]*b
        a_safe = a if a > alpha_floor else alpha_floor
        Unew[IDX_BETA, j]  += dt*rho[j]*(gamma2_signed/a_safe - dudx_i*b)

    # Exact-exponential BGK relaxation on interior cells
    decay = np.exp(-dt/tau)
    for j in range(n_ghost, n_ghost + N):
        rho_n = Unew[IDX_RHO, j]
        u_n   = Unew[IDX_MOM, j]/rho_n
        Pxx_n = Unew[IDX_EXX, j] - rho_n*u_n*u_n
        Pp_n  = Unew[IDX_PP, j]
        a_n   = Unew[IDX_ALPHA, j]/rho_n
        b_n   = Unew[IDX_BETA, j]/rho_n
        M3_n  = Unew[IDX_M3, j]
        Q_n   = M3_n - rho_n*u_n*u_n*u_n - 3.0*u_n*Pxx_n

        P_iso = (Pxx_n + 2.0*Pp_n)/3.0
        Pxx_new = P_iso + (Pxx_n - P_iso)*decay
        Pp_new  = P_iso + (Pp_n  - P_iso)*decay
        b_new = b_n*decay
        Q_new = Q_n*decay

        Sigma_vv_new = max(Pxx_new, rho_floor)/max(rho_n, rho_floor)
        beta_max = realizability_headroom*np.sqrt(Sigma_vv_new)
        if b_new > beta_max:    b_new = beta_max
        elif b_new < -beta_max: b_new = -beta_max

        Unew[IDX_EXX, j]  = rho_n*u_n*u_n + Pxx_new
        Unew[IDX_PP,  j]  = Pp_new
        Unew[IDX_BETA, j] = rho_n*b_new
        Unew[IDX_M3,  j]  = rho_n*u_n*u_n*u_n + 3.0*u_n*Pxx_new + Q_new

    return Unew


@nb.njit(cache=True, fastmath=False)
def hll_step_periodic(U, dx, dt, tau):
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

    Fleft = np.empty((n_fields, N))
    for i in range(N):
        l = (i-1) % N
        F0, F1, F2, F3, F4, F5, F6, F7 = hll_edge_flux(U[:, l], U[:, i], cs[l], cs[i])
        Fleft[0, i] = F0; Fleft[1, i] = F1; Fleft[2, i] = F2; Fleft[3, i] = F3
        Fleft[4, i] = F4; Fleft[5, i] = F5; Fleft[6, i] = F6; Fleft[7, i] = F7

    inv_dx = 1.0/dx
    for i in range(N):
        ip = (i+1) % N
        for k in range(n_fields):
            Unew[k, i] = U[k, i] - dt*inv_dx*(Fleft[k, ip] - Fleft[k, i])

    # Liouville sources for alpha and beta:
    #   D alpha/Dt = beta
    #   D beta/Dt  = (Sigma_vv - beta^2) / alpha - (du/dx) beta
    # The (Sigma_vv - beta^2) factor is gamma^2; we do NOT clip it to >= 0
    # in the source -- when beta^2 exceeds Sigma_vv the source becomes
    # negative, which is a restoring force that prevents runaway growth.
    # The clip-to-zero is only used in visualization where gamma must be
    # a real width.
    for i in range(N):
        Sigma_vv_i = Pxx[i]/max(rho[i], 1e-30)
        a = alpha[i]; b = beta[i]
        gamma2_signed = Sigma_vv_i - b*b           # may be negative
        ip2 = (i+1) % N; im2 = (i-1) % N
        dudx_i = (u[ip2] - u[im2])/(2.0*dx)
        Unew[IDX_ALPHA, i] += dt*rho[i]*b
        a_safe = a if a > 1e-15 else 1e-15
        Unew[IDX_BETA, i] += dt*rho[i]*(gamma2_signed/a_safe - dudx_i*b)

    # BGK relaxation
    decay = np.exp(-dt/tau)
    for i in range(N):
        rho_n = Unew[IDX_RHO, i]
        u_n   = Unew[IDX_MOM, i]/rho_n
        Pxx_n = Unew[IDX_EXX, i] - rho_n*u_n*u_n
        Pp_n  = Unew[IDX_PP, i]
        a_n   = Unew[IDX_ALPHA, i]/rho_n
        b_n   = Unew[IDX_BETA, i]/rho_n
        M3_n  = Unew[IDX_M3, i]
        Q_n   = M3_n - rho_n*u_n*u_n*u_n - 3.0*u_n*Pxx_n

        # Pressure isotropization
        P_iso = (Pxx_n + 2.0*Pp_n)/3.0
        Pxx_new = P_iso + (Pxx_n - P_iso)*decay
        Pp_new  = P_iso + (Pp_n  - P_iso)*decay
        # Cross-moment relaxation: beta decays exponentially
        b_new = b_n*decay
        # Heat flux relaxation
        Q_new = Q_n*decay

        # Realizability clip on beta:  |beta| < sqrt(Sigma_vv) since gamma^2 >= 0.
        # We clip to 0.999*sqrt(Sigma_vv) to keep gamma > 0 strictly.
        Sigma_vv_new = max(Pxx_new, 1e-30)/max(rho_n, 1e-30)
        beta_max = 0.999*np.sqrt(Sigma_vv_new)
        if b_new > beta_max:    b_new = beta_max
        elif b_new < -beta_max: b_new = -beta_max

        Unew[IDX_EXX, i] = rho_n*u_n*u_n + Pxx_new
        Unew[IDX_PP,  i] = Pp_new
        Unew[IDX_BETA,i] = rho_n*b_new
        Unew[IDX_M3,  i] = rho_n*u_n*u_n*u_n + 3.0*u_n*Pxx_new + Q_new
        # alpha unchanged by collisions

    return Unew


@nb.njit(cache=True, fastmath=False)
def hll_step_transmissive(U, dx, dt, tau):
    n_fields, N = U.shape
    Unew = np.empty_like(U)

    rho   = U[IDX_RHO]; u = U[IDX_MOM]/rho
    Pxx   = U[IDX_EXX] - rho*u*u; Pp = U[IDX_PP]; L1 = U[IDX_L1]/rho
    alpha = U[IDX_ALPHA]/rho; beta = U[IDX_BETA]/rho
    M3    = U[IDX_M3]; Q = M3 - rho*u*u*u - 3.0*u*Pxx
    cs    = np.sqrt(CSCOEF*np.maximum(Pxx, 1e-30)/np.maximum(rho, 1e-30))

    Fleft = np.empty((n_fields, N+1))
    for i in range(N+1):
        if i == 0:   l = 0;   r = 0
        elif i == N: l = N-1; r = N-1
        else:        l = i-1; r = i
        F0, F1, F2, F3, F4, F5, F6, F7 = hll_edge_flux(U[:, l], U[:, r], cs[l], cs[r])
        Fleft[0, i] = F0; Fleft[1, i] = F1; Fleft[2, i] = F2; Fleft[3, i] = F3
        Fleft[4, i] = F4; Fleft[5, i] = F5; Fleft[6, i] = F6; Fleft[7, i] = F7
    inv_dx = 1.0/dx
    for i in range(N):
        for k in range(n_fields):
            Unew[k, i] = U[k, i] - dt*inv_dx*(Fleft[k, i+1] - Fleft[k, i])
    for i in range(N):
        Sigma_vv_i = Pxx[i]/max(rho[i], 1e-30)
        a = alpha[i]; b = beta[i]
        gamma2 = Sigma_vv_i - b*b           # signed; negative -> restoring force on beta
        if i == 0:        dudx_i = (u[1] - u[0])/dx
        elif i == N-1:    dudx_i = (u[N-1] - u[N-2])/dx
        else:             dudx_i = (u[i+1] - u[i-1])/(2.0*dx)
        Unew[IDX_ALPHA, i] += dt*rho[i]*b
        a_safe = a if a > 1e-15 else 1e-15
        Unew[IDX_BETA, i] += dt*rho[i]*(gamma2/a_safe - dudx_i*b)
    decay = np.exp(-dt/tau)
    for i in range(N):
        rho_n = Unew[IDX_RHO, i]; u_n = Unew[IDX_MOM, i]/rho_n
        Pxx_n = Unew[IDX_EXX, i] - rho_n*u_n*u_n; Pp_n = Unew[IDX_PP, i]
        a_n = Unew[IDX_ALPHA, i]/rho_n; b_n = Unew[IDX_BETA, i]/rho_n
        M3_n = Unew[IDX_M3, i]; Q_n = M3_n - rho_n*u_n*u_n*u_n - 3.0*u_n*Pxx_n
        P_iso = (Pxx_n + 2.0*Pp_n)/3.0
        Pxx_new = P_iso + (Pxx_n - P_iso)*decay
        Pp_new  = P_iso + (Pp_n  - P_iso)*decay
        b_new = b_n*decay
        Q_new = Q_n*decay
        Sigma_vv_new = max(Pxx_new, 1e-30)/max(rho_n, 1e-30)
        beta_max = 0.999*np.sqrt(Sigma_vv_new)
        if b_new > beta_max:    b_new = beta_max
        elif b_new < -beta_max: b_new = -beta_max
        Unew[IDX_EXX, i] = rho_n*u_n*u_n + Pxx_new
        Unew[IDX_PP,  i] = Pp_new
        Unew[IDX_BETA,i] = rho_n*b_new
        Unew[IDX_M3,  i] = rho_n*u_n*u_n*u_n + 3.0*u_n*Pxx_new + Q_new
    return Unew


# ---------- Drivers ----------
def run_sine(N=400, t_end=0.5, tau=1e3, cfl=0.3, A=1.0, T0=1e-3, sigma_x0=0.02,
             snap_times=None):
    if snap_times is None: snap_times = [0.0, 0.1, 0.2, 0.5]
    x = np.linspace(0, 1, N, endpoint=False) + 0.5/N
    dx = 1.0/N
    rho = np.ones(N)
    u   = A*np.sin(2*np.pi*x)
    p   = np.full(N, T0)
    Pxx0 = p.copy(); Pp0 = p.copy()
    alpha0 = np.full(N, sigma_x0)              # alpha = sqrt(Sigma_xx_0)
    beta0  = np.zeros(N)                        # initially uncorrelated
    Q0     = np.zeros(N)
    M30    = rho*u*u*u + 3.0*u*Pxx0 + Q0
    U = np.array([rho, rho*u, rho*u*u + Pxx0, Pp0, rho*x, rho*alpha0, rho*beta0, M30])
    snaps = []
    snap_idx = 0; t = 0.0; nsteps = 0
    if snap_times[0] <= t:
        snaps.append((t, x.copy(), U.copy())); snap_idx += 1
    while t < t_end and snap_idx < len(snap_times):
        smax = max_signal_speed(U)
        dt = cfl*dx/smax
        next_snap = snap_times[snap_idx]
        dt = min(dt, next_snap - t, t_end - t)
        if dt <= 0: break
        U = hll_step_periodic(U, dx, dt, tau)
        t += dt; nsteps += 1
        if t >= snap_times[snap_idx] - 1e-12:
            snaps.append((t, x.copy(), U.copy())); snap_idx += 1
    return snaps, nsteps


def run_sod(N=800, t_end=0.2, tau=1e-6, cfl=0.4, sigma_x0=0.02):
    x = np.linspace(0, 1, N)
    dx = x[1]-x[0]
    rho = np.where(x < 0.5, 1.0, 0.125)
    u   = np.zeros(N)
    p   = np.where(x < 0.5, 1.0, 0.1)
    Pxx0 = p.copy(); Pp0 = p.copy()
    alpha0 = np.full(N, sigma_x0); beta0 = np.zeros(N)
    Q0 = np.zeros(N)
    M30 = rho*u*u*u + 3.0*u*Pxx0 + Q0
    U = np.array([rho, rho*u, rho*u*u + Pxx0, Pp0, rho*x, rho*alpha0, rho*beta0, M30])
    t = 0.0; nsteps = 0
    while t < t_end:
        smax = max_signal_speed(U)
        dt = min(cfl*dx/smax, t_end - t)
        U = hll_step_transmissive(U, dx, dt, tau)
        t += dt; nsteps += 1
    return x, U, nsteps


# ---------- Phase-space rendering (same as Step 2) ----------
@nb.njit(cache=True, fastmath=True, parallel=True)
def build_f_periodic(x_arr, u_arr, Sxx, Sxv, Svv, rho, dx_cell, x_grid, v_grid):
    Nx = len(x_grid); Nv = len(v_grid); Nc = len(x_arr)
    f = np.zeros((Nv, Nx))
    inv_two_pi = 1.0/(2.0*np.pi)
    for i in nb.prange(Nc):
        a = Sxx[i]; b = Sxv[i]; c = Svv[i]
        det = a*c - b*b
        if det < 1e-30: det = 1e-30
        inv00 = c/det; inv11 = a/det; inv01 = -b/det
        amp = rho[i]*dx_cell*inv_two_pi/np.sqrt(det)
        u_i = u_arr[i]
        for shift in (-1.0, 0.0, 1.0):
            x_c = x_arr[i] + shift
            for ix in range(Nx):
                dx_ = x_grid[ix] - x_c
                if dx_*dx_ > 25.0*a: continue
                pre_x = inv00*dx_*dx_; cross = 2.0*inv01*dx_
                for iv in range(Nv):
                    dv_ = v_grid[iv] - u_i
                    quad = pre_x + cross*dv_ + inv11*dv_*dv_
                    if quad < 50.0:
                        f[iv, ix] += amp*np.exp(-0.5*quad)
    return f

def build_f(x_arr, U, x_grid, v_grid):
    rho, u, Pxx, Pp, L1, alpha, beta, gamma, Sxx, Sxv, Svv, Q = primitives(U)
    Sxx_safe = np.maximum(Sxx, 1e-30)
    dx_cell = x_arr[1] - x_arr[0]
    f = build_f_periodic(x_arr, u, Sxx_safe, Sxv, Svv, rho, dx_cell, x_grid, v_grid)
    return f, u, Sxv, Sxx_safe, Svv, rho, alpha, beta, gamma
