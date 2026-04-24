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

CSCOEF = 3.0 + np.sqrt(6.0)            # 13-moment max-eigenvalue^2 coefficient


def primitives(U):
    rho   = U[0]
    u     = U[1]/rho
    Pxx   = U[2] - rho*u*u
    Pp    = U[3]
    L1    = U[4]/rho
    alpha = U[5]/rho
    beta  = U[6]/rho
    M3    = U[7]
    Q     = M3 - rho*u*u*u - 3.0*u*Pxx
    Sigma_vv = np.maximum(Pxx, 1e-30)/np.maximum(rho, 1e-30)
    Sigma_xx = alpha*alpha
    Sigma_xv = alpha*beta
    gamma2   = np.maximum(Sigma_vv - beta*beta, 0.0)
    gamma    = np.sqrt(gamma2)
    return rho, u, Pxx, Pp, L1, alpha, beta, gamma, Sigma_xx, Sigma_xv, Sigma_vv, Q


@nb.njit(cache=True, fastmath=True)
def max_signal_speed(U):
    rho = U[0]; u = U[1]/rho
    Pxx = U[2] - rho*u*u
    cs = np.sqrt(CSCOEF*np.maximum(Pxx, 1e-30)/np.maximum(rho, 1e-30))
    return float(np.max(np.abs(u) + cs))


@nb.njit(cache=True, fastmath=True)
def hll_step_periodic(U, dx, dt, tau):
    n_fields, N = U.shape
    Unew = np.empty_like(U)

    rho   = U[0]
    u     = U[1]/rho
    Pxx   = U[2] - rho*u*u
    Pp    = U[3]
    L1    = U[4]/rho
    alpha = U[5]/rho
    beta  = U[6]/rho
    M3    = U[7]
    Q     = M3 - rho*u*u*u - 3.0*u*Pxx
    cs    = np.sqrt(CSCOEF*np.maximum(Pxx, 1e-30)/np.maximum(rho, 1e-30))

    Fleft = np.empty((n_fields, N))
    for i in range(N):
        l = (i-1) % N
        rho_L = rho[l]; u_L = u[l]; Pxx_L = Pxx[l]; Pp_L = Pp[l]; L1_L = L1[l]
        a_L = alpha[l]; b_L = beta[l]; Q_L = Q[l]; cs_L = cs[l]
        rho_R = rho[i]; u_R = u[i]; Pxx_R = Pxx[i]; Pp_R = Pp[i]; L1_R = L1[i]
        a_R = alpha[i]; b_R = beta[i]; Q_R = Q[i]; cs_R = cs[i]

        SL = min(u_L - cs_L, u_R - cs_R)
        SR = max(u_L + cs_L, u_R + cs_R)

        FL0 = rho_L*u_L
        FL1 = rho_L*u_L*u_L + Pxx_L
        FL2 = rho_L*u_L*u_L*u_L + 3.0*u_L*Pxx_L + Q_L
        FL3 = u_L*Pp_L
        FL4 = rho_L*L1_L*u_L
        FL5 = rho_L*a_L*u_L
        FL6 = rho_L*b_L*u_L
        FL7 = rho_L*u_L**4 + 6.0*u_L*u_L*Pxx_L + 4.0*u_L*Q_L + 3.0*Pxx_L*Pxx_L/rho_L
        FR0 = rho_R*u_R
        FR1 = rho_R*u_R*u_R + Pxx_R
        FR2 = rho_R*u_R*u_R*u_R + 3.0*u_R*Pxx_R + Q_R
        FR3 = u_R*Pp_R
        FR4 = rho_R*L1_R*u_R
        FR5 = rho_R*a_R*u_R
        FR6 = rho_R*b_R*u_R
        FR7 = rho_R*u_R**4 + 6.0*u_R*u_R*Pxx_R + 4.0*u_R*Q_R + 3.0*Pxx_R*Pxx_R/rho_R

        if SL >= 0.0:
            Fleft[0,i]=FL0; Fleft[1,i]=FL1; Fleft[2,i]=FL2; Fleft[3,i]=FL3
            Fleft[4,i]=FL4; Fleft[5,i]=FL5; Fleft[6,i]=FL6; Fleft[7,i]=FL7
        elif SR <= 0.0:
            Fleft[0,i]=FR0; Fleft[1,i]=FR1; Fleft[2,i]=FR2; Fleft[3,i]=FR3
            Fleft[4,i]=FR4; Fleft[5,i]=FR5; Fleft[6,i]=FR6; Fleft[7,i]=FR7
        else:
            invDS = 1.0/(SR - SL + 1e-30)
            Fleft[0,i] = (SR*FL0 - SL*FR0 + SL*SR*(U[0,i] - U[0,l]))*invDS
            Fleft[1,i] = (SR*FL1 - SL*FR1 + SL*SR*(U[1,i] - U[1,l]))*invDS
            Fleft[2,i] = (SR*FL2 - SL*FR2 + SL*SR*(U[2,i] - U[2,l]))*invDS
            Fleft[3,i] = (SR*FL3 - SL*FR3 + SL*SR*(U[3,i] - U[3,l]))*invDS
            Fleft[4,i] = (SR*FL4 - SL*FR4 + SL*SR*(U[4,i] - U[4,l]))*invDS
            Fleft[5,i] = (SR*FL5 - SL*FR5 + SL*SR*(U[5,i] - U[5,l]))*invDS
            Fleft[6,i] = (SR*FL6 - SL*FR6 + SL*SR*(U[6,i] - U[6,l]))*invDS
            Fleft[7,i] = (SR*FL7 - SL*FR7 + SL*SR*(U[7,i] - U[7,l]))*invDS

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
        Unew[5, i] += dt*rho[i]*b
        a_safe = a if a > 1e-15 else 1e-15
        Unew[6, i] += dt*rho[i]*(gamma2_signed/a_safe - dudx_i*b)

    # BGK relaxation
    decay = np.exp(-dt/tau)
    for i in range(N):
        rho_n = Unew[0, i]
        u_n   = Unew[1, i]/rho_n
        Pxx_n = Unew[2, i] - rho_n*u_n*u_n
        Pp_n  = Unew[3, i]
        a_n   = Unew[5, i]/rho_n
        b_n   = Unew[6, i]/rho_n
        M3_n  = Unew[7, i]
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

        Unew[2, i] = rho_n*u_n*u_n + Pxx_new
        Unew[3, i] = Pp_new
        Unew[6, i] = rho_n*b_new
        Unew[7, i] = rho_n*u_n*u_n*u_n + 3.0*u_n*Pxx_new + Q_new
        # alpha unchanged by collisions

    return Unew


@nb.njit(cache=True, fastmath=True)
def hll_step_transmissive(U, dx, dt, tau):
    n_fields, N = U.shape
    Unew = np.empty_like(U)

    rho   = U[0]; u = U[1]/rho
    Pxx   = U[2] - rho*u*u; Pp = U[3]; L1 = U[4]/rho
    alpha = U[5]/rho; beta = U[6]/rho
    M3    = U[7]; Q = M3 - rho*u*u*u - 3.0*u*Pxx
    cs    = np.sqrt(CSCOEF*np.maximum(Pxx, 1e-30)/np.maximum(rho, 1e-30))

    Fleft = np.empty((n_fields, N+1))
    for i in range(N+1):
        if i == 0: l = 0; r = 0
        elif i == N: l = N-1; r = N-1
        else: l = i-1; r = i
        rho_L = rho[l]; u_L = u[l]; Pxx_L = Pxx[l]; Pp_L = Pp[l]; L1_L = L1[l]
        a_L = alpha[l]; b_L = beta[l]; Q_L = Q[l]; cs_L = cs[l]
        rho_R = rho[r]; u_R = u[r]; Pxx_R = Pxx[r]; Pp_R = Pp[r]; L1_R = L1[r]
        a_R = alpha[r]; b_R = beta[r]; Q_R = Q[r]; cs_R = cs[r]
        SL = min(u_L - cs_L, u_R - cs_R)
        SR = max(u_L + cs_L, u_R + cs_R)
        FL0 = rho_L*u_L; FL1 = rho_L*u_L*u_L + Pxx_L
        FL2 = rho_L*u_L*u_L*u_L + 3.0*u_L*Pxx_L + Q_L
        FL3 = u_L*Pp_L; FL4 = rho_L*L1_L*u_L
        FL5 = rho_L*a_L*u_L; FL6 = rho_L*b_L*u_L
        FL7 = rho_L*u_L**4 + 6.0*u_L*u_L*Pxx_L + 4.0*u_L*Q_L + 3.0*Pxx_L*Pxx_L/rho_L
        FR0 = rho_R*u_R; FR1 = rho_R*u_R*u_R + Pxx_R
        FR2 = rho_R*u_R*u_R*u_R + 3.0*u_R*Pxx_R + Q_R
        FR3 = u_R*Pp_R; FR4 = rho_R*L1_R*u_R
        FR5 = rho_R*a_R*u_R; FR6 = rho_R*b_R*u_R
        FR7 = rho_R*u_R**4 + 6.0*u_R*u_R*Pxx_R + 4.0*u_R*Q_R + 3.0*Pxx_R*Pxx_R/rho_R
        if SL >= 0.0:
            Fleft[0,i]=FL0; Fleft[1,i]=FL1; Fleft[2,i]=FL2; Fleft[3,i]=FL3
            Fleft[4,i]=FL4; Fleft[5,i]=FL5; Fleft[6,i]=FL6; Fleft[7,i]=FL7
        elif SR <= 0.0:
            Fleft[0,i]=FR0; Fleft[1,i]=FR1; Fleft[2,i]=FR2; Fleft[3,i]=FR3
            Fleft[4,i]=FR4; Fleft[5,i]=FR5; Fleft[6,i]=FR6; Fleft[7,i]=FR7
        else:
            invDS = 1.0/(SR - SL + 1e-30)
            Fleft[0,i] = (SR*FL0 - SL*FR0 + SL*SR*(U[0,r] - U[0,l]))*invDS
            Fleft[1,i] = (SR*FL1 - SL*FR1 + SL*SR*(U[1,r] - U[1,l]))*invDS
            Fleft[2,i] = (SR*FL2 - SL*FR2 + SL*SR*(U[2,r] - U[2,l]))*invDS
            Fleft[3,i] = (SR*FL3 - SL*FR3 + SL*SR*(U[3,r] - U[3,l]))*invDS
            Fleft[4,i] = (SR*FL4 - SL*FR4 + SL*SR*(U[4,r] - U[4,l]))*invDS
            Fleft[5,i] = (SR*FL5 - SL*FR5 + SL*SR*(U[5,r] - U[5,l]))*invDS
            Fleft[6,i] = (SR*FL6 - SL*FR6 + SL*SR*(U[6,r] - U[6,l]))*invDS
            Fleft[7,i] = (SR*FL7 - SL*FR7 + SL*SR*(U[7,r] - U[7,l]))*invDS
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
        Unew[5, i] += dt*rho[i]*b
        a_safe = a if a > 1e-15 else 1e-15
        Unew[6, i] += dt*rho[i]*(gamma2/a_safe - dudx_i*b)
    decay = np.exp(-dt/tau)
    for i in range(N):
        rho_n = Unew[0, i]; u_n = Unew[1, i]/rho_n
        Pxx_n = Unew[2, i] - rho_n*u_n*u_n; Pp_n = Unew[3, i]
        a_n = Unew[5, i]/rho_n; b_n = Unew[6, i]/rho_n
        M3_n = Unew[7, i]; Q_n = M3_n - rho_n*u_n*u_n*u_n - 3.0*u_n*Pxx_n
        P_iso = (Pxx_n + 2.0*Pp_n)/3.0
        Pxx_new = P_iso + (Pxx_n - P_iso)*decay
        Pp_new  = P_iso + (Pp_n  - P_iso)*decay
        b_new = b_n*decay
        Q_new = Q_n*decay
        Sigma_vv_new = max(Pxx_new, 1e-30)/max(rho_n, 1e-30)
        beta_max = 0.999*np.sqrt(Sigma_vv_new)
        if b_new > beta_max:    b_new = beta_max
        elif b_new < -beta_max: b_new = -beta_max
        Unew[2, i] = rho_n*u_n*u_n + Pxx_new
        Unew[3, i] = Pp_new
        Unew[6, i] = rho_n*b_new
        Unew[7, i] = rho_n*u_n*u_n*u_n + 3.0*u_n*Pxx_new + Q_new
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
