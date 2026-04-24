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
import numpy as np
import numba as nb

from ..schemes.cholesky import max_signal_speed, CSCOEF


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


@nb.njit(cache=True, fastmath=True)
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
            Fleft[0,i] = (SR*FL0 - SL*FR0 + SL*SR*(U[0,i]-U[0,l]))*invDS
            Fleft[1,i] = (SR*FL1 - SL*FR1 + SL*SR*(U[1,i]-U[1,l]))*invDS
            Fleft[2,i] = (SR*FL2 - SL*FR2 + SL*SR*(U[2,i]-U[2,l]))*invDS
            Fleft[3,i] = (SR*FL3 - SL*FR3 + SL*SR*(U[3,i]-U[3,l]))*invDS
            Fleft[4,i] = (SR*FL4 - SL*FR4 + SL*SR*(U[4,i]-U[4,l]))*invDS
            Fleft[5,i] = (SR*FL5 - SL*FR5 + SL*SR*(U[5,i]-U[5,l]))*invDS
            Fleft[6,i] = (SR*FL6 - SL*FR6 + SL*SR*(U[6,i]-U[6,l]))*invDS
            Fleft[7,i] = (SR*FL7 - SL*FR7 + SL*SR*(U[7,i]-U[7,l]))*invDS

    # Conservative update
    inv_dx = 1.0/dx
    for i in range(N):
        ip = (i+1) % N
        for k in range(n_fields):
            Unew[k, i] = U[k, i] - dt*inv_dx*(Fleft[k, ip] - Fleft[k, i])

    # Exact-exponential BGK relaxation
    decay = np.exp(-dt/tau)
    for i in range(N):
        rho_n = Unew[0, i]
        u_n   = Unew[1, i]/rho_n
        Pxx_n = Unew[2, i] - rho_n*u_n*u_n
        Pp_n  = Unew[3, i]
        M3_n  = Unew[7, i]
        Q_n   = M3_n - rho_n*u_n*u_n*u_n - 3.0*u_n*Pxx_n
        b_n   = Unew[6, i]/rho_n
        P_iso = (Pxx_n + 2.0*Pp_n)/3.0
        Pxx_new = P_iso + (Pxx_n - P_iso)*decay
        Pp_new  = P_iso + (Pp_n  - P_iso)*decay
        Q_new = Q_n * decay
        b_new = b_n * decay
        Svv_n = max(Pxx_new/rho_n, 1e-30)
        b_max = 0.999*np.sqrt(Svv_n)
        if b_new >  b_max: b_new =  b_max
        elif b_new < -b_max: b_new = -b_max
        Unew[2, i] = rho_n*u_n*u_n + Pxx_new
        Unew[3, i] = Pp_new
        Unew[6, i] = rho_n*b_new
        Unew[7, i] = rho_n*u_n*u_n*u_n + 3.0*u_n*Pxx_new + Q_new

    # Energy-conservative noise injection
    for i in range(N):
        rho_i = Unew[0, i]
        u_i   = Unew[1, i]/rho_i
        Exx_old = Unew[2, i]
        Pxx_old = Exx_old - rho_i*u_i*u_i
        Pp_old  = Unew[3, i]
        M3_old  = Unew[7, i]
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

        rhou_new = Unew[1, i] + delta_rhou
        u_new = rhou_new/rho_i

        Delta_KE_vol = u_i * delta_rhou + 0.5 * delta_rhou**2 / rho_i

        # Debit internal energy to preserve total energy
        Pxx_new = Pxx_old - (2.0/3.0)*Delta_KE_vol
        Pp_new  = Pp_old  - (2.0/3.0)*Delta_KE_vol
        if Pxx_new < 1e-8: Pxx_new = 1e-8
        if Pp_new  < 1e-8: Pp_new  = 1e-8

        Unew[1, i] = rhou_new
        Unew[2, i] = rho_i*u_new*u_new + Pxx_new
        Unew[3, i] = Pp_new
        Unew[7, i] = rho_i*u_new**3 + 3.0*u_new*Pxx_new + Q_old

    return Unew


def run_noise(U, t_end, save_times, C_A, C_B, ell_corr=2.0, cfl=0.3,
              tau=1e-3, seed=0):
    """Energy-conservative driver with calibrated noise injection.

    Parameters
    ----------
    U : ndarray, shape (8, N)
        Initial state.
    t_end : float
    save_times : array-like
    C_A, C_B : float
        Calibrated noise coefficients (typically loaded from
        data/noise_model_params.npz).
    ell_corr : float, default 2.0
        Spatial correlation length of the noise, in cells.
    cfl : float, default 0.3
    tau : float, default 1e-3
        BGK relaxation time.
    seed : int
        RNG seed for the Laplace draws.

    Returns
    -------
    snapshots : list of (float, ndarray)
        Matches the convention of `momentlag.integrate.run_to`.
    """
    dx = 1.0/U.shape[1]
    rng = np.random.default_rng(seed)
    t = 0.0
    snapshots = [(0.0, U.copy())]
    save_idx = 0
    while t < t_end and save_idx < len(save_times):
        next_save = save_times[save_idx]
        smax = max_signal_speed(U)
        dt = min(cfl*dx/smax, next_save - t, t_end - t)
        if dt <= 1e-14:
            snapshots.append((t, U.copy()))
            save_idx += 1
            continue
        eta_white = rng.laplace(scale=1.0/np.sqrt(2.0), size=U.shape[1])
        eta = smooth_gaussian_periodic(eta_white, ell_corr) if ell_corr > 0 \
              else eta_white
        U = hll_step_noise_econsv(U, dx, dt, tau, C_A, C_B, eta)
        t += dt
        if t >= next_save - 1e-12:
            snapshots.append((t, U.copy()))
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
