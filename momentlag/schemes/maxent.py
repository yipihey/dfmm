"""
Polynomial-exponent maximum-entropy fourth-moment closure.

Replaces the default Wick-Gaussian closure
   <(v-u)^4>_Wick = 3 (P_xx/rho)^2
with a closure derived from a positive-definite distribution
   f(v) ∝ exp[-c_2 (v-u)^2/2 - c_3 (v-u)^3/6 - c_4 (v-u)^4/24]

with c_2, c_3, c_4 chosen so that
   <1> = 1, <(v-u)^2> = Sigma_vv = P_xx/rho, <(v-u)^3> = Q/rho.

Working in standardized variables xi = (v-u)/sqrt(Sigma_vv), define
  A_2 = c_2*Sigma_vv - 1    (excess inverse variance over Gaussian)
  A_3 = c_3*Sigma_vv^{3/2}/6
  A_4 = c_4*Sigma_vv^2/24
so the exponent becomes
   Q_v(xi) = xi^2/2 + A_2 xi^2/2 + A_3 xi^3 + A_4 xi^4
and the Gaussian limit is A_2 = A_3 = A_4 = 0.

The constraint equations are
   <xi^2>(A) = 1,  <xi^3>(A) = s = Q / (rho * Sigma_vv^{3/2}),  normalization auto.
Three constraints in three unknowns A = (A_2, A_3, A_4).  Newton's method.

Once A converged, the kurtosis is
   kappa(A) = <xi^4>(A)
and the corrected fourth moment is
   <(v-u)^4>_exp = kappa * Sigma_vv^2
"""
import numpy as np
import numba as nb

# ---------- Gauss-Hermite quadrature (weight e^{-xi^2/2}) ----------
# Use scipy to generate physicist's Hermite (weight e^{-xi^2}) and rescale.
from scipy.special import roots_hermite
N_GH = 24
_GH_xi_phys, _GH_w_phys = roots_hermite(N_GH)
# physicist's: nodes are roots of H_n, weight function e^{-xi^2}.
# To convert to "probabilist's" with weight e^{-xi^2/2}, substitute xi -> xi/sqrt(2):
GH_XI = _GH_xi_phys * np.sqrt(2.0)        # nodes for weight e^{-xi^2/2}
GH_W  = _GH_w_phys * np.sqrt(2.0)         # weights so that  sum(w_i g(xi_i)) = int g(xi) e^{-xi^2/2} dxi

# Sanity check: int e^{-xi^2/2} dxi = sqrt(2 pi).
# Let's verify numerically:
_check = float(np.sum(GH_W))
assert abs(_check - np.sqrt(2*np.pi)) < 1e-12, f'GH nodes wrong: {_check} vs {np.sqrt(2*np.pi)}'
# int xi^2 e^{-xi^2/2} dxi = sqrt(2 pi):
_check2 = float(np.sum(GH_W * GH_XI**2))
assert abs(_check2 - np.sqrt(2*np.pi)) < 1e-12

CSCOEF = 8.0   # larger than 13-moment value (3+sqrt(6)~5.45) for safety with variable kappa closure


# ---------- Max-entropy closure: solve for A = (A2, A3, A4) given target s ----------

@nb.njit(cache=True, fastmath=False, inline='always')
def _moments_at(A2, A3, A4, xi, w):
    Z = 0.0; m2 = 0.0; m3 = 0.0; m4 = 0.0
    for i in range(xi.shape[0]):
        x = xi[i]
        delta = 0.5*A2*x*x + A3*x*x*x + A4*x*x*x*x
        if delta < -50.0: delta = -50.0
        elif delta > 200.0:
            continue
        e = w[i] * np.exp(-delta)
        Z  += e
        m2 += e * x*x
        m3 += e * x*x*x
        m4 += e * x*x*x*x
    if Z < 1e-200:
        return 1e-200, 1.0, 0.0, 3.0      # bail with Gaussian-equivalent moments
    return Z, m2/Z, m3/Z, m4/Z


@nb.njit(cache=True, fastmath=False)
def solve_maxent(s_target, xi, w, max_iter=12, tol=1e-7):
    """Find (A2, A3, A4) such that <xi^2>=1, <xi^3>=s_target, with A4 set
    by integrability heuristic A4 = max(A3^2/2, 1e-4).  Returns (A2, A3, A4, kappa)."""
    # Hard cap on s_target to stay inside the realizability cone of the
    # cubic+quartic exponential family.
    if s_target >  1.4: s_target =  1.4
    if s_target < -1.4: s_target = -1.4

    A2 = 0.0; A3 = 0.0
    A4 = max(A3*A3*0.5, 1e-4)
    Z, m2, m3, m4 = _moments_at(A2, A3, A4, xi, w)
    if not (m2 > 0.0):
        return 0.0, 0.0, 1e-4, 3.0   # bail to Gaussian

    for it in range(max_iter):
        f0 = m2 - 1.0
        f1 = m3 - s_target
        if f0*f0 + f1*f1 < tol*tol:
            return A2, A3, A4, m4

        eps = 1e-4
        A4p = max((A3)*(A3)*0.5, 1e-4)
        Z2, m2p, m3p, _ = _moments_at(A2+eps, A3, A4p, xi, w)
        if not (m2p > 0.0): return 0.0, 0.0, 1e-4, 3.0
        df0_dA2 = (m2p - m2)/eps
        df1_dA2 = (m3p - m3)/eps

        A4q = max((A3+eps)*(A3+eps)*0.5, 1e-4)
        Z3, m2p, m3p, _ = _moments_at(A2, A3+eps, A4q, xi, w)
        if not (m2p > 0.0): return 0.0, 0.0, 1e-4, 3.0
        df0_dA3 = (m2p - m2)/eps
        df1_dA3 = (m3p - m3)/eps

        det = df0_dA2*df1_dA3 - df0_dA3*df1_dA2
        # NaN-safe: NaN comparisons are False, so this catches NaN det too
        if not (abs(det) > 1e-10):
            return 0.0, 0.0, 1e-4, 3.0

        dA2 = (-df1_dA3*f0 + df0_dA3*f1)/det
        dA3 = ( df1_dA2*f0 - df0_dA2*f1)/det
        # NaN-safe step size cap
        if not (abs(dA2) < 100.0 and abs(dA3) < 100.0):
            return 0.0, 0.0, 1e-4, 3.0

        damp = 1.0
        if abs(dA2) > 0.5: damp = min(damp, 0.5/abs(dA2))
        if abs(dA3) > 0.5: damp = min(damp, 0.5/abs(dA3))
        A2 += damp*dA2
        A3 += damp*dA3
        if A2 < -0.9: A2 = -0.9
        if A2 >  5.0: A2 =  5.0
        if A3 >  3.0: A3 =  3.0
        if A3 < -3.0: A3 = -3.0
        A4 = max(A3*A3*0.5, 1e-4)
        Z, m2, m3, m4 = _moments_at(A2, A3, A4, xi, w)
        if not (m2 > 0.0):
            return 0.0, 0.0, 1e-4, 3.0
    return A2, A3, A4, m4


# ---------- The full step ----------
@nb.njit(cache=True, fastmath=True)
def hll_step_periodic(U, dx, dt, tau, GH_XI, GH_W):
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
    Sigma_vv = np.maximum(Pxx, 1e-30)/np.maximum(rho, 1e-30)

    # Compute kappa per cell via max-entropy fit.
    # NOTE: bound kappa to a safe range to prevent unphysical oscillations
    # in the post-shell-crossing regime, where the cubic+quartic exponential
    # family realizability boundary is crossed.  At |s| > 1.4 the closure
    # becomes ambiguous; we conservatively bound kappa near the Wick value 3.
    kappa = np.empty(N)
    A2_arr = np.empty(N); A3_arr = np.empty(N); A4_arr = np.empty(N)
    for i in range(N):
        denom = rho[i] * np.power(max(Sigma_vv[i], 1e-30), 1.5)
        if denom < 1e-30: denom = 1e-30
        s = Q[i]/denom
        if s >  1.4: s =  1.4
        if s < -1.4: s = -1.4
        A2, A3, A4, k = solve_maxent(s, GH_XI, GH_W)
        # Bound kappa to [1, 5] to prevent characteristic-speed explosion
        # when the cell is near closure failure
        if k < 1.0: k = 1.0
        if k > 5.0: k = 5.0
        kappa[i] = k
        A2_arr[i] = A2; A3_arr[i] = A3; A4_arr[i] = A4

    # M4 from max-entropy:  <(v-u)^4>_exp = kappa * Sigma_vv^2
    # M4_full = rho u^4 + 6 u^2 Pxx + 4 u Q + rho * kappa * Sigma_vv^2
    #        = rho u^4 + 6 u^2 Pxx + 4 u Q + kappa * Pxx^2/rho
    M4 = rho*u**4 + 6.0*u*u*Pxx + 4.0*u*Q + kappa*Pxx*Pxx/np.maximum(rho, 1e-30)

    Fleft = np.empty((n_fields, N))
    for i in range(N):
        l = (i-1) % N
        rho_L = rho[l]; u_L = u[l]; Pxx_L = Pxx[l]; Pp_L = Pp[l]; L1_L = L1[l]
        a_L = alpha[l]; b_L = beta[l]; Q_L = Q[l]; cs_L = cs[l]; M4_L = M4[l]
        rho_R = rho[i]; u_R = u[i]; Pxx_R = Pxx[i]; Pp_R = Pp[i]; L1_R = L1[i]
        a_R = alpha[i]; b_R = beta[i]; Q_R = Q[i]; cs_R = cs[i]; M4_R = M4[i]

        SL = min(u_L - cs_L, u_R - cs_R)
        SR = max(u_L + cs_L, u_R + cs_R)

        FL0 = rho_L*u_L
        FL1 = rho_L*u_L*u_L + Pxx_L
        FL2 = rho_L*u_L*u_L*u_L + 3.0*u_L*Pxx_L + Q_L
        FL3 = u_L*Pp_L
        FL4 = rho_L*L1_L*u_L
        FL5 = rho_L*a_L*u_L
        FL6 = rho_L*b_L*u_L
        FL7 = M4_L
        FR0 = rho_R*u_R
        FR1 = rho_R*u_R*u_R + Pxx_R
        FR2 = rho_R*u_R*u_R*u_R + 3.0*u_R*Pxx_R + Q_R
        FR3 = u_R*Pp_R
        FR4 = rho_R*L1_R*u_R
        FR5 = rho_R*a_R*u_R
        FR6 = rho_R*b_R*u_R
        FR7 = M4_R

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

    # Liouville sources for alpha and beta (same as Step 3)
    for i in range(N):
        Sigma_vv_i = Pxx[i]/max(rho[i], 1e-30)
        a = alpha[i]; b = beta[i]
        gamma2 = Sigma_vv_i - b*b
        ip2 = (i+1) % N; im2 = (i-1) % N
        dudx_i = (u[ip2] - u[im2])/(2.0*dx)
        Unew[5, i] += dt*rho[i]*b
        a_safe = a if a > 1e-15 else 1e-15
        Unew[6, i] += dt*rho[i]*(gamma2/a_safe - dudx_i*b)

    # BGK
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

        # Pressure positivity floor (essential for stability of variable-closure system)
        if Pxx_n < 1e-6: Pxx_n = 1e-6
        if Pp_n  < 1e-6: Pp_n  = 1e-6

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


@nb.njit(cache=True, fastmath=True)
def max_signal_speed(U):
    rho = U[0]; u = U[1]/rho
    Pxx = U[2] - rho*u*u
    cs = np.sqrt(CSCOEF*np.maximum(Pxx, 1e-30)/np.maximum(rho, 1e-30))
    return float(np.max(np.abs(u) + cs))


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
    return rho, u, Pxx, Pp, L1, alpha, beta, Sigma_vv, Q


def diagnose_kappa(U):
    """For visualization: per-cell skewness s and the corresponding kappa from max-entropy."""
    rho, u, Pxx, Pp, L1, alpha, beta, Svv, Q = primitives(U)
    s = Q/(rho * Svv**1.5 + 1e-30)
    s_clip = np.clip(s, -3.0, 3.0)
    kappa = np.empty_like(s)
    A2_arr = np.empty_like(s); A3_arr = np.empty_like(s); A4_arr = np.empty_like(s)
    for i in range(len(s)):
        A2, A3, A4, k = solve_maxent(s_clip[i], GH_XI, GH_W)
        kappa[i] = k
        A2_arr[i] = A2; A3_arr[i] = A3; A4_arr[i] = A4
    return s, kappa, A2_arr, A3_arr, A4_arr


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
    alpha0 = np.full(N, sigma_x0); beta0 = np.zeros(N)
    Q0 = np.zeros(N)
    M30 = rho*u*u*u + 3.0*u*Pxx0 + Q0
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
        U = hll_step_periodic(U, dx, dt, tau, GH_XI, GH_W)
        t += dt; nsteps += 1
        if t >= snap_times[snap_idx] - 1e-12:
            snaps.append((t, x.copy(), U.copy())); snap_idx += 1
    return snaps, nsteps


# ---------- Phase-space rendering with NON-Gaussian f(v) per cell ----------
@nb.njit(cache=True, fastmath=True, parallel=True)
def build_f_periodic_maxent(x_arr, u_arr, alpha_arr, beta_arr, Sigma_vv_arr, rho,
                             A2_arr, A3_arr, A4_arr,
                             dx_cell, x_grid, v_grid):
    """Render phase space using the max-entropy v-marginal at each cell.
    Position dimension still uses the Cholesky parametrization (Gaussian in x, correlated to v)."""
    Nx = len(x_grid); Nv = len(v_grid); Nc = len(x_arr)
    f = np.zeros((Nv, Nx))
    inv_two_pi = 1.0/(2.0*np.pi)
    for i in nb.prange(Nc):
        a = alpha_arr[i]; b = beta_arr[i]
        Svv = Sigma_vv_arr[i]
        Sxx = a*a; Sxv = a*b
        # Position-space Gaussian uses (Sxx, Sxv, Svv);
        # v-marginal at fixed x' is a polynomial-exponent correction.
        # For simplicity, we do the joint Gaussian (Step 3 style) and then
        # multiply each cell's contribution by exp(-A2 xi^2/2 - A3 xi^3 - A4 xi^4)
        # in the v-direction with xi = (v-u)/sqrt(Svv), normalized by the cell's Z.
        det = Sxx*Svv - Sxv*Sxv
        if det < 1e-30: det = 1e-30
        inv00 = Svv/det; inv11 = Sxx/det; inv01 = -Sxv/det
        amp = rho[i]*dx_cell*inv_two_pi/np.sqrt(det)
        u_i = u_arr[i]
        sqrtSvv = np.sqrt(max(Svv, 1e-30))
        A2 = A2_arr[i]; A3 = A3_arr[i]; A4 = A4_arr[i]

        # Compute v-marginal normalization Z by Gauss-Hermite
        # (already encoded in A2,A3,A4 via solve_maxent; use it to normalize the perturbation)
        for shift in (-1.0, 0.0, 1.0):
            x_c = x_arr[i] + shift
            for ix in range(Nx):
                dx_ = x_grid[ix] - x_c
                if dx_*dx_ > 25.0*Sxx: continue
                pre_x = inv00*dx_*dx_; cross = 2.0*inv01*dx_
                for iv in range(Nv):
                    dv_ = v_grid[iv] - u_i
                    quad = pre_x + cross*dv_ + inv11*dv_*dv_
                    if quad > 100.0: continue
                    # Polynomial-exponent correction in standardized v
                    xi = dv_/sqrtSvv
                    delta = 0.5*A2*xi*xi + A3*xi*xi*xi + A4*xi*xi*xi*xi
                    if delta > 50.0: continue
                    f[iv, ix] += amp*np.exp(-0.5*quad - delta)
    return f


def build_f(x_arr, U, x_grid, v_grid):
    rho, u, Pxx, Pp, L1, alpha, beta, Svv, Q = primitives(U)
    s = Q/(rho * Svv**1.5 + 1e-30)
    s_clip = np.clip(s, -3.0, 3.0)
    A2 = np.empty_like(s); A3 = np.empty_like(s); A4 = np.empty_like(s)
    for i in range(len(s)):
        A2[i], A3[i], A4[i], _ = solve_maxent(s_clip[i], GH_XI, GH_W)
    dx_cell = x_arr[1] - x_arr[0]
    f = build_f_periodic_maxent(x_arr, u, alpha, beta, Svv, rho, A2, A3, A4,
                                 dx_cell, x_grid, v_grid)
    return f, u, alpha, beta, Svv, rho, A2, A3, A4

