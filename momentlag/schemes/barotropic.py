"""
EOS variants for the steady-shock test.

Three regimes:
  (a) Adiabatic (full energy equation): our existing 8-field scheme with
      gamma_eff set by the moment tower. Rankine-Hugoniot gives a
      Mach-dependent compression ratio with saturation at (gamma+1)/(gamma-1).
  (b) Polytropic barotropic P = K rho^Gamma: pressure algebraically slaved
      to density. Energy equation redundant. Compression ratio solves
      implicit equation M^2 (1 - 1/r) = (r^Gamma - 1)/Gamma, unbounded as
      Gamma -> 1.
  (c) Isothermal P = c_T^2 rho: special case Gamma = 1 with compression
      ratio r = M^2 exactly.

For cases (b) and (c) we implement a REDUCED 5-field moment tower:
  (rho, rho*u, rho*L1, rho*alpha, rho*beta)
with P algebraically closed by P = K rho^Gamma at each flux evaluation.
This demonstrates the structural point that barotropic + our scheme
gives "full closure" at the first-moment level.
"""
import numpy as np
import numba as nb
from scipy.optimize import brentq

CSCOEF_BARO = 3.0   # barotropic wave speed factor (no heat flux contribution)


# ---------- Theoretical compression ratios ----------

def compression_ratio_adiabatic(M1, gamma=5.0/3.0):
    """Rankine-Hugoniot density compression ratio for adiabatic shock."""
    return (gamma + 1) * M1**2 / ((gamma - 1) * M1**2 + 2)

def compression_ratio_isothermal(M1):
    """Exact: r = M^2 for isothermal shock."""
    return M1**2

def compression_ratio_polytropic(M1, Gamma):
    """Solve M^2 (1 - 1/r) = (r^Gamma - 1)/Gamma for r > 1."""
    if abs(Gamma - 1) < 1e-6:
        return M1**2
    def f(r):
        return M1**2 * (1.0 - 1.0/r) - (r**Gamma - 1.0)/Gamma
    # r > 1 always; solve in bracket.
    # For Gamma > 1, RHS grows faster than LHS eventually, so f crosses zero.
    # Upper bound: for Gamma ~ 1 need very large r; for Gamma ~ 5/3 need r < (Gamma+1)/(Gamma-1) well.
    r_lo, r_hi = 1.0 + 1e-6, 1e6
    if f(r_hi) > 0:
        # No solution found in range -- use loose upper bound
        r_hi = 1e12
    return brentq(f, r_lo, r_hi)


# ---------- Barotropic / polytropic moment scheme (5 fields) ----------

@nb.njit(cache=True, fastmath=False)
def _baro_pressure(rho, K, Gamma):
    """P = K rho^Gamma, handling small rho."""
    return K * max(rho, 1e-30)**Gamma


@nb.njit(cache=True, fastmath=False)
def baro_max_signal_speed(U, K, Gamma):
    rho = U[0]
    u = U[1]/np.maximum(rho, 1e-30)
    cs = np.empty_like(rho)
    for i in range(len(rho)):
        P = _baro_pressure(rho[i], K, Gamma)
        cs[i] = np.sqrt(CSCOEF_BARO * P / max(rho[i], 1e-30))
    return float(np.max(np.abs(u) + cs))


@nb.njit(cache=True, fastmath=False)
def baro_hll_step(U, dx, dt, tau, K, Gamma, boundary_mode, U_inflow=None):
    """Barotropic 5-field HLL step.
    Fields: U[0]=rho, U[1]=rho u, U[2]=rho L1, U[3]=rho alpha, U[4]=rho beta.
    P is algebraically closed by P = K rho^Gamma.
    Momentum flux: rho u^2 + P; passive scalars advect with u.
    Liouville sources for alpha, beta (same as before but with Sigma_vv = P/rho).
    """
    n_fields, N = U.shape
    Unew = np.empty_like(U)

    rho = U[0]
    u   = U[1]/np.maximum(rho, 1e-30)
    L1  = U[2]/np.maximum(rho, 1e-30)
    alpha = U[3]/np.maximum(rho, 1e-30)
    beta  = U[4]/np.maximum(rho, 1e-30)
    # Pressure and sound speed per cell
    P = np.empty_like(rho); cs = np.empty_like(rho); Svv = np.empty_like(rho)
    for i in range(N):
        P[i] = _baro_pressure(rho[i], K, Gamma)
        Svv[i] = P[i]/max(rho[i], 1e-30)
        cs[i] = np.sqrt(CSCOEF_BARO * Svv[i])

    # Compute fluxes at each left face
    if boundary_mode == 0:   # periodic
        offset_range = N
    else:                     # transmissive (ghost = edge values)
        offset_range = N + 1
    Fleft = np.empty((n_fields, offset_range))
    for i in range(offset_range):
        if boundary_mode == 0:
            l = (i - 1) % N; r = i
        else:
            if i == 0: l = 0; r = 0
            elif i == N: l = N-1; r = N-1
            else: l = i-1; r = i
        rho_L = rho[l]; u_L = u[l]; P_L = P[l]; cs_L = cs[l]
        L1_L = L1[l]; a_L = alpha[l]; b_L = beta[l]
        rho_R = rho[r]; u_R = u[r]; P_R = P[r]; cs_R = cs[r]
        L1_R = L1[r]; a_R = alpha[r]; b_R = beta[r]
        SL = min(u_L - cs_L, u_R - cs_R)
        SR = max(u_L + cs_L, u_R + cs_R)
        FL0 = rho_L*u_L
        FL1 = rho_L*u_L*u_L + P_L
        FL2 = rho_L*L1_L*u_L
        FL3 = rho_L*a_L*u_L
        FL4 = rho_L*b_L*u_L
        FR0 = rho_R*u_R
        FR1 = rho_R*u_R*u_R + P_R
        FR2 = rho_R*L1_R*u_R
        FR3 = rho_R*a_R*u_R
        FR4 = rho_R*b_R*u_R
        if SL >= 0.0:
            Fleft[0,i]=FL0; Fleft[1,i]=FL1; Fleft[2,i]=FL2; Fleft[3,i]=FL3; Fleft[4,i]=FL4
        elif SR <= 0.0:
            Fleft[0,i]=FR0; Fleft[1,i]=FR1; Fleft[2,i]=FR2; Fleft[3,i]=FR3; Fleft[4,i]=FR4
        else:
            invDS = 1.0/(SR - SL + 1e-30)
            Fleft[0,i] = (SR*FL0 - SL*FR0 + SL*SR*(U[0,r]-U[0,l]))*invDS
            Fleft[1,i] = (SR*FL1 - SL*FR1 + SL*SR*(U[1,r]-U[1,l]))*invDS
            Fleft[2,i] = (SR*FL2 - SL*FR2 + SL*SR*(U[2,r]-U[2,l]))*invDS
            Fleft[3,i] = (SR*FL3 - SL*FR3 + SL*SR*(U[3,r]-U[3,l]))*invDS
            Fleft[4,i] = (SR*FL4 - SL*FR4 + SL*SR*(U[4,r]-U[4,l]))*invDS

    # Update
    inv_dx = 1.0/dx
    for i in range(N):
        if boundary_mode == 0:
            ip = (i+1) % N
            for k in range(n_fields):
                Unew[k, i] = U[k, i] - dt*inv_dx*(Fleft[k, ip] - Fleft[k, i])
        else:
            for k in range(n_fields):
                Unew[k, i] = U[k, i] - dt*inv_dx*(Fleft[k, i+1] - Fleft[k, i])

    # Liouville sources for alpha, beta
    for i in range(N):
        a = alpha[i]; b = beta[i]
        gamma2 = Svv[i] - b*b
        if boundary_mode == 0:
            ip2 = (i+1) % N; im2 = (i-1) % N
            dudx = (u[ip2] - u[im2])/(2.0*dx)
        else:
            if i == 0:        dudx = (u[1] - u[0])/dx
            elif i == N-1:    dudx = (u[N-1] - u[N-2])/dx
            else:             dudx = (u[i+1] - u[i-1])/(2.0*dx)
        Unew[3, i] += dt*rho[i]*b
        a_safe = a if a > 1e-15 else 1e-15
        Unew[4, i] += dt*rho[i]*(gamma2/a_safe - dudx*b)

    # Realizability clip on beta (barotropic has exact Svv = K rho^Gamma/rho)
    for i in range(N):
        rho_n = Unew[0, i]
        b_n   = Unew[4, i]/max(rho_n, 1e-30)
        Svv_n = _baro_pressure(rho_n, K, Gamma)/max(rho_n, 1e-30)
        b_max = 0.999*np.sqrt(max(Svv_n, 1e-30))
        if b_n >  b_max: b_n =  b_max
        elif b_n < -b_max: b_n = -b_max
        Unew[4, i] = rho_n * b_n

    # Apply inflow BC (overwrite first 2 cells)
    if boundary_mode == 1 and U_inflow is not None:
        for k in range(n_fields):
            Unew[k, 0] = U_inflow[k]
            Unew[k, 1] = U_inflow[k]

    return Unew


def baro_primitives(U, K, Gamma):
    rho = U[0]
    u   = U[1]/np.maximum(rho, 1e-30)
    L1  = U[2]/np.maximum(rho, 1e-30)
    alpha = U[3]/np.maximum(rho, 1e-30)
    beta  = U[4]/np.maximum(rho, 1e-30)
    P = K * np.maximum(rho, 1e-30)**Gamma
    Svv = P/np.maximum(rho, 1e-30)
    gamma_ch = np.sqrt(np.maximum(Svv - beta**2, 0.0))
    return dict(rho=rho, u=u, P=P, L1=L1, alpha=alpha, beta=beta,
                Svv=Svv, gamma=gamma_ch)


def run_baro_shock(M1, Gamma, K=1.0, rho1=1.0, N=400, t_end=3.0, cfl=0.3,
                    sigma_x0=0.02):
    """Barotropic steady shock. c_s1^2 = Gamma * K * rho1^(Gamma-1), u1 = M1 * c_s1."""
    c_s1 = np.sqrt(Gamma * K * rho1**(Gamma-1))
    u1 = M1 * c_s1
    r = compression_ratio_polytropic(M1, Gamma)
    rho2 = rho1 * r
    u2 = u1/r
    x = np.linspace(0, 1, N)
    dx = x[1] - x[0]
    rho = np.where(x < 0.5, rho1, rho2)
    u   = np.where(x < 0.5, u1,   u2)
    U = np.zeros((5, N))
    U[0] = rho; U[1] = rho*u
    U[2] = rho*x
    U[3] = rho*sigma_x0
    U[4] = 0.0
    U_inflow = np.array([rho1, rho1*u1, 0.0, rho1*sigma_x0, 0.0])
    t = 0.0; nsteps = 0
    while t < t_end:
        smax = baro_max_signal_speed(U, K, Gamma)
        dt = min(cfl*dx/smax, t_end - t)
        if dt <= 0: break
        U = baro_hll_step(U, dx, dt, 1e10, K, Gamma, 1, U_inflow)   # tau=1e10 dummy
        t += dt; nsteps += 1
    return x, U, nsteps


