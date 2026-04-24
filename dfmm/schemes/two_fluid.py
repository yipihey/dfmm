"""
Step 5: Two-species moment-tower framework with pluggable cross-coupling kernels.

State: 16 fields per cell (8 per species, 2 species A and B).
  U[0..7]   = species A: rho, rho*u, E_xx, P_perp, rho*L1, rho*alpha, rho*beta, M3
  U[8..15]  = species B: same layout

Each species has its own self-BGK timescale (tau_AA, tau_BB) and its own
Wick-closure 13-moment system using the Step 3 Cholesky formulation.

Cross-coupling has its own physics kernel: given local state, it produces
two rates per cell (nu_p for momentum, nu_T for temperature).  The operator
relaxes u^A -> u^B (and vice versa) toward common u*, similarly for T,
with exact-exponential update that preserves total momentum and total energy.

Mass ratio m^A/m^B is explicit throughout.

This file contains the generic infrastructure.  Application-specific kernels
(Epstein, Coulomb, SIDM, hard-sphere) are in separate helper functions.
"""
import numpy as np
import numba as nb

# Species A occupies U[0..7]; species B occupies U[8..15]; within each
# 8-field block the field layout is shared with the single-fluid state
# (see dfmm.schemes._common).
from ._common import (CSCOEF, IDX_RHO, IDX_MOM, IDX_EXX, IDX_PP,
                      IDX_L1, IDX_ALPHA, IDX_BETA, IDX_M3)


# -------- Generic single-species primitives --------

@nb.njit(cache=True, fastmath=False, inline='always')
def _species_primitives(U_s):
    """Return (rho, u, P_xx, P_perp, L1, alpha, beta, Svv, Q) for one species."""
    rho   = U_s[IDX_RHO]
    u     = U_s[IDX_MOM]/max(rho, 1e-30)
    Pxx   = U_s[IDX_EXX] - rho*u*u
    Pp    = U_s[IDX_PP]
    L1    = U_s[IDX_L1]/max(rho, 1e-30)
    alpha = U_s[IDX_ALPHA]/max(rho, 1e-30)
    beta  = U_s[IDX_BETA]/max(rho, 1e-30)
    M3    = U_s[IDX_M3]
    Q     = M3 - rho*u*u*u - 3.0*u*Pxx
    Svv   = max(Pxx, 1e-30)/max(rho, 1e-30)
    return rho, u, Pxx, Pp, L1, alpha, beta, Svv, Q


def primitives(U, species='A', m=1.0):
    """Numpy version for diagnostics — returns arrays.
    Temperature is in energy units: T = P * m / rho = P/n."""
    off = 0 if species == 'A' else 8
    rho   = U[off + IDX_RHO]
    u     = U[off + IDX_MOM]/np.maximum(rho, 1e-30)
    Pxx   = U[off + IDX_EXX] - rho*u*u
    Pp    = U[off + IDX_PP]
    L1    = U[off + IDX_L1]/np.maximum(rho, 1e-30)
    alpha = U[off + IDX_ALPHA]/np.maximum(rho, 1e-30)
    beta  = U[off + IDX_BETA]/np.maximum(rho, 1e-30)
    M3    = U[off + IDX_M3]
    Q     = M3 - rho*u*u*u - 3.0*u*Pxx
    Svv   = np.maximum(Pxx, 1e-30)/np.maximum(rho, 1e-30)
    Piso  = (Pxx + 2.0*Pp)/3.0
    T     = Piso * m / np.maximum(rho, 1e-30)         # P = n T, n = rho/m  =>  T = P m/rho
    return dict(rho=rho, u=u, Pxx=Pxx, Pp=Pp, L1=L1, alpha=alpha, beta=beta,
                Svv=Svv, Q=Q, T=T, Piso=Piso)


# -------- HLL flux for one species; periodic or transmissive BC --------

@nb.njit(cache=True, fastmath=False)
def _hll_flux_one_species(Ul, Ur, cs_l, cs_r):
    """8-component HLL flux given left and right 8-field states and their wave speeds."""
    F = np.zeros(8)
    rho_L = Ul[IDX_RHO]; u_L = Ul[IDX_MOM]/max(rho_L, 1e-30)
    Pxx_L = Ul[IDX_EXX] - rho_L*u_L*u_L
    Pp_L  = Ul[IDX_PP]
    L1_L  = Ul[IDX_L1]/max(rho_L, 1e-30)
    a_L   = Ul[IDX_ALPHA]/max(rho_L, 1e-30)
    b_L   = Ul[IDX_BETA]/max(rho_L, 1e-30)
    M3_L  = Ul[IDX_M3]
    Q_L   = M3_L - rho_L*u_L*u_L*u_L - 3.0*u_L*Pxx_L

    rho_R = Ur[IDX_RHO]; u_R = Ur[IDX_MOM]/max(rho_R, 1e-30)
    Pxx_R = Ur[IDX_EXX] - rho_R*u_R*u_R
    Pp_R  = Ur[IDX_PP]
    L1_R  = Ur[IDX_L1]/max(rho_R, 1e-30)
    a_R   = Ur[IDX_ALPHA]/max(rho_R, 1e-30)
    b_R   = Ur[IDX_BETA]/max(rho_R, 1e-30)
    M3_R  = Ur[IDX_M3]
    Q_R   = M3_R - rho_R*u_R*u_R*u_R - 3.0*u_R*Pxx_R

    SL = min(u_L - cs_l, u_R - cs_r)
    SR = max(u_L + cs_l, u_R + cs_r)

    FL0 = rho_L*u_L
    FL1 = rho_L*u_L*u_L + Pxx_L
    FL2 = rho_L*u_L*u_L*u_L + 3.0*u_L*Pxx_L + Q_L
    FL3 = u_L*Pp_L
    FL4 = rho_L*L1_L*u_L
    FL5 = rho_L*a_L*u_L
    FL6 = rho_L*b_L*u_L
    # Wick fourth moment: <(v-u)^4> = 3 Svv^2, so M4 = rho u^4 + 6 u^2 Pxx + 4 u Q + 3 Pxx^2/rho
    FL7 = rho_L*u_L**4 + 6.0*u_L*u_L*Pxx_L + 4.0*u_L*Q_L + 3.0*Pxx_L*Pxx_L/max(rho_L, 1e-30)

    FR0 = rho_R*u_R
    FR1 = rho_R*u_R*u_R + Pxx_R
    FR2 = rho_R*u_R*u_R*u_R + 3.0*u_R*Pxx_R + Q_R
    FR3 = u_R*Pp_R
    FR4 = rho_R*L1_R*u_R
    FR5 = rho_R*a_R*u_R
    FR6 = rho_R*b_R*u_R
    FR7 = rho_R*u_R**4 + 6.0*u_R*u_R*Pxx_R + 4.0*u_R*Q_R + 3.0*Pxx_R*Pxx_R/max(rho_R, 1e-30)

    if SL >= 0.0:
        F[0]=FL0; F[1]=FL1; F[2]=FL2; F[3]=FL3; F[4]=FL4; F[5]=FL5; F[6]=FL6; F[7]=FL7
    elif SR <= 0.0:
        F[0]=FR0; F[1]=FR1; F[2]=FR2; F[3]=FR3; F[4]=FR4; F[5]=FR5; F[6]=FR6; F[7]=FR7
    else:
        invDS = 1.0/(SR - SL + 1e-30)
        F[0] = (SR*FL0 - SL*FR0 + SL*SR*(Ur[0]-Ul[0]))*invDS
        F[1] = (SR*FL1 - SL*FR1 + SL*SR*(Ur[1]-Ul[1]))*invDS
        F[2] = (SR*FL2 - SL*FR2 + SL*SR*(Ur[2]-Ul[2]))*invDS
        F[3] = (SR*FL3 - SL*FR3 + SL*SR*(Ur[3]-Ul[3]))*invDS
        F[4] = (SR*FL4 - SL*FR4 + SL*SR*(Ur[4]-Ul[4]))*invDS
        F[5] = (SR*FL5 - SL*FR5 + SL*SR*(Ur[5]-Ul[5]))*invDS
        F[6] = (SR*FL6 - SL*FR6 + SL*SR*(Ur[6]-Ul[6]))*invDS
        F[7] = (SR*FL7 - SL*FR7 + SL*SR*(Ur[7]-Ul[7]))*invDS
    return F


# -------- Single-species step (Steps 1-3 machinery) --------

@nb.njit(cache=True, fastmath=False)
def _species_step_periodic(U_full, off, dx, dt, tau_self, N):
    """Apply HLL fluxes + Liouville sources + self-BGK for one species (8-field block
    starting at offset `off` in the full 16-field array).  Updates U_full in place."""
    # Precompute primitives and signal speeds for this species
    rho   = np.empty(N); u = np.empty(N); Pxx = np.empty(N); Pp = np.empty(N)
    L1    = np.empty(N); alpha = np.empty(N); beta = np.empty(N); Q = np.empty(N)
    cs    = np.empty(N)
    for i in range(N):
        r, v, pxx, pp, l1, a, b, svv, q = _species_primitives(U_full[off:off+8, i])
        rho[i]=r; u[i]=v; Pxx[i]=pxx; Pp[i]=pp; L1[i]=l1
        alpha[i]=a; beta[i]=b; Q[i]=q
        cs[i] = np.sqrt(CSCOEF*max(pxx, 1e-30)/max(r, 1e-30))

    # HLL flux at left face of each cell (periodic)
    Fleft = np.empty((8, N))
    for i in range(N):
        l = (i-1) % N
        F = _hll_flux_one_species(U_full[off:off+8, l], U_full[off:off+8, i], cs[l], cs[i])
        for k in range(8):
            Fleft[k, i] = F[k]

    # Conservative update for this species' 8 fields
    inv_dx = 1.0/dx
    for i in range(N):
        ip = (i+1) % N
        for k in range(8):
            U_full[off+k, i] = U_full[off+k, i] - dt*inv_dx*(Fleft[k, ip] - Fleft[k, i])

    # Liouville sources on alpha, beta
    for i in range(N):
        Svv_i = Pxx[i]/max(rho[i], 1e-30)
        a = alpha[i]; b = beta[i]
        gamma2 = Svv_i - b*b     # signed
        ip2 = (i+1) % N; im2 = (i-1) % N
        dudx_i = (u[ip2] - u[im2])/(2.0*dx)
        U_full[off+IDX_ALPHA, i] += dt*rho[i]*b
        a_safe = a if a > 1e-15 else 1e-15
        U_full[off+IDX_BETA,  i] += dt*rho[i]*(gamma2/a_safe - dudx_i*b)

    # Self-BGK relaxation (exact exponential)
    decay = np.exp(-dt/tau_self)
    for i in range(N):
        rho_n = U_full[off+IDX_RHO, i]
        u_n   = U_full[off+IDX_MOM, i]/max(rho_n, 1e-30)
        Pxx_n = U_full[off+IDX_EXX, i] - rho_n*u_n*u_n
        Pp_n  = U_full[off+IDX_PP, i]
        a_n   = U_full[off+IDX_ALPHA, i]/max(rho_n, 1e-30)
        b_n   = U_full[off+IDX_BETA,  i]/max(rho_n, 1e-30)
        M3_n  = U_full[off+IDX_M3, i]
        Q_n   = M3_n - rho_n*u_n*u_n*u_n - 3.0*u_n*Pxx_n

        # Positivity floors
        if Pxx_n < 1e-8: Pxx_n = 1e-8
        if Pp_n  < 1e-8: Pp_n  = 1e-8

        P_iso = (Pxx_n + 2.0*Pp_n)/3.0
        Pxx_new = P_iso + (Pxx_n - P_iso)*decay
        Pp_new  = P_iso + (Pp_n  - P_iso)*decay
        b_new   = b_n*decay
        Q_new   = Q_n*decay

        # Realizability clip
        Svv_new = max(Pxx_new, 1e-30)/max(rho_n, 1e-30)
        b_max = 0.999*np.sqrt(Svv_new)
        if b_new >  b_max: b_new =  b_max
        elif b_new < -b_max: b_new = -b_max

        U_full[off+IDX_EXX, i] = rho_n*u_n*u_n + Pxx_new
        U_full[off+IDX_PP,  i] = Pp_new
        U_full[off+IDX_BETA, i] = rho_n*b_new
        U_full[off+IDX_M3,  i] = rho_n*u_n*u_n*u_n + 3.0*u_n*Pxx_new + Q_new


@nb.njit(cache=True, fastmath=False)
def _species_step_transmissive(U_full, off, dx, dt, tau_self, N):
    """Transmissive BC version (for Sod-like tests)."""
    rho   = np.empty(N); u = np.empty(N); Pxx = np.empty(N); Pp = np.empty(N)
    L1    = np.empty(N); alpha = np.empty(N); beta = np.empty(N); Q = np.empty(N)
    cs    = np.empty(N)
    for i in range(N):
        r, v, pxx, pp, l1, a, b, svv, q = _species_primitives(U_full[off:off+8, i])
        rho[i]=r; u[i]=v; Pxx[i]=pxx; Pp[i]=pp; L1[i]=l1
        alpha[i]=a; beta[i]=b; Q[i]=q
        cs[i] = np.sqrt(CSCOEF*max(pxx, 1e-30)/max(r, 1e-30))

    Fleft = np.empty((8, N+1))
    for i in range(N+1):
        if i == 0:      l = 0; r = 0
        elif i == N:    l = N-1; r = N-1
        else:           l = i-1; r = i
        F = _hll_flux_one_species(U_full[off:off+8, l], U_full[off:off+8, r], cs[l], cs[r])
        for k in range(8):
            Fleft[k, i] = F[k]

    inv_dx = 1.0/dx
    for i in range(N):
        for k in range(8):
            U_full[off+k, i] = U_full[off+k, i] - dt*inv_dx*(Fleft[k, i+1] - Fleft[k, i])

    for i in range(N):
        Svv_i = Pxx[i]/max(rho[i], 1e-30)
        a = alpha[i]; b = beta[i]
        gamma2 = Svv_i - b*b
        if i == 0:        dudx_i = (u[1] - u[0])/dx
        elif i == N-1:    dudx_i = (u[N-1] - u[N-2])/dx
        else:             dudx_i = (u[i+1] - u[i-1])/(2.0*dx)
        U_full[off+IDX_ALPHA, i] += dt*rho[i]*b
        a_safe = a if a > 1e-15 else 1e-15
        U_full[off+IDX_BETA,  i] += dt*rho[i]*(gamma2/a_safe - dudx_i*b)

    decay = np.exp(-dt/tau_self)
    for i in range(N):
        rho_n = U_full[off+IDX_RHO, i]
        u_n   = U_full[off+IDX_MOM, i]/max(rho_n, 1e-30)
        Pxx_n = U_full[off+IDX_EXX, i] - rho_n*u_n*u_n
        Pp_n  = U_full[off+IDX_PP, i]
        a_n   = U_full[off+IDX_ALPHA, i]/max(rho_n, 1e-30)
        b_n   = U_full[off+IDX_BETA,  i]/max(rho_n, 1e-30)
        M3_n  = U_full[off+IDX_M3, i]
        Q_n   = M3_n - rho_n*u_n*u_n*u_n - 3.0*u_n*Pxx_n

        if Pxx_n < 1e-8: Pxx_n = 1e-8
        if Pp_n  < 1e-8: Pp_n  = 1e-8

        P_iso = (Pxx_n + 2.0*Pp_n)/3.0
        Pxx_new = P_iso + (Pxx_n - P_iso)*decay
        Pp_new  = P_iso + (Pp_n  - P_iso)*decay
        b_new   = b_n*decay
        Q_new   = Q_n*decay

        Svv_new = max(Pxx_new, 1e-30)/max(rho_n, 1e-30)
        b_max = 0.999*np.sqrt(Svv_new)
        if b_new >  b_max: b_new =  b_max
        elif b_new < -b_max: b_new = -b_max

        U_full[off+IDX_EXX, i] = rho_n*u_n*u_n + Pxx_new
        U_full[off+IDX_PP,  i] = Pp_new
        U_full[off+IDX_BETA, i] = rho_n*b_new
        U_full[off+IDX_M3,  i] = rho_n*u_n*u_n*u_n + 3.0*u_n*Pxx_new + Q_new


# -------- Cross-coupling operator (exact exponential) --------
#
# Equations in the cross-coupling step (all conserve total momentum and total energy):
#   d u^A / dt = -nu_A * (u^A - u^B)
#   d u^B / dt = -nu_B * (u^B - u^A)
#   nu_A = nu_p * rho^B / (rho^A + rho^B)
#   nu_B = nu_p * rho^A / (rho^A + rho^B)
#
# Delta u = u^A - u^B decays as:   Delta u(t) = Delta u(0) exp(-nu_p * t)
# u^* = (rho^A u^A + rho^B u^B)/(rho^A+rho^B)  is conserved.
#
# Energy: let e^s = P^s_iso/(rho^s * (gamma - 1)) be specific thermal energy
# Two effects:
#   (1) Thermal equilibration T^A <-> T^B at rate nu_T
#   (2) Frictional heating from relative motion
#
# Total thermal energy rate of change (conservative formulation):
#   d (rho^A e^A)/dt = Fric + (-)*ThermEq term
#   d (rho^B e^B)/dt = Fric + (+)*ThermEq term
# where Fric = (1/2) * mu_AB * (n^A n^B / (n^A+n^B)) * nu_p * (Delta u)^2 goes to each fluid
# weighted by inverse mass fraction of the partner.
#
# For simplicity and exact conservation, we do this in two sub-steps:
#   (a) Momentum exchange (exact exp); compute frictional heating added to a common
#       energy reservoir; partition it between species proportional to their number densities.
#   (b) Thermal equilibration (exact exp) between T^A, T^B toward common T^*.
# This ordering conserves total momentum and total energy to machine precision.
#
# Higher moments (Q, beta, anisotropy P_xx - P_perp) relax toward zero with rate nu_p
# weighted by opposite-species number fraction:
#   d Q^s / dt = -nu_p * (n^{s'}/(n^A+n^B)) * Q^s
# This damps the non-Maxwellian features on the cross-coupling timescale.

@nb.njit(cache=True, fastmath=False)
def _apply_cross_coupling_cell(U, dt, nu_p, nu_T, m_A, m_B):
    """In-place cross-coupling update on one cell's 16 fields.
    Conserves total rho*u and total thermal+kinetic energy to machine precision."""
    # Extract species A primitives
    rhoA = U[IDX_RHO]; uA = U[IDX_MOM]/max(rhoA, 1e-30)
    PxxA = U[IDX_EXX] - rhoA*uA*uA; PpA = U[IDX_PP]
    bA   = U[IDX_BETA]/max(rhoA, 1e-30)
    M3A  = U[IDX_M3]; QA = M3A - rhoA*uA*uA*uA - 3.0*uA*PxxA
    PisoA = (PxxA + 2.0*PpA)/3.0
    TA    = PisoA/max(rhoA, 1e-30) * m_A     # T^A in energy units: P = n T = (rho/m) T
    # wait -- P_iso = n * T means T = P_iso * m / rho.  Need to be careful.
    TA    = PisoA * m_A / max(rhoA, 1e-30)

    # Species B
    rhoB = U[8+IDX_RHO]; uB = U[8+IDX_MOM]/max(rhoB, 1e-30)
    PxxB = U[8+IDX_EXX] - rhoB*uB*uB; PpB = U[8+IDX_PP]
    bB   = U[8+IDX_BETA]/max(rhoB, 1e-30)
    M3B  = U[8+IDX_M3]; QB = M3B - rhoB*uB*uB*uB - 3.0*uB*PxxB
    PisoB = (PxxB + 2.0*PpB)/3.0
    TB    = PisoB * m_B / max(rhoB, 1e-30)

    nA = rhoA/m_A; nB = rhoB/m_B
    ntot = nA + nB
    if ntot < 1e-30: return

    # ----- (a) Momentum exchange, exact exponential -----
    u_star = (rhoA*uA + rhoB*uB)/max(rhoA + rhoB, 1e-30)
    Delta_u = uA - uB
    Delta_u_new = Delta_u * np.exp(-nu_p * dt)
    # Frictional kinetic energy converted to thermal:
    #   dKE = (1/2) * rho_red * ((Delta_u)^2 - (Delta_u_new)^2)
    # where rho_red = rho^A rho^B / (rho^A + rho^B) is the reduced mass density
    rho_red = rhoA*rhoB/max(rhoA + rhoB, 1e-30)
    dKE = 0.5 * rho_red * (Delta_u*Delta_u - Delta_u_new*Delta_u_new)
    # Partition this among species by number density (mean-free-path argument):
    dE_A = dKE * nB/ntot         # goes to species A: B's scatterers heat A
    dE_B = dKE * nA/ntot
    # Update velocities
    uA_new = u_star + (rhoB/max(rhoA+rhoB, 1e-30))*Delta_u_new
    uB_new = u_star - (rhoA/max(rhoA+rhoB, 1e-30))*Delta_u_new
    # Add frictional heat to each species' isotropic pressure (distribute 1/3 to Pxx, 2/3 to Pp)
    dPiso_A = (2.0/3.0) * dE_A     # since P = (2/3) * thermal energy density for 3D
    dPiso_B = (2.0/3.0) * dE_B
    # Add 1/3 of delta P_iso to P_xx, 2/3 split between the two P_perp directions (one scalar here, holding 2 degrees)
    PxxA_new = PxxA + dPiso_A                  # 1/3 per direction * 3 directions if isotropic
    PpA_new  = PpA  + dPiso_A                  # NOTE: this must give correct trace update
    PxxB_new = PxxB + dPiso_B
    PpB_new  = PpB  + dPiso_B
    # Check: P_iso_new = (Pxx_new + 2 Pp_new)/3 = (PxxA + dPiso_A + 2(PpA + dPiso_A))/3
    #       = (PxxA + 2 PpA)/3 + 3*dPiso_A/3 = PisoA + dPiso_A
    # And dE/d(P_iso) for ideal gas, e_th = (3/2) n T = (3/2) P_iso, so dE = (3/2) dP_iso -> dP_iso = (2/3) dE. ✓

    # ----- (b) Thermal equilibration -----
    PisoA_new = (PxxA_new + 2.0*PpA_new)/3.0
    PisoB_new = (PxxB_new + 2.0*PpB_new)/3.0
    TA_new = PisoA_new * m_A / max(rhoA, 1e-30)
    TB_new = PisoB_new * m_B / max(rhoB, 1e-30)
    # Common temperature T_star conserves total thermal energy:
    # E_th_total = (3/2) (n_A T_A + n_B T_B)
    # T_star = (n_A T_A + n_B T_B)/(n_A + n_B)
    T_star = (nA*TA_new + nB*TB_new)/ntot
    Delta_T = TA_new - TB_new
    Delta_T_new = Delta_T * np.exp(-nu_T * dt)
    TA_final = T_star + (nB/ntot)*Delta_T_new
    TB_final = T_star - (nA/ntot)*Delta_T_new
    # Back to pressures
    PisoA_final = nA*TA_final       # P = n T
    PisoB_final = nB*TB_final
    # Update P_xx and P_perp preserving their *anisotropy ratio* (don't let thermal
    # equilibration isotropize them — the self-BGK does that separately).
    anisA = PxxA_new - PpA_new
    anisB = PxxB_new - PpB_new
    # New P_xx, P_perp such that their trace is PisoA_final but anisotropy preserved
    PxxA_final = PisoA_final + (2.0/3.0)*anisA
    PpA_final  = PisoA_final - (1.0/3.0)*anisA
    PxxB_final = PisoB_final + (2.0/3.0)*anisB
    PpB_final  = PisoB_final - (1.0/3.0)*anisB

    # ----- (c) Damp higher moments and cross-correlation -----
    # Q and beta relax with rate nu_p * (opposite species fraction)
    damp_A_coef = nA/ntot   # species A's Q is damped by rate nu_p * n_B/n_tot... wait
    # Reading my notes: d Q^A/dt = -nu_p * (n^B / n_tot) * Q^A
    damp_A = np.exp(-nu_p * dt * nB/ntot)
    damp_B = np.exp(-nu_p * dt * nA/ntot)
    QA_new_damped = QA * damp_A
    QB_new_damped = QB * damp_B
    bA_new = bA * damp_A
    bB_new = bB * damp_B

    # Also damp the pressure anisotropy (same rate)
    PxxA_final = PisoA_final + (2.0/3.0)*anisA*damp_A
    PpA_final  = PisoA_final - (1.0/3.0)*anisA*damp_A
    PxxB_final = PisoB_final + (2.0/3.0)*anisB*damp_B
    PpB_final  = PisoB_final - (1.0/3.0)*anisB*damp_B

    # ----- Write back -----
    U[IDX_MOM] = rhoA * uA_new
    U[IDX_EXX] = rhoA*uA_new*uA_new + PxxA_final
    U[IDX_PP]  = PpA_final
    U[IDX_BETA] = rhoA * bA_new
    # M3 = rho u^3 + 3 u P_xx + Q
    U[IDX_M3]  = rhoA*uA_new*uA_new*uA_new + 3.0*uA_new*PxxA_final + QA_new_damped

    U[8+IDX_MOM] = rhoB * uB_new
    U[8+IDX_EXX] = rhoB*uB_new*uB_new + PxxB_final
    U[8+IDX_PP]  = PpB_final
    U[8+IDX_BETA] = rhoB * bB_new
    U[8+IDX_M3]  = rhoB*uB_new*uB_new*uB_new + 3.0*uB_new*PxxB_final + QB_new_damped


@nb.njit(cache=True, fastmath=False)
def apply_cross_coupling(U, dt, nu_p_arr, nu_T_arr, m_A, m_B, N):
    """Apply cross-coupling to all N cells in the 16-field state."""
    for i in range(N):
        _apply_cross_coupling_cell(U[:, i], dt, nu_p_arr[i], nu_T_arr[i], m_A, m_B)


# -------- Physical kernels: return (nu_p, nu_T) per cell --------

@nb.njit(cache=True, fastmath=False)
def kernel_epstein(U, m_A, m_B, grain_radius, grain_mat_density, N):
    """Dust-in-gas, Epstein regime.
    Species A = dust (grain of mass m_A, radius a, internal density rho_mat).
    Species B = gas (molecular mass m_B).
    t_s = rho_mat * a / (rho_g * v_bar_th);  nu_p = 1/t_s;
    nu_T = (2 m_B / m_A) * nu_p = extreme thermal inefficiency for heavy grains.
    """
    nu_p = np.empty(N); nu_T = np.empty(N)
    for i in range(N):
        rhoB = U[8+IDX_RHO, i]
        uB   = U[8+IDX_MOM, i]/max(rhoB, 1e-30)
        PxxB = U[8+IDX_EXX, i] - rhoB*uB*uB
        PpB  = U[8+IDX_PP, i]
        PisoB = (PxxB + 2.0*PpB)/3.0
        TB = PisoB * m_B / max(rhoB, 1e-30)      # gas temperature
        v_bar = np.sqrt(8.0*TB/(np.pi*m_B))
        t_s = grain_mat_density * grain_radius / (rhoB * v_bar + 1e-30)
        nu_p[i] = 1.0/max(t_s, 1e-30)
        nu_T[i] = 2.0 * m_B / m_A * nu_p[i]
    return nu_p, nu_T


@nb.njit(cache=True, fastmath=False)
def kernel_coulomb(U, m_A, m_B, Z_A, Z_B, lnLambda, N):
    """Electron-ion Coulomb coupling.  Species A = electrons, B = ions.

    Using Gaussian-cgs-like units with e = 1:
      nu_p = (4/3) sqrt(2 pi) * n_i Z^2 e^4 lnLambda /
             (m_e * (T_e/m_e + T_i/m_i)^{3/2})
    We absorb e^4 into a single coupling-strength parameter `coupling` = e^4
    (user can set = 1 in dimensionless units).
    nu_T = (2 m_A m_B / (m_A + m_B)^2) * nu_p
    """
    nu_p = np.empty(N); nu_T = np.empty(N)
    # pre-factor 4/3 sqrt(2 pi)
    pref = 4.0/3.0 * np.sqrt(2.0*np.pi)
    # In dimensionless units we set e^4 = 1 by convention
    e4 = 1.0
    for i in range(N):
        rhoA = U[IDX_RHO, i]; uA = U[IDX_MOM, i]/max(rhoA, 1e-30)
        PxxA = U[IDX_EXX, i] - rhoA*uA*uA; PpA = U[IDX_PP, i]
        PisoA = (PxxA + 2.0*PpA)/3.0
        TA = PisoA * m_A / max(rhoA, 1e-30)

        rhoB = U[8+IDX_RHO, i]; uB = U[8+IDX_MOM, i]/max(rhoB, 1e-30)
        PxxB = U[8+IDX_EXX, i] - rhoB*uB*uB; PpB = U[8+IDX_PP, i]
        PisoB = (PxxB + 2.0*PpB)/3.0
        TB = PisoB * m_B / max(rhoB, 1e-30)

        nB = rhoB/m_B
        Zeff2 = Z_A*Z_A * Z_B*Z_B
        velscale = TA/m_A + TB/m_B
        if velscale < 1e-30: velscale = 1e-30
        nu_p[i] = pref * nB * Zeff2 * e4 * lnLambda / (m_A * velscale**1.5)
        nu_T[i] = 2.0 * m_A * m_B / ((m_A + m_B)**2) * nu_p[i]
    return nu_p, nu_T


@nb.njit(cache=True, fastmath=False)
def kernel_hard_sphere(U, m_A, m_B, sigma, N):
    """Two neutral species with a hard-sphere cross-section sigma.
    v_rel = sqrt(8 T*/pi mu) where T* = (n_A T_A + n_B T_B)/(n_A+n_B), mu=reduced mass.
    nu_p = n_{s'} * sigma * v_rel.
    nu_T = 2 mu_AB / (m_A + m_B) * nu_p  for equal/comparable masses."""
    nu_p = np.empty(N); nu_T = np.empty(N)
    mu_AB = m_A*m_B/(m_A+m_B)
    for i in range(N):
        rhoA = U[IDX_RHO, i]; uA = U[IDX_MOM, i]/max(rhoA, 1e-30)
        PxxA = U[IDX_EXX, i] - rhoA*uA*uA; PpA = U[IDX_PP, i]
        PisoA = (PxxA + 2.0*PpA)/3.0
        TA = PisoA * m_A / max(rhoA, 1e-30)

        rhoB = U[8+IDX_RHO, i]; uB = U[8+IDX_MOM, i]/max(rhoB, 1e-30)
        PxxB = U[8+IDX_EXX, i] - rhoB*uB*uB; PpB = U[8+IDX_PP, i]
        PisoB = (PxxB + 2.0*PpB)/3.0
        TB = PisoB * m_B / max(rhoB, 1e-30)

        nA = rhoA/m_A; nB = rhoB/m_B
        ntot = nA + nB
        T_star = (nA*TA + nB*TB)/max(ntot, 1e-30)
        v_rel = np.sqrt(8.0*T_star/(np.pi*mu_AB) + (uA-uB)**2)
        nu_p[i] = nB * sigma * v_rel    # rate for species A; symmetric handling in operator
        nu_T[i] = 2.0 * mu_AB / (m_A + m_B) * nu_p[i]
    return nu_p, nu_T


@nb.njit(cache=True, fastmath=False)
def kernel_sidm(U, m_A, m_B, sigma_over_m, N):
    """Self-interacting dark matter. sigma_over_m in cm^2/g (or dimensionless units).
    nu_p = rho_{s'} * (sigma/m) * v_rel."""
    nu_p = np.empty(N); nu_T = np.empty(N)
    mu_AB = m_A*m_B/(m_A+m_B)
    for i in range(N):
        rhoA = U[IDX_RHO, i]; uA = U[IDX_MOM, i]/max(rhoA, 1e-30)
        PxxA = U[IDX_EXX, i] - rhoA*uA*uA; PpA = U[IDX_PP, i]
        PisoA = (PxxA + 2.0*PpA)/3.0
        TA = PisoA * m_A / max(rhoA, 1e-30)

        rhoB = U[8+IDX_RHO, i]; uB = U[8+IDX_MOM, i]/max(rhoB, 1e-30)
        PxxB = U[8+IDX_EXX, i] - rhoB*uB*uB; PpB = U[8+IDX_PP, i]
        PisoB = (PxxB + 2.0*PpB)/3.0
        TB = PisoB * m_B / max(rhoB, 1e-30)

        v_rel = np.sqrt(8.0*(TA/m_A + TB/m_B)/np.pi + (uA-uB)**2)
        nu_p[i] = rhoB * sigma_over_m * v_rel
        nu_T[i] = 2.0 * m_A * m_B / ((m_A + m_B)**2) * nu_p[i]
    return nu_p, nu_T


# -------- Top-level step (operator split) --------

@nb.njit(cache=True, fastmath=False)
def max_signal_speed_both(U, N):
    smax = 0.0
    for i in range(N):
        for off in (0, 8):
            rho = U[off+IDX_RHO, i]
            u = U[off+IDX_MOM, i]/max(rho, 1e-30)
            Pxx = U[off+IDX_EXX, i] - rho*u*u
            cs = np.sqrt(CSCOEF*max(Pxx, 1e-30)/max(rho, 1e-30))
            s = abs(u) + cs
            if s > smax: smax = s
    return smax


def step_two_species(U, dx, dt, tau_AA, tau_BB, kernel_fn, kernel_params, m_A, m_B,
                     bc='periodic'):
    """One full step: single-species updates for each, then cross-coupling."""
    N = U.shape[1]
    # Species updates (order-independent since they're spatial-only; the cross-coupling
    # operator happens in a separate substep).
    if bc == 'periodic':
        _species_step_periodic(U, 0, dx, dt, tau_AA, N)
        _species_step_periodic(U, 8, dx, dt, tau_BB, N)
    else:
        _species_step_transmissive(U, 0, dx, dt, tau_AA, N)
        _species_step_transmissive(U, 8, dx, dt, tau_BB, N)
    # Compute kernel rates, apply cross-coupling
    nu_p, nu_T = kernel_fn(U, m_A, m_B, *kernel_params, N)
    apply_cross_coupling(U, dt, nu_p, nu_T, m_A, m_B, N)
    return U


# -------- Initial-state builder --------

def make_initial_state(N, rho_A, u_A, P_A, rho_B, u_B, P_B, sigma_x0=0.02,
                        x_coords=None):
    """Build a 16-field initial state.  rho_* etc. are arrays of length N."""
    if x_coords is None:
        x_coords = np.linspace(0, 1, N, endpoint=False) + 0.5/N
    U = np.zeros((16, N))
    # Species A
    U[0] = rho_A
    U[1] = rho_A * u_A
    U[2] = rho_A * u_A * u_A + P_A            # E_xx = rho u^2 + P_xx; start isotropic so Pxx=P_iso
    U[3] = P_A                                  # P_perp = P_iso initially
    U[4] = rho_A * x_coords                     # L1 = x
    U[5] = rho_A * sigma_x0                     # alpha = sigma_x0
    U[6] = rho_A * 0.0                          # beta = 0
    U[7] = rho_A * u_A**3 + 3.0*u_A*P_A         # M3 with Q=0
    # Species B
    U[8]  = rho_B
    U[9]  = rho_B * u_B
    U[10] = rho_B * u_B * u_B + P_B
    U[11] = P_B
    U[12] = rho_B * x_coords
    U[13] = rho_B * sigma_x0
    U[14] = rho_B * 0.0
    U[15] = rho_B * u_B**3 + 3.0*u_B*P_B
    return U
