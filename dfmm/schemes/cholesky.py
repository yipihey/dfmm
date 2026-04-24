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
                      hll_edge_flux, limit, LIMITER_MINMOD)


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
             rho_floor, alpha_floor, realizability_headroom,
             Unew, Fleft):
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

    `Unew` and `Fleft` are pre-allocated scratch buffers (see
    `Workspace.for_padded_state`): `Unew` shape matches `U`, `Fleft`
    has shape `(n_fields, N + 2*n_ghost + 1)`. The kernel writes into
    these in place and returns `Unew` (so the caller can ping-pong
    the two buffers across steps without re-allocation).
    """
    n_fields, N_tot = U.shape
    N = N_tot - 2 * n_ghost

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
def hll_step_muscl(U, dx, dt, tau, n_ghost,
                   rho_floor, alpha_floor, realizability_headroom,
                   limiter_code,
                   Unew, Fleft, U_L_edge, U_R_edge):
    """Second-order MUSCL-Hancock HLL + BGK step on a ghost-padded state.

    Requires n_ghost >= 2 so the reconstruction stencil (per-cell
    slope uses cells [j-1, j, j+1]) has room at the interior-face
    boundaries. Reconstruction is in primitive variables
    (rho, u, Pxx, Pp, L1, alpha, beta, Q) to avoid nonphysical
    reconstructed states (e.g. negative rho); the limited slope is
    controlled by `limiter_code` (0=minmod, 1=MC, 2=van_leer).

    Workspace buffers `U_L_edge` / `U_R_edge` hold the reconstructed
    conserved state at each cell's left/right edge after the Hancock
    half-step predictor. The face loop then uses HLL on these
    predicted states.

    Post-flux logic (Liouville sources, BGK relaxation) is identical
    to `hll_step`.
    """
    n_fields, N_tot = U.shape
    N = N_tot - 2 * n_ghost
    inv_dx = 1.0 / dx
    half_dt_over_dx = 0.5 * dt * inv_dx

    # Vectorized primitives on the full padded array (used below for
    # slopes, Liouville source, and BGK).
    rho   = U[IDX_RHO]
    u     = U[IDX_MOM]/rho
    Pxx   = U[IDX_EXX] - rho*u*u
    Pp    = U[IDX_PP]
    L1    = U[IDX_L1]/rho
    alpha = U[IDX_ALPHA]/rho
    beta  = U[IDX_BETA]/rho
    M3    = U[IDX_M3]
    Q     = M3 - rho*u*u*u - 3.0*u*Pxx

    # --- MUSCL reconstruction + Hancock predictor ---
    # For each cell j in [1, N_tot-2], compute slope-limited edge
    # states and evolve them by dt/2 via the analytic flux Jacobian
    # applied to the reconstructed states. We skip cells 0 and
    # N_tot-1 (need ghosts on both sides). With n_ghost >= 2 this
    # still covers every cell that contributes to an interior-face
    # flux.
    for j in range(1, N_tot - 1):
        # Primitive slopes via the selected limiter
        s_rho = limit(rho[j]   - rho[j-1],   rho[j+1]   - rho[j],   limiter_code)
        s_u   = limit(u[j]     - u[j-1],     u[j+1]     - u[j],     limiter_code)
        s_Pxx = limit(Pxx[j]   - Pxx[j-1],   Pxx[j+1]   - Pxx[j],   limiter_code)
        s_Pp  = limit(Pp[j]    - Pp[j-1],    Pp[j+1]    - Pp[j],    limiter_code)
        s_L1  = limit(L1[j]    - L1[j-1],    L1[j+1]    - L1[j],    limiter_code)
        s_a   = limit(alpha[j] - alpha[j-1], alpha[j+1] - alpha[j], limiter_code)
        s_b   = limit(beta[j]  - beta[j-1],  beta[j+1]  - beta[j],  limiter_code)
        s_Q   = limit(Q[j]     - Q[j-1],     Q[j+1]     - Q[j],     limiter_code)

        # Reconstructed primitives at left edge (minus) and right edge (plus)
        rho_L = rho[j]   - 0.5 * s_rho; rho_R = rho[j]   + 0.5 * s_rho
        u_L   = u[j]     - 0.5 * s_u;   u_R   = u[j]     + 0.5 * s_u
        Pxx_L = Pxx[j]   - 0.5 * s_Pxx; Pxx_R = Pxx[j]   + 0.5 * s_Pxx
        Pp_L  = Pp[j]    - 0.5 * s_Pp;  Pp_R  = Pp[j]    + 0.5 * s_Pp
        L1_L  = L1[j]    - 0.5 * s_L1;  L1_R  = L1[j]    + 0.5 * s_L1
        a_L   = alpha[j] - 0.5 * s_a;   a_R   = alpha[j] + 0.5 * s_a
        b_L   = beta[j]  - 0.5 * s_b;   b_R   = beta[j]  + 0.5 * s_b
        Q_L   = Q[j]     - 0.5 * s_Q;   Q_R   = Q[j]     + 0.5 * s_Q

        # Floor rho to keep fluxes finite
        rho_L_safe = rho_L if rho_L > rho_floor else rho_floor
        rho_R_safe = rho_R if rho_R > rho_floor else rho_floor

        # Analytic flux at each reconstructed edge state
        FL0 = rho_L * u_L
        FL1 = rho_L * u_L * u_L + Pxx_L
        FL2 = rho_L * u_L**3 + 3.0 * u_L * Pxx_L + Q_L
        FL3 = u_L * Pp_L
        FL4 = rho_L * L1_L * u_L
        FL5 = rho_L * a_L * u_L
        FL6 = rho_L * b_L * u_L
        FL7 = rho_L * u_L**4 + 6.0 * u_L * u_L * Pxx_L + 4.0 * u_L * Q_L \
              + 3.0 * Pxx_L * Pxx_L / rho_L_safe

        FR0 = rho_R * u_R
        FR1 = rho_R * u_R * u_R + Pxx_R
        FR2 = rho_R * u_R**3 + 3.0 * u_R * Pxx_R + Q_R
        FR3 = u_R * Pp_R
        FR4 = rho_R * L1_R * u_R
        FR5 = rho_R * a_R * u_R
        FR6 = rho_R * b_R * u_R
        FR7 = rho_R * u_R**4 + 6.0 * u_R * u_R * Pxx_R + 4.0 * u_R * Q_R \
              + 3.0 * Pxx_R * Pxx_R / rho_R_safe

        # Conserved-form left/right edges, then Hancock half-step predictor:
        #   U_edge_pred = U_edge - (dt/2/dx) * (F_R - F_L)
        U_L_edge[IDX_RHO,   j] = rho_L                          - half_dt_over_dx * (FR0 - FL0)
        U_L_edge[IDX_MOM,   j] = rho_L * u_L                    - half_dt_over_dx * (FR1 - FL1)
        U_L_edge[IDX_EXX,   j] = rho_L * u_L * u_L + Pxx_L      - half_dt_over_dx * (FR2 - FL2)
        U_L_edge[IDX_PP,    j] = Pp_L                           - half_dt_over_dx * (FR3 - FL3)
        U_L_edge[IDX_L1,    j] = rho_L * L1_L                   - half_dt_over_dx * (FR4 - FL4)
        U_L_edge[IDX_ALPHA, j] = rho_L * a_L                    - half_dt_over_dx * (FR5 - FL5)
        U_L_edge[IDX_BETA,  j] = rho_L * b_L                    - half_dt_over_dx * (FR6 - FL6)
        U_L_edge[IDX_M3,    j] = rho_L * u_L**3 + 3.0 * u_L * Pxx_L + Q_L - half_dt_over_dx * (FR7 - FL7)

        U_R_edge[IDX_RHO,   j] = rho_R                          - half_dt_over_dx * (FR0 - FL0)
        U_R_edge[IDX_MOM,   j] = rho_R * u_R                    - half_dt_over_dx * (FR1 - FL1)
        U_R_edge[IDX_EXX,   j] = rho_R * u_R * u_R + Pxx_R      - half_dt_over_dx * (FR2 - FL2)
        U_R_edge[IDX_PP,    j] = Pp_R                           - half_dt_over_dx * (FR3 - FL3)
        U_R_edge[IDX_L1,    j] = rho_R * L1_R                   - half_dt_over_dx * (FR4 - FL4)
        U_R_edge[IDX_ALPHA, j] = rho_R * a_R                    - half_dt_over_dx * (FR5 - FL5)
        U_R_edge[IDX_BETA,  j] = rho_R * b_R                    - half_dt_over_dx * (FR6 - FL6)
        U_R_edge[IDX_M3,    j] = rho_R * u_R**3 + 3.0 * u_R * Pxx_R + Q_R - half_dt_over_dx * (FR7 - FL7)

    # --- Face fluxes from reconstructed edge states ---
    # At face i (between cell i-1 and cell i), take the left state
    # from the right edge of cell i-1 and the right state from the
    # left edge of cell i. Compute cs from each reconstructed state.
    for i in range(n_ghost, n_ghost + N + 1):
        # Left side = U_R_edge[:, i-1], right side = U_L_edge[:, i]
        rho_L = U_R_edge[IDX_RHO, i-1]
        rho_L_safe = rho_L if rho_L > rho_floor else rho_floor
        u_L_face = U_R_edge[IDX_MOM, i-1] / rho_L_safe
        Pxx_L_face = U_R_edge[IDX_EXX, i-1] - rho_L * u_L_face * u_L_face
        cs_L = np.sqrt(CSCOEF * max(Pxx_L_face, rho_floor) / rho_L_safe)

        rho_R = U_L_edge[IDX_RHO, i]
        rho_R_safe = rho_R if rho_R > rho_floor else rho_floor
        u_R_face = U_L_edge[IDX_MOM, i] / rho_R_safe
        Pxx_R_face = U_L_edge[IDX_EXX, i] - rho_R * u_R_face * u_R_face
        cs_R = np.sqrt(CSCOEF * max(Pxx_R_face, rho_floor) / rho_R_safe)

        F0, F1, F2, F3, F4, F5, F6, F7 = hll_edge_flux(
            U_R_edge[:, i-1], U_L_edge[:, i], cs_L, cs_R)
        Fleft[0, i] = F0; Fleft[1, i] = F1; Fleft[2, i] = F2; Fleft[3, i] = F3
        Fleft[4, i] = F4; Fleft[5, i] = F5; Fleft[6, i] = F6; Fleft[7, i] = F7

    # --- Conservative update for interior cells ---
    for j in range(n_ghost, n_ghost + N):
        for k in range(n_fields):
            Unew[k, j] = U[k, j] - dt * inv_dx * (Fleft[k, j+1] - Fleft[k, j])
    # Copy ghost cells through (apply_* will rewrite them on next step)
    for g in range(n_ghost):
        for k in range(n_fields):
            Unew[k, g] = U[k, g]
            Unew[k, n_ghost + N + g] = U[k, n_ghost + N + g]

    # --- Liouville sources for alpha and beta (same stencil as first-order) ---
    for j in range(n_ghost, n_ghost + N):
        Sigma_vv_i = Pxx[j] / max(rho[j], rho_floor)
        a = alpha[j]; b = beta[j]
        gamma2_signed = Sigma_vv_i - b * b
        dudx_i = (u[j+1] - u[j-1]) / (2.0 * dx)
        Unew[IDX_ALPHA, j] += dt * rho[j] * b
        a_safe = a if a > alpha_floor else alpha_floor
        Unew[IDX_BETA, j]  += dt * rho[j] * (gamma2_signed / a_safe - dudx_i * b)

    # --- Exact-exponential BGK relaxation on interior cells ---
    decay = np.exp(-dt / tau)
    for j in range(n_ghost, n_ghost + N):
        rho_n = Unew[IDX_RHO, j]
        u_n   = Unew[IDX_MOM, j] / rho_n
        Pxx_n = Unew[IDX_EXX, j] - rho_n * u_n * u_n
        Pp_n  = Unew[IDX_PP, j]
        b_n   = Unew[IDX_BETA, j] / rho_n
        M3_n  = Unew[IDX_M3, j]
        Q_n   = M3_n - rho_n * u_n**3 - 3.0 * u_n * Pxx_n

        P_iso = (Pxx_n + 2.0 * Pp_n) / 3.0
        Pxx_new = P_iso + (Pxx_n - P_iso) * decay
        Pp_new  = P_iso + (Pp_n  - P_iso) * decay
        b_new = b_n * decay
        Q_new = Q_n * decay

        Sigma_vv_new = max(Pxx_new, rho_floor) / max(rho_n, rho_floor)
        beta_max = realizability_headroom * np.sqrt(Sigma_vv_new)
        if b_new > beta_max:    b_new = beta_max
        elif b_new < -beta_max: b_new = -beta_max

        Unew[IDX_EXX,  j] = rho_n * u_n * u_n + Pxx_new
        Unew[IDX_PP,   j] = Pp_new
        Unew[IDX_BETA, j] = rho_n * b_new
        Unew[IDX_M3,   j] = rho_n * u_n**3 + 3.0 * u_n * Pxx_new + Q_new

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
