"""Shared constants, HLL flux helper, and workspace for the 8-field
moment state.

Imported by the cholesky scheme (`dfmm.schemes.cholesky`), the
energy-conservative noise scheme (`dfmm.closure.noise_model`), and
the two-fluid scheme (`dfmm.schemes.two_fluid`, where each species
occupies its own 8-field block). The maxent and barotropic schemes
use different wave-speed coefficients and do not share this module.

The IDX_* constants are plain ints; Numba inlines them into the
compiled code, so using `U[IDX_RHO]` has identical cost to `U[0]`.
"""
from dataclasses import dataclass
import numpy as np
import numba as nb

IDX_RHO   = 0   # rho
IDX_MOM   = 1   # rho u
IDX_EXX   = 2   # E_xx = rho u^2 + P_xx
IDX_PP    = 3   # P_perp
IDX_L1    = 4   # rho L_1 (advected Lagrangian label)
IDX_ALPHA = 5   # rho alpha (Cholesky 11-component)
IDX_BETA  = 6   # rho beta  (Cholesky 21-component)
IDX_M3    = 7   # rho <v^3>

# 13-moment max-eigenvalue coefficient: cs = sqrt(CSCOEF * P_xx / rho)
CSCOEF = 3.0 + np.sqrt(6.0)


# Slope-limiter codes used by MUSCL reconstruction (`hll_step`
# dispatches by this integer). Keep synced with the string aliases
# accepted by `dfmm.config.SimulationConfig.limiter`.
LIMITER_MINMOD = 0
LIMITER_MC = 1
LIMITER_VAN_LEER = 2


@nb.njit(cache=True, fastmath=False, inline='always')
def minmod(a, b):
    """Minmod limiter — the most diffusive of the common options but
    always TVD.
        minmod(a, b) = a  if 0 < a < b or b < a < 0
                     = b  if 0 < b < a or a < b < 0
                     = 0  otherwise (opposite signs)
    """
    if a * b <= 0.0:
        return 0.0
    if a > 0.0:
        return a if a < b else b
    return a if a > b else b


@nb.njit(cache=True, fastmath=False, inline='always')
def mc(a, b):
    """Monotonized-central (MC) limiter — 2× less diffusive than minmod
    on smooth data, reduces to minmod near extrema.
        mc(a, b) = minmod(2*a, 2*b, (a+b)/2)
    """
    if a * b <= 0.0:
        return 0.0
    c = 0.5 * (a + b)
    two_a = 2.0 * a
    two_b = 2.0 * b
    # minmod of three same-signed values
    if a > 0.0:
        m = two_a if two_a < two_b else two_b
        return m if m < c else c
    m = two_a if two_a > two_b else two_b
    return m if m > c else c


@nb.njit(cache=True, fastmath=False, inline='always')
def van_leer(a, b):
    """Van-Leer harmonic limiter — smooth interpolation between minmod
    and MC; TVD and symmetric.
        van_leer(a, b) = 2*a*b/(a+b)  if a*b > 0
                       = 0           otherwise
    """
    if a * b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


@nb.njit(cache=True, fastmath=False, inline='always')
def limit(a, b, limiter_code):
    """Dispatch to the selected limiter by integer code."""
    if limiter_code == LIMITER_MINMOD:
        return minmod(a, b)
    if limiter_code == LIMITER_MC:
        return mc(a, b)
    if limiter_code == LIMITER_VAN_LEER:
        return van_leer(a, b)
    return minmod(a, b)  # safe fallback


@dataclass
class Workspace:
    """Pre-allocated scratch buffers for the HLL step kernel.

    Bundling them in one object lets `run_to` allocate once at entry
    and reuse across every step, eliminating per-step GC pressure
    that can dominate runtime on long simulations. The kernel
    receives the individual ndarrays as positional arguments (Numba
    can't destructure a dataclass).

    Sized to match the ghost-padded shape `(n_fields, N + 2*n_ghost)`.
    `Fleft` has one extra column to hold fluxes at all interior faces
    plus the two boundary faces when n_ghost >= 1.

    `U_L_edge` / `U_R_edge` are additional buffers only populated
    when the caller requests `reconstruction='muscl'`; they carry the
    Hancock-predicted reconstructed left/right cell-edge states. They
    are allocated zero-size when not needed (Numba sees valid arrays
    but no memory is wasted).
    """
    Unew: np.ndarray
    Fleft: np.ndarray
    U_L_edge: np.ndarray
    U_R_edge: np.ndarray

    @classmethod
    def for_padded_state(cls, U_ghost, reconstruction='first'):
        """Build a fresh workspace sized to `U_ghost.shape`.

        Pass `reconstruction='muscl'` to also allocate the
        reconstruction/predictor buffers; first-order runs get
        zero-sized placeholders.
        """
        n_fields, N_padded = U_ghost.shape
        dt = U_ghost.dtype
        Unew = np.empty((n_fields, N_padded), dtype=dt)
        Fleft = np.empty((n_fields, N_padded + 1), dtype=dt)
        if reconstruction == 'muscl':
            U_L_edge = np.empty((n_fields, N_padded), dtype=dt)
            U_R_edge = np.empty((n_fields, N_padded), dtype=dt)
        else:
            U_L_edge = np.empty((n_fields, 0), dtype=dt)
            U_R_edge = np.empty((n_fields, 0), dtype=dt)
        return cls(Unew=Unew, Fleft=Fleft,
                   U_L_edge=U_L_edge, U_R_edge=U_R_edge)


@nb.njit(cache=True, fastmath=False, inline='always')
def hll_edge_flux(Ul, Ur, cs_L, cs_R):
    """HLL numerical flux at one edge for the 8-field moment state.

    The left/right conserved states `Ul` and `Ur` are 1D length-8 views
    (typically `U[:, l]` and `U[:, r]`). `cs_L` and `cs_R` are the
    precomputed sound speeds; the caller is responsible for choosing
    the wave-speed estimator (CSCOEF * Pxx / rho with a floor is what
    every current 8-field kernel uses).

    Returns the 8 flux components as a tuple; Numba with
    `inline='always'` lowers this to the same scalar registers the
    previous hand-inlined kernels produced.
    """
    rho_L = Ul[IDX_RHO]
    rho_L_safe = rho_L if rho_L > 1e-30 else 1e-30
    u_L   = Ul[IDX_MOM]/rho_L_safe
    Pxx_L = Ul[IDX_EXX] - rho_L*u_L*u_L
    Pp_L  = Ul[IDX_PP]
    L1_L  = Ul[IDX_L1]/rho_L_safe
    a_L   = Ul[IDX_ALPHA]/rho_L_safe
    b_L   = Ul[IDX_BETA]/rho_L_safe
    Q_L   = Ul[IDX_M3] - rho_L*u_L*u_L*u_L - 3.0*u_L*Pxx_L

    rho_R = Ur[IDX_RHO]
    rho_R_safe = rho_R if rho_R > 1e-30 else 1e-30
    u_R   = Ur[IDX_MOM]/rho_R_safe
    Pxx_R = Ur[IDX_EXX] - rho_R*u_R*u_R
    Pp_R  = Ur[IDX_PP]
    L1_R  = Ur[IDX_L1]/rho_R_safe
    a_R   = Ur[IDX_ALPHA]/rho_R_safe
    b_R   = Ur[IDX_BETA]/rho_R_safe
    Q_R   = Ur[IDX_M3] - rho_R*u_R*u_R*u_R - 3.0*u_R*Pxx_R

    SL = min(u_L - cs_L, u_R - cs_R)
    SR = max(u_L + cs_L, u_R + cs_R)

    FL0 = rho_L*u_L
    FL1 = rho_L*u_L*u_L + Pxx_L
    FL2 = rho_L*u_L*u_L*u_L + 3.0*u_L*Pxx_L + Q_L
    FL3 = u_L*Pp_L
    FL4 = rho_L*L1_L*u_L
    FL5 = rho_L*a_L*u_L
    FL6 = rho_L*b_L*u_L
    # Wick fourth moment: M4 = rho u^4 + 6 u^2 Pxx + 4 u Q + 3 Pxx^2 / rho
    FL7 = rho_L*u_L**4 + 6.0*u_L*u_L*Pxx_L + 4.0*u_L*Q_L + 3.0*Pxx_L*Pxx_L/rho_L_safe

    FR0 = rho_R*u_R
    FR1 = rho_R*u_R*u_R + Pxx_R
    FR2 = rho_R*u_R*u_R*u_R + 3.0*u_R*Pxx_R + Q_R
    FR3 = u_R*Pp_R
    FR4 = rho_R*L1_R*u_R
    FR5 = rho_R*a_R*u_R
    FR6 = rho_R*b_R*u_R
    FR7 = rho_R*u_R**4 + 6.0*u_R*u_R*Pxx_R + 4.0*u_R*Q_R + 3.0*Pxx_R*Pxx_R/rho_R_safe

    if SL >= 0.0:
        return FL0, FL1, FL2, FL3, FL4, FL5, FL6, FL7
    if SR <= 0.0:
        return FR0, FR1, FR2, FR3, FR4, FR5, FR6, FR7
    invDS = 1.0/(SR - SL + 1e-30)
    F0 = (SR*FL0 - SL*FR0 + SL*SR*(Ur[IDX_RHO]   - Ul[IDX_RHO]))  *invDS
    F1 = (SR*FL1 - SL*FR1 + SL*SR*(Ur[IDX_MOM]   - Ul[IDX_MOM]))  *invDS
    F2 = (SR*FL2 - SL*FR2 + SL*SR*(Ur[IDX_EXX]   - Ul[IDX_EXX]))  *invDS
    F3 = (SR*FL3 - SL*FR3 + SL*SR*(Ur[IDX_PP]    - Ul[IDX_PP]))   *invDS
    F4 = (SR*FL4 - SL*FR4 + SL*SR*(Ur[IDX_L1]    - Ul[IDX_L1]))   *invDS
    F5 = (SR*FL5 - SL*FR5 + SL*SR*(Ur[IDX_ALPHA] - Ul[IDX_ALPHA]))*invDS
    F6 = (SR*FL6 - SL*FR6 + SL*SR*(Ur[IDX_BETA]  - Ul[IDX_BETA])) *invDS
    F7 = (SR*FL7 - SL*FR7 + SL*SR*(Ur[IDX_M3]    - Ul[IDX_M3]))   *invDS
    return F0, F1, F2, F3, F4, F5, F6, F7
