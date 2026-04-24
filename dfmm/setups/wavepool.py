"""
Wave-pool initial conditions.

Broadband random velocity field with Kolmogorov-like spectrum, periodic
domain [0, 1], uniform density and pressure. Used as the decaying
compressible-turbulence test flow for the Kramers-Moyal LES closure.

The field is

    u(x, 0) = u_0 * sum_{k=1}^{K_max} A_k * cos(2 pi k x + phi_k)

with A_k ~ k^(-alpha) (alpha = 5/6 gives E(k) ~ k^(-5/3)) and normalized
so that the RMS velocity equals u_0.

Random phases phi_k may be drawn from the RNG seed (make_wave_pool_ic) or
supplied explicitly (make_wave_pool_ic_paired), supporting Angulo-Pontzen
paired-fixed ensembles.
"""
import numpy as np


def make_wave_pool_ic(N, u0, P0=1e-2, K_max=16, alpha=5.0/6.0, rho0=1.0,
                     sigma_x0=0.02, seed=42):
    """Build a broadband random-phase wave-pool initial state.

    Parameters
    ----------
    N : int
        Number of cells.
    u0 : float
        Target RMS velocity.
    P0 : float
        Initial uniform pressure (equal for P_xx and P_perp).
    K_max : int
        Highest forced Fourier mode.
    alpha : float
        Amplitude power-law exponent: A_k ~ k^{-alpha}.
    rho0 : float
        Uniform initial density.
    sigma_x0 : float
        Initial Cholesky alpha-field (sqrt of Sigma_xx); small value so
        phase-space co-factors are not degenerate.
    seed : int
        RNG seed for phase draws.

    Returns
    -------
    U : ndarray, shape (8, N)
        Conserved-variable state vector.
    x : ndarray, shape (N,)
        Cell-center positions.
    """
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2*np.pi, size=K_max)
    return make_wave_pool_ic_paired(N, u0, phases=phases, P0=P0, K_max=K_max,
                                     alpha=alpha, rho0=rho0, sigma_x0=sigma_x0,
                                     flip=False)


def make_wave_pool_ic_paired(N, u0, phases, P0=1e-2, K_max=16, alpha=5.0/6.0,
                              rho0=1.0, sigma_x0=0.02, flip=False):
    """Build an IC with prescribed phases, optionally pi-shifted.

    The paired ensemble of Angulo & Pontzen (2016) runs each phase
    realization {phi_k} alongside the sign-flipped companion {phi_k + pi}
    so that u_pair(x) = -u(x).  Averaging over paired spectra cancels
    odd-order nonlinear contaminations to leading order.

    Parameters
    ----------
    phases : ndarray, shape (K_max,)
        Phase vector for the Fourier modes k = 1 .. K_max.
    flip : bool
        If True, add pi to every phase, yielding the paired companion.

    Returns
    -------
    U, x : same as `make_wave_pool_ic`.
    """
    x = np.linspace(0, 1, N, endpoint=False) + 0.5/N
    k = np.arange(1, K_max+1)
    Ak = k**(-alpha)
    phi = phases.copy() + (np.pi if flip else 0.0)
    u = np.zeros(N)
    for ki, Aki, phii in zip(k, Ak, phi):
        u += Aki * np.cos(2*np.pi*ki*x + phii)
    u *= u0/np.sqrt(np.mean(u**2))

    rho = np.full(N, rho0)
    Pxx = np.full(N, P0); Pp = np.full(N, P0)
    alpha_fld = np.full(N, sigma_x0); beta_fld = np.zeros(N)
    Q0 = np.zeros(N)
    M3 = rho*u**3 + 3*u*Pxx + Q0
    U = np.array([rho, rho*u, rho*u*u + Pxx, Pp, rho*x, rho*alpha_fld,
                   rho*beta_fld, M3])
    return U, x
