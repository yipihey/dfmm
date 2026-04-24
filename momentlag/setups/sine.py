"""
Cold sinusoidal velocity perturbation on uniform density.

A standard test for phase-space structure: start with u(x) = A sin(2 pi x)
on rho = rho_0, T = T_0 small, and evolve. In the cold limit the flow
develops shell crossing where phase-space folds; in the collisional limit
the same IC forms shocks. The contrast between the two is useful for
diagnosing the closure scheme's phase-space rank indicator.
"""
import numpy as np


def make_sine_ic(N, A=1.0, T0=1e-3, rho0=1.0, sigma_x0=0.02):
    """Cold-sinusoid initial state.

    Parameters
    ----------
    N : int
        Number of cells.
    A : float
        Velocity amplitude.
    T0 : float
        Thermal pressure (P_xx = P_perp = rho_0 * T0).
    rho0 : float
        Uniform density.
    sigma_x0 : float
        Initial Cholesky alpha-field.

    Returns
    -------
    U : ndarray, shape (8, N)
    x : ndarray, shape (N,)
    """
    x = np.linspace(0, 1, N, endpoint=False) + 0.5/N
    rho = np.full(N, rho0)
    u = A*np.sin(2*np.pi*x)
    Pxx = np.full(N, rho0*T0); Pp = np.full(N, rho0*T0)
    alpha_fld = np.full(N, sigma_x0); beta_fld = np.zeros(N)
    M3 = rho*u**3 + 3*u*Pxx
    U = np.array([rho, rho*u, rho*u*u + Pxx, Pp, rho*x, rho*alpha_fld,
                   rho*beta_fld, M3])
    return U, x
