"""
Sod shock-tube initial conditions.

Standard one-dimensional Riemann problem:

    (rho, u, P) = (1, 0, 1)      for x < 0.5
                  (0.125, 0, 0.1) for x >= 0.5

on the domain [0, 1] with reflective or outflow boundaries at x = 0, 1.
"""
import numpy as np


def make_sod_ic(N, sigma_x0=0.02):
    """Build a Sod-tube IC compatible with the 8-field Cholesky scheme.

    Parameters
    ----------
    N : int
        Number of cells.
    sigma_x0 : float
        Initial Cholesky alpha-field (phase-space position-variance
        square root). Small uniform value; not physically important for
        Sod, but keeps the Cholesky factor non-degenerate.

    Returns
    -------
    U : ndarray, shape (8, N)
    x : ndarray, shape (N,)
    """
    x = np.linspace(0, 1, N, endpoint=False) + 0.5/N
    rho = np.where(x < 0.5, 1.0, 0.125)
    u = np.zeros(N)
    P = np.where(x < 0.5, 1.0, 0.1)
    Pxx = P.copy()
    Pp = P.copy()
    alpha_fld = np.full(N, sigma_x0)
    beta_fld = np.zeros(N)
    M3 = rho*u**3 + 3*u*Pxx
    U = np.array([rho, rho*u, rho*u*u + Pxx, Pp, rho*x, rho*alpha_fld,
                   rho*beta_fld, M3])
    return U, x
