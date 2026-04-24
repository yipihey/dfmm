"""
Steady-state shock initial conditions and inflow-outflow stepper.

For the steady-shock tests we set up a sharp discontinuity at x = 0.5
between Rankine-Hugoniot-consistent upstream and downstream states, then
let the flow relax to a steady profile under inflow (Dirichlet) /
outflow (extrapolation) boundary conditions.

This module exposes:
    rankine_hugoniot(rho1, u1, P1)
        Analytic downstream state.
    build_state_from_primitives(rho, u, P)
        Package (rho, u, P) arrays into the 8-field conserved state.
    step_inflow_outflow(U, dx, dt, tau, U_inflow)
        Single-step update that reinstates the upstream state in the
        leftmost two cells after every transmissive HLL step.
    run_steady_shock(M1, tau, N, t_end)
        Convenience driver: builds the IC, steps to steady state, returns.
"""
import numpy as np
import numba as nb

from ..schemes.cholesky import max_signal_speed, hll_step_transmissive

GAMMA = 5.0/3.0


def rankine_hugoniot(rho1, u1, P1, gamma=GAMMA):
    """Downstream state (rho2, u2, P2) given upstream (rho1, u1, P1).

    Returns
    -------
    rho2, u2, P2, M1 : float
        Post-shock density, velocity, pressure; upstream Mach number.
    """
    c_s1 = np.sqrt(gamma * P1/rho1)
    M1 = u1 / c_s1
    rho_ratio = (gamma + 1) * M1**2 / ((gamma - 1) * M1**2 + 2)
    P_ratio = (2 * gamma * M1**2 - (gamma - 1)) / (gamma + 1)
    rho2 = rho1 * rho_ratio
    u2 = u1 / rho_ratio
    P2 = P1 * P_ratio
    return rho2, u2, P2, M1


def build_state_from_primitives(rho, u, P, sigma_x0=0.02):
    """Assemble the 8-field conserved state from primitive arrays.

    Assumes isotropic Maxwellian: P_xx = P_perp = P and Q = 0.
    """
    N = len(rho)
    Pxx = P.copy()
    Pp = P.copy()
    alpha0 = np.full(N, sigma_x0)
    beta0 = np.zeros(N)
    x = np.linspace(0, 1, N)
    M30 = rho*u**3 + 3.0*u*Pxx
    U = np.array([rho, rho*u, rho*u*u + Pxx, Pp, rho*x, rho*alpha0,
                   rho*beta0, M30])
    return U


@nb.njit(cache=True, fastmath=True)
def step_inflow_outflow(U, dx, dt, tau, U_inflow):
    """One time step with transmissive HLL + upstream Dirichlet reinject.

    The transmissive stepper in cholesky.py provides the underlying
    HLL update; this wrapper overwrites cells 0 and 1 with the upstream
    state after each step, which is numerically equivalent to inflow
    Dirichlet boundary conditions with a thin sponge.
    """
    Unew = hll_step_transmissive(U, dx, dt, tau)
    for k in range(8):
        Unew[k, 0] = U_inflow[k]
        Unew[k, 1] = U_inflow[k]
    return Unew


def run_steady_shock(M1, tau, N=400, t_end=10.0, cfl=0.3,
                      rho1=1.0, P1=1.0, sigma_x0=0.02):
    """Run the inflow-outflow steady-shock test to steady state.

    Parameters
    ----------
    M1 : float
        Upstream Mach number.
    tau : float
        BGK relaxation time.
    N : int
        Number of cells on [0, 1].
    t_end : float
        Wall-time to run; should exceed the cascade time to steady.
    cfl : float
        CFL coefficient.
    rho1, P1 : float
        Upstream density and pressure.
    sigma_x0 : float
        Initial Cholesky alpha-field.

    Returns
    -------
    x : (N,) cell centers.
    U : (8, N) final state.
    nsteps : int, number of integration steps taken.
    """
    c_s1 = np.sqrt(GAMMA * P1/rho1)
    u1 = M1 * c_s1
    rho2, u2, P2, _ = rankine_hugoniot(rho1, u1, P1)
    x = np.linspace(0, 1, N)
    rho = np.where(x < 0.5, rho1, rho2)
    u = np.where(x < 0.5, u1, u2)
    P = np.where(x < 0.5, P1, P2)
    U = build_state_from_primitives(rho, u, P, sigma_x0=sigma_x0)
    dx = 1.0/(N-1)
    U_inflow = np.array([rho1, rho1*u1, rho1*u1*u1 + P1, P1,
                          rho1*0.0, rho1*sigma_x0, 0.0,
                          rho1*u1**3 + 3*u1*P1])
    t = 0.0; nsteps = 0
    while t < t_end:
        smax = max_signal_speed(U)
        dt = min(cfl*dx/smax, t_end - t)
        if dt <= 0: break
        U = step_inflow_outflow(U, dx, dt, tau, U_inflow)
        t += dt; nsteps += 1
    return x, U, nsteps
