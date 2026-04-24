"""
Time integration and coarse-graining utilities.

Integrators are thin drivers over the per-step kernels in
`dfmm.schemes`: they handle CFL selection, save-time scheduling,
and snapshot bookkeeping while delegating the actual update to a
scheme's `hll_step_*` function.

The `coarse_grain` helper box-averages a fine-resolution state onto a
coarser grid. Because every field in the 8-field state is conservative
or additive (rho, rho u, E_xx, P_perp, rho L_1, rho alpha, rho beta,
M_3), simple averaging is the correct filter operator.
"""
import numpy as np

from .schemes.cholesky import (hll_step_periodic, hll_step_transmissive,
                                max_signal_speed)


_STEPPERS = {
    "periodic": hll_step_periodic,
    "transmissive": hll_step_transmissive,
}


def run_to(U, t_end, cfl=0.3, tau=1e-3, save_times=None, checkpoint_dt=None,
           bc="periodic"):
    """Integrate a state forward to `t_end`, saving at requested times.

    The per-step update is dispatched by `bc`:

        bc="periodic"      periodic HLL + BGK (default; wave-pool, sine)
        bc="transmissive"  zero-gradient outflow on both ends (Sod-like)

    Parameters
    ----------
    U : ndarray, shape (8, N)
        Initial state.
    t_end : float
        Final integration time.
    cfl : float, default 0.3
        CFL coefficient.
    tau : float, default 1e-3
        BGK relaxation time.
    save_times : array-like or None
        Times at which to record snapshots. Either this or
        `checkpoint_dt` must be given.
    checkpoint_dt : float or None
        If given, overrides `save_times` with uniform spacing.
    bc : {"periodic", "transmissive"}
        Boundary-condition variant (see above).

    Returns
    -------
    snapshots : list of (float, ndarray)
        (time, state) pairs, including the t=0 initial snapshot.
    nsteps : int
        Number of HLL steps taken.
    """
    try:
        step = _STEPPERS[bc]
    except KeyError:
        raise ValueError(f"unknown bc={bc!r}; use one of {list(_STEPPERS)}")
    dx = 1.0/U.shape[1]
    t = 0.0
    snapshots = [(0.0, U.copy())]
    if save_times is None:
        assert checkpoint_dt is not None, "pass save_times or checkpoint_dt"
        save_times = np.arange(checkpoint_dt, t_end + 1e-9, checkpoint_dt)
    save_idx = 0
    nsteps = 0
    while t < t_end and save_idx < len(save_times):
        next_save = save_times[save_idx]
        smax = max_signal_speed(U)
        dt = min(cfl*dx/smax, next_save - t, t_end - t)
        if dt <= 1e-14:
            snapshots.append((t, U.copy()))
            save_idx += 1
            continue
        U = step(U, dx, dt, tau)
        t += dt
        nsteps += 1
        if t >= next_save - 1e-12:
            snapshots.append((t, U.copy()))
            save_idx += 1
    return snapshots, nsteps


def coarse_grain(U_fine, factor):
    """Box-average a fine-resolution state onto a coarser grid.

    Parameters
    ----------
    U_fine : ndarray, shape (n_fields, N_fine)
    factor : int
        Refinement ratio; must divide N_fine.

    Returns
    -------
    U_coarse : ndarray, shape (n_fields, N_fine // factor)
    """
    n_fields, N_fine = U_fine.shape
    N_coarse = N_fine // factor
    assert N_coarse * factor == N_fine, (
        f"factor {factor} does not divide N_fine {N_fine}")
    return U_fine.reshape(n_fields, N_coarse, factor).mean(axis=2)
