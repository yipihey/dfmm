"""
Time integration and coarse-graining utilities.

`run_to` is a thin driver over the unified `hll_step` kernel in
`dfmm.schemes.cholesky`, with boundary conditions applied via the
ghost-cell machinery in `dfmm.schemes.boundaries`. Each step pads the
(n_fields, N) interior state to (n_fields, N + 2*n_ghost), fills the
ghosts with the configured BC, invokes the kernel, and unpads the
result for storage/snapshot.

The `coarse_grain` helper box-averages a fine-resolution state onto a
coarser grid. Because every field in the 8-field state is conservative
or additive (rho, rho u, E_xx, P_perp, rho L_1, rho alpha, rho beta,
M_3), simple averaging is the correct filter operator.
"""
from dataclasses import replace
import numpy as np

from .config import SimulationConfig
from .schemes._common import Workspace
from .schemes.boundaries import (pad_with_ghosts, unpad_ghosts,
                                  apply_mixed)
from .schemes.cholesky import hll_step, hll_step_muscl, max_signal_speed

# String -> integer code for the limiter dispatch inside hll_step_muscl.
_LIMITER_CODES = {'minmod': 0, 'mc': 1, 'van_leer': 2}


def run_to(U, t_end, cfl=None, tau=None, save_times=None, checkpoint_dt=None,
           bc="periodic", cfg=None):
    """Integrate a state forward to `t_end`, saving at requested times.

    Parameters
    ----------
    U : ndarray, shape (8, N)
        Initial state (interior only; ghost padding is handled
        internally).
    t_end : float
        Final integration time.
    cfl : float, optional
        CFL coefficient; overrides `cfg.cfl` if both given.
    tau : float, optional
        BGK relaxation time; overrides `cfg.tau`.
    save_times : array-like or None
        Times at which to record snapshots.
    checkpoint_dt : float or None
        If given, overrides `save_times` with uniform spacing.
    bc : {"periodic", "transmissive", "reflective"}, default "periodic"
        Backward-compat shortcut — sets both `cfg.bc_left` and
        `cfg.bc_right` to this value. Ignored if `cfg` is passed
        with explicit BC fields.
    cfg : SimulationConfig, optional
        Full configuration. If None, one is built from the `cfl`,
        `tau`, and `bc` shortcuts (plus all defaults).

    Returns
    -------
    snapshots : list of (float, ndarray)
        (time, state) pairs, including the t=0 initial snapshot.
        States are the unpadded (8, N) interior view.
    nsteps : int
        Number of HLL steps taken.
    """
    if cfg is None:
        kw = {}
        if cfl is not None: kw['cfl'] = cfl
        if tau is not None: kw['tau'] = tau
        kw['bc_left'] = bc
        kw['bc_right'] = bc
        cfg = SimulationConfig(**kw)
    else:
        # Allow the positional shortcuts to override corresponding
        # fields on an explicitly-supplied cfg.
        overrides = {}
        if cfl is not None: overrides['cfl'] = cfl
        if tau is not None: overrides['tau'] = tau
        if overrides:
            cfg = replace(cfg, **overrides)

    n_ghost = cfg.n_ghost
    dx = 1.0 / U.shape[1]
    t = 0.0
    snapshots = [(0.0, U.copy())]
    if save_times is None:
        assert checkpoint_dt is not None, "pass save_times or checkpoint_dt"
        save_times = np.arange(checkpoint_dt, t_end + 1e-9, checkpoint_dt)
    save_idx = 0
    nsteps = 0

    # Pad to ghost layout once; allocate the workspace once; ping-
    # pong `U_curr` / `U_next` across steps to avoid per-step
    # allocations.
    U_curr = pad_with_ghosts(U, n_ghost)
    ws = Workspace.for_padded_state(U_curr, reconstruction=cfg.reconstruction)
    U_next = ws.Unew
    limiter_code = _LIMITER_CODES[cfg.limiter]

    while t < t_end and save_idx < len(save_times):
        next_save = save_times[save_idx]
        # Apply BCs so max_signal_speed sees a consistent field
        # (and so the Liouville-source du/dx stencil at interior
        # edges uses the correct ghost neighbour).
        apply_mixed(U_curr, n_ghost, cfg.bc_left, cfg.bc_right,
                    state_left=cfg.bc_state_left,
                    state_right=cfg.bc_state_right)
        smax = max_signal_speed(unpad_ghosts(U_curr, n_ghost))
        dt = min(cfg.cfl * dx / smax, next_save - t, t_end - t)
        if dt <= 1e-14:
            snapshots.append((t, unpad_ghosts(U_curr, n_ghost).copy()))
            save_idx += 1
            continue
        if cfg.reconstruction == 'first':
            hll_step(U_curr, dx, dt, cfg.tau, n_ghost,
                     cfg.rho_floor, cfg.alpha_floor,
                     cfg.realizability_headroom,
                     U_next, ws.Fleft)
        else:  # 'muscl'
            hll_step_muscl(U_curr, dx, dt, cfg.tau, n_ghost,
                           cfg.rho_floor, cfg.alpha_floor,
                           cfg.realizability_headroom,
                           limiter_code,
                           U_next, ws.Fleft, ws.U_L_edge, ws.U_R_edge)
        # Swap roles — U_next holds the new state; reuse the old
        # U_curr slot as scratch on the next iteration.
        U_curr, U_next = U_next, U_curr
        t += dt
        nsteps += 1
        if t >= next_save - 1e-12:
            snapshots.append((t, unpad_ghosts(U_curr, n_ghost).copy()))
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
