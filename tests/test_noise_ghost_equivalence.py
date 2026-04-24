"""Verify that the ghost-cell `hll_step_noise` reproduces the legacy
`hll_step_noise_econsv` bit-identically under the periodic
boundary condition.

Gates the Phase 3d migration of the noise kernel the same way the
wave-pool regression snapshot gates cholesky's migration.
"""
import numpy as np
import pytest

from dfmm.setups.wavepool import make_wave_pool_ic
from dfmm.closure.noise_model import (hll_step_noise, hll_step_noise_econsv,
                                       smooth_gaussian_periodic,
                                       total_energy)
from dfmm.schemes._common import Workspace
from dfmm.schemes.boundaries import pad_with_ghosts, apply_periodic, unpad_ghosts
from dfmm.schemes.cholesky import max_signal_speed


def test_ghost_cell_noise_step_bit_identical_to_legacy():
    """One noise step via the ghost-cell kernel matches the legacy
    periodic kernel to machine precision for the same (U, dt, eta)."""
    U, _ = make_wave_pool_ic(N=128, u0=1.0, P0=0.1, seed=42)
    dx = 1.0 / U.shape[1]
    smax = max_signal_speed(U)
    dt = 0.3 * dx / smax

    rng = np.random.default_rng(0)
    eta_white = rng.laplace(scale=1.0 / np.sqrt(2.0), size=128)
    eta = smooth_gaussian_periodic(eta_white, 2.0)

    U_legacy = hll_step_noise_econsv(
        U, dx, dt, tau=1e-3, C_A=0.34, C_B=0.55, eta_draw=eta)

    U_ghost = pad_with_ghosts(U, 1)
    apply_periodic(U_ghost, 1)
    ws = Workspace.for_padded_state(U_ghost)
    hll_step_noise(U_ghost, dx, dt, 1e-3, 1,
                   1e-30, 1e-15, 0.999, 1e-8, 0.25,
                   0.34, 0.55, eta,
                   ws.Unew, ws.Fleft)
    U_new = unpad_ghosts(ws.Unew, 1).copy()

    assert np.array_equal(U_legacy, U_new), \
        f"max abs diff {np.max(np.abs(U_legacy - U_new)):.3e}"


def test_run_noise_preserves_total_energy():
    """After the ghost-cell migration, run_noise still preserves total
    energy to machine precision (same invariant as the legacy path's
    test_noise_scheme_energy_conservation in test_conservation.py)."""
    from dfmm.closure.noise_model import run_noise
    U, _ = make_wave_pool_ic(64, u0=1.0, P0=0.1, seed=42)
    E0 = float(total_energy(U).sum())
    snaps = run_noise(U.copy(), t_end=0.05, save_times=[0.05],
                      C_A=0.34, C_B=0.55, ell_corr=2.0, seed=0, tau=1e-3)
    E_final = float(total_energy(snaps[-1][1]).sum())
    drift = abs(E_final - E0) / max(abs(E0), 1e-30)
    assert drift < 1e-10, f"noise-run total-energy drift {drift:.3e}"
