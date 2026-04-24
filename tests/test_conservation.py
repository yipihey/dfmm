"""Conservation smoke tests.

Verify that the integrator preserves mass, momentum, and energy to
machine precision in periodic flows, and that the energy-conservative
noise scheme preserves total energy to machine precision as well.
"""
import numpy as np
import pytest

from dfmm.setups.wavepool import make_wave_pool_ic
from dfmm.integrate import run_to
from dfmm.closure.noise_model import run_noise, total_energy


def _totals(U):
    return float(U[0].sum()), float(U[1].sum()), float(total_energy(U).sum())


def test_periodic_conservation():
    """Mass, momentum, and energy are conserved by the periodic stepper."""
    U0, _ = make_wave_pool_ic(64, u0=1.0, P0=0.1, seed=42)
    M0, P0, E0 = _totals(U0)
    snaps, _ = run_to(U0, t_end=0.2, save_times=[0.1, 0.2], tau=1e-3)
    for t, U in snaps[1:]:
        M, P, E = _totals(U)
        assert abs(M - M0) < 1e-10, f"mass drift {abs(M - M0)} at t={t}"
        assert abs(P - P0) < 1e-10, f"momentum drift {abs(P - P0)} at t={t}"
        assert abs(E - E0) < 1e-10, f"energy drift {abs(E - E0)} at t={t}"


def test_noise_scheme_energy_conservation():
    """The noise-augmented scheme preserves total energy to machine precision.

    Noise injection changes momentum per cell, but internal-energy debits
    are applied such that the total energy (KE + IE) per cell is unchanged
    up to floating-point roundoff.
    """
    U0, _ = make_wave_pool_ic(64, u0=1.0, P0=0.1, seed=42)
    E0 = float(total_energy(U0).sum())
    snaps = run_noise(U0.copy(), t_end=0.2, save_times=[0.1, 0.2],
                      C_A=0.34, C_B=0.55, ell_corr=2.0, seed=0, tau=1e-3)
    for t, U in snaps[1:]:
        E = float(total_energy(U).sum())
        drift = abs(E - E0) / max(abs(E0), 1e-30)
        assert drift < 1e-10, f"noise scheme energy drift {drift:.3e} at t={t}"
