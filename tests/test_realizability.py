"""Realizability smoke tests.

The Cholesky-factored 2x2 phase-space covariance is realizable by
construction: Sigma = L L^T with L lower triangular and non-negative
diagonal. These tests check that the scheme actually maintains this
property through a representative flow, by verifying that
Sigma_xx = alpha^2 >= 0 and gamma^2 = Sigma_vv - beta^2 >= 0 pointwise.
"""
import numpy as np
import pytest

from momentlag.setups.wavepool import make_wave_pool_ic
from momentlag.setups.sod import make_sod_ic
from momentlag.integrate import run_to
from momentlag.diagnostics import extract_diagnostics


def test_wavepool_realizability():
    """Realizability diagnostics stay non-negative in wave-pool evolution."""
    U0, _ = make_wave_pool_ic(64, u0=1.0, P0=0.1, seed=42)
    snaps, _ = run_to(U0, t_end=0.2, save_times=[0.1, 0.2], tau=1e-3)
    for t, U in snaps:
        d = extract_diagnostics(U)
        assert np.all(d['rho'] > 0), f"negative density at t={t}"
        assert np.all(d['Pxx'] > 0), f"negative P_xx at t={t}"
        assert np.all(d['Pp']  > 0), f"negative P_perp at t={t}"
        assert np.all(d['gamma'] >= 0), f"negative gamma at t={t}"


def test_sod_realizability():
    """Realizability diagnostics hold through a Sod shock."""
    U0, _ = make_sod_ic(128)
    snaps, _ = run_to(U0, t_end=0.15, save_times=[0.1, 0.15],
                       tau=1e-3, bc="transmissive")
    for t, U in snaps:
        d = extract_diagnostics(U)
        assert np.all(d['rho'] > 0), f"negative density at t={t}"
        assert np.all(d['Pxx'] > 0), f"negative P_xx at t={t}"
        assert np.all(d['Pp']  > 0), f"negative P_perp at t={t}"
        assert np.all(d['gamma'] >= 0), f"negative gamma at t={t}"
