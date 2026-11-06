"""Tests for the two-level AMR driver and the Germano/K bookkeeping."""
import numpy as np
import pytest

from dfmm.setups.wavepool import make_wave_pool_ic
from dfmm.integrate import run_to, coarse_grain
from dfmm.config import SimulationConfig
from dfmm.amr import (TwoLevelAMR, TurbModel, run_single, restrict,
                      resolved_subcell_ke)

CFG = SimulationConfig()
NC, R = 32, 4


@pytest.fixture(scope="module")
def ic():
    Uf0, _ = make_wave_pool_ic(NC * R, u0=1.0, P0=0.1, K_max=8, seed=3)
    return coarse_grain(Uf0, R), Uf0


def _run(amr, t):
    recs = []
    while amr.t < t - 1e-12:
        recs.append(amr.step(t_stop=t))
    return recs


def test_k_off_level_step_matches_run_to(ic):
    Uc0, _ = ic
    a, _ = run_to(Uc0, t_end=0.05, save_times=[0.05], tau=CFG.tau)
    b = run_single(Uc0, 0.05, [0.05], CFG, TurbModel())
    assert np.abs(a[-1][1] - b[-1][1]).max() < 1e-12


@pytest.mark.parametrize("model", [TurbModel(),
                                   TurbModel(enabled=True, C_eps=2.0, f_K=0.1)])
def test_patch_conservation_with_reflux(ic, model):
    Uc0, Uf0 = ic
    i0, i1 = 8, 24
    amr = TwoLevelAMR(Uc=Uc0.copy(), Uf=Uf0[:, i0 * R:i1 * R].copy(),
                      i0=i0, i1=i1, r=R, cfg=CFG, model=model)
    T0 = amr.composite_totals()
    _run(amr, 0.1)
    T1 = amr.composite_totals()
    for k in T0:
        assert abs(T1[k] - T0[k]) < 1e-12, k


def test_reflux_matters(ic):
    Uc0, Uf0 = ic
    i0, i1 = 8, 24
    amr = TwoLevelAMR(Uc=Uc0.copy(), Uf=Uf0[:, i0 * R:i1 * R].copy(),
                      i0=i0, i1=i1, r=R, cfg=CFG, reflux=False)
    T0 = amr.composite_totals()
    _run(amr, 0.1)
    assert abs(amr.composite_totals()["mass"] - T0["mass"]) > 1e-8


def test_full_coverage_projection_and_defect(ic):
    Uc0, Uf0 = ic
    amr = TwoLevelAMR(Uc=Uc0.copy(), Uf=Uf0.copy(), i0=0, i1=NC, r=R, cfg=CFG)
    recs = _run(amr, 0.05)
    assert np.array_equal(restrict(amr.Uf, R), amr.Uc)
    # the defect is a genuine closure error, not identically zero
    assert np.abs(recs[-1]["eps"][1]).max() > 0


def test_germano_projection_partitions_energy(ic):
    """After projection K_c = <K_f> + K_res, so the thermal pressure
    P_xx - 2K restricts as a plain average and K_res >= 0."""
    Uc0, Uf0 = ic
    amr = TwoLevelAMR(Uc=Uc0.copy(), Uf=Uf0.copy(), i0=0, i1=NC, r=R, cfg=CFG,
                      model=TurbModel(enabled=True, C_eps=2.0, f_K=0.1))
    _run(amr, 0.05)
    assert (resolved_subcell_ke(amr.Uf, R) >= 0).all()
    def pth(U, K):
        rho = U[0]; u = U[1] / rho
        return U[2] - rho * u * u - 2 * K
    np.testing.assert_allclose(pth(amr.Uc, amr.Kc),
                               restrict(pth(amr.Uf, amr.Kf), R),
                               rtol=1e-10, atol=1e-13)


def test_dynamic_fit_independent_of_initial_guess():
    # The early C_eps shapes the K trajectory the fit then sees, leaving a
    # small path dependence that shrinks with resolution (~5% at Nc=32,
    # ~2% at 64, ~1e-5 at 256 in experiment 19).
    nc = 64
    Uf0, _ = make_wave_pool_ic(nc * R, u0=1.0, P0=0.1, K_max=8, seed=3)
    Uc0 = coarse_grain(Uf0, R)
    out = []
    for C0 in (0.3, 3.0):
        amr = TwoLevelAMR(Uc=Uc0.copy(), Uf=Uf0.copy(), i0=0, i1=nc, r=R,
                          cfg=CFG, model=TurbModel(enabled=True, C_eps=C0),
                          dynamic_C_eps=True, dynamic_f_K=True)
        _run(amr, 0.3)
        out.append((amr.model.C_eps, amr.model.f_K))
    assert abs(out[0][0] - out[1][0]) < 5e-2 * max(1.0, out[0][0])
    assert abs(out[0][1] - out[1][1]) < 1e-2
