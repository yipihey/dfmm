"""
Experiment 20: what should a refinement event do with the unresolved K?

A coarse-only run with the Semenov K model is advanced to t_ref, then the
whole domain is refined by r = 8 and the two-level hierarchy continues.
Ways to build the new fine level (`dfmm.amr.refine_from_coarse`):

  heat     smooth prolongation, K handed to the thermal pressure (what a
           hierarchy without a K field does)
  keepK    smooth prolongation, K_f = K_c - K_res (Germano-consistent, but
           the newly resolvable band starts empty)
  nyquist  the band share phi = 1 - r^(1-n) of K_c (n = 2: phi = 0.875) is
           turned into random-phase velocity fluctuations on the newly
           resolved scales 2 dx_f .. 2 dx_c (zero momentum per coarse cell,
           KE taken out of K, total energy exact)
  nyquist-cal  as nyquist, but the injected energy is phi_band * K_c on
           top of the slope-restored part, with phi_band = (KE above k_c) /
           K_res measured online on the calibration run's fine level (for a
           box-filter K most of K_res is leakage of modes just below k_c)
  front    same energy as nyquist, but placed in a tanh velocity front (a sub-grid
           shock) in each compressive coarse cell

Both the first-order HLL scheme and MUSCL-Hancock (MC limiter) are run;
(C_eps, f_K) for each scheme come from a short online Germano fit
(experiment-19 machinery). Reference: the fine DNS from t = 0 with the
same scheme. Diagnostics on the fine level after the event:
  - velocity power in the new band N_c/2 < k <= N_f/2, / DNS
  - total sub-coarse-cell KE  K_res + <K_f>  (resolved + modelled), / DNS
    K_res -- the quantity a Semenov-type star-formation criterion sees

Produces: paper/figs/amr_refinement_nyquist.png and stdout summary.
Runtime:  a few minutes (QUICK=1 for a smoke version).
"""
import os, time
import numpy as np
import matplotlib.pyplot as plt

from dfmm.setups.wavepool import make_wave_pool_ic
from dfmm.integrate import run_to, coarse_grain
from dfmm.config import SimulationConfig
from dfmm.amr import (TwoLevelAMR, TurbModel, run_single, restrict,
                      resolved_subcell_ke, refine_from_coarse, pad_periodic,
                      band_fraction)

HERE = os.path.dirname(__file__)
FIG_DIR = os.path.join(HERE, "..", "paper", "figs")
DATA_DIR = os.path.join(HERE, "..", "data")

QUICK = os.environ.get("QUICK", "0") == "1"
N_C = 64 if QUICK else 256
R = 8
N_F = N_C * R
TAU = 1e-3
SCHEMES = {"first-order HLL": SimulationConfig(tau=TAU),
           "MUSCL (MC)": SimulationConfig(tau=TAU, reconstruction="muscl",
                                          limiter="mc")}
SEEDS = [42] if QUICK else [42, 77, 101]
PHASE_DRAWS = 1 if QUICK else 2
T_REF = 0.1
LAGS = np.array([0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1])
MODES = {"heat": dict(mode="heat"),
         "keepK": dict(mode="keepK"),
         "nyquist": dict(mode="nyquist", n_slope=2.0),
         "nyquist-cal": dict(mode="nyquist", n_slope=2.0, phi_band_only=True),
         "front": dict(mode="nyquist", shape="front", front_width=1.0)}


def calibrate(cfg, seed=42, t_end=0.5):
    """Short online 2-parameter Germano fit of (C_eps, f_K) for a scheme,
    plus the band share phi_band measured on the fine level."""
    Uf0, _ = make_wave_pool_ic(N_F, u0=1.0, P0=0.1, K_max=16, seed=seed)
    amr = TwoLevelAMR(Uc=coarse_grain(Uf0, R), Uf=Uf0, i0=0, i1=N_C, r=R,
                      cfg=cfg, model=TurbModel(enabled=True, C_eps=3.0, f_K=0.1),
                      dynamic_C_eps=True, dynamic_f_K=True)
    fb = []
    for tt in np.arange(0.05, t_end + 1e-9, 0.05):
        while amr.t < tt - 1e-12: amr.step(t_stop=tt)
        fb.append(band_fraction(amr.Uf, R))
    h = np.array(amr.C_eps_history); f = np.array(amr.f_K_history)
    late = h[:, 0] > 0.1
    return (float(np.median(h[late, 1])), float(np.median(f[late, 1])),
            float(np.median(fb)))


def spectrum(U):
    u = U[1] / U[0]
    return np.abs(np.fft.rfft(u) / u.size)**2   # E(k), k = 0..N/2


def band_power(U):
    E = spectrum(U)
    k = np.arange(E.size)
    return E[(k > N_C // 2)].sum()


def run_scheme(cfg, C_eps, f_K, phi_band):
    model = lambda: TurbModel(enabled=True, C_eps=C_eps, f_K=f_K)
    res = {m: {"band": [], "ktot": [], "spec": []} for m in list(MODES) + ["dns"]}
    for seed in SEEDS:
        Uf0, _ = make_wave_pool_ic(N_F, u0=1.0, P0=0.1, K_max=16, seed=seed)
        Uc0 = coarse_grain(Uf0, R)
        times = [float(T_REF + l) for l in LAGS]
        dns = [U for _, U in run_to(Uf0, t_end=times[-1], save_times=times,
                                    cfg=cfg)[0][1:]]
        res["dns"]["band"].append([band_power(U) for U in dns])
        res["dns"]["ktot"].append([resolved_subcell_ke(U, R).mean() for U in dns])
        res["dns"]["spec"].append(spectrum(dns[list(LAGS).index(0.01)]))
        _, Uc, Kc = run_single(Uc0, T_REF, [T_REF], cfg, model())[-1]
        for name, kw in MODES.items():
            if kw.get("phi_band_only"):
                kw = dict(kw, phi=phi_band)
            draws = PHASE_DRAWS if kw.get("shape", "random") == "random" \
                and kw["mode"] == "nyquist" else 1
            for d in range(draws):
                Uf, Kf, _ = refine_from_coarse(
                    pad_periodic(Uc, 1), pad_periodic(Kc, 1), R,
                    rng=np.random.default_rng(1000 * seed + d), **kw)
                amr = TwoLevelAMR(Uc=restrict(Uf, R), Uf=Uf, i0=0, i1=N_C, r=R,
                                  cfg=cfg, model=model(),
                                  Kc0=restrict(Kf, R) + resolved_subcell_ke(Uf, R),
                                  Kf0=Kf)
                amr.t = T_REF
                band, ktot = [], []
                for l in LAGS:
                    while amr.t < T_REF + l - 1e-12:
                        amr.step(t_stop=T_REF + l)
                    band.append(band_power(amr.Uf))
                    ktot.append(resolved_subcell_ke(amr.Uf, R).mean()
                                + amr.Kf.mean())
                    if abs(l - 0.01) < 1e-12:
                        res[name]["spec"].append(spectrum(amr.Uf))
                res[name]["band"].append(band)
                res[name]["ktot"].append(ktot)
    return res


def main():
    Uw, _ = make_wave_pool_ic(32, u0=0.1)
    run_to(Uw, t_end=0.002, save_times=[0.002])
    allres = {}
    for sname, cfg in SCHEMES.items():
        t0 = time.time()
        C_eps, f_K, phi_band = calibrate(cfg)
        res = run_scheme(cfg, C_eps, f_K, phi_band)
        allres[sname] = res
        dB = np.mean(res["dns"]["band"], 0); dK = np.mean(res["dns"]["ktot"], 0)
        print(f"\n=== {sname}: C_eps = {C_eps:.2f}, f_K = {f_K:.3f}, "
              f"phi_band = {phi_band:.3f}, refinement at t = {T_REF} ({time.time() - t0:.0f}s) ===")
        print("  lag after refinement     " + "  ".join(f"{l:>6g}" for l in LAGS))
        for m in MODES:
            res[m]["band_r"] = np.mean(res[m]["band"], 0) / dB
            res[m]["ktot_r"] = np.mean(res[m]["ktot"], 0) / dK
            print(f"  {m:11s} band / DNS     " + "  ".join(f"{x:6.3f}" for x in res[m]["band_r"]))
            print(f"  {'':11s} K_res+<K_f>/DNS " + "  ".join(f"{x:6.3f}" for x in res[m]["ktot_r"]))

    cols = {"heat": "C0", "keepK": "C2", "nyquist": "C3", "nyquist-cal": "C4",
            "front": "C1"}
    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    k = np.arange(N_F // 2 + 1)
    for row, (sname, res) in enumerate(allres.items()):
        for m in MODES:
            ax[row, 0].semilogy(LAGS, res[m]["band_r"], "o-", color=cols[m], label=m)
            ax[row, 1].semilogy(LAGS, res[m]["ktot_r"], "o-", color=cols[m], label=m)
            ax[row, 2].loglog(k[1:], np.mean(res[m]["spec"], 0)[1:], color=cols[m],
                              lw=0.9, label=m)
        ax[row, 2].loglog(k[1:], np.mean(res["dns"]["spec"], 0)[1:], "k", lw=1.2,
                          label="DNS")
        ax[row, 2].axvline(N_C // 2, color="gray", ls=":", lw=1)
        ax[row, 2].set_ylim(1e-14, None); ax[row, 2].set_xlabel("k")
        ax[row, 2].set_title(f"{sname}: velocity spectrum 0.01 after refinement",
                             fontsize=10)
        for c, t in zip((0, 1), ("velocity power in the new band "
                                 r"$N_c/2<k\leq N_f/2$, / DNS",
                                 r"total sub-coarse KE $K_{\rm res}+\langle K_f\rangle$, / DNS")):
            a = ax[row, c]
            a.axhline(1, color="k", lw=0.8)
            a.set_xlabel("time since refinement")
            a.set_title(f"{sname}: {t}", fontsize=10)
        for a in ax[row]:
            a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
    out = os.path.join(FIG_DIR, "amr_refinement_nyquist.png")
    fig.savefig(out, dpi=130)
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
