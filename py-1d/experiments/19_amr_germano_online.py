"""
Experiment 19: two-level AMR as an online closure laboratory.

Three parts, all on the decaying wave pool of experiments 10-14
(u0 = 1, P0 = 0.1, K_max = 16, tau = 1e-3, N_coarse = 256, r = 8):

  1. AMR verification. Static fine patch with Berger-Oliger subcycling,
     time-interpolated prolonged ghosts, projection and refluxing:
     composite mass / momentum / energy conservation with and without
     refluxing; full-coverage hierarchy reproduces the fine run.

  2. Online Kramers-Moyal calibration. Because projection resets the
     coarse state to R(U_f) every coarse step, the pre-projection defect
     eps = U_c^{n+1} - R(U_f^{n+1}) IS the experiment-11 closure error
     with interval = one coarse step. We fit C_A, C_B from (a) the
     per-step defect, (b) "shadow" coarse runs launched from the
     projected state every 0.01 (the experiment-11 interval, but online),
     (c) the experiment-11 offline protocol on the same seeds, and
     (d) the per-step defect restricted to a half-domain patch.

  3. Semenov-style subgrid K with Germano bookkeeping. Projection sets
     K_c <- <K_f> + K_res (energy-conserving); the pre-projection
     mismatch calibrates C_eps dynamically. A-posteriori test: coarse-only
     runs with and without K against the filtered fine DNS, on the
     pressure anisotropy <P_xx - P_perp> (restriction deposits sub-cell
     bulk KE into P_xx; without K the coarse BGK isotropizes it as heat).

Produces: paper/figs/amr_germano_online.png, data/amr_online_params.npz,
          stdout summary.
Runtime:  a few minutes (set QUICK=1 for a ~30 s smoke version).
"""
import os, time
import numpy as np
import matplotlib.pyplot as plt

from dfmm.setups.wavepool import make_wave_pool_ic
from dfmm.integrate import run_to, coarse_grain
from dfmm.config import SimulationConfig
from dfmm.amr import (TwoLevelAMR, TurbModel, run_single, restrict,
                      resolved_subcell_ke)

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")
FIG_DIR = os.path.join(HERE, "..", "paper", "figs")
os.makedirs(FIG_DIR, exist_ok=True)

QUICK = os.environ.get("QUICK", "0") == "1"
N_C = 64 if QUICK else 256
R = 8
TAU = 1e-3
U0, P0, KMAX = 1.0, 0.1, 16
T_END = 0.3 if QUICK else 1.0
DT_WIN = 0.01
SEEDS = [42] if QUICK else [42, 77, 101]
CFG = SimulationConfig(tau=TAU)


# ----------------------------------------------------------------------
def fit_km(eps, pA, pB, n_bins=20):
    """Experiment-11 estimator: binned drift regression, then binned
    residual-variance regression. Returns C_A, C_B, floor, kurtosis."""
    q = np.quantile(pA, np.linspace(0, 1, n_bins + 1))
    xc, ym, ys = [], [], []
    for j in range(n_bins):
        m = (pA >= q[j]) & (pA < q[j + 1])
        if m.sum() < 5: continue
        xc.append(pA[m].mean()); ym.append(eps[m].mean())
        ys.append(eps[m].std() / np.sqrt(m.sum()))
    xc, ym, ys = map(np.array, (xc, ym, ys))
    w = 1 / np.maximum(ys**2, 1e-30)
    C_A = float(np.sum(w * xc * ym) / np.sum(w * xc**2))
    res = eps - C_A * pA
    msk = pB > 1e-12 * max(pB.max(), 1e-300)
    q = np.quantile(pB[msk], np.linspace(0, 1, n_bins + 1))
    xb, yb = [], []
    for j in range(n_bins):
        m = (pB >= q[j]) & (pB < q[j + 1])
        if m.sum() < 5: continue
        xb.append(pB[m].mean()); yb.append(res[m].var())
    xb, yb = np.array(xb), np.array(yb)
    A = np.vstack([xb, np.ones_like(xb)]).T
    slope, icpt = np.linalg.lstsq(A, yb, rcond=None)[0]
    rn = res / res.std()
    return dict(C_A=C_A, C_B=float(np.sqrt(max(slope, 0))),
                floor=float(max(icpt, 0)), kurt=float(np.mean(rn**4) - 3),
                n=int(eps.size))


def samples_from_rec(rec, dxc):
    rho = rec["Uc_n"][0]
    return rec["eps"][1], rho * rec["dudx_n"] * rec["dt"], \
        rho**2 * np.maximum(-rec["dudx_n"], 0) * rec["dt"]


def dudx_periodic(U, dx):
    u = U[1] / U[0]
    return (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)


def make_ic(seed):
    Uf0, _ = make_wave_pool_ic(N_C * R, u0=U0, P0=P0, K_max=KMAX, seed=seed)
    return coarse_grain(Uf0, R), Uf0


# ----------------------------------------------------------------------
def part1_verification():
    print("\n=== Part 1: AMR verification ===")
    Uc0, Uf0 = make_ic(SEEDS[0])
    i0, i1 = N_C // 4, 3 * N_C // 4
    out = {}
    for reflux in (True, False):
        amr = TwoLevelAMR(Uc=Uc0.copy(), Uf=Uf0[:, i0 * R:i1 * R].copy(),
                          i0=i0, i1=i1, r=R, cfg=CFG, reflux=reflux)
        T0 = amr.composite_totals()
        while amr.t < 0.3 - 1e-12: amr.step(t_stop=0.3)
        T1 = amr.composite_totals()
        out[reflux] = {k: abs(T1[k] - T0[k]) for k in T0}
        print(f"  patch, reflux={reflux!s:5}: |d mass|={out[reflux]['mass']:.1e}"
              f"  |d mom|={out[reflux]['mom']:.1e}  |d E|={out[reflux]['energy']:.1e}")
    amr = TwoLevelAMR(Uc=Uc0.copy(), Uf=Uf0.copy(), i0=0, i1=N_C, r=R, cfg=CFG)
    while amr.t < 0.1 - 1e-12: amr.step(t_stop=0.1)
    ref, _ = run_to(Uf0, t_end=amr.t, save_times=[amr.t], tau=TAU)
    d = np.abs(ref[-1][1] - amr.Uf).max()
    print(f"  full coverage vs standalone fine run, max|dU| at t=0.1: {d:.2e}"
          "  (differs only through the coarse-limited time step)")
    return out


# ----------------------------------------------------------------------
def part2_online_km():
    print("\n=== Part 2: online Kramers-Moyal calibration ===")
    S = {k: ([], [], []) for k in ("online_step", "online_window",
                                    "offline_exp11", "online_patch")}
    dns_snaps = {}
    for seed in SEEDS:
        t0 = time.time()
        Uc0, Uf0 = make_ic(seed)
        # (a)+(b) full coverage: per-step defects + shadow windows
        amr = TwoLevelAMR(Uc=Uc0.copy(), Uf=Uf0.copy(), i0=0, i1=N_C,
                          r=R, cfg=CFG)
        snaps = [(0.0, amr.Uc.copy(), np.zeros(N_C))]
        for tw in np.arange(DT_WIN, T_END + 1e-9, DT_WIN):
            U_start = amr.Uc.copy()
            while amr.t < tw - 1e-12:
                rec = amr.step(t_stop=tw)
                for L, v in zip(S["online_step"], samples_from_rec(rec, amr.dxc)):
                    L.append(v)
            snaps.append((amr.t, amr.Uc.copy(), resolved_subcell_ke(amr.Uf, R)))
            sh = run_single(U_start, DT_WIN, [DT_WIN], CFG, TurbModel())
            dd = dudx_periodic(U_start, amr.dxc)
            S["online_window"][0].append(sh[-1][1][1] - amr.Uc[1])
            S["online_window"][1].append(U_start[0] * dd * DT_WIN)
            S["online_window"][2].append(U_start[0]**2 * np.maximum(-dd, 0) * DT_WIN)
        dns_snaps[seed] = snaps
        # (c) offline experiment-11 protocol on a standalone fine run
        save_times = np.arange(DT_WIN, T_END + 1e-9, DT_WIN)
        sf, _ = run_to(Uf0, t_end=T_END, tau=TAU, save_times=save_times)
        for i in range(len(sf) - 1):
            Ucs = coarse_grain(sf[i][1], R); Uct = coarse_grain(sf[i + 1][1], R)
            sc, _ = run_to(Ucs, t_end=sf[i + 1][0] - sf[i][0], tau=TAU,
                           save_times=[sf[i + 1][0] - sf[i][0]])
            dd = np.gradient(Ucs[1] / Ucs[0], 1.0 / N_C)
            S["offline_exp11"][0].append(sc[-1][1][1] - Uct[1])
            S["offline_exp11"][1].append(Ucs[0] * dd * DT_WIN)
            S["offline_exp11"][2].append(Ucs[0]**2 * np.maximum(-dd, 0) * DT_WIN)
        # (d) half-domain patch, per-step defect
        i0, i1 = N_C // 4, 3 * N_C // 4
        amr = TwoLevelAMR(Uc=Uc0.copy(), Uf=Uf0[:, i0 * R:i1 * R].copy(),
                          i0=i0, i1=i1, r=R, cfg=CFG)
        while amr.t < T_END - 1e-12:
            rec = amr.step(t_stop=T_END)
            for L, v in zip(S["online_patch"], samples_from_rec(rec, amr.dxc)):
                L.append(v)
        print(f"  seed {seed}: {time.time() - t0:.1f}s")
    fits = {}
    for k, (e, a, b) in S.items():
        fits[k] = fit_km(np.concatenate(e), np.concatenate(a), np.concatenate(b))
        f = fits[k]
        print(f"  {k:15s} C_A={f['C_A']:.3f}  C_B={f['C_B']:.3f}  "
              f"floor={f['floor']:.2e}  kurt={f['kurt']:.2f}  n={f['n']}")
    return fits, dns_snaps


# ----------------------------------------------------------------------
def part3_germano_K(dns_snaps):
    print("\n=== Part 3: Semenov K + Germano bookkeeping ===")
    hist, fhist, C1 = {}, {}, {}
    for seed in SEEDS:
        for C0 in (0.3, 3.0):
            Uc0, Uf0 = make_ic(seed)
            amr = TwoLevelAMR(Uc=Uc0, Uf=Uf0, i0=0, i1=N_C, r=R, cfg=CFG,
                              model=TurbModel(enabled=True, C_eps=C0, f_K=1.0),
                              dynamic_C_eps=True, dynamic_f_K=True)
            while amr.t < T_END - 1e-12: amr.step(t_stop=T_END)
            hist[(seed, C0)] = np.array(amr.C_eps_history)
            fhist[(seed, C0)] = np.array(amr.f_K_history)
        # one-parameter fit (f_K fixed at 1, i.e. all numerical
        # dissipation feeds K as in Semenov 2025) for contrast
        Uc0, Uf0 = make_ic(seed)
        amr = TwoLevelAMR(Uc=Uc0, Uf=Uf0, i0=0, i1=N_C, r=R, cfg=CFG,
                          model=TurbModel(enabled=True, C_eps=3.0, f_K=1.0),
                          dynamic_C_eps=True)
        while amr.t < T_END - 1e-12: amr.step(t_stop=T_END)
        C1[seed] = np.array(amr.C_eps_history)
    late = lambda H: np.concatenate([h[h[:, 0] > 0.1, 1] for h in H.values()])
    C_eps, f_K = float(np.median(late(hist))), float(np.median(late(fhist)))
    C_eps_1 = float(np.median(late(C1)))
    spread = max(abs(hist[(s, 0.3)][-1, 1] - hist[(s, 3.0)][-1, 1]) for s in SEEDS)
    print(f"  2-parameter dynamic fit: C_eps = {C_eps:.2f}, f_K = {f_K:.3f}"
          f"  (median over t>0.1, seeds, initial guesses)")
    print(f"  max |C_eps(guess 0.3) - C_eps(guess 3.0)| at t_end = {spread:.2e}")
    print(f"  1-parameter fit with f_K = 1 (all dissipation -> K): C_eps = {C_eps_1:.2f}")

    times = np.array([t for t, _, _ in dns_snaps[SEEDS[0]]])
    def aniso(U):
        rho = U[0]; u = U[1] / rho
        return np.mean(U[2] - rho * u * u - U[3])
    runs = {"coarse": TurbModel(),
            "coarseK1": TurbModel(enabled=True, C_eps=C_eps_1, f_K=1.0),
            "coarseK": TurbModel(enabled=True, C_eps=C_eps, f_K=f_K)}
    curves = {k: np.zeros((len(SEEDS), len(times))) for k in
              ("dns", "Kres", "K", "K1", *runs)}
    err = {k: [] for k in runs}
    m = (times > 0.02) & (times < 0.5)
    for si, seed in enumerate(SEEDS):
        Uc0, _ = make_ic(seed)
        for ti, (t, Ud, Kr) in enumerate(dns_snaps[seed]):
            curves["dns"][si, ti] = aniso(Ud); curves["Kres"][si, ti] = Kr.mean()
        for k, mod in runs.items():
            sn = run_single(Uc0, T_END, times[1:], CFG, mod)
            for ti, (_, U, K) in enumerate(sn):
                curves[k][si, ti] = aniso(U)
                if k == "coarseK": curves["K"][si, ti] = K.mean()
                if k == "coarseK1": curves["K1"][si, ti] = K.mean()
            err[k].append(np.mean(np.abs(curves[k][si, m] - curves["dns"][si, m]))
                          / np.mean(np.abs(curves["dns"][si, m])))
    mK = times > 0
    for k in ("K", "K1"):
        e = np.mean(np.abs(np.log(curves[k][:, mK] / curves["Kres"][:, mK])))
        print(f"  mean |ln(K/K_res)| over the run, {k:3s}: {e:.2f}")
    for k in err:
        print(f"  <Pxx-Pperp> rel. error vs filtered DNS, t in (0.02,0.5): "
              f"{k:9s} {np.mean(err[k]):.2f}")
    return dict(C_eps=C_eps, f_K=f_K, C_eps_1=C_eps_1, hist=hist, fhist=fhist,
                times=times, curves=curves, err=err)


# ----------------------------------------------------------------------
def main():
    print("JIT warmup...", flush=True)
    Uw, _ = make_wave_pool_ic(32, u0=0.1)
    run_to(Uw, t_end=0.002, save_times=[0.002])
    p1 = part1_verification()
    fits, dns = part2_online_km()
    P3 = part3_germano_K(dns)
    times, curves, err = P3["times"], P3["curves"], P3["err"]

    np.savez(os.path.join(DATA_DIR, "amr_online_params.npz" if not QUICK
                          else "amr_online_params_quick.npz"),
             **{f"{k}_{q}": v[q] for k, v in fits.items() for q in ("C_A", "C_B", "kurt")},
             C_eps_dynamic=P3["C_eps"], f_K_dynamic=P3["f_K"],
             C_eps_one_param=P3["C_eps_1"])

    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    a = ax[0, 0]
    keys = list(fits); xs = np.arange(len(keys))
    a.bar(xs - 0.2, [fits[k]["C_A"] for k in keys], 0.4, label="$C_A$")
    a.bar(xs + 0.2, [fits[k]["C_B"] for k in keys], 0.4, label="$C_B$")
    a.set_xticks(xs); a.set_xticklabels([k.replace("_", "\n") for k in keys])
    a.set_title("Kramers-Moyal coefficients: online (AMR defect) vs offline")
    a.legend(); a.grid(alpha=0.3, axis="y")
    a = ax[0, 1]; a2 = a.twinx()
    for i, ((seed, C0), h) in enumerate(P3["hist"].items()):
        ls = "-" if C0 == 0.3 else "--"
        a.plot(h[:, 0], h[:, 1], lw=1, ls=ls, color=f"C{i // 2}",
               label=f"seed {seed}, guess {C0}")
        f = P3["fhist"][(seed, C0)]
        a2.plot(f[:, 0], f[:, 1], lw=0.8, ls=ls, color=f"C{i // 2}", alpha=0.5)
    a.set_ylim(0, 8); a2.set_ylim(0, 0.16)
    a.set_xlabel("t"); a.set_ylabel(r"$C_\epsilon$ (dark)"); a2.set_ylabel(r"$f_K$ (light)")
    a.set_title(rf"Dynamic Germano fit: $C_\epsilon$={P3['C_eps']:.2f}, $f_K$={P3['f_K']:.3f}")
    a.legend(fontsize=7, ncol=2, loc="upper right"); a.grid(alpha=0.3)
    a = ax[1, 0]
    for k, c, lab in (("dns", "k", "filtered fine DNS"),
                      ("coarse", "C0", "coarse"),
                      ("coarseK1", "C1", r"coarse + K, $f_K$=1"),
                      ("coarseK", "C3", r"coarse + K, 2-param fit")):
        e = f" (err {np.mean(err[k]):.2f})" if k in err else ""
        a.plot(times, curves[k].mean(0), color=c, label=lab + e)
    a.set_xlabel("t"); a.set_ylabel(r"$\langle P_{xx} - P_\perp \rangle$")
    a.set_xlim(0, 0.6); a.set_title("Pressure anisotropy (seed mean)")
    a.legend(); a.grid(alpha=0.3)
    a = ax[1, 1]
    a.semilogy(times[1:], curves["Kres"].mean(0)[1:], "k", label=r"$K_{\rm res}$ from DNS")
    a.semilogy(times[1:], curves["K1"].mean(0)[1:], "C1", label=r"coarse K, $f_K$=1")
    a.semilogy(times[1:], curves["K"].mean(0)[1:], "C3", label="coarse K, 2-param fit")
    a.set_xlabel("t"); a.set_ylabel("domain mean")
    a.set_title("Unresolved sub-cell kinetic energy")
    a.legend(); a.grid(alpha=0.3, which="both")
    out = os.path.join(FIG_DIR, "amr_germano_online.png")
    fig.savefig(out, dpi=130)
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
