"""
Experiment 12: Scan of noise spatial correlation length ell_corr.

Runs the noise-augmented coarse scheme with a range of spatial
correlation lengths ell_corr (in cells) for the injected noise, and
compares the resulting spectrum to a fine-DNS reference. ell_corr = 0
(per-cell uncorrelated) puts too much power at the grid scale;
over-smoothing (ell_corr >> 1) suppresses mid-k. The optimal is ell_corr
~ 2 cells, matching roughly the coarse-graining scale in fine cells.

Produces: paper/figs/kmles_stageF_corr.png
Runtime:  ~20 seconds
"""
import os, time
import numpy as np
import matplotlib.pyplot as plt

from dfmm.setups.wavepool import make_wave_pool_ic
from dfmm.integrate import run_to, coarse_grain
from dfmm.analysis import compute_spectrum, log_bin_spectrum
from dfmm.closure.noise_model import run_noise

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    print("JIT warmup...", end=' ', flush=True)
    U_warm, _ = make_wave_pool_ic(32, u0=0.1)
    _ = run_to(U_warm, t_end=0.002, save_times=[0.002])
    _ = run_noise(U_warm, t_end=0.002, save_times=[0.002],
                  C_A=0.34, C_B=0.55, ell_corr=2.0, seed=0)
    print("done.")

    calib = np.load(os.path.join(DATA_DIR, "noise_model_params.npz"))
    C_A = float(calib['C_A']); C_B = float(calib['C_B'])

    N_coarse = 256; refinement = 8; N_fine = N_coarse*refinement
    u0 = 1.0; P0 = 0.1; K_max = 16; seed_ic = 42; tau = 1e-3
    t_end = 1.0
    save_dt = 0.05
    save_times = np.arange(save_dt, t_end + 1e-9, save_dt)

    def get_u(snaps, t_target):
        i = np.argmin(np.abs(np.array([s[0] for s in snaps]) - t_target))
        U = snaps[i][1]
        return U[1]/U[0]

    def lbin(u):
        k, P = compute_spectrum(u)
        return log_bin_spectrum(k, P, n_bins=25)

    # Baseline: det coarse + fine DNS
    print("  running det coarse...")
    U_c0, _ = make_wave_pool_ic(N_coarse, u0=u0, P0=P0, K_max=K_max, seed=seed_ic)
    t0 = time.time()
    snaps_c, _ = run_to(U_c0.copy(), t_end=t_end, tau=tau, save_times=save_times)
    print(f"    {time.time()-t0:.1f}s")

    print("  running fine DNS...")
    U_f0, _ = make_wave_pool_ic(N_fine, u0=u0, P0=P0, K_max=K_max, seed=seed_ic)
    t0 = time.time()
    snaps_f, _ = run_to(U_f0, t_end=t_end, tau=tau, save_times=save_times)
    snaps_f_cg = [(t, coarse_grain(U, refinement)) for t, U in snaps_f]
    print(f"    {time.time()-t0:.1f}s")

    # Scan
    ell_corrs = [0, 1, 2, 4, 8]
    noise_runs = {}
    for ell in ell_corrs:
        print(f"  noise ell={ell}...")
        t0 = time.time()
        noise_runs[ell] = run_noise(U_c0.copy(), t_end=t_end, tau=tau,
                                      save_times=save_times,
                                      C_A=C_A, C_B=C_B, ell_corr=ell, seed=1)
        print(f"    {time.time()-t0:.1f}s")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, t_target in zip(axes, [0.10, 0.30, 0.70]):
        u_c = get_u(snaps_c, t_target); u_f = get_u(snaps_f_cg, t_target)
        k_b, Pc = lbin(u_c); _, Pf = lbin(u_f)
        ax.loglog(k_b, Pc, 'C0-', lw=1.0, alpha=0.8, label='det. coarse')
        ax.loglog(k_b, Pf, 'k-', lw=1.4, label='fine DNS (cg)')
        colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(ell_corrs)))
        for ell, c in zip(ell_corrs, colors):
            u_n = get_u(noise_runs[ell], t_target)
            _, Pn = lbin(u_n)
            ax.loglog(k_b, Pn, '--', color=c, lw=1.0,
                       label=rf'$\ell_{{\rm corr}}={ell}$', alpha=0.9)
        if t_target == 0.10:
            kref = np.array([3.0, 60.0])
            ax.loglog(kref, 0.1*kref**(-2), 'k:', lw=0.7, label=r'$k^{-2}$')
        ax.set_title(f"t = {t_target:.2f}")
        ax.set_xlabel('k'); ax.set_ylabel('E(k)')
        ax.set_ylim(1e-10, 1); ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=7, loc='lower left')
    fig.suptitle(r'Energy spectra, log-$k$ binned: scan of noise correlation length $\ell_{\rm corr}$',
                 fontsize=11)
    out = os.path.join(FIG_DIR, "kmles_stageF_corr.png")
    fig.savefig(out, dpi=130)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
