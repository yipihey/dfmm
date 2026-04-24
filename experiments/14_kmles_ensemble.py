"""
Experiment 14: Paired-fixed ensemble validation of the noise-augmented LES scheme.

Runs N_pairs paired-phase realizations at coarse resolution (both
deterministic and noise-augmented) and at high-resolution DNS, measures
velocity spectra with time-window averaging and logarithmic k-binning,
and plots ensemble-mean spectra at four focus times with confidence
bands.

This is the main validation figure of the paper's Section 9.

Produces: paper/figs/kmles_stageH_ensemble.png
Runtime:  ~7-15 minutes, depending on N_coarse/N_fine.

To use a smaller config for quick testing, set SMALL=1 in the environment.
"""
import os, time
import numpy as np
import matplotlib.pyplot as plt

from momentlag.setups.wavepool import make_wave_pool_ic_paired
from momentlag.integrate import run_to, coarse_grain
from momentlag.analysis import compute_spectrum, log_bin_spectrum
from momentlag.closure.noise_model import run_noise
from momentlag.closure.ensemble import build_focus_times, draw_paired_phases

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    small = bool(int(os.environ.get("SMALL", "0")))
    if small:
        N_coarse, refinement, N_pairs = 128, 8, 5
        print("Running in SMALL mode (reduced N_coarse and N_pairs)")
    else:
        N_coarse, refinement, N_pairs = 512, 8, 20
    N_fine = N_coarse*refinement

    calib = np.load(os.path.join(DATA_DIR, "noise_model_params.npz"))
    C_A = float(calib['C_A']); C_B = float(calib['C_B'])
    print(f"C_A = {C_A:.4f}, C_B = {C_B:.4f}")
    print(f"N_coarse = {N_coarse}, N_fine = {N_fine}, N_pairs = {N_pairs}")

    u0 = 1.0; P0 = 0.1; K_max = 16; tau = 1e-3; ell_corr = 2.0
    focus_times = [0.1, 0.3, 0.5, 0.8]
    save_times, focus_slices = build_focus_times(focus_times,
                                                   n_per_window=5,
                                                   half_width=0.025)
    t_end = float(save_times[-1]) + 0.01
    n_times = len(save_times)

    print("JIT warmup...", end=' ', flush=True)
    phi_w = draw_paired_phases(0, 8)
    U_w, _ = make_wave_pool_ic_paired(32, u0=0.1, phases=phi_w,
                                        P0=0.1, K_max=8)
    _ = run_to(U_w, t_end=0.002, save_times=[0.002])
    _ = run_noise(U_w, t_end=0.002, save_times=[0.002],
                    C_A=C_A, C_B=C_B, ell_corr=2, seed=0)
    print("done.")

    u_det   = np.zeros((N_pairs, 2, n_times, N_coarse))
    u_noise = np.zeros((N_pairs, 2, n_times, N_coarse))
    u_fine  = np.zeros((N_pairs, 2, n_times, N_coarse))

    t_start = time.time()
    for pair_idx in range(N_pairs):
        t_p = time.time()
        phases = draw_paired_phases(10000 + pair_idx, K_max)
        for m, flip in enumerate([False, True]):
            U_c, _ = make_wave_pool_ic_paired(N_coarse, u0=u0, phases=phases,
                                                P0=P0, K_max=K_max, flip=flip)
            U_f, _ = make_wave_pool_ic_paired(N_fine, u0=u0, phases=phases,
                                                P0=P0, K_max=K_max, flip=flip)
            # det coarse
            snaps_c, _ = run_to(U_c.copy(), t_end=t_end, tau=tau,
                                 save_times=save_times)
            for ti, (t, U) in enumerate(snaps_c[1:]):
                u_det[pair_idx, m, ti] = U[1]/U[0]
            # noise coarse
            nseed = 30000 + 2*pair_idx + m
            snaps_n = run_noise(U_c.copy(), t_end=t_end, tau=tau,
                                  save_times=save_times,
                                  C_A=C_A, C_B=C_B, ell_corr=ell_corr,
                                  seed=nseed)
            for ti, (t, U) in enumerate(snaps_n[1:]):
                u_noise[pair_idx, m, ti] = U[1]/U[0]
            # fine DNS (coarse-grained on save)
            snaps_f, _ = run_to(U_f.copy(), t_end=t_end, tau=tau,
                                 save_times=save_times)
            for ti, (t, U) in enumerate(snaps_f[1:]):
                u_fine[pair_idx, m, ti] = coarse_grain(U, refinement)[1]/ \
                                            coarse_grain(U, refinement)[0]
        eta_p = time.time() - t_p
        eta_rem = (N_pairs - pair_idx - 1)*eta_p
        print(f"  pair {pair_idx+1:2d}/{N_pairs}: {eta_p:.1f}s "
              f"(total {time.time()-t_start:.0f}s, ETA {eta_rem:.0f}s)",
              flush=True)

    print(f"\nTotal ensemble time: {time.time()-t_start:.1f}s")

    # Compute spectra
    def all_spec(u):
        k, _ = compute_spectrum(u[0, 0, 0])
        spec = np.empty(u.shape[:-1] + (len(k),))
        for p in range(u.shape[0]):
            for m in range(u.shape[1]):
                for ti in range(u.shape[2]):
                    _, P = compute_spectrum(u[p, m, ti])
                    spec[p, m, ti] = P
        return k, spec

    k, S_det = all_spec(u_det)
    _, S_noise = all_spec(u_noise)
    _, S_fine = all_spec(u_fine)

    # Window-average and log-bin
    def bin_window(S, sl):
        win = S[:, :, sl, :]   # (Npairs, 2, nwin, nk)
        _, out = log_bin_spectrum(k, win, n_bins=30)
        return out

    # Main figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for ax, ti_slice, tfocus in zip(axes.flat, focus_slices, focus_times):
        k_b, Pd = log_bin_spectrum(k, S_det[:, :, ti_slice, :], n_bins=30)
        _, Pn = log_bin_spectrum(k, S_noise[:, :, ti_slice, :], n_bins=30)
        _, Pf = log_bin_spectrum(k, S_fine[:, :, ti_slice, :], n_bins=30)
        mask = np.isfinite(Pf)
        ax.loglog(k_b[mask], Pf[mask], 'k-', lw=1.8, label='fine DNS (cg)')
        ax.loglog(k_b[mask], Pd[mask], 'C0-', lw=1.3, label='det coarse')
        ax.loglog(k_b[mask], Pn[mask], 'C3--', lw=1.3, label='noise coarse')
        if len(k_b[mask]) > 4:
            j = len(k_b[mask])//3
            amp = Pf[mask][j]*k_b[mask][j]**2
            kref = np.logspace(0.3, 2.2, 10)
            ax.loglog(kref, amp*kref**(-2.0)*0.5, 'k:', alpha=0.4, lw=0.8,
                        label=r'$k^{-2}$')
        ax.set_xlabel('k'); ax.set_ylabel('E(k)')
        ax.set_title(f't = {tfocus}')
        ax.legend(fontsize=8, loc='lower left')
        ax.grid(alpha=0.3, which='both')
        ax.set_ylim(1e-10, 2)
    fig.suptitle(f'Window + ensemble + log-k binned spectra, '
                 f'$N_{{\\rm coarse}} = {N_coarse}$ ({N_pairs} pairs = '
                 f'{2*N_pairs} realizations)')
    out = os.path.join(FIG_DIR, "kmles_stageH_ensemble.png")
    fig.savefig(out, dpi=130)
    print(f"  saved {out}")

    # Quantitative summary
    print("\n=== Spectral match (RMS log-distance to fine DNS, k in [3, 100]) ===")
    for ti_slice, tfocus in zip(focus_slices, focus_times):
        k_b, Pd = log_bin_spectrum(k, S_det[:, :, ti_slice, :], n_bins=30)
        _, Pn = log_bin_spectrum(k, S_noise[:, :, ti_slice, :], n_bins=30)
        _, Pf = log_bin_spectrum(k, S_fine[:, :, ti_slice, :], n_bins=30)
        m = np.isfinite(Pf) & (k_b >= 3) & (k_b <= 100)
        d_det = np.sqrt(np.mean((np.log10(Pd[m]) - np.log10(Pf[m]))**2))
        d_noi = np.sqrt(np.mean((np.log10(Pn[m]) - np.log10(Pf[m]))**2))
        print(f"  t={tfocus}: det={d_det:.3f}, noise={d_noi:.3f}, "
              f"improvement {d_det/d_noi:.2f}x")


if __name__ == "__main__":
    main()
