"""
Experiment 10: Wave-pool flow evolution at a single realization.

Sets up the broadband decaying-compressible-turbulence test and plots
key integrated diagnostics over time: kinetic/internal energy, RMS Mach,
peak |s|, minimum gamma. This is the reference flow used throughout the
Kramers-Moyal LES closure pipeline (experiments 11 onward).

Produces: paper/figs/kmles_stageA_evolution.png
Runtime:  ~10 seconds
"""
import os, time
import numpy as np
import matplotlib.pyplot as plt

from momentlag.setups.wavepool import make_wave_pool_ic
from momentlag.integrate import run_to
from momentlag.diagnostics import extract_diagnostics

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    N = 256
    u0 = 1.0; P0 = 0.1; K_max = 16
    t_end = 5.0
    save_dt = 0.02
    save_times = np.arange(save_dt, t_end + 1e-9, save_dt)

    print("JIT warmup...", end=' ', flush=True)
    U_warm, _ = make_wave_pool_ic(32, u0=0.1)
    _ = run_to(U_warm, t_end=0.002, save_times=[0.002])
    print("done.")

    print(f"Running wave-pool IC N={N}, t_end={t_end}...")
    t0 = time.time()
    U0, x = make_wave_pool_ic(N, u0=u0, P0=P0, K_max=K_max, seed=42)
    snaps, nsteps = run_to(U0, t_end=t_end, tau=1e-3, save_times=save_times)
    print(f"  {nsteps} steps in {time.time()-t0:.1f}s")

    # Extract time series
    times = np.array([s[0] for s in snaps])
    KE = np.empty(len(times)); IE = np.empty(len(times))
    Mrms = np.empty(len(times)); smax = np.empty(len(times))
    gmin = np.empty(len(times))
    for i, (_, U) in enumerate(snaps):
        d = extract_diagnostics(U)
        KE[i] = 0.5*np.mean(d['rho']*d['u']**2)
        IE[i] = 0.5*np.mean(d['Pxx'] + 2*d['Pp'])
        Mrms[i] = np.sqrt(np.mean(d['Mach']**2))
        smax[i] = np.max(np.abs(d['s']))
        gmin[i] = np.min(d['gamma'])

    IE0 = IE[0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(times, KE, label='KE', lw=1.3)
    axes[0, 0].plot(times, IE - IE0, '--', label='IE - IE(0)', lw=1.3)
    axes[0, 0].plot(times, (KE + IE) - (KE[0] + IE0), ':k', lw=1.0,
                     label='total - init')
    axes[0, 0].set_ylabel('energy'); axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_title('Kinetic energy decay, thermal energy rise')
    axes[0, 1].plot(times, Mrms, color='C2', lw=1.3)
    axes[0, 1].set_ylabel(r'$M_{\rm rms}$')
    axes[0, 1].set_title('RMS Mach number evolution')
    axes[1, 0].plot(times, smax, color='C3', lw=1.3)
    axes[1, 0].axhline(np.sqrt(2), color='k', ls=':', lw=0.8,
                        label='realizability')
    axes[1, 0].set_ylabel(r'$\max |s|$'); axes[1, 0].legend(fontsize=9)
    axes[1, 0].set_title('Peak skewness diagnostic')
    axes[1, 1].semilogy(times, gmin, color='C4', lw=1.3)
    axes[1, 1].axhline(1e-1, color='k', ls=':', lw=0.8)
    axes[1, 1].set_ylabel(r'$\min \gamma / \sqrt{\Sigma_{vv}}$')
    axes[1, 1].set_title('Minimum Cholesky diagnostic')
    axes[1, 1].set_ylim(1e-2, 2)
    for ax in axes.flat:
        ax.set_xlabel('t'); ax.grid(alpha=0.3)
    out = os.path.join(FIG_DIR, "kmles_stageA_evolution.png")
    fig.savefig(out, dpi=130)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
