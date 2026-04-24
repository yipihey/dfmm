"""
Experiment 02: Cold sinusoid across three Knudsen regimes.

Initial condition is a cold sinusoidal velocity perturbation,
u(x, 0) = A * sin(2 pi x), rho = 1, P = P0 small, on a periodic
[0, 1] domain.

The initial velocity maximum is at x = 0.25 with value A. In the
collisionless (Knudsen -> infinity) limit, ballistic parcels move at
constant velocity; the parcel initially at x = 0.25 arrives at x = 0.85
after time t = 0.6/A. We choose t_end = 0.6/A so the caustic appears
cleanly away from the periodic boundary. At the same time the Euler
case has formed a strong shock and the intermediate case is partially
thermalized.

This script produces two figures:

    sine_three_tau.png        Hydrodynamic profiles + closure diagnostics
                              at t_end for all three tau values.
    sine_phase_space.png      Phase-space distribution f(x, v) rendered
                              from the 8-field moment state for each tau.

Runtime: ~15 seconds.
"""
import os, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from momentlag.setups.sine import make_sine_ic
from momentlag.integrate import run_to
from momentlag.diagnostics import extract_diagnostics
from momentlag.schemes.cholesky import primitives, build_f

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def _render_phase_space(U, x, Nv=256, v_margin=1.6):
    """Render f(x, v) from the moment state, using the Cholesky Gaussian."""
    rho, u, Pxx, Pp, L1, alpha, beta, gamma, Sxx, Sxv, Svv, Q = primitives(U)
    v_max = v_margin * max(np.max(np.abs(u)), np.sqrt(np.max(Svv)))
    x_grid = np.linspace(0, 1, len(x))
    v_grid = np.linspace(-v_max, v_max, Nv)
    f, *_ = build_f(x_grid, U, x_grid, v_grid)
    return x_grid, v_grid, f


def main():
    A = 1.0
    t_end = 0.6 / A     # caustic at x = 0.85 in collisionless limit
    N = 512
    cases = [(1e-5, 'Euler',          'C0'),
             (1e-2, 'intermediate',   'C1'),
             (1e3,  'collisionless',  'C2')]

    print(f"Running cold-sinusoid tests at A={A}, t_end={t_end:.3f}...")

    # JIT warmup
    U0, _ = make_sine_ic(32, A=0.1)
    _ = run_to(U0, t_end=0.002, save_times=[0.002], tau=1e-3)

    final = {}
    for tau, label, color in cases:
        t0 = time.time()
        U0, x = make_sine_ic(N, A=A)
        snaps, ns = run_to(U0, t_end=t_end, save_times=[t_end], tau=tau)
        print(f"  tau={tau:g} ({label}): {ns} steps in {time.time()-t0:.2f}s")
        final[label] = (x, snaps[-1][1])

    # -------- Fig 1: hydro + closure diagnostics --------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for (tau, label, color) in cases:
        x, U = final[label]
        d = extract_diagnostics(U)
        axes[0, 0].plot(x, d['rho'], color=color,
                         label=rf"{label} $\tau={tau:g}$", lw=1.1)
        axes[0, 1].plot(x, d['u'],   color=color, lw=1.1)
        axes[1, 0].plot(x, np.abs(d['s']), color=color, lw=1.1)
        axes[1, 1].semilogy(x, np.maximum(d['gamma'], 1e-8), color=color, lw=1.1)

    axes[0, 0].set_ylabel(r'$\rho$')
    axes[0, 0].legend(fontsize=9)
    axes[0, 1].set_ylabel('u')
    axes[0, 1].axvline(0.25, color='k', ls=':', lw=0.6, alpha=0.5)
    axes[0, 1].axvline(0.85, color='k', ls=':', lw=0.6, alpha=0.5)
    axes[1, 0].set_ylabel(r'$|s|$ (standardized skewness)')
    axes[1, 0].axhline(np.sqrt(2), color='k', ls=':', lw=0.8,
                        label='realizability')
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set_ylabel(r'$\gamma / \sqrt{\Sigma_{vv}}$ (Cholesky diagnostic)')
    axes[1, 1].set_ylim(1e-6, 2)
    for ax in axes[1]:
        ax.set_xlabel('x')
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle(rf"Cold sinusoid at three $\tau$ values, "
                 rf"$A = {A}$, $t = {t_end:.2f}$")
    out1 = os.path.join(FIG_DIR, "sine_three_tau.png")
    fig.savefig(out1, dpi=130)
    print(f"  saved {out1}")

    # -------- Fig 2: phase-space portrait --------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, (tau, label, color) in zip(axes, cases):
        x, U = final[label]
        x_grid, v_grid, f = _render_phase_space(U, x, Nv=256, v_margin=1.6)
        im = ax.pcolormesh(x_grid, v_grid, f + 1e-12,
                            norm=LogNorm(vmin=1e-3, vmax=f.max()),
                            shading='auto', cmap='viridis')
        ax.set_xlabel('x'); ax.set_ylabel('v')
        ax.set_title(rf"{label}, $\tau = {tau:g}$")
        ax.axvline(0.25, color='w', ls=':', lw=0.6, alpha=0.5)
        ax.axvline(0.85, color='w', ls=':', lw=0.6, alpha=0.5)
        plt.colorbar(im, ax=ax, label='f(x, v)')
    fig.suptitle(rf"Phase-space portrait at $t = {t_end:.2f}$ "
                 rf"(caustic expected near $x = 0.85$)")
    out2 = os.path.join(FIG_DIR, "sine_phase_space.png")
    fig.savefig(out2, dpi=130)
    print(f"  saved {out2}")


if __name__ == "__main__":
    main()
