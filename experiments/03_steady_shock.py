"""
Experiment 03: Steady shocks — profiles, diagnostics, tau scan, convergence.

Runs the inflow-outflow steady-shock setup across several representative
parameter combinations, producing four figures in one pass:

    steady_shock_profiles.png    Hydro profiles at several Mach numbers.
    shock_diagnostics.png        M=3 profile with full closure diagnostics.
    shock_tau_scan.png           Shock thickness vs tau at fixed M.
    convergence.png              Grid-convergence test at fixed (M, tau).

Runtime: ~2 minutes total.
"""
import os, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from momentlag.setups.shock import run_steady_shock, rankine_hugoniot, GAMMA
from momentlag.diagnostics import extract_diagnostics

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def measure_thickness(x, rho, rho1, rho2):
    norm = (rho - rho1)/(rho2 - rho1)
    idx = np.argmin(np.abs(norm - 0.5))
    x0_guess = x[idx]
    def f(x, x0, delta):
        return 0.5*(1 + np.tanh((x - x0)/delta))
    try:
        popt, _ = curve_fit(f, x, norm, p0=[x0_guess, 0.02],
                             bounds=([0, 1e-4], [1.0, 0.5]))
        return popt[1], popt[0]
    except (RuntimeError, ValueError):
        return np.nan, x0_guess


def main():
    _ = run_steady_shock(M1=2.0, tau=1e-3, N=64, t_end=0.02)     # JIT warmup
    N_default = 400
    t_end = 3.0

    # === Fig 1: profiles at several M ===
    print("Fig 1: profiles at several Mach numbers...")
    M_list = [1.5, 2.0, 3.0, 5.0, 10.0]
    colors_M = plt.cm.viridis(np.linspace(0.15, 0.85, len(M_list)))
    results_M = {}
    for M1 in M_list:
        t0 = time.time()
        x, U, ns = run_steady_shock(M1=M1, tau=1e-3, N=N_default, t_end=t_end)
        rho2, u2, P2, _ = rankine_hugoniot(1.0, M1*np.sqrt(GAMMA), 1.0)
        d = extract_diagnostics(U)
        _, x0 = measure_thickness(x, d['rho'], 1.0, rho2)
        results_M[M1] = (x - x0, d, rho2, u2, P2)
        print(f"  M={M1}: {ns} steps in {time.time()-t0:.2f}s")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for M1, color in zip(M_list, colors_M):
        x_s, d, rho2, u2, P2 = results_M[M1]
        P = (d['Pxx'] + 2*d['Pp'])/3
        axes[0, 0].plot(x_s, (d['rho']-1)/(rho2-1), color=color, lw=1.1,
                         label=f'M={M1}')
        axes[0, 1].plot(x_s, d['u'], color=color, lw=1.1)
        axes[1, 0].plot(x_s, P, color=color, lw=1.1)
        axes[1, 1].plot(x_s, (d['Pxx']-d['Pp'])/np.maximum(P, 1e-6),
                         color=color, lw=1.1)
    axes[0, 0].set_ylabel(r'$(\rho - \rho_1)/(\rho_2 - \rho_1)$')
    axes[0, 0].legend(fontsize=9, loc='best')
    axes[0, 1].set_ylabel('u')
    axes[1, 0].set_ylabel('P')
    axes[1, 1].set_ylabel(r'$(P_{xx} - P_\perp)/P$')
    for ax in axes.flat:
        ax.set_xlabel('x - x_shock')
        ax.set_xlim(-0.2, 0.2); ax.grid(alpha=0.3)
    fig.suptitle(rf"Steady shocks at $\tau = 10^{{-3}}$, $N = {N_default}$")
    fig.savefig(os.path.join(FIG_DIR, "steady_shock_profiles.png"), dpi=130)

    # === Fig 2: diagnostics through M=3 ===
    print("Fig 2: closure diagnostics at M=3...")
    x_s, d, rho2, u2, P2 = results_M[3.0]
    P = (d['Pxx'] + 2*d['Pp'])/3
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(x_s, d['rho'], color='C0', lw=1.2)
    axes[0, 0].axhline(1.0, color='k', ls=':', lw=0.6)
    axes[0, 0].axhline(rho2, color='k', ls=':', lw=0.6)
    axes[0, 0].set_ylabel(r'$\rho$')
    axes[0, 1].plot(x_s, (d['Pxx']-d['Pp'])/np.maximum(P, 1e-6),
                     color='C1', lw=1.2)
    axes[0, 1].axhline(0, color='k', lw=0.3)
    axes[0, 1].set_ylabel(r'pressure anisotropy $(P_{xx}-P_\perp)/P$')
    axes[1, 0].plot(x_s, d['Q'], color='C2', lw=1.2)
    axes[1, 0].axhline(0, color='k', lw=0.3)
    axes[1, 0].set_ylabel('heat flux Q')
    axes[1, 1].plot(x_s, np.abs(d['s']), color='C3', lw=1.2)
    axes[1, 1].axhline(np.sqrt(2), color='k', ls=':', lw=0.8,
                        label='realizability')
    axes[1, 1].set_ylabel(r'$|s|$ (standardized skewness)')
    axes[1, 1].legend(fontsize=9)
    for ax in axes.flat:
        ax.set_xlabel('x - x_shock')
        ax.set_xlim(-0.1, 0.15); ax.grid(alpha=0.3)
    fig.suptitle(r'Closure diagnostics through $M=3$ shock, $\tau = 10^{-3}$')
    fig.savefig(os.path.join(FIG_DIR, "shock_diagnostics.png"), dpi=130)

    # === Fig 3: tau scan ===
    print("Fig 3: tau scan at M=3...")
    tau_list = np.logspace(-4, -1, 7)
    colors_t = plt.cm.plasma(np.linspace(0.15, 0.85, len(tau_list)))
    thick_tau = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    rho2_3, _, _, _ = rankine_hugoniot(1.0, 3.0*np.sqrt(GAMMA), 1.0)
    for tau, color in zip(tau_list, colors_t):
        x, U, _ = run_steady_shock(M1=3.0, tau=tau, N=N_default, t_end=t_end)
        d = extract_diagnostics(U)
        th, x0 = measure_thickness(x, d['rho'], 1.0, rho2_3)
        thick_tau.append(th)
        axes[0].plot(x - x0, (d['rho']-1)/(rho2_3-1), color=color, lw=1.1,
                      label=rf'$\tau = {tau:.1e}$')
    axes[0].set_xlabel('x - x_shock')
    axes[0].set_ylabel(r'$(\rho - \rho_1)/(\rho_2 - \rho_1)$')
    axes[0].set_xlim(-0.2, 0.2)
    axes[0].legend(fontsize=7, ncol=2, loc='lower right')
    axes[0].grid(alpha=0.3)
    axes[1].loglog(tau_list, thick_tau, 'o-', lw=1.2)
    t_ref = np.array([1e-4, 1e-1])
    axes[1].loglog(t_ref, 30*t_ref, 'k:', lw=0.7, label=r'$\delta \propto \tau$')
    axes[1].set_xlabel(r'$\tau$')
    axes[1].set_ylabel(r'shock thickness $\delta$')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, which='both')
    fig.suptitle(r'$\tau$ scan at $M_1 = 3$')
    fig.savefig(os.path.join(FIG_DIR, "shock_tau_scan.png"), dpi=130)

    # === Fig 4: grid convergence ===
    print("Fig 4: grid convergence at M=3, tau=1e-3...")
    N_list = [100, 200, 400, 800]
    thick_N = []
    for N in N_list:
        x, U, _ = run_steady_shock(M1=3.0, tau=1e-3, N=N, t_end=t_end)
        rho = U[0]
        th, _ = measure_thickness(x, rho, 1.0, rho2_3)
        thick_N.append(th)
        print(f"    N={N}: thickness={th:.4f}")
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.semilogx(N_list, thick_N, 'o-', lw=1.2, color='C0')
    ax.set_xlabel('N (grid cells)')
    ax.set_ylabel(r'shock thickness $\delta$')
    ax.set_title(r'Convergence: $M_1 = 3$, $\tau = 10^{-3}$')
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIG_DIR, "convergence.png"), dpi=130)
    print("All shock figures saved.")


if __name__ == "__main__":
    main()
