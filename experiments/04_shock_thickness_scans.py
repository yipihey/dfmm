"""
Experiment 04: Shock thickness across (M, EOS) parameter space.

Measures shock thickness from tanh fits to the density profile and
compares adiabatic vs barotropic EOS. The adiabatic shocks show the
characteristic non-monotonic thickness-vs-M curve (with a minimum near
M~3) that is a signature of the closure transition from NS-like to
bimodal regimes; the barotropic reduction gives monotonically
decreasing thickness because first-moment closure is exact.

    shock_thickness_scans.png    Thickness vs M for adiabatic + three
                                 polytropic indices.
    eos_profiles.png             Shock profiles for adiabatic vs
                                 isothermal at the same M.

Runtime: ~5 minutes.
"""
import os, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from dfmm.setups.shock import run_steady_shock, rankine_hugoniot, GAMMA
from dfmm.schemes.barotropic import (run_baro_shock,
                                              compression_ratio_polytropic,
                                              baro_primitives)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def measure_thickness(x, rho, rho1, rho2):
    norm = (rho - rho1)/(rho2 - rho1)
    idx = np.argmin(np.abs(norm - 0.5))
    x0 = x[idx]
    def f(x, x0, delta):
        return 0.5*(1 + np.tanh((x - x0)/delta))
    try:
        popt, _ = curve_fit(f, x, norm, p0=[x0, 0.02],
                             bounds=([0, 1e-4], [1.0, 0.5]))
        return popt[1], popt[0]
    except (RuntimeError, ValueError):
        return np.nan, x0


def main():
    # JIT warmup
    _ = run_steady_shock(M1=2.0, tau=1e-3, N=64, t_end=0.02)
    _ = run_baro_shock(M1=2.0, Gamma=5.0/3.0, N=64, t_end=0.02)

    N = 400; t_end = 3.0; tau = 1e-3
    M_list = np.array([1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0])

    print("Adiabatic shock thickness vs M (full 8-field scheme)...")
    thick_adia = []
    for M1 in M_list:
        x, U, _ = run_steady_shock(M1=M1, tau=tau, N=N, t_end=t_end)
        rho2, _, _, _ = rankine_hugoniot(1.0, M1*np.sqrt(GAMMA), 1.0)
        th, _ = measure_thickness(x, U[0], 1.0, rho2)
        thick_adia.append(th)
        print(f"  M={M1}: delta={th:.4f}")

    baro_cases = [(5.0/3.0, r'$\Gamma = 5/3$ (adiabatic baro)'),
                  (1.0,     r'$\Gamma = 1$ (isothermal)'),
                  (2.0,     r'$\Gamma = 2$ (stiff)')]
    thick_baro = {}
    for G, label in baro_cases:
        print(f"Barotropic {label} thickness vs M...")
        vals = []
        for M1 in M_list:
            x, U, _ = run_baro_shock(M1=M1, Gamma=G, K=1.0, N=N, t_end=t_end)
            rho2 = compression_ratio_polytropic(M1, G)
            th, _ = measure_thickness(x, U[0], 1.0, rho2)
            vals.append(th)
        thick_baro[G] = vals
        print(f"  thick list: {[f'{t:.3f}' for t in vals]}")

    # Figure: thickness vs M
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.semilogx(M_list, thick_adia, 'o-', lw=1.4, color='C3',
                 label='adiabatic (full 8-field)')
    styles = ['--', '-.', ':']
    for (G, label), ls in zip(baro_cases, styles):
        ax.semilogx(M_list, thick_baro[G], 'o'+ls, lw=1.2, label=label)
    ax.set_xlabel(r'Mach number $M_1$')
    ax.set_ylabel(r'shock thickness $\delta$')
    ax.set_title(rf"Shock thickness vs $M_1$ ($\tau = 10^{{-3}}$)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')
    fig.savefig(os.path.join(FIG_DIR, "shock_thickness_scans.png"), dpi=130)
    print("Saved shock_thickness_scans.png")

    # Figure: profile comparison at M=3
    print("Fig: adiabatic vs isothermal profiles at M=3...")
    x_a, U_a, _ = run_steady_shock(M1=3.0, tau=tau, N=N, t_end=t_end)
    x_i, U_i, _ = run_baro_shock(M1=3.0, Gamma=1.0, K=1.0, N=N, t_end=t_end)
    x_p, U_p, _ = run_baro_shock(M1=3.0, Gamma=2.0, K=1.0, N=N, t_end=t_end)

    rho2_a, _, _, _ = rankine_hugoniot(1.0, 3.0*np.sqrt(GAMMA), 1.0)
    rho2_i = compression_ratio_polytropic(3.0, 1.0)
    rho2_p = compression_ratio_polytropic(3.0, 2.0)

    _, x0_a = measure_thickness(x_a, U_a[0], 1.0, rho2_a)
    _, x0_i = measure_thickness(x_i, U_i[0], 1.0, rho2_i)
    _, x0_p = measure_thickness(x_p, U_p[0], 1.0, rho2_p)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(x_a - x0_a, U_a[0], 'C3-', lw=1.3, label='adiabatic')
    axes[0].plot(x_i - x0_i, U_i[0], 'C0--', lw=1.3, label=r'isothermal $\Gamma=1$')
    axes[0].plot(x_p - x0_p, U_p[0], 'C1-.', lw=1.3, label=r'stiff $\Gamma=2$')
    axes[0].set_xlabel('x - x_shock'); axes[0].set_ylabel(r'$\rho$')
    axes[0].set_xlim(-0.2, 0.2); axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    u_a = U_a[1]/U_a[0]
    u_i = U_i[1]/U_i[0]
    u_p = U_p[1]/U_p[0]
    axes[1].plot(x_a - x0_a, u_a, 'C3-', lw=1.3, label='adiabatic')
    axes[1].plot(x_i - x0_i, u_i, 'C0--', lw=1.3, label=r'isothermal')
    axes[1].plot(x_p - x0_p, u_p, 'C1-.', lw=1.3, label=r'$\Gamma=2$')
    axes[1].set_xlabel('x - x_shock'); axes[1].set_ylabel('u')
    axes[1].set_xlim(-0.2, 0.2); axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.suptitle(r"Profile comparison across EOS at $M_1 = 3$")
    fig.savefig(os.path.join(FIG_DIR, "eos_profiles.png"), dpi=130)
    print("Saved eos_profiles.png")


if __name__ == "__main__":
    main()
