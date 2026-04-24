"""
Experiment 01: Sod shock tube at three Knudsen numbers.

Runs the Sod initial condition at three relaxation times tau spanning
the Euler (tau -> 0), intermediate, and collisionless (tau -> infinity)
regimes. In the Euler limit the moment scheme reproduces the classical
exact Riemann solution; in the collisionless limit it reproduces free
streaming. The intermediate case shows regularized shock structure.

Produces: paper/figs/sod_three_tau.png
Runtime:  ~5 seconds (after JIT warmup)
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from momentlag.setups.sod import make_sod_ic
from momentlag.integrate import run_to
from momentlag.diagnostics import extract_diagnostics

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def exact_riemann_sod(x, t, gamma=5/3, x0=0.5):
    """Self-similar Sod exact solution, adapted for gamma = 5/3."""
    # Based on Toro (1999). Left state rho1=1, u1=0, P1=1; right rho4=0.125, u4=0, P4=0.1.
    rho1, u1, P1 = 1.0, 0.0, 1.0
    rho4, u4, P4 = 0.125, 0.0, 0.1
    c1 = np.sqrt(gamma*P1/rho1)
    c4 = np.sqrt(gamma*P4/rho4)

    # Solve P3 iteratively (Newton-Raphson in Toro); here use bisection
    def f_shock(P, Pk, rho_k):
        A = 2.0/((gamma+1.0)*rho_k)
        B = (gamma-1.0)/(gamma+1.0)*Pk
        return (P - Pk)*np.sqrt(A/(P + B))

    def f_rare(P, Pk, rho_k):
        ck = np.sqrt(gamma*Pk/rho_k)
        return (2*ck/(gamma-1.0))*((P/Pk)**((gamma-1.0)/(2*gamma)) - 1)

    def f(P):
        f_L = f_rare(P, P1, rho1) if P <= P1 else f_shock(P, P1, rho1)
        f_R = f_rare(P, P4, rho4) if P <= P4 else f_shock(P, P4, rho4)
        return f_L + f_R + (u4 - u1)

    P_lo, P_hi = 1e-6, 10.0
    for _ in range(200):
        P_mid = 0.5*(P_lo + P_hi)
        if f(P_mid) > 0: P_hi = P_mid
        else: P_lo = P_mid
    P3 = 0.5*(P_lo + P_hi)

    u3 = 0.5*(u1 + u4) + 0.5*((f_shock(P3, P4, rho4) if P3 > P4 else f_rare(P3, P4, rho4))
                               - (f_shock(P3, P1, rho1) if P3 > P1 else f_rare(P3, P1, rho1)))
    # Left wave: rarefaction for P3 < P1
    c3L = c1*(P3/P1)**((gamma-1.0)/(2*gamma))
    rho3L = gamma*P3/c3L**2

    # Right wave: shock for P3 > P4
    rho3R = rho4*((P3/P4 + (gamma-1.0)/(gamma+1.0))
                   / ((gamma-1.0)/(gamma+1.0)*P3/P4 + 1.0))
    S_shock = u4 + c4*np.sqrt((gamma+1.0)/(2*gamma)*P3/P4 + (gamma-1.0)/(2*gamma))

    # Rarefaction speeds on left
    S_head = u1 - c1
    S_tail = u3 - c3L

    xi = (x - x0)/t
    rho = np.empty_like(x); u = np.empty_like(x); P = np.empty_like(x)
    for i, s in enumerate(xi):
        if s < S_head:
            rho[i], u[i], P[i] = rho1, u1, P1
        elif s < S_tail:
            u[i] = 2/(gamma+1.0)*(c1 + (gamma-1.0)/2*u1 + s)
            c_loc = c1 - (gamma-1.0)/2*(u[i]-u1)
            rho[i] = rho1*(c_loc/c1)**(2/(gamma-1.0))
            P[i] = P1*(c_loc/c1)**(2*gamma/(gamma-1.0))
        elif s < u3:
            rho[i], u[i], P[i] = rho3L, u3, P3
        elif s < S_shock:
            rho[i], u[i], P[i] = rho3R, u3, P3
        else:
            rho[i], u[i], P[i] = rho4, u4, P4
    return rho, u, P


def main():
    print("Running Sod tests at three tau values...")
    N = 400
    t_end = 0.2
    cases = [
        (1e-5, r"Kn$\to 0$  (Euler limit, $\tau=10^{-5}$)", 'C0'),
        (5e-3, r"intermediate  ($\tau=5\times 10^{-3}$)", 'C1'),
        (1e3,  r"Kn$\to\infty$  (collisionless, $\tau=10^{3}$)", 'C2'),
    ]

    # JIT warmup
    print("  JIT warmup...", end=' ', flush=True)
    U0, _ = make_sod_ic(32)
    _ = run_to(U0, t_end=0.005, save_times=[0.005], tau=1e-3, bc="transmissive")
    print("done.")

    x_ex = np.linspace(0, 1, 2000)
    rho_ex, u_ex, P_ex = exact_riemann_sod(x_ex, t_end)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for tau, label, color in cases:
        t0 = time.time()
        U0, x = make_sod_ic(N)
        snaps, ns = run_to(U0, t_end=t_end, save_times=[t_end], tau=tau,
                            bc="transmissive")
        dt = time.time() - t0
        print(f"  tau={tau:g}: {ns} steps in {dt:.2f}s")
        U = snaps[-1][1]
        d = extract_diagnostics(U)
        P = (d['Pxx'] + 2*d['Pp'])/3
        anis = (d['Pxx'] - d['Pp'])/np.maximum(P, 1e-12)
        axes[0, 0].plot(x, d['rho'], color=color, label=label, lw=1.2)
        axes[0, 1].plot(x, d['u'],   color=color, lw=1.2)
        axes[1, 0].plot(x, P,        color=color, lw=1.2)
        axes[1, 1].plot(x, anis,     color=color, lw=1.2)

    # Exact solution overlays
    axes[0, 0].plot(x_ex, rho_ex, 'k:', lw=0.8, label='exact (Euler)')
    axes[0, 1].plot(x_ex, u_ex,   'k:', lw=0.8)
    axes[1, 0].plot(x_ex, P_ex,   'k:', lw=0.8)

    axes[0, 0].set_ylabel(r'$\rho$'); axes[0, 0].legend(fontsize=8, loc='lower left')
    axes[0, 1].set_ylabel('u')
    axes[1, 0].set_ylabel('P'); axes[1, 0].set_xlabel('x')
    axes[1, 1].set_ylabel(r'$(P_{xx} - P_\perp) / P$'); axes[1, 1].set_xlabel('x')
    axes[1, 1].axhline(0, color='k', lw=0.3)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Sod tube at three $\\tau$ values, $t = {t_end}$, $N = {N}$")
    out = os.path.join(FIG_DIR, "sod_three_tau.png")
    fig.savefig(out, dpi=130)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
