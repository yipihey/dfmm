"""
Experiment 05: Dust-in-gas cold sinusoid across three stopping-time regimes.

Two species, same initial sinusoidal velocity field:
    A (dust): large self-tau (collisionless), Epstein drag to gas
    B (gas):  small self-tau (Eulerian hydro)

Scans the grain radius through three decades to span the stopping-time
regimes: tightly coupled, intermediate, decoupled.

Produces: paper/figs/dust_gas_sinusoid.png
Runtime:  ~30 seconds
"""
import os, time
import numpy as np
import matplotlib.pyplot as plt

from dfmm.schemes import two_fluid as tf

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def run_dust_gas(N, t_end, grain_radius, A=1.0, T0_gas=1e-3, T0_dust=1e-5,
                 dust_to_gas=0.1, tau_dd=1e3, tau_gg=1e-5, cfl=0.3):
    x = np.linspace(0, 1, N, endpoint=False) + 0.5/N
    dx = 1.0/N
    rho_d = np.full(N, dust_to_gas)
    u_d   = A*np.sin(2*np.pi*x)
    P_d   = rho_d * T0_dust
    rho_g = np.ones(N)
    u_g   = A*np.sin(2*np.pi*x)
    P_g   = rho_g * T0_gas
    U = tf.make_initial_state(N, rho_d, u_d, P_d, rho_g, u_g, P_g, x_coords=x)
    t = 0.0; nsteps = 0
    kparams = (grain_radius, 1.0)
    while t < t_end:
        smax = tf.max_signal_speed_both(U, N)
        dt = min(cfl*dx/smax, t_end - t)
        if dt <= 0: break
        U = tf.step_two_species(U, dx, dt, tau_dd, tau_gg,
                                  tf.kernel_epstein, kparams, 1.0, 1.0,
                                  bc='periodic')
        t += dt; nsteps += 1
    return x, U, nsteps


def main():
    print("JIT warmup...", end=' ', flush=True)
    _ = run_dust_gas(N=32, t_end=0.001, grain_radius=1e-3)
    print("done.")

    N = 400; t_end = 0.5
    # Three regimes of Epstein stopping time
    cases = [(1e-5, 'tightly coupled', 'C0'),
             (1e-3, 'intermediate',    'C1'),
             (1e-1, 'decoupled',       'C2')]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for a_grain, label, color in cases:
        t0 = time.time()
        x, U, ns = run_dust_gas(N=N, t_end=t_end, grain_radius=a_grain)
        print(f"  a={a_grain:g} ({label}): {ns} steps in {time.time()-t0:.2f}s")
        # Extract per-species primitives (returns a dict)
        p_d = tf.primitives(U, species='A', m=1.0)
        p_g = tf.primitives(U, species='B', m=1.0)
        axes[0, 0].plot(x, p_d['u'], color=color, lw=1.1, label=f'{label}')
        axes[0, 1].plot(x, p_g['u'], color=color, lw=1.1)
        axes[1, 0].plot(x, p_d['rho'], color=color, lw=1.1)
        axes[1, 1].plot(x, p_d['u'] - p_g['u'], color=color, lw=1.1)

    axes[0, 0].set_ylabel('dust velocity'); axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_ylabel('gas velocity')
    axes[1, 0].set_ylabel('dust density')
    axes[1, 1].set_ylabel(r'$u_d - u_g$ (slip velocity)')
    for ax in axes.flat:
        ax.set_xlabel('x'); ax.grid(alpha=0.3)
    fig.suptitle(rf"Dust-in-gas at $t = {t_end}$ across three grain sizes")
    out = os.path.join(FIG_DIR, "dust_gas_sinusoid.png")
    fig.savefig(out, dpi=130)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
