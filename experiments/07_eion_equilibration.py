"""
Experiment 07: Electron-ion Coulomb equilibration.

Uniform-density, uniform-velocity plasma with initially different
electron and ion temperatures. Under Coulomb collisions the
temperatures relax toward a common value at a rate set by the
mass-ratio-dependent cross-coupling. No spatial gradients, no hydro
signal speed — this is the cleanest test of the two-fluid kernel.

Analytic expectation:

    d(T_e - T_i)/dt = -nu_T * (T_e - T_i)

with nu_T / nu_p = 2 m_e m_i / (m_e + m_i)^2. For m_e << m_i the
ratio nu_p / nu_T ~ m_i / (2 m_e), so momentum equilibrates fast,
temperature slow: the defining feature of plasmas.

Produces: paper/figs/eion_equilibration.png
Runtime:  ~5 seconds.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from momentlag.schemes import two_fluid as tf

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def run_equilibration(t_end=10.0, m_e=0.01, m_i=1.0, Z_e=1.0, Z_i=1.0,
                       lnLambda=10.0, Te0=2.0, Ti0=0.5, n_snaps=200,
                       N=32):
    x = np.linspace(0, 1, N, endpoint=False) + 0.5/N
    rho_e = np.full(N, m_e)          # n = 1, rho = n*m
    rho_i = np.full(N, m_i)
    u_e = np.zeros(N); u_i = np.zeros(N)
    P_e = rho_e * Te0 / m_e
    P_i = rho_i * Ti0 / m_i
    U = tf.make_initial_state(N, rho_e, u_e, P_e, rho_i, u_i, P_i,
                                 x_coords=x)

    history = []
    t = 0.0; nsteps = 0
    snap_dt = t_end/n_snaps
    next_snap = 0.0
    while t < t_end:
        nu_p, nu_T = tf.kernel_coulomb(U, m_e, m_i, Z_e, Z_i, lnLambda, N)
        dt = min(0.1/max(nu_p.max(), 1e-30), t_end - t,
                  next_snap - t + snap_dt)
        if dt <= 0:
            dt = snap_dt/10
        tf.apply_cross_coupling(U, dt, nu_p, nu_T, m_e, m_i, N)
        t += dt; nsteps += 1
        if t >= next_snap:
            pe = tf.primitives(U, 'A', m=m_e)
            pi = tf.primitives(U, 'B', m=m_i)
            history.append(dict(t=t, Te=pe['T'][0], Ti=pi['T'][0]))
            next_snap += snap_dt
    return history


def main():
    print("JIT warmup...", end=' ', flush=True)
    _ = run_equilibration(t_end=0.01, n_snaps=2)
    print("done.")

    mass_ratios = [(1.0,  r'$m_e/m_i = 1$',    'C0'),
                   (0.1,  r'$m_e/m_i = 0.1$',  'C1'),
                   (0.01, r'$m_e/m_i = 0.01$', 'C3')]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for mr, label, color in mass_ratios:
        hist = run_equilibration(m_e=mr, m_i=1.0, lnLambda=10.0, t_end=10.0)
        ts = np.array([h['t'] for h in hist])
        Te = np.array([h['Te'] for h in hist])
        Ti = np.array([h['Ti'] for h in hist])
        axes[0].plot(ts, Te, color=color, lw=1.3, label=label + ' $T_e$')
        axes[0].plot(ts, Ti, color=color, lw=1.3, ls='--')
        DT = Te - Ti
        axes[1].semilogy(ts, np.abs(DT)/np.abs(DT[0]),
                          color=color, lw=1.3, label=label)

    axes[0].set_xlabel('t'); axes[0].set_ylabel('T')
    axes[0].set_title('Electron (solid) and ion (dashed) temperatures')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('t')
    axes[1].set_ylabel(r'$|T_e - T_i| / |T_e - T_i|_{t=0}$')
    axes[1].set_title('Relative temperature difference (exponential decay)')
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3, which='both')

    out = os.path.join(FIG_DIR, "eion_equilibration.png")
    fig.savefig(out, dpi=130)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
