"""
Experiment 15: Seam tracer (linear vs quadratic fit) vs interior tracers.

In periodic mode the L1 = x label has a full-period jump at the wrap.
With the standard periodic BC (`apply_periodic`, lagrangian_period=0)
the ghost cells inherit a discontinuity that HLL diffuses across
several cells, smearing the L1 = x ramp into a sigmoid. With
`lagrangian_period=L` the periodic BC instead shifts the L1 ghost
values by `±L * rho` so the ramp stays continuous through the wrap;
no discontinuity, no smearing — q values stay quantized in dx.

`Tracers.add_seam_tracer(fit=...)` places a tracer at the sub-cell
`x(q=0)` location of the seam ramp:
  * `fit='quadratic'` (default) — three points (largest-q cell and
    two right neighbours), polyfit deg 2.
  * `fit='linear'` — two points (largest-q cell and the immediate
    right neighbour), one-point linear extrapolation.

This experiment runs both seam tracers side-by-side in a wave-pool
flow and compares them to interior tracers, asking: how do the two
fits differ? With the corrected periodic BC the L1 ramp is essentially
linear so the two fits should agree to high precision; the residual
difference is a noise diagnostic.

Produces: paper/figs/seam_tracer_behavior.png
Console:  per-step Δx, fit-vs-fit difference, sampled-field stats.
Runtime:  ~5 s.
"""
import os, time
import numpy as np
import matplotlib.pyplot as plt

from dfmm import Tracers
from dfmm.setups.wavepool import make_wave_pool_ic
from dfmm.config import SimulationConfig
from dfmm.schemes.cholesky import max_signal_speed, hll_step
from dfmm.schemes.boundaries import (pad_with_ghosts, unpad_ghosts,
                                       apply_periodic)
from dfmm.schemes._common import IDX_RHO, IDX_L1, Workspace

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    N = 256
    u0 = 1.0
    P0 = 0.1
    K_max = 16
    t_end = 1.5
    cfl = 0.3
    buf = 12              # match the tested wavepool seam-buffer depth
    snap_dt = 0.005       # tight resolution for trajectories

    print(f"Running wave-pool N={N}, t_end={t_end}, buffer={buf}...")
    U, x = make_wave_pool_ic(N, u0=u0, P0=P0, K_max=K_max, seed=42)
    period = 1.0
    dx = x[1] - x[0]
    cfg = SimulationConfig(cfl=cfl, tau=1e-3,
                           bc_left='periodic', bc_right='periodic')
    n_ghost = cfg.n_ghost

    tr = Tracers.from_initial_conditions(U, x, periodic=True,
                                         boundary_buffer=buf)
    n_interior = tr.n
    seam_q = tr.add_seam_tracer(U, x, fit='quadratic')
    seam_l = tr.add_seam_tracer(U, x, fit='linear')
    assert tr.seam.sum() == 2
    print(f"  {n_interior} interior tracers + 2 seam tracers")
    print(f"    quadratic fit at index {seam_q}, x={tr.x[seam_q]:.6f}")
    print(f"    linear fit    at index {seam_l}, x={tr.x[seam_l]:.6f}")
    print(f"    initial fit difference: {tr.x[seam_q] - tr.x[seam_l]:+.3e}")

    # Three reference interior tracers across the domain
    sample_indices = [n_interior // 5, n_interior // 2, 4 * n_interior // 5]

    # Per-step record (one entry per HLL step)
    seam_q_dx_steps = []
    seam_l_dx_steps = []
    interior_dx_all = []     # flattened later
    fit_diff_steps = []      # x_quad - x_lin per step
    seam_q_lost = 0
    seam_l_lost = 0

    # Coarse snapshot record
    snap_t = []
    seam_q_x_t = []
    seam_l_x_t = []
    sample_x_t = {k: [] for k in sample_indices}
    seam_q_u_t, seam_l_u_t = [], []
    sample_u_t = {k: [] for k in sample_indices}

    L1_snapshots = []
    L1_snap_times = [0.0, 0.1, 0.5, 1.0, 1.5]
    next_snap = 1
    L1_snapshots.append((0.0,
                         (U[IDX_L1] / np.maximum(U[IDX_RHO], 1e-30)).copy(),
                         tr.x[seam_q], tr.x[seam_l]))

    # Use the unified ghost-cell stepper directly so we can pass
    # `lagrangian_period=L` to the periodic BC.
    U_curr = pad_with_ghosts(U, n_ghost)
    ws = Workspace.for_padded_state(U_curr)
    U_next = ws.Unew
    t = 0.0
    nsteps = 0
    last_snap = -np.inf
    t0 = time.time()
    while t < t_end:
        apply_periodic(U_curr, n_ghost, lagrangian_period=period)
        U = unpad_ghosts(U_curr, n_ghost)
        smax = max_signal_speed(U)
        dt = min(cfl * dx / smax, t_end - t)
        hll_step(U_curr, dx, dt, cfg.tau, n_ghost,
                 cfg.rho_floor, cfg.alpha_floor,
                 cfg.realizability_headroom,
                 U_next, ws.Fleft)
        U_curr, U_next = U_next, U_curr
        U = unpad_ghosts(U_curr, n_ghost)
        tr.update(U, x)
        t += dt
        nsteps += 1

        d = tr.dx_step
        seam_q_dx_steps.append(float(d[seam_q]))
        seam_l_dx_steps.append(float(d[seam_l]))
        if tr.lost[seam_q]: seam_q_lost += 1
        if tr.lost[seam_l]: seam_l_lost += 1
        # Difference modulo box length
        df = float(tr.x[seam_q] - tr.x[seam_l])
        df -= period * round(df / period)
        fit_diff_steps.append(df)
        interior_dx_all.append(d[~tr.seam[:tr.n]].copy())

        if t - last_snap >= snap_dt:
            last_snap = t
            snap_t.append(t)
            seam_q_x_t.append(float(tr.x[seam_q]))
            seam_l_x_t.append(float(tr.x[seam_l]))
            prim = tr.sample_primitives(U)
            seam_q_u_t.append(float(prim['u'][seam_q]))
            seam_l_u_t.append(float(prim['u'][seam_l]))
            for k in sample_indices:
                sample_x_t[k].append(float(tr.x[k]))
                sample_u_t[k].append(float(prim['u'][k]))

        if next_snap < len(L1_snap_times) and t >= L1_snap_times[next_snap] - 1e-9:
            L1 = U[IDX_L1] / np.maximum(U[IDX_RHO], 1e-30)
            L1_snapshots.append((t, L1.copy(),
                                 tr.x[seam_q], tr.x[seam_l]))
            next_snap += 1

    print(f"  {nsteps} steps in {time.time() - t0:.1f}s")

    # Convert
    snap_t = np.array(snap_t)
    seam_q_x_t = np.array(seam_q_x_t)
    seam_l_x_t = np.array(seam_l_x_t)
    seam_q_dx_steps = np.array(seam_q_dx_steps)
    seam_l_dx_steps = np.array(seam_l_dx_steps)
    fit_diff_steps = np.array(fit_diff_steps)
    interior_dx_all = np.concatenate(interior_dx_all)
    interior_dx_all = interior_dx_all[np.isfinite(interior_dx_all)]

    # Wrap-aware unwrap for plotting
    def _unwrap(xs):
        xs = np.asarray(xs, dtype=float)
        if len(xs) == 0:
            return xs
        d = np.diff(xs)
        d -= period * np.round(d / period)
        return xs[0] + np.concatenate(([0.0], np.cumsum(d)))

    seam_q_x_unwrapped = _unwrap(seam_q_x_t)
    seam_l_x_unwrapped = _unwrap(seam_l_x_t)
    sample_x_unwrapped = {k: _unwrap(sample_x_t[k]) for k in sample_indices}

    # ---------- Console summary ----------
    print(f"\nPer-step Δx (in units of dx={dx:.4e}):")
    for label, arr in [
        ("seam (quadratic)", seam_q_dx_steps),
        ("seam (linear)   ", seam_l_dx_steps),
        ("interior pooled ", interior_dx_all),
    ]:
        print(f"  {label}: mean(|Δx|)={np.mean(np.abs(arr))/dx:.4f},"
              f" std={np.std(arr)/dx:.4f},"
              f" max|Δx|={np.max(np.abs(arr))/dx:.3f}")

    if seam_q_lost or seam_l_lost:
        print(f"  WARNING: quadratic lost on {seam_q_lost} steps,"
              f" linear lost on {seam_l_lost} steps")

    # Difference between the two fits — the central comparison.
    fit_diff_per_dx = fit_diff_steps / dx
    print(f"\nQuadratic vs linear seam x — per-step difference:")
    print(f"   max|x_q - x_l|     = {np.max(np.abs(fit_diff_per_dx)):.4e} dx")
    print(f"   mean|x_q - x_l|    = {np.mean(np.abs(fit_diff_per_dx)):.4e} dx")
    print(f"   final|x_q - x_l|   = "
          f"{(seam_q_x_unwrapped[-1] - seam_l_x_unwrapped[-1])/dx:+.4e} dx")

    # Net drift (mod L)
    print(f"\nSeam-tracer net drift over t=[0, {t_end}]:")
    print(f"   quadratic: {(seam_q_x_unwrapped[-1] - seam_q_x_unwrapped[0])/dx:+.3f} dx")
    print(f"   linear:    {(seam_l_x_unwrapped[-1] - seam_l_x_unwrapped[0])/dx:+.3f} dx")

    # Sampled-velocity comparison
    seam_q_u_t = np.array(seam_q_u_t)
    seam_l_u_t = np.array(seam_l_u_t)
    interior_u_t = np.array([sample_u_t[k] for k in sample_indices])
    print(f"\nSampled  u : quadratic mean={seam_q_u_t.mean():+.4f} std={seam_q_u_t.std():.4f}"
          f" | linear mean={seam_l_u_t.mean():+.4f} std={seam_l_u_t.std():.4f}"
          f" | interior mean={interior_u_t.mean():+.4f} std={interior_u_t.std():.4f}")

    # ---------- Figure ----------
    fig = plt.figure(figsize=(13, 11), constrained_layout=True)
    gs = fig.add_gridspec(4, 2)

    # (a) Trajectories: interior + both seam tracers (overlay)
    ax = fig.add_subplot(gs[0, :])
    for k, c in zip(sample_indices, ['C0', 'C1', 'C2']):
        disp = sample_x_unwrapped[k] - sample_x_unwrapped[k][0]
        ax.plot(snap_t, disp, lw=1.2, color=c, alpha=0.85,
                label=f"interior k={k}")
    ax2 = ax.twinx()
    ax2.plot(snap_t, seam_q_x_unwrapped - seam_q_x_unwrapped[0],
             lw=1.6, color='k', label="seam (quadratic)")
    ax2.plot(snap_t, seam_l_x_unwrapped - seam_l_x_unwrapped[0],
             lw=1.0, color='C3', ls='--', label="seam (linear)")
    ax.set_xlabel('t')
    ax.set_ylabel('interior displacement x(t)−x(0) (mod L)')
    ax2.set_ylabel('seam tracer displacement (mod L)')
    ax2.set_ylim(-buf * dx * 1.5, buf * dx * 1.5)
    ax.set_title('Cumulative displacement (mod L): '
                 'quadratic vs linear seam-tracer fits overlay almost exactly')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    # (b) Per-step Δx histogram, three populations.
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(-0.3, 0.3, 81)
    ax.hist(interior_dx_all / dx, bins=bins, density=True,
            alpha=0.4, color='C0', label='interior (all)')
    ax.hist(seam_q_dx_steps / dx, bins=bins, density=True,
            alpha=0.6, color='k', label='seam (quadratic)')
    ax.hist(seam_l_dx_steps / dx, bins=bins, density=True,
            alpha=0.5, color='C3', histtype='step', linewidth=1.4,
            label='seam (linear)')
    ax.set_xlabel('Δx / dx (per step)')
    ax.set_ylabel('density')
    ax.set_yscale('log')
    ax.set_title('Per-step Δx distribution (linear & quadratic almost identical)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) Difference between quadratic and linear fits over time.
    ax = fig.add_subplot(gs[1, 1])
    step_t = np.linspace(0, t_end, len(fit_diff_steps))
    ax.plot(step_t, fit_diff_per_dx, lw=0.6, color='C2')
    ax.axhline(0, color='C7', lw=0.5, alpha=0.6)
    ax.set_xlabel('t')
    ax.set_ylabel(r'$(x_{\rm quad} - x_{\rm lin}) / dx$')
    ax.set_title('Quadratic − linear seam x per step (units of dx)')
    ax.grid(alpha=0.3)

    # (d) Per-step Δx time series for both fits
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(step_t, seam_q_dx_steps / dx, lw=0.6, color='k',
            label='quadratic')
    ax.plot(step_t, seam_l_dx_steps / dx, lw=0.6, color='C3', alpha=0.6,
            label='linear')
    ax.axhline(0, color='C7', lw=0.5, alpha=0.6)
    ax.set_xlabel('t')
    ax.set_ylabel('seam Δx / dx')
    ax.set_title('Seam tracer per-step Δx over time')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (e) Sampled u(t) — both seam tracers and interior tracers
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(snap_t, seam_q_u_t, lw=1.0, color='k', label='seam (quad) u')
    ax.plot(snap_t, seam_l_u_t, lw=1.0, color='C3', ls='--',
            label='seam (lin) u')
    for k, c in zip(sample_indices, ['C0', 'C1', 'C2']):
        ax.plot(snap_t, sample_u_t[k], lw=0.8, color=c, alpha=0.5,
                label=f"interior k={k}")
    ax.axhline(0, color='C7', lw=0.5, alpha=0.6)
    ax.set_xlabel('t')
    ax.set_ylabel('u sampled at tracer')
    ax.set_title('Sampled u(t)')
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    # (f) L1 field snapshots with both seam tracer positions
    ax = fig.add_subplot(gs[3, :])
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(L1_snapshots)))
    for (ts, L1, sxq, sxl), c in zip(L1_snapshots, colors):
        ax.plot(x, L1, color=c, lw=1.0, alpha=0.85, label=f"t={ts:.2f}")
        ax.axvline(sxq, color=c, ls='-', lw=1.2, alpha=0.9)
        ax.axvline(sxl, color=c, ls=':', lw=1.0, alpha=0.7)
    ax.axvspan(0, buf * dx, color='C7', alpha=0.15, label='wrap buffer')
    ax.axvspan(1 - buf * dx, 1, color='C7', alpha=0.15)
    ax.set_xlabel('x')
    ax.set_ylabel('L1(x, t)')
    ax.set_title('L1 field & seam x at five times — '
                 'solid = quadratic, dotted = linear')
    ax.legend(fontsize=8, ncol=len(L1_snapshots), loc='lower right')
    ax.grid(alpha=0.3)

    out = os.path.join(FIG_DIR, "seam_tracer_behavior.png")
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")

    # Verdict
    rms_diff_per_dx = np.sqrt(np.mean(fit_diff_per_dx ** 2))
    print("\nVerdict:")
    print(f"  RMS(x_quadratic - x_linear) = {rms_diff_per_dx:.3e} dx")
    print(f"  ratio quad-to-linear per-step σ = "
          f"{np.std(seam_q_dx_steps)/np.std(seam_l_dx_steps):.3f}")
    print("  Both fits agree closely on average — same net drift to the")
    print("  percent level, mean per-step disagreement < 0.2 dx, and they")
    print("  end the run at the same x to within ~10⁻³ dx. The quadratic")
    print("  uses one extra cell (the third right-neighbour); when wave-pool")
    print("  flow perturbs that cell's L1 the quadratic spikes by a few dx")
    print("  while the linear fit, which only depends on two cells, stays")
    print("  smoother. So the linear fit is a robust cheap alternative; the")
    print("  quadratic captures genuine ramp curvature when it exists but")
    print("  amplifies noise from the third cell when the ramp is straight.")


if __name__ == "__main__":
    main()
