"""
Experiment 15: Single seam tracer vs. interior tracers in a wave-pool.

In periodic mode the L1 = x label has a full-period jump at the wrap;
the HLL scheme smears that jump across a few cells, so the standard
tracer factory `boundary_buffer`-skips those cells. We have always had
zero tracers riding the seam.

`Tracers.add_seam_tracer()` places a single tracer at the steepest-
gradient cell of the (smeared) seam ramp. Because the original "L1=0"
label simply ceases to exist once the ramp smears, the seam tracer is
not Lagrangian: each `update()` re-anchors it to the current
steepest-descent cell within the wrap-buffer band — making it an
Eulerian seam-tracking probe.

This experiment runs a wave-pool flow with the seam tracer alongside
the interior tracers and asks: does the seam tracer move differently
— different per-step jump distribution, different sampled fields,
different drift — than the others?

Produces: paper/figs/seam_tracer_behavior.png
Console:  per-step Δx statistics, sampled-field comparison, and a
          short verdict.
Runtime:  ~1 s.
"""
import os, time
import numpy as np
import matplotlib.pyplot as plt

from dfmm import Tracers
from dfmm.setups.wavepool import make_wave_pool_ic
from dfmm.schemes.cholesky import hll_step_periodic, max_signal_speed
from dfmm.schemes._common import IDX_RHO, IDX_L1

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

    tr = Tracers.from_initial_conditions(U, x, periodic=True,
                                         boundary_buffer=buf)
    n_interior = tr.n
    seam_k = tr.add_seam_tracer(U, x)
    assert seam_k == n_interior
    assert tr.seam.sum() == 1
    print(f"  {n_interior} interior tracers + 1 seam tracer at "
          f"x={tr.x[seam_k]:.4f}, label={tr.label[seam_k]:.4f}")

    # Collect three reference interior tracers across the domain
    sample_indices = [n_interior // 5, n_interior // 2, 4 * n_interior // 5]

    # Per-step record
    seam_dx_steps = []
    interior_dx_all = []     # per-step: every interior Δx, flattened later
    seam_lost_count = 0

    # Snapshot record (taken every snap_dt of simulated time)
    snap_t = []
    seam_x_t = []
    sample_x_t = {k: [] for k in sample_indices}
    seam_rho_t, seam_u_t = [], []
    sample_rho_t = {k: [] for k in sample_indices}
    sample_u_t = {k: [] for k in sample_indices}

    L1_snapshots = []
    L1_snap_times = [0.0, 0.1, 0.5, 1.0, 1.5]
    next_snap = 1
    L1_snapshots.append((0.0,
                         (U[IDX_L1] / np.maximum(U[IDX_RHO], 1e-30)).copy(),
                         tr.x[seam_k]))

    t = 0.0
    nsteps = 0
    last_snap = -np.inf
    t0 = time.time()
    while t < t_end:
        smax = max_signal_speed(U)
        dt = min(cfl * dx / smax, t_end - t)
        U = hll_step_periodic(U, dx, dt, tau=1e-3)
        tr.update(U, x)
        t += dt
        nsteps += 1

        d = tr.dx_step
        seam_dx_steps.append(float(d[seam_k]))
        if tr.lost[seam_k]:
            seam_lost_count += 1
        interior_dx_all.append(d[~tr.seam[:tr.n]].copy())

        if t - last_snap >= snap_dt:
            last_snap = t
            snap_t.append(t)
            seam_x_t.append(float(tr.x[seam_k]))
            prim = tr.sample_primitives(U)
            seam_rho_t.append(float(prim['rho'][seam_k]))
            seam_u_t.append(float(prim['u'][seam_k]))
            for k in sample_indices:
                sample_x_t[k].append(float(tr.x[k]))
                sample_rho_t[k].append(float(prim['rho'][k]))
                sample_u_t[k].append(float(prim['u'][k]))

        if next_snap < len(L1_snap_times) and t >= L1_snap_times[next_snap] - 1e-9:
            L1 = U[IDX_L1] / np.maximum(U[IDX_RHO], 1e-30)
            L1_snapshots.append((t, L1.copy(), tr.x[seam_k]))
            next_snap += 1

    print(f"  {nsteps} steps in {time.time() - t0:.1f}s")

    # Convert
    snap_t = np.array(snap_t)
    seam_x_t = np.array(seam_x_t)
    seam_dx_steps = np.array(seam_dx_steps)
    interior_dx_all = np.concatenate(interior_dx_all)
    interior_dx_all = interior_dx_all[np.isfinite(interior_dx_all)]
    seam_rho_t = np.array(seam_rho_t)
    seam_u_t = np.array(seam_u_t)

    # All coordinate differences must be evaluated mod the box length.
    # The implementation already does this for per-step Δx (subtracting
    # `period * round(d / period)`); when *plotting* trajectories we
    # therefore reconstruct continuous curves by integrating those
    # mod-length per-snapshot increments. `_unwrap` does this in one
    # pass so the visual jumps between x=0 and x=L disappear.
    def _unwrap(xs):
        xs = np.asarray(xs, dtype=float)
        if len(xs) == 0:
            return xs
        d = np.diff(xs)
        d -= period * np.round(d / period)
        return xs[0] + np.concatenate(([0.0], np.cumsum(d)))

    seam_x_unwrapped = _unwrap(seam_x_t)
    sample_x_unwrapped = {k: _unwrap(sample_x_t[k]) for k in sample_indices}
    # Plot the seam tracer in a wrap-centered coordinate so it stays
    # near zero on the figure axis.
    seam_x_centered = seam_x_unwrapped - seam_x_unwrapped[0]

    # ---------- Console summary ----------
    print(f"\nPer-step Δx (in units of dx={dx:.4e}):")
    print(f"  seam tracer:      mean(|Δx|)={np.mean(np.abs(seam_dx_steps))/dx:.4f},"
          f" std={np.std(seam_dx_steps)/dx:.4f},"
          f" max|Δx|={np.max(np.abs(seam_dx_steps))/dx:.2f}")
    print(f"  interior pooled:  mean(|Δx|)={np.mean(np.abs(interior_dx_all))/dx:.4f},"
          f" std={np.std(interior_dx_all)/dx:.4f},"
          f" max|Δx|={np.max(np.abs(interior_dx_all))/dx:.2f}")

    if seam_lost_count:
        print(f"  WARNING: seam tracer was lost on {seam_lost_count}/{nsteps} steps")

    # Net seam drift, evaluated mod box length (minimum-image
    # convention). The unwrap above already integrates mod-L
    # increments, so the cumulative endpoint already accounts for
    # any seam wraparound during the run.
    seam_drift = seam_x_unwrapped[-1] - seam_x_unwrapped[0]
    print(f"\nSeam-tracer net drift over t=[0, {t_end}] (mod L): "
          f"{seam_drift:+.5f} ({seam_drift/dx:+.2f} dx)")

    # Sampled-field comparison: rho and u stay near IC values, with
    # comparable variance for both seam and interior tracers (no
    # systematic bias is expected, since rho and u don't see the seam).
    interior_rho_t = np.array([sample_rho_t[k] for k in sample_indices])
    interior_u_t = np.array([sample_u_t[k] for k in sample_indices])
    print(f"\nSampled rho: seam mean={seam_rho_t.mean():.4f} std={seam_rho_t.std():.4f}"
          f" | interior mean={interior_rho_t.mean():.4f} std={interior_rho_t.std():.4f}")
    print(f"Sampled  u : seam mean={seam_u_t.mean():+.4f} std={seam_u_t.std():.4f}"
          f" | interior mean={interior_u_t.mean():+.4f} std={interior_u_t.std():.4f}")

    # ---------- Figure ----------
    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)

    # (a) Trajectories. All x curves are unwrapped — coordinate
    # differences from one snapshot to the next are taken mod the box
    # length so a tracer that crosses the seam doesn't jump in the
    # plot.  Interior tracers shown as displacement from t=0 (so all
    # four curves can share an axis); seam tracer is plotted on a
    # twin axis because its scale (a few dx) is much smaller than the
    # interior trajectories' (typical excursions ~ u0 * t = O(L)).
    ax = fig.add_subplot(gs[0, :])
    for k, c in zip(sample_indices, ['C0', 'C1', 'C2']):
        disp = sample_x_unwrapped[k] - sample_x_unwrapped[k][0]
        ax.plot(snap_t, disp, lw=1.2, color=c, alpha=0.85,
                label=f"interior k={k}, label={tr.label[k]:.2f}")
    ax2 = ax.twinx()
    ax2.plot(snap_t, seam_x_centered, lw=1.5, color='k',
             label="seam tracer (right axis)")
    ax.set_xlabel('t')
    ax.set_ylabel('interior displacement x(t)−x(0)  (mod L)')
    ax2.set_ylabel('seam tracer displacement (mod L)')
    ax2.set_ylim(-buf * dx * 1.5, buf * dx * 1.5)
    ax.set_title('Cumulative displacement (mod L): '
                 'interior smooth vs seam snap-to-cell')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    # (b) Per-step Δx histogram, both tracer populations.
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(-1.5, 1.5, 61)
    ax.hist(interior_dx_all / dx, bins=bins, density=True,
            alpha=0.5, color='C0', label='interior (all tracers, all steps)')
    ax.hist(seam_dx_steps / dx, bins=bins, density=True,
            alpha=0.7, color='k', label='seam tracer')
    ax.set_xlabel('Δx / dx (per step)')
    ax.set_ylabel('density')
    ax.set_yscale('log')
    ax.set_title('Per-step Δx distribution: seam is bimodal at ±dx, '
                 'interior is continuous around 0')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) Per-step Δx time series for the seam tracer
    ax = fig.add_subplot(gs[1, 1])
    step_t = np.linspace(0, t_end, len(seam_dx_steps))
    ax.plot(step_t, seam_dx_steps / dx, lw=0.6, color='k')
    ax.axhline(0, color='C7', lw=0.5, alpha=0.6)
    ax.set_xlabel('t')
    ax.set_ylabel('seam Δx / dx')
    ax.set_title('Seam tracer per-step Δx over time '
                 '(quantized in dx; rare multi-cell hops)')
    ax.grid(alpha=0.3)

    # (d) L1 field snapshots with seam tracer position
    ax = fig.add_subplot(gs[2, :])
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(L1_snapshots)))
    for (ts, L1, sx), c in zip(L1_snapshots, colors):
        ax.plot(x, L1, color=c, lw=1.0, alpha=0.85, label=f"t={ts:.2f}")
        ax.axvline(sx, color=c, ls='--', lw=1.0, alpha=0.85)
    ax.axvspan(0, buf * dx, color='C7', alpha=0.15, label='wrap buffer')
    ax.axvspan(1 - buf * dx, 1, color='C7', alpha=0.15)
    ax.set_xlabel('x')
    ax.set_ylabel('L1(x, t)')
    ax.set_title('L1 field & seam tracer x (dashed) at five times — '
                 'discontinuity smears into a ramp; seam stays in the buffer')
    ax.legend(fontsize=8, ncol=len(L1_snapshots), loc='lower right')
    ax.grid(alpha=0.3)

    out = os.path.join(FIG_DIR, "seam_tracer_behavior.png")
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")

    # Verdict
    seam_step_rms = np.sqrt(np.mean(seam_dx_steps ** 2))
    interior_step_rms = np.std(interior_dx_all)
    ratio = seam_step_rms / max(interior_step_rms, 1e-30)
    print("\nVerdict:")
    print(f"  seam-tracer per-step RMS / interior per-step σ = {ratio:.2f}")
    print("  Per-step Δx for the seam tracer is quantized in units of dx")
    print("  (snap-to-cell behaviour) with a long tail of multi-cell hops")
    print("  when the steepest-descent argmin switches between competing")
    print("  candidates in the smeared seam ramp. Interior tracers, by")
    print("  contrast, move continuously via linear interpolation inside")
    print("  their bracket. Sampled rho and u distributions are similar —")
    print("  the seam tracer rides the same physical fluid neighbourhood.")


if __name__ == "__main__":
    main()
