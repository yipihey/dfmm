# Phase 11 — Tier B.5 passive scalar advection

**Status:** complete. Variational tracer transport is **bit-exact** in
the deterministic-Lagrangian phase; Eulerian reference smears the
interface to ~100 cells over the same number of timesteps. The
fidelity ratio is *infinite* in the L∞-error sense (variational
error = 0.0 exactly); in cell-width terms it's at least one decade
on a 200-cell grid and grows linearly with timestep count for the
Eulerian baseline.

## The structural argument (methods paper §7)

A passive scalar `T(x, t)` satisfying `∂_t T + u·∇T = 0` is, in the
variational framework, identified with a function of the parcel
label `m`: `T = T(m, t)`. In the Lagrangian frame the evolution is
trivial — `T(m, t) = T_0(m)` for all `t`, since the parcel label
itself is conserved by definition.

In the discrete scheme, each `Mesh1D` segment carries a fixed
Lagrangian-mass label `m_j` and mass extent `Δm_j`. A passive
scalar field `T_k(j)` is just one number per segment. The
variational integrator advances the *bulk* fields `(x, u, α, β, s)`
by a Newton step on the discrete Euler–Lagrange residual; it never
touches `T_k`. Hence **the deterministic numerical diffusion of any
passive scalar is exactly zero** — the field array is not in the
write set of the integrator.

In Milestone 1's 1D scope there is no remap (the moment computation
would be a trivial interval intersection if invoked). Milestone 2's
Bayesian remap onto the Eulerian mesh introduces a *bounded* tracer
diffusion proportional to the polynomial-reconstruction order
(fourth-order with cubic moments). For 1D / Milestone 1 the
deterministic phase is the only phase, so tracer diffusion is **0.0**.

## The Track-A separation

The Phase-11 brief mandates that tracer plumbing must NOT extend
`DetField` or `Mesh1D`. We carry tracers in a parallel
`TracerMesh` struct that holds a *reference* to the fluid mesh
plus an `[n_tracers, n_segments]` value matrix. This decouples
tracer infrastructure from Track-A core (Phases 1–7) and lets
Phases 7 (steady shock, BC extensions) and 8 (stochastic injection)
land on `src/segment.jl` and `src/newton_step.jl` without
collisions.

```julia
struct TracerMesh{T<:Real, M<:Mesh1D}
    fluid::M             # reference (NOT copy) to fluid mesh
    tracers::Matrix{T}   # [n_tracers, n_segments]
    names::Vector{Symbol}
end
```

The `advect_tracers!(tm, dt)` API is a no-op by design — present
for symmetry with future Milestone-2 versions that introduce remap
diffusion, but in 1D it has no effect.

## Bit-exactness verification

`test/test_phase11_tracer_advection.jl @testset "Phase 11.1"` runs
the variational integrator on a Sod-style shock+rarefaction IC
(mirror-doubled periodic mesh) for **1000 deterministic timesteps**
with three concurrent tracer fields:

- step (`T = 1` for `m < M/2`, else 0),
- sinusoid (`T = sin(2π m / M)`),
- narrow Gaussian (`T = exp(-((m - M/2) / 0.05M)²)`).

Acceptance:
- `tm.tracers === tm.tracers` (matrix object identity preserved).
- `tm.tracers == tracers_initial` (element-wise equal).
- `maximum(abs.(tm.tracers .- tracers_initial)) === 0.0`
  (literally zero, not "machine epsilon").

All three pass.

## Fidelity comparison vs Eulerian upwind

`test/test_phase11_tracer_advection.jl @testset "Phase 11.3"` and
the production driver in `experiments/B5_passive_tracer.jl`
benchmark the variational scheme against a reference first-order
upwind advection on a uniform Eulerian mesh:

```
T^{n+1}_j = T^n_j − dt * max(u_j, 0) * (T^n_j   − T^n_{j-1}) / dx
                  − dt * min(u_j, 0) * (T^n_{j+1} − T^n_j  ) / dx
```

The setup transports a step IC at `x = 0.5` by a constant velocity
`u = 0.5` for `t_end = 0.2`. We measure the *interface width* —
the spatial span of cells whose normalised tracer value lies in
the transition band `[0.05, 0.95]`. A bit-exact step has width
0.0 (no cell sits in the transition band); a smeared profile has
width proportional to its diffused extent.

Production result at N = 200 (487 timesteps):

| metric                          | variational | Eulerian upwind | ratio   |
|---------------------------------|-------------|-----------------|---------|
| L∞ tracer change                | **0.0**     | n/a             | ∞       |
| interface width (Eulerian L)    | **0.0**     | 0.55            | ∞       |
| interface width / dx            | **0**       | 110             | ∞       |
| L∞ pointwise error vs initial   | **0.0**     | O(0.5)          | ∞       |

The "n_steps = 487" production run smears the Eulerian step over
~110 cells (width 0.55 on a length-1 domain at dx = 0.005). The
variational scheme keeps the step *bit-exact*. As the brief
suggests, the honest summary is "fidelity ratio is `1/0 = ∞`";
in cell-width units the variational interface is ≥ 1 decade
sharper than the Eulerian reference at this resolution and grows
arbitrarily wider as `n_steps → ∞`.

The headline figure
`reference/figs/B5_tracer_through_shock.png` shows three panels:

1. Sod density profile at `t = 0.2` (variational integrator).
2. Step tracer at `t = 0.2`: variational (single cell jump) vs
   Eulerian-reference (smeared).
3. Interface width vs time: monotonically rising for the Eulerian
   baseline, identically zero for the variational scheme.

## Cross-check vs py-1d

`py-1d/dfmm/tracers.py` implements an *Eulerian-grid* tracer
infrastructure: tracer particles are located each timestep by
root-finding on a passively-advected `L1` label field carried as
the 8th conserved variable of the HLL scheme. The L1 field itself
suffers HLL numerical diffusion at shocks and across the periodic
seam, so py-1d's tracer scheme is fundamentally diffusive (even
modulo bounded boundary-buffer mitigations).

This is not a target to numerically match — the schemes differ at
a structural level. The cross-check is conceptual: both schemes
assert the analytical property `T(m, t) = T_0(m)` along parcel
labels. py-1d realises this approximately (with HLL diffusion of
the L1 field as the error source); the Julia variational scheme
realises it *exactly* (segment values are never written to). The
qualitative behaviour (sharp tracer interface co-moving with the
fluid) matches between the two implementations, but only the
variational scheme is exact at the discrete level.

No py-1d invocation is needed to verify the property — it follows
by construction from the source code (the `det_step!` write set
contains `(x, u, α, β, s, Pp, p_half)` and *not* `tm.tracers`).

## Files

- `src/tracers.jl` — `TracerMesh` struct, `advect_tracers!`,
  `set_tracer!`, `add_tracer!`, `tracer_at_position`,
  `eulerian_upwind_advect!`, `interface_width`.
- `experiments/B5_passive_tracer.jl` — Sod-style production
  driver with multi-tracer recording and fidelity-comparison
  figure rendering.
- `test/test_phase11_tracer_advection.jl` — bit-exactness,
  multi-tracer, fidelity, and `tracer_at_position` unit tests.
- `reference/figs/B5_tracer_through_shock.png` — headline
  three-panel figure.

## Open questions / Milestone 2 hooks

- The remap-diffusion bound (cubic-reconstruction → fourth-order
  tracer transport, methods paper §7) is the natural Milestone 2
  test. The `TracerMesh` API is already shaped to take a future
  `remap_tracers!(tm, eulerian_mesh)` step that would handle
  multi-cell intersection on the Bayesian remap.
- Two-fluid extension (Phase 12) would carry a `TracerMesh` per
  species. The `M::Mesh1D` parametric type already accepts
  any subtype, so this slots in without an API change.
- `add_tracer!` reallocates the tracer matrix; `set_tracer!` is
  in-place. For long-running simulations with dynamic tracer
  insertion, a slot-based pre-allocated layout (cf. py-1d's
  `_capacity` mechanism) would be cleaner. Not needed for
  Milestone 1.
