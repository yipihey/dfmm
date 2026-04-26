# M3-3 — 2D Cholesky + Berry connection on the native HG path

> **Status (2026-04-26):** *Design note*, no code written. This document
> briefs the agent that will execute Phase M3-3 of
> `reference/MILESTONE_3_PLAN.md`. Read it together with the §10
> references — especially `src/berry.jl` and the HG halo/sparsity/BC
> source files — and you will have everything you need to start coding
> the per-cell EL residual on top of the HG substrate.

## 1. Goal recap

M1 + M2 shipped a 1D variational Hamilton–Pontryagin scheme on the
hand-rolled `Mesh1D{T,DetField{T}}`. M3-0/1/2 ported every M1+M2 path
onto HierarchicalGrids.jl (HG) `SimplicialMesh{1, T}` + `PolynomialFieldSet`
through a thin `cache_mesh::Mesh1D` shim, with **bit-exact parity to M1
verified to 0.0 absolute** across 2806 tests
(`reference/notes_M3_2_phase7811_m2_port.md`). M3-prep then derived,
verified, and pre-computed the Berry-connection stencils in pure Julia:
`src/berry.jl` ships the 2D / 3D / off-diagonal Berry forms, all
cross-checked against the SymPy scripts under `scripts/`
(`reference/notes_M3_prep_berry_stencil.md`). M3-2b is in flight in
parallel and will retire some of the per-test cache_mesh round-tripping
in the AMR/realizability paths.

**M3-3 is the next major scientific phase**: extend the per-cell EL
system from M1's scalar-1D `(x, u, α, β, s)` layout to the 2D
principal-axis-decomposed layout
$(x_a, u_a, \alpha_a, \beta_a, \theta_R, s)$ for `a = 1, 2`, with the
Berry-connection coupling
$\Theta_{\rm rot}^{(2D)} = \tfrac{1}{3}(\alpha_1^3\beta_2 - \alpha_2^3\beta_1)\,d\theta_R$
making the per-cell Lagrangian invariant under principal-axis rotation.
M3-3 also delivers the **first native HG-side** EL residual
(consuming `HaloView` + `cell_adjacency_sparsity` directly), retiring
the M3-0/1/2 cache_mesh shim for the dimension-lifted code path.

The math is *already done* and verified — see §2 below for the EL
structure and §10 for the full reference list. M3-3 is "encode the
verified stencils + new EL residual on top of HG". This is mostly
engineering with two small theoretical hot-spots: (a) the
solvability constraint $dH \perp v_{\rm ker}$ that fixes
$\mathcal{H}_{\rm rot}$ in the 5D diagonal sector
(`notes_M3_phase0_berry_connection.md` §6.6); (b) the off-diagonal
$\beta_{12}, \beta_{21}$ sector, which is gated for D.1 KH and which
M3-3 should leave optional / dormant (see §4 below).

## 2. New EL system structure

### 2.1 Per-cell unknowns: 1D → 2D → 3D

| D | Unknowns per leaf cell | Count |
|---|---|---|
| 1 | $(x, u, \alpha, \beta, s)$ | 5 (with $P_\perp, Q$ post-Newton: 7) |
| 2 (M3-3) | $(x_a, u_a)_{a=1,2}, (\alpha_a, \beta_a)_{a=1,2}, \theta_R, s$ | 10 (12 with $P_\perp, Q$) |
| 2 (M3-3 + off-diag, D.1 KH) | + $(\beta_{12}, \beta_{21})$ | 12 (14) |
| 3 (M3-7) | $(x_a, u_a)_{a=1,2,3}, (\alpha_a, \beta_a)_{a=1,2,3}, \theta_{12}, \theta_{13}, \theta_{23}, s$ | 15 (17) |

The Cholesky-sector subset that the M3-3 EL residual evaluates
directly (i.e., the unknowns for which Newton has to drive the residual
to zero) is

$$\boxed{\;
(x_a, u_a, \alpha_a, \beta_a)_{a=1,2}\;+\;\theta_R\;+\;s \quad\text{= 10 dof per leaf cell.}
\;}$$

The other M1+M2 sectors — $P_\perp$ (deviatoric / Phase 5), $Q$ (heat
flux / Phase 7), tracers (Phase 11), realizability projection
(M2-3) — remain post-Newton operator-split (or explicit) updates,
exactly as they are in 1D today, but with per-axis decomposition
applied where they need it (e.g., per-axis stochastic injection
already lives in `src/stochastic_injection.jl` for M2-3 / Phase 8).

The **discrete EL system grows from 4N → 10N** per leaf cell (vs the
M3 plan's earlier "4N → 5N" estimate, which counted only the
Cholesky-sector subset $(\alpha_a, \beta_a, \theta_R)$; the
implementer also needs to advance $(x_a, u_a)$ per axis. Only
$\alpha, \beta, \theta_R, s$ are cell-centred; $x, u$ are
vertex-centred in M1's 1D code, but on HG's `SimplicialMesh` with
`PolynomialFieldSet` they all sit per-cell as polynomial coefficients
of order 0 + their ghost-cell halos. See `src/setups_2d.jl` for the
D-generic field set already in use for the Tier-C ICs.).

If M3-3 enables the off-diagonal $\beta_{ab}$ sector for D.1 KH
(see §4.4), unknowns grow to 12N. We **recommend pinning
$\beta_{12} = \beta_{21} = 0$ initially** (realizability boundary,
see §5 of the off-diag derivation in
`notes_M3_phase0_berry_connection.md` §7), keeping M3-3 at 10N. D.1
will gate the off-diagonal evolution and is naturally separated as
M3-6 work.

### 2.2 Where the EL residual terms come from

The discrete Hamilton–Pontryagin Lagrangian per cell, with the M1
sign convention (v2 §3.1; methods paper §3.2 / §9.4 / §5.2), is

$$
L_{\rm cell}^{(2D)} \;=\;
\underbrace{\tfrac12 \sum_{a=1,2} \dot x_a^2}_{\rm kinetic\ position}
\;+\;
\underbrace{\sum_{a=1,2}\bigl[-\tfrac{\alpha_a^3}{3}\,\mathcal{D}_t^{(1)}\beta_a + \tfrac12 \alpha_a^2 \gamma_a^2\bigr]}_{\rm per\!-\!axis\ Cholesky\ (M1\ form)}
\;-\;
\underbrace{\tfrac{1}{3}(\alpha_1^3\beta_2 - \alpha_2^3\beta_1)\,\dot\theta_R}_{\rm Berry\ kinetic\ (M3\!-\!3\ NEW)}
\;-\;
\underbrace{\mathcal{H}_{\rm rot}(\alpha_a, \beta_a)}_{\rm rotation\ Hamiltonian}
\;-\;
\underbrace{P(J, s)\,\dot J/\rho}_{\rm EOS\ coupling}
$$

with $\gamma_a^2 = M_{vv,a}(J, s) - \beta_a^2$ as in M1 (per axis now)
and $J$ the per-cell Jacobian. Each term contributes to the EL residual
for a specific subset of unknowns:

| Term in $L$ | Drives residual w.r.t. |
|---|---|
| $\tfrac12 \dot x_a^2$ + EOS | $x_a$ (forces from $\partial P/\partial x$ via the $J$-coupling stencil) |
| $-\tfrac{\alpha_a^3}{3}\mathcal{D}_t^{(1)}\beta_a$ | $\alpha_a, \beta_a$ (per-axis Cholesky; M1 boxed equations) |
| $\tfrac12 \alpha_a^2 \gamma_a^2$ | $\alpha_a, \beta_a, s$ (γ-block) |
| $-\tfrac{1}{3}(\alpha_1^3\beta_2 - \alpha_2^3\beta_1)\dot\theta_R$ | $\alpha_1, \alpha_2, \beta_1, \beta_2, \theta_R$ (Berry — closed forms in `src/berry.jl`) |
| $\mathcal{H}_{\rm rot}$ | $\theta_R$ (constraint-determined; see §2.3) |
| $P\,\dot J/\rho$ | $x_a, u_a, s$ |

The M1 + M2 1D residual already implements rows 1, 2, 3, 6 (per-axis
in the 2D code). The **only structurally new pieces** in M3-3 are:

1. The Berry kinetic term (row 4), which couples the four
   $(\alpha_a, \beta_a)$ scalars to the new $\theta_R$ unknown. The
   stencil is in `src/berry.jl::berry_partials_2d` and is closed-form
   — no AD needed, though AD is fine too (allocation-free under
   `SVector` inputs).
2. The rotation Hamiltonian term (row 5). In the 5D
   principal-axis-diagonal sector, $\mathcal{H}_{\rm rot}$ is *not* a
   free input — it is fixed by the kernel-orthogonality constraint
   $dH \perp v_{\rm ker}$ from
   `notes_M3_phase0_berry_connection.md` §6.6:
   $$\frac{\partial \mathcal{H}_{\rm rot}}{\partial \theta_R}\bigg|_{\rm constraint}
   = \frac{\gamma_1^2 \alpha_2^3}{3\alpha_1} - \frac{\gamma_2^2 \alpha_1^3}{3\alpha_2} - (\alpha_1^2 - \alpha_2^2)\beta_1\beta_2.$$
   See §6.4 below for the verification gate that exercises this
   formula (sample $(\alpha, \beta, \gamma)$ points + check kernel
   solvability).

### 2.3 Discrete-time mid-point stencil

The M1 mid-point rule for $\mathcal{D}_t^{(q)}$ (charge-q parallel
transport) generalizes per principal axis. The Berry term contributes
its own mid-point stencil for $\dot\theta_R$:

$$
\mathcal{D}_t^{(\rm rot)}\theta_R \;\approx\; \frac{\theta_R^{n+1} - \theta_R^n}{\Delta t}
\;+\; (\text{lab-frame strain rotation correction})
$$

where the strain-rotation correction is the kinematic-equation (3.1)
contribution from `notes_M3_phase0_berry_connection.md` §3:
$\tilde W_{12} - \tilde S_{12}\,(\alpha_1^2 + \alpha_2^2)/(\alpha_1^2 - \alpha_2^2)$.
M3-3 should reuse the M1 mid-point convention for the per-axis sectors
and treat the $\theta_R$ mid-point exactly the same way (averaging at
half-step, finite-difference of the angle increment). The resulting
discrete EL residual is symmetric under $n \leftrightarrow n+1$ swap
modulo the M1 sign convention, exactly like the per-axis 1D form.

The pseudo-code skeleton (one cell, per Newton iteration):

```
function det_el_residual_cell_2D(Ω_n, Ω_{n+1}, halo_view, cell_idx, params):
    # 1. Pull halos: Ω_{n+1} = (x, u, α₁, α₂, β₁, β₂, θ_R, s) at cell + neighbors
    self    = halo_view[cell_idx, (0, 0)]
    east    = halo_view[cell_idx, (+1, 0)]   # may be `nothing` at boundary
    west    = halo_view[cell_idx, (-1, 0)]
    north   = halo_view[cell_idx, (0, +1)]
    south   = halo_view[cell_idx, (0, -1)]

    # 2. Mid-point quantities per axis a = 1, 2
    α_mid_a = (self.α_a + halo_n.α_a) / 2  # per axis 'a'
    β_mid_a = (self.β_a + halo_n.β_a) / 2
    γ_mid_a² = M_vv_a(J_mid, s_mid) - β_mid_a²

    # 3. Per-axis Cholesky-sector residual (M1 form, per axis; reuse cholesky_sector.jl)
    R_α[a] = M1_residual_α(α_mid_a, β_mid_a, γ_mid_a, dt)
    R_β[a] = M1_residual_β(α_mid_a, β_mid_a, γ_mid_a, dt)

    # 4. Per-axis position/velocity residual (per-axis pressure stencil from neighbors)
    R_x[a] = (Ω_{n+1}.x_a - Ω_n.x_a) - dt * u_mid_a
    R_u[a] = (Ω_{n+1}.u_a - Ω_n.u_a) - dt * (-∂_a P / ρ + body forces)

    # 5. Berry-connection contribution (M3-3 NEW)
    α_mid = SVector(α_mid_1, α_mid_2)
    β_mid = SVector(β_mid_1, β_mid_2)
    dθ_R  = Ω_{n+1}.θ_R - Ω_n.θ_R
    dB    = berry_partials_2d(α_mid, β_mid, dθ_R)        # 5-vector
    R_α[1] += dB[1]  # ∂Θ/∂α₁ contribution
    R_α[2] += dB[2]
    R_β[1] += dB[3]
    R_β[2] += dB[4]
    R_θR    = dB[5]                                       # ∂Θ/∂θ_R + H_rot constraint

    # 6. Rotation-Hamiltonian solvability constraint (forces R_θR completion)
    R_θR += partial_H_rot_partial_θR(α, β, γ)

    # 7. Entropy + coupling
    R_s = M1_entropy_residual(...)

    return SVector(R_x[1], R_x[2], R_u[1], R_u[2], R_α[1], R_α[2],
                   R_β[1], R_β[2], R_θR, R_s)
end
```

The 10-vector residual feeds the Newton solve; sparsity comes from HG's
`cell_adjacency_sparsity(mesh; depth=1)` (see §3.3 below).

## 3. Native HG-side EL residual

Critical M3-3 deliverable: **retire the `cache_mesh::Mesh1D` shim**
that M3-0/1/2 used. The 2D code can't go through cache_mesh because
M1's 1D `Mesh1D` doesn't carry $\theta_R$ or per-axis fields, and
faking a 1D-symmetric configuration for every 2D cell is wasteful.
M3-3 writes a native HG-side residual that consumes `PolynomialFieldSet`
+ `HaloView` directly.

The four HG primitives that make this practical landed in HG commit
`e7a286c` (HaloView + cell_adjacency_sparsity) and `97c5035` (BCKind +
FrameBoundaries). Read them before writing the residual:

### 3.1 Per-cell unknowns via `PolynomialFieldSet`

`src/setups_2d.jl` already allocates a `PolynomialFieldSet` with named
scalar fields `(rho, ux, uy, P)` for the Tier-C tests. M3-3 extends
that allocation to include the Cholesky-sector fields:

```julia
# 2D: 10 dof per cell (excluding optional off-diagonal β)
fields_2D_cholesky = allocate_polynomial_fields(
    mesh,
    SoA(MonomialBasis{2, 0}(),
        :x_1, :x_2,                  # per-axis position
        :u_1, :u_2,                  # per-axis velocity
        :α_1, :α_2,                  # per-axis Cholesky 11
        :β_1, :β_2,                  # per-axis Cholesky 21 diagonal
        :θ_R,                        # principal-axis rotation angle
        :s,                          # entropy
        :P_perp, :Q                  # post-Newton sectors (M1 carry-over)
    )
)
```

Order-0 ("piecewise-constant") storage suffices for the first cut,
mirroring M3-0/1/2's 1D path. Higher-order Bernstein expansion (per
methods paper §9.2) is a follow-up — start with order 0, then bump.

### 3.2 Neighbor access via `HaloView`

**Read first:** `~/.julia/dev/HierarchicalGrids/src/Storage/HaloView.jl`.

`HaloView` wraps a `PolynomialFieldView` plus the lazily-built
`NeighborGraph` and supports

```julia
hv = halo_view(field, mesh, depth=1)
hv[i, off::NTuple{D, Int}]   # = field[neighbor at offset off] or `nothing` at boundary
```

with `off = (0, 0)` for the cell itself, `off = (+1, 0)` for the
east neighbor, `off = (0, -1)` for the south neighbor, etc. For
`depth = 1`, `|off[1]| + |off[2]| ≤ 1` is enforced. The contract:

- O(1) construction; no allocations on the hot path after the
  neighbor graph is built.
- Out-of-domain offsets return `nothing` — the residual must handle
  `nothing` gracefully (use the BC framework, see §3.4).
- For coarse-fine faces (when `mesh.balanced == false`), only the
  representative neighbor is returned. Call
  `face_fine_neighbors(mesh, i, face)` for the full list. M3-3 should
  start with `mesh.balanced == true` and defer hanging-node handling
  to a follow-up phase.
- `HaloView.depth` controls the maximum face-hop distance. A
  conservative second-order stencil needs `depth ≥ 1`; richer
  reconstructions (e.g., for the action-error indicator) might
  eventually need `depth = 2`.

**Defensive coding around HaloView (M3-3 risk).** The field-view
type returned by `hv[i, off]` is a *coefficient column* (a
`PolynomialView` or `Tuple` for order > 0). The 1D shim simply
indexed scalars. Before writing the residual against it, run a
smoke test that walks all 4 (or 8) face neighbors of a corner cell
in a 2-level uniform mesh and verifies the returned coefficient
matches what `field[neighbor_idx]` returns directly. This is the
single biggest unknown left for M3-3 — see §7.

### 3.3 Newton sparsity via `cell_adjacency_sparsity`

**Read first:** `~/.julia/dev/HierarchicalGrids/src/Mesh/Neighbors.jl`,
specifically the `cell_adjacency_sparsity` block at lines 401–510.

```julia
sparsity = cell_adjacency_sparsity(mesh; depth=1, leaves_only=true)
# returns SparseMatrixCSC{Bool, Int32}: M[i,j] = true iff cells i, j are within 1 face-hop
```

Newton-Jacobian sparsity for the 10N-dof system is then the
**Kronecker product** of `sparsity` (cell-cell) with a dense
$10 \times 10$ block (within-cell). The structural pattern is
exactly what `SparseConnectivityTracer` + `SparseMatrixColorings`
consume. The M1 Newton path is in `src/newton_step.jl` (M1) and
`src/newton_step_HG.jl` (M3-0/1/2 shim); the 2D native path is a new
file `src/newton_step_HG.jl` extension (see §5 below).

**Convergence-rate expectation.** M1's 4-dof-per-cell Newton
converged in 2-3 iterations on smooth ICs and 5-8 iterations on
shocks (with line-search). 10-dof-per-cell with a closed-form Berry
block + per-axis decoupling on smooth ICs should converge in a
similar 2-4 iterations on the dimension-lifted (1D-symmetric)
configurations and 5-10 on Tier-D ICs. If you see >20 iterations on
a smooth IC, check (a) the $\theta_R$ initialization matches the
principal-axis frame of $M_{xx}$ at $t = 0$, (b) the
$\mathcal{H}_{\rm rot}$ constraint of §2.2 is enforced, and (c) the
$\beta_{12}, \beta_{21}$ pinning is consistent (no off-diagonal
leakage).

### 3.4 Boundary conditions via `BCKind` + `FrameBoundaries`

**Read first:** `~/.julia/dev/HierarchicalGrids/src/Mesh/BoundaryConditions.jl`
and `~/.julia/dev/HierarchicalGrids/src/Overlap/frame_boundaries.jl`.

The HG BC framework is per-axis-half:

```julia
spec = ((PERIODIC, PERIODIC), (REFLECTING, REFLECTING))   # 2D, x-periodic + y-reflecting
fb   = FrameBoundaries(spec)
is_periodic_axis(fb, 1)   # true
is_periodic_axis(fb, 2)   # false
bc(fb, axis, side)        # → BCKind
```

`face_neighbors_with_bcs(mesh, i, fb)` is the BC-aware variant of
`face_neighbors`: for each periodic axis it post-processes the
boundary entries to wrap to the opposite-wall leaf with overlapping
other-axes extent. Non-periodic kinds (`INFLOW`, `OUTFLOW`,
`REFLECTING`, `DIRICHLET`) are advisory at the HG layer — **dfmm's
EL-residual code consumes them** to assemble the appropriate ghost
state when `hv[i, off]` returns `nothing`.

M3-3 should:

1. Construct a `FrameBoundaries{2}` per problem (Tier-C / Tier-D ICs
   declare it explicitly). M3-2b is delivering the BC framework
   integration on the 1D path in parallel — coordinate with that work
   to keep the 1D and 2D conventions identical.
2. In the residual, wrap `hv[cell, off]` access in a small helper
   `halo_or_ghost(hv, cell, off, fb)` that:
   - returns the cell value if not `nothing`,
   - if `nothing`, dispatches on `bc(fb, axis, side)` to construct the
     appropriate ghost state (mirror for `REFLECTING`, supplied state
     for `INFLOW`, extrapolated for `OUTFLOW`, pinned for `DIRICHLET`,
     wrap-around handled upstream by `face_neighbors_with_bcs` for
     `PERIODIC`).
3. Pin Tier-C C.1 to periodic-y / inflow-outflow-x to mirror M1's
   1D Sod boundary conditions exactly. C.2 (cold sinusoid) is
   doubly periodic. C.3 (plane wave) is doubly periodic.

### 3.5 Data-flow pipeline

```
                    ┌──────────────────────────────────────────────┐
                    │     HierarchicalMesh{2} + FrameBoundaries{2} │
                    └───────────────┬──────────────────────────────┘
                                    │
          ┌─────────────────────────┴─────────────────────────────┐
          │                                                       │
          ▼                                                       ▼
PolynomialFieldSet (per-cell                       cell_adjacency_sparsity(mesh; depth=1)
  α₁, α₂, β₁, β₂, θ_R, x₁, x₂, u₁, u₂, s, …)                       │
          │                                                       │
          ▼                                                       ▼
  halo_view(field, mesh, depth=1)                  Sparse Newton-Jacobian skeleton
          │                                              (Kron with 10×10 dense block)
          ▼                                                       │
   det_el_residual_cell_2D!                                       │
   (ghost-cell aware via fb)                                      │
          │                                                       │
          ├─ AD (ForwardDiff or Enzyme over halo'd cell tuple) ◀──┘
          │
          ▼
   sparse_jacobian (fed to NonlinearSolve.jl Newton)
          │
          ▼
   (x, u, α, β, θ_R, s)_{n+1} per cell
          │
          ▼
   post-Newton operator splits (M1 + M2 carry-over):
       Pp relaxation, Q decay, stochastic injection (per-axis),
       realizability projection (per-axis)
```

The pipeline mirrors M1's 1D Newton-then-post-Newton structure exactly.
The sole structural change is per-axis decomposition for the
operator-split sectors and the new $\theta_R$ unknown in the Newton
block.

## 4. Berry coupling integration

### 4.1 Reuse `src/berry.jl` directly

M3-prep already shipped `src/berry.jl` with verified Berry stencils
(see `reference/notes_M3_prep_berry_stencil.md`). M3-3 consumes
the module directly with **no source-code changes**:

```julia
using StaticArrays
using dfmm: berry_F_2d, berry_partials_2d, BerryStencil2D, kinetic_offdiag_2d

α_mid = SVector(α_mid_1, α_mid_2)
β_mid = SVector(β_mid_1, β_mid_2)
dθ_R  = θ_R_np1 - θ_R_n

dB    = berry_partials_2d(α_mid, β_mid, dθ_R)   # 5-vector: (∂α₁, ∂α₂, ∂β₁, ∂β₂, ∂θ_R)

# Add to the residual rows for (α₁, α₂, β₁, β₂, θ_R)
R[idx_α1] += dB[1]
R[idx_α2] += dB[2]
R[idx_β1] += dB[3]
R[idx_β2] += dB[4]
R[idx_θR] += dB[5]
```

The closed-form partials are verified in
`test/test_M3_prep_berry_stencil.jl` (181 asserts, including
finite-difference cross-check at `h = 1e-6`, rel-error ≤ 1e-7). Hot-path
allocation count is 0 with `@allocated`-warmed `SVector` inputs.

If the residual evaluation is JIT'd through ForwardDiff or Enzyme,
`berry_partials_2d` can be replaced by raw `berry_F_2d` and let AD
handle differentiation — both produce identical numbers up to FP
order.

### 4.2 BerryStencil2D pre-compute (optional cache)

`BerryStencil2D(α, β)` pre-computes the 2D Berry function and its α/β
gradients at a given $(\alpha, \beta)$. The per-step `dθ_R` factor
multiplies at use-time. If the 2D residual is evaluated from a
mid-point state $(\alpha_{\rm mid}, \beta_{\rm mid})$ that gets reused
across a Newton iteration, pre-computing the stencil amortizes
2-3 multiplications per cell:

```julia
stencil = BerryStencil2D(α_mid, β_mid)
contribution = apply(stencil, dθ_R)  # = F · dθ_R
∂α  = stencil.dF_dα * dθ_R           # SVector{2, T}
∂β  = stencil.dF_dβ * dθ_R           # SVector{2, T}
∂θR = stencil.F                      # T
```

For an order-0 cell-average residual the savings are marginal; for an
order-3 Bernstein per-cell cubic reconstruction the stencil should be
pre-computed once per quadrature point. M3-3 starts at order 0 and
need not optimize this; flag for a future performance pass.

### 4.3 Per-axis γ diagnostic

A central Tier-C deliverable is the **per-axis γ diagnostic**:

$$\gamma_a^2 \;=\; M_{vv,a}(J, s) \;-\; \beta_a^2,
\qquad \gamma_a \;=\; \sqrt{\max(\gamma_a^2, 0)}.$$

This is the per-axis analog of M1's `gamma_from_Mvv(β, M_vv)` and
plugs into:

- **AMR indicator selectivity** (per-axis rather than scalar). The
  action-error indicator `action_error_indicator_HG` (M3-2 carry-over)
  should accept a per-axis γ and refine only along axes where γ → 0
  (compressive direction). This is what enables the C.2 "per-axis γ
  selectivity" test (`γ_1 → 1e-5`, `γ_2 → O(1)` in a 1D-symmetric
  cold-sinusoid).
- **Hessian-degeneracy regularization** (M1 Phase 3 carry-over). The
  per-axis γ-floor protects the per-axis Newton solve when one axis
  collapses while another doesn't.
- **Per-axis stochastic injection** (M2-3 / Phase 8 carry-over). The
  realizability-checked injection acts on the compressive axis only;
  this is the D.1 KH falsifier (`reference/notes_dfmm_2d_overview.md`
  Tier D.1).

Per-axis γ is a derived quantity (not a Newton unknown). Compute it in
`src/diagnostics.jl` as `gamma_per_axis_2d(field, J, s)` returning
`SVector{2, T}`. M3-3 should add this helper and wire it through the
existing AMR / realizability / stochastic-injection code paths.

### 4.4 Off-diagonal $\beta_{12}, \beta_{21}$ — pin to zero in M3-3

The off-diagonal $L_2$ sector (full 7D phase space per cell:
$\alpha_a, \beta_a, \beta_{12}, \beta_{21}, \theta_R$) is needed for
**D.1 KH** and for matching the asymptotic Drazin–Reid growth rate
(see `notes_M3_phase0_berry_connection.md` §7.6). The
verified off-diagonal kinetic 1-form

$$\theta_{\rm offdiag} \;=\; -\tfrac{1}{2}(\alpha_1^2 \alpha_2\,d\beta_{21} + \alpha_1 \alpha_2^2\,d\beta_{12})$$

is in `src/berry.jl::kinetic_offdiag_2d`, with verified partials in
`scripts/verify_berry_connection_offdiag.py`. **For M3-3 we recommend
pinning $\beta_{12} = \beta_{21} = 0$** and treating them as
realizability-boundary constants:

- The diagonal-sector physics (Tier-C tests) does not exercise
  $\beta_{12}, \beta_{21}$; the kinematic equation (3.1) of the 2D
  doc reduces to vorticity + strain misalignment without the
  intrinsic-rotation term.
- The 5D Casimir kernel of $\Omega$ has a closed-form
  $\mathcal{H}_{\rm rot}$ constraint
  (`notes_M3_phase0_berry_connection.md` §6.6); the 7D Casimir
  tilts away from $\theta_R$ and requires a $\beta$-dependent
  $\mathcal{H}_{\rm rot}^{\rm off}$ that has to be calibrated against
  the linearised KH dispersion (see §7.5 of the off-diag derivation).
  This is research-level work and is **explicitly out of M3-3 scope**.
- D.1 KH is a Tier-D headline for M3-6, and M3-6 will gate the
  off-diagonal sector activation. M3-3 just needs to leave the door
  open: store $\beta_{12} = \beta_{21} = 0$ explicitly in the field
  set so the Newton residual doesn't read garbage, but **do not
  evolve** them in the Newton block.

When M3-6 / D.1 needs the off-diagonal sector, M3-3 should have made
this clean to enable: (a) define a `EnableOffDiagonal` flag in the
solver options; (b) when it's true, route through
`kinetic_offdiag_2d` for the $\beta_{12}, \beta_{21}$ kinetic form
and add the corresponding rows to the Newton residual; (c) extend the
field set with $:\beta_{12}, :\beta_{21}$ scalars; (d) supply the
$\mathcal{H}_{\rm rot}^{\rm off}$ form (calibrated). The structural
hooks are easy if planned for from the start; retrofitting them later
is harder.

## 5. New / modified files

| File | Change in M3-3 |
|---|---|
| `src/types.jl` | NEW: a concrete `DetField{2, T}` 2D variant with the 10-dof layout (or extend `DetFieldND{2, T}` to be the working struct rather than a doc tag). The legacy 1D `DetField{T}` remains as the M1 regression baseline (do **not** delete; M3-2 parity tests still consume it). |
| `src/cholesky_DD.jl` | NEW: per-axis Cholesky decomposition driver. Construct $(\alpha_a, \beta_a, \theta_R)$ from a $2 \times 2$ $L_1$ via SVD-style spectral decomposition; reverse for diagnostics. Also: per-cell Berry coupling driver consuming `src/berry.jl` (with `BerryStencil2D` cache if profitable). Closed-form $\mathcal{H}_{\rm rot}$ constraint enforcement. |
| `src/eom.jl` | EXTENDED: add `det_el_residual_HG_2D!` (the native HG-side 2D residual). The 1D path stays at `det_el_residual_HG_1D!` (M3-0/1/2 cache_mesh shim). New file is the native 2D, with a `det_el_residual_HG!{D}` dispatch entry point that picks 1D or 2D. Once verified, route the 1D path through the native HG-side residual too and **drop the cache_mesh shim** (M3-3 retires it for both 1D and 2D — see §6 verification gates). |
| `src/newton_step_HG.jl` | EXTENDED: full Newton solve on the native HG path. Replace the cache_mesh-via-`det_step!` delegation with native sparsity + AD (ForwardDiff initially, Enzyme later). Forward all M1/M2 kwargs (`bc`, `inflow_state`, `q_kind`, …). Maintain the M3-2 bit-exact-parity test as a regression gate. |
| `src/setups_2d.jl` | EXTENDED: 2D ICs initialize the full 10-dof per cell, including $\theta_R^{(0)}$ (set to the principal-axis-frame angle of the $M_{xx}$ at $t = 0$; for a 1D-symmetric IC this is identically 0). The Tier-C IC factories already in place (`tier_c_sod_ic`, `tier_c_cold_sinusoid_ic`, `tier_c_plane_wave_ic`) need only the new field-set fields plumbed through. |
| `src/diagnostics.jl` | EXTENDED: `gamma_per_axis_2d(field, J, s) -> SVector{2, T}`. Used by AMR + realizability + Tier-C C.2 selectivity test. |
| `src/action_amr_helpers.jl` | EXTENDED: action-error indicator generalized to per-axis γ (see §4.3). |
| `src/stochastic_injection.jl` | EXTENDED: per-axis VG injection (the M2-3 carry-over already supports per-axis decomposition; verify the wiring is intact for the 2D field set). |
| `test/test_M3_3_*.jl` | NEW per sub-phase: 2D zero-strain (analog of M1 Phase-1 stay-quiet test), 2D uniform-strain, 2D acoustic, 2D-symmetric-1D-reduction parity (the **dimension-lift gate** from §6.1 below), Berry CHECK 1–7 reproduced numerically. |
| `reference/notes_M3_3_2d_cholesky_berry.md` | THIS FILE. |

Files explicitly **not** modified in M3-3:

- `src/cholesky_sector.jl` — M1 1D Cholesky kernel; reused per axis.
- `src/berry.jl` — M3-prep ships this; treat as an immutable dep.
- `src/segment.jl`, `src/amr_1d.jl` — M1 1D legacy; retired in M3-2 if
  the cache_mesh shim is dropped, but **kept as the M1 regression
  baseline until M3-3 ships**. M3-3 may delete them in a follow-up
  commit *after* the dimension-lift gate (§6.1) passes.
- `src/eos.jl`, `src/stochastic.jl`, `src/calibration.jl` — pure
  functions; D-generic by virtue of taking scalar inputs.

## 6. Verification gates

### 6.1 Dimension-lift parity gate (CRITICAL)

**The single most important M3-3 acceptance criterion.**

The 2D code in a 1D-symmetric configuration (e.g. y-direction trivial,
$k_y = 0$ for cold sinusoid; $\beta_2 = \theta_R = 0$ initially;
$\alpha_2 = \alpha_2^{(\rm IC)}$ constant) **must reproduce M3-2b's
1D bit-exact results to machine precision** ($|\Delta| \leq 5\!\times\!10^{-13}$).

Operationally:

1. Take an M3-2 1D test (e.g., the Phase-2 zero-strain test, the
   Phase-7 steady-shock test, the Phase-8 stochastic test).
2. Construct the 2D analog by replicating the 1D state along y and
   pinning $\beta_2 = 0$, $\theta_R = 0$, $\alpha_2 = $ const.
3. Run both the M1 1D path and the M3-3 2D path with identical
   timesteps + RNG seeds.
4. Assert per-cell-state agreement at every timestep:
   `(x_1, u_1, α_1, β_1, s)_{2D, y=any column}` $\equiv$
   `(x, u, α, β, s)_{1D}` to 0.0 absolute (or 5e-13 tolerance if
   summation order causes ULP drift).

**If this gate fails, the Berry coupling has a bug** — the 2D code
is supposed to *reduce* to 1D in the dimension-lifted sub-slice
(`notes_M3_phase0_berry_connection.md` §1: "the 1D analog is the
slice $(\alpha_2, \beta_2, \theta_R) = (\text{const}, 0, 0)$").

This is the M3-3 equivalent of M3-0's "M1 Phase 1 parity to 5e-13"
gate. Run it on **every** 1D test M3-2 ships (~40 tests; 10-15 min
total runtime).

### 6.2 Berry verification reproduction

The seven SymPy-verified Berry checks in
`scripts/verify_berry_connection.py` (see
`notes_M3_phase0_berry_connection.md` §6) must reproduce numerically
in the implementation. Sample 5–10 random $(\alpha, \beta, \theta_R)$
points (with realizability constraints satisfied) and verify:

- **CHECK 1 closedness**: $d\omega_{2D} = 0$ structurally — the
  Newton Jacobian at any point is symmetric in the
  $(\alpha_a, \beta_a, \theta_R)$ block w.r.t. cyclic-sum identities.
- **CHECK 2 per-axis match**: setting $\dot\theta_R = 0$ in the
  2D residual and dropping the off-axis terms reproduces M1's per-axis
  residual to 0.0.
- **CHECK 3 iso-pullback**: at $\alpha_1 = \alpha_2$, $\beta_1 = \beta_2$,
  the Berry-block contribution to the residual vanishes (modulo the
  factor-of-2 normalization caveat in §6.4 of the 2D doc — adopt the
  $\tfrac12$-normalization recommended there).
- **CHECK 5 explicit numerical values**: cross-check
  `berry_F_2d(α, β)` at the 5 reference points hard-coded in
  `test_M3_prep_berry_stencil.jl` (already done in M3-prep; M3-3
  inherits the regression).

### 6.3 Iso-pullback ε-expansion check

At $\alpha_1 = \alpha_2 = \alpha_0 + \epsilon\,\delta_\alpha$,
$\beta_1 = \beta_2 = \beta_0 + \epsilon\,\delta_\beta$,
$\theta_R = \epsilon\,\delta_\theta$, the Berry term reduces per the
ε-expansion in `notes_M3_phase0_berry_connection.md` §6 to
$O(\epsilon^2)$ per cell — i.e., the linear-in-ε contribution from
the Berry block vanishes on the iso slice. Verify by computing the
residual at three ε values (e.g., $10^{-2}, 10^{-4}, 10^{-6}$) and
checking the leading-order behavior is $O(\epsilon^2)$, not $O(\epsilon)$.
Tolerance: rel-error ≤ 1e-3 (the ε-extrapolation has limited precision).

### 6.4 $\mathcal{H}_{\rm rot}$ solvability constraint

Sample 5 generic $(\alpha, \beta, \gamma)$ points (no axis degeneracy,
no realizability boundary). Compute the kernel direction $v_{\rm ker}$
of the 5D $\Omega$ symbolically (or extract from the verification
script's output). Verify $dH \cdot v_{\rm ker} = 0$ where $H = H_{\rm Ch} + H_{\rm rot}$
and $H_{\rm rot}$ uses the closed-form expression in
`notes_M3_phase0_berry_connection.md` §6.6:

$$\frac{\partial \mathcal{H}_{\rm rot}}{\partial \theta_R}
= \frac{\gamma_1^2 \alpha_2^3}{3\alpha_1} - \frac{\gamma_2^2 \alpha_1^3}{3\alpha_2} - (\alpha_1^2 - \alpha_2^2)\beta_1\beta_2.$$

Tolerance: rel-error ≤ 1e-10 (this is a closed-form algebraic check).

If this fails, the rotation Hamiltonian is wrong — Newton will not
converge on configurations with non-trivial $\theta_R$ dynamics
(plane wave at oblique angle, axis-aligned strain).

### 6.5 Per-axis γ selectivity

In a 1D-symmetric collapsing-axis configuration (cold sinusoid with
$k_x = $ small, $k_y = 0$, low-pressure IC), verify:

- $\gamma_1 \to 10^{-5}$ (the collapsing axis hits the realizability
  floor),
- $\gamma_2 \to O(1)$ (the trivial axis stays in the bulk),
- The action-error indicator fires only along the collapsing-axis
  direction (per-axis selectivity).

This is the C.2 acceptance criterion (`notes_dfmm_2d_overview.md`
Tier C.2) and is the M3-3 demonstration that per-axis decomposition
is operationally meaningful.

### 6.6 Cache_mesh shim retirement

After §6.1 + §6.2 + §6.3 + §6.4 + §6.5 all pass, **drop the
cache_mesh::Mesh1D shim** from `src/newton_step_HG.jl` and re-run the
full M3-2 test suite (2806 + 1 deferred). All tests must still pass at
the same tolerances. This retires the M3-0/1/2 cache_mesh round-trip
on the 1D path too.

If retiring breaks any test:

- **0.0-absolute tests** (the M3-2 parity tests) failing means the
  native HG-side residual is not bit-exact to M1; debug the per-axis
  reduction.
- **Tolerance-bounded tests** (the action-AMR refine/coarsen mass
  tests) failing means the rebuild path needs care; check that the
  HG path through `refine_cells!` + `coarsen_cells!` produces the
  same cell counts at the same step.

The cache_mesh retirement is the M3-2 "open issue 2" handoff item;
M3-3 is the natural place to close it. Coordinate with M3-2b which
may already have made progress on this in parallel.

## 7. Wall time + risk

### Wall time

Per `MILESTONE_3_PLAN.md` Phase M3-3 estimate: **~2 weeks** (down from
the original 2-3 weeks now that HG sparsity + halo are upstream and
the Berry stencils are pre-shipped). Block breakdown:

| Block | Days |
|---|---|
| 2D field-set + types extension; per-axis Cholesky decomposition driver | 2 |
| Native HG-side EL residual (without Berry) | 2 |
| Berry coupling integration; $\mathcal{H}_{\rm rot}$ constraint | 2 |
| Newton sparsity + AD wiring | 2 |
| Dimension-lift parity gate + Berry check tests | 2 |
| Cache_mesh retirement + full M3-2 regression | 2 |
| **Total** | **~12 days (~2 weeks)** |

### Main risk

**HaloView's polynomial-field access pattern may not match what
dfmm's per-cell residual assumes.** The cache_mesh shim hid this for
M3-0/1/2 because the residual ran on the 1D `Mesh1D{T,DetField{T}}`
struct, which is dfmm-native. The native HG-side residual reads
`PolynomialFieldView`'s coefficient column directly, and the order-0
case is conceptually a scalar but operationally a `Tuple{T}` or
`PolynomialView{1, T}`.

**Defensive coding plan:**

1. Before writing the residual, write a `halo_smoke_test_2D.jl` that
   builds a 4×4 uniform 2D mesh, populates the `PolynomialFieldSet`
   with a known per-cell pattern (e.g., `α_1[i, j] = 10*i + j`), and
   walks all 4 face-neighbors of a corner cell + an interior cell via
   `halo_view`, asserting the returned values match the pattern.
2. Wrap `hv[i, off]` access in a typed helper `halo_value(hv, i, off, ::Val{:α_1})`
   that accesses the named field's scalar value, returning `T` not a
   `Tuple{T}`. This isolates the order-0 pattern; when M3-3 bumps to
   order > 0, the helper changes (returning a coefficient column),
   not the residual itself.
3. Run the smoke test early; **block the residual development** on
   the smoke test passing. This is the M3-0 lesson: HG's 1D coverage
   was tested first, before the cholesky_step_HG! shim was written.

### Secondary risks

- **Newton convergence on the 10N-dof system.** M1's 4-dof Newton
  converged in 2-3 iterations on smooth ICs; the 10-dof Newton with a
  closed-form Berry block should be similar. If you see >20
  iterations, debug the $\theta_R$ initialization (set to the
  principal-axis frame of the IC's $M_{xx}$, not zero).

- **Sparsity-pattern propagation** through the Newton's
  `SparseMatrixCSC{Bool, Int32}`. M1's Newton uses `Float64` sparsity
  via `SparseConnectivityTracer.jacobian_sparsity`. M3-3 should
  cross-check that `cell_adjacency_sparsity ⊗ ones(10, 10)` produces
  the same sparsity pattern as `SparseConnectivityTracer` would
  derive automatically. If they disagree, the manually-constructed
  pattern is missing a coupling — likely the $\theta_R$-to-$\beta_a$
  off-cell coupling from the strain-rotation correction in §2.3.

- **Off-diagonal $\beta_{ab}$ creeping into the residual.** Even
  pinned at zero, the Newton may try to push them off zero if the
  residual has incorrectly-zeroed gradient terms. Add a sanity
  assertion at the end of every Newton step:
  `@assert abs(β_12) ≤ ε && abs(β_21) ≤ ε` for all cells. M3-6 will
  release this check when the off-diagonal sector activates.

- **3D forward-compatibility.** The 3D EL residual generalizes the
  2D one with three pair-rotation angles $\theta_{12}, \theta_{13}, \theta_{23}$
  and pair-Berry stencils (already in `src/berry.jl::berry_partials_3d`).
  Write the 2D residual so the structure obviously generalizes —
  parameterize by `D` where possible, use `SVector{D, T}` for per-axis
  fields. M3-7 should be able to copy the 2D residual into the 3D
  case without rewriting. The M3 plan's wall-time estimate for M3-7
  (3-5 weeks) banks on this.

## 8. Critical references

Read all of these before drafting code. The **bold** entries are the
ones M3-3 will touch directly; the rest are for context.

- **`reference/notes_M3_phase0_berry_connection.md`** — 2D Berry
  derivation + verification. §3 (kinematic equation), §5 (ansatz),
  §6 (verifications), §6.6 ($\mathcal{H}_{\rm rot}$ constraint), §7
  (off-diagonal $L_2$).
- `reference/notes_M3_phase0_berry_connection_3D.md` — SO(3)
  extension. Read for forward-compat awareness; M3-3 does not
  implement it.
- **`reference/notes_M3_prep_berry_stencil.md`** — Berry stencil API
  (the `src/berry.jl` interface).
- `scripts/verify_berry_connection.py` — SymPy 2D verification (run
  it once to see the output; M3-3 reproduces it numerically).
- `scripts/verify_berry_connection_3D.py` — SymPy 3D verification.
- `scripts/verify_berry_connection_offdiag.py` — SymPy off-diag
  verification.
- **`src/berry.jl`** — M3-prep stencil implementation (consumed
  directly).
- `reference/notes_dfmm_2d_overview.md` — eight differences for 2D;
  read §5 (per-principal-axis stochastic injection) and Tier C list.
- `specs/01_methods_paper.tex` §5 — Berry connection prose, revised in
  M3-prep (`reference/notes_M3_prep_paper_section5_revision_applied.md`).
- `specs/01_methods_paper.tex` §6 — Bayesian remap (M3-5 / M3-6
  context; M3-3 does not touch this).
- `specs/01_methods_paper.tex` §9.6 — per-principal-axis stochastic
  injection (D.1 KH falsifier; M3-3 leaves this dormant but the
  field-set wiring should already accommodate it).
- **`~/.julia/dev/HierarchicalGrids/src/Storage/HaloView.jl`** —
  HG halo API. Read in full before drafting.
- **`~/.julia/dev/HierarchicalGrids/src/Mesh/Neighbors.jl`** —
  HG sparsity API. Read in full.
- **`~/.julia/dev/HierarchicalGrids/src/Mesh/BoundaryConditions.jl`** —
  HG BC framework.
- **`~/.julia/dev/HierarchicalGrids/src/Overlap/frame_boundaries.jl`** —
  `FrameBoundaries` accessor.
- `reference/notes_M3_0_hg_integration.md` — M3-0 patterns
  (workarounds + version pins).
- `reference/notes_M3_2_phase7811_m2_port.md` — M3-2 patterns + open
  issues (in particular the cache_mesh retirement note in §"Open
  issues / handoff to M3-3").
- `reference/notes_phaseA0.md` — M1 Phase 1 1D EL residual derivation.
  The 2D residual is the per-axis lift of this with the Berry block
  added.
- `reference/notes_phase2_discretization.md` — M1 Phase 2 multi-cell
  discretization patterns.
- **`src/cholesky_sector.jl`** — M1 per-axis kernel; reused per axis
  in 2D.
- **`src/eom.jl`**, **`src/newton_step_HG.jl`** — M3-0/1/2 1D
  residual + Newton (the entry points M3-3 extends).
- `src/setups_2d.jl` — Tier-C IC factories (already D-generic; M3-3
  extends the field set).
- `src/types.jl` — existing `DetField{T}` (M1 1D) +
  `DetFieldND{D, T}` (M3-0/1 doc-tag); M3-3 promotes the doc tag to
  a working struct for D=2.

## 9. Sub-phase split (suggested for the M3-3 launch agent)

The 2-week wall time is comfortable for one agent. If the M3-3 agent
prefers an explicit sub-phase split (mirroring M1's Phase-N pattern):

- **M3-3a** (3 days). Extend the 2D field set in `src/setups_2d.jl`
  + `src/types.jl`; write the per-axis Cholesky decomposition driver
  in `src/cholesky_DD.jl`; smoke-test halo access (§7 risk
  defensive plan). Dimension-lift gate stays passing (no functional
  change to the 1D residual).

- **M3-3b** (4 days). Native HG-side 2D EL residual in `src/eom.jl`,
  *without* Berry (only per-axis Cholesky + per-axis pressure stencil).
  Verify the dimension-lift gate (§6.1) for a zero-strain 2D test.
  Cache_mesh shim still in place.

- **M3-3c** (3 days). Berry coupling integration: consume `src/berry.jl`
  in the residual; enforce $\mathcal{H}_{\rm rot}$ constraint;
  wire $\theta_R$ as a Newton unknown. Re-verify dimension-lift gate.
  Run §6.2 + §6.3 + §6.4.

- **M3-3d** (2 days). Per-axis γ diagnostic + AMR/realizability
  per-axis wiring. Run §6.5 (per-axis γ selectivity in a
  1D-symmetric cold sinusoid).

- **M3-3e** (2 days). Cache_mesh shim retirement (§6.6) — drop the
  shim from `src/newton_step_HG.jl`; re-run full M3-2 regression
  (2806 + 1 deferred). Only after this passes does M3-3 close.

## 10. Open questions / decisions for the M3-3 launch agent

The M3-3 brief asked whether anything requires human judgement or
additional research. The four open items:

1. **HaloView coefficient access for order-0 fields.** Is `hv[i, off]`
   for an order-0 `MonomialBasis{D, 0}` field returning a `Tuple{T}`,
   a `PolynomialView{1, T}`, or a scalar? The HaloView source comment
   says "Tuple of length n_coeffs(basis) for zero allocation"; M3-3
   should verify with the smoke test (§7 risk plan) before drafting
   the residual. *Action: write the smoke test first.*

2. **`MonomialBasis` for the 2D Cholesky-sector fields.** Stay at
   order 0 for the first cut, or jump to order 3 (Bernstein cubic per
   methods paper §9.2) immediately? Recommend order 0 → match M3-2's
   1D substrate and keep the dimension-lift gate clean. Order 3 is
   M3-4 / M3-5 work. *Action: stay order 0 in M3-3; flag for M3-4.*

3. **$\beta_{12}, \beta_{21}$ pinning enforcement.** Hard-code in the
   field-set initializer (`= 0`) and assert in the Newton residual,
   or omit the fields entirely (so the Newton residual has 10N rows
   not 12N)? Recommend **omit**: simpler, no risk of leakage. M3-6
   will re-add them when D.1 activates. *Action: omit the
   off-diagonal-β fields from the M3-3 field set.*

4. **`mesh.balanced` requirement.** M3-3 should require
   `mesh.balanced == true` so each face has at most one neighbor.
   Coarse-fine face handling (hanging nodes) is a follow-up phase
   (M3-5 is the natural place). *Action: add an assertion in the
   2D solver that `mesh.balanced == true`; document in the M3-3
   status note that hanging-node handling is deferred.*

These four decisions shape the rest of M3-3's design. The recommended
defaults above (smoke-test first, order 0, omit off-diag β, balanced
only) are the conservative path; the M3-3 launch agent may revisit
any of them with cause.

5. **Coordination with M3-2b (in flight in parallel).** M3-2b is
   doing the HG-feature swap-in (Swap 1–8 per the M3-2b brief). Swap
   8 lands the BC framework integration on the 1D path. M3-3 should
   coordinate to use the same BC framework convention so the 2D
   path doesn't diverge from the 1D path. *Action: rebase M3-3
   onto M3-2b's HEAD before starting M3-3c (Berry integration);
   confirm the 1D BC integration pattern and mirror it in 2D.*

---

*End of M3-3 design note. ~700 lines including math + pseudocode +
reference list. Estimated reading time for the M3-3 launch agent:
~30 min, plus another ~60 min reading the §8 references in full.
After that, M3-3a (the field-set extension + halo smoke test) should
take ~half a day to start coding.*
