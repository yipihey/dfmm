# M3-7 — 3D extension: SO(3) Cholesky + Berry, Tier C + D headlines

> **Status (2026-04-26):** *Design note*, no code written. This document
> briefs the agent that will execute Phase M3-7 of
> `reference/MILESTONE_3_PLAN.md`. Read it together with the §10
> references — especially `src/berry.jl` (3D stencils already shipped
> + verified per `notes_M3_prep_3D_berry_verification.md`), the M3-3
> 2D design note (`notes_M3_3_2d_cholesky_berry.md`), and the M3-3a..e
> sub-phase status notes — and you will have everything you need to
> start coding the 3D per-cell EL residual on top of the HG substrate.
>
> **Mirror pattern.** This note follows the structure of the 2D design
> note `notes_M3_3_2d_cholesky_berry.md` exactly. M3-7 is the SO(3)
> generalization of M3-3's SO(2) Berry coupling on a 3D HG mesh; the
> 2D pattern carries over per pair-rotation generator with three
> Euler angles in place of the single 2D rotation angle.

## 1. Goal recap

M3-3 (2026-04-26, sub-phases a/b/c/d/e) shipped the 2D principal-axis
Cholesky + SO(2) Berry coupling on the native HG path: 9-dof per leaf
cell, dimension-lift gate at 0.0 absolute against M1's 1D Phase-1
trajectory, all Berry CHECKs reproduced numerically, per-axis γ
diagnostic + per-axis realizability + AMR all wired through, and
**`cache_mesh::Mesh1D` retired** at the end of M3-3e-5. M3-4 (in
flight) is delivering the periodic-x coordinate wrap + the C.1/C.2/C.3
Tier-C drivers on the 2D path. M3-5 (2026-04-26) closed the Bayesian
L↔E remap on top of HG's `compute_overlap` + `polynomial_remap_*`,
including a 3D-ready API (HG's remap is dimension-generic). M3-6 will
deliver the Tier-D 2D headlines (D.1 KH, D.4 pancake, D.7 dust traps,
D.10 ISM tracers); M3-6 may activate the 2D off-diagonal $\beta_{12},
\beta_{21}$ sector for D.1 KH.

**M3-7 is the 3D-extension phase.** Lift the M3-3 2D Cholesky-sector
EL residual + SO(2) Berry coupling onto a 3D HG mesh, run 3D versions
of the M3-4 Tier-C consistency tests (3D Sod, 3D cold sinusoid, 3D
plane wave), and run 3D versions of the M3-6 Tier-D headlines —
foremost the **3D Zel'dovich pancake collapse** (the cosmological
reference test from methods paper §10.5 D.4), plus 3D KH and 3D dust
traps. The math is *already done* and verified — see
`notes_M3_phase0_berry_connection_3D.md` for the 3D Berry derivation
and `notes_M3_prep_3D_berry_verification.md` for the 8 SymPy CHECKs
all reproduced in Julia (797 asserts, 0.0 / 1e-13 / 1e-9 tolerances
depending on CHECK). M3-7 is "encode the verified 3D stencils + the
new 13-dof EL residual on top of HG".

The HG substrate is ready: `HierarchicalMesh{3}` and
`SimplicialMesh{3, T}` ship in HG @ 81914c2; HG's IntExact backend
(`cc6ed70`+, adopted by M3-5) supports D=4 polynomial moments on 3D
tetrahedra; `HaloView` and `cell_adjacency_sparsity` are
dimension-generic (the 1D path used D=1, the 2D path used D=2 — the
3D path consumes the same APIs at D=3); `BCKind` + `FrameBoundaries`
are per-axis-half so the 3D extension is just a longer specifier
tuple.

The structurally-new pieces in M3-7 are: (a) **three Euler angles**
$\theta_{12}, \theta_{13}, \theta_{23}$ as Newton unknowns (vs M3-3's
single $\theta_R$); (b) the **per-axis Cholesky decomposition for 3×3
SPD** (vs the 2×2 closed form in `src/cholesky_DD.jl`); (c) the
**3-component $\mathcal{H}_{\rm rot}^{ab}$ kernel-orthogonality
constraint** (vs M3-3c's scalar `h_rot_partial_dtheta`); (d) wiring
the existing 3D Berry stencils in `src/berry.jl` (`berry_F_3d`,
`berry_term_3d`, `berry_partials_3d`, `BerryStencil3D`) into the
residual exactly the way M3-3c wired the 2D stencils.

## 2. New EL system structure (3D)

### 2.1 Per-cell unknowns: 1D → 2D → 3D

| D | Unknowns per leaf cell | Newton dof | + post-Newton |
|---|---|---:|---:|
| 1 (M1) | $(x, u, \alpha, \beta, s)$ | 5 | 7 (+ $P_\perp$, $Q$) |
| 2 (M3-3) | $(x_a, u_a, \alpha_a, \beta_a)_{a=1,2}, \theta_R, s$ | 9 | 11 |
| 2 (D.1 off-diag, M3-6 gated) | + $(\beta_{12}, \beta_{21})$ | 11 | 13 |
| **3 (M3-7)** | $(x_a, u_a, \alpha_a, \beta_a)_{a=1,2,3}, (\theta_{12}, \theta_{13}, \theta_{23}), s$ | **13** | **15** |
| 3 (D.1 3D KH off-diag, gated post-M3-7) | + 6 off-diagonal $\beta_{ab}$ | 19 | 21 |

The Cholesky-sector subset that the M3-7 EL residual evaluates
directly (i.e., the unknowns for which Newton has to drive the
residual to zero) is

$$\boxed{\;
(x_a, u_a, \alpha_a, \beta_a)_{a=1,2,3}\;+\;(\theta_{12}, \theta_{13}, \theta_{23})\;+\;s
\quad\text{= 13 dof per leaf cell.}
\;}$$

Counting: $4\times 3$ per-axis position/velocity/Cholesky pairs +
3 Euler angles + 1 entropy = 13. Entropy `s` continues to be frozen
across the Newton step (M1 / M3-3 convention; entropy updates are
operator-split). $P_\perp, Q$ ride along as post-Newton sectors,
exactly as in 1D / 2D.

The discrete EL system grows from 9N (M3-3) to **13N** per leaf cell.
By comparison, M1's 4N (1D) and M3-3's 9N (2D) — the per-cell block
size is monotone in $D$, which is exactly the Hamilton–Pontryagin
structure: Cholesky-sector dof scales as $D \cdot 4 + \binom{D}{2}$
(four per axis + one Euler angle per pair-rotation), plus 1 for
entropy.

If M3-7 enables the off-diagonal $\beta_{ab}$ sector for a 3D D.1 KH
follow-up (six off-diagonal entries — see §6 of
`notes_M3_phase0_berry_connection_3D.md`), unknowns grow to 19N. We
**recommend pinning** all six off-diagonals to zero throughout M3-7
(realizability boundary, analog of the 2D §4.4 default). 3D KH with
off-diagonal sector is naturally separated as M3-9 work, after the
2D D.1 calibration in M3-6 has settled the off-diagonal
$\mathcal{H}_{\rm rot}^{\rm off}$ form.

### 2.2 Where the 3D EL residual terms come from

The discrete Hamilton–Pontryagin Lagrangian per cell, with the M1
sign convention generalized per pair-rotation generator (see
`notes_M3_phase0_berry_connection_3D.md` §3):

$$
L_{\rm cell}^{(3D)} \;=\;
\underbrace{\tfrac12 \sum_{a=1,2,3} \dot x_a^2}_{\rm kinetic\ position}
\;+\;
\underbrace{\sum_{a=1,2,3}\bigl[-\tfrac{\alpha_a^3}{3}\,\mathcal{D}_t^{(1)}\beta_a + \tfrac12 \alpha_a^2 \gamma_a^2\bigr]}_{\rm per\!-\!axis\ Cholesky\ (M1\ form\ per\ axis)}
\;-\;
\underbrace{\tfrac{1}{3}\sum_{a<b}(\alpha_a^3\beta_b - \alpha_b^3\beta_a)\,\dot\theta_{ab}}_{\rm Berry\ kinetic\ (M3\!-\!7\ NEW;\ 3\ pair\ terms)}
\;-\;
\underbrace{\mathcal{H}_{\rm rot}(\alpha_a, \beta_a, \theta_{ab})}_{\rm rotation\ Hamiltonian}
\;-\;
\underbrace{P(J, s)\,\dot J/\rho}_{\rm EOS\ coupling}
$$

with $\gamma_a^2 = M_{vv,a}(J, s) - \beta_a^2$ as in M1 / M3-3 (per
axis now over $a = 1, 2, 3$) and $J$ the per-cell Jacobian. Each term
contributes to the EL residual for a specific subset of unknowns:

| Term in $L$ | Drives residual w.r.t. |
|---|---|
| $\tfrac12 \dot x_a^2$ + EOS | $x_a$ (forces from $\partial P/\partial x$ via per-axis pressure stencil) |
| $-\tfrac{\alpha_a^3}{3}\mathcal{D}_t^{(1)}\beta_a$ | $\alpha_a, \beta_a$ (per-axis Cholesky; M1 boxed equations) |
| $\tfrac12 \alpha_a^2 \gamma_a^2$ | $\alpha_a, \beta_a, s$ (γ-block per axis) |
| $-\tfrac{1}{3}\sum_{a<b}(\alpha_a^3\beta_b - \alpha_b^3\beta_a)\dot\theta_{ab}$ | $\alpha_a, \beta_a, \theta_{ab}$ for all six $(a, a, b)$ index slots — three Berry blocks, closed forms in `src/berry.jl::berry_partials_3d` |
| $\mathcal{H}_{\rm rot}^{ab}(\theta_{ab})$ | $\theta_{ab}$ for $(a, b) \in \{(1,2), (1,3), (2,3)\}$ — three closed-form constraints |
| $P\,\dot J/\rho$ | $x_a, u_a, s$ |

The M1 + M2 1D residual already implements rows 1, 2, 3, 6 (per axis
in the 3D code, summed over $a = 1, 2, 3$). M3-3 implemented row 4
for the single 2D pair $(a, b) = (1, 2)$ + row 5 for the scalar
$\theta_R$. The **only structurally new pieces** in M3-7 are:

1. **The Berry kinetic term (row 4) summed over three pairs.** Each
   pair-Berry block is structurally identical to M3-3's 2D block, so
   the residual integration is "M3-3c times three" with index
   bookkeeping. The closed-form partials are in
   `src/berry.jl::berry_partials_3d` — already verified (CHECK 7 of
   `notes_M3_prep_3D_berry_verification.md`, FD vs closed-form at
   1e-9 and closed-form vs SymPy at 1e-14).
2. **The rotation Hamiltonian term (row 5) per pair-generator.** In
   the 9D principal-axis-diagonal sector, $\mathcal{H}_{\rm rot}^{ab}$
   per pair is fixed by the kernel-orthogonality constraint
   $dH \cdot v_{\rm ker} = 0$ from
   `notes_M3_phase0_berry_connection_3D.md` §6 + CHECK 4 of the
   verification note. **Important: the 9D Casimir kernel direction is
   1-dimensional, not 3-dimensional.** SymPy's `nullspace` gives the
   kernel direction with all nine components nonzero generically;
   $\theta_{23}$ is normalized to 1 in the default output (CHECK 4 of
   `notes_M3_prep_3D_berry_verification.md`). The kernel-orthogonality
   residual must therefore be added to the **$\theta_{23}$ row** (the
   row where the kernel direction has the largest, normalized
   component) — analog of M3-3c's `h_rot_kernel_orthogonality_residual`
   added to the $\theta_R$ row in 2D.

   The closed-form $\partial\mathcal{H}_{\rm rot}/\partial\theta_{ab}$
   per pair is the 3D analog of M3-3's

   $$\frac{\partial \mathcal{H}_{\rm rot}}{\partial \theta_R}\bigg|_{\rm 2D} = \frac{\gamma_1^2 \alpha_2^3}{3\alpha_1} - \frac{\gamma_2^2 \alpha_1^3}{3\alpha_2} - (\alpha_1^2 - \alpha_2^2)\beta_1\beta_2.$$

   Per pair $(a, b)$ in 3D, the analogous formula is (deferred
   derivation; see §6 of the 3D Berry note):

   $$\frac{\partial \mathcal{H}_{\rm rot}}{\partial \theta_{ab}}\bigg|_{\rm 3D, (a,b)} = \frac{\gamma_a^2 \alpha_b^3}{3\alpha_a} - \frac{\gamma_b^2 \alpha_a^3}{3\alpha_b} - (\alpha_a^2 - \alpha_b^2)\beta_a\beta_b.$$

   Each pair's closed form is *structurally* the 2D form on the
   $(\alpha_a, \alpha_b, \beta_a, \beta_b, \gamma_a, \gamma_b)$ pair
   restriction. The 2D form falls out at $(a, b) = (1, 2)$ on the
   $\alpha_3 = $ const, $\beta_3 = 0$ slice (CHECK 3b of the
   verification note: 2D reduction reproduced exactly).

   See §7.4 below for the verification gate that exercises this
   formula at 5–10 sample points.

### 2.3 Discrete-time mid-point stencil (3D)

The M1 mid-point rule for $\mathcal{D}_t^{(q)}$ generalizes per
principal axis $a = 1, 2, 3$. Each pair-rotation $\theta_{ab}$
contributes its own mid-point stencil:

$$
\mathcal{D}_t^{(\rm rot,ab)}\theta_{ab} \;\approx\; \frac{\theta_{ab}^{n+1} - \theta_{ab}^n}{\Delta t}
\;+\; (\text{lab-frame strain rotation correction per pair}).
$$

The strain-rotation correction is the kinematic-equation contribution
from the $(a, b)$ off-diagonal of the $3\times 3$ kinematic-equation
matrix (`notes_M3_phase0_berry_connection_3D.md` §3): per pair,

$$\dot\theta_{ab}^{\rm drive} = \tilde W_{ab} - \tilde S_{ab}\,\frac{\alpha_a^2 + \alpha_b^2}{\alpha_a^2 - \alpha_b^2}.$$

Each pair has its own $1/(\alpha_a^2 - \alpha_b^2)$ singularity at
the corresponding axis-degeneracy boundary. M3-7 should reuse the M1
mid-point convention for the per-axis sectors and treat each
$\theta_{ab}$ mid-point exactly the same way (averaging at half-step,
finite-difference of the angle increment). The resulting discrete EL
residual is symmetric under $n \leftrightarrow n+1$ swap modulo the
M1 sign convention, exactly like the 2D form.

The pseudo-code skeleton (one cell, per Newton iteration; mirrors
M3-3c's pattern):

```julia
function det_el_residual_cell_3D(Ω_n, Ω_{n+1}, halo_view, cell_idx, params):
    # 1. Pull halos: Ω_{n+1} = (x_a, u_a, α_a, β_a, θ_{ab}, s) for a=1,2,3 + 3 pairs
    self    = halo_view[cell_idx, (0, 0, 0)]
    east    = halo_view[cell_idx, (+1, 0, 0)]   # may be `nothing` at boundary
    west    = halo_view[cell_idx, (-1, 0, 0)]
    north   = halo_view[cell_idx, (0, +1, 0)]
    south   = halo_view[cell_idx, (0, -1, 0)]
    top     = halo_view[cell_idx, (0, 0, +1)]
    bottom  = halo_view[cell_idx, (0, 0, -1)]

    # 2. Mid-point quantities per axis a = 1, 2, 3
    α_mid_a = (self.α_a + halo_n.α_a) / 2
    β_mid_a = (self.β_a + halo_n.β_a) / 2
    γ_mid_a² = M_vv_a(J_mid, s_mid) - β_mid_a²

    # 3. Per-axis Cholesky-sector residual (M1 form per axis; reuse cholesky_sector.jl)
    R_α[a] = M1_residual_α(α_mid_a, β_mid_a, γ_mid_a, dt)  # for a = 1, 2, 3
    R_β[a] = M1_residual_β(α_mid_a, β_mid_a, γ_mid_a, dt)

    # 4. Per-axis position/velocity residual (per-axis pressure stencil)
    R_x[a] = (Ω_{n+1}.x_a - Ω_n.x_a) - dt * u_mid_a
    R_u[a] = (Ω_{n+1}.u_a - Ω_n.u_a) - dt * (-∂_a P / ρ + body forces)

    # 5. Berry-connection contribution (M3-7 NEW — three pair blocks)
    α_mid = SVector(α_mid_1, α_mid_2, α_mid_3)
    β_mid = SVector(β_mid_1, β_mid_2, β_mid_3)
    dθ    = SMatrix{3,3}(...)   # antisymmetric: dθ[a,b] = θ_ab_np1 - θ_ab_n
    dB    = berry_partials_3d(α_mid, β_mid, dθ)
    # dB packs: 3 entries for ∂Θ/∂α_a, 3 for ∂Θ/∂β_a, 3 for ∂Θ/∂θ_{ab}
    R_α[a] += dB.dα[a]                         # for a = 1, 2, 3
    R_β[a] += dB.dβ[a]                         # for a = 1, 2, 3
    R_θ[ab] = dB.dθ[ab]                        # for (a, b) ∈ {(1,2), (1,3), (2,3)}

    # 6. Rotation-Hamiltonian solvability constraint (forces the θ_{23} row to close)
    R_θ[23] += partial_H_rot_partial_θ_3d(α, β, γ; pair=:23)
    # Note: M3-3c added the closed-form ∂H_rot/∂θ_R to ALL θ rows in 2D
    #       (single θ_R row); in 3D, kernel-orthogonality requires it on
    #       the θ_{23} row only — the row where v_ker has its largest
    #       component (CHECK 4 of the 3D verification note).

    # 7. Entropy + coupling
    R_s = M1_entropy_residual(...)

    return SVector{13}(R_x[1], R_x[2], R_x[3], R_u[1], R_u[2], R_u[3],
                       R_α[1], R_α[2], R_α[3], R_β[1], R_β[2], R_β[3],
                       R_θ[12], R_θ[13], R_θ[23], R_s)
end
```

The 13-vector residual feeds the Newton solve; sparsity comes from
HG's `cell_adjacency_sparsity(mesh; depth=1)` Kron-producted with a
dense $13 \times 13$ within-cell block (see §3.3 below).

## 3. Native HG-side EL residual (3D)

M3-3 retired the `cache_mesh::Mesh1D` shim end-to-end (M3-3e-5). The
M3-7 3D residual inherits a fully native HG-side path: there is no
1D / 2D shim to traverse, and the dimension-generic structure of
`PolynomialFieldSet` + `HaloView` + `cell_adjacency_sparsity` carries
straight over. **M3-7 writes the 3D residual against
`HierarchicalMesh{3}` directly.**

### 3.1 Per-cell unknowns via `PolynomialFieldSet` (3D)

`src/setups_2d.jl::allocate_cholesky_2d_fields` allocates a
12-named-field `PolynomialFieldSet` over 2D leaves. M3-7 extends this
to 3D:

```julia
# 3D: 13 Newton unknowns + 2 post-Newton (P_perp, Q) per cell = 15 named scalar fields
fields_3D_cholesky = allocate_cholesky_3d_fields(
    mesh::HierarchicalMesh{3},
    SoA(MonomialBasis{3, 0}(),
        :x_1, :x_2, :x_3,            # per-axis position
        :u_1, :u_2, :u_3,            # per-axis velocity
        :α_1, :α_2, :α_3,            # per-axis Cholesky diagonals
        :β_1, :β_2, :β_3,            # per-axis Cholesky 21 diagonals
        :θ_12, :θ_13, :θ_23,         # three Euler angles (Newton unknowns)
        :s,                          # entropy
        :Pp, :Q                      # post-Newton sectors
    )
)
```

Order-0 ("piecewise-constant") storage suffices for the first cut,
mirroring M3-3's 2D path. Higher-order Bernstein expansion (per
methods paper §9.2) is M3-8 / M3-9 work — keep order 0 in M3-7 to
match the dimension-lift gate (§7.1 below).

The total field count grows from 2D's 12 (10 Newton + 2 post-Newton)
to 15 (13 + 2). HG's `PolynomialFieldSet` is dimension-generic; the
3D version differs from 2D only in the `MonomialBasis{3, 0}()` basis
and the addition of three named scalars (`x_3`, `u_3`, `α_3`,
`β_3`, `θ_13`, `θ_23` minus the 2D-only `θ_R`).

**Naming convention.** M3-3 used `:θ_R` for the single 2D rotation
angle. In 3D we use `:θ_12, :θ_13, :θ_23` to match the SO(3)
pair-generator notation in `notes_M3_phase0_berry_connection_3D.md`.
The 2D field `:θ_R` is **not** renamed in the 2D field set; the 2D
and 3D field sets coexist (they target different mesh dimensions).
The 3D `:θ_12` corresponds to the 2D `:θ_R` in the 3D-to-2D
restriction (§7.1 dimension-lift gate verifies this concretely).

### 3.2 Neighbor access via `HaloView` (3D)

**Read first:** `~/.julia/dev/HierarchicalGrids/src/Storage/HaloView.jl`.

`HaloView` is dimension-generic. The 3D version supports 6 face
neighbors per interior cell, indexed by `off::NTuple{3, Int}`:

```julia
hv = halo_view(field, mesh::HierarchicalMesh{3}, depth=1)
hv[i, (0, 0, 0)]      # = field[cell i] (self)
hv[i, (+1, 0, 0)]     # = field[east neighbor]  (axis 1, hi)
hv[i, (-1, 0, 0)]     # = field[west neighbor]  (axis 1, lo)
hv[i, (0, +1, 0)]     # = field[north neighbor] (axis 2, hi)
hv[i, (0, -1, 0)]     # = field[south neighbor] (axis 2, lo)
hv[i, (0, 0, +1)]     # = field[top neighbor]   (axis 3, hi)
hv[i, (0, 0, -1)]     # = field[bottom neighbor](axis 3, lo)
```

For `depth = 1`, `|off[1]| + |off[2]| + |off[3]| ≤ 1` is enforced
(i.e., only direct face neighbors, no diagonal / corner / edge cells
at depth 1).

**M3-3a discovery (carries to 3D):** the field-view returned by
`hv[i, off]` is a `PolynomialView`, not a `Tuple`. Index `pv[1]` for
the scalar order-0 cell-average coefficient. The smoke test
(`test/test_M3_3a_halo_smoke.jl`) pins this contract on a 2D mesh;
M3-7a should write a 3D analog
(`test/test_M3_7a_halo_smoke.jl`) before drafting the residual to
confirm the same `PolynomialView` contract holds at D=3.

**HaloView depth in 3D — open question.** dfmm has used `depth = 1`
throughout 1D / 2D. For a face-only stencil at order 0 in 3D,
`depth = 1` is sufficient. **Higher-order stencils or richer
neighbor patterns (e.g., diagonal-fluxes for the 3D Zel'dovich
pancake) might want `depth = 2`** to access edge / corner neighbors.
HG supports `depth = 2`; it has been tested at D=2 in HG's own test
suite, but not exercised in dfmm scope. M3-7a's smoke test should
also walk a `depth = 2` 3D HaloView on a small mesh to confirm the
pattern matches what HG's docstring claims. **Recommendation:** stay
at `depth = 1` in M3-7a/b/c (matches 2D path); revisit only if the
Tier-D drivers in M3-7e need wider stencils.

### 3.3 Newton sparsity via `cell_adjacency_sparsity` (3D)

```julia
sparsity = cell_adjacency_sparsity(mesh::HierarchicalMesh{3}; depth=1, leaves_only=true)
# returns SparseMatrixCSC{Bool, Int32}: M[i,j] = true iff cells i, j are within 1 face-hop in 3D
```

Newton-Jacobian sparsity for the 13N-dof system is the **Kronecker
product** of `sparsity` (cell-cell) with a dense $13 \times 13$ block
(within-cell). The M3-3c 2D Newton solve uses `sparsity ⊗ ones(9, 9)`;
M3-7 extends to `sparsity ⊗ ones(13, 13)`.

The structural pattern is what `SparseConnectivityTracer` +
`SparseMatrixColorings` consume. M3-3c verified that the
manually-constructed Kron sparsity matches what
`SparseConnectivityTracer.jacobian_sparsity` derives automatically
on the 2D residual. M3-7c should re-verify this at D=3.

**Convergence-rate expectation.** M3-3c's 9-dof-per-cell 2D Newton
converges in 2-3 iterations on smooth ICs (zero-strain, plane wave)
and 5-8 iterations on shocked ICs (Sod). M3-7's 13-dof-per-cell 3D
Newton with three Berry blocks should converge in **2-5 iterations
on smooth ICs** and **5-10 iterations on Tier-D ICs** (3D Zel'dovich
pancake, 3D KH). Note the §7.4 gate is **≤ 7 iterations** (not 5)
because the 3D system is 44% larger than the 2D system per cell. If
you see >20 iterations on a smooth IC, debug:

- (a) the $\theta_{ab}$ initialization matches the principal-axis
  frame of $M_{xx}$ at $t = 0$ for all three pairs (set the three
  Euler angles by aligning the initial $L_1$ to its eigenframe);
- (b) all three $\mathcal{H}_{\rm rot}^{ab}$ closed-form constraints
  are enforced (the kernel-orthogonality residual on the $\theta_{23}$
  row);
- (c) all six off-diagonal $\beta_{ab}$ are pinned to zero — no
  off-diagonal leakage from the residual rows.

### 3.4 Boundary conditions via `BCKind` + `FrameBoundaries{3}`

**Read first:** `~/.julia/dev/HierarchicalGrids/src/Mesh/BoundaryConditions.jl`
and `~/.julia/dev/HierarchicalGrids/src/Overlap/frame_boundaries.jl`.

The HG BC framework is per-axis-half; in 3D the spec is six entries
(2 sides × 3 axes):

```julia
# 3D: x-periodic + y-periodic + z-reflecting
spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC), (REFLECTING, REFLECTING))
fb   = FrameBoundaries(spec)
is_periodic_axis(fb, 1)   # true
is_periodic_axis(fb, 2)   # true
is_periodic_axis(fb, 3)   # false
bc(fb, axis, side)        # → BCKind for each (axis, side) pair
```

`face_neighbors_with_bcs(mesh, i, fb)` is the BC-aware variant; for
each periodic axis it post-processes the boundary entries to wrap to
the opposite-wall leaf with overlapping other-axes extent (the 3D
generalization of the M3-4 periodic-x wrap pattern). Non-periodic
kinds (`INFLOW`, `OUTFLOW`, `REFLECTING`, `DIRICHLET`) are advisory
at the HG layer — dfmm's EL-residual code consumes them to assemble
the appropriate ghost state when `hv[i, off]` returns `nothing`.

M3-7 should:

1. Construct a `FrameBoundaries{3}` per problem (Tier-C / Tier-D ICs
   declare it explicitly). The 2D pattern from M3-4 carries straight
   over; the 3D-only addition is the third axis.
2. **Generalize the M3-4 periodic-coordinate-wrap helper from 2D to
   3D.** `build_periodic_wrap_tables` in `src/eom.jl` currently
   handles per-axis wrap on 2D; M3-7 extends it to handle per-axis
   wrap on 3D (the rule is identical per axis; just compute the
   third-axis offset alongside the other two).
3. Pin the 3D Tier-C drivers with appropriate BCs:
   - **3D Sod:** doubly-periodic in y + z, inflow-outflow in x
     (analog of M3-4's C.1 1D pattern).
   - **3D cold sinusoid:** triply-periodic.
   - **3D plane wave:** triply-periodic (with the wave vector along
     a specified axis combination).

### 3.5 Data-flow pipeline (3D)

```
                  ┌─────────────────────────────────────────────────────┐
                  │     HierarchicalMesh{3} + FrameBoundaries{3}        │
                  └───────────────┬─────────────────────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────────┐
        │                                                       │
        ▼                                                       ▼
PolynomialFieldSet (per-cell                       cell_adjacency_sparsity(mesh; depth=1)
  α_a, β_a, θ_{ab}, x_a, u_a, s, …)                         │
        │                                                       │
        ▼                                                       ▼
halo_view(field, mesh::HierarchicalMesh{3}, depth=1)   Sparse Newton-Jacobian skeleton
        │                                              (Kron with 13×13 dense block)
        ▼                                                       │
det_el_residual_cell_3D!                                        │
(ghost-cell aware via fb;                                       │
 periodic-wrap per axis in 3D)                                  │
        │                                                       │
        ├─ AD (ForwardDiff or Enzyme over halo'd cell tuple) ◀──┘
        │
        ▼
sparse_jacobian (fed to NonlinearSolve.jl Newton)
        │
        ▼
(x_a, u_a, α_a, β_a, θ_{ab}, s)_{n+1} per cell
        │
        ▼
post-Newton operator splits (M1 + M2 + M3-3 carry-over):
    Pp relaxation (per axis), Q decay (per axis),
    stochastic injection (per axis), realizability projection (per axis)
```

The pipeline mirrors M3-3's 2D Newton-then-post-Newton structure
exactly. The sole structural change is the third axis everywhere
(per-axis decomposition over $a = 1, 2, 3$ instead of $a = 1, 2$)
and three Euler angles instead of one.

## 4. Berry coupling integration (3D)

### 4.1 Reuse `src/berry.jl` directly

M3-prep already shipped 3D Berry stencils in `src/berry.jl` (commit
`99af5c4`). M3-7 consumes the module directly with **no source-code
changes**:

```julia
using StaticArrays
using dfmm: berry_F_3d, berry_term_3d, berry_partials_3d, BerryStencil3D

α_mid = SVector(α_mid_1, α_mid_2, α_mid_3)
β_mid = SVector(β_mid_1, β_mid_2, β_mid_3)
# dθ packed as antisymmetric SMatrix{3, 3}: dθ[a, b] = θ_ab_np1 - θ_ab_n for a < b
dθ    = @SMatrix [0           dθ_12       dθ_13;
                  -dθ_12      0           dθ_23;
                  -dθ_13     -dθ_23       0]

# berry_partials_3d returns: per-axis (∂α_a, ∂β_a) + per-pair (∂θ_{ab})
dB    = berry_partials_3d(α_mid, β_mid, dθ)

# Add to the residual rows for (α_a, β_a, θ_{ab})
R[idx_α[a]] += dB.dα[a]    # for a = 1, 2, 3
R[idx_β[a]] += dB.dβ[a]    # for a = 1, 2, 3
R[idx_θ[ab]] += dB.dθ[ab]  # for (a,b) ∈ {(1,2), (1,3), (2,3)}
```

The closed-form partials are verified in
`test/test_M3_prep_3D_berry_verification.jl` (797 asserts, including
finite-difference cross-check at `h = 1e-6`, rel-error ≤ 1e-9).
Hot-path allocation count is 0 with `@allocated`-warmed `SVector`
inputs (mirrors M3-3c).

If the residual evaluation is JIT'd through ForwardDiff or Enzyme,
`berry_partials_3d` can be replaced by raw `berry_F_3d` and let AD
handle differentiation — both produce identical numbers up to FP
order. M3-3c chose the closed-form path because it's allocation-free
and faster; M3-7c should do the same.

### 4.2 BerryStencil3D pre-compute (optional cache)

`BerryStencil3D(α, β)` pre-computes the 3D Berry function (an
`SVector{3, T}` with one entry per pair) and its α/β gradients at a
given $(\alpha, \beta)$. The per-step `dθ` factor multiplies at
use-time:

```julia
stencil = BerryStencil3D(α_mid, β_mid)
contribution = apply(stencil, dθ)              # F[ab] · dθ[ab] summed over pairs
∂α  = stencil.dF_dα * dθ                       # SVector{3, T}
∂β  = stencil.dF_dβ * dθ                       # SVector{3, T}
∂θ  = stencil.F                                # SVector{3, T} — F per pair
```

For an order-0 cell-average residual the savings are marginal; for
an order-3 Bernstein per-cell cubic reconstruction the stencil
should be pre-computed once per quadrature point. M3-7 starts at
order 0 and need not optimize this; flag for a future performance
pass (M3-8).

### 4.3 Per-axis γ diagnostic (3D)

The per-axis γ diagnostic generalizes from 2D's `SVector{2, T}` to
3D's `SVector{3, T}`:

$$\gamma_a^2 \;=\; M_{vv,a}(J, s) \;-\; \beta_a^2,
\qquad \gamma_a \;=\; \sqrt{\max(\gamma_a^2, 0)}, \qquad a = 1, 2, 3.$$

This is the per-axis analog of M1's `gamma_from_Mvv(β, M_vv)`
generalized over three axes. M3-7d adds
`gamma_per_axis_3d(β::SVector{3}, M_vv_diag::SVector{3}) -> SVector{3, T}`
to `src/cholesky_DD.jl` (alongside the existing 2D form).

Per-axis γ plugs into:

- **AMR indicator selectivity** (per-axis rather than scalar). The
  action-error indicator `action_error_indicator_3d_per_axis`
  generalizes M3-3d's 2D form. The C.2 3D selectivity test should
  show: with `k = (1, 1, 0)` the third axis stays trivial (γ_3
  uniform), the first two axes show correlated collapse along their
  respective directions.
- **Hessian-degeneracy regularization** (M1 Phase 3 carry-over). The
  per-axis γ-floor protects the per-axis Newton solve when one or
  two axes collapse while another doesn't. The 3D Zel'dovich pancake
  is the canonical case: collapse on a single axis with γ_1 → small,
  γ_2, γ_3 → O(1) — and the realizability projection should regularize
  axis 1 only.
- **Per-axis stochastic injection** (M2-3 / Phase 8 carry-over). The
  realizability-checked injection acts on the compressive axis (or
  axes) only.

Per-axis γ is a derived quantity (not a Newton unknown). Compute it
in `src/diagnostics.jl` as `gamma_per_axis_3d_field(fields, leaves; …)`
returning a `3 × N` matrix (mirrors M3-3d's
`gamma_per_axis_2d_field`).

### 4.4 Off-diagonal $\beta_{ab}$ in 3D — pin to zero in M3-7

The 3D off-diagonal $\tilde L_2$ sector has six entries: three
antisymmetric combinations (angular-momentum-like) and three
symmetric (intrinsic-rotation-like). The 2D off-diagonal derivation
in `notes_M3_phase0_berry_connection.md` §7 generalizes (per
`notes_M3_phase0_berry_connection_3D.md` §6):

$$\theta_{\rm offdiag}^{(3D)} \;=\; -\tfrac{1}{2}\sum_{a<b}(\alpha_a^2 \alpha_b\,d\beta_{ba} + \alpha_a \alpha_b^2\,d\beta_{ab})$$

per pair, by direct extension. **For M3-7 we recommend pinning
$\beta_{ab} = \beta_{ba} = 0$ for all $(a, b)$** and treating them
as realizability-boundary constants. Reasons (mirror M3-3 §4.4):

- The diagonal-sector physics (3D Tier-C tests + 3D Zel'dovich
  pancake + 3D dust traps) does not exercise off-diagonal $\beta$.
- The 9D Casimir kernel of $\Omega$ has a closed-form
  $\mathcal{H}_{\rm rot}^{ab}$ constraint per pair (§2.2 above); the
  19D Casimir tilts away from $\theta_{ab}$ and requires a
  $\beta$-dependent $\mathcal{H}_{\rm rot}^{\rm off}$ that has to be
  calibrated against the linearised KH dispersion (3D-KH-specific
  research-level work).
- 3D D.1 KH is post-M3-7 work (M3-9 the natural place); M3-7 just
  needs to leave the door open: store all six off-diagonals
  explicitly as zero in the field set so the Newton residual doesn't
  read garbage, but **do not evolve** them in the Newton block.

When 3D D.1 needs the off-diagonal sector, M3-7 should have made this
clean to enable: (a) an `EnableOffDiagonal` flag in the solver
options; (b) when true, route through an analog of M3-3's
`kinetic_offdiag_2d` (a `kinetic_offdiag_3d` that M3-7 does **not**
implement now); (c) extend the field set with 6 off-diagonal scalars;
(d) supply the 3D off-diagonal $\mathcal{H}_{\rm rot}^{\rm off}$ form
(calibrated against 3D KH dispersion). Same advice as M3-3 §4.4:
plan the structural hooks now, retrofit later if the 3D KH calibration
isn't ready.

## 5. Per-axis γ in 3D (test setup)

For the 3D γ-selectivity gate (§7.5 below), three setups span the
relevant symmetry classes:

| Setup | Wave vector | Active axes | Trivial axes | Expected γ pattern |
|---|---|---|---|---|
| 1D-symmetric in 3D | $k = (1, 0, 0)$ | axis 1 | axes 2, 3 (γ_2 = γ_3 uniform) | γ_1 collapses; γ_2, γ_3 ≡ const |
| 2D-symmetric in 3D | $k = (1, 1, 0)$ | axes 1, 2 (in plane) | axis 3 (γ_3 uniform) | γ_1, γ_2 collapse correlated; γ_3 ≡ const |
| Full 3D | $k = (1, 1, 1)$ | all three axes | none | all three γ's non-trivial |

The 1D-symmetric and 2D-symmetric setups are the **dimension-lift
gates** (§7.1 below) — they should reduce exactly to M3-2 1D and
M3-3 2D respectively. The full-3D setup is the genuinely new
configuration: we expect Newton convergence to be slower (5-7 iter on
smooth IC vs 2-3 in symmetric reductions) and the per-axis γ
diagnostic to show **all three** axes participating.

The 3D Zel'dovich pancake (§7.7 below) is a degenerate-1D case in
the per-axis γ sense: the pancake collapses on a single axis (say
axis 1, the compressive direction), so γ_1 → 0 while γ_2, γ_3 stay
at O(1). This is the per-axis γ identifying the pancake-collapse
direction without input.

## 6. New / modified files

| File | Change in M3-7 |
|---|---|
| `src/types.jl` | NEW: `DetField3D{T}` working struct carrying the 13 Newton unknowns + 2 post-Newton sectors. Mirror M3-3a's `DetField2D{T}`. The legacy 1D `DetField{T}` and 2D `DetField2D{T}` remain (regression baselines). |
| `src/cholesky_DD.jl` | EXTEND: `cholesky_decompose_3d`, `cholesky_recompose_3d` — 3×3 SPD ↔ (α_a, θ_{ab}) round-trip via 3×3 eigendecomposition (use `LinearAlgebra.eigen` or the closed-form 3×3 cubic; pick whichever maintains allocation-free with `SMatrix{3,3}`). Plus `gamma_per_axis_3d(β::SVector{3}, M_vv_diag::SVector{3})`, `h_rot_partial_dtheta_3d(α, β, γ; pair)` (per pair), `h_rot_kernel_orthogonality_residual_3d(α, β, γ)` (numerical probe over 9D). ~150-200 LOC. |
| `src/eom.jl` | EXTEND: `cholesky_el_residual_3D!` (no Berry first), then `cholesky_el_residual_3D_berry!` (with the three Berry blocks + the kernel-orthogonality residual on the θ_{23} row). Plus pack/unpack helpers (`pack_state_3d_berry`, `unpack_state_3d_berry!`). The 3D analog of `build_face_neighbor_tables` for 6 face neighbors. The 3D analog of `build_periodic_wrap_tables` (M3-4). ~400-500 LOC. |
| `src/newton_step_HG.jl` | EXTEND: `det_step_3d_HG!` Newton driver (no Berry), then `det_step_3d_berry_HG!` (with Berry). Mirror `det_step_2d_berry_HG!` from M3-3c. Sparsity is `cell_adjacency_sparsity ⊗ ones(13, 13)`. ~150 LOC. |
| `src/setups_2d.jl` → rename to `src/setups_multid.jl` (or add `src/setups_3d.jl` alongside) | EXTEND or NEW: `allocate_cholesky_3d_fields`, `read_detfield_3d`, `write_detfield_3d!`. The Tier-C IC factories should be made dimension-generic where possible; the 3D-specific factories (`tier_c_3d_sod_ic`, `tier_c_3d_cold_sinusoid_ic`, `tier_c_3d_plane_wave_ic`) join the existing 2D ones. |
| `src/diagnostics.jl` | EXTEND: `gamma_per_axis_3d_field(fields, leaves; …)` returning a `3 × N` matrix. |
| `src/action_amr_helpers.jl` | EXTEND: per-axis action-AMR for 3D: `action_error_indicator_3d_per_axis`, `register_field_set_on_refine!` for 3D field set, `step_with_amr_3d!` driver. Mirror M3-3d's 2D version. The 3D analog of M3-3d's listener pattern. |
| `src/stochastic_injection.jl` | EXTEND: `realizability_project_3d!` per-axis VG injection on the 3D field set. The M2-3 carry-over with three axes. |
| `src/remap.jl` | EXTEND: 3D Bayesian remap. M3-5's `BayesianRemapState{D, T}` is dimension-generic; HG's `polynomial_remap_l_to_e!` and `compute_overlap` already support D=3. M3-7 just threads through the 3D path and adds tests for 3D mass/momentum/energy round-trip. |
| `test/test_M3_7a_*.jl` | NEW: halo smoke (3D), field-set 3D, per-axis Cholesky 3D round-trip + iso pullback 3D + per-axis γ 3D. |
| `test/test_M3_7b_*.jl` | NEW: 3D EL residual without Berry; 3D zero-strain regression; **dimension-lift gates 3D ⊂ 1D and 3D ⊂ 2D**. |
| `test/test_M3_7c_*.jl` | NEW: 3D Berry coupling; CHECK 1–8 numerical reproduction in residual (delegate to existing `test/test_M3_prep_3D_berry_verification.jl` for the stencil-level + add residual-level checks); iso-pullback ε-expansion 3D; 3D H_rot solvability. |
| `test/test_M3_7d_*.jl` | NEW: per-axis γ 3D AMR + realizability + selectivity. |
| `test/test_M3_7e_*.jl` | NEW per Tier-C/D 3D test (3D Sod, 3D cold sinusoid, 3D plane wave, 3D Zel'dovich pancake, 3D KH, 3D dust traps). |
| `experiments/M3_7_*.jl` | NEW per Tier-C/D 3D driver (analog of `experiments/M3_3d_*.jl` etc.). |
| `reference/notes_M3_7_*.md` | NEW per sub-phase status (M3-7a/b/c/d/e). |
| `reference/notes_M3_7_3d_extension.md` | THIS FILE. |

Files explicitly **not** modified in M3-7:

- `src/cholesky_sector.jl` — M1 1D Cholesky kernel; reused per axis.
- `src/berry.jl` — M3-prep ships the 3D stencils; treat as immutable.
- `src/eos.jl`, `src/stochastic.jl`, `src/calibration.jl` — pure
  functions; D-generic by virtue of taking scalar inputs.
- `src/setups.jl` — M1 1D / 2D Tier-A setups; M3-7 adds 3D setups in
  `setups_multid.jl` (or `setups_3d.jl`).

## 7. Verification gates

### 7.1 Dimension-lift gate (CRITICAL)

**The single most important M3-7 acceptance criterion.**

Two sub-gates: 3D ⊂ 1D and 3D ⊂ 2D.

#### 7.1a 3D 1D-symmetric ⊂ 1D to ≤ 1e-12

The 3D code in a 1D-symmetric configuration ($k_y = k_z = 0$, axes
2 and 3 trivial: $\beta_2 = \beta_3 = 0$, $\theta_{12} = \theta_{13} = \theta_{23} = 0$,
$\alpha_2 = \alpha_3 = $ const) **must reproduce M3-2's 1D bit-exact
results to ≤ 1e-12** absolute (preferably to 0.0 absolute as in M3-3).

This is structurally guaranteed: on the 1D-symmetric slice, all three
Berry $F_{ab}$'s vanish identically (because $\beta_b = 0$ for $b \neq 1$
and $\alpha_2 = \alpha_3$ kills the $(2, 3)$-pair via CHECK 6 of the
3D verification note).

#### 7.1b 3D 2D-symmetric ⊂ 2D to ≤ 1e-12

The 3D code in a 2D-symmetric configuration ($k_z = 0$, axis 3
trivial: $\beta_3 = 0$, $\theta_{13} = \theta_{23} = 0$, $\alpha_3 = $
const) **must reproduce M3-3c's 2D bit-exact results to ≤ 1e-12**
(preferably to 0.0). This is the sharper test — it verifies the
SO(3) Berry reduces to 2D's SO(2) correctly, which is **CHECK 3b** of
`notes_M3_prep_3D_berry_verification.md` (the 5×5 sub-block on
$(\alpha_1, \beta_1, \alpha_2, \beta_2, \theta_{12})$ matches the 2D
$\Omega$).

Operationally, mirror M3-3c's gate:

1. Take an M3-3c 2D test (e.g., the M3-3c dimension-lift gate, the
   M3-3d selectivity test, the M3-4 periodic-wrap test).
2. Construct the 3D analog by replicating the 2D state along z and
   pinning $\beta_3 = 0$, $\theta_{13} = \theta_{23} = 0$, $\alpha_3 = $
   const.
3. Run both the M3-3c 2D path and the M3-7 3D path with identical
   timesteps + RNG seeds.
4. Assert per-cell-state agreement at every timestep for the
   non-trivial axes:
   $(x_1, x_2, u_1, u_2, \alpha_1, \alpha_2, \beta_1, \beta_2, \theta_{12}, s)_{\rm 3D, z\text{-any column}} \equiv (x_1, x_2, u_1, u_2, \alpha_1, \alpha_2, \beta_1, \beta_2, \theta_R, s)_{\rm 2D}$
   to 0.0 absolute (or 1e-12 if summation order causes ULP drift).

**If 7.1b fails, the SO(3) Berry coupling has a bug** — the 3D code
is supposed to *reduce* to 2D in the dimension-lifted sub-slice
(`notes_M3_phase0_berry_connection_3D.md` §1: "the 2D analog is the
slice $(\alpha_3, \beta_3, \theta_{13}, \theta_{23}) = (\text{const}, 0, 0, 0)$").

This is the M3-7 equivalent of M3-3's "2D 1D-symmetric ⊂ 1D"
dimension-lift gate. Run it on **every** 2D test M3-3 + M3-4 ship
(~50 tests; 15-25 min total runtime).

### 7.2 Berry verification reproduction (3D)

The 8 SymPy-verified Berry checks in
`scripts/verify_berry_connection_3D.py` (see
`notes_M3_phase0_berry_connection_3D.md` §4) are already reproduced
numerically at the **stencil level** in
`test/test_M3_prep_3D_berry_verification.jl` (797 asserts, all
passing). M3-7c re-tests the relevant subset at the **residual
level**:

- **CHECK 1 closedness (84 cyclic triples in 9D)**: structurally — the
  Newton Jacobian at any point is symmetric in the
  $(\alpha_a, \beta_a, \theta_{ab})$ block w.r.t. cyclic-sum
  identities. Sample 5 random points; verify the Jacobian sub-block
  is closed.
- **CHECK 2 per-axis match**: setting $\dot\theta_{ab} = 0$ for all
  three pairs in the 3D residual and dropping the off-axis terms
  reproduces M1's per-axis residual to 0.0. This is the 1D-symmetric
  reduction rephrased as a residual check.
- **CHECK 3a iso-pullback**: at $\alpha_1 = \alpha_2 = \alpha_3$,
  $\beta_1 = \beta_2 = \beta_3$, the Berry-block contribution to the
  residual vanishes. The $(\partial_\alpha, \partial_\beta)$ pullback
  to the iso slice is $3\alpha^2$, three times the 1D weight (CHECK
  3a of the 3D verification note).
- **CHECK 3b 2D reduction**: on the slice $\alpha_3 = $ const,
  $\beta_3 = 0$, $\theta_{13} = \theta_{23} = 0$, the 9D residual
  pulls back to the 5D 2D residual. This is the same as §7.1b of
  this gate list.
- **CHECK 5 explicit numerical values**: cross-check
  `berry_F_3d(α, β)` at the reference points hard-coded in
  `test_M3_prep_3D_berry_verification.jl` (already done in M3-prep;
  M3-7c inherits the regression).
- **CHECK 7 exactness (∂Θ/∂θ_{ab} = F_{ab})**: M3-7c asserts that the
  residual's per-pair $\theta_{ab}$ row produces the stencil's
  $F_{ab}$ at the appropriate point sample, to FD tolerance ≤ 1e-9
  (mirrors CHECK 7 of the 3D verification note).

### 7.3 Iso-pullback ε-expansion (3D)

At $\alpha_1 = \alpha_2 = \alpha_3 = \alpha_0 + \epsilon\,\delta_\alpha$,
$\beta_1 = \beta_2 = \beta_3 = \beta_0 + \epsilon\,\delta_\beta$,
$\theta_{12} = \theta_{13} = \theta_{23} = \epsilon\,\delta_\theta$,
the Berry term reduces per the ε-expansion (3D analog of the 2D §6.3
ε-expansion) to $O(\epsilon^2)$ per cell — i.e., the linear-in-ε
contribution from all three Berry blocks vanishes on the iso slice.
Verify by computing the residual at three ε values (e.g., $10^{-2}$,
$10^{-4}$, $10^{-6}$) and checking the leading-order behavior is
$O(\epsilon^2)$, not $O(\epsilon)$.

Tolerance: rel-error ≤ 1e-3 (the ε-extrapolation has limited
precision, same as 2D).

### 7.4 $\mathcal{H}_{\rm rot}$ solvability constraint (3D SO(3))

Sample 5–10 generic $(\alpha, \beta, \gamma)$ points (no axis
degeneracy, no realizability boundary). Compute the kernel direction
$v_{\rm ker}$ of the 9D $\Omega$ via SymPy (or extract from the
verification script's output — note SymPy normalises the
$\theta_{23}$-component to 1 by default, so the kernel direction has
all 9 components nonzero). Verify $dH \cdot v_{\rm ker} = 0$ where
$H = H_{\rm Ch}^{(3D)} + \sum_{a<b} \mathcal{H}_{\rm rot}^{ab}$ and
each $\mathcal{H}_{\rm rot}^{ab}$ uses the closed-form expression
from §2.2 above.

Tolerance: rel-error ≤ 1e-9 (looser than 2D's 1e-10 because the 3D
closed form has compound numerator/denominator polynomials reaching
magnitudes ~10⁵ at random points; SVD-style precision degrades —
this matches CHECK 4's tolerance in the verification note).

If this fails, the rotation Hamiltonian is wrong — Newton will not
converge on configurations with non-trivial $\theta_{ab}$ dynamics
(3D plane wave at oblique angle, 3D KH).

**Newton convergence sub-gate.** On a non-isotropic 3D IC (random
$\alpha, \beta, \theta_{ab}$ within realizability), Newton converges
in **≤ 7 iterations** (note: 7 not 5 because the 13-dof system is
larger than the 9-dof 2D system; the per-pair Berry blocks are
diagonal-dominant within their pair so the convergence rate is set
by the largest eigenvalue of the 13×13 within-cell Jacobian, not by
the cross-pair coupling).

### 7.5 Per-axis γ selectivity (3D)

Three configurations (per §5 above):

- **1D-symmetric in 3D** ($k = (1, 0, 0)$): γ_1 → small (collapsing
  axis); γ_2, γ_3 ≡ const (axes 2, 3 trivial). The action-error
  indicator fires only along axis 1. This is the 1D ⊂ 3D dimension
  lift of the 1D / M1 cold-sinusoid test.
- **2D-symmetric in 3D** ($k = (1, 1, 0)$): γ_1, γ_2 → small
  correlated; γ_3 ≡ const. The action-error indicator fires along
  axes 1 and 2 only, axis 3 stays trivial. This is the 2D ⊂ 3D
  dimension lift of M3-3d's selectivity test.
- **Full 3D** ($k = (1, 1, 1)$): all three γ's non-trivial. Each axis
  shows independent collapse dynamics.

The acceptance criterion is **per-axis selectivity**: the action-error
indicator must fire only along axes where γ_a → 0 (compressive
directions); axes where γ_a stays at O(1) (trivial directions) must
not trigger spurious refinement. This is the 3D demonstration that
per-axis decomposition is operationally meaningful — the same
scientific gate as M3-3d for the 2D code.

### 7.6 3D Bayesian remap conservation

M3-5's `BayesianRemapState{D, T}` is dimension-generic; HG's remap
is dimension-generic. M3-7e re-runs the M3-5 conservation gate on
3D meshes:

- Mass / momentum / energy preserved through L→E→L round-trip in 3D
  to ≤ 1e-12 (`:float` backend);
- Liouville monotone-increase necessary condition holds
  (n_negative_jacobian_cells == 0, total_volume conserved,
  total_overlap_volume ≈ box volume) per cycle;
- 5-cycle round-trip on a sinusoidally-deformed 3D Lagrangian mesh
  shows ≤ 6.7e-16 per cycle (matches M3-5's 2D cycle drift result);
- IntExact backend audit on a 3D-canonical polytope battery (HG's
  3D analog of the 2D 9-polytope battery; HG ships this — confirm at
  M3-7 launch).

### 7.7 3D Zel'dovich pancake (the cosmological reference test)

Methods paper §10.5 D.4: 3D Zel'dovich pancake collapse. The 3D
analog of dust collapse onto a single shell. This is the **headline
M3-7 scientific deliverable** — the cosmological reference test
that the 3D code must reproduce.

Acceptance criteria:

- **Pancake-collapse direction correctly identified by per-axis γ.**
  At the collapse time $t_{\rm crossing}$, $\gamma_1$ (the
  compressive axis) has dropped to $\sim 10^{-3}$ at the collapse
  cell, while $\gamma_2, \gamma_3$ stay at O(1). The per-axis γ
  diagnostic identifies axis 1 as the pancake axis.
- **Stochastic regularization on the compressive axis only.** The
  per-axis stochastic injection (M2-3 / Phase 8 carry-over,
  3D-extended in `realizability_project_3d!`) regularizes axis 1
  (raises $\gamma_1^2 \geq$ headroom × $\beta_1^2$); axes 2 and 3
  are not perturbed. This is the per-axis selectivity gate
  reformulated as the pancake test.
- **Density profile matches Zel'dovich's analytic shell solution
  to ≤ 5%** away from the singular sheet, **≤ 20%** at the sheet
  (where finite-resolution + stochastic regularization have biggest
  effect). Cross-check against the 1D collapse test from M2 +
  AppendixA1 of methods paper §10.5 — the 3D pancake should reduce
  to the 1D collapse on a slice perpendicular to the trivial axes.
- **Conservation:** mass, momentum, energy preserved (with explicit
  energy-injection bookkeeping from stochastic regularization) to ≤
  1e-10 across the run.

The 3D Zel'dovich IC is non-trivial; see §9 Risk for the IC-setup
caveats.

## 8. Wall time + risk

### Wall time

Per `MILESTONE_3_PLAN.md` Phase M3-7: **3-5 weeks**, down from the
original estimate (the HG 3D primitives + IntExact D=4 are ready,
and M3-prep already verified the 3D Berry stencils). Block breakdown:

| Block | Days |
|---|---|
| M3-7a — 3D field set + per-axis Cholesky 3D + halo smoke | 3 |
| M3-7b — Native HG-side 3D EL residual (no Berry) + dimension-lift gates | 4 |
| M3-7c — SO(3) Berry coupling integration + H_rot constraint + Berry verification | 3 |
| M3-7d — per-axis γ 3D + AMR/realizability 3D wiring | 2 |
| M3-7e — 3D Tier C consistency + 3D Tier D headlines (Zel'dovich, KH, dust traps) | 5 |
| **Total** | **~17 days (~3-3.5 weeks)** |

This is at the low end of the M3-plan estimate (3-5 weeks). Buffer
days for the secondary risks below should push toward 4 weeks
realistic.

### Main risk — 3D HaloView at depth=2

dfmm has used `depth = 1` throughout 1D / 2D. For face-only stencils
at order 0, `depth = 1` suffices. **Higher-order stencils or
diagonal-flux contributions** (e.g., for a 3D Zel'dovich pancake
where the compressive axis isn't axis-aligned) **might need
`depth = 2`** to access edge / corner neighbors.

HG supports `depth = 2`; HG's own 2D test suite exercises it but
dfmm hasn't. Defensive plan:

1. Before writing the 3D residual, write a `halo_smoke_test_3D.jl`
   that builds a 4×4×4 uniform 3D mesh, populates the
   `PolynomialFieldSet` with a known per-cell pattern, and walks
   *all 6 face neighbors* of a corner cell + an interior cell via
   `halo_view` at `depth = 1`, asserting the returned values match
   the pattern.
2. Add a second smoke test at `depth = 2` (12 face-edge-corner
   neighbors per cell) on a 6×6×6 mesh, asserting the same.
3. Run the smoke tests early; **block the residual development on
   the depth=1 smoke test passing**, and gate the M3-7e Tier-D
   drivers on depth=2 if any of them needs it.

This is the M3-3 lesson reapplied to 3D.

### Secondary risks

- **3D mesh AMR.** HG's `register_on_refine!` listener should work
  directly on `HierarchicalMesh{3}` (M3-3d already uses this on
  `HierarchicalMesh{2}`); the SimplicialMesh{3, T} Lagrangian AMR is
  hand-rolled per M3-3e-3's Option-B pattern (see
  `notes_M3_3e_3_amr_tracers_native.md` for the 2D pattern). M3-7d
  needs to verify that the Option-B pattern generalizes to 3D
  Lagrangian tetrahedra.

- **Cosmological IC for D.4 (3D Zel'dovich pancake).** The Zel'dovich
  approximation IC requires a power-spectrum-sampled displacement
  field. For 3D this is a 3-vector field on a regular grid, FFT'd to
  produce coherent perturbations in $k$-space, then realised onto the
  Lagrangian mesh. dfmm doesn't ship an FFT helper today; M3-7e will
  need to either: (a) borrow a minimal FFT-IC routine (e.g., from
  AbstractFFTs / FFTW), (b) hand-roll a single-mode pancake IC (a
  single-wavelength sine-perturbation along the compressive axis,
  which is the simplest 1D-collapse-dressed-up-as-3D test), or (c)
  load a pre-computed IC from disk.

  **Recommendation:** start with (b) — single-mode along axis 1.
  This is the minimal pancake test, and it cleanly reduces to the
  1D collapse from M2 (which is verified). If a multi-mode test is
  needed for paper figures, escalate to (a) or (c).

- **Newton convergence on the 13N-dof system.** M3-3c's 9N-dof
  Newton converges in 2-3 iterations on smooth ICs. M3-7's 13N-dof
  Newton with three Berry blocks should converge in **2-5 iterations
  on smooth ICs** and **5-10 on Tier-D ICs**. If you see >20
  iterations, debug the $\theta_{ab}$ initialization (set to the
  principal-axis frame of $M_{xx}^{(0)}$ via `cholesky_decompose_3d`,
  not zero) and the kernel-orthogonality residual on the $\theta_{23}$
  row (this is the analog of M3-3c's H_rot fix).

- **Sparsity-pattern propagation through `SparseMatrixCSC{Bool, Int32}`.**
  Same risk as M3-3c: cross-check `cell_adjacency_sparsity ⊗ ones(13, 13)`
  against `SparseConnectivityTracer.jacobian_sparsity` derived from
  the 3D residual. If they disagree, the manual pattern is missing
  a coupling — likely the $\theta_{ab}$-to-$\beta_c$ off-cell
  coupling from the strain-rotation correction in §2.3.

- **Off-diagonal $\beta$ creeping in.** The 3D Newton has six pinned
  off-diagonal scalars; the residual must not leak gradients into
  them. Add a sanity assertion at the end of every Newton step:
  `@assert all(abs.(β_offdiag_3d) .≤ ε)` for all cells. This catches
  off-diagonal leakage early (lesson from M3-3c).

- **GPU forward-compat.** M3-8's GPU port is downstream; the 3D
  residual should be allocation-free and `KernelAbstractions.jl`
  -compatible from day 1 (use `SVector{13}` for the residual,
  `SMatrix{3, 3}` for `dθ`, no allocating temporaries on the hot
  path). M3-3c achieved this; M3-7c should too.

## 9. Sub-phase split (suggested for the M3-7 launch agent)

The 3-5 week wall time is comfortable for one agent. Following M3-3's
M3-3a..e structure exactly:

- **M3-7a** (3 days). 3D field set in `src/setups_multid.jl` +
  `src/types.jl`; per-axis Cholesky decomposition driver in
  `src/cholesky_DD.jl` (`cholesky_decompose_3d`,
  `cholesky_recompose_3d`, `gamma_per_axis_3d`); halo smoke test on
  a 4×4×4 3D mesh (analog of M3-3a's smoke test). Dimension-lift
  gates stay passing (no functional change to the 1D / 2D residuals).

- **M3-7b** (4 days). Native HG-side 3D EL residual in `src/eom.jl`
  (`cholesky_el_residual_3D!`) and Newton driver in
  `src/newton_step_HG.jl` (`det_step_3d_HG!`), both **without**
  Berry. The 3D analog of M3-3b. Verify the dimension-lift gates
  §7.1a + §7.1b for zero-strain 3D tests:
  - 3D 1D-symmetric ⊂ 1D to ≤ 1e-12 (load-bearing): re-run all
    M3-2 1D tests in 3D 1D-symmetric mode.
  - 3D 2D-symmetric ⊂ 2D to ≤ 1e-12: re-run all M3-3b 2D tests in
    3D 2D-symmetric mode.

- **M3-7c** (3 days). SO(3) Berry coupling integration: consume
  `src/berry.jl::berry_partials_3d` in the residual; enforce the
  three $\mathcal{H}_{\rm rot}^{ab}$ kernel-orthogonality
  constraints (with the residual on the $\theta_{23}$ row); wire
  $(\theta_{12}, \theta_{13}, \theta_{23})$ as Newton unknowns.
  Re-verify the dimension-lift gates §7.1a + §7.1b at the Berry
  level (the Berry contributions must vanish on the dimension-lift
  slices). Run §7.2 (Berry verification reproduction at residual
  level), §7.3 (iso-pullback ε-expansion 3D), §7.4 (H_rot solvability
  3D + Newton convergence ≤ 7 iter).

- **M3-7d** (2 days). Per-axis γ 3D diagnostic + AMR/realizability
  per-axis 3D wiring. Run §7.5 (per-axis γ 3D selectivity in three
  setups: 1D-symmetric / 2D-symmetric / full 3D).

- **M3-7e** (5 days). 3D Tier C consistency drivers (3D Sod, 3D cold
  sinusoid, 3D plane wave) + 3D Tier D headlines (3D Zel'dovich
  pancake — §7.7; 3D KH; 3D dust traps). Run §7.6 (3D Bayesian
  remap conservation). Tier-D acceptance per methods paper §10.5
  D.4 / D.7 / D.10.

## 10. Critical references

Read all of these before drafting code. The **bold** entries are the
ones M3-7 will touch directly; the rest are for context.

### Berry-derivation + verification

- **`reference/notes_M3_phase0_berry_connection_3D.md`** — 3D SO(3)
  Berry derivation + 8 SymPy CHECKs. §1 (setup), §3 (per-pair
  ansatz), §4 (verifications), §5 (use for M3-7), §6 (open issues:
  off-diagonal + H_rot determination).
- `reference/notes_M3_phase0_berry_connection.md` — 2D Berry
  derivation. M3-7 generalizes per pair-generator; §6.6 (2D H_rot
  constraint) is the per-pair template.
- **`reference/notes_M3_prep_3D_berry_verification.md`** — Julia
  reproduction of all 8 SymPy CHECKs (797 asserts). M3-7c launch
  implications + pseudo-code in §"M3-7 launch implications" already
  pre-figure the residual integration.
- `scripts/verify_berry_connection_3D.py` — SymPy authority for the
  3D Berry connection. Run it once to see the output; M3-7c
  reproduces it numerically.
- `scripts/verify_berry_connection.py` — SymPy 2D verification (for
  cross-reference with the 2D-reduction CHECK 3b).
- **`src/berry.jl`** — M3-prep stencil implementation, 2D and 3D.
  M3-7 consumes `berry_F_3d`, `berry_term_3d`, `berry_partials_3d`,
  `BerryStencil3D` directly with no source changes.

### M3-3 sub-phase status notes (the 2D pattern M3-7 generalizes)

- **`reference/notes_M3_3_2d_cholesky_berry.md`** — the 2D design
  note. Mirror its structure section by section.
- `reference/notes_M3_3a_field_set_cholesky.md` — 2D field-set
  + per-axis Cholesky driver + HaloView smoke test. M3-7a does the
  3D analog.
- `reference/notes_M3_3b_native_residual.md` — 2D native HG-side
  residual without Berry. M3-7b does the 3D analog.
- `reference/notes_M3_3c_berry_integration.md` — 2D Berry residual
  integration + θ_R as Newton unknown + H_rot constraint. M3-7c
  does the 3D analog with three Euler angles.
- `reference/notes_M3_3d_per_axis_gamma_amr.md` — 2D per-axis γ +
  AMR/realizability/selectivity. M3-7d does the 3D analog.
- `reference/notes_M3_3e_*.md` (5 sub-notes) — 2D cache_mesh
  retirement + native det_step + native stochastic + native AMR/tracers
  + native realizability. M3-7 inherits the cache_mesh-free path; no
  retirement work required.

### M3-4 / M3-5 / M3-6 cross-references

- **`reference/notes_M3_4_tier_c_consistency.md`** — M3-4 IC bridge
  pattern (periodic-x coordinate wrap; C.1/C.2/C.3 drivers in flight).
  M3-7e generalizes the IC bridge to 3D.
- **`reference/notes_M3_5_bayesian_remap.md`** — M3-5 remap; M3-5
  is dimension-generic, so 3D extension is mostly threading the API
  through and adding 3D conservation tests.
- `reference/MILESTONE_3_STATUS.md` — M3 synthesis (read for
  cross-reference and current-state ground-truth).

### HG dependencies

- **`~/.julia/dev/HierarchicalGrids/`** — HG @ 81914c2 for
  `HierarchicalMesh{3}` + `SimplicialMesh{3, T}` + IntExact D=4.
- `~/.julia/dev/HierarchicalGrids/src/Storage/HaloView.jl` — HG
  halo API. Read in full before drafting; M3-3a discovered the
  return contract (`PolynomialView`, not `Tuple`).
- `~/.julia/dev/HierarchicalGrids/src/Mesh/Neighbors.jl` — HG
  sparsity API. Read the `cell_adjacency_sparsity` block (lines
  401–510 in 2D; check the same range for 3D).
- `~/.julia/dev/HierarchicalGrids/src/Mesh/BoundaryConditions.jl` —
  HG BC framework.
- `~/.julia/dev/HierarchicalGrids/src/Overlap/frame_boundaries.jl` —
  `FrameBoundaries{3}` constructor + accessors.
- `~/.julia/dev/HierarchicalGrids/src/Overlap/IntExact/` — IntExact
  backend (commit `cc6ed70`+) for 3D conservation regression.

### Methods-paper sections

- `specs/01_methods_paper.tex` §5.4 — Berry connection 3D SO(3) prose
  (the "3D extension is straightforward" flagged section). Revised
  in M3-prep + reflects the actual derivation in
  `notes_M3_phase0_berry_connection_3D.md`.
- `specs/01_methods_paper.tex` §10.5 D.4 — Zel'dovich pancake
  reference test description. M3-7e implements this.
- `specs/01_methods_paper.tex` §10.5 D.7 — dust trap reference test.
- `specs/01_methods_paper.tex` §10.5 D.10 — ISM tracer reference
  test.
- `reference/notes_M3_3_prep_paper_section10_revision_applied.md` —
  paper §10 revisions done in M3-3 prep.

### M1 / M2 carry-over

- **`src/cholesky_sector.jl`** — M1 per-axis kernel; reused per axis
  in 3D (just like 2D).
- **`src/cholesky_DD.jl`** — M3-3a's 2D per-axis Cholesky driver +
  M3-3c's `h_rot_partial_dtheta` + `h_rot_kernel_orthogonality_residual`.
  M3-7 extends with 3D analogs.
- **`src/eom.jl`** — M3-3b/c 2D residual + M3-4 periodic-wrap
  helper. M3-7 extends with 3D residual.
- **`src/newton_step_HG.jl`** — post M3-3e-5 cache_mesh retirement,
  this is the native HG Newton driver. M3-7 adds `det_step_3d_HG!`
  + `det_step_3d_berry_HG!`.
- `src/types.jl` — `DetField{T}` (M1 1D), `DetField2D{T}` (M3-3a),
  `DetFieldND{D, T}` (M3-0 doc tag). M3-7 promotes the doc tag to a
  working struct for D=3 (`DetField3D{T}`).
- `reference/notes_phase2_discretization.md` — M1 Phase 2 multi-cell
  discretization patterns.
- `reference/notes_phase8_stochastic_injection.md` — M1 Phase 8
  stochastic injection.

### Forward-compat (M3-8/M3-9)

- `reference/MILESTONE_3_PLAN.md` Phase M3-8 (Tier E + GPU/MPI) — for
  GPU/MPI forward-compat awareness (the 3D residual must be
  allocation-free and KernelAbstractions-compatible).
- `reference/MILESTONE_3_PLAN.md` Phase M3-9 — wrap-up.

### Julia-r3d port (M3-7 awareness)

- **Memory note:** `r3d_julia_port.md` — Tom is porting r3d to Julia;
  for M3-7 (and the broader M3 sequence) skip custom polygon-moment
  work and add it as a dependency. M3-5 already adopted this pattern
  via HG's IntExact backend; M3-7e should rely on the same path for
  3D conservation.

## 11. Open questions / decisions for the M3-7 launch agent

Mirror the M3-3 design note's §10. The seven open items:

1. **Choice of 3D Euler-angle parameterization.** The SymPy script
   `scripts/verify_berry_connection_3D.py` uses one specific
   convention (Cardan / extrinsic ZYZ — confirm by reading the script;
   `notes_M3_phase0_berry_connection_3D.md` §1 doesn't pin the
   convention). The choices:
   - **ZYZ extrinsic** (rotate about lab z by θ_{12}, then about new y
     by θ_{13}, then about new z by θ_{23}).
   - **ZXZ extrinsic** (the "physics" convention; rotate about lab z
     by θ_{12}, then about new x by θ_{13}, then about new z by
     θ_{23}).
   - **Cardan angles** (3 successive rotations about distinct axes:
     X-Y-Z or roll-pitch-yaw).
   - **Pair-generator parameterization** (the natural one for the
     SO(3) Berry derivation: each θ_{ab} is the rotation angle in
     the (e_a, e_b) plane, with the three not necessarily commuting).
   The 3D Berry derivation in
   `notes_M3_phase0_berry_connection_3D.md` uses pair-generator
   parameterization; the SymPy script verifies this. **Recommendation:**
   adopt pair-generator parameterization in M3-7; document the
   convention explicitly in the M3-7a sub-phase status note. *Action:
   confirm by reading the SymPy script + verification note; pin the
   convention in M3-7a; document.*

2. **HaloView depth in 3D: 1 vs 2.** dfmm has used `depth = 1`
   throughout. For M3-7 face-only stencils at order 0, `depth = 1`
   suffices. **Recommendation:** stay at `depth = 1` in M3-7a/b/c;
   smoke-test `depth = 2` in M3-7a as a forward-compat check;
   revisit in M3-7e if any Tier-D driver needs wider stencils.
   *Action: write the smoke test; default to depth=1 in the residual.*

3. **3D `MonomialBasis` order: 0 in M3-7.** Mirror M3-3's default.
   Higher-order (order 3, Bernstein cubic per methods paper §9.2)
   is M3-8 / M3-9 work — keep order 0 in M3-7 to match the
   dimension-lift gate. *Action: stay order 0; flag for M3-8.*

4. **Off-diagonal $\beta_{ab}$ in 3D: omit per M3-3 / M3-6 default
   unless a 3D-only D test needs them.** Same recommendation as M3-3
   §10 Q3: omit the six off-diagonals from the M3-7 field set;
   M3-9 (3D KH) will re-add them. *Action: omit; document.*

5. **3D `mesh.balanced` requirement.** Same as M3-3 §10 Q4: assume
   true; document deferred. M3-7 should require `mesh.balanced == true`;
   hanging-node handling in 3D is post-M3-9 work. *Action: assert in
   the 3D solver; document deferred.*

6. **3D Cholesky decomposition algorithm.** 3×3 SPD eigendecomposition
   is non-trivial closed-form (the 3×3 cubic is closed-form but
   numerically less stable than the 2×2 closed form M3-3a uses). Choices:
   - **`LinearAlgebra.eigen(SMatrix{3,3})`** — robust but allocates;
     slow on hot path.
   - **Closed-form 3×3 cubic** — analytic, allocation-free with
     `SMatrix`, but more sensitive to round-off near degeneracy
     boundaries.
   - **Jacobi rotations / Givens-style 3×3 SPD diagonalization** —
     iterative but converges in ~5 sweeps; numerically robust;
     allocation-free with `SMatrix`.
   **Recommendation:** start with `LinearAlgebra.eigen` for
   correctness; profile in M3-7d; switch to Jacobi-3x3 if hot-path
   allocation is the bottleneck. The 3D Cholesky decomposition is
   only called at IC + during AMR refinement — not on every Newton
   step — so the allocation cost is bounded. *Action: pick `eigen`
   for the first cut; profile in M3-7d.*

7. **3D Zel'dovich pancake IC source.** §9 Risk discusses three
   options: hand-rolled single-mode (b), borrowed FFT-IC (a), or
   pre-computed disk load (c). **Recommendation:** start with (b)
   (single-mode along axis 1); this is the minimal 3D pancake test
   and reduces cleanly to the 1D collapse. Multi-mode IC is a
   follow-up if paper figures need it. *Action: implement (b) in
   M3-7e; flag (a) for follow-up.*

These seven decisions shape the rest of M3-7's design. The recommended
defaults above (pair-generator parameterization, depth=1,
order 0, omit off-diag β, balanced only, eigen-based 3×3 decomposition,
single-mode pancake) are the conservative path; the M3-7 launch agent
may revisit any of them with cause.

### 11.8 Coordination with M3-6 (in flight)

M3-6 is doing the 2D Tier-D headlines. M3-6's D.1 KH may activate the
2D off-diagonal $\beta_{12}, \beta_{21}$ sector + supply the calibrated
2D off-diagonal $\mathcal{H}_{\rm rot}^{\rm off}$. M3-7 should
coordinate: (a) if M3-6 lands the 2D off-diagonal sector, M3-7's
dimension-lift gate §7.1b should re-run with the M3-6 2D code (to
ensure the SO(3) reduction reproduces 2D-with-off-diagonals correctly).
(b) M3-7 leaves the 3D off-diagonal sector dormant; the calibrated 2D
$\mathcal{H}_{\rm rot}^{\rm off}$ is the per-pair template that
M3-9 will use for 3D KH, but M3-7 does **not** ship a 3D
$\mathcal{H}_{\rm rot}^{\rm off}$. *Action: rebase M3-7 onto M3-6's
HEAD before starting M3-7c (Berry integration); confirm the 2D
off-diagonal pattern (if landed) and ensure M3-7's 2D-symmetric
dimension-lift gate is regression-proof against it.*

### 11.9 Coordination with M3-4 (in flight)

M3-4 Phase 1 (periodic-x wrap) is done; Phase 2 (IC bridge + C.1/C.2/C.3
drivers) is in flight. M3-7e's 3D Tier-C drivers should mirror the
M3-4 IC-bridge pattern. *Action: rebase M3-7 onto M3-4's HEAD before
starting M3-7e (3D Tier-C drivers); reuse the IC factory pattern as-is.*

### 11.10 Forward-compat with the r3d Julia port

The Julia r3d port (per the auto-memory note) is in flight by Tom;
when complete, M3-7 should adopt it as a dependency rather than
hand-rolling 3D polytope-moment computations. M3-5 already adopted
this pattern via HG's IntExact backend, which delegates to r3djl
when available. M3-7e's 3D conservation gate should pick up r3djl
the same way M3-5 did. *Action: track r3d-Julia status; if r3djl is
ready by M3-7e, adopt; else fall back to HG's current backend.*

---

*End of M3-7 design note. ~870 lines including math + pseudocode +
reference list. Estimated reading time for the M3-7 launch agent:
~35 min, plus another ~75 min reading the §10 references in full
(especially the 2D design note + M3-3 sub-phase status notes + the
3D Berry derivation + verification note). After that, M3-7a (the
3D field-set extension + 3D halo smoke test) should take ~half a
day to start coding.*
