# M5 plan — physics-extension closures + scaling + paper resubmission

> **Status (2026-04-26):** Drafted at M4 close (M4 wrap-up). M5
> starts from `main` post-M4-wrap-up. The three M4 sub-phases that
> shipped (M4-1, M4-2, M4-3) closed the substrates for D.1 closed-
> loop β coupling and D.7 per-species momentum and produced three
> honest-partial scientific verdicts; the literal D.1 eigenmode and
> literal D.7 centrifugal-accumulation predictions remain open in the
> precise sense of needing additional physics terms. The original
> M4 plan's GPU/MPI/E.4/D.6-D.8-D.9/Bernstein/paper-resubmission
> phases were deferred to this milestone.

## Scope summary

M5 is a *physics-extension closure + scaling + paper resubmission*
milestone. The two M4 honest-partial loops close at M5-1 / M5-2; the
upstream HG `Backend` PR + downstream Metal/CUDA + MPI scaling close
the engineering deliverables (M5-3 through M5-6); the paper
resubmission with consolidated M3 + M4 + M5 findings closes the
writing deliverable (M5-7).

## Phase plan

### M5-1 — Centrifugal-force kernel + two-way momentum coupling (D.7 closure)

**Headline goal:** activate the literal D.7 centrifugal-accumulation
prediction. Dust in a steady vortex should drift inward (decoupled
limit) under the centrifugal force; dust + gas should equilibrate via
two-way drag exchange.

**Substrate work:** two new kernels.

1. `apply_centrifugal_drift!(psm, frame, leaves, dt; lo, hi, wrap)`
   — for each species k and cell ci, compute `(u_k · ∇)u_gas` via
   finite differences across the cell's four neighbors (2D) or six
   (3D); apply `du_k = -dt · (u_k · ∇)u_gas` as an inertial drag-like
   term. This activates inward spiral drift for decoupled dust in a
   steady vortex.

2. Two-way momentum exchange: extend
   `cholesky_el_residual_2D_berry!` (and the 3D analog) with a back-
   reaction term `Σ_k≠gas ρ_k (u_k − u_gas) / τ_drag_k` on the gas
   momentum equation (the Cholesky-sector velocity unknowns).
   Requires per-species ρ_k accessible at residual evaluation time
   (currently implicit in `tm.tracers` as concentration · ρ_per_cell;
   the per-species momentum struct already carries the velocity).

**Acceptance gates (D.7 falsifier):**
- Decoupled (τ_drag = ∞) dust in steady Taylor-Green vortex shows
  inward radial drift toward vortex centres (peak/mean ratio of
  remapped concentration grows over T_KH; collapse_fraction rises
  from 0 toward 1).
- Tightly-coupled (τ_drag → 0) dust tracks gas exactly (peak/mean
  identity to passive scalar).
- Intermediate τ_drag produces size-dependent drift between the two
  limits.
- Two-way momentum exchange: gas energy injection from drag drops
  out under conservative coupling; total (gas + dust) momentum
  preserved bit-exactly under periodic BCs.
- Bit-exact regression preservation: at zero per-species momentum,
  M3-6 Phase 4 / M4-3 path stays byte-equal.

**Expected wall time:** ~3 weeks. The kernels are O(N) in cells; the
gas residual two-way coupling is a localized add at finite-difference
cost. The bulk of the time is the falsifier driver tuning (matching
peak/mean against PencilCode / AREPO reference data).

**Prerequisites:** none beyond M4-3 substrate.

**References:**
- `notes_M4_phase3_per_species_momentum.md` §"M4 Phase 4 handoff
  items" (the source spec).
- M4 closing memo on M4-3.

### M5-2 — Higher-order Hamiltonian (per-cell linearised Rayleigh) (D.1 closure)

**Headline goal:** activate the literal D.1 Drazin-Reid eigenmode.
The KH growth trajectory should switch from linear-in-t (kinematic
forced) to exp-in-t (eigenmode self-amplified) with γ_measured
matching γ_KH = U/(2w) within 20%.

**Substrate work:** three candidate routes (M5-2a / M5-2b / M5-2c).
Pick one in the M5-2 prep phase; the most promising is M5-2a.

**M5-2a (recommended): per-cell linearised Rayleigh reconstruction.**
At the stencil level, project the perturbation onto the eigenvectors
of the linearised Rayleigh operator. This activates the eigenmode
dynamics directly. Algorithmically: at each cell, compute the local
linearised operator across nearest neighbours; diagonalize; project
the perturbation onto the unstable eigenvector; advance with the
matching γ. Adds O(stencil_size³) per-cell cost (k×k matrix
diagonalization where k = stencil size = 5 in 2D, 7 in 3D).

**M5-2b: cubic/quartic Hamiltonian extension.** Add a cubic or
quartic term in (β_off, β_a) perturbation amplitudes that creates a
positive-eigenvalue block in the linearised matrix. Mathematically
elegant; requires careful sign-bookkeeping to preserve axis-swap
antisymmetry and bit-exact regression at β_off = 0.

**M5-2c: symplectic-natural F^β_off form.** Replace Phase 1a's
kinematic `β̇_off + G̃·α/2` row with the full symplectic-row form
(no `β̇_off` term; α̇'s and θ̇_R balancing G̃·α/2 provide the
constraint). Mathematically more correct but requires Newton-system
regularization to handle the rank-6 Casimir kernel; M4 Phase 1c
explored this briefly.

**Acceptance gates (D.1 falsifier):**
- KH growth γ_measured/γ_DR ∈ [0.8, 1.2] at level 4 / level 5
  (tighter aspiration band).
- Linear-in-t vs exp-in-t fit comparison: exp-in-t fits the
  trajectory significantly better than linear-in-t (ssr_exp ≪
  ssr_lin).
- Bit-exact regression preservation: at β_off = 0 IC, M3-6 Phase 0
  / Phase 1a / M4-1 / M4-2 paths stay byte-equal.
- 3D lift: same eigenmode behaviour holds in 3D under the M4-2 21-
  dof path (closes the M4-2 honest-falsification-lifted loop).
- n_negative_jacobian = 0 throughout the trajectory (realizability
  preserved).

**Expected wall time:** ~6 weeks. M5-2a per-cell diagonalization is
the heaviest path; M5-2b sign-bookkeeping is delicate; M5-2c
requires Newton-system regularization. Expected total dev time
across explore + implement + falsifier validation.

**Prerequisites:** none beyond M4-1/M4-2 substrate.

**References:**
- `notes_M4_phase1_closed_loop_beta.md` §"Honest scientific finding"
  interpretations 1-3.
- `notes_M4_phase2_3d_kh_falsifier.md` §"Honest scientific finding"
  routes (a-c).
- `scripts/verify_berry_connection_offdiag.py` CHECK 9 (KH-shear
  linearization).

### M5-3 — HG `Backend`-parameterized `PolynomialFieldSet` (upstream PR)

**Headline goal:** unblock the GPU port. The HG `PolynomialFieldSet`
SoA storage is currently CPU-only; lifting it to a `Backend`-
parameterized form (`PolynomialFieldSet{<:KA.Backend}` in the sense
of `KernelAbstractions.jl`) is the prerequisite for the per-leaf
residual kernel GPU port.

**Substrate work:** PR to `HierarchicalGrids.jl` upstream. The PR
should:
1. Replace `Vector{T}` storage with `KA.AbstractArray{T}` parameterized
   on a `Backend` type parameter.
2. Add `device_array(::Backend, ::Type{T}, dims...)` constructor
   helpers consistent with KA's design.
3. Preserve the CPU-only fast path: when `Backend = CPU`, the layout
   reduces to `Vector{T}` exactly (no pointer-chasing penalty).
4. Update the existing `compute_overlap`, `polynomial_remap_l_to_e!`,
   `face_neighbors_with_bcs`, `HaloView` machinery to work
   transparently with the new storage.

**Acceptance gates:**
- HG test suite passes byte-equal at `Backend = CPU` against the
  pre-PR baseline.
- Smoke test on `Backend = CUDABackend()` (or `MetalBackend()`) for
  basic field-set construction + reduction.
- The dfmm M3-3 / M3-4 / M3-7 / M4 regression battery passes byte-
  equal at `Backend = CPU` against the pre-PR baseline (verifies
  the dfmm code uses HG transparently).

**Expected wall time:** ~4 weeks (HG-side work + upstream PR review +
dfmm-side adapter validation).

**Prerequisites:** none.

**References:**
- `notes_M3_8b_metal_gpu_port.md` (M3 GPU readiness audit).
- `notes_HG_design_guidance.md` (HG architectural notes).

### M5-4 — Apple Metal kernel port for the per-cell residual

**Headline goal:** 5×–10× speedup at level 5–7 on Apple Silicon.

**Substrate work:** port the per-leaf residual kernel via
`KernelAbstractions.jl` `@kernel` definition. The matrix-free Newton-
Krylov outer driver (M3-8b) already supports `concrete_jac = false +
jvp_autodiff = AutoForwardDiff()`; the GPU work is the per-cell
residual evaluation. Three deliverables:
1. `cholesky_el_residual_2D_berry_GPU!` `@kernel` definition.
2. `cholesky_el_residual_3D_berry_kh_GPU!` `@kernel` definition (M4-2
   21-dof path).
3. `det_step_*_HG_matrix_free_GPU!` outer driver routing the
   residual through the GPU kernel.

**Acceptance gates:**
- Bit-equal to round-off vs CPU matrix-free path (`max abs diff ≤
  1e-10` rel-error contract from M3-8b).
- 5×–10× speedup at level 5–7 (the targets named in the M3-8a GPU
  readiness audit).
- Realizability projection still works correctly (currently CPU-side;
  may need a separate GPU port).
- D.4 / D.7 / D.10 falsifier results reproduced byte-equal-modulo-
  rounding under the GPU path.

**Expected wall time:** ~6 weeks.

**Prerequisites:** M5-3 (HG `Backend` upstream).

### M5-5 — CUDA backend

**Headline goal:** match Metal port performance on NVIDIA hardware.

**Substrate work:** with the KA `@kernel` definition from M5-4, CUDA
support comes essentially free; the work is regression-testing
across CUDA versions, debugging any KA-CUDA-specific issues, and
documenting the supported configuration matrix.

**Acceptance gates:** same as M5-4 but on CUDA backend.

**Expected wall time:** ~3 weeks (mostly testing + debugging).

**Prerequisites:** M5-4 (Metal port stabilized).

### M5-6 — MPI scaling on HG chunk structure

**Headline goal:** demonstrate weak scaling at 4 / 16 / 64 MPI ranks
and strong scaling at fixed problem size.

**Substrate work:** domain decomposition via HG's
`partition_for_threads(mesh, n_chunks)` chunk structure (one chunk =
one MPI rank). Three deliverables:
1. Halo exchange wraps HG's `HaloView` + `face_neighbors_with_bcs`
   machinery; one MPI Sendrecv per neighbour rank per substep.
2. Stochastic injection on multi-rank: byte-equal RNG invariant
   requires careful design (the M3-3e-2 single-rank invariant is the
   starting point; per-chunk seeded streams).
3. Action-AMR refinement events serialized across ranks via global
   reduction on the per-cell `Δ S_cell` indicator.

**Acceptance gates:**
- Weak scaling at 4 / 16 / 64 ranks (efficiency > 80% at level 4;
  fixed work per rank).
- Strong scaling at fixed problem size (level 6, ~64k leaves)
  across 1, 4, 16, 64 ranks.
- Bit-exact regression on single-rank vs multi-rank under
  deterministic-only path (RNG invariant for stochastic separately).
- Falsifier results (D.4, D.7, D.10) reproduced under multi-rank
  configuration.

**Expected wall time:** ~5 weeks.

**Prerequisites:** none beyond M3-8b matrix-free path; can run in
parallel with M5-4 / M5-5.

### M5-7 — Methods paper resubmission

**Headline goal:** ship the methods paper as a journal submission
with M3 + M4 + M5 findings consolidated.

**Final figures:**
- D.1 KH eigenmode growth: linear-in-t baseline (M4-1, M4-2) +
  exp-in-t with eigenmode (post-M5-2). 4-panel figure.
- D.7 vortex-centre accumulation: M3-6 Phase 4 substrate baseline +
  M4-3 per-species momentum + M5-1 centrifugal drift. 4-panel figure.
- D.4 3D Zel'dovich pancake (already landed in M3-7e).
- D.10 ISM tracers (already landed).
- GPU/MPI scaling plots (post-M5-4/5/6).

**Two-fluid 2D content:** D.6 / D.8 / D.9 paragraphs in §10.5 if those
are pursued as orthogonal M5+ items; otherwise leave the §10.5 D.6 /
D.8 / D.9 entries as "framework supports; example deliverable to
follow" placeholders.

**Methods paper supplementary materials:**
- `reference/MILESTONE_3_FINAL.md` 1-page synthesis as appendix §A.
- `reference/MILESTONE_4_FINAL.md` 1-page synthesis as appendix §B.
- `reference/MILESTONE_5_FINAL.md` 1-page synthesis as appendix §C.
- The honest-revision protocol (§10.7 honest characterization +
  M3-9 / M4-wrap-up close memos) as appendix §D.

**Submission package:** journal target (J.~Comput.~Phys. or
M.N.R.A.S. methods paper), cover letter, response-to-reviewers
template.

**Acceptance gates:** none beyond editorial review.

**Expected wall time:** ~6 weeks (writing + figure polish + journal
formatting + cover letter).

**Prerequisites:** M5-1 (D.7 closure) + M5-2 (D.1 closure) + M5-4 or
M5-5 (GPU performance plot) + M5-6 (MPI scaling plot).

## Test count growth (projected)

| Phase | Tests added | Cumulative |
|---|---:|---:|
| M4 close (M4 wrap-up entry) | 37,433 + 1 deferred | 37,433 + 1 |
| M5-1 D.7 centrifugal closure | ~2500 | ~39,900 |
| M5-2 D.1 eigenmode closure | ~2500 | ~42,400 |
| M5-3 HG `Backend` upstream | 0 (HG-side) | ~42,400 |
| M5-4 Metal kernel port | ~1500 | ~43,900 |
| M5-5 CUDA backend | ~1000 | ~44,900 |
| M5-6 MPI scaling | ~1500 | ~46,400 |
| M5-7 paper resubmission | 0 (paper-only) | ~46,400 |

Total estimated M5 test growth: ~9,000 asserts.

## Critical-path constraints

1. **M5-3 HG `Backend` upstream must land before M5-4.** Without it
   the per-leaf residual kernel cannot be cleanly ported to GPU. The
   M3-8b matrix-free driver is the algorithm-side prerequisite
   already in place.

2. **M5-1 and M5-2 are independent and can run in parallel.** Each
   closes one literal-claim loop; both are needed for M5-7 paper
   resubmission.

3. **M5-4 and M5-6 are independent and can run in parallel.** Both
   are engineering scaling deliverables; M5-7 needs at least one for
   the GPU/MPI scaling plot.

4. **M5-2 is the single highest-risk item.** All three sub-routes
   (per-cell Rayleigh, cubic/quartic H, symplectic-natural F^β_off)
   are non-trivial; if all three falsify, M5-2 closes in
   HONEST_FALSIFICATION mode and the methods paper §10.5 D.1
   declares the eigenmode as a known unsolved physics-substrate
   problem. The M3-9 / M4-wrap-up honest-reporting protocol carries
   forward.

## Honest-reporting protocol carry-forward

The M3-9 §10.5 honest revision (D.1 / D.7 framing) and M4 wrap-up
§10.7 "Honest scientific characterization" subsection are the
precedents. Each M5 phase report should declare upfront which of the
methods-paper claims it *verifies natively* vs *requires extension
for* vs *falsifies*. The M5-1 / M5-2 deliverables are exactly such
extensions; the rest of M5 should similarly declare its honest
findings as it lands.

## Deferred items beyond M5

These remain tracked from the original M4 plan but are orthogonal to
M5-1 through M5-7 and may slip to M6 if M5 lands quickly:

- **D.6 cold-sinusoid dust-in-gas** (per-species per-axis
  diagnostics; orthogonal to M5-1 centrifugal closure).
- **D.8 plasma equilibration with anisotropy** (intra- vs inter-
  species relaxation via BGK τ_P / ν^T_AB cross-coupling kernels
  preserved from dfmm).
- **D.9 tracer fidelity in 2D KH** (sharp-interface preservation
  2-3× better than standard schemes; piggybacks on M4-1/M4-2/M5-2
  KH driver).
- **Higher-order Bernstein reconstruction** (Open Issue #2 carry-
  forward; HG already supports MonomialBasis{D, k} for k > 0).
- **E.4 cosmological-with-self-gravity** (multigrid Poisson on the
  Eulerian quadtree; gravitational potential interpolated to
  Lagrangian vertices; CDM-style 2D collapse IC).
- **3D extensions of D.7 (per-species momentum) and D.10 (ISM
  tracers).** D.7 3D needs `PerSpeciesMomentumHG3D` +
  `tier_d_dust_trap_3d_per_species_ic_full`; D.10 3D needs
  `tier_d_ism_tracers_3d_ic_full`.
- **ColDICE / PM N-body cross-comparison** for D.4 (external dataset
  ingest path).

## References

- `reference/MILESTONE_4_FINAL.md` — M4 final synthesis (this is
  the M5 entry point's parent context).
- `reference/MILESTONE_4_STATUS.md` §"M4 close" — full close ledger.
- `reference/notes_M4_phase1_closed_loop_beta.md` §"Honest scientific
  finding" — M5-2 design constraint (2D).
- `reference/notes_M4_phase2_3d_kh_falsifier.md` §"Honest scientific
  finding" — M5-2 design constraint (3D).
- `reference/notes_M4_phase3_per_species_momentum.md` §"M4 Phase 4
  handoff items" — M5-1 design constraint.
- `reference/notes_M3_8b_metal_gpu_port.md` — M5-4 prerequisite
  (HG `Backend` parameterisation; M3 GPU readiness audit).
- `reference/notes_M3_8a_tier_e_gpu_prep.md` — M5-4 / M5-5 port
  plan.
- `reference/notes_M3_7e_3d_tier_cd_drivers.md` — 3D substrate +
  D.4 3D headline (M5-1 3D path starting point).
- `specs/01_methods_paper.tex` §10.5 / §10.7 / §10.9 — the methods-
  paper claims that M5 needs to extend or close.
- `reference/notes_M4_plan.md` — the M4 plan (original); items
  rolled forward to M5 via deferral.
