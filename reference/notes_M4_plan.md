# M4 plan — physics extensions + scaling + paper resubmission

> **Status (2026-04-26):** Drafted at M3 close (M3-9). M4 starts
> from `main` post-M3-9 wrap-up. M3 has closed all six 2D Tier-D
> headlines (D.1 broad-band, D.4 2D + 3D, D.7 substrate sound,
> D.10 bit-exact); M4 picks up the four follow-ups M3 honestly
> flagged plus the engineering scaling (GPU, MPI) and the methods-
> paper resubmission package.

## Scope summary

M4 is a *physics + scaling* milestone, in contrast to M3's
*substrate + Tier-D* milestone. The M3 honest findings on D.1
(closed-loop β coupling missing) and D.7 (per-species momentum
coupling missing) define the headline M4 physics extensions. The
3D Tier-D headlines, full GPU port, and MPI scaling complete the
engineering deliverables. The methods-paper resubmission package
is the writing deliverable.

## Phase plan (proposed)

### M4-1 — Off-diagonal β closed-loop coupling (D.1 follow-up)

**Headline goal:** make the 2D KH falsifier go *eigenmode* (self-
amplified Drazin-Reid exponential growth) rather than *kinematic*
(forced linear response to base-flow strain).

**Substrate work:** add the closed-loop coupling between the along-
axis Cholesky factors β_a and the perturbation amplitude β_off in
`cholesky_el_residual_2D_berry!`. The current Phase 1a residual
wires the strain coupling H_rot^off = G̃_12·(α_1·β_21 + α_2·β_12)/2
into F^β_a, F^β_12, F^β_21, F^θ_R rows but not the back-reaction
of β_off onto β_a's evolution beyond the residual's natural
coupling. The closed-loop addition needs:

- A consistent linearised Rayleigh-equation reduction of the residual.
- Verification at the linear-instability gate: γ_measured =
  γ_DR exp(γ_DR·t) (exponential), not γ_DR + linear forcing term.
- Mesh-convergence + n_negative_jacobian = 0 retained.
- Bit-exact regression on the Tier-C / pre-Phase-1a paths.

**Deliverable:** D.1 falsifier *eigenmode-passing* gate, with
γ_measured/γ_DR ∈ [0.8, 1.2] at the tighter calibration band
(replacing the current broad-band [0.5, 2.0]).

**Reference:** `reference/notes_M3_6_phase1c_D1_kh_falsifier.md`
§"Honest scientific finding" for the design constraint.

### M4-2 — Per-species momentum coupling (D.7 follow-up)

**Headline goal:** capture sub-cell centrifugal drift of dust toward
vortex centres, so the literal D.7 prediction "vortex-centre
accumulation matches reference codes (PencilCode, AREPO with dust)"
becomes reproducible.

Two design routes (one will be selected in M4-2 prep):

**Route A: per-species momentum + drag.** Each species k carries a
per-cell velocity u_k and a drag relaxation toward gas:
∂_t (ρ_k u_k) = -ρ_k (u_k - u_gas) / τ_drag(size). Adds
n_species × 2 dof per cell. For the 2-species `[:gas, :dust]` case
this is +4 dof, ~22-dof Newton system. Pros: physically
transparent, exposes the full size-dependent drift. Cons:
substantial residual surgery + new IC bridge.

**Route B: Lagrangian volume update via M3-5 Bayesian remap.** The
existing M3-5 `bayesian_remap_l_to_e!` / `_e_to_l!` machinery
already supports per-species mass tracking; composing it with the
deterministic step lets the Lagrangian frame's volume-tracking
update the per-cell tracer concentration in response to local
volume change. Pros: uses existing substrate; minimal residual
surgery; the M3-5 IntExact backend gives a conservation gate by
construction. Cons: doesn't expose *size-dependent* drift; only
captures the gas-frame compression-induced concentration.

Recommended: M4-2a Route B (cheap, fast); M4-2b Route A (full).

**Deliverable:** D.7 falsifier *literal-claim-passing* gate, with
peak/mean ratio matching reference-code data within published
tolerance.

**Reference:** `reference/notes_M3_6_phase4_D7_dust_traps.md`.

### M4-3 — 3D Tier-D headlines

The 3D analogues of M3-6's Tier-D battery, leveraging the M3-7e
substrate. Three sub-phases:

**M4-3a: D.1 3D KH.** Lift `tier_d_kh_ic_full` to 3D via the
M3-7e prep pattern; calibrate γ_measured/γ_DR; verify the M4-1
closed-loop coupling extends to 3D (it should, by the structural
identity of the Rayleigh equation).

**M4-3b: D.7 3D dust traps.** 3D Taylor-Green vortex (e.g., ABC
flow) + 3D `TracerMeshHG3D[:gas, :dust]`; the M4-2 per-species
momentum coupling extends per-axis. Cross-check the 2D ⊂ 3D
selectivity: a 2D-symmetric IC should reproduce the 2D D.7
verdicts byte-equal.

**M4-3c: D.10 3D ISM.** 3D analogue of `tier_d_ism_tracers_ic_full`;
the bit-exact preservation should hold (the structural argument
extends per-cell).

### M4-4 — Full Metal/CUDA port

**Prerequisite:** HG ships `PolynomialFieldSet{<:KA.Backend}` (the
upstream `Backend` parameterisation flagged in
`notes_M3_8b_metal_gpu_port.md`).

**Substrate work:** port the per-leaf residual kernel via
`KernelAbstractions.jl` `@kernel` definition; maintain the
matrix-free Newton-Krylov outer driver. Deliverable: 5×–10× speedup
at level 5–7 (the targets named in the M3-8a GPU readiness audit).

**Bit-exact regression:** the GPU path must match the CPU matrix-
free path to round-off (`max abs diff ≤ 1e-10` rel-error contract
documented in `notes_M3_8b_metal_gpu_port.md`).

### M4-5 — MPI scaling

Domain decomposition via HG's `partition_for_threads(mesh, n_chunks)`
chunk structure (one chunk = one MPI rank). Halo exchange wraps
HG's `HaloView` + `face_neighbors_with_bcs` machinery. Stochastic
injection on multi-rank: byte-equal RNG invariant requires careful
design (the M3-3e-2 single-rank invariant is the starting point).
Test gates: weak scaling at 4 / 16 / 64 ranks; strong scaling at
fixed problem size.

### M4-6 — E.4 cosmological IC + ColDICE / PM N-body comparison

Multigrid Poisson on the Eulerian quadtree; gravitational potential
interpolated to Lagrangian vertices. CDM-style 2D collapse IC at
the loaded scale; cross-comparison to ColDICE 2D and a PM N-body
reference. The D.4 ColDICE comparison (mentioned in §10.5 D.4)
also lands here, since both require external-dataset ingest.

### M4-7 — D.6 / D.8 / D.9 two-fluid 2D content

Orthogonal to D.7's per-species momentum coupling: D.6 cold-sinusoid
dust-in-gas (per-species per-axis diagnostics); D.8 plasma
equilibration with anisotropy (intra- vs inter-species relaxation);
D.9 tracer fidelity in 2D KH (sharp-interface preservation 2-3×
better than standard schemes). The D.6 driver is the simplest;
D.8 needs the BGK τ_P / ν^T_AB cross-coupling kernels (preserved
from dfmm); D.9 piggybacks on the M4-1 KH falsifier driver.

### M4-8 — Higher-order Bernstein reconstruction

Open Issue #2 carry-forward: the variational scheme's
shock-capture L∞ is loose vs HLL golden (~10-20% in the C.1 1D-
symmetric Sod gate; ~50% at high Mach in E.1). Higher-order
Bernstein per-cell reconstruction would tighten this. Substrate:
HG already supports MonomialBasis{D, k} for k > 0; the residual
needs adapting. Test gate: C.1 1D-symmetric Sod L∞ tightens by
~3×.

### M4-9 — Methods paper resubmission

**Final figures:** D.1 KH eigenmode growth (post-M4-1); D.7 vortex-
centre accumulation (post-M4-2); D.4 3D Zel'dovich pancake (already
landed in M3-7e); D.10 ISM tracers (already landed); GPU/MPI scaling
plots (post-M4-4/5).

**Two-fluid 2D content:** D.6, D.8, D.9 paragraphs in §10.5 (post-
M4-7).

**Methods paper supplementary materials:** the
`reference/MILESTONE_3_FINAL.md` 1-page synthesis as appendix §A;
the M3-9 honest-revision protocol as appendix §B.

**Submission package:** journal target (J.~Comput.~Phys. or
M.N.R.A.S. methods paper), cover letter, response-to-reviewers
template.

## Test count growth (projected)

| Phase | Tests added | Cumulative |
|---|---:|---:|
| M3 close (M3-9 entry) | 33940 + 3 deferred | 33940 + 3 |
| M4-1 D.1 closed-loop | ~2000 | ~36000 |
| M4-2 D.7 per-species | ~2500 | ~38500 |
| M4-3 3D Tier-D | ~6000 | ~44500 |
| M4-4 Metal/CUDA port | ~1500 | ~46000 |
| M4-5 MPI scaling | ~1000 | ~47000 |
| M4-6 E.4 cosmological + ColDICE | ~2000 | ~49000 |
| M4-7 D.6 / D.8 / D.9 | ~3000 | ~52000 |
| M4-8 Bernstein reconstruction | ~1500 | ~53500 |
| M4-9 paper resubmission | 0 (paper-only) | ~53500 |

Total estimated M4 test growth: ~20,000 asserts (matching the M3
growth rate of ~14,000 asserts over 9 phases).

## Critical-path constraints

1. **HG `Backend` parameterisation must land before M4-4.** Without
   it the per-leaf residual kernel cannot be cleanly ported to GPU.
   The M3-8b matrix-free driver is the algorithm-side prerequisite
   already in place.

2. **M4-1 closed-loop β coupling must precede M4-3a (3D KH).** The
   3D D.1 driver should exercise the same eigenmode dynamics; landing
   it on the kinematic-only residual would only repeat the M3-6
   Phase 1c finding in 3D.

3. **M4-2 per-species momentum must precede M4-3b (3D D.7).** Same
   reasoning: the 3D dust-trap driver needs the physics extension
   to be a meaningful test.

4. **M4-7 (D.6 / D.8 / D.9) is largely independent of M4-1/M4-2.**
   Can run in parallel.

## Honest-reporting protocol carry-forward

The M3-9 §10.5 honest revision (D.1 / D.7 framing) is the precedent.
Each M4 phase report should declare upfront which of the methods-
paper claims it *verifies natively* vs *requires extension for* vs
*falsifies*. The H4-1 / M4-2 deliverables are exactly such
extensions; the rest of M4 should similarly declare its honest
findings as it lands.

## References

- `reference/MILESTONE_3_FINAL.md` — M3 final synthesis (this is
  the M4 entry point's parent context).
- `reference/MILESTONE_3_STATUS.md` §"M3 close" — full close ledger.
- `reference/notes_M3_6_phase1c_D1_kh_falsifier.md` §"Honest
  scientific finding" — M4-1 design constraint.
- `reference/notes_M3_6_phase4_D7_dust_traps.md` §"Headline
  scientific finding (HONEST)" — M4-2 design constraint.
- `reference/notes_M3_8b_metal_gpu_port.md` — M4-4 prerequisite
  (HG `Backend` parameterisation).
- `reference/notes_M3_8a_gpu_readiness_audit.md` — M4-4 / M4-5
  port plan.
- `reference/notes_M3_7e_3d_tier_cd_drivers.md` — 3D substrate +
  D.4 3D headline (M4-3 starting point).
- `specs/01_methods_paper.tex` §10.5 / §10.6 / §11 — the methods-
  paper claims that M4 needs to extend or verify.
- `reference/MILESTONE_3_PLAN.md` — the M3-as-planned plan, for
  comparison to M3-as-shipped (M4 should track an equivalent
  `MILESTONE_4_PLAN.md` as it gets drafted).
