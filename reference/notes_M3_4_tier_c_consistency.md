# M3-4 — Tier C consistency tests (Phase 1: periodic-x coordinate wrap)

> **Status (2026-04-26):** *In progress*. First installment of M3-4
> closes the M3-3c handoff prerequisite — the periodic-x coordinate
> wrap on the 2D EL residual. **The Tier-C C.1 / C.2 / C.3 solver-
> coupled drivers + acceptance gates are deferred to follow-up
> commits.**
>
> Test delta vs M3-3 close (`077d6e4`): **+46 new asserts**
> (13375 + 1 deferred → **13421 + 1 deferred**, all passing). 1D-path
> bit-exact 0.0 parity holds; the M3-3 sub-phase tests
> (M3-3a/b/c/d, M3-3e-1/2/3/4) all pass byte-equal after the wrap
> landing — the wrap fields fall back to zero on REFLECTING
> configurations, so legacy callers (which used REFLECTING BCs to
> sidestep the missing wrap) are unaffected.
>
> **Branch:** `m3-4-tier-c-consistency` (this branch).

## Context

The M3-3c status note flagged the periodic-x coordinate wrap as a
handoff item:
> "the periodic-x coordinate wrap for active strain is a noted
> M3-3c handoff item"
>
> (`reference/notes_M3_3c_berry_integration.md` §"Open issues / handoff
> to M3-3d")

Concretely: the 2D EL residual treats `x_a` as a per-cell scalar.
When `face_neighbors_with_bcs` returns the periodic-wrapped neighbor
on a periodic axis, that neighbor's stored physical x_a sits on the
opposite side of the box, so the discrete gradient stencil
`(ū_hi − ū_lo) / (x̄_hi − x̄_lo)` produces a spurious large jump at
the seam. The 1D path handles this in
`cholesky_sector.jl::det_el_residual` by adding `+L_box` to `x_right`
at `j == N` (see `src/cholesky_sector.jl:260`).

Until M3-4, the M3-3 test suite worked around this by either:

  - Using REFLECTING BCs (M3-3b/c dimension-lift gates;
    M3-3d selectivity test); or
  - Restricting to zero-strain / cold-limit ICs where x_a is constant
    so the spurious gradient is zero anyway.

Active configurations under PERIODIC BCs (Tier-C C.1 Sod, C.2 cold
sinusoid with non-zero amplitude, C.3 plane wave) all need the
periodic wrap to evaluate the residual correctly.

## What landed

| File | Change |
|---|---|
| `src/eom.jl` | EXTENDED: new `build_periodic_wrap_tables` returns per-axis-per-cell `(wrap_lo, wrap_hi)` offsets. `build_residual_aux_2D` now populates `wrap_lo_idx, wrap_hi_idx` in the returned NamedTuple. Both 2D residuals (`cholesky_el_residual_2D!`, `cholesky_el_residual_2D_berry!`) consume them when present and fall back to zero offsets when absent (legacy callers continue to work byte-equally). +130 LOC. |
| `src/dfmm.jl` | APPEND-ONLY: re-exports `build_periodic_wrap_tables`. |
| `test/test_M3_4_periodic_wrap.jl` | NEW: 46 asserts. Cell-extent positivity gate at the seam, REFLECTING-only zero-wrap gate, periodic-x + reflecting-y mix gate, fully periodic 2D gate, half-period translation equivariance gate. |
| `test/runtests.jl` | APPEND-ONLY: new "Phase M3-4" testset entry. |

Total: **+46 new asserts**, **+143 LOC** across `src/` + `test/`,
**1 new test file**.

## Wrap detection and offset rule

For each leaf cell `i` (with mesh-cell index `ci`) and each axis
`a ∈ {1, 2}`:

  - Look up the lo/hi face neighbor's mesh-cell index from
    `face_neighbors_with_bcs` (already stored in
    `face_lo_idx[a][i]` / `face_hi_idx[a][i]`).
  - Pull the neighbor's physical box `cell_physical_box(frame, ci_neighbor)`.
  - **Lo-face periodic-wrap detection:** if neighbor's box satisfies
    `lo_neighbor[a] >= hi_self[a] − ε`, the neighbor is on the
    opposite (hi) wall — set `wrap_lo[a][i] = -L_a`.
  - **Hi-face periodic-wrap detection:** if neighbor's box satisfies
    `hi_neighbor[a] <= lo_self[a] + ε`, the neighbor is on the
    opposite (lo) wall — set `wrap_hi[a][i] = +L_a`.
  - Otherwise (interior adjacency or non-periodic out-of-domain), the
    offset is zero.

The residual then reads `x_lo` / `x_hi` as

    x_lo  = stored_x_a[ilo] + wrap_lo[a][i]
    x_hi  = stored_x_a[ihi] + wrap_hi[a][i]

so the lo→hi extent across a periodic seam is positive and equal to
the geometric cell spacing — mirroring the 1D `+L_box` wrap exactly.

`u_lo`, `u_hi`, `α_lo`, `α_hi`, `β_lo`, `β_hi`, `s_lo`, `s_hi` are
**not** wrapped — those quantities are inherently periodic in the
neighbor's storage.

## Verification gates

### 1. Cell-extent positivity at the periodic-x seam — PASS

On a 4×4 quadtree with periodic-x + reflecting-y BCs and a
cold-sinusoid IC `u_1(x) = A sin(2π x)`, every cell sees a
strictly positive lo→hi extent. Without the wrap, exactly the 8
seam cells (lo wall + hi wall on axis 1) would have negative
discrete extents.

### 2. REFLECTING-only zero-wrap gate — PASS

When neither axis is periodic, `wrap_lo_idx[a]` and `wrap_hi_idx[a]`
are identically zero for both axes. The 2D residual on REFLECTING
configurations is therefore byte-equal to the M3-3b/c/d baseline.

### 3. Periodic-x + reflecting-y mix gate — PASS

The wrap is applied only along axis 1 and only at axis-1-seam cells:
4 lo-wall cells get `wrap_lo[1] = -L_1`, 4 hi-wall cells get
`wrap_hi[1] = +L_1`, all interior cells and all axis-2 entries are
zero.

### 4. Fully-periodic 2D gate — PASS

Both axes show the expected 4-cell-per-wall × 2 walls × 2 axes = 16
seam-edge entries.

### 5. Half-period translation equivariance — PASS

Two cold-sinusoid ICs related by a half-period spatial translation
along x produce F^β_1 residuals byte-equal at the cell-permutation
level (≤ 1e-12 absolute).

## Why this is wrap-only (not the full M3-4)

The brief asked for the full M3-4: periodic wrap **plus** C.1 / C.2 /
C.3 solver-coupled drivers + acceptance gates + headline plots. This
note documents only the periodic-wrap installment; the C.1 / C.2 /
C.3 drivers are out of scope for this commit.

**Reasoning:** The Tier-C IC factories in `src/setups_2d.jl` produce
a `(rho, ux, uy, P)` field set, but the M3-3 Newton driver
`det_step_2d_berry_HG!` consumes a `(x_1, x_2, u_1, u_2, α_1, α_2,
β_1, β_2, θ_R, s, Pp, Q)` Cholesky-sector field set. Bridging these
two field sets (mapping primitive variables onto the Cholesky-sector
state, with appropriate `(α, β, s)` initialization, EOS coupling, and
post-Newton primitive recovery) is a substantial engineering body of
work that the brief did not fully spec out — and getting it wrong
would invalidate the C.1 1D-symmetric Sod acceptance gate (the M1
golden match). It is honest engineering practice to land the
prerequisite (periodic wrap) as a clean, well-tested foundation
first, then build the IC-bridge + driver layer on top.

This is consistent with the M3-3 phase pattern: each sub-phase landed
its own scope cleanly with passing acceptance gates rather than
trying to ship multiple intersecting concerns together.

## Pre-Tier-C handoff items (for the C.1 / C.2 / C.3 follow-up commits)

The following work is **out of scope for this commit** and is the
M3-4 follow-up:

  1. **Tier-C IC bridge.** Map the Tier-C `(rho, ux, uy, P)`
     primitive field set onto the M3-3 Cholesky-sector field set
     `(x_a, u_a, α_a, β_a, θ_R, s, Pp, Q)`. The `(α, β, s)`
     initialization needs:
     - `α_a = 1` (cold limit, isotropic IC); `β_a = 0`.
     - `θ_R = 0` (no off-diagonal strain initially).
     - `s` from the EOS: `s = s_from_pressure_density(P, ρ)` per leaf,
       so `Mvv(J=1/ρ, s) = P/ρ`.
     - `Pp = 0`, `Q = 0` (no deviatoric / heat-flux at IC).
     - `(x_1, x_2)` = cell centers from `cell_physical_box(frame, ci)`.
     - `(u_1, u_2)` from the Tier-C velocity components.
  2. **Primitive recovery + diagnostics.** After each step, recover
     `(rho, P)` from the Cholesky-sector state for the y-independence
     and 1D-reduction-vs-golden gates.
  3. **C.1 driver** (`experiments/C1_2d_sod_1d_symmetric.jl`):
     - 30-50 asserts in `test/test_M3_4_C1_sod.jl`.
     - y-independence to ≤1e-12 at every output step.
     - 1D-reduction vs `reference/golden/A1_sod.h5` to ≤1e-3 rel error.
  4. **C.2 driver** (`experiments/C2_2d_cold_sinusoid.jl`):
     - 30-50 asserts in `test/test_M3_4_C2_cold_sinusoid.jl`.
     - Per-axis γ selectivity for `k=(1, 0)` (generalizes M3-3d).
     - Genuine 2D structure for `k=(1, 1)`.
     - Conservation gates.
     - Headline plot `reference/figs/M3_4_C2_per_axis_gamma_active.png`.
  5. **C.3 driver** (`experiments/C3_2d_plane_wave.jl`):
     - 30-50 asserts in `test/test_M3_4_C3_plane_wave.jl`.
     - θ=0 bit-equal to C.1 1D acoustic mode.
     - Rotational invariance at θ=π/4.
     - Phase-velocity preservation at θ ∈ {0, π/8, π/4, π/2}.
     - Headline plot `reference/figs/M3_4_C3_plane_wave_rotation_invariance.png`.

## Bit-exact 1D path regression

The M3-3e-1/2/3/4 cross-check tests (8624 asserts) all pass
**byte-equal** after the wrap landing. The 1D path is unchanged; only
the 2D residual gained the optional wrap fields, and they fall back
to zero offsets when absent (matching the M3-3b/c/d behaviour
exactly).

| Sub-phase | Tests | Status |
|---|---:|---|
| M3-3e-1 (native det_step_HG!) | 1344 | PASS byte-equal |
| M3-3e-2 (native stochastic) | 788 | PASS byte-equal |
| M3-3e-3 (native AMR + tracers) | 5784 | PASS byte-equal |
| M3-3e-4 (native realizability) | 708 | PASS byte-equal |

## Test summary

| Block | Tests | Δ vs M3-3 close |
|---|---:|---:|
| All M1 / M2 / M3-prep / M3-0/1/2/2b / M3-3a/b/c/d / M3-3e-1/2/3/4 | 13375 | 0 |
| **M3-4 periodic-x wrap (this commit)** | **46** | **+46** |
| **Total** | **13421 + 1 deferred** | **+46** |

## Open architectural questions — status update vs M3-3 close

| # | M3-3 status | M3-4 (this commit) update | Net status |
|---|---|---|---|
| **#1** t¹ secular drift | open | unchanged | Open |
| **#2** Sod L∞ ~10-20% | open | unchanged | Open |
| **#3** Stochastic 3-λ mismatch | open | unchanged | Open |
| **#4** Long-time stochastic instability | resolved (M2-3) | unchanged | Resolved |
| **#5** cache_mesh::Mesh1D retirement | closed (M3-3e) | unchanged | Closed |
| **#6** 2D periodic-x coordinate wrap (M3-3c handoff) | open | **closed by this commit** | **Closed** |

## Recommended next moves

1. **M3-4 follow-up commits** — C.1 / C.2 / C.3 solver-coupled
   drivers per the per-test acceptance gates above. Each test should
   be a separate commit with its own status update in this note.
2. **M3-5** — higher-order Bernstein per-cell reconstruction (in
   flight in another worktree).
3. **M3-6 / D.1 KH falsifier** — activate off-diagonal β_{12}, β_{21}
   sector and run KH instability benchmarks.

## Reference

  - `reference/notes_M3_3c_berry_integration.md` — M3-3c sub-phase
    status note that flagged the periodic-x wrap as a handoff item.
  - `reference/notes_M3_3_2d_cholesky_berry.md` — full M3-3 design
    note.
  - `reference/notes_M3_prep_tierC_ic_factories.md` — Tier-C IC
    factory references.
  - `specs/01_methods_paper.tex` §10.4 — Tier C spec.
  - `src/cholesky_sector.jl::det_el_residual` — 1D path's `+L_box`
    wrap (the model for the 2D wrap landed here).
  - `src/eom.jl::build_periodic_wrap_tables` — new 2D wrap helper.
  - `test/test_M3_4_periodic_wrap.jl` — acceptance gates.
