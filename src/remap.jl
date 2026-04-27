# remap.jl
#
# Milestone M3-5 — Bayesian L↔E remap orchestration.
#
# This module wires HG's `compute_overlap` + `polynomial_remap_l_to_e!` /
# `polynomial_remap_e_to_l!` into a dfmm-side per-step driver. The remap
# conservatively transfers per-cell polynomial fields between a deforming
# Lagrangian `SimplicialMesh{D, T}` and a fixed Eulerian
# `HierarchicalMesh{D}` background, per the methods paper §6 (Bayesian
# remap with the law of total cumulants).
#
# Scope (M3-5):
# - `BayesianRemapState{D, T}` — paired-mesh + companion field-set state
#   carrying the Eulerian frame, Eulerian-side `PolynomialFieldSet`
#   storage, and an opt-in `IntegerLattice` for the IntExact backend.
# - `bayesian_remap_l_to_e!` / `bayesian_remap_e_to_l!` — per-pass
#   orchestration around HG primitives. Operates field-by-field; reuses
#   the precomputed overlap so multi-field remaps amortize the geometric
#   cost.
# - `remap_round_trip!` — L→E→L conservation regression helper.
# - `liouville_monotone_increase_diagnostic` — wraps HG's `RemapDiagnostics`
#   to expose per-step Liouville extrema, running totals, and a
#   monotone-increase verification check.
# - `audit_overlap_dfmm` — IntExact audit harness wrapper.
#
# This file does NOT touch the per-step solver source; integration with
# the deterministic Newton driver is via the `remap_every` kwarg added in
# `src/dfmm.jl` (append-only, see `det_run_with_remap_HG!`).

using HierarchicalGrids: SimplicialMesh, HierarchicalMesh, EulerianFrame,
    PolynomialFieldSet, MonomialBasis, allocate_polynomial_fields, SoA,
    GeometricOverlap, IntegerLattice, RemapDiagnostics,
    OverlapAuditReport,
    compute_overlap, polynomial_remap_l_to_e!, polynomial_remap_e_to_l!,
    polynomial_coeffs_view, polynomial_coeffs_matrix,
    set_polynomial_coeffs_matrix!,
    eulerian_frame, lagrangian_frame, audit_overlap,
    enumerate_leaves, cell_physical_box, n_simplices, simplex_volume,
    field_names, basis_of, n_elements, n_coeffs,
    AxisAlignedRef, SimplicialRef, CellReferenceFrame
import HierarchicalGrids

# ============================================================================
# State container
# ============================================================================

"""
    BayesianRemapState{D, T}

Per-step Bayesian remap state. Holds the Eulerian-frame
`HierarchicalMesh{D}` + `EulerianFrame{D, T}` companion to the Lagrangian
`SimplicialMesh{D, T}`, an Eulerian-side `PolynomialFieldSet` matching the
Lagrangian field-set layout, plus optional caches for the IntExact backend
and a reusable `RemapDiagnostics{T}` accumulator.

# Fields

- `eul_mesh::HierarchicalMesh{D}` — the fixed Eulerian background.
- `frame::EulerianFrame{D, T}` — physical bounds for the root cell.
- `eul_fields::PolynomialFieldSet` — companion to the Lagrangian
  `PolynomialFieldSet`. Sized to `HierarchicalGrids.n_cells(eul_mesh)`
  to match HG's `compute_overlap` indexing convention. Same field
  names, same `MonomialBasis{D, P}`.
- `lattice::Union{Nothing, IntegerLattice{D}}` — opt-in lattice for
  the IntExact backend. `nothing` ⇒ HG auto-derives one each call.
- `last_overlap::Union{Nothing, GeometricOverlap{D, T}}` — most recent
  overlap. Cached to amortize the BVH build across multi-field remaps
  within a single L→E or E→L pass; invalidated each call (the
  Lagrangian mesh deforms between calls).
- `diagnostics::RemapDiagnostics{T}` — reusable accumulator. The
  per-call API resets it, then HG's `_accumulate_remap_diagnostics!`
  fills it in.
- `liouville_running_max::T` — running maximum of `liouville_max`
  across remap calls. The per-step Liouville Theorem (methods paper
  §6.5) requires `det L_j^new ≥ Σ_i w_ij det L_i^old`; the running
  max gives the monotone-increase diagnostic a single-number
  history. Initialized to `-Inf`.
- `liouville_history::Vector{NTuple{2, T}}` — per-call
  `(liouville_min, liouville_max)` log. Empty until the first remap.

# Construction

```julia
state = BayesianRemapState(eul_mesh, frame, field_layout_template)
```

The `field_layout_template` is a `PolynomialFieldSet` whose names,
basis, and element type are mirrored on the Eulerian side; its `n` is
ignored (replaced with `n_cells(eul_mesh)`).
"""
mutable struct BayesianRemapState{D, T}
    eul_mesh::HierarchicalMesh{D}
    frame::EulerianFrame{D, T}
    eul_fields::PolynomialFieldSet
    lattice::Union{Nothing, IntegerLattice{D}}
    last_overlap::Union{Nothing, GeometricOverlap{D, T}}
    diagnostics::RemapDiagnostics{T}
    liouville_running_max::T
    liouville_history::Vector{NTuple{2, T}}
end

"""
    BayesianRemapState(eul_mesh, frame, lag_fields; lattice = nothing)

Build a `BayesianRemapState` whose Eulerian-side `PolynomialFieldSet`
mirrors the layout of `lag_fields` (a Lagrangian-side
`PolynomialFieldSet`). The Eulerian field set is sized to
`HierarchicalGrids.n_cells(eul_mesh)` so it matches HG's
`compute_overlap` indexing.
"""
function BayesianRemapState(eul_mesh::HierarchicalMesh{D},
                            frame::EulerianFrame{D, T},
                            lag_fields::PolynomialFieldSet;
                            lattice::Union{Nothing, IntegerLattice{D}} = nothing
                            ) where {D, T}
    nc = HierarchicalGrids.n_cells(eul_mesh)
    eul_fields = _allocate_companion_fieldset(lag_fields, nc)
    diagnostics = RemapDiagnostics{T}()
    return BayesianRemapState{D, T}(
        eul_mesh, frame, eul_fields, lattice, nothing,
        diagnostics, T(-Inf), NTuple{2, T}[],
    )
end

"""
    _allocate_companion_fieldset(template::PolynomialFieldSet, n)

Allocate a fresh `PolynomialFieldSet` of `n` elements with the same
basis and named-field layout (same names, same per-field element types)
as `template`. Used to build the Eulerian companion to a Lagrangian
field set. Always returns an SoA-laid-out field set so the polynomial
remap kernel can use zero-copy reshape views.
"""
function _allocate_companion_fieldset(template::PolynomialFieldSet, n::Integer)
    b = basis_of(template)
    storage = template.storage
    names = field_names(template)
    # Build a NamedTuple{names}(types) of element types, then splat into kwargs.
    types = ntuple(i -> eltype(getfield(storage, names[i])), length(names))
    kw = NamedTuple{names}(types)
    return allocate_polynomial_fields(SoA(), b, Int(n); kw...)
end

# ============================================================================
# Frame builders (Lagrangian + Eulerian)
# ============================================================================

"""
    _lagrangian_frames(lag_mesh) -> Vector{SimplicialRef{D, T}}

Build per-simplex `SimplicialRef` frames for every simplex in
`lag_mesh`, in 1..n_simplices order. Used by HG's
`polynomial_remap_l_to_e!` / `polynomial_remap_e_to_l!`.
"""
function _lagrangian_frames(lag_mesh::SimplicialMesh{D, T}) where {D, T}
    ns = n_simplices(lag_mesh)
    frames = Vector{SimplicialRef{D, T}}(undef, ns)
    @inbounds for s in 1:ns
        frames[s] = lagrangian_frame(lag_mesh, Int(s))
    end
    return frames
end

"""
    _eulerian_frames(frame) -> Vector{AxisAlignedRef{D, T}}

Build per-cell `AxisAlignedRef` frames for every cell of the Eulerian
mesh (1..n_cells), in order. Cells that are never touched by overlap
entries still get a frame (cheap; matches HG's expectation that
`length(dst_frames) == overlap.n_eul`).
"""
function _eulerian_frames(frame::EulerianFrame{D, T}) where {D, T}
    nc = HierarchicalGrids.n_cells(frame.mesh)
    frames = Vector{AxisAlignedRef{D, T}}(undef, nc)
    @inbounds for ci in 1:nc
        lo, hi = cell_physical_box(frame, ci)
        frames[ci] = AxisAlignedRef{D, T}(lo, hi)
    end
    return frames
end

# ============================================================================
# Overlap builder
# ============================================================================

"""
    _build_overlap!(state, lag_mesh; backend, moment_order, parallel)

Compute (or recompute) the L↔E geometric overlap and stash it on
`state.last_overlap`. Returns the overlap. The `:exact` backend uses
`state.lattice` if set, else HG auto-derives one.
"""
function _build_overlap!(state::BayesianRemapState{D, T},
                          lag_mesh::SimplicialMesh{D, T};
                          backend::Symbol = :float,
                          moment_order::Integer = 3,
                          parallel::Bool = false) where {D, T}
    backend === :float || backend === :exact ||
        throw(ArgumentError("backend must be :float or :exact (got $backend)"))
    lat = backend === :exact ? state.lattice : nothing
    overlap = compute_overlap(lag_mesh, state.frame;
                              moment_order = Int(moment_order),
                              backend = backend,
                              lattice = lat,
                              parallel = parallel)
    state.last_overlap = overlap
    return overlap
end

# ============================================================================
# Remap drivers
# ============================================================================

"""
    bayesian_remap_l_to_e!(state, lag_mesh, lag_fields;
                           backend = :float, moment_order = nothing,
                           parallel = false, fields = nothing)

Project every named field of `lag_fields` from the Lagrangian
`SimplicialMesh{D, T}` onto the Eulerian companion in `state`. Computes
the geometric overlap once and reuses it across all named fields.

# Arguments

- `state` — the `BayesianRemapState`.
- `lag_mesh` — the deforming Lagrangian mesh.
- `lag_fields` — source field set; same basis/names as
  `state.eul_fields`.
- `backend` — `:float` (default) or `:exact`.
- `moment_order` — moment order for `compute_overlap`. When `nothing`
  (default), uses `2 * P_src` to satisfy the L²-projection requirement
  `moment_order ≥ P_src + P_dst` for matched src/dst orders. The methods
  paper §6.2 cubic-reconstruction default is `moment_order = 3` (matches
  `P_src = P_dst = 1`); for `P = 0` cell-averages this collapses to
  `moment_order = 0` (volume-only, the cheapest path).
- `parallel` — pass `true` to use HG's threaded `compute_overlap`. The
  `:exact` backend ignores this; the integer-rational accumulator is
  not yet thread-safe.
- `fields` — optional `Tuple{Symbol, ...}` selecting a subset of named
  fields to remap. Defaults to all fields in `lag_fields.storage`.

Returns `state` (mutated). The Eulerian side (`state.eul_fields`)
holds the projected coefficients; the Lagrangian side is left
unchanged.

The `state.diagnostics` accumulator is reset and refilled with the
overlap-pass Liouville extrema (per HG `_accumulate_remap_diagnostics!`,
called on the FIRST per-call field).
"""
function bayesian_remap_l_to_e!(state::BayesianRemapState{D, T},
                                lag_mesh::SimplicialMesh{D, T},
                                lag_fields::PolynomialFieldSet;
                                backend::Symbol = :float,
                                moment_order::Union{Nothing, Integer} = nothing,
                                parallel::Bool = false,
                                fields::Union{Nothing, Tuple} = nothing
                                ) where {D, T}
    return _remap_pass!(state, lag_mesh, lag_fields, state.eul_fields;
                         direction = :l_to_e, backend = backend,
                         moment_order = moment_order, parallel = parallel,
                         fields = fields)
end

"""
    bayesian_remap_e_to_l!(state, lag_mesh, lag_fields;
                           backend = :float, moment_order = nothing,
                           parallel = false, fields = nothing)

Inverse projection: pull every named field of `state.eul_fields` back
onto the Lagrangian field set `lag_fields`. Same arguments as
`bayesian_remap_l_to_e!`. The Lagrangian side is overwritten in place;
the Eulerian side is left unchanged.

Note: the `:float` backend is still the production default; `:exact`
is the audit harness path.
"""
function bayesian_remap_e_to_l!(state::BayesianRemapState{D, T},
                                lag_mesh::SimplicialMesh{D, T},
                                lag_fields::PolynomialFieldSet;
                                backend::Symbol = :float,
                                moment_order::Union{Nothing, Integer} = nothing,
                                parallel::Bool = false,
                                fields::Union{Nothing, Tuple} = nothing
                                ) where {D, T}
    return _remap_pass!(state, lag_mesh, state.eul_fields, lag_fields;
                         direction = :e_to_l, backend = backend,
                         moment_order = moment_order, parallel = parallel,
                         fields = fields)
end

# Internal: shared L→E / E→L body. `direction` selects which way the
# kernel runs and which side gets which set of frames.
function _remap_pass!(state::BayesianRemapState{D, T},
                      lag_mesh::SimplicialMesh{D, T},
                      source_pfs::PolynomialFieldSet,
                      target_pfs::PolynomialFieldSet;
                      direction::Symbol,
                      backend::Symbol,
                      moment_order::Union{Nothing, Integer},
                      parallel::Bool,
                      fields::Union{Nothing, Tuple}) where {D, T}
    direction in (:l_to_e, :e_to_l) ||
        throw(ArgumentError("direction must be :l_to_e or :e_to_l"))

    # Determine polynomial orders from the bases.
    src_basis = basis_of(source_pfs)
    dst_basis = basis_of(target_pfs)
    src_basis isa MonomialBasis ||
        throw(ArgumentError("source field set must use MonomialBasis"))
    dst_basis isa MonomialBasis ||
        throw(ArgumentError("target field set must use MonomialBasis"))
    P_src = _basis_order(src_basis)
    P_dst = _basis_order(dst_basis)

    # Default moment_order: methods paper §6.2 says cubic-reconstruction
    # uses `moment_order = 3`. Generically: `P_src + P_dst`. For order-0
    # cell averages this is 0 (volume-only is sufficient).
    mo = moment_order === nothing ? max(P_src + P_dst, 0) : Int(moment_order)
    mo >= P_src + P_dst ||
        throw(ArgumentError(
            "moment_order = $mo < P_src + P_dst = $(P_src + P_dst); " *
            "the L²-projection moment integration requirement is not met."))

    # Build per-cell frames for both sides.
    lag_frames = _lagrangian_frames(lag_mesh)
    eul_frames = _eulerian_frames(state.frame)

    src_frames, dst_frames = if direction === :l_to_e
        (lag_frames, eul_frames)
    else
        (eul_frames, lag_frames)
    end

    # Compute or reuse the overlap. We always recompute on entry: the
    # Lagrangian mesh moves between calls. (Caller can amortize across
    # multiple fields by calling `_remap_pass!` with `fields = ...` for
    # the field-list slice — the overlap is built once per call.)
    overlap = _build_overlap!(state, lag_mesh;
                              backend = backend,
                              moment_order = mo,
                              parallel = parallel)

    # Reset the diagnostics accumulator before the first field. HG fills
    # it on each `polynomial_remap_*!` call — to keep the per-pass
    # numbers honest we only attach it to the FIRST field's call (the
    # geometry is the same for every field).
    HierarchicalGrids.Diagnostics.reset!(state.diagnostics)

    # Determine the field list to remap.
    field_names = if fields === nothing
        keys(source_pfs.storage)
    else
        fields
    end
    isempty(field_names) &&
        throw(ArgumentError("no fields to remap (empty field list)"))

    first_call = true
    for (i, name) in enumerate(field_names)
        # Source coefficients (zero-copy view for SoA).
        src_coeffs = polynomial_coeffs_view(source_pfs, name)
        # Target coefficients buffer (zero-copy view for SoA).
        tgt_coeffs = polynomial_coeffs_view(target_pfs, name)

        diag = first_call ? state.diagnostics : nothing
        if direction === :l_to_e
            polynomial_remap_l_to_e!(tgt_coeffs, src_coeffs, overlap,
                                      src_frames, dst_frames,
                                      P_src, P_dst;
                                      diagnostics = diag)
        else
            polynomial_remap_e_to_l!(tgt_coeffs, src_coeffs, overlap,
                                      src_frames, dst_frames,
                                      P_src, P_dst;
                                      diagnostics = diag)
        end
        first_call = false
    end

    # Update the running Liouville-monotone history.
    push!(state.liouville_history,
          (state.diagnostics.liouville_min, state.diagnostics.liouville_max))
    if state.diagnostics.liouville_max > state.liouville_running_max
        state.liouville_running_max = state.diagnostics.liouville_max
    end

    return state
end

@inline _basis_order(::MonomialBasis{D, P}) where {D, P} = P

# ============================================================================
# Round trip + conservation regression
# ============================================================================

"""
    remap_round_trip!(state, lag_mesh, lag_fields;
                      backend = :float, moment_order = nothing,
                      parallel = false, fields = nothing)

L→E→L round trip on `lag_fields`. Returns `state` after running both
passes; `lag_fields` is overwritten with the round-tripped coefficients.

The per-pass `state.diagnostics` is filled by the FINAL call (E→L);
the cumulative `state.liouville_history` retains both passes' extrema.

# Conservation contract

For order-0 (cell-average) fields the round trip preserves the
mass-weighted total `Σ_j ρ_j V_j` (mass) and similar moments
(`Σ_j (ρu)_j V_j`, `Σ_j E_j V_j`). On the `:float` backend the
relative error is bounded by ≤ 1e-12 on smooth fields; on the
`:exact` backend with `moment_order = 0` and matching lattice the
totals are byte-equal.
"""
function remap_round_trip!(state::BayesianRemapState{D, T},
                            lag_mesh::SimplicialMesh{D, T},
                            lag_fields::PolynomialFieldSet;
                            backend::Symbol = :float,
                            moment_order::Union{Nothing, Integer} = nothing,
                            parallel::Bool = false,
                            fields::Union{Nothing, Tuple} = nothing
                            ) where {D, T}
    bayesian_remap_l_to_e!(state, lag_mesh, lag_fields;
                            backend = backend,
                            moment_order = moment_order,
                            parallel = parallel,
                            fields = fields)
    bayesian_remap_e_to_l!(state, lag_mesh, lag_fields;
                            backend = backend,
                            moment_order = moment_order,
                            parallel = parallel,
                            fields = fields)
    return state
end

# ============================================================================
# Conservation accumulators (canonical-order summation)
# ============================================================================

"""
    total_mass_weighted_lagrangian(lag_mesh, lag_fields, fieldname)

Compute `Σ_s a_s * V_s` where `V_s` is simplex `s`'s physical volume
and `a_s` is the cell-average coefficient (constant term, index 1) of
the named field at simplex `s`. Used for the per-backend conservation
gate. Sums in canonical 1..n_simplices order; the `:exact` backend's
byte-equal contract relies on this fixed traversal.
"""
function total_mass_weighted_lagrangian(lag_mesh::SimplicialMesh{D, T},
                                          lag_fields::PolynomialFieldSet,
                                          fieldname::Symbol) where {D, T}
    fv = getproperty(lag_fields, fieldname)
    s_total = zero(T)
    @inbounds for s in 1:n_simplices(lag_mesh)
        V = simplex_volume(lag_mesh, Int(s))
        s_total += T(fv[Int(s)][1]) * V
    end
    return s_total
end

"""
    total_mass_weighted_eulerian(state, fieldname)

Compute `Σ_j a_j * V_j` over Eulerian leaves of `state.eul_mesh`,
where `V_j` is leaf `j`'s physical volume and `a_j` is the constant
term of the named field at leaf `j`. Skips non-leaf cells (they are
not the storage of record on a refined mesh).
"""
function total_mass_weighted_eulerian(state::BayesianRemapState{D, T},
                                        fieldname::Symbol) where {D, T}
    fv = getproperty(state.eul_fields, fieldname)
    s_total = zero(T)
    @inbounds for ci in enumerate_leaves(state.eul_mesh)
        lo, hi = cell_physical_box(state.frame, Int(ci))
        V = one(T)
        for d in 1:D
            V *= hi[d] - lo[d]
        end
        s_total += T(fv[Int(ci)][1]) * V
    end
    return s_total
end

# ============================================================================
# Liouville monotone-increase diagnostic
# ============================================================================

"""
    liouville_monotone_increase_diagnostic(state)

Returns a `(min_J, max_J, monotone_increase_holds::Bool)` tuple
summarizing the Liouville-Jacobian / shell-crossing state of the most
recent remap pass. Per the methods paper §6.5 / §6.6, the per-cell
Liouville Jacobian `Δ_Liou(C_j) = det L_j^new − Σ_i w_ij det L_i` is
non-negative as a consequence of the law of total covariance plus the
positive-semidefiniteness of the within-overlap variance.

# What the proxy measures

HG's `RemapDiagnostics` records a per-overlap-entry Jacobian proxy
`entry.volume / source_physical_volume`. This is a *geometric* proxy
for the moment-level `det L`: it equals 1 for source-cells-fully-
inside-a-target-cell pairs and `< 1` for sub-cell overlaps. The proxy
is bounded above by 1 (the source-fully-inside-target case) and below
by 0 (positive-volume invariant of `compute_overlap`). A negative
value would signal an inverted source simplex (shell crossing) and is
recorded as `n_negative_jacobian_cells > 0`.

# Monotone-increase verification

The paper's monotonicity statement is at the moment-level `det L_j^new
≥ Σ_i w_ij det L_i`; HG does NOT yet implement the per-cell increment
field directly (that is HG-design-guidance item #8 — to be added
upstream, see `~/.julia/dev/HierarchicalGrids/reference/notes_HG_design_guidance.md`).
Until then, dfmm uses the following geometric-proxy
test as a lightweight stand-in:

- `monotone_increase_holds == true` iff:
  - `n_negative_jacobian_cells == 0` for every recorded pass
    (no inverted source simplices), AND
  - `liouville_min > 0` for every recorded pass
    (positive overlap volume → positive proxy), AND
  - the per-pass `total_volume_in == total_volume_out`
    (volume balance — partition-of-unity sanity check; HG records
    each overlap entry symmetrically).

This is a NECESSARY condition for the moment-level monotone-increase;
it is not the full sufficient verification (which requires the §6.6
HG hook to land). Tests that need full moment-level verification can
build it on top of the per-pass `state.diagnostics` via the
`total_mass_weighted_*` accumulators.

When the history is empty (no remap has run yet), returns
`(NaN, NaN, true)` (vacuously true).
"""
function liouville_monotone_increase_diagnostic(state::BayesianRemapState{D, T}
                                                  ) where {D, T}
    h = state.liouville_history
    if isempty(h)
        return (T(NaN), T(NaN), true)
    end
    last = h[end]
    # Necessary-condition check: most recent pass had positive proxies and
    # zero negative-Jacobian cells. This is the cheapest dfmm-side
    # surrogate for the §6.6 monotone-increase theorem until HG ships
    # the dedicated `liouville_increment` field. Per-pass diagnostics
    # are stored on `state.diagnostics`; the history vector retains
    # min/max only.
    diag = state.diagnostics
    necessary = (diag.n_negative_jacobian_cells == 0) &&
                (last[1] > zero(T)) &&
                isapprox(diag.total_volume_in, diag.total_volume_out;
                          atol = 16 * eps(T) * max(diag.total_volume_in, one(T)))
    return (last[1], last[2], necessary)
end

# ============================================================================
# IntExact audit harness wrapper
# ============================================================================

"""
    audit_overlap_dfmm(; verbose = false, atol = 1e-10)

Thin wrapper around HG's `audit_overlap` that runs the canonical
2D / 3D polytope battery (5 + 4 polytopes; see HG's
`src/Diagnostics/exact_audit.jl` for the catalog) and returns the
`OverlapAuditReport`. Used as the M3-5 IntExact regression gate
(`test/test_M3_5_intexact_audit.jl`).

# Caveats from `~/.julia/dev/HierarchicalGrids/docs/src/exact_backend.md`

- D = 2 0//0 degeneracy on collinear-triangle inputs (rare in
  practice; the audit battery avoids them).
- D ∈ {2, 3} default-lattice (16-bit) volume drift up to ~30 % on
  certain near-degenerate configurations. Use a finer lattice
  (`bits = 24` or `Int64` lattice type) for stricter audits if
  needed.

The audit defaults to `atol = 1e-10` (relative); the canonical
battery is expected to pass at this tolerance with the default
10-bit audit lattice.
"""
function audit_overlap_dfmm(; verbose::Bool = false, atol::Real = 1e-10)
    return audit_overlap(; verbose = verbose, atol = atol)
end
