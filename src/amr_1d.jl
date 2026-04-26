# amr_1d.jl
#
# Phase M2-1 — Tier B.3 1D adaptive mesh refinement on the Lagrangian
# segment mesh.
#
# Provides three primitives + two indicators:
#   refine_segment!(mesh, j[, tracers])     splits segment j in two.
#   coarsen_segment_pair!(mesh, j[, tracers]) merges segments j and j+1.
#   action_error_indicator(mesh)            per-segment ΔS_cell proxy.
#   gradient_indicator(mesh; field)         per-segment |∇ρ|/ρ etc.
#   amr_step!(mesh, indicator, τ_refine[, τ_coarsen]; tracers, max_depth)
#                                           applies indicator and
#                                           performs refine/coarsen
#                                           events.
#
# Conservation properties (verified by tests):
#   1. Mass per merged-or-split cell preserved bit-exactly.
#   2. Momentum preserved exactly (Σ m̄_i u_i invariant under refine
#      and coarsen).
#   3. Tracer fields:
#        * refine: bit-exact daughter copy (the parent tracer value).
#        * coarsen: mass-weighted average — conservative but not
#          bit-exact (per the methods paper §6.5 documented entropy
#          increase).
#   4. Energy: refine is exactly conservative (linear-interpolated
#      mid-vertex velocity preserves KE exactly when u is linear in
#      mass coordinate; otherwise sub-grid kinetic energy is lost
#      analogous to a Bayesian remap). Coarsen produces a Liouville-
#      monotone KE *decrease* (mass-weighted velocity averaging),
#      compensated by entropy increase (law of total covariance).
#
# Variable-Δm support. The Phase-2 `Mesh1D(positions, velocities, …;
# Δm = Δm_vec)` constructor already accepts a per-segment Δm vector;
# `det_step!` reads `seg.Δm` per segment, so no integrator change is
# needed. The new code below mutates `mesh.segments` and `mesh.p_half`
# in place to add or remove segments while preserving the periodic
# topology.
#
# Indicators. The action-error indicator is the per-segment EL
# residual produced by a Newton-converged step
# (`det_el_residual` evaluated at the current state pair) plus the
# proxy `|D_t^{(0)}α - β| + |D_t^{(1)}β - γ²/α|` from cholesky_sector.
# The gradient indicator is centered |Δρ|/(2ρ̄) (for `:rho`) or
# |Δu|/c_s (for `:u`).

using StaticArrays: SVector

# ──────────────────────────────────────────────────────────────────────
# Refine
# ──────────────────────────────────────────────────────────────────────

"""
    refine_segment!(mesh::Mesh1D, j::Integer; tracers = nothing)
        -> Mesh1D

Split segment `j` into two equal-mass daughter segments at the
mid-mass-coordinate. Modifies the mesh in place: `n_segments` grows
by 1 and `mesh.p_half` gains one entry.

Conservation:
  * Mass: bit-exact (each daughter gets `Δm/2`).
  * Momentum: exact (the inserted mid-vertex velocity is the
    arithmetic mean of the parent's left and right vertex velocities;
    vertex masses redistribute symmetrically so total `Σ m̄_i u_i`
    is unchanged).
  * Tracers (if `tracers::TracerMesh` supplied): both daughters
    receive *bit-identical* copies of the parent's tracer value
    (Phase-11 exactness preserved).

The daughters' inherited per-segment Cholesky / EOS state:
  * `α, β, s, Pp, Q`: copied from parent (uniform within parent
    under the M1 first-order reconstruction).

Vertex positions: parent's left vertex stays put. The inserted mid
vertex sits at `x_left + Δm_parent/(2·ρ_parent)` — i.e. the
geometric midpoint in mass-coordinate, which equals the Eulerian
midpoint when ρ is constant in the parent (true under the constant
reconstruction).
"""
function refine_segment!(mesh::Mesh1D{T,DetField{T}}, j::Integer;
                         tracers::Union{Nothing,TracerMesh} = nothing) where {T<:Real}
    N = n_segments(mesh)
    @assert 1 ≤ j ≤ N "refine: segment index $j out of range 1:$N"
    @assert N + 1 ≥ 2

    parent_seg = mesh.segments[j]
    parent_state = parent_seg.state
    Δm_parent = parent_seg.Δm
    Δm_half = Δm_parent / 2

    # Mid-vertex position: parent's left vertex + half its Eulerian
    # extent. Compute Δx via segment_length to get the periodic-wrap
    # right vertex.
    Δx_parent = segment_length(mesh, j)
    x_mid = parent_state.x + Δx_parent / 2

    # Right-vertex velocity: the segment's right vertex is the next
    # segment's left vertex (periodic).
    j_right = j == N ? 1 : j + 1
    u_right = mesh.segments[j_right].state.u
    u_mid = (parent_state.u + u_right) / 2  # linear interpolation, momentum-conserving

    # Build the two daughters. Both inherit α, β, s, Pp, Q exactly.
    left_state = DetField{T}(
        parent_state.x, parent_state.u,           # left vertex unchanged
        parent_state.α, parent_state.β,
        parent_state.s, parent_state.Pp, parent_state.Q,
    )
    right_state = DetField{T}(
        x_mid, u_mid,                              # new mid vertex
        parent_state.α, parent_state.β,
        parent_state.s, parent_state.Pp, parent_state.Q,
    )
    # Mass-coordinate centers: parent.m is the cell center; daughters
    # have centers at parent.m ± Δm_half/2.
    m_center_left  = parent_seg.m - Δm_half / 2
    m_center_right = parent_seg.m + Δm_half / 2

    daughter_left  = Segment{T,DetField{T}}(m_center_left,  Δm_half, left_state)
    daughter_right = Segment{T,DetField{T}}(m_center_right, Δm_half, right_state)

    # Splice into the segment vector: replace position j with daughter_left
    # and insert daughter_right at j+1.
    mesh.segments[j] = daughter_left
    insert!(mesh.segments, j + 1, daughter_right)

    # p_half: vertex-indexed (one entry per segment, equal to N after
    # the splice). Recompute from scratch — cheap and avoids subtle
    # bugs in periodic indexing under topology changes.
    refresh_p_half!(mesh)

    # Tracers: bit-exact daughter copy.
    if tracers !== nothing
        refine_tracer!(tracers, j)
    end

    return mesh
end

# Coarsen ──────────────────────────────────────────────────────────────

"""
    coarsen_segment_pair!(mesh::Mesh1D, j::Integer; tracers = nothing)
        -> Mesh1D

Merge segments `j` and `j+1` into a single segment of mass
`Δm_j + Δm_{j+1}`. Modifies the mesh in place: `n_segments` shrinks by
1 and the shared vertex (`mesh.segments[j+1].state.x`) is removed.

Conservation:
  * Mass: bit-exact.
  * Momentum: exact. The disappearing vertex's momentum is
    redistributed onto the two surviving vertices in proportion to
    the masses they newly absorb (vertex `j` absorbs `Δm_{j+1}/2`,
    vertex `j+2` absorbs `Δm_j/2`).
  * Cholesky `(α, β)` use the law-of-total-covariance:
      M_vv,new = (Δm_j M_vv,j + Δm_{j+1} M_vv,{j+1}) / Δm_new
                + Δm_j Δm_{j+1} / Δm_new² · (u_j − u_{j+1})²
      β_new   = (Δm_j β_j + Δm_{j+1} β_{j+1}) / Δm_new
      γ_new   = √(M_vv,new − β_new²)
      s_new   = log(M_vv,new) − (1−Γ) log(J_new)
                where J_new = Δx_new / Δm_new.
    With this scheme γ stays real-valued (intra-cell + between-cell
    variance is non-negative by construction), and the merged P_xx
    equals ρ_new · M_vv,new self-consistently.
  * Tracers: mass-weighted average (conservative; bit-exactness lost).

The "between-cell" velocity for the law-of-total-covariance uses the
cell-center velocity `u_cell = (u_left + u_right)/2`, taking into
account the periodic right vertex.
"""
function coarsen_segment_pair!(mesh::Mesh1D{T,DetField{T}}, j::Integer;
                               tracers::Union{Nothing,TracerMesh} = nothing,
                               Gamma::Real = GAMMA_LAW_DEFAULT) where {T<:Real}
    N = n_segments(mesh)
    @assert 1 ≤ j ≤ N "coarsen: segment index $j out of range 1:$N"
    @assert N ≥ 2 "coarsen: cannot reduce mesh below 1 segment"
    j_next = j == N ? 1 : j + 1
    @assert j != j_next "coarsen: degenerate j == j+1 (N = 1)"

    seg_l = mesh.segments[j]
    seg_r = mesh.segments[j_next]
    state_l = seg_l.state
    state_r = seg_r.state

    Δm_l = seg_l.Δm
    Δm_r = seg_r.Δm
    Δm_new = Δm_l + Δm_r

    # Eulerian lengths (periodic-aware via segment_length helper).
    Δx_l = segment_length(mesh, j)
    Δx_r = segment_length(mesh, j_next)
    Δx_new = Δx_l + Δx_r
    ρ_l = Δm_l / Δx_l
    ρ_r = Δm_r / Δx_r
    J_new = Δx_new / Δm_new

    # Cell-center velocities (between-cell variance source). The right
    # vertex of a segment is the next segment's left vertex, modulo
    # periodic wrap (we don't need the wrap here because `u` is just
    # the velocity scalar, not a position).
    j_after = j_next == N ? 1 : j_next + 1
    u_cell_l = (state_l.u + state_r.u) / 2
    u_cell_r = (state_r.u + mesh.segments[j_after].state.u) / 2

    # Pre-merge M_vv per segment from the EOS.
    M_vv_l = Mvv(Δx_l / Δm_l, state_l.s; Gamma = Gamma)
    M_vv_r = Mvv(Δx_r / Δm_r, state_r.s; Gamma = Gamma)

    # Law of total covariance. Intra-cell variance:
    M_vv_intra = (Δm_l * M_vv_l + Δm_r * M_vv_r) / Δm_new
    # Between-cell variance (mass-weighted bilinear in cell-center
    # velocity differences). The factor Δm_l Δm_r / Δm_new² is the
    # standard total-variance correction for two-population mixing.
    M_vv_between = (Δm_l * Δm_r / Δm_new^2) * (u_cell_l - u_cell_r)^2
    M_vv_new = M_vv_intra + M_vv_between

    # Mass-weighted β (charge-1 bookkeeping; methods paper §6.3).
    β_new = (Δm_l * state_l.β + Δm_r * state_r.β) / Δm_new
    # Realizability: γ² = M_vv − β² must be ≥ 0. Numerically clamp.
    γ²_new = max(M_vv_new - β_new^2, zero(T))
    if M_vv_new - β_new^2 < zero(T)
        # Round-off; absorb the deficit by lifting M_vv just to β² so
        # γ ≥ 0 holds bit-exactly.
        M_vv_new = β_new^2
    end

    # Entropy chosen so that Mvv(J_new, s_new) == M_vv_new exactly.
    # M_vv = J^(1-Γ) exp(s/cv) ⇒ s = cv·(log M_vv − (1−Γ) log J).
    cv = T(CV_DEFAULT)
    s_new = M_vv_new > 0 ?
            cv * (log(M_vv_new) - (1 - Gamma) * log(J_new)) :
            (state_l.s + state_r.s) / 2

    # α: mass-weighted average. (α evolves under D_t^{(0)}α = β, so
    # its merge is the simple linear average — no variance correction
    # because α is charge-0 / "first moment" in this picture.)
    α_new = (Δm_l * state_l.α + Δm_r * state_r.α) / Δm_new

    # Pp (P_⊥) and Q: mass-weighted averages.
    Pp_new = (Δm_l * state_l.Pp + Δm_r * state_r.Pp) / Δm_new
    Q_new  = (Δm_l * state_l.Q  + Δm_r * state_r.Q ) / Δm_new

    # Vertex velocity update (momentum-conserving redistribution of
    # the disappearing vertex j+1's momentum).
    j_left_of_seam = j == 1 ? N : j - 1
    j_right_of_seam = j_next == N ? 1 : j_next + 1

    # m̄ at vertex j, j+1, j+2 (pre-merge cyclic).
    m̄_left_pre  = (mesh.segments[j_left_of_seam].Δm + Δm_l) / 2
    m̄_seam_pre  = (Δm_l + Δm_r) / 2
    m̄_rite_pre  = (Δm_r + mesh.segments[j_right_of_seam].Δm) / 2

    # Post-merge vertex masses (vertex j_left_of_seam is unchanged at
    # its own slot; vertex j_next disappears; vertex j_right_of_seam
    # is unchanged — but the index of the right-of-seam vertex shifts
    # in the array because we're deleting one entry).
    m̄_left_post = (mesh.segments[j_left_of_seam].Δm + Δm_new) / 2
    m̄_rite_post = (Δm_new + mesh.segments[j_right_of_seam].Δm) / 2

    # Vertex-i (segment j's left) absorbs Δm_r/2 of new mass; vertex
    # i+2 (segment j+2's left) absorbs Δm_l/2. Redistribute the lost
    # vertex i+1's momentum proportionally:
    u_seam_pre = state_r.u  # the vertex being eliminated (left vertex of seg_r = right vertex of seg_l)
    u_left_pre = state_l.u
    # Right-of-seam vertex velocity (might be eliminated if N=2; handle
    # the N=2 case below by skipping if j_right_of_seam == j).
    u_rite_pre = mesh.segments[j_right_of_seam].state.u

    # Momentum redistribution (split u_seam_pre's momentum 50/50 by
    # absorbed-mass fractions Δm_r/2 and Δm_l/2):
    p_left_new = m̄_left_pre * u_left_pre + (Δm_r / 2) * u_seam_pre
    p_rite_new = m̄_rite_pre * u_rite_pre + (Δm_l / 2) * u_seam_pre
    u_left_new = p_left_new / m̄_left_post
    u_rite_new = p_rite_new / m̄_rite_post

    # Build the merged segment. Mass-coordinate center: average of
    # parent centers weighted by Δm.
    m_center_new = (Δm_l * seg_l.m + Δm_r * seg_r.m) / Δm_new
    merged_state = DetField{T}(
        state_l.x,            # left vertex of merged = left vertex of seg_l
        u_left_new,           # updated for momentum conservation
        α_new, β_new,
        s_new, Pp_new, Q_new,
    )
    mesh.segments[j] = Segment{T,DetField{T}}(m_center_new, Δm_new, merged_state)

    # Update the right-of-seam vertex's velocity in its own segment.
    if j_right_of_seam != j  # skip if N=2 (right-of-seam wraps back to merged)
        seg_r_of_seam = mesh.segments[j_right_of_seam]
        srs = seg_r_of_seam.state
        new_state_rs = DetField{T}(srs.x, u_rite_new, srs.α, srs.β,
                                    srs.s, srs.Pp, srs.Q)
        mesh.segments[j_right_of_seam] = Segment{T,DetField{T}}(
            seg_r_of_seam.m, seg_r_of_seam.Δm, new_state_rs,
        )
    end

    # Delete segment j+1 (the eliminated one).
    deleteat!(mesh.segments, j_next)

    # Recompute p_half from the now-shrunken vertex set.
    refresh_p_half!(mesh)

    # Tracers: mass-weighted average and delete the second row's column.
    if tracers !== nothing
        coarsen_tracer!(tracers, j; Δm_l = Δm_l, Δm_r = Δm_r)
    end

    return mesh
end

# ──────────────────────────────────────────────────────────────────────
# Indicators
# ──────────────────────────────────────────────────────────────────────

"""
    action_error_indicator(mesh::Mesh1D{T,DetField{T}};
                           dt = nothing) -> Vector{T}

Per-segment ΔS_cell proxy for refinement triggering. Methods paper
§9.7 / design/03_action_note_v2.tex eq:DS-cell. With a constant
reconstruction (Phase-2 default), the leading term reduces to the
discrete EL residual (since the order-(p+1) reconstruction = order-p
reconstruction up to round-off when both are constant). We therefore
use the *normalised* per-segment EL residual

    indicator_j = |D_t^{(0)}α_j - β_j| + |D_t^{(1)}β_j - γ²_j/α_j|

evaluated as a *finite-difference* surrogate: at fixed state (no
timestep info), the indicator collapses to a *stationary* probe
of the local field's smoothness, which we compute as

    |β_{j+1} − 2β_j + β_{j-1}|         (curvature of β in mass)
    + |α_{j+1} − 2α_j + α_{j-1}|       (curvature of α)
    + |s_{j+1} − 2s_j + s_{j-1}|       (curvature of s)

— a discrete second-derivative norm in the segment label, which
flags shocks (large jumps) and shell-crossings (γ → 0 induces
α → ∞ via D_t α = β with growing β/α). This is the *physics-aware*
surrogate of the §9.7 indicator that doesn't require a higher-order
reconstruction (which would require Phase-2 of M2 to implement).

Cells at the boundary use one-sided second differences (periodic
wrap; for `bc = :inflow_outflow` we copy from the nearest interior
neighbour to avoid spurious indicator spikes at the Dirichlet
plane).
"""
function action_error_indicator(mesh::Mesh1D{T,DetField{T}};
                                dt::Union{Real,Nothing} = nothing,
                                Gamma::Real = GAMMA_LAW_DEFAULT) where {T<:Real}
    N = n_segments(mesh)
    out = zeros(T, N)
    if N < 3
        return out
    end
    is_periodic = mesh.bc == :periodic

    # Pre-compute per-segment ρ, M_vv, sound-speed, c_s for the
    # γ-collapse / Mach-curvature term.
    ρ_arr   = T[segment_density(mesh, j) for j in 1:N]
    Mvv_arr = T[Mvv(one(T)/ρ_arr[j], mesh.segments[j].state.s; Gamma = Gamma) for j in 1:N]
    cs_arr  = T[sqrt(max(Gamma * Mvv_arr[j], T(eps(T)))) for j in 1:N]

    @inbounds for j in 1:N
        if is_periodic
            jl = j == 1 ? N : j - 1
            jr = j == N ? 1 : j + 1
        else
            jl = max(j - 1, 1)
            jr = min(j + 1, N)
            if jl == j || jr == j
                continue  # boundary cells: keep indicator at zero
            end
        end
        sl = mesh.segments[jl].state
        sj = mesh.segments[j].state
        sr = mesh.segments[jr].state

        # 2nd-difference (curvature) of the dynamical variables on
        # mass coordinate. Each term is the leading contribution to
        # the discrete EL residual in the corresponding sector.
        d2_α = abs(sr.α - 2*sj.α + sl.α)
        d2_β = abs(sr.β - 2*sj.β + sl.β)
        d2_s = abs(sr.s - 2*sj.s + sl.s)

        # Velocity curvature relative to the local sound speed.
        # This *is* the primary action-residual contribution in
        # smooth-flow regions — the EL momentum equation (∂_t u +
        # ρ⁻¹ ∂_x P_xx) discretised at midpoint has leading-order
        # truncation ∝ ∂²u (3-point) / c_s on a uniform-mass grid.
        d2_u = abs(sr.u - 2*sj.u + sl.u)
        d2_u_norm = d2_u / max(cs_arr[j], T(eps(T)))

        # γ-collapse marker (Hessian degeneracy, methods paper §3).
        # Drops to ~1 when γ² → 0 ⇒ EL residual blows up via
        # `γ²/α` term of the boxed Hamilton equations.
        γ²_j = max(Mvv_arr[j] - sj.β^2, T(0))
        γ_inv = sqrt(max(Mvv_arr[j], T(eps(T)))) / sqrt(max(γ²_j, T(eps(T))))
        γ_marker = (γ_inv - one(T))   # 0 when γ ≈ √M_vv (β=0); large near collapse

        out[j] = d2_α + d2_β + d2_s + d2_u_norm + T(0.01) * γ_marker
    end
    return out
end

"""
    gradient_indicator(mesh::Mesh1D{T,DetField{T}};
                       field::Symbol = :rho) -> Vector{T}

Centered relative-gradient indicator on segment `j`:

    field = :rho  →  |ρ_{j+1} − ρ_{j-1}| / (2 ρ_j)
    field = :u    →  |u_{j+1} − u_{j-1}| / (2 c_s,j)   (Mach gradient)
    field = :P    →  |P_{j+1} − P_{j-1}| / (2 P_j)

The classic AMR gradient detector. Used as the comparison baseline
against `action_error_indicator` in the Tier B.3 acceptance test.

Boundary handling matches `action_error_indicator`.
"""
function gradient_indicator(mesh::Mesh1D{T,DetField{T}};
                            field::Symbol = :rho,
                            Gamma::Real = GAMMA_LAW_DEFAULT) where {T<:Real}
    N = n_segments(mesh)
    out = zeros(T, N)
    if N < 3
        return out
    end
    is_periodic = mesh.bc == :periodic

    # Pre-compute per-segment scalars we may need.
    ρs = T[segment_density(mesh, j) for j in 1:N]
    @inbounds for j in 1:N
        if is_periodic
            jl = j == 1 ? N : j - 1
            jr = j == N ? 1 : j + 1
        else
            jl = max(j - 1, 1)
            jr = min(j + 1, N)
            if jl == j || jr == j
                continue
            end
        end
        if field == :rho
            scale = max(ρs[j], eps(T))
            out[j] = abs(ρs[jr] - ρs[jl]) / (2 * scale)
        elseif field == :u
            sj = mesh.segments[j].state
            J_j = one(T) / ρs[j]
            Mvv_j = Mvv(J_j, sj.s; Gamma = Gamma)
            c_s = sqrt(max(Gamma * Mvv_j, T(eps(T))))
            ul = mesh.segments[jl].state.u
            ur = mesh.segments[jr].state.u
            out[j] = abs(ur - ul) / (2 * c_s)
        elseif field == :P
            P(j_) = ρs[j_] * Mvv(one(T) / ρs[j_], mesh.segments[j_].state.s; Gamma = Gamma)
            scale = max(P(j), eps(T))
            out[j] = abs(P(jr) - P(jl)) / (2 * scale)
        else
            error("gradient_indicator: unknown field $field. Use :rho, :u, or :P.")
        end
    end
    return out
end

# ──────────────────────────────────────────────────────────────────────
# AMR driver
# ──────────────────────────────────────────────────────────────────────

"""
    amr_step!(mesh::Mesh1D, indicator::AbstractVector,
              τ_refine::Real, τ_coarsen::Real = τ_refine / 4;
              tracers = nothing, max_segments = 4096,
              min_segments = 4) -> NamedTuple

Apply one round of refine / coarsen events to `mesh` based on the
per-segment `indicator` vector, with hysteresis
`τ_coarsen ≤ τ_refine / 4` (methods paper §9.7).

Algorithm (single-pass; multiple rounds permitted by the caller):
  1. Mark segments where indicator > τ_refine for refinement.
  2. Mark adjacent pairs where both indicators < τ_coarsen for
     coarsening (and neither was marked for refinement).
  3. Apply coarsens first (right-to-left to keep indices stable),
     then refines (right-to-left).
  4. Cap with `max_segments` and `min_segments` to avoid pathological
     growth/shrinkage.

Returns `(n_refined, n_coarsened)` for diagnostics.
"""
function amr_step!(mesh::Mesh1D{T,DetField{T}},
                   indicator::AbstractVector{<:Real},
                   τ_refine::Real,
                   τ_coarsen::Real = τ_refine / 4;
                   tracers::Union{Nothing,TracerMesh} = nothing,
                   max_segments::Int = 4096,
                   min_segments::Int = 4) where {T<:Real}
    @assert τ_coarsen ≤ τ_refine / 4 + eps(τ_refine) "hysteresis violation: τ_coarsen ($τ_coarsen) must be ≤ τ_refine/4 ($(τ_refine/4))"
    N = n_segments(mesh)
    @assert length(indicator) == N "indicator length $(length(indicator)) ≠ N=$N"

    # Refine candidates (descending j so deletes/inserts don't shift
    # earlier indices).
    refine_set = Int[]
    for j in 1:N
        if indicator[j] > τ_refine
            push!(refine_set, j)
        end
    end

    # Coarsen candidates: adjacent pairs (j, j+1) where both
    # indicators < τ_coarsen and neither is in the refine set.
    coarsen_pairs = Int[]
    for j in 1:N-1   # don't coarsen across periodic wrap (keeps the box stable)
        if indicator[j] < τ_coarsen && indicator[j+1] < τ_coarsen &&
           !(j in refine_set) && !(j+1 in refine_set)
            push!(coarsen_pairs, j)
        end
    end

    # Apply coarsens first, right-to-left, ensuring no overlap.
    n_coarsened = 0
    sort!(coarsen_pairs; rev = true)
    last_j = typemax(Int)
    for j in coarsen_pairs
        # Skip overlapping pairs (shouldn't happen given the j+1 step
        # below, but defensive).
        if j + 1 >= last_j
            continue
        end
        if n_segments(mesh) ≤ min_segments
            break
        end
        coarsen_segment_pair!(mesh, j; tracers = tracers)
        last_j = j
        n_coarsened += 1
    end

    # Two strategies:
    #   * If no coarsens fired, the original refine_set indices are
    #     valid — apply directly (right-to-left).
    #   * If coarsens fired, indices may have shifted; we skip
    #     refines on this round to keep state consistent. The caller
    #     should call amr_step! again with a freshly-evaluated
    #     indicator to pick up the deferred refines. Most production
    #     drivers re-evaluate every step anyway, so this is fine.
    n_refined = 0
    if n_coarsened == 0
        sort!(refine_set; rev = true)
        for j in refine_set
            if n_segments(mesh) ≥ max_segments
                break
            end
            refine_segment!(mesh, j; tracers = tracers)
            n_refined += 1
        end
    end

    return (n_refined = n_refined, n_coarsened = n_coarsened)
end

# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

"""
    refresh_p_half!(mesh::Mesh1D)

Recompute `mesh.p_half` from the current segment vector after a
topology change. `p_half[i] = m̄_i · u_i` with cyclic vertex masses.
"""
function refresh_p_half!(mesh::Mesh1D{T,DetField{T}}) where {T<:Real}
    N = n_segments(mesh)
    if length(mesh.p_half) != N
        resize!(mesh.p_half, N)
    end
    @inbounds for i in 1:N
        i_left = i == 1 ? N : i - 1
        m̄ = (mesh.segments[i_left].Δm + mesh.segments[i].Δm) / 2
        mesh.p_half[i] = m̄ * mesh.segments[i].state.u
    end
    return mesh
end

"""
    refine_tracer!(tm::TracerMesh, j::Integer)

Insert a duplicate of column `j` at column `j+1` in the tracer
matrix (bit-exact). Resizes the matrix; mutates in place where
possible by allocating a new matrix and replacing it in-struct.
Phase 11's bit-exactness is preserved across refine.
"""
function refine_tracer!(tm::TracerMesh{T}, j::Integer) where {T<:Real}
    K, N = size(tm.tracers)
    @assert 1 ≤ j ≤ N "refine_tracer!: column $j out of range 1:$N"
    new_mat = Matrix{T}(undef, K, N + 1)
    @inbounds for col in 1:j
        for row in 1:K
            new_mat[row, col] = tm.tracers[row, col]
        end
    end
    @inbounds for row in 1:K
        new_mat[row, j + 1] = tm.tracers[row, j]   # bit-exact daughter copy
    end
    @inbounds for col in (j + 1):N
        for row in 1:K
            new_mat[row, col + 1] = tm.tracers[row, col]
        end
    end
    # Replace the tracer matrix. Since `tm.tracers` is a field of an
    # immutable struct, we rebuild the matrix's storage by copy.
    # (TracerMesh's tracers field is a Matrix, not a typed array
    # alias; we resize via copyto! after a length-bump.)
    _replace_tracer_matrix!(tm, new_mat)
    return tm
end

"""
    coarsen_tracer!(tm::TracerMesh, j::Integer; Δm_l, Δm_r)

Replace columns `j, j+1` with a single column of mass-weighted
averages (`Δm_l ⊕ Δm_r`). Conservative; bit-exactness deliberately
broken for two-tracer-per-pair merges.
"""
function coarsen_tracer!(tm::TracerMesh{T}, j::Integer;
                         Δm_l::Real, Δm_r::Real) where {T<:Real}
    K, N = size(tm.tracers)
    @assert N ≥ 2 && 1 ≤ j ≤ N "coarsen_tracer!: invalid index"
    Δm_total = T(Δm_l + Δm_r)
    # Allow periodic wrap: j+1 may wrap to 1 if j == N. (For
    # consistency with coarsen_segment_pair! which only deletes the
    # j_next column.)
    j_next = j == N ? 1 : j + 1
    new_mat = Matrix{T}(undef, K, N - 1)
    @inbounds for row in 1:K
        # Mass-weighted merged value.
        merged_val = (T(Δm_l) * tm.tracers[row, j] +
                      T(Δm_r) * tm.tracers[row, j_next]) / Δm_total
        # Build the new row: keep cols 1..j-1, set col j to merged_val,
        # then keep j+2..N (or wrap if j_next == 1, i.e. last column was
        # eliminated and we shift everything else down).
        if j_next == 1
            # j == N path: the "next" column is column 1. Shift the
            # array left by 1 and set the new last column to merged.
            for col in 2:N-1
                new_mat[row, col - 1] = tm.tracers[row, col]
            end
            new_mat[row, N - 1] = merged_val
        else
            for col in 1:j-1
                new_mat[row, col] = tm.tracers[row, col]
            end
            new_mat[row, j] = merged_val
            for col in (j + 2):N
                new_mat[row, col - 1] = tm.tracers[row, col]
            end
        end
    end
    _replace_tracer_matrix!(tm, new_mat)
    return tm
end

"""
    _replace_tracer_matrix!(tm::TracerMesh, new_mat::Matrix)

Internal: swap `tm.tracers`'s contents to those of `new_mat`. The
TracerMesh struct is immutable (its `tracers` Matrix reference is
fixed), but we need to reshape on refine/coarsen. We do so by
mutating the struct via `setfield!` from `Base`. This is
deliberately scoped to the AMR primitives and not exported.
"""
function _replace_tracer_matrix!(tm::TracerMesh, new_mat::AbstractMatrix)
    # `tm` is a mutable struct (so we can re-point the .tracers field
    # at the resized matrix). Necessary on refine (matrix grows) and
    # on coarsen (matrix shrinks). The Phase-11 set_tracer! semantics
    # are unaffected: those mutations work on the underlying Matrix
    # in place, which is also still supported here.
    tm.tracers = new_mat
    return tm
end
