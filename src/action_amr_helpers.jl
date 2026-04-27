# action_amr_helpers.jl — HG-side Phase M3-2 (M2-1) action-AMR primitives.
#
# This file implements the conservative refine/coarsen + indicator
# bookkeeping for a `DetMeshHG{T}` wrapper, mirroring `src/amr_1d.jl`'s
# operations on `Mesh1D{T,DetField{T}}`.
#
# **M3-3e-3 (2026-04-26):** the 1D AMR primitives previously delegated to
# M1's `refine_segment!` / `coarsen_segment_pair!` on the cached
# `Mesh1D` shim and rebuilt the HG `SimplicialMesh{1, T}` from scratch
# afterwards. Per Option B of `reference/notes_M3_3e_cache_mesh_retirement.md`
# §C, the M3-3e-3 lift hand-rolls 1D-Lagrangian refine/coarsen primitives
# operating directly on `(SimplicialMesh{1, T}, PolynomialFieldSet,
# Δm::Vector, p_half::Vector)` without going through Mesh1D. The cache
# mesh field stays allocated on `DetMeshHG` (M3-3e-4 / M3-3e-5 own its
# eventual retirement) but is no longer touched by these helpers.
#
# Bit-exact parity. The native primitives reproduce M1's per-segment
# scalar arithmetic byte-for-byte: each `seg.state.{α, β, x, u, s,
# Pp, Q}` access becomes a `read_detfield(fields, j)` read, each
# `seg.Δm` becomes `mesh.Δm[j]`, and the law-of-total-covariance
# computation is character-identical.

using HierarchicalGrids: SimplicialMesh

# ─────────────────────────────────────────────────────────────────────
# Helpers — periodic-aware Eulerian length / density on HG storage.
# ─────────────────────────────────────────────────────────────────────

"""
    _segment_length_HG_native(fields, j::Integer, N::Integer, L_box::Real)

Eulerian length of segment `j` on an HG `PolynomialFieldSet`,
periodic-wrapped. Mirrors M1's `segment_length(mesh::Mesh1D, j)`. Not
exported — internal helper used by the AMR primitives.
"""
@inline function _segment_length_HG_native(fields, j::Integer, N::Integer, L_box::T) where {T<:Real}
    x_l = T(fields.x[j][1])
    if j < N
        x_r = T(fields.x[j + 1][1])
    else
        x_r = T(fields.x[1][1]) + L_box
    end
    return x_r - x_l
end

"""
    _segment_density_HG_native(fields, j::Integer, Δm_j::Real, N::Integer, L_box::Real)

Eulerian density `ρ_j = Δm_j / Δx_j` of segment `j` on HG storage.
"""
@inline function _segment_density_HG_native(fields, j::Integer, Δm_j::T,
                                             N::Integer, L_box::T) where {T<:Real}
    return Δm_j / _segment_length_HG_native(fields, j, N, L_box)
end

"""
    _refresh_p_half_HG!(mesh::DetMeshHG)

Recompute `mesh.p_half[i] = m̄_i · u_i` from the current HG field set
and `mesh.Δm`. Mirrors M1's `refresh_p_half!` on `Mesh1D`. Used after
a topology change.
"""
@inline function _refresh_p_half_HG!(mesh::DetMeshHG{T}) where {T<:Real}
    N = length(mesh.Δm)
    if length(mesh.p_half) != N
        resize!(mesh.p_half, N)
    end
    fs = mesh.fields
    @inbounds for i in 1:N
        i_left = i == 1 ? N : i - 1
        m̄ = (mesh.Δm[i_left] + mesh.Δm[i]) / 2
        mesh.p_half[i] = m̄ * T(fs.u[i][1])
    end
    return mesh
end

"""
    _rebuild_simplicial_mesh_HG!(mesh::DetMeshHG)

After `mesh.Δm` and `mesh.fields` have been resized to a new `N`,
reassemble the periodic-wrap `SimplicialMesh{1, T}` and store it back
on `mesh.mesh`. Cumulative-mass coordinate exactly mirrors
`DetMeshHG_from_arrays`. Periodic-wrap topology is used regardless of
`bc` (M1's convention — `det_step_HG_native!` honours BC via the BC
spec, not via the simplex-neighbour matrix).
"""
function _rebuild_simplicial_mesh_HG!(mesh::DetMeshHG{T}) where {T<:Real}
    N = length(mesh.Δm)
    cum_m = T(0)
    pos_m = Vector{NTuple{1, T}}(undef, N + 1)
    pos_m[1] = (T(0),)
    @inbounds for j in 1:N
        cum_m += T(mesh.Δm[j])
        pos_m[j + 1] = (cum_m,)
    end
    sv = Matrix{Int32}(undef, 2, N)
    sn = zeros(Int32, 2, N)
    @inbounds for j in 1:N
        sv[1, j] = j
        sv[2, j] = j + 1
        sn[1, j] = j > 1 ? Int32(j - 1) : Int32(N)
        sn[2, j] = j < N ? Int32(j + 1) : Int32(1)
    end
    mesh.mesh = SimplicialMesh{1, T}(pos_m, sv, sn)
    return mesh
end

# M3-3e-5: `_resize_cache_mesh_HG!` and `rebuild_HG_from_cache!` have
# been DROPPED with the cache_mesh field. The native AMR primitives
# below operate directly on `mesh.fields` + `mesh.Δm` + `mesh.p_half`;
# no parallel `Mesh1D` snapshot is maintained.

# ─────────────────────────────────────────────────────────────────────
# Native refine / coarsen primitives.
# ─────────────────────────────────────────────────────────────────────

"""
    refine_segment_HG!(mesh::DetMeshHG, j::Integer; tracers = nothing)
        -> DetMeshHG

Split simplex `j` of the HG-substrate mesh into two equal-mass
daughter simplices at the mid-mass-coordinate. Native HG-side
implementation operating directly on `mesh.fields::PolynomialFieldSet`,
`mesh.Δm`, `mesh.p_half`, and `mesh.mesh::SimplicialMesh{1, T}`. The
cache mesh is **not** touched.

Conservation properties (mirrored from M1's `refine_segment!`):
  * Mass: bit-exact (each daughter gets `Δm/2`).
  * Momentum: exact (the inserted mid-vertex velocity is the
    arithmetic mean of the parent's left and right vertex velocities).
  * Cholesky `(α, β, s, Pp, Q)`: copied from parent (uniform within
    parent under the M1 first-order reconstruction).
  * Tracers: bit-exact daughter copy (Phase-11 exactness preserved).

Bit-exact parity to M1's `refine_segment!` on a parallel `Mesh1D` is
verified by `test_M3_2_M2_1_amr_HG.jl` and the M3-3e-3 cross-check
test.
"""
function refine_segment_HG!(mesh::DetMeshHG{T}, j::Integer;
                             tracers = nothing) where {T<:Real}
    fs = mesh.fields
    N = length(mesh.Δm)
    @assert 1 ≤ j ≤ N "refine: segment index $j out of range 1:$N"

    # Read parent state — mirrors M1's `parent_seg.state` access.
    parent = read_detfield(fs, j)
    Δm_parent = mesh.Δm[j]
    Δm_half = Δm_parent / 2

    # Mid-vertex position: parent's left vertex + half the periodic-aware
    # Eulerian length. Bit-identical to M1's
    # `parent_state.x + segment_length(mesh, j) / 2`.
    Δx_parent = _segment_length_HG_native(fs, j, N, mesh.L_box)
    x_mid = parent.x + Δx_parent / 2

    # Right-vertex velocity: the segment's right vertex is the next
    # segment's left vertex (periodic).
    j_right = j == N ? 1 : j + 1
    u_right = T(fs.u[j_right][1])
    u_mid = (parent.u + u_right) / 2  # linear interpolation, momentum-conserving

    # Build the two daughters.
    left_state = DetField{T}(parent.x, parent.u,
                              parent.α, parent.β,
                              parent.s, parent.Pp, parent.Q)
    right_state = DetField{T}(x_mid, u_mid,
                               parent.α, parent.β,
                               parent.s, parent.Pp, parent.Q)

    # Resize the field set and Δm, shifting everything right of j by
    # one slot. Easiest path: reallocate the field set and copy. This
    # mirrors M1's `splice!` (replace at j, insert at j+1) byte-equally
    # because the per-cell scalar values are unchanged.
    new_fields = allocate_detfield_HG(N + 1; T = T)
    @inbounds for k in 1:(j - 1)
        write_detfield!(new_fields, k, read_detfield(fs, k))
    end
    write_detfield!(new_fields, j,     left_state)
    write_detfield!(new_fields, j + 1, right_state)
    @inbounds for k in (j + 1):N
        write_detfield!(new_fields, k + 1, read_detfield(fs, k))
    end
    mesh.fields = new_fields

    # Δm splice — replace at j, insert at j+1 (mirrors M1).
    mesh.Δm[j] = Δm_half
    insert!(mesh.Δm, j + 1, Δm_half)

    # Rebuild the simplicial mesh on the new mass coordinate, refresh
    # p_half from the (now N+1)-cell field set + Δm.
    _rebuild_simplicial_mesh_HG!(mesh)
    _refresh_p_half_HG!(mesh)

    # M3-3e-5: cache mesh dropped — no lock-step resync needed.

    # Tracers: bit-exact daughter copy.
    if tracers !== nothing
        _refine_tracer_HG!(tracers, j)
    end

    return mesh
end

"""
    coarsen_segment_pair_HG!(mesh::DetMeshHG, j::Integer; tracers = nothing,
                             Gamma = GAMMA_LAW_DEFAULT) -> DetMeshHG

Merge simplices `j` and `j+1` of the HG-substrate mesh into one,
operating directly on HG storage. Native HG-side counterpart to M1's
`coarsen_segment_pair!`. The cache mesh is **not** touched.

Conservation properties (mirrored from M1's `coarsen_segment_pair!`):
  * Mass: bit-exact (`Δm_new = Δm_l + Δm_r`).
  * Momentum: exact (vertex-velocity redistribution preserves
    `Σ m̄_i u_i` to round-off).
  * Cholesky `(α, β)` use the law-of-total-covariance (intra-cell +
    between-cell variance); γ stays real-valued.
  * Tracers: mass-weighted average (conservative; bit-exactness lost,
    matching M1's behaviour).

Bit-exact parity to M1's `coarsen_segment_pair!` on a parallel
`Mesh1D` is verified by `test_M3_2_M2_1_amr_HG.jl` and the M3-3e-3
cross-check test.
"""
function coarsen_segment_pair_HG!(mesh::DetMeshHG{T}, j::Integer;
                                   tracers = nothing,
                                   Gamma::Real = GAMMA_LAW_DEFAULT) where {T<:Real}
    fs = mesh.fields
    N = length(mesh.Δm)
    @assert 1 ≤ j ≤ N "coarsen: segment index $j out of range 1:$N"
    @assert N ≥ 2 "coarsen: cannot reduce mesh below 1 segment"
    j_next = j == N ? 1 : j + 1
    @assert j != j_next "coarsen: degenerate j == j+1 (N = 1)"

    # Per-segment state reads — mirror M1's `state_l = seg_l.state`.
    state_l = read_detfield(fs, j)
    state_r = read_detfield(fs, j_next)

    Δm_l = mesh.Δm[j]
    Δm_r = mesh.Δm[j_next]
    Δm_new = Δm_l + Δm_r

    # Eulerian lengths (periodic-aware, byte-equal to M1's
    # `segment_length(mesh, j)` reads).
    Δx_l = _segment_length_HG_native(fs, j,      N, mesh.L_box)
    Δx_r = _segment_length_HG_native(fs, j_next, N, mesh.L_box)
    Δx_new = Δx_l + Δx_r
    ρ_l = Δm_l / Δx_l
    ρ_r = Δm_r / Δx_r
    J_new = Δx_new / Δm_new

    # Cell-center velocity for the law-of-total-covariance — read the
    # vertex-velocity of the segment after j_next (periodic).
    j_after = j_next == N ? 1 : j_next + 1
    u_cell_l = (state_l.u + state_r.u) / 2
    u_cell_r = (state_r.u + T(fs.u[j_after][1])) / 2

    # Pre-merge M_vv per segment from the EOS.
    M_vv_l = Mvv(Δx_l / Δm_l, state_l.s; Gamma = Gamma)
    M_vv_r = Mvv(Δx_r / Δm_r, state_r.s; Gamma = Gamma)

    # Law of total covariance.
    M_vv_intra = (Δm_l * M_vv_l + Δm_r * M_vv_r) / Δm_new
    M_vv_between = (Δm_l * Δm_r / Δm_new^2) * (u_cell_l - u_cell_r)^2
    M_vv_new = M_vv_intra + M_vv_between

    β_new = (Δm_l * state_l.β + Δm_r * state_r.β) / Δm_new
    γ²_new = max(M_vv_new - β_new^2, zero(T))
    if M_vv_new - β_new^2 < zero(T)
        # Round-off; absorb the deficit by lifting M_vv just to β².
        M_vv_new = β_new^2
    end

    cv = T(CV_DEFAULT)
    s_new = M_vv_new > 0 ?
            cv * (log(M_vv_new) - (1 - Gamma) * log(J_new)) :
            (state_l.s + state_r.s) / 2

    α_new = (Δm_l * state_l.α + Δm_r * state_r.α) / Δm_new
    Pp_new = (Δm_l * state_l.Pp + Δm_r * state_r.Pp) / Δm_new
    Q_new  = (Δm_l * state_l.Q  + Δm_r * state_r.Q ) / Δm_new

    # Vertex-velocity redistribution (mirrors M1 byte-equally).
    j_left_of_seam  = j == 1 ? N : j - 1
    j_right_of_seam = j_next == N ? 1 : j_next + 1

    m̄_left_pre  = (mesh.Δm[j_left_of_seam] + Δm_l) / 2
    m̄_seam_pre  = (Δm_l + Δm_r) / 2
    m̄_rite_pre  = (Δm_r + mesh.Δm[j_right_of_seam]) / 2

    m̄_left_post = (mesh.Δm[j_left_of_seam] + Δm_new) / 2
    m̄_rite_post = (Δm_new + mesh.Δm[j_right_of_seam]) / 2

    u_seam_pre = state_r.u
    u_left_pre = state_l.u
    u_rite_pre = T(fs.u[j_right_of_seam][1])

    p_left_new = m̄_left_pre * u_left_pre + (Δm_r / 2) * u_seam_pre
    p_rite_new = m̄_rite_pre * u_rite_pre + (Δm_l / 2) * u_seam_pre
    u_left_new = p_left_new / m̄_left_post
    u_rite_new = p_rite_new / m̄_rite_post

    # Build the merged segment's state.
    merged_state = DetField{T}(
        state_l.x,            # left vertex of merged = left vertex of seg_l
        u_left_new,
        α_new, β_new,
        s_new, Pp_new, Q_new,
    )

    # Snapshot the right-of-seam state so we can apply the new u_rite
    # consistently when N != 2 (when N == 2 the right-of-seam wraps
    # back onto the merged cell; M1 skips the rewrite in that case).
    right_of_seam_state = if j_right_of_seam != j
        rs = read_detfield(fs, j_right_of_seam)
        DetField{T}(rs.x, u_rite_new, rs.α, rs.β, rs.s, rs.Pp, rs.Q)
    else
        merged_state  # unused
    end

    # Rebuild the field set into N - 1 cells.
    new_fields = allocate_detfield_HG(N - 1; T = T)
    if j_next == 1
        # j == N path. M1's `deleteat!(mesh.segments, j_next = 1)` shifts
        # the array down by 1 and overwrites segment N (the merged) with
        # `merged_state`. The periodic right-of-seam is the (now-)last
        # segment which already got its u updated.
        @inbounds for k in 2:(N - 1)
            write_detfield!(new_fields, k - 1, read_detfield(fs, k))
        end
        # The merged cell sits at the new last index (N - 1). It absorbs
        # the (now-removed) segment 1.
        write_detfield!(new_fields, N - 1, merged_state)
        # Right-of-seam segment is `j_right_of_seam = 2` pre-merge in the
        # j == N path. Its post-merge index is 1 (after the shift).
        if j_right_of_seam != j
            # j_right_of_seam = 2 (post-shift index 1)
            write_detfield!(new_fields, j_right_of_seam - 1, right_of_seam_state)
        end
    else
        @inbounds for k in 1:(j - 1)
            write_detfield!(new_fields, k, read_detfield(fs, k))
        end
        write_detfield!(new_fields, j, merged_state)
        # Update the right-of-seam segment if it's not the merged cell
        # itself. In M1's path j_right_of_seam pre-merge is at
        # `j_next + 1`; post-merge that index becomes `j_next` (since
        # we deleted j_next), but only if j_right_of_seam was right of
        # j_next. In the wrap case j_right_of_seam == 1 (when j_next ==
        # N), but that branch is the j_next == 1 case handled above.
        # Here j_next ∈ 2:N and j_right_of_seam = j_next + 1 unless
        # j_next == N (in which case j_right_of_seam = 1 — but j_next ==
        # N implies j == N - 1, which is handled in the else branch).
        @inbounds for k in (j + 2):N
            if k == j_right_of_seam
                write_detfield!(new_fields, k - 1, right_of_seam_state)
            else
                write_detfield!(new_fields, k - 1, read_detfield(fs, k))
            end
        end
        # Edge case: when j_next == N (i.e. j == N - 1 with N ≥ 3), the
        # right-of-seam wraps to 1. The loop above starts at j + 2 == N + 1
        # and is empty. We need to apply right_of_seam_state to index 1.
        if j_right_of_seam < j  # = wrap-around to 1 (so j_right_of_seam = 1, j_next == N)
            write_detfield!(new_fields, j_right_of_seam, right_of_seam_state)
        end
    end
    mesh.fields = new_fields

    # Δm splice — bit-equally mirrors M1's mass bookkeeping.
    if j_next == 1
        # j == N path. M1: deleteat!(segments, 1) shifts down, then
        # mesh.segments[j] = merged_segment overwrote segment N earlier.
        # Equivalently: remove Δm[1], replace merged at the (now-shifted)
        # slot N - 1.
        deleteat!(mesh.Δm, 1)
        mesh.Δm[end] = Δm_new
    else
        mesh.Δm[j] = Δm_new
        deleteat!(mesh.Δm, j_next)
    end

    # Rebuild simplicial mesh + p_half from the new (N-1)-cell state.
    _rebuild_simplicial_mesh_HG!(mesh)
    _refresh_p_half_HG!(mesh)

    # M3-3e-5: cache mesh dropped — no lock-step resync needed.

    # Tracers: mass-weighted average and remove the second column.
    if tracers !== nothing
        _coarsen_tracer_HG!(tracers, j; Δm_l = Δm_l, Δm_r = Δm_r)
    end

    return mesh
end

# ─────────────────────────────────────────────────────────────────────
# Indicators on HG storage
# ─────────────────────────────────────────────────────────────────────

"""
    action_error_indicator_HG(mesh::DetMeshHG; dt = nothing,
                              Gamma = GAMMA_LAW_DEFAULT) -> Vector

Per-segment ΔS_cell action-error indicator on the HG-substrate mesh.
Bit-equal-equivalent to M1's `action_error_indicator(mesh_M1)` on a
parallel Mesh1D with the same per-cell state. M3-3e-3: native HG
implementation; the cache mesh is no longer touched.
"""
function action_error_indicator_HG(mesh::DetMeshHG{T};
                                    dt::Union{Real,Nothing} = nothing,
                                    Gamma::Real = GAMMA_LAW_DEFAULT) where {T<:Real}
    fs = mesh.fields
    N = length(mesh.Δm)
    out = zeros(T, N)
    if N < 3
        return out
    end
    is_periodic = mesh.bc == :periodic

    # Pre-compute per-segment ρ, M_vv, c_s — same arithmetic as M1's
    # `action_error_indicator`, just sourced from HG storage.
    ρ_arr   = Vector{T}(undef, N)
    Mvv_arr = Vector{T}(undef, N)
    cs_arr  = Vector{T}(undef, N)
    @inbounds for j in 1:N
        Δx_j = _segment_length_HG_native(fs, j, N, mesh.L_box)
        ρ_arr[j] = mesh.Δm[j] / Δx_j
        s_j = T(fs.s[j][1])
        Mvv_arr[j] = Mvv(one(T) / ρ_arr[j], s_j; Gamma = Gamma)
        cs_arr[j]  = sqrt(max(Gamma * Mvv_arr[j], T(eps(T))))
    end

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
        αl = T(fs.alpha[jl][1]);  αj = T(fs.alpha[j][1]);  αr = T(fs.alpha[jr][1])
        βl = T(fs.beta[jl][1]);   βj = T(fs.beta[j][1]);   βr = T(fs.beta[jr][1])
        sl = T(fs.s[jl][1]);      sj = T(fs.s[j][1]);      sr = T(fs.s[jr][1])
        ul = T(fs.u[jl][1]);      uj = T(fs.u[j][1]);      ur = T(fs.u[jr][1])

        d2_α = abs(αr - 2 * αj + αl)
        d2_β = abs(βr - 2 * βj + βl)
        d2_s = abs(sr - 2 * sj + sl)

        d2_u = abs(ur - 2 * uj + ul)
        d2_u_norm = d2_u / max(cs_arr[j], T(eps(T)))

        γ²_j  = max(Mvv_arr[j] - βj^2, T(0))
        γ_inv = sqrt(max(Mvv_arr[j], T(eps(T)))) /
                sqrt(max(γ²_j, T(eps(T))))
        γ_marker = (γ_inv - one(T))

        out[j] = d2_α + d2_β + d2_s + d2_u_norm + T(0.01) * γ_marker
    end
    return out
end

"""
    gradient_indicator_HG(mesh::DetMeshHG; field = :rho,
                          Gamma = GAMMA_LAW_DEFAULT) -> Vector

Per-segment relative-gradient indicator on the HG-substrate mesh.
Bit-equal-equivalent to M1's `gradient_indicator(mesh_M1; field)`.
M3-3e-3: native HG implementation; the cache mesh is no longer touched.
"""
function gradient_indicator_HG(mesh::DetMeshHG{T};
                                field::Symbol = :rho,
                                Gamma::Real = GAMMA_LAW_DEFAULT) where {T<:Real}
    fs = mesh.fields
    N = length(mesh.Δm)
    out = zeros(T, N)
    if N < 3
        return out
    end
    is_periodic = mesh.bc == :periodic

    ρs = Vector{T}(undef, N)
    @inbounds for j in 1:N
        Δx_j = _segment_length_HG_native(fs, j, N, mesh.L_box)
        ρs[j] = mesh.Δm[j] / Δx_j
    end

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
            J_j = one(T) / ρs[j]
            s_j = T(fs.s[j][1])
            Mvv_j = Mvv(J_j, s_j; Gamma = Gamma)
            c_s = sqrt(max(Gamma * Mvv_j, T(eps(T))))
            ul = T(fs.u[jl][1])
            ur = T(fs.u[jr][1])
            out[j] = abs(ur - ul) / (2 * c_s)
        elseif field == :P
            P(j_) = ρs[j_] * Mvv(one(T) / ρs[j_], T(fs.s[j_][1]); Gamma = Gamma)
            scale = max(P(j), eps(T))
            out[j] = abs(P(jr) - P(jl)) / (2 * scale)
        else
            error("gradient_indicator_HG: unknown field $field. Use :rho, :u, or :P.")
        end
    end
    return out
end

# ─────────────────────────────────────────────────────────────────────
# AMR driver — mirrors M1's `amr_step!`.
# ─────────────────────────────────────────────────────────────────────

"""
    amr_step_HG!(mesh::DetMeshHG, indicator, τ_refine,
                 τ_coarsen = τ_refine/4; tracers = nothing,
                 max_segments = 4096, min_segments = 4)

Apply one round of refine / coarsen events to a `DetMeshHG{T}` mesh
based on `indicator`. M3-3e-3: native HG implementation, mirrors
M1's `amr_step!` byte-equally. Returns `(n_refined, n_coarsened)`.
"""
function amr_step_HG!(mesh::DetMeshHG{T},
                      indicator::AbstractVector{<:Real},
                      τ_refine::Real,
                      τ_coarsen::Real = τ_refine / 4;
                      tracers = nothing,
                      max_segments::Int = 4096,
                      min_segments::Int = 4) where {T<:Real}
    @assert τ_coarsen ≤ τ_refine / 4 + eps(τ_refine) "hysteresis violation: τ_coarsen ($τ_coarsen) must be ≤ τ_refine/4 ($(τ_refine/4))"
    N = length(mesh.Δm)
    @assert length(indicator) == N "indicator length $(length(indicator)) ≠ N=$N"

    refine_set = Int[]
    for j in 1:N
        if indicator[j] > τ_refine
            push!(refine_set, j)
        end
    end

    coarsen_pairs = Int[]
    for j in 1:N-1
        if indicator[j] < τ_coarsen && indicator[j+1] < τ_coarsen &&
           !(j in refine_set) && !(j+1 in refine_set)
            push!(coarsen_pairs, j)
        end
    end

    n_coarsened = 0
    sort!(coarsen_pairs; rev = true)
    last_j = typemax(Int)
    for j in coarsen_pairs
        if j + 1 >= last_j
            continue
        end
        if length(mesh.Δm) ≤ min_segments
            break
        end
        coarsen_segment_pair_HG!(mesh, j; tracers = tracers)
        last_j = j
        n_coarsened += 1
    end

    n_refined = 0
    if n_coarsened == 0
        sort!(refine_set; rev = true)
        for j in refine_set
            if length(mesh.Δm) ≥ max_segments
                break
            end
            refine_segment_HG!(mesh, j; tracers = tracers)
            n_refined += 1
        end
    end

    return (n_refined = n_refined, n_coarsened = n_coarsened)
end

# ─────────────────────────────────────────────────────────────────────
# Tracer refine / coarsen helpers — operate on the native
# `_TracerStorageHG{T}` storage (defined in `src/newton_step_HG_M3_2.jl`).
# ─────────────────────────────────────────────────────────────────────

"""
    _refine_tracer_HG!(tracers::TracerMeshHG, j::Integer)

Insert a duplicate of column `j` at column `j+1` in the tracer
matrix (bit-exact). Mirrors M1's `refine_tracer!` byte-equally.
"""
function _refine_tracer_HG!(tracers, j::Integer)
    mat = tracers.tm.tracers
    K, N = size(mat)
    @assert 1 ≤ j ≤ N "_refine_tracer_HG!: column $j out of range 1:$N"
    new_mat = Matrix{eltype(mat)}(undef, K, N + 1)
    @inbounds for col in 1:j
        for row in 1:K
            new_mat[row, col] = mat[row, col]
        end
    end
    @inbounds for row in 1:K
        new_mat[row, j + 1] = mat[row, j]   # bit-exact daughter copy
    end
    @inbounds for col in (j + 1):N
        for row in 1:K
            new_mat[row, col + 1] = mat[row, col]
        end
    end
    tracers.tm.tracers = new_mat
    return tracers
end

"""
    _coarsen_tracer_HG!(tracers::TracerMeshHG, j::Integer; Δm_l, Δm_r)

Replace columns `j, j+1` (cyclic) with a single column of
mass-weighted averages. Mirrors M1's `coarsen_tracer!` byte-equally.
"""
function _coarsen_tracer_HG!(tracers, j::Integer;
                              Δm_l::Real, Δm_r::Real)
    mat = tracers.tm.tracers
    K, N = size(mat)
    @assert N ≥ 2 && 1 ≤ j ≤ N "_coarsen_tracer_HG!: invalid index"
    Tt = eltype(mat)
    Δm_total = Tt(Δm_l + Δm_r)
    j_next = j == N ? 1 : j + 1
    new_mat = Matrix{Tt}(undef, K, N - 1)
    @inbounds for row in 1:K
        merged_val = (Tt(Δm_l) * mat[row, j] +
                      Tt(Δm_r) * mat[row, j_next]) / Δm_total
        if j_next == 1
            for col in 2:N-1
                new_mat[row, col - 1] = mat[row, col]
            end
            new_mat[row, N - 1] = merged_val
        else
            for col in 1:j-1
                new_mat[row, col] = mat[row, col]
            end
            new_mat[row, j] = merged_val
            for col in (j + 2):N
                new_mat[row, col - 1] = mat[row, col]
            end
        end
    end
    tracers.tm.tracers = new_mat
    return tracers
end

# ─────────────────────────────────────────────────────────────────────
# Phase M3-3d: per-axis action-AMR indicator on HierarchicalMesh{2}
# ─────────────────────────────────────────────────────────────────────
#
# The 2D Cholesky-sector field set carries 10 Newton-unknown scalars
# `(x_a, u_a, α_a, β_a)_{a=1,2} + θ_R + s` per leaf (12 with the post-
# Newton `Pp, Q` slots). The M2-1 action-AMR indicator generalises per-
# axis: we evaluate ΔS_cell separately along each axis using the
# axis-a quantities `(α_a, β_a, u_a)` and the per-axis γ collapse
# marker, then aggregate as `max_a` so a cell refines along the most
# action-energetic axis. This selects collapsing axes (γ → 0) without
# triggering on the trivial axis (γ = O(1)). See §4.3 + §6.5 of
# `reference/notes_M3_3_2d_cholesky_berry.md`.
#
# The indicator is evaluated using the per-axis 4-stencil `(j_lo, j,
# j_hi)` along axis `a`, where `(j_lo, j_hi)` come from the same
# face-neighbor table that the 2D residual consumes
# (`build_face_neighbor_tables`).

using HierarchicalGrids: HierarchicalMesh, EulerianFrame,
    register_refinement_listener!, unregister_refinement_listener!,
    enumerate_leaves, refine_by_indicator!,
    n_cells as hg_n_cells_amr, RefinementEvent, ListenerHandle
using HierarchicalGrids.AMR: step_with_amr!

"""
    action_error_indicator_2d_per_axis(fields, mesh, frame, leaves, bc_spec;
                                        ρ_ref=1.0, Gamma=GAMMA_LAW_DEFAULT,
                                        M_vv_override=nothing)
        -> Vector{Float64}

Per-leaf, per-axis-aggregated action-error indicator on a 2D
`HierarchicalMesh{2}`. The output `out[i]` is

    max_a [|d2(α_a)| + |d2(β_a)| + |d2(u_a)|/c_s + 0.01·γ_inv_marker_a]

evaluated as a 3-point second difference along axis `a`, where
`(j_lo_a, j_hi_a)` are pulled from `build_face_neighbor_tables(mesh,
leaves, bc_spec)` (so periodic wrap-around / reflecting BCs are
honoured upstream). The shared `s`-curvature term `|d2(s)|` is added
once per leaf (not per axis) since `s` is a scalar shared across axes.

The per-axis γ marker `γ_inv_marker_a = √Mvv / max(γ_a, ε) − 1` is the
same Hessian-degeneracy proxy used in the 1D `action_error_indicator`,
evaluated per axis. It is what gives the indicator its **per-axis
selectivity**: when `γ_a → 0` along the collapsing axis (`a = 1` for
the C.2 cold sinusoid with `k_y = 0`), the marker spikes along that
axis; the trivial axis stays `O(1)` so its marker stays near zero.
The `max_a` aggregation then picks the collapsing axis as the cell's
indicator value.

# Boundary handling

When `face_lo_idx[a][i] == 0` or `face_hi_idx[a][i] == 0` (out-of-domain
after BC processing), the indicator drops the corresponding axis-a
contribution to zero (one-sided second-difference is unreliable at
the indicator level — the 1D action-error indicator does the same on
non-periodic boundaries).

# Use cases

  • Drives `refine_by_indicator!` on `HierarchicalMesh{2}` via
    `step_with_amr_2d!` (this file).
  • The §6.5 selectivity test: with `k_y = 0` cold sinusoid and the
    indicator running per step, refinement events fire only on cells
    along the collapsing-x axis; the y-direction stays unrefined.
"""
function action_error_indicator_2d_per_axis(fields, mesh::HierarchicalMesh{2},
                                            frame::EulerianFrame{2, T},
                                            leaves::AbstractVector{<:Integer},
                                            bc_spec;
                                            ρ_ref::Real = 1.0,
                                            Gamma::Real = GAMMA_LAW_DEFAULT,
                                            M_vv_override = nothing) where {T}
    N = length(leaves)
    out = zeros(Float64, N)
    if N < 3
        return out
    end

    face_lo_idx, face_hi_idx = build_face_neighbor_tables(mesh, leaves, bc_spec)

    # Pre-compute per-leaf primitives.
    α1 = Vector{Float64}(undef, N); α2 = Vector{Float64}(undef, N)
    β1 = Vector{Float64}(undef, N); β2 = Vector{Float64}(undef, N)
    u1 = Vector{Float64}(undef, N); u2 = Vector{Float64}(undef, N)
    s_v = Vector{Float64}(undef, N)
    Mvv_v = Vector{Float64}(undef, N)
    cs_v = Vector{Float64}(undef, N)
    @inbounds for (i, ci) in enumerate(leaves)
        α1[i] = Float64(fields.α_1[ci][1])
        α2[i] = Float64(fields.α_2[ci][1])
        β1[i] = Float64(fields.β_1[ci][1])
        β2[i] = Float64(fields.β_2[ci][1])
        u1[i] = Float64(fields.u_1[ci][1])
        u2[i] = Float64(fields.u_2[ci][1])
        s_v[i] = Float64(fields.s[ci][1])
        if M_vv_override !== nothing
            Mvv_v[i] = max(Float64(M_vv_override[1]),
                            Float64(M_vv_override[2]))
        else
            Mvv_v[i] = Float64(Mvv(1.0 / Float64(ρ_ref), s_v[i];
                                    Gamma = Float64(Gamma)))
        end
        cs_v[i] = sqrt(max(Float64(Gamma) * Mvv_v[i], eps(Float64)))
    end

    @inbounds for i in 1:N
        # Shared s-curvature contribution (uses axis-1 stencil; the
        # 2D field set has a single scalar `s` per leaf).
        d2_s_axis1 = zero(Float64)
        ilo = face_lo_idx[1][i]
        ihi = face_hi_idx[1][i]
        if ilo > 0 && ihi > 0
            d2_s_axis1 = abs(s_v[ihi] - 2 * s_v[i] + s_v[ilo])
        end

        # Per-axis contributions.
        max_axis = zero(Float64)
        for a in 1:2
            ilo_a = face_lo_idx[a][i]
            ihi_a = face_hi_idx[a][i]
            if ilo_a == 0 || ihi_a == 0
                continue
            end
            if a == 1
                d2_α = abs(α1[ihi_a] - 2 * α1[i] + α1[ilo_a])
                d2_β = abs(β1[ihi_a] - 2 * β1[i] + β1[ilo_a])
                d2_u = abs(u1[ihi_a] - 2 * u1[i] + u1[ilo_a])
                β_self = β1[i]
            else
                d2_α = abs(α2[ihi_a] - 2 * α2[i] + α2[ilo_a])
                d2_β = abs(β2[ihi_a] - 2 * β2[i] + β2[ilo_a])
                d2_u = abs(u2[ihi_a] - 2 * u2[i] + u2[ilo_a])
                β_self = β2[i]
            end
            d2_u_norm = d2_u / max(cs_v[i], eps(Float64))
            γ²_a = max(Mvv_v[i] - β_self * β_self, 0.0)
            γ_inv = sqrt(max(Mvv_v[i], eps(Float64))) /
                    sqrt(max(γ²_a, eps(Float64)))
            γ_marker = γ_inv - 1.0
            axis_val = d2_α + d2_β + d2_u_norm + 0.01 * γ_marker
            max_axis = max(max_axis, axis_val)
        end
        out[i] = max_axis + d2_s_axis1
    end
    return out
end

"""
    register_field_set_on_refine!(fields, mesh::HierarchicalMesh{D};
                                   default_value=0.0)
        -> ListenerHandle

Register an HG `register_refinement_listener!` callback that keeps a
`PolynomialFieldSet` consistent across refine/coarsen events on
`mesh`. After each `refine_cells!` / `coarsen_cells!` (or
`refine_by_indicator!`) call, the listener:

  1. Resizes every named field's coefficient storage to
     `n_cells(mesh)` post-event (HG does NOT track field sets — the
     dfmm-side has to do it explicitly).
  2. For each refined parent → children mapping in
     `event.new_children`, copies the parent's order-0 cell-average
     into every child (piecewise-constant prolongation, the natural
     order-0 reconstruction).
  3. For each coarsened parent in `event.coarsened_parents`, averages
     the (now-removed) children's order-0 values into the parent
     coefficient (mass-conservative for `:rho`-like fields when the
     children are equal-volume, which is true for isotropic refinement).

The listener handle is returned so callers can `unregister_refinement_listener!`
when the field set's lifecycle ends. **The listener captures `fields`
by reference** — any subsequent reallocation of `fields` (which
shouldn't happen for `PolynomialFieldSet` SoA storage; the storage
arrays are mutated in place) would be a use-after-free.

# Limits

  • Order-0 piecewise-constant prolongation only. Higher-order
    Bernstein restriction/prolongation is M3-4 / M3-5 work.
  • The listener walks all named fields in the field set; the dfmm
    Cholesky-sector layout (12 named fields per the M3-3a allocator)
    is fully covered.
"""
function register_field_set_on_refine!(fields, mesh::HierarchicalMesh{D};
                                       default_value::Real = 0.0) where {D}
    # Names of the public named scalar fields. The HG `PolynomialFieldSet`
    # exposes the field names via `field_names(fields)`, with the SoA
    # storage backing accessible at `fields.storage` as a NamedTuple of
    # `Vector{T}` (one entry per coefficient × cell). For `MonomialBasis{D, 0}`
    # there is exactly one coefficient per cell, so the storage vector
    # length equals `n_cells(mesh)`.
    names = HierarchicalGrids.field_names(fields)
    T_default = Float64(default_value)
    cb = function(event::RefinementEvent)
        new_n = Int(HierarchicalGrids.n_cells(mesh))
        index_remap = event.index_remap
        old_n = length(index_remap)
        # Per name: snapshot old values, reallocate, prolongate into
        # children, average across coarsened groups.
        for name in names
            storage_arr = getproperty(fields.storage, name)
            old_vals = copy(storage_arr)::Vector{Float64}
            new_vals = fill(T_default, new_n)
            @inbounds for old_i in 1:old_n
                new_i = Int(index_remap[old_i])
                if new_i != 0
                    new_vals[new_i] = old_vals[old_i]
                end
            end
            # Prolongate refined parents into their children
            # (piecewise-constant order-0 prolongation).
            for k in eachindex(event.refined_parents)
                old_parent = Int(event.refined_parents[k])
                parent_val = old_vals[old_parent]
                child_range = event.new_children[k]
                for child in child_range
                    new_vals[Int(child)] = parent_val
                end
            end
            # Coarsened parents: HG's coarsen_cells! event populates
            # `removed_old_indices` in groups of 2^D per parent in
            # `coarsened_parents` order. Average each group.
            n_per_group = 1 << D
            for k in eachindex(event.coarsened_parents)
                new_parent = Int(event.coarsened_parents[k])
                child_acc = 0.0
                for offset in 0:(n_per_group - 1)
                    old_child = Int(event.removed_old_indices[
                        (k - 1) * n_per_group + offset + 1])
                    child_acc += old_vals[old_child]
                end
                new_vals[new_parent] = child_acc / n_per_group
            end
            # In-place resize + write-back.
            resize!(storage_arr, new_n)
            @inbounds for i in 1:new_n
                storage_arr[i] = new_vals[i]
            end
        end
        return nothing
    end
    return register_refinement_listener!(mesh, cb)
end

"""
    step_with_amr_2d!(fields, mesh::HierarchicalMesh{2},
                      frame::EulerianFrame{2, T},
                      bc_spec, dt, n_steps;
                      refine_threshold,
                      coarsen_threshold = refine_threshold / 4,
                      hysteresis_steps = 3,
                      max_level = typemax(Int),
                      isotropic = true,
                      M_vv_override = nothing,
                      ρ_ref = 1.0,
                      project_kind = :none,
                      realizability_headroom = 1.05,
                      Mvv_floor = 1e-2,
                      pressure_floor = 1e-8,
                      proj_stats = nothing,
                      newton_kwargs = (;)) where {T}

End-to-end M3-3d AMR-driven 2D run on a `HierarchicalMesh{2}`. Wraps
HG's `step_with_amr!` driver with:

  • A `step!` callback that (a) calls `det_step_2d_berry_HG!` per step
    on the current leaves, then (b) refreshes the leaves vector after
    each AMR cycle.
  • An `indicator(mesh)` callback that returns the per-axis-aggregated
    action-error indicator `action_error_indicator_2d_per_axis` mapped
    onto **mesh-cell indices** (length `n_cells(mesh)`; non-leaf cells
    get value 0 so they can never trigger refinement).
  • A refinement listener registered via `register_field_set_on_refine!`
    that prolongates / restricts the field set across each
    `refine_by_indicator!` event.

Returns `n_steps` (the number of physics steps taken). Mutates
`fields` and `mesh` in place. Closes M3-2b's deferred Swaps 2+3 for
the 2D scope (per the M3-3d brief): this is the first dfmm path that
uses HG's `register_refinement_listener!` + `step_with_amr!` natively.
"""
function step_with_amr_2d!(fields, mesh::HierarchicalMesh{2},
                            frame::EulerianFrame{2, T},
                            bc_spec, dt::Real, n_steps::Integer;
                            refine_threshold::Real,
                            coarsen_threshold::Real = refine_threshold / 4,
                            hysteresis_steps::Integer = 3,
                            max_level::Integer = typemax(Int),
                            isotropic::Bool = true,
                            M_vv_override = nothing,
                            ρ_ref::Real = 1.0,
                            project_kind::Symbol = :none,
                            realizability_headroom::Real = 1.05,
                            Mvv_floor::Real = 1e-2,
                            pressure_floor::Real = 1e-8,
                            proj_stats = nothing,
                            newton_kwargs::NamedTuple = NamedTuple()) where {T}
    @assert mesh.balanced "step_with_amr_2d! requires mesh.balanced == true"

    # State carried across the step!/indicator closures.
    state = Ref(enumerate_leaves(mesh))

    handle = register_field_set_on_refine!(fields, mesh)
    try
        step_fn = function(state_ref, frame_arg)
            leaves_now = state_ref[]
            # Refresh leaves if a refinement happened since last call
            # (the listener has updated `fields` by now).
            if length(leaves_now) != length(enumerate_leaves(frame_arg.mesh))
                leaves_now = enumerate_leaves(frame_arg.mesh)
                state_ref[] = leaves_now
            end
            det_step_2d_berry_HG!(fields, frame_arg.mesh, frame_arg, leaves_now,
                                   bc_spec, dt;
                                   M_vv_override = M_vv_override,
                                   ρ_ref = ρ_ref,
                                   project_kind = project_kind,
                                   realizability_headroom = realizability_headroom,
                                   Mvv_floor = Mvv_floor,
                                   pressure_floor = pressure_floor,
                                   proj_stats = proj_stats,
                                   newton_kwargs...)
            return nothing
        end

        ind_fn = function(mesh_arg)
            leaves_now = enumerate_leaves(mesh_arg)
            state[] = leaves_now
            ind_leaf = action_error_indicator_2d_per_axis(
                fields, mesh_arg, frame, leaves_now, bc_spec;
                ρ_ref = ρ_ref, M_vv_override = M_vv_override)
            # Map leaf-major indicator to mesh-cell indicator. Non-leaves
            # are not eligible for refine_by_indicator anyway, but HG
            # still calls the indicator across all cells; we return 0
            # for non-leaves so they never trigger.
            nc = HierarchicalGrids.n_cells(mesh_arg)
            full = zeros(Float64, nc)
            @inbounds for (k, ci) in enumerate(leaves_now)
                full[ci] = ind_leaf[k]
            end
            return full
        end

        return step_with_amr!(fields, frame, step_fn, ind_fn, n_steps;
                               refine_threshold = refine_threshold,
                               coarsen_threshold = coarsen_threshold,
                               hysteresis_steps = hysteresis_steps,
                               max_level = max_level,
                               isotropic = isotropic)
    finally
        unregister_refinement_listener!(mesh, handle)
    end
end

# ─────────────────────────────────────────────────────────────────────
# M3-6 Phase 3 (a): 2D tracer refinement listener.
# ─────────────────────────────────────────────────────────────────────

"""
    _hierarchical_mesh_dim(mesh::HierarchicalMesh{D}) -> Int

Extract the spatial dimension `D` from a `HierarchicalMesh{D, M}`
type parameter. Used by `register_tracers_on_refine_2d!` since HG
does not export `spatial_dimension` for the cubical mesh (only for
`SimplicialMesh`).
"""
@inline _hierarchical_mesh_dim(::HierarchicalMesh{D, M}) where {D, M} = D

"""
    register_tracers_on_refine_2d!(tm::TracerMeshHG2D) -> ListenerHandle

Register an HG `register_refinement_listener!` callback that keeps
`tm.tracers` consistent across refine/coarsen events on `tm.mesh`.
After each `refine_cells!` / `coarsen_cells!` (or
`refine_by_indicator!`) call, the listener:

  1. Resizes the tracer matrix's column dimension to `n_cells(mesh)`
     post-event (preserving the per-species row count).
  2. For each refined parent → children mapping in
     `event.new_children`, copies the parent's per-species
     concentration into every child (piecewise-constant
     prolongation, exactly mass-conservative under equal-volume
     refinement: `c_child = c_parent`, so total mass `Σ c·V` is
     preserved bit-exactly).
  3. For each coarsened parent in `event.coarsened_parents`,
     averages the (now-removed) children's per-species
     concentrations into the parent (volume-weighted mean; equal-
     volume children make this the plain arithmetic mean, also
     exactly mass-conservative).

Mirrors `register_field_set_on_refine!`'s listener shape but operates
on the `tm.tracers::Matrix{T}` storage instead of a
`PolynomialFieldSet`. Captures `tm` by reference; in-place
`resize!` keeps the matrix's identity stable across events.

# Multi-species independence

Each species row is processed independently (no cross-row reads or
writes). Refining or coarsening species `k` does not perturb species
`k′ ≠ k` — verified by the M3-6 Phase 3 multi-species independence
gate.

# Lifecycle

Returns the `ListenerHandle` so callers can
`unregister_refinement_listener!(tm.mesh, handle)` at the end of
the run. Forgetting to unregister is a leak (the listener stays
captured in `tm.mesh.refinement_listeners`); not a correctness
issue for short-lived test runs but should be paired with the
`register` call in production drivers.
"""
function register_tracers_on_refine_2d!(tm)
    mesh = tm.mesh
    # Spatial dimension D is encoded in the mesh's type parameter:
    # `HierarchicalMesh{D}`. HG does NOT export `spatial_dimension` for
    # the cubical mesh (it's defined only on `SimplicialMesh`), so we
    # extract D from the type itself.
    D = _hierarchical_mesh_dim(mesh)
    cb = function(event::RefinementEvent)
        new_n = Int(HierarchicalGrids.n_cells(mesh))
        index_remap = event.index_remap
        old_n = length(index_remap)
        K = size(tm.tracers, 1)
        # Snapshot old values, reallocate column-wise via a temporary,
        # prolongate refined parents into children, average across
        # coarsened groups. The matrix reallocation pattern matches
        # `register_field_set_on_refine!` field-by-field, applied per
        # species row.
        old_mat = copy(tm.tracers)
        new_mat = zeros(eltype(tm.tracers), K, new_n)
        @inbounds for old_i in 1:old_n
            new_i = Int(index_remap[old_i])
            if new_i != 0
                for k in 1:K
                    new_mat[k, new_i] = old_mat[k, old_i]
                end
            end
        end
        # Refined parents: piecewise-constant prolongation per species.
        for k_event in eachindex(event.refined_parents)
            old_parent = Int(event.refined_parents[k_event])
            child_range = event.new_children[k_event]
            for child in child_range
                for k in 1:K
                    new_mat[k, Int(child)] = old_mat[k, old_parent]
                end
            end
        end
        # Coarsened parents: volume-weighted (equal-volume mean) per
        # species.
        n_per_group = 1 << D
        for k_event in eachindex(event.coarsened_parents)
            new_parent = Int(event.coarsened_parents[k_event])
            for k in 1:K
                child_acc = 0.0
                for offset in 0:(n_per_group - 1)
                    old_child = Int(event.removed_old_indices[
                        (k_event - 1) * n_per_group + offset + 1])
                    child_acc += old_mat[k, old_child]
                end
                new_mat[k, new_parent] = child_acc / n_per_group
            end
        end
        # Replace `tm.tracers` storage in place.
        tm.tracers = new_mat
        return nothing
    end
    return register_refinement_listener!(mesh, cb)
end
