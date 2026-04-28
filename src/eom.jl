# eom.jl — HierarchicalGrids-based discrete Euler–Lagrange residual.
#
# Phase M3-0 scope: a thin shim that ports M1's Phase-1 Cholesky-sector
# integrator (single autonomous cell) onto an HG `SimplicialMesh{1, T}`
# + `PolynomialFieldSet` substrate. The numerical kernel — the discrete
# EL residual `cholesky_el_residual` — is reused **byte-identically**
# from `cholesky_sector.jl` so the HG-based path must produce
# bit-identical output to M1 modulo storage layout (no algorithmic
# divergence is allowed in M3-0).
#
# Storage convention (1D, Phase-1 / M3-0):
#
#   - `mesh::SimplicialMesh{1, T}` carries `n_simplices` segments. Each
#     simplex is two vertices; `simplex_volume(mesh, j)` is the
#     Lagrangian segment length (later phases use this as `Δm` after
#     the EOS coupling). For M3-0 (autonomous Cholesky cell) the
#     Lagrangian-mass coordinate is irrelevant; we only need a
#     per-cell container.
#
#   - `fields::PolynomialFieldSet` allocated with `MonomialBasis{1, 0}`
#     (one coefficient per cell — i.e. piecewise constant per simplex)
#     and named scalar fields `(alpha, beta)`. In later phases
#     (M3-1/2) this lifts to higher-order polynomial reconstruction.
#
# The discrete EL residual is exactly M1's `cholesky_el_residual`. The
# Newton iteration is exactly M1's `cholesky_step` (NonlinearSolve
# `SimpleNewtonRaphson` with `AutoForwardDiff`). The new wrapper
# `cholesky_step_HG!` simply reads `(α, β)` from the field set, calls
# `cholesky_step`, and writes the result back. Per-cell parallelism is
# wired through HG's `parallel_for_cells`.
#
# This file deliberately does NOT introduce a new EL residual function.
# M3-0's correctness gate is byte-equality with M1; introducing a new
# kernel risks divergence that the parity tests then have to debug.
# The HG-substrate generalization happens in M3-3 (2D Berry coupling),
# where the residual genuinely changes.

using StaticArrays: SVector
using HierarchicalGrids: SimplicialMesh, PolynomialFieldSet,
    n_simplices, n_elements, field_names, basis_of, n_coeffs,
    HierarchicalMesh, halo_view, enumerate_leaves, is_leaf,
    n_cells as hg_n_cells, cell_physical_box, EulerianFrame,
    face_neighbors, face_neighbors_with_bcs, FrameBoundaries

"""
    read_alphabeta(fields::PolynomialFieldSet, j::Integer)

Read the `(α, β)` pair stored at simplex `j` of an HG order-0
polynomial field set. Returns an `SVector{2,T}` matching M1's
`cholesky_step` calling convention.

The field set must have the structure produced by
`allocate_polynomial_fields(SoA(), MonomialBasis{1, 0}(), n;
alpha=T, beta=T)`. Each polynomial-view's single coefficient is
read via index `[1]`.
"""
@inline function read_alphabeta(fields::PolynomialFieldSet, j::Integer)
    α = fields.alpha[j][1]
    β = fields.beta[j][1]
    T = promote_type(typeof(α), typeof(β))
    return SVector{2,T}(T(α), T(β))
end

"""
    write_alphabeta!(fields::PolynomialFieldSet, j::Integer, q)

Write the `(α, β)` pair `q::SVector{2,T}` to simplex `j`'s order-0
coefficients in the HG field set. Mirrors `read_alphabeta`.
"""
@inline function write_alphabeta!(fields::PolynomialFieldSet,
                                   j::Integer, q)
    fields.alpha[j] = (q[1],)
    fields.beta[j]  = (q[2],)
    return q
end

"""
    cholesky_el_residual_HG(fields, j, q_np1, q_n, M_vv, divu_half, dt)

Per-simplex discrete Euler–Lagrange residual on an HG-based field
set. Algebraically identical to M1's `cholesky_el_residual`; the only
difference is the calling convention (the HG path receives the
field-set + simplex index for context-aware later phases). In Phase
M3-0 the body is a single delegation to `cholesky_el_residual` so
that bit-equality is automatic.

Arguments:
- `fields::PolynomialFieldSet` — current field set (unused in
  Phase M3-0; reserved for later phases that need neighbour data).
- `j::Integer`                  — simplex index (unused in M3-0;
  retained for the future per-cell EOS-coupling phases).
- `q_np1::SVector{2}`           — unknown `(α_{n+1}, β_{n+1})`.
- `q_n::SVector{2}`             — known `(α_n, β_n)`.
- `M_vv`                        — externally-supplied second moment.
- `divu_half`                   — midpoint strain rate.
- `dt`                          — timestep.

Returns: `SVector{2}` residual matching `cholesky_el_residual`.
"""
@inline function cholesky_el_residual_HG(::PolynomialFieldSet,
                                          ::Integer,
                                          q_np1, q_n,
                                          M_vv, divu_half, dt)
    return cholesky_el_residual(q_np1, q_n, M_vv, divu_half, dt)
end

# ─────────────────────────────────────────────────────────────────────
# Phase M3-1: Phase-2/5/5b field-set helpers for the HG path.
# ─────────────────────────────────────────────────────────────────────
#
# The Phase-2 deterministic state per cell (1D) is the 7-tuple
# `(α, β, x, u, s, Pp, Q)`. The HG-side storage uses an order-0
# `PolynomialFieldSet` with seven Float64 scalar fields, one
# coefficient per cell (piecewise constant), keyed by those names.
# The boundary conditions, periodic box length `L_box`, vertex
# half-step momenta `p_half`, and per-segment masses `Δm` are not
# expressible as polynomial-cell fields and ride along on a thin
# wrapper struct (see `src/newton_step_HG.jl`).

"""
    read_detfield(fields::PolynomialFieldSet, j::Integer)

Read the 7-tuple `(x, u, α, β, s, Pp, Q)` stored at simplex `j` of
an HG order-0 polynomial field set populated for the Phase-2/5
state. Returns the M1 `DetField{T}` so downstream consumers (e.g.
the M1 `det_step!` driver) can use the legacy code path
byte-identically.

The field set must have the structure produced by
`allocate_detfield_HG`. Each polynomial-view's single coefficient is
read via index `[1]`.
"""
@inline function read_detfield(fields::PolynomialFieldSet, j::Integer)
    α  = fields.alpha[j][1]
    β  = fields.beta[j][1]
    x  = fields.x[j][1]
    u  = fields.u[j][1]
    s  = fields.s[j][1]
    Pp = fields.Pp[j][1]
    Q  = fields.Q[j][1]
    T = promote_type(typeof(α), typeof(β), typeof(x), typeof(u),
                     typeof(s), typeof(Pp), typeof(Q))
    return DetField{T}(T(x), T(u), T(α), T(β), T(s), T(Pp), T(Q))
end

"""
    write_detfield!(fields::PolynomialFieldSet, j::Integer, det::DetField)

Write the M1 `DetField{T}` `det` to simplex `j`'s order-0 coefficients
in the HG field set. Mirrors `read_detfield`.
"""
@inline function write_detfield!(fields::PolynomialFieldSet,
                                  j::Integer,
                                  det::DetField)
    fields.alpha[j] = (det.α,)
    fields.beta[j]  = (det.β,)
    fields.x[j]     = (det.x,)
    fields.u[j]     = (det.u,)
    fields.s[j]     = (det.s,)
    fields.Pp[j]    = (det.Pp,)
    fields.Q[j]     = (det.Q,)
    return det
end

# ─────────────────────────────────────────────────────────────────────
# Phase M3-3b: native HG-side 2D EL residual (no Berry; θ_R fixed)
# ─────────────────────────────────────────────────────────────────────
#
# This is the first native HG-side EL residual on the 2D substrate. It
# is the dimension-lifted per-axis Cholesky-sector residual from M1's
# 1D `det_el_residual`, with the additions deferred to later sub-phases:
#
#   • NO Berry coupling. The Berry kinetic 1-form
#     `Θ_rot^(2D) = (1/3)(α_1³ β_2 − α_2³ β_1) dθ_R` is omitted from
#     the residual rows; M3-3c will plug it in via `src/berry.jl`.
#   • NO θ_R as a Newton unknown. θ_R is read from the input field set
#     and held fixed across the step. M3-3c will promote it.
#   • NO off-diagonal β_{12}, β_{21}. Per Q3 of the M3-3 design note
#     (`reference/notes_M3_3_2d_cholesky_berry.md` §10), these are
#     omitted from the field set and pinned to zero.
#
# Per-cell unknowns (8 dof per leaf cell):
#
#     (x_1, x_2, u_1, u_2, α_1, α_2, β_1, β_2)
#
# Entropy `s` is frozen across the Newton step (mirrors M1; entropy
# updates are operator-split by Phase 5b's q-dissipation pathway, which
# is not exercised in M3-3b's zero-strain tests). `Pp`, `Q`, `θ_R` are
# read at IC and ride along on the field-set storage.
#
# The residual operates on a flat `y_np1::AbstractVector` of length
# `8 N` packed as
#
#     y[8(i-1) + k] for k = 1:8 ↔ (x_1, x_2, u_1, u_2, α_1, α_2, β_1, β_2)
#
# at leaf-cell `i ∈ 1:N`. The reference state at time `n` is read from
# the same packing into `y_n`; per-cell auxiliary data (s, θ_R, mesh
# geometry, BC spec) come in as flat arrays / NamedTuple parameters.
#
# Pressure stencil. Per axis a ∈ {1, 2}, we use the M1 1D pressure
# stencil applied along that axis: the lo-side neighbor along axis a
# supplies the upstream pressure for the F^u_a momentum residual at
# the cell's lo-face; the hi-side neighbor (via the halo view) supplies
# the downstream pressure at the hi-face. The pressure itself is
# `P = ρ M_vv(J, s)` where for the 2D Eulerian-cell substrate the
# specific volume per axis is `J = (x_a^{n+1/2}_hi - x_a^{n+1/2}_lo) /
# Δm_a` with `Δm_a` the per-axis Lagrangian mass step. For the M3-3b
# zero-strain regression this evaluates identically to M1's 1D form on
# any 1D-symmetric configuration. Higher-fidelity 2D physics (genuinely
# multi-D pressure gradients, vorticity, KH growth) is M3-3c+ scope.

using HierarchicalGrids: HierarchicalMesh, halo_view, enumerate_leaves,
    cell_physical_box, EulerianFrame

"""
    cholesky_el_residual_2D!(F::AbstractVector,
                             y_np1::AbstractVector,
                             y_n::AbstractVector,
                             aux::NamedTuple,
                             dt::Real)

Native HG-side 2D Cholesky-sector EL residual without Berry coupling
(M3-3b). Writes the residual `F` for the 8-dof-per-cell flat unknown
vector `y_np1` against the reference state `y_n`. Both vectors are
packed leaf-cell-major as

    y[8(i-1) + 1] = x_1
    y[8(i-1) + 2] = x_2
    y[8(i-1) + 3] = u_1
    y[8(i-1) + 4] = u_2
    y[8(i-1) + 5] = α_1
    y[8(i-1) + 6] = α_2
    y[8(i-1) + 7] = β_1
    y[8(i-1) + 8] = β_2

for `i ∈ 1:N` (N = number of leaf cells in mesh order).

Auxiliary data carried in `aux::NamedTuple`:

  • `s_vec::AbstractVector{T}`           — frozen entropy per leaf
  • `θR_vec::AbstractVector{T}`          — fixed θ_R per leaf (M3-3b
                                            does not evolve it)
  • `Δm_per_axis::NTuple{2, AbstractVector{T}}` — per-axis Lagrangian
                                            mass step at each leaf
                                            (= ρ_ref × cell extent
                                            along axis a; for the
                                            zero-strain test ρ_ref=1
                                            and Δm_a = cell_size_a)
  • `face_lo_idx::NTuple{2, Vector{Int}}` — per-axis lo-face neighbor
                                            cell idx (0 for boundary;
                                            populated from
                                            `face_neighbors_with_bcs`
                                            so periodic wrap-around is
                                            already resolved)
  • `face_hi_idx::NTuple{2, Vector{Int}}` — per-axis hi-face neighbor
                                            cell idx
  • `M_vv_override::Union{Nothing, NTuple{2, T}}` — when not `nothing`,
                                            override `M_vv(J, s)` per
                                            axis with the supplied
                                            constants; used by M3-3b's
                                            unit tests to decouple from
                                            EOS specifics
  • `ρ_ref::T`                            — reference density used to
                                            convert `Δm_a → cell extent
                                            and to compute the per-axis
                                            pressure `P = ρ_ref · M_vv`.
                                            For zero-strain ICs the
                                            density is uniform and this
                                            simple form suffices; M3-3c
                                            will reintroduce the
                                            J-dependent EOS coupling
                                            for genuine compressive
                                            flow.

The residual is the per-axis lift of M1's `det_el_residual` (see
`src/cholesky_sector.jl`) applied independently along axis a = 1, 2.
Term-by-term per axis (mirroring M1 §3.2):

    F^x_a    = (x_a^{n+1} − x_a^n)/dt − ū_a
    F^u_a    = (u_a^{n+1} − u_a^n)/dt + (P̄_a^hi − P̄_a^lo) / m̄_a
    F^α_a    = (α_a^{n+1} − α_a^n)/dt − β̄_a                            [D_t^{(0)} α_a = β_a]
    F^β_a    = (β_a^{n+1} − β_a^n)/dt + (∂_a u_a) β̄_a − γ²_a / ᾱ_a    [D_t^{(1)} β_a = γ²/α]

where `(∂_a u_a) = (ū_a^hi − ū_a^lo) / Δx̄_a` and `γ²_a = M_vv_a − β̄_a²`.

Boundary handling: when `face_lo_idx[a][i] == 0`, the residual treats
the lo-face as a wall (mirror) — pressure equal to the cell's own,
velocity equal to the cell's own — so no spurious gradient at the
boundary. Periodic wrap should already be resolved upstream by
`face_neighbors_with_bcs`; this `0`-as-wall fallback only fires for
genuinely closed boundaries (REFLECTING / DIRICHLET-pinned).

# Verification

  • For the cold-limit fixed point (`M_vv = 0` on every axis,
    `β = 0`, `u = 0`, uniform cells), the residual evaluates to the
    machine-precision zero vector when `y_n == y_np1`.
  • In a 1D-symmetric configuration (axis 2 trivial / fixed-point),
    the axis-1 sub-residual reduces bit-for-bit to M1's 1D
    `det_el_residual` per cell — this is the M3-3b dimension-lift gate
    (§6.1 of the design note).
"""
function cholesky_el_residual_2D!(F::AbstractVector,
                                   y_np1::AbstractVector,
                                   y_n::AbstractVector,
                                   aux::NamedTuple,
                                   dt::Real)
    s_vec       = aux.s_vec
    Δm_per_axis = aux.Δm_per_axis
    face_lo     = aux.face_lo_idx
    face_hi     = aux.face_hi_idx
    M_vv_over   = aux.M_vv_override
    ρ_ref       = aux.ρ_ref
    # M3-4: optional per-axis-per-cell coordinate-wrap offsets for
    # periodic axes. When absent (e.g., legacy callers built before
    # M3-4), default to zero so the residual stays bit-exact.
    wrap_lo     = haskey(aux, :wrap_lo_idx) ? aux.wrap_lo_idx : nothing
    wrap_hi     = haskey(aux, :wrap_hi_idx) ? aux.wrap_hi_idx : nothing

    N = length(s_vec)
    @assert length(y_np1) == 8 * N "y_np1 length $(length(y_np1)) does not match 8 * N = $(8 * N)"
    @assert length(y_n)   == 8 * N
    @assert length(F)     == 8 * N
    @assert length(Δm_per_axis[1]) == N
    @assert length(Δm_per_axis[2]) == N

    Tres = promote_type(eltype(y_np1), eltype(y_n), eltype(s_vec), typeof(dt))

    # Per-cell index helpers (closed over y arrays).
    @inline get_x(y, a, i) = y[8 * (i - 1) + a]              # a ∈ {1, 2}
    @inline get_u(y, a, i) = y[8 * (i - 1) + 2 + a]
    @inline get_α(y, a, i) = y[8 * (i - 1) + 4 + a]
    @inline get_β(y, a, i) = y[8 * (i - 1) + 6 + a]

    @inbounds for i in 1:N
        s_i = s_vec[i]
        for a in 1:2
            # Self midpoints.
            x_n   = get_x(y_n,   a, i)
            x_np1 = get_x(y_np1, a, i)
            u_n   = get_u(y_n,   a, i)
            u_np1 = get_u(y_np1, a, i)
            α_n   = get_α(y_n,   a, i)
            α_np1 = get_α(y_np1, a, i)
            β_n   = get_β(y_n,   a, i)
            β_np1 = get_β(y_np1, a, i)

            x̄ = (x_n   + x_np1)   / 2
            ū = (u_n   + u_np1)   / 2
            ᾱ = (α_n   + α_np1)   / 2
            β̄ = (β_n   + β_np1)   / 2

            # Neighbor along axis a (lo, hi).
            ilo = face_lo[a][i]
            ihi = face_hi[a][i]

            # M3-4 periodic wrap: when this lo/hi-face neighbor is the
            # periodic wrap from the opposite wall, its stored x_a is on
            # the opposite side of the box; shift by ±L_a so the discrete
            # gradient stencil sees a consistent monotonic coordinate
            # (mirror of the 1D `+L_box` wrap at j == N).
            wrap_lo_off = wrap_lo === nothing ? zero(Tres) : Tres(wrap_lo[a][i])
            wrap_hi_off = wrap_hi === nothing ? zero(Tres) : Tres(wrap_hi[a][i])

            # Lo-face neighbor data; mirror-self when out-of-domain.
            if ilo == 0
                x_lo_n   = x_n
                x_lo_np1 = x_np1
                u_lo_n   = u_n
                u_lo_np1 = u_np1
                α_lo_n   = α_n
                α_lo_np1 = α_np1
                β_lo_n   = β_n
                β_lo_np1 = β_np1
                s_lo     = s_i
            else
                x_lo_n   = get_x(y_n,   a, ilo) + wrap_lo_off
                x_lo_np1 = get_x(y_np1, a, ilo) + wrap_lo_off
                u_lo_n   = get_u(y_n,   a, ilo)
                u_lo_np1 = get_u(y_np1, a, ilo)
                α_lo_n   = get_α(y_n,   a, ilo)
                α_lo_np1 = get_α(y_np1, a, ilo)
                β_lo_n   = get_β(y_n,   a, ilo)
                β_lo_np1 = get_β(y_np1, a, ilo)
                s_lo     = s_vec[ilo]
            end
            x̄_lo = (x_lo_n + x_lo_np1) / 2
            ū_lo = (u_lo_n + u_lo_np1) / 2

            # Hi-face neighbor data; mirror-self when out-of-domain.
            if ihi == 0
                x_hi_n   = x_n
                x_hi_np1 = x_np1
                u_hi_n   = u_n
                u_hi_np1 = u_np1
                α_hi_n   = α_n
                α_hi_np1 = α_np1
                β_hi_n   = β_n
                β_hi_np1 = β_np1
                s_hi     = s_i
            else
                x_hi_n   = get_x(y_n,   a, ihi) + wrap_hi_off
                x_hi_np1 = get_x(y_np1, a, ihi) + wrap_hi_off
                u_hi_n   = get_u(y_n,   a, ihi)
                u_hi_np1 = get_u(y_np1, a, ihi)
                α_hi_n   = get_α(y_n,   a, ihi)
                α_hi_np1 = get_α(y_np1, a, ihi)
                β_hi_n   = get_β(y_n,   a, ihi)
                β_hi_np1 = get_β(y_np1, a, ihi)
                s_hi     = s_vec[ihi]
            end
            x̄_hi = (x_hi_n + x_hi_np1) / 2
            ū_hi = (u_hi_n + u_hi_np1) / 2

            # Per-axis Lagrangian mass step, taken at the cell.
            Δm_i = Δm_per_axis[a][i]

            # Per-axis M_vv. For the M3-3b zero-strain regression we
            # consume `M_vv_override` directly so the test decouples
            # from the EOS thermodynamic factorisation; the J-dependent
            # `Mvv(J, s)` form is reinstated for genuine compressive
            # flow in M3-3c+.
            M̄vv_a = if M_vv_over !== nothing
                Tres(M_vv_over[a])
            else
                # Cell-centred J for the EOS branch; computed from the
                # cell's own mid-step extent (lo→hi half-cell extents
                # would couple boundary cells differently from interior
                # cells, breaking the dimension-lift gate). For zero-
                # strain ICs the cell extent does not change so this
                # is a constant.
                # Use the lo→hi extent normalised by 2·Δm_i if both
                # neighbours are present; otherwise fall back to a
                # one-sided stencil with the present neighbour.
                Δx_avg = if ilo == 0 && ihi == 0
                    zero(Tres)
                elseif ilo == 0
                    # mirror lo: x̄ - x̄_lo = 0, use x̄_hi - x̄ as the half-cell
                    # extent and double for full cell.
                    2 * (x̄_hi - x̄)
                elseif ihi == 0
                    2 * (x̄ - x̄_lo)
                else
                    x̄_hi - x̄_lo
                end
                J̄_self = Δx_avg > 0 ? Δx_avg / (2 * Δm_i) : zero(Tres)
                J̄_self > 0 ? Mvv(J̄_self, s_i) : zero(Tres)
            end

            # Pressure per axis. With ρ ≈ ρ_ref (uniform IC) the
            # cell pressure is `P = ρ_ref · M_vv`. The face pressures
            # are then mid-point averages of the two adjacent cell
            # pressures. For the M3-3b zero-strain test all cells share
            # `M_vv` per axis so face pressures are identical to cell
            # pressures, the gradient is zero, and `F^u_a = 0`.
            P_self = Tres(ρ_ref) * M̄vv_a
            P_lo_neighbor = if ilo == 0
                P_self
            else
                Mvv_lo = if M_vv_over !== nothing
                    Tres(M_vv_over[a])
                else
                    Tres(ρ_ref) * Mvv(one(Tres) / Tres(ρ_ref), s_lo)
                end
                Tres(ρ_ref) * Mvv_lo
            end
            P_hi_neighbor = if ihi == 0
                P_self
            else
                Mvv_hi = if M_vv_over !== nothing
                    Tres(M_vv_over[a])
                else
                    Tres(ρ_ref) * Mvv(one(Tres) / Tres(ρ_ref), s_hi)
                end
                Tres(ρ_ref) * Mvv_hi
            end
            P̄_lo = (P_self + P_lo_neighbor) / 2
            P̄_hi = (P_self + P_hi_neighbor) / 2

            # Self-cell strain rate along axis a (Eulerian (∂_a u_a) at
            # the cell midpoint), using the lo↔hi extent.
            Δx̄_full = x̄_hi - x̄_lo
            div̄u_a = if Δx̄_full > 0
                (ū_hi - ū_lo) / Δx̄_full
            else
                zero(Tres)
            end

            # Lagrangian mass at cell faces — for the F^u stencil.
            # Use the per-axis Δm_i directly (cell-centered momentum
            # equation; no half-mass averaging needed here because both
            # P̄_lo and P̄_hi are face pressures).
            m̄_a = Δm_i

            # γ² per axis.
            γ²_a = M̄vv_a - β̄^2

            # Residual rows.
            base = 8 * (i - 1)
            F[base + a]     = (x_np1 - x_n) / dt - ū                         # F^x_a
            F[base + 2 + a] = (u_np1 - u_n) / dt + (P̄_hi - P̄_lo) / m̄_a       # F^u_a
            # F^α_a: D_t^{(0)} α = β  (charge 0, no strain coupling)
            F[base + 4 + a] = (α_np1 - α_n) / dt - β̄                         # F^α_a
            # F^β_a: D_t^{(1)} β = γ²/α  (charge 1; per-axis strain coupling)
            F[base + 6 + a] = (β_np1 - β_n) / dt + div̄u_a * β̄ -
                              (ᾱ != 0 ? γ²_a / ᾱ : zero(Tres))                # F^β_a
        end
    end
    return F
end

"""
    cholesky_el_residual_2D(y_np1, y_n, aux, dt)

Allocating wrapper around `cholesky_el_residual_2D!`. Returns a fresh
residual vector of the same eltype and length as `y_np1`. Used in
tests where allocation cost is irrelevant.
"""
function cholesky_el_residual_2D(y_np1::AbstractVector,
                                  y_n::AbstractVector,
                                  aux::NamedTuple,
                                  dt::Real)
    Tres = promote_type(eltype(y_np1), eltype(y_n), eltype(aux.s_vec), typeof(dt))
    F = similar(y_np1, Tres, length(y_np1))
    cholesky_el_residual_2D!(F, y_np1, y_n, aux, dt)
    return F
end

# ─────────────────────────────────────────────────────────────────────
# 2D field-set ↔ flat state-vector packing helpers (M3-3b)
# ─────────────────────────────────────────────────────────────────────

"""
    pack_state_2d(fields::PolynomialFieldSet, leaves::AbstractVector{<:Integer})
        -> Vector{Float64}

Pack the per-leaf 8-dof Newton state `(x_1, x_2, u_1, u_2, α_1, α_2, β_1, β_2)`
out of the M3-3a 2D Cholesky-sector field set `fields` into a flat
length-`8 N` vector, where `N = length(leaves)`. Iteration order is
`leaves[k]` for `k ∈ 1:N`. The post-Newton sectors `Pp, Q, θ_R, s` are
NOT packed — they are read directly from the field set as auxiliary
data (see `cholesky_el_residual_2D!` `aux`).
"""
function pack_state_2d(fields::PolynomialFieldSet,
                        leaves::AbstractVector{<:Integer})
    N = length(leaves)
    # `PolynomialView`'s eltype is `Any` (the field set is heterogeneous);
    # the actual scalar coefficient type is `typeof(view[1])`.
    T = typeof(fields.x_1[leaves[1]][1])
    y = Vector{T}(undef, 8 * N)
    @inbounds for (i, ci) in enumerate(leaves)
        base = 8 * (i - 1)
        y[base + 1] = fields.x_1[ci][1]
        y[base + 2] = fields.x_2[ci][1]
        y[base + 3] = fields.u_1[ci][1]
        y[base + 4] = fields.u_2[ci][1]
        y[base + 5] = fields.α_1[ci][1]
        y[base + 6] = fields.α_2[ci][1]
        y[base + 7] = fields.β_1[ci][1]
        y[base + 8] = fields.β_2[ci][1]
    end
    return y
end

"""
    unpack_state_2d!(fields::PolynomialFieldSet,
                      leaves::AbstractVector{<:Integer},
                      y::AbstractVector)

Inverse of `pack_state_2d`: write the flat 8-dof per-leaf state back
into the 2D field set. Leaves the `:θ_R, :s, :Pp, :Q` slots untouched.
"""
function unpack_state_2d!(fields::PolynomialFieldSet,
                           leaves::AbstractVector{<:Integer},
                           y::AbstractVector)
    N = length(leaves)
    @assert length(y) == 8 * N "y length $(length(y)) does not match 8 * N = $(8 * N)"
    @inbounds for (i, ci) in enumerate(leaves)
        base = 8 * (i - 1)
        fields.x_1[ci] = (y[base + 1],)
        fields.x_2[ci] = (y[base + 2],)
        fields.u_1[ci] = (y[base + 3],)
        fields.u_2[ci] = (y[base + 4],)
        fields.α_1[ci] = (y[base + 5],)
        fields.α_2[ci] = (y[base + 6],)
        fields.β_1[ci] = (y[base + 7],)
        fields.β_2[ci] = (y[base + 8],)
    end
    return fields
end

"""
    build_face_neighbor_tables(mesh::HierarchicalMesh{2},
                                leaves::AbstractVector{<:Integer},
                                bc_spec)
        -> (face_lo_idx::NTuple{2, Vector{Int}},
            face_hi_idx::NTuple{2, Vector{Int}})

Pre-compute per-leaf face-neighbor leaf indices for both axes. Output
is a pair of length-`N` `Int` vectors per axis, where `face_lo_idx[a][i]`
is the leaf-major index of the lo-side neighbor of `leaves[i]` along
axis `a` (or 0 if out-of-domain after BC processing). Built once per
mesh; consumed by `cholesky_el_residual_2D!` to drive the per-axis
pressure stencil.

`bc_spec` is forwarded to `face_neighbors_with_bcs` so periodic
wrap-around is already resolved before the residual sees the table.
"""
function build_face_neighbor_tables(mesh::HierarchicalMesh{2},
                                     leaves::AbstractVector{<:Integer},
                                     bc_spec)
    N = length(leaves)
    # Map mesh-cell-index -> leaf-major position (1..N), 0 if not a leaf.
    pos = zeros(Int, hg_n_cells(mesh))
    @inbounds for (k, ci) in enumerate(leaves)
        pos[ci] = k
    end
    face_lo_1 = zeros(Int, N)
    face_hi_1 = zeros(Int, N)
    face_lo_2 = zeros(Int, N)
    face_hi_2 = zeros(Int, N)
    @inbounds for (i, ci) in enumerate(leaves)
        fn = face_neighbors_with_bcs(mesh, ci, bc_spec)
        # face order: (axis 1 lo, axis 1 hi, axis 2 lo, axis 2 hi)
        face_lo_1[i] = fn[1] == 0 ? 0 : pos[Int(fn[1])]
        face_hi_1[i] = fn[2] == 0 ? 0 : pos[Int(fn[2])]
        face_lo_2[i] = fn[3] == 0 ? 0 : pos[Int(fn[3])]
        face_hi_2[i] = fn[4] == 0 ? 0 : pos[Int(fn[4])]
    end
    return ((face_lo_1, face_lo_2), (face_hi_1, face_hi_2))
end

"""
    build_periodic_wrap_tables(mesh::HierarchicalMesh{2},
                                frame::EulerianFrame{2, T},
                                leaves::AbstractVector{<:Integer},
                                face_lo_idx::NTuple{2, Vector{Int}},
                                face_hi_idx::NTuple{2, Vector{Int}})
        -> (wrap_lo::NTuple{2, Vector{T}}, wrap_hi::NTuple{2, Vector{T}})

Pre-compute per-leaf-per-axis additive coordinate-wrap offsets for the
2D EL residual. For each leaf `i ∈ 1:N` and each axis `a ∈ {1, 2}`,
returns `wrap_lo[a][i]` and `wrap_hi[a][i]` such that the periodic
neighbor's `x_a` should be read as

    x_a_lo_neighbor + wrap_lo[a][i],   x_a_hi_neighbor + wrap_hi[a][i]

so the lo→hi extent across a periodic seam is positive and equal to
the geometric cell spacing — mirroring the 1D path's `+L_box` wrap on
`x_right` at `j == N` (see `src/cholesky_sector.jl::det_el_residual`,
M3-3c handoff item).

Detection rule: if the neighbor's physical box on axis `a` lies on the
opposite wall (i.e., `box_neighbor.hi[a] ≤ box_self.lo[a] + ε` for the
lo-face, or `box_neighbor.lo[a] ≥ box_self.hi[a] − ε` for the hi-face)
then the neighbor is the periodic wrap; the offset is `-L_a` for the
lo-face and `+L_a` for the hi-face, where `L_a = frame.hi[a] − frame.lo[a]`.
For interior cells (real adjacency, not a wrap), the offset is zero.
For boundary cells with `face_*_idx == 0` (out-of-domain;
non-periodic BC), the offset is also zero — the residual treats those
as mirror-self.

The output is a pair of `(NTuple{2, Vector{T}}, NTuple{2, Vector{T}})`,
where `wrap_lo[a]` and `wrap_hi[a]` are length-`N` `T`-vectors.
"""
function build_periodic_wrap_tables(mesh::HierarchicalMesh{2},
                                     frame::EulerianFrame{2, T},
                                     leaves::AbstractVector{<:Integer},
                                     face_lo_idx::NTuple{2, Vector{Int}},
                                     face_hi_idx::NTuple{2, Vector{Int}}
                                     ) where {T}
    N = length(leaves)
    L = ntuple(a -> T(frame.hi[a] - frame.lo[a]), 2)
    wrap_lo_1 = zeros(T, N)
    wrap_hi_1 = zeros(T, N)
    wrap_lo_2 = zeros(T, N)
    wrap_hi_2 = zeros(T, N)
    eps_a = ntuple(a -> T(1e-12) * L[a], 2)

    @inbounds for (i, ci) in enumerate(leaves)
        lo_self, hi_self = cell_physical_box(frame, ci)
        for a in 1:2
            wrap_arrs_lo = a == 1 ? wrap_lo_1 : wrap_lo_2
            wrap_arrs_hi = a == 1 ? wrap_hi_1 : wrap_hi_2
            face_lo = face_lo_idx[a]
            face_hi = face_hi_idx[a]

            # Lo-face wrap detection.
            ilo = face_lo[i]
            if ilo != 0
                ci_lo = leaves[ilo]
                lo_n, hi_n = cell_physical_box(frame, ci_lo)
                # If neighbor's box sits on the hi-wall side (i.e.,
                # `lo_n[a] >= hi_self[a]`), the neighbor is the
                # periodic wrap from the lo-face, so its physical x_a
                # is L_a too high and should be shifted by -L_a.
                if lo_n[a] >= hi_self[a] - eps_a[a]
                    wrap_arrs_lo[i] = -L[a]
                end
            end

            # Hi-face wrap detection.
            ihi = face_hi[i]
            if ihi != 0
                ci_hi = leaves[ihi]
                lo_n, hi_n = cell_physical_box(frame, ci_hi)
                # If neighbor's box sits on the lo-wall side (i.e.,
                # `hi_n[a] <= lo_self[a]`), the neighbor is the
                # periodic wrap from the hi-face; shift by +L_a.
                if hi_n[a] <= lo_self[a] + eps_a[a]
                    wrap_arrs_hi[i] = L[a]
                end
            end
        end
    end
    return ((wrap_lo_1, wrap_lo_2), (wrap_hi_1, wrap_hi_2))
end

"""
    build_residual_aux_2D(fields::PolynomialFieldSet,
                          mesh::HierarchicalMesh{2},
                          frame::EulerianFrame{2, T},
                          leaves::AbstractVector{<:Integer},
                          bc_spec;
                          M_vv_override = nothing,
                          ρ_ref = 1.0) where {T}

Convenience builder for the `aux` NamedTuple consumed by
`cholesky_el_residual_2D!`. Reads `s` and `θ_R` per leaf from `fields`,
extracts per-axis cell extents from the Eulerian frame, and computes
per-axis Lagrangian mass steps `Δm_a = ρ_ref * extent_a` per cell.
"""
function build_residual_aux_2D(fields::PolynomialFieldSet,
                                mesh::HierarchicalMesh{2},
                                frame::EulerianFrame{2, T},
                                leaves::AbstractVector{<:Integer},
                                bc_spec;
                                M_vv_override = nothing,
                                ρ_ref::Real = 1.0) where {T}
    N = length(leaves)
    s_vec  = Vector{T}(undef, N)
    θR_vec = Vector{T}(undef, N)
    Δm_1   = Vector{T}(undef, N)
    Δm_2   = Vector{T}(undef, N)
    ρ_t = T(ρ_ref)
    @inbounds for (i, ci) in enumerate(leaves)
        s_vec[i]  = fields.s[ci][1]
        θR_vec[i] = fields.θ_R[ci][1]
        lo, hi = cell_physical_box(frame, ci)
        Δm_1[i] = ρ_t * (hi[1] - lo[1])
        Δm_2[i] = ρ_t * (hi[2] - lo[2])
    end
    face_lo_idx, face_hi_idx = build_face_neighbor_tables(mesh, leaves, bc_spec)
    # M3-4: per-axis-per-cell coordinate-wrap offsets for periodic axes.
    # For non-periodic axes the offsets are zero, so the residual stays
    # bit-exact to M3-3c on REFLECTING/INFLOW/OUTFLOW configurations.
    wrap_lo_idx, wrap_hi_idx = build_periodic_wrap_tables(mesh, frame, leaves,
                                                            face_lo_idx,
                                                            face_hi_idx)
    return (
        s_vec       = s_vec,
        θR_vec      = θR_vec,
        Δm_per_axis = (Δm_1, Δm_2),
        face_lo_idx = face_lo_idx,
        face_hi_idx = face_hi_idx,
        wrap_lo_idx = wrap_lo_idx,
        wrap_hi_idx = wrap_hi_idx,
        M_vv_override = M_vv_override,
        ρ_ref       = ρ_t,
    )
end

# ─────────────────────────────────────────────────────────────────────
# Phase M3-3c: Berry coupling + θ_R Newton unknown
# ─────────────────────────────────────────────────────────────────────
#
# M3-3c extends the M3-3b residual by:
#
#   1. Promoting θ_R from an auxiliary fixed value to a Newton unknown.
#      Per-cell unknowns grow from 8 to 9:
#
#          (x_1, x_2, u_1, u_2, α_1, α_2, β_1, β_2, θ_R).
#
#      The flat residual / unknown vector is therefore length 9 N.
#
#   2. Adding the Berry-coupling contributions to the per-axis (α_a, β_a)
#      rows. The closed-form Berry partials live in `src/berry.jl`. The
#      sign convention is derived from the rows of Ω · X = -dH where Ω
#      is the closed 5×5 symplectic / Poisson form
#      `ω_{2D} = α_1²dα_1∧dβ_1 + α_2²dα_2∧dβ_2 + dF∧dθ_R` with
#      `F = (α_1³β_2 − α_2³β_1)/3`. See
#      `reference/notes_M3_phase0_berry_connection.md` §6 and
#      `reference/notes_M3_3_2d_cholesky_berry.md` §4.
#
#      The Berry-modified per-axis Hamilton equations are
#
#          α̇_1 = β_1 − (α_2³/(3α_1²)) · θ̇_R
#          α̇_2 = β_2 + (α_1³/(3α_2²)) · θ̇_R
#          β̇_1 = γ_1²/α_1 − β_2 · θ̇_R
#          β̇_2 = γ_2²/α_2 + β_1 · θ̇_R
#
#      The discrete EL residual rows mirror M1's midpoint convention
#      (see `src/cholesky_sector.jl::cholesky_el_residual`), with the
#      Berry contribution added at midpoint state and `θ̇_R = (θ_R_np1
#      − θ_R_n)/dt`:
#
#          F^α_1 = (α_1_np1 − α_1_n)/dt − β̄_1 + (ᾱ_2³/(3ᾱ_1²)) · θ̇_R_h
#          F^α_2 = (α_2_np1 − α_2_n)/dt − β̄_2 − (ᾱ_1³/(3ᾱ_2²)) · θ̇_R_h
#          F^β_1 = (β_1_np1 − β_1_n)/dt + (∂_1 u_1) β̄_1 − γ̄_1²/ᾱ_1 + β̄_2 θ̇_R_h
#          F^β_2 = (β_2_np1 − β_2_n)/dt + (∂_2 u_2) β̄_2 − γ̄_2²/ᾱ_2 − β̄_1 θ̇_R_h
#
#      At the 1D-symmetric slice (α_2 = const, β_2 = 0, θ_R = 0,
#      θ_R_np1 = 0 by construction since the trivial θ_R row pins it),
#      every Berry-modification term vanishes identically (β̄_2 = 0
#      makes `(ᾱ_2³/(3ᾱ_1²)) θ̇_R_h = 0` only if θ̇_R_h = 0; the
#      θ_R-row pinning ensures this) — so the dimension-lift gate
#      §6.1 holds at 0.0 absolute exactly as in M3-3b.
#
#   3. Adding a 9th residual row F^θ_R encoding the kinematic equation
#      `θ̇_R = drive` from the Berry derivation §3. In the M3-3c first
#      cut (no off-diagonal β, no off-diagonal velocity gradient stencil
#      yet), the kinematic drive `W̃_12 − S̃_12·… + 2M̃_xv,12/…` reduces
#      to 0 — so
#
#          F^θ_R = (θ_R_np1 − θ_R_n)/dt
#
#      i.e., θ_R is conserved per cell. This is the analog of the M1
#      §6.6 H_rot solvability constraint embedded structurally in the
#      residual: with the per-axis rows derived from Ω · X = -dH and
#      F^θ_R = θ̇_R, the discrete system inherits the continuous
#      solvability identity (cross-checked in `test_M3_3c_h_rot_solvability.jl`).
#      M3-3d / M3-6 will activate non-trivial off-diagonal strain-rate
#      drives (D.1 KH falsifier).
#
# Off-diagonal β is still pinned (per Q3 of the design note's §10);
# `β_{12} = β_{21} = 0` is enforced by their absence from the field set.

"""
    cholesky_el_residual_2D_berry!(F, y_np1, y_n, aux, dt)

Native HG-side 2D Cholesky-sector EL residual **with Berry coupling,
θ_R as a Newton unknown, and (M3-6 Phase 0) the off-diagonal Cholesky
pair `β_12, β_21` re-activated**. Writes the residual `F` for the
11-dof-per-cell flat unknown vector `y_np1` against the reference
state `y_n`. Both vectors are packed leaf-cell-major as

    y[11(i-1) + 1 ] = x_1
    y[11(i-1) + 2 ] = x_2
    y[11(i-1) + 3 ] = u_1
    y[11(i-1) + 4 ] = u_2
    y[11(i-1) + 5 ] = α_1
    y[11(i-1) + 6 ] = α_2
    y[11(i-1) + 7 ] = β_1
    y[11(i-1) + 8 ] = β_2
    y[11(i-1) + 9 ] = β_12   (M3-6 Phase 0)
    y[11(i-1) + 10] = β_21   (M3-6 Phase 0)
    y[11(i-1) + 11] = θ_R

for `i ∈ 1:N` (N = number of leaf cells in mesh order). M3-3c had 9
dof per cell; M3-6 Phase 0 adds two new rows for `β_12, β_21`.

Auxiliary data carried in `aux::NamedTuple` (same layout as the M3-3b
`build_residual_aux_2D`; `θR_vec` is now read at `n` only — `θ_R_np1`
is a Newton unknown):

  • `s_vec`, `Δm_per_axis`, `face_lo_idx`, `face_hi_idx`,
    `M_vv_override`, `ρ_ref` — see `cholesky_el_residual_2D!`.

The residual rows are:

    F^x_a    = (x_a_np1 − x_a_n)/dt − ū_a
    F^u_a    = (u_a_np1 − u_a_n)/dt + (P̄_a^hi − P̄_a^lo) / m̄_a
    F^α_a    = (α_a_np1 − α_a_n)/dt − β̄_a + (Berry α-modification)
    F^β_a    = (β_a_np1 − β_a_n)/dt + (∂_a u_a) β̄_a − γ̄_a²/ᾱ_a
                + (Berry β-modification, M3-3c + M3-6 Phase 0 + M3-6 Phase 1a)
    F^β_12   = (β_12_np1 − β_12_n)/dt + G̃_12·ᾱ_2/2    (M3-6 Phase 1a strain drive)
    F^β_21   = (β_21_np1 − β_21_n)/dt + G̃_12·ᾱ_1/2    (M3-6 Phase 1a strain drive)
    F^θ_R    = (θ_R_np1 − θ_R_n)/dt + W_12·F_off       (M3-6 Phase 1a vorticity drive)

with G̃_12 = (∂_2 u_1 + ∂_1 u_2)/2 (symmetric strain), W_12 =
(∂_2 u_1 − ∂_1 u_2)/2 (antisymmetric / vorticity), and F_off =
(α_1²·α_2·β_12 − α_1·α_2²·β_21)/2 the off-diagonal Berry function
(`scripts/verify_berry_connection_offdiag.py`). At axis-aligned ICs
(every M3-3c regression and M3-4 driver test, plus M3-6 Phase 0)
the off-diagonal velocity gradients vanish identically ⇒ G̃_12 = W_12
= 0 ⇒ every Phase 1a addition vanishes and the residual reduces byte-
equally to the M3-6 Phase 0 form. See `test_M3_6_phase1a_strain_coupling.jl`.

with the per-axis Berry α/β-modifications detailed in the file-level
comment block above and consistent with `src/berry.jl::berry_partials_2d`
(diagonal block) and `kinetic_offdiag_coeffs_2d` (M3-6 Phase 0
off-diagonal block; sign convention from
`scripts/verify_berry_connection_offdiag.py`).

# Verification

  • The §Dimension-lift gate (M3-3c §6.1; M3-6 Phase 0 reverification)
    holds at 0.0 absolute when `α_2 = const, β_2 = 0, β_12 = β_21 = 0,
    θ_R_n = 0` (the Berry α/β-modification terms vanish; F^θ_R, F^β_12,
    F^β_21 = 0 keep their unknowns at 0 across the step).
  • The Berry partials (∂F^α_a/∂θ_R_np1, ∂F^β_a/∂θ_R_np1) match
    `berry_partials_2d` up to the (1/dt) factor and the midpoint
    averaging — verified in `test_M3_3c_berry_residual.jl`.
  • The §Berry-offdiag CHECKs 1-9 are reproduced in
    `test_M3_6_phase0_offdiag_residual.jl` at the residual-Jacobian
    level (FD probes of ∂F^β_a / ∂β_12_np1, ∂F^β_a / ∂β_21_np1, plus
    iso-pullback ε-expansion and the diagonal-reduction byte-equal
    gate).
"""
function cholesky_el_residual_2D_berry!(F::AbstractVector,
                                          y_np1::AbstractVector,
                                          y_n::AbstractVector,
                                          aux::NamedTuple,
                                          dt::Real)
    s_vec       = aux.s_vec
    Δm_per_axis = aux.Δm_per_axis
    face_lo     = aux.face_lo_idx
    face_hi     = aux.face_hi_idx
    M_vv_over   = aux.M_vv_override
    ρ_ref       = aux.ρ_ref
    # M3-4: optional per-axis-per-cell coordinate-wrap offsets.
    wrap_lo     = haskey(aux, :wrap_lo_idx) ? aux.wrap_lo_idx : nothing
    wrap_hi     = haskey(aux, :wrap_hi_idx) ? aux.wrap_hi_idx : nothing

    N = length(s_vec)
    # M3-6 Phase 0: 11 dof per cell.
    #   y[11(i-1) + 1 ] = x_1
    #   y[11(i-1) + 2 ] = x_2
    #   y[11(i-1) + 3 ] = u_1
    #   y[11(i-1) + 4 ] = u_2
    #   y[11(i-1) + 5 ] = α_1
    #   y[11(i-1) + 6 ] = α_2
    #   y[11(i-1) + 7 ] = β_1
    #   y[11(i-1) + 8 ] = β_2
    #   y[11(i-1) + 9 ] = β_12   (M3-6 Phase 0)
    #   y[11(i-1) + 10] = β_21   (M3-6 Phase 0)
    #   y[11(i-1) + 11] = θ_R
    @assert length(y_np1) == 11 * N "y_np1 length $(length(y_np1)) does not match 11 * N = $(11 * N)"
    @assert length(y_n)   == 11 * N
    @assert length(F)     == 11 * N
    @assert length(Δm_per_axis[1]) == N
    @assert length(Δm_per_axis[2]) == N

    Tres = promote_type(eltype(y_np1), eltype(y_n), eltype(s_vec), typeof(dt))

    # 11-dof per-cell index helpers (M3-6 Phase 0).
    @inline get_x(y, a, i) = y[11 * (i - 1) + a]              # a ∈ {1, 2}
    @inline get_u(y, a, i) = y[11 * (i - 1) + 2 + a]
    @inline get_α(y, a, i) = y[11 * (i - 1) + 4 + a]
    @inline get_β(y, a, i) = y[11 * (i - 1) + 6 + a]
    @inline get_β12(y, i)  = y[11 * (i - 1) + 9]
    @inline get_β21(y, i)  = y[11 * (i - 1) + 10]
    @inline get_θR(y, i)   = y[11 * (i - 1) + 11]

    @inbounds for i in 1:N
        s_i = s_vec[i]

        # Self θ_R midpoint (shared by both axes).
        θR_n   = get_θR(y_n,   i)
        θR_np1 = get_θR(y_np1, i)
        dθR_dt = (θR_np1 - θR_n) / Tres(dt)

        # M3-6 Phase 0: off-diagonal Cholesky entries `β_12, β_21` and
        # their time derivatives. The corrected sign convention from
        # `scripts/verify_berry_connection_offdiag.py` puts them in the
        # symplectic form Ω with the entries
        #     Ω[α_1, β_12] = -α_2² / 2,   Ω[α_1, β_21] = -α_1·α_2,
        #     Ω[α_2, β_12] = -α_1·α_2,   Ω[α_2, β_21] = -α_1² / 2,
        # and the Berry coefficient acquires
        #     ΔΩ[α_1, θ_R] = +α_1·α_2·β_12 - (α_2²/2)·β_21,
        #     ΔΩ[α_2, θ_R] = +(α_1²/2)·β_12 - α_1·α_2·β_21.
        # Inverting the per-axis Hamilton equations from rows α_a of
        # Ω·X + dH = 0 yields the F^β_a couplings below; the
        # constraint rows for β_12, β_21 themselves are kinematic-
        # drive equations identical in shape to F^θ_R (free-flight
        # / no-off-diag-strain limit ⇒ trivial drive ⇒ they are
        # conserved per cell, the M3-3c regression configurations
        # IC them at zero so they stay at zero).
        β12_n   = get_β12(y_n,   i)
        β12_np1 = get_β12(y_np1, i)
        β21_n   = get_β21(y_n,   i)
        β21_np1 = get_β21(y_np1, i)
        dβ12_dt = (β12_np1 - β12_n) / Tres(dt)
        dβ21_dt = (β21_np1 - β21_n) / Tres(dt)
        β̄12     = (β12_n + β12_np1) / 2
        β̄21     = (β21_n + β21_np1) / 2

        # Cache per-axis self midpoints (needed for Berry cross-axis terms).
        ᾱ_self_1 = (get_α(y_n, 1, i) + get_α(y_np1, 1, i)) / 2
        ᾱ_self_2 = (get_α(y_n, 2, i) + get_α(y_np1, 2, i)) / 2
        β̄_self_1 = (get_β(y_n, 1, i) + get_β(y_np1, 1, i)) / 2
        β̄_self_2 = (get_β(y_n, 2, i) + get_β(y_np1, 2, i)) / 2

        # ──────────────────────────────────────────────────────────
        # M3-6 Phase 1a: off-diagonal velocity-gradient stencil.
        #
        # The off-diagonal Hamiltonian H_rot^off ∝ G̃_12 · (α_1·β_21
        # + α_2·β_12)/2 (per §7.5 of the 2D Berry note) sources β_12
        # and β_21 from a sheared base flow. The per-cell off-diagonal
        # velocity gradients are
        #
        #     ∂_2 u_1 ≈ (u_1[hi-along-2] − u_1[lo-along-2]) / (Δx_2 lo→hi)
        #     ∂_1 u_2 ≈ (u_2[hi-along-1] − u_2[lo-along-1]) / (Δx_1 lo→hi)
        #
        # G̃_12 = (∂_2 u_1 + ∂_1 u_2)/2 — symmetric strain (sources β_12, β_21).
        # W_12 = (∂_2 u_1 − ∂_1 u_2)/2 — antisymmetric (vorticity; sources θ_R).
        #
        # At axis-aligned ICs (every M3-3c regression and M3-4 C.1/C.2/
        # C.3 driver test), u_2 = 0 ⇒ ∂_1 u_2 = 0 and u_1 only varies
        # along axis 1 ⇒ ∂_2 u_1 = 0, so both G̃_12 = 0 and W_12 = 0.
        # Every Phase 1a addition vanishes multiplicatively ⇒ the
        # M3-3c / M3-4 / M3-6 Phase 0 §Dimension-lift gate holds at
        # bit-exact 0.0 absolute (CRITICAL).

        # ∂_2 u_1: read u_1 at axis-2 neighbors (cells "above" and "below"
        # in the y direction).
        ilo2 = face_lo[2][i]
        ihi2 = face_hi[2][i]
        wrap_lo_off_2 = wrap_lo === nothing ? zero(Tres) : Tres(wrap_lo[2][i])
        wrap_hi_off_2 = wrap_hi === nothing ? zero(Tres) : Tres(wrap_hi[2][i])

        # u_1 and x_2 at axis-2 lo neighbor.
        if ilo2 == 0
            u1_lo2_n   = get_u(y_n,   1, i)
            u1_lo2_np1 = get_u(y_np1, 1, i)
            x2_lo2_n   = get_x(y_n,   2, i)
            x2_lo2_np1 = get_x(y_np1, 2, i)
        else
            u1_lo2_n   = get_u(y_n,   1, ilo2)
            u1_lo2_np1 = get_u(y_np1, 1, ilo2)
            x2_lo2_n   = get_x(y_n,   2, ilo2) + wrap_lo_off_2
            x2_lo2_np1 = get_x(y_np1, 2, ilo2) + wrap_lo_off_2
        end
        if ihi2 == 0
            u1_hi2_n   = get_u(y_n,   1, i)
            u1_hi2_np1 = get_u(y_np1, 1, i)
            x2_hi2_n   = get_x(y_n,   2, i)
            x2_hi2_np1 = get_x(y_np1, 2, i)
        else
            u1_hi2_n   = get_u(y_n,   1, ihi2)
            u1_hi2_np1 = get_u(y_np1, 1, ihi2)
            x2_hi2_n   = get_x(y_n,   2, ihi2) + wrap_hi_off_2
            x2_hi2_np1 = get_x(y_np1, 2, ihi2) + wrap_hi_off_2
        end
        ū1_lo2 = (u1_lo2_n + u1_lo2_np1) / 2
        ū1_hi2 = (u1_hi2_n + u1_hi2_np1) / 2
        x̄2_lo2 = (x2_lo2_n + x2_lo2_np1) / 2
        x̄2_hi2 = (x2_hi2_n + x2_hi2_np1) / 2
        Δx̄2 = x̄2_hi2 - x̄2_lo2
        d2_u1 = Δx̄2 > 0 ? (ū1_hi2 - ū1_lo2) / Δx̄2 : zero(Tres)

        # ∂_1 u_2: read u_2 at axis-1 neighbors.
        ilo1 = face_lo[1][i]
        ihi1 = face_hi[1][i]
        wrap_lo_off_1 = wrap_lo === nothing ? zero(Tres) : Tres(wrap_lo[1][i])
        wrap_hi_off_1 = wrap_hi === nothing ? zero(Tres) : Tres(wrap_hi[1][i])

        if ilo1 == 0
            u2_lo1_n   = get_u(y_n,   2, i)
            u2_lo1_np1 = get_u(y_np1, 2, i)
            x1_lo1_n   = get_x(y_n,   1, i)
            x1_lo1_np1 = get_x(y_np1, 1, i)
        else
            u2_lo1_n   = get_u(y_n,   2, ilo1)
            u2_lo1_np1 = get_u(y_np1, 2, ilo1)
            x1_lo1_n   = get_x(y_n,   1, ilo1) + wrap_lo_off_1
            x1_lo1_np1 = get_x(y_np1, 1, ilo1) + wrap_lo_off_1
        end
        if ihi1 == 0
            u2_hi1_n   = get_u(y_n,   2, i)
            u2_hi1_np1 = get_u(y_np1, 2, i)
            x1_hi1_n   = get_x(y_n,   1, i)
            x1_hi1_np1 = get_x(y_np1, 1, i)
        else
            u2_hi1_n   = get_u(y_n,   2, ihi1)
            u2_hi1_np1 = get_u(y_np1, 2, ihi1)
            x1_hi1_n   = get_x(y_n,   1, ihi1) + wrap_hi_off_1
            x1_hi1_np1 = get_x(y_np1, 1, ihi1) + wrap_hi_off_1
        end
        ū2_lo1 = (u2_lo1_n + u2_lo1_np1) / 2
        ū2_hi1 = (u2_hi1_n + u2_hi1_np1) / 2
        x̄1_lo1 = (x1_lo1_n + x1_lo1_np1) / 2
        x̄1_hi1 = (x1_hi1_n + x1_hi1_np1) / 2
        Δx̄1 = x̄1_hi1 - x̄1_lo1
        d1_u2 = Δx̄1 > 0 ? (ū2_hi1 - ū2_lo1) / Δx̄1 : zero(Tres)

        # M3-6 Phase 1a strain decomposition.
        G̃12 = (d2_u1 + d1_u2) / 2     # symmetric (off-diag) strain rate
        W12 = (d2_u1 - d1_u2) / 2     # antisymmetric (vorticity)

        for a in 1:2
            # Self midpoints.
            x_n   = get_x(y_n,   a, i)
            x_np1 = get_x(y_np1, a, i)
            u_n   = get_u(y_n,   a, i)
            u_np1 = get_u(y_np1, a, i)
            α_n   = get_α(y_n,   a, i)
            α_np1 = get_α(y_np1, a, i)
            β_n   = get_β(y_n,   a, i)
            β_np1 = get_β(y_np1, a, i)

            x̄ = (x_n   + x_np1)   / 2
            ū = (u_n   + u_np1)   / 2
            ᾱ = (α_n   + α_np1)   / 2
            β̄ = (β_n   + β_np1)   / 2

            # Neighbor along axis a (lo, hi).
            ilo = face_lo[a][i]
            ihi = face_hi[a][i]

            # M3-4 periodic wrap (see `cholesky_el_residual_2D!`).
            wrap_lo_off = wrap_lo === nothing ? zero(Tres) : Tres(wrap_lo[a][i])
            wrap_hi_off = wrap_hi === nothing ? zero(Tres) : Tres(wrap_hi[a][i])

            # Lo-face neighbor data; mirror-self when out-of-domain.
            if ilo == 0
                x_lo_n   = x_n
                x_lo_np1 = x_np1
                u_lo_n   = u_n
                u_lo_np1 = u_np1
                s_lo     = s_i
            else
                x_lo_n   = get_x(y_n,   a, ilo) + wrap_lo_off
                x_lo_np1 = get_x(y_np1, a, ilo) + wrap_lo_off
                u_lo_n   = get_u(y_n,   a, ilo)
                u_lo_np1 = get_u(y_np1, a, ilo)
                s_lo     = s_vec[ilo]
            end
            x̄_lo = (x_lo_n + x_lo_np1) / 2
            ū_lo = (u_lo_n + u_lo_np1) / 2

            # Hi-face neighbor data; mirror-self when out-of-domain.
            if ihi == 0
                x_hi_n   = x_n
                x_hi_np1 = x_np1
                u_hi_n   = u_n
                u_hi_np1 = u_np1
                s_hi     = s_i
            else
                x_hi_n   = get_x(y_n,   a, ihi) + wrap_hi_off
                x_hi_np1 = get_x(y_np1, a, ihi) + wrap_hi_off
                u_hi_n   = get_u(y_n,   a, ihi)
                u_hi_np1 = get_u(y_np1, a, ihi)
                s_hi     = s_vec[ihi]
            end
            x̄_hi = (x_hi_n + x_hi_np1) / 2
            ū_hi = (u_hi_n + u_hi_np1) / 2

            Δm_i = Δm_per_axis[a][i]

            # Per-axis M_vv (M_vv_override or EOS branch); mirrors the
            # M3-3b residual exactly so the dimension-lift gate carries.
            M̄vv_a = if M_vv_over !== nothing
                Tres(M_vv_over[a])
            else
                Δx_avg = if ilo == 0 && ihi == 0
                    zero(Tres)
                elseif ilo == 0
                    2 * (x̄_hi - x̄)
                elseif ihi == 0
                    2 * (x̄ - x̄_lo)
                else
                    x̄_hi - x̄_lo
                end
                J̄_self = Δx_avg > 0 ? Δx_avg / (2 * Δm_i) : zero(Tres)
                J̄_self > 0 ? Mvv(J̄_self, s_i) : zero(Tres)
            end

            # Pressure stencil (mirrors M3-3b).
            P_self = Tres(ρ_ref) * M̄vv_a
            P_lo_neighbor = if ilo == 0
                P_self
            else
                Mvv_lo = if M_vv_over !== nothing
                    Tres(M_vv_over[a])
                else
                    Tres(ρ_ref) * Mvv(one(Tres) / Tres(ρ_ref), s_lo)
                end
                Tres(ρ_ref) * Mvv_lo
            end
            P_hi_neighbor = if ihi == 0
                P_self
            else
                Mvv_hi = if M_vv_over !== nothing
                    Tres(M_vv_over[a])
                else
                    Tres(ρ_ref) * Mvv(one(Tres) / Tres(ρ_ref), s_hi)
                end
                Tres(ρ_ref) * Mvv_hi
            end
            P̄_lo = (P_self + P_lo_neighbor) / 2
            P̄_hi = (P_self + P_hi_neighbor) / 2

            Δx̄_full = x̄_hi - x̄_lo
            div̄u_a = if Δx̄_full > 0
                (ū_hi - ū_lo) / Δx̄_full
            else
                zero(Tres)
            end

            m̄_a = Δm_i
            γ²_a = M̄vv_a - β̄^2

            # Berry-modification coefficients for this axis (M3-3c +
            # M3-6 Phase 0 off-diag β extension). Signs come from rows
            # of Ω · X = -dH (see file-level comment block above).
            #
            # axis 1: α̇_1 = β_1 - (α_2³/(3α_1²)) θ̇_R           ⇒ +(ᾱ_2³/(3ᾱ_1²)) θ̇_R
            #         β̇_1 = γ_1²/α_1 - β_2 θ̇_R + ...           ⇒ +β̄_2 θ̇_R + (M3-6) off-diag β coupling
            # axis 2: α̇_2 = β_2 + (α_1³/(3α_2²)) θ̇_R           ⇒ -(ᾱ_1³/(3ᾱ_2²)) θ̇_R
            #         β̇_2 = γ_2²/α_2 + β_1 θ̇_R + ...           ⇒ -β̄_1 θ̇_R + (M3-6) off-diag β coupling
            #
            # M3-6 Phase 0 additions to F^β_a (from rows α_a of Ω·X+dH=0
            # with the new Ω entries). With H = H_Ch only (no off-diag
            # strain in M3-6 Phase 0 free-flight cut), ∂H/∂α_a is the
            # same as M3-3c, but the row α_a now contains β̇_12, β̇_21
            # terms via Ω[α_a, β_12], Ω[α_a, β_21]; and Ω[α_a, θ_R]
            # acquires β_12, β_21 contributions from F_off:
            #
            #   axis 1 (row α_1):
            #     α_1²·β̇_1 - (α_2²/2)·β̇_12 - α_1·α_2·β̇_21
            #       + (α_1²β_2 + α_1·α_2·β_12 - (α_2²/2)·β_21)·θ̇_R
            #       - α_1·γ_1² = 0
            #     ⇒ β̇_1 = γ_1²/α_1
            #              - (β_2 + (α_2/α_1)·β_12 - (α_2²/(2α_1²))·β_21)·θ̇_R
            #              + (α_2²/(2α_1²))·β̇_12 + (α_2/α_1)·β̇_21
            #
            #   axis 2 (row α_2):
            #     α_2²·β̇_2 - α_1·α_2·β̇_12 - (α_1²/2)·β̇_21
            #       + (-α_2²β_1 + (α_1²/2)·β_12 - α_1·α_2·β_21)·θ̇_R
            #       - α_2·γ_2² = 0
            #     ⇒ β̇_2 = γ_2²/α_2
            #              - (-β_1 + (α_1²/(2α_2²))·β_12 - (α_1/α_2)·β_21)·θ̇_R
            #              + (α_1/α_2)·β̇_12 + (α_1²/(2α_2²))·β̇_21
            #
            # All M3-6 additions are multiplied by β_12, β_21, β̇_12, or
            # β̇_21. At β_12=β_21=0 (M3-3c regression IC + trivial-drive
            # F^β_12, F^β_21 rows below), every M3-6 term vanishes
            # identically and the residual reduces byte-equally to the
            # M3-3c form. This is the §Dimension-lift gate.
            berry_α_term, berry_β_term = if a == 1
                ( (ᾱ_self_2^3) / (3 * ᾱ_self_1^2) * dθR_dt,
                   β̄_self_2 * dθR_dt
                   # M3-6 Phase 0 off-diag β coupling on F^β_1.
                   + (ᾱ_self_2 / ᾱ_self_1) * β̄12 * dθR_dt
                   - (ᾱ_self_2^2 / (2 * ᾱ_self_1^2)) * β̄21 * dθR_dt
                   - (ᾱ_self_2^2 / (2 * ᾱ_self_1^2)) * dβ12_dt
                   - (ᾱ_self_2 / ᾱ_self_1) * dβ21_dt
                   # M3-6 Phase 1a off-diagonal strain coupling.
                   # H_rot^off = G̃_12·(α_1·β_21 + α_2·β_12)/2 contributes
                   # ∂H_rot^off/∂α_1 = G̃_12·β_21/2 to the F^β_1 row at
                   # midpoint. Vanishes when G̃_12 = 0 (axis-aligned ICs).
                   + G̃12 * β̄21 / 2
                )
            else
                ( -(ᾱ_self_1^3) / (3 * ᾱ_self_2^2) * dθR_dt,
                  -β̄_self_1 * dθR_dt
                   # M3-6 Phase 0 off-diag β coupling on F^β_2.
                   - (ᾱ_self_1^2 / (2 * ᾱ_self_2^2)) * β̄12 * dθR_dt
                   + (ᾱ_self_1 / ᾱ_self_2) * β̄21 * dθR_dt
                   - (ᾱ_self_1 / ᾱ_self_2) * dβ12_dt
                   - (ᾱ_self_1^2 / (2 * ᾱ_self_2^2)) * dβ21_dt
                   # M3-6 Phase 1a off-diagonal strain coupling.
                   # ∂H_rot^off/∂α_2 = G̃_12·β_12/2 contributes to F^β_2.
                   + G̃12 * β̄12 / 2
                )
            end

            # Residual rows.
            base = 11 * (i - 1)
            F[base + a]     = (x_np1 - x_n) / dt - ū                                         # F^x_a
            F[base + 2 + a] = (u_np1 - u_n) / dt + (P̄_hi - P̄_lo) / m̄_a                       # F^u_a
            F[base + 4 + a] = (α_np1 - α_n) / dt - β̄ + berry_α_term                          # F^α_a
            F[base + 6 + a] = (β_np1 - β_n) / dt + div̄u_a * β̄ -
                              (ᾱ != 0 ? γ²_a / ᾱ : zero(Tres)) + berry_β_term                # F^β_a
        end

        # M3-6 Phase 0 off-diag β rows + M3-6 Phase 1a strain coupling.
        #
        # Phase 0 had trivial drive (F^β_12 = (β_12_np1 − β_12_n)/dt) so
        # β_12, β_21 stayed at IC. Phase 1a adds the symmetric-strain
        # drive sourced by H_rot^off = G̃_12·(α_1·β_21 + α_2·β_12)/2:
        #
        #     F^β_12 += ∂H_rot^off/∂β_12 = G̃_12 · ᾱ_2 / 2
        #     F^β_21 += ∂H_rot^off/∂β_21 = G̃_12 · ᾱ_1 / 2
        #
        # At G̃_12 = 0 (axis-aligned ICs of M3-3c / M3-4 / M3-6 Phase 0
        # tests), the addition vanishes identically and the rows reduce
        # to the Phase 0 trivial-drive form ⇒ §Dimension-lift gate is
        # bit-exact.
        #
        # The smoke test in `test_M3_6_phase1a_strain_coupling.jl`
        # confirms that a sheared base flow u_1(x_2) drives β_12, β_21
        # off rest after a single Newton step.
        F[11 * (i - 1) + 9]  = (β12_np1 - β12_n) / dt + G̃12 * ᾱ_self_2 / 2
        F[11 * (i - 1) + 10] = (β21_np1 - β21_n) / dt + G̃12 * ᾱ_self_1 / 2

        # F^θ_R: kinematic-equation-driven evolution. M3-3c first cut had
        # zero drive ⇒ θ_R conserved per cell. M3-6 Phase 1a adds the
        # antisymmetric-strain (vorticity) coupling to the off-diagonal
        # Berry block:
        #
        #     F^θ_R += W_12 · F_off
        #
        # where F_off = (α_1²·α_2·β_12 − α_1·α_2²·β_21)/2 is the
        # off-diagonal piece of the Berry function (see
        # `scripts/verify_berry_connection_offdiag.py` CHECK 7). At
        # axis-aligned ICs W_12 = 0; at β_12 = β_21 = 0 IC F_off = 0;
        # in either case the addition vanishes and the §Dimension-lift
        # gate is preserved.
        F_off = (ᾱ_self_1^2 * ᾱ_self_2 * β̄12 -
                 ᾱ_self_1 * ᾱ_self_2^2 * β̄21) / 2
        F[11 * (i - 1) + 11] = (θR_np1 - θR_n) / dt + W12 * F_off
    end
    return F
end

"""
    cholesky_el_residual_2D_berry(y_np1, y_n, aux, dt)

Allocating wrapper around `cholesky_el_residual_2D_berry!`. Returns a
fresh residual vector. Used in tests where allocation cost is
irrelevant.
"""
function cholesky_el_residual_2D_berry(y_np1::AbstractVector,
                                         y_n::AbstractVector,
                                         aux::NamedTuple,
                                         dt::Real)
    Tres = promote_type(eltype(y_np1), eltype(y_n), eltype(aux.s_vec), typeof(dt))
    F = similar(y_np1, Tres, length(y_np1))
    cholesky_el_residual_2D_berry!(F, y_np1, y_n, aux, dt)
    return F
end

"""
    pack_state_2d_berry(fields::PolynomialFieldSet,
                         leaves::AbstractVector{<:Integer})
        -> Vector{Float64}

11-dof variant of `pack_state_2d` (M3-6 Phase 0): packs
`(x_1, x_2, u_1, u_2, α_1, α_2, β_1, β_2, β_12, β_21, θ_R)` per
leaf into a flat length-`11 N` vector. The off-diag pair `β_12, β_21`
was added in M3-6 Phase 0; pre-M3-6 IC factories set them to zero.
"""
function pack_state_2d_berry(fields::PolynomialFieldSet,
                              leaves::AbstractVector{<:Integer})
    N = length(leaves)
    T = typeof(fields.x_1[leaves[1]][1])
    y = Vector{T}(undef, 11 * N)
    @inbounds for (i, ci) in enumerate(leaves)
        base = 11 * (i - 1)
        y[base + 1]  = fields.x_1[ci][1]
        y[base + 2]  = fields.x_2[ci][1]
        y[base + 3]  = fields.u_1[ci][1]
        y[base + 4]  = fields.u_2[ci][1]
        y[base + 5]  = fields.α_1[ci][1]
        y[base + 6]  = fields.α_2[ci][1]
        y[base + 7]  = fields.β_1[ci][1]
        y[base + 8]  = fields.β_2[ci][1]
        y[base + 9]  = fields.β_12[ci][1]   # M3-6 Phase 0
        y[base + 10] = fields.β_21[ci][1]   # M3-6 Phase 0
        y[base + 11] = fields.θ_R[ci][1]
    end
    return y
end

"""
    unpack_state_2d_berry!(fields::PolynomialFieldSet,
                            leaves::AbstractVector{<:Integer},
                            y::AbstractVector)

Inverse of `pack_state_2d_berry`: write the flat 11-dof per-leaf state
back into the 2D field set (M3-6 Phase 0; off-diag β included).
Leaves the `:s, :Pp, :Q` slots untouched.
"""
function unpack_state_2d_berry!(fields::PolynomialFieldSet,
                                 leaves::AbstractVector{<:Integer},
                                 y::AbstractVector)
    N = length(leaves)
    @assert length(y) == 11 * N "y length $(length(y)) does not match 11 * N = $(11 * N)"
    @inbounds for (i, ci) in enumerate(leaves)
        base = 11 * (i - 1)
        fields.x_1[ci]  = (y[base + 1],)
        fields.x_2[ci]  = (y[base + 2],)
        fields.u_1[ci]  = (y[base + 3],)
        fields.u_2[ci]  = (y[base + 4],)
        fields.α_1[ci]  = (y[base + 5],)
        fields.α_2[ci]  = (y[base + 6],)
        fields.β_1[ci]  = (y[base + 7],)
        fields.β_2[ci]  = (y[base + 8],)
        fields.β_12[ci] = (y[base + 9],)    # M3-6 Phase 0
        fields.β_21[ci] = (y[base + 10],)   # M3-6 Phase 0
        fields.θ_R[ci]  = (y[base + 11],)
    end
    return fields
end

# ─────────────────────────────────────────────────────────────────────
# Phase M3-7b: native HG-side 3D EL residual (no Berry; θ_{ab} trivial)
# ─────────────────────────────────────────────────────────────────────
#
# Direct dimension-lift of the M3-3b 2D Cholesky-sector EL residual to
# `HierarchicalMesh{3}`: per-axis sums over `a ∈ {1, 2, 3}` and the
# face-neighbor stencil expanded to 6 faces (lo/hi along each of the 3
# axes). The Newton-driven row count grows from 8 (M3-3b 2D) to 15 per
# leaf — 3 x_a + 3 u_a + 3 α_a + 3 β_a + 3 θ_{ab} — with a 16th named
# slot `s` carried as auxiliary frozen entropy.
#
# Per the M3-7 design note §3 + §9, M3-7b implements the residual
# WITHOUT Berry coupling (M3-7c integrates the verified
# `berry_partials_3d` stencils). The three θ-rows are TRIVIAL-DRIVEN:
#
#     F^θ_{ab} = (θ_{ab}_np1 - θ_{ab}_n) / dt
#
# i.e., each Euler angle is conserved per cell across the Newton step.
# When the IC has all three angles at zero (the §7.1 dimension-lift
# gates), the trivial drive pins them at zero throughout the trajectory
# and the per-axis Cholesky-sector reduction matches the 2D / 1D
# residuals byte-equally on the dimension-lifted slices.
#
# The pseudo-residual structure mirrors M3-3b's 2D form on each axis:
#
#     F^x_a    = (x_a_np1 − x_a_n)/dt − ū_a                          (a = 1, 2, 3)
#     F^u_a    = (u_a_np1 − u_a_n)/dt + (P̄_a^hi − P̄_a^lo) / m̄_a     (a = 1, 2, 3)
#     F^α_a    = (α_a_np1 − α_a_n)/dt − β̄_a                          (a = 1, 2, 3)
#     F^β_a    = (β_a_np1 − β_a_n)/dt + (∂_a u_a) β̄_a − γ̄_a²/ᾱ_a    (a = 1, 2, 3)
#     F^θ_ab   = (θ_ab_np1 − θ_ab_n) / dt                            (ab ∈ {12,13,23})
#
# Off-diagonal `β_{ab}` is intentionally NOT carried on the 3D field set
# (per M3-3a Q3 default + M3-7 design note §4.4); 3D D.1 KH (M3-9)
# will lift the field set to 19 dof when off-diagonal coupling is
# activated.

"""
    cholesky_el_residual_3D!(F::AbstractVector,
                             y_np1::AbstractVector,
                             y_n::AbstractVector,
                             aux::NamedTuple,
                             dt::Real)

Native HG-side 3D Cholesky-sector EL residual without Berry coupling
(M3-7b). Writes the residual `F` for the 15-dof-per-cell flat unknown
vector `y_np1` against the reference state `y_n`. Both vectors are
packed leaf-cell-major as

    y[15(i-1) +  1] = x_1
    y[15(i-1) +  2] = x_2
    y[15(i-1) +  3] = x_3
    y[15(i-1) +  4] = u_1
    y[15(i-1) +  5] = u_2
    y[15(i-1) +  6] = u_3
    y[15(i-1) +  7] = α_1
    y[15(i-1) +  8] = α_2
    y[15(i-1) +  9] = α_3
    y[15(i-1) + 10] = β_1
    y[15(i-1) + 11] = β_2
    y[15(i-1) + 12] = β_3
    y[15(i-1) + 13] = θ_12
    y[15(i-1) + 14] = θ_13
    y[15(i-1) + 15] = θ_23

for `i ∈ 1:N` (N = number of leaf cells in mesh order).

Auxiliary data carried in `aux::NamedTuple`:

  • `s_vec::AbstractVector{T}`           — frozen entropy per leaf
  • `Δm_per_axis::NTuple{3, AbstractVector{T}}` — per-axis Lagrangian
                                            mass step at each leaf
                                            (= ρ_ref × cell extent
                                            along axis a)
  • `face_lo_idx::NTuple{3, Vector{Int}}` — per-axis lo-face neighbor
                                            cell idx (0 for boundary;
                                            populated from
                                            `face_neighbors_with_bcs`
                                            so periodic wrap-around is
                                            already resolved)
  • `face_hi_idx::NTuple{3, Vector{Int}}` — per-axis hi-face neighbor
                                            cell idx
  • `wrap_lo_idx::NTuple{3, Vector{T}}`    — per-axis-per-cell additive
                                            coordinate-wrap offsets for
                                            periodic axes (M3-4
                                            generalisation)
  • `wrap_hi_idx::NTuple{3, Vector{T}}`    — likewise for hi-faces
  • `M_vv_override::Union{Nothing, NTuple{3, T}}` — when not `nothing`,
                                            override `M_vv(J, s)` per
                                            axis with the supplied
                                            constants
  • `ρ_ref::T`                             — reference density used by
                                            the per-axis pressure
                                            stencil

Boundary handling: when `face_lo_idx[a][i] == 0`, the residual treats
the lo-face as a wall (mirror) — pressure equal to the cell's own,
velocity equal to the cell's own — so no spurious gradient at the
boundary. Periodic wrap is already resolved upstream by
`face_neighbors_with_bcs`; this `0`-as-wall fallback only fires for
genuinely closed boundaries (REFLECTING / DIRICHLET-pinned).

# Verification

  • For the cold-limit fixed point (`M_vv = 0` on every axis,
    `β = 0`, `u = 0`, uniform cells, all θ_{ab} = 0), the residual
    evaluates to the machine-precision zero vector when
    `y_n == y_np1`.
  • In a 1D-symmetric configuration (axes 2, 3 trivial / fixed-point;
    all θ_{ab} = 0), the axis-1 sub-residual reduces bit-for-bit to
    M1's 1D `det_el_residual` per cell — this is the 3D ⊂ 1D
    dimension-lift gate (M3-7 design note §7.1a).
  • In a 2D-symmetric configuration (axis 3 trivial; θ_13 = θ_23 = 0,
    θ_12 free), the axis-1+axis-2 sub-residual reduces to M3-3b's 2D
    residual byte-equally — the 3D ⊂ 2D dimension-lift gate
    (M3-7 design note §7.1b). Note: in M3-7b's no-Berry residual the
    F^θ_12 row is trivial-driven, so θ_12 stays at IC; this gate
    therefore exercises the per-axis Cholesky-sector reduction at +1
    axis only, mirroring M3-3b's 2D path which also pins θ_R fixed.
"""
function cholesky_el_residual_3D!(F::AbstractVector,
                                   y_np1::AbstractVector,
                                   y_n::AbstractVector,
                                   aux::NamedTuple,
                                   dt::Real)
    s_vec       = aux.s_vec
    Δm_per_axis = aux.Δm_per_axis
    face_lo     = aux.face_lo_idx
    face_hi     = aux.face_hi_idx
    M_vv_over   = aux.M_vv_override
    ρ_ref       = aux.ρ_ref
    wrap_lo     = haskey(aux, :wrap_lo_idx) ? aux.wrap_lo_idx : nothing
    wrap_hi     = haskey(aux, :wrap_hi_idx) ? aux.wrap_hi_idx : nothing

    N = length(s_vec)
    @assert length(y_np1) == 15 * N "y_np1 length $(length(y_np1)) does not match 15 * N = $(15 * N)"
    @assert length(y_n)   == 15 * N
    @assert length(F)     == 15 * N
    @assert length(Δm_per_axis[1]) == N
    @assert length(Δm_per_axis[2]) == N
    @assert length(Δm_per_axis[3]) == N

    Tres = promote_type(eltype(y_np1), eltype(y_n), eltype(s_vec), typeof(dt))

    # 15-dof per-cell index helpers (closed over y arrays).
    # a ∈ {1, 2, 3} for per-axis x/u/α/β; ab ∈ {1, 2, 3} maps to
    # {(1,2), (1,3), (2,3)} in order for θ_{ab}.
    @inline get_x(y, a, i) = y[15 * (i - 1) + a]              # 1..3
    @inline get_u(y, a, i) = y[15 * (i - 1) + 3 + a]          # 4..6
    @inline get_α(y, a, i) = y[15 * (i - 1) + 6 + a]          # 7..9
    @inline get_β(y, a, i) = y[15 * (i - 1) + 9 + a]          # 10..12
    @inline get_θ(y, ab, i) = y[15 * (i - 1) + 12 + ab]       # 13..15

    @inbounds for i in 1:N
        s_i = s_vec[i]
        for a in 1:3
            # Self midpoints.
            x_n   = get_x(y_n,   a, i)
            x_np1 = get_x(y_np1, a, i)
            u_n   = get_u(y_n,   a, i)
            u_np1 = get_u(y_np1, a, i)
            α_n   = get_α(y_n,   a, i)
            α_np1 = get_α(y_np1, a, i)
            β_n   = get_β(y_n,   a, i)
            β_np1 = get_β(y_np1, a, i)

            x̄ = (x_n   + x_np1)   / 2
            ū = (u_n   + u_np1)   / 2
            ᾱ = (α_n   + α_np1)   / 2
            β̄ = (β_n   + β_np1)   / 2

            # Neighbor along axis a (lo, hi).
            ilo = face_lo[a][i]
            ihi = face_hi[a][i]

            # M3-7b periodic-coordinate wrap (3D analog of M3-4 Phase 1).
            wrap_lo_off = wrap_lo === nothing ? zero(Tres) : Tres(wrap_lo[a][i])
            wrap_hi_off = wrap_hi === nothing ? zero(Tres) : Tres(wrap_hi[a][i])

            # Lo-face neighbor data; mirror-self when out-of-domain.
            if ilo == 0
                x_lo_n   = x_n
                x_lo_np1 = x_np1
                u_lo_n   = u_n
                u_lo_np1 = u_np1
                s_lo     = s_i
            else
                x_lo_n   = get_x(y_n,   a, ilo) + wrap_lo_off
                x_lo_np1 = get_x(y_np1, a, ilo) + wrap_lo_off
                u_lo_n   = get_u(y_n,   a, ilo)
                u_lo_np1 = get_u(y_np1, a, ilo)
                s_lo     = s_vec[ilo]
            end
            x̄_lo = (x_lo_n + x_lo_np1) / 2
            ū_lo = (u_lo_n + u_lo_np1) / 2

            # Hi-face neighbor data; mirror-self when out-of-domain.
            if ihi == 0
                x_hi_n   = x_n
                x_hi_np1 = x_np1
                u_hi_n   = u_n
                u_hi_np1 = u_np1
                s_hi     = s_i
            else
                x_hi_n   = get_x(y_n,   a, ihi) + wrap_hi_off
                x_hi_np1 = get_x(y_np1, a, ihi) + wrap_hi_off
                u_hi_n   = get_u(y_n,   a, ihi)
                u_hi_np1 = get_u(y_np1, a, ihi)
                s_hi     = s_vec[ihi]
            end
            x̄_hi = (x_hi_n + x_hi_np1) / 2
            ū_hi = (u_hi_n + u_hi_np1) / 2

            # Per-axis Lagrangian mass step.
            Δm_i = Δm_per_axis[a][i]

            # Per-axis M_vv (override or EOS branch); mirrors M3-3b.
            M̄vv_a = if M_vv_over !== nothing
                Tres(M_vv_over[a])
            else
                Δx_avg = if ilo == 0 && ihi == 0
                    zero(Tres)
                elseif ilo == 0
                    2 * (x̄_hi - x̄)
                elseif ihi == 0
                    2 * (x̄ - x̄_lo)
                else
                    x̄_hi - x̄_lo
                end
                J̄_self = Δx_avg > 0 ? Δx_avg / (2 * Δm_i) : zero(Tres)
                J̄_self > 0 ? Mvv(J̄_self, s_i) : zero(Tres)
            end

            # Pressure per axis. Mirror M3-3b's stencil exactly so the
            # 3D ⊂ 1D / 3D ⊂ 2D dimension-lift gates carry.
            P_self = Tres(ρ_ref) * M̄vv_a
            P_lo_neighbor = if ilo == 0
                P_self
            else
                Mvv_lo = if M_vv_over !== nothing
                    Tres(M_vv_over[a])
                else
                    Tres(ρ_ref) * Mvv(one(Tres) / Tres(ρ_ref), s_lo)
                end
                Tres(ρ_ref) * Mvv_lo
            end
            P_hi_neighbor = if ihi == 0
                P_self
            else
                Mvv_hi = if M_vv_over !== nothing
                    Tres(M_vv_over[a])
                else
                    Tres(ρ_ref) * Mvv(one(Tres) / Tres(ρ_ref), s_hi)
                end
                Tres(ρ_ref) * Mvv_hi
            end
            P̄_lo = (P_self + P_lo_neighbor) / 2
            P̄_hi = (P_self + P_hi_neighbor) / 2

            # Self-cell strain rate along axis a (Eulerian (∂_a u_a)
            # at the cell midpoint) using the lo↔hi extent.
            Δx̄_full = x̄_hi - x̄_lo
            div̄u_a = if Δx̄_full > 0
                (ū_hi - ū_lo) / Δx̄_full
            else
                zero(Tres)
            end

            m̄_a = Δm_i
            γ²_a = M̄vv_a - β̄^2

            # Residual rows.
            base = 15 * (i - 1)
            F[base + a]      = (x_np1 - x_n) / dt - ū                          # F^x_a
            F[base + 3 + a]  = (u_np1 - u_n) / dt + (P̄_hi - P̄_lo) / m̄_a       # F^u_a
            # F^α_a: D_t^{(0)} α = β
            F[base + 6 + a]  = (α_np1 - α_n) / dt - β̄                          # F^α_a
            # F^β_a: D_t^{(1)} β = γ²/α
            F[base + 9 + a]  = (β_np1 - β_n) / dt + div̄u_a * β̄ -
                               (ᾱ != 0 ? γ²_a / ᾱ : zero(Tres))                # F^β_a
        end

        # F^θ_{ab}: trivial-drive (M3-7b; M3-7c will activate Berry).
        # Each Euler angle is conserved across the Newton step.
        base = 15 * (i - 1)
        for ab in 1:3
            θ_n   = get_θ(y_n,   ab, i)
            θ_np1 = get_θ(y_np1, ab, i)
            F[base + 12 + ab] = (θ_np1 - θ_n) / dt                              # F^θ_{12,13,23}
        end
    end
    return F
end

"""
    cholesky_el_residual_3D(y_np1, y_n, aux, dt)

Allocating wrapper around `cholesky_el_residual_3D!`. Returns a fresh
residual vector. Used in tests where allocation cost is irrelevant.
"""
function cholesky_el_residual_3D(y_np1::AbstractVector,
                                  y_n::AbstractVector,
                                  aux::NamedTuple,
                                  dt::Real)
    Tres = promote_type(eltype(y_np1), eltype(y_n), eltype(aux.s_vec), typeof(dt))
    F = similar(y_np1, Tres, length(y_np1))
    cholesky_el_residual_3D!(F, y_np1, y_n, aux, dt)
    return F
end

# ─────────────────────────────────────────────────────────────────────
# 3D field-set ↔ flat state-vector packing helpers (M3-7b)
# ─────────────────────────────────────────────────────────────────────

"""
    pack_state_3d(fields::PolynomialFieldSet, leaves::AbstractVector{<:Integer})
        -> Vector{Float64}

Pack the per-leaf 15-dof Newton state
`(x_a, u_a, α_a, β_a)_{a=1,2,3} + (θ_12, θ_13, θ_23)` out of the
M3-7a 3D Cholesky-sector field set `fields` into a flat length-`15 N`
vector, where `N = length(leaves)`. Iteration order is `leaves[k]` for
`k ∈ 1:N`. The frozen-across-Newton entropy `s` is NOT packed — it is
read directly from the field set as auxiliary data (see
`cholesky_el_residual_3D!` `aux`).
"""
function pack_state_3d(fields::PolynomialFieldSet,
                        leaves::AbstractVector{<:Integer})
    N = length(leaves)
    T = typeof(fields.x_1[leaves[1]][1])
    y = Vector{T}(undef, 15 * N)
    @inbounds for (i, ci) in enumerate(leaves)
        base = 15 * (i - 1)
        y[base +  1] = fields.x_1[ci][1]
        y[base +  2] = fields.x_2[ci][1]
        y[base +  3] = fields.x_3[ci][1]
        y[base +  4] = fields.u_1[ci][1]
        y[base +  5] = fields.u_2[ci][1]
        y[base +  6] = fields.u_3[ci][1]
        y[base +  7] = fields.α_1[ci][1]
        y[base +  8] = fields.α_2[ci][1]
        y[base +  9] = fields.α_3[ci][1]
        y[base + 10] = fields.β_1[ci][1]
        y[base + 11] = fields.β_2[ci][1]
        y[base + 12] = fields.β_3[ci][1]
        y[base + 13] = fields.θ_12[ci][1]
        y[base + 14] = fields.θ_13[ci][1]
        y[base + 15] = fields.θ_23[ci][1]
    end
    return y
end

"""
    unpack_state_3d!(fields::PolynomialFieldSet,
                      leaves::AbstractVector{<:Integer},
                      y::AbstractVector)

Inverse of `pack_state_3d`: write the flat 15-dof per-leaf state back
into the 3D field set. Leaves the `:s` slot untouched (entropy is
operator-split, frozen across the Newton step).
"""
function unpack_state_3d!(fields::PolynomialFieldSet,
                           leaves::AbstractVector{<:Integer},
                           y::AbstractVector)
    N = length(leaves)
    @assert length(y) == 15 * N "y length $(length(y)) does not match 15 * N = $(15 * N)"
    @inbounds for (i, ci) in enumerate(leaves)
        base = 15 * (i - 1)
        fields.x_1[ci]  = (y[base +  1],)
        fields.x_2[ci]  = (y[base +  2],)
        fields.x_3[ci]  = (y[base +  3],)
        fields.u_1[ci]  = (y[base +  4],)
        fields.u_2[ci]  = (y[base +  5],)
        fields.u_3[ci]  = (y[base +  6],)
        fields.α_1[ci]  = (y[base +  7],)
        fields.α_2[ci]  = (y[base +  8],)
        fields.α_3[ci]  = (y[base +  9],)
        fields.β_1[ci]  = (y[base + 10],)
        fields.β_2[ci]  = (y[base + 11],)
        fields.β_3[ci]  = (y[base + 12],)
        fields.θ_12[ci] = (y[base + 13],)
        fields.θ_13[ci] = (y[base + 14],)
        fields.θ_23[ci] = (y[base + 15],)
    end
    return fields
end

"""
    build_face_neighbor_tables_3d(mesh::HierarchicalMesh{3},
                                   leaves::AbstractVector{<:Integer},
                                   bc_spec)
        -> (face_lo_idx::NTuple{3, Vector{Int}},
            face_hi_idx::NTuple{3, Vector{Int}})

3D analog of `build_face_neighbor_tables`. Pre-computes per-leaf
face-neighbor leaf indices for all three axes. Output is a pair of
length-`N` `Int` vectors per axis; `face_lo_idx[a][i]` is the
leaf-major index of the lo-side neighbor of `leaves[i]` along axis
`a` (or 0 if out-of-domain after BC processing). Periodic wrap is
already resolved upstream via `face_neighbors_with_bcs(mesh, ci, bc_spec)`.

`face_neighbors_with_bcs` for `HierarchicalMesh{3}` returns an
`NTuple{6, UInt32}` with face order `(axis1 lo, axis1 hi, axis2 lo,
axis2 hi, axis3 lo, axis3 hi)`.
"""
function build_face_neighbor_tables_3d(mesh::HierarchicalMesh{3},
                                        leaves::AbstractVector{<:Integer},
                                        bc_spec)
    N = length(leaves)
    pos = zeros(Int, hg_n_cells(mesh))
    @inbounds for (k, ci) in enumerate(leaves)
        pos[ci] = k
    end
    face_lo_1 = zeros(Int, N); face_hi_1 = zeros(Int, N)
    face_lo_2 = zeros(Int, N); face_hi_2 = zeros(Int, N)
    face_lo_3 = zeros(Int, N); face_hi_3 = zeros(Int, N)
    @inbounds for (i, ci) in enumerate(leaves)
        fn = face_neighbors_with_bcs(mesh, ci, bc_spec)
        face_lo_1[i] = fn[1] == 0 ? 0 : pos[Int(fn[1])]
        face_hi_1[i] = fn[2] == 0 ? 0 : pos[Int(fn[2])]
        face_lo_2[i] = fn[3] == 0 ? 0 : pos[Int(fn[3])]
        face_hi_2[i] = fn[4] == 0 ? 0 : pos[Int(fn[4])]
        face_lo_3[i] = fn[5] == 0 ? 0 : pos[Int(fn[5])]
        face_hi_3[i] = fn[6] == 0 ? 0 : pos[Int(fn[6])]
    end
    return ((face_lo_1, face_lo_2, face_lo_3),
            (face_hi_1, face_hi_2, face_hi_3))
end

"""
    build_periodic_wrap_tables_3d(mesh::HierarchicalMesh{3},
                                    frame::EulerianFrame{3, T},
                                    leaves::AbstractVector{<:Integer},
                                    face_lo_idx::NTuple{3, Vector{Int}},
                                    face_hi_idx::NTuple{3, Vector{Int}})
        -> (wrap_lo::NTuple{3, Vector{T}}, wrap_hi::NTuple{3, Vector{T}})

3D analog of `build_periodic_wrap_tables` (M3-4 Phase 1). Pre-computes
per-leaf-per-axis additive coordinate-wrap offsets for the 3D EL
residual. For each leaf `i ∈ 1:N` and each axis `a ∈ {1, 2, 3}`,
returns `wrap_lo[a][i]` and `wrap_hi[a][i]` such that the periodic
neighbor's `x_a` should be read as

    x_a_lo_neighbor + wrap_lo[a][i],   x_a_hi_neighbor + wrap_hi[a][i]

so the lo→hi extent across a periodic seam stays positive — the 3D
generalisation of the 1D `+L_box` wrap pattern.

The detection rule per axis is identical to the 2D version: if the
neighbor's physical box on axis `a` lies on the opposite wall, the
neighbor is the periodic wrap; the offset is `-L_a` for the lo-face
and `+L_a` for the hi-face.
"""
function build_periodic_wrap_tables_3d(mesh::HierarchicalMesh{3},
                                         frame::EulerianFrame{3, T},
                                         leaves::AbstractVector{<:Integer},
                                         face_lo_idx::NTuple{3, Vector{Int}},
                                         face_hi_idx::NTuple{3, Vector{Int}}
                                         ) where {T}
    N = length(leaves)
    L = ntuple(a -> T(frame.hi[a] - frame.lo[a]), 3)
    wrap_lo_1 = zeros(T, N); wrap_hi_1 = zeros(T, N)
    wrap_lo_2 = zeros(T, N); wrap_hi_2 = zeros(T, N)
    wrap_lo_3 = zeros(T, N); wrap_hi_3 = zeros(T, N)
    eps_a = ntuple(a -> T(1e-12) * L[a], 3)

    @inbounds for (i, ci) in enumerate(leaves)
        lo_self, hi_self = cell_physical_box(frame, ci)
        for a in 1:3
            wrap_arr_lo = a == 1 ? wrap_lo_1 : (a == 2 ? wrap_lo_2 : wrap_lo_3)
            wrap_arr_hi = a == 1 ? wrap_hi_1 : (a == 2 ? wrap_hi_2 : wrap_hi_3)
            face_lo = face_lo_idx[a]
            face_hi = face_hi_idx[a]

            # Lo-face wrap detection.
            ilo = face_lo[i]
            if ilo != 0
                ci_lo = leaves[ilo]
                lo_n, hi_n = cell_physical_box(frame, ci_lo)
                if lo_n[a] >= hi_self[a] - eps_a[a]
                    wrap_arr_lo[i] = -L[a]
                end
            end

            # Hi-face wrap detection.
            ihi = face_hi[i]
            if ihi != 0
                ci_hi = leaves[ihi]
                lo_n, hi_n = cell_physical_box(frame, ci_hi)
                if hi_n[a] <= lo_self[a] + eps_a[a]
                    wrap_arr_hi[i] = L[a]
                end
            end
        end
    end
    return ((wrap_lo_1, wrap_lo_2, wrap_lo_3),
            (wrap_hi_1, wrap_hi_2, wrap_hi_3))
end

"""
    build_residual_aux_3D(fields::PolynomialFieldSet,
                          mesh::HierarchicalMesh{3},
                          frame::EulerianFrame{3, T},
                          leaves::AbstractVector{<:Integer},
                          bc_spec;
                          M_vv_override = nothing,
                          ρ_ref = 1.0) where {T}

Convenience builder for the `aux` NamedTuple consumed by
`cholesky_el_residual_3D!`. Reads `s` per leaf from `fields`, extracts
per-axis cell extents from the Eulerian frame, computes per-axis
Lagrangian mass steps `Δm_a = ρ_ref * extent_a` per cell, and builds
the 3-axis face-neighbor + periodic-wrap tables. The 3D analog of
`build_residual_aux_2D`.
"""
function build_residual_aux_3D(fields::PolynomialFieldSet,
                                mesh::HierarchicalMesh{3},
                                frame::EulerianFrame{3, T},
                                leaves::AbstractVector{<:Integer},
                                bc_spec;
                                M_vv_override = nothing,
                                ρ_ref::Real = 1.0) where {T}
    N = length(leaves)
    s_vec  = Vector{T}(undef, N)
    Δm_1   = Vector{T}(undef, N)
    Δm_2   = Vector{T}(undef, N)
    Δm_3   = Vector{T}(undef, N)
    ρ_t = T(ρ_ref)
    @inbounds for (i, ci) in enumerate(leaves)
        s_vec[i] = fields.s[ci][1]
        lo, hi = cell_physical_box(frame, ci)
        Δm_1[i] = ρ_t * (hi[1] - lo[1])
        Δm_2[i] = ρ_t * (hi[2] - lo[2])
        Δm_3[i] = ρ_t * (hi[3] - lo[3])
    end
    face_lo_idx, face_hi_idx = build_face_neighbor_tables_3d(mesh, leaves, bc_spec)
    wrap_lo_idx, wrap_hi_idx = build_periodic_wrap_tables_3d(mesh, frame, leaves,
                                                              face_lo_idx,
                                                              face_hi_idx)
    return (
        s_vec       = s_vec,
        Δm_per_axis = (Δm_1, Δm_2, Δm_3),
        face_lo_idx = face_lo_idx,
        face_hi_idx = face_hi_idx,
        wrap_lo_idx = wrap_lo_idx,
        wrap_hi_idx = wrap_hi_idx,
        M_vv_override = M_vv_override,
        ρ_ref       = ρ_t,
    )
end

# ─────────────────────────────────────────────────────────────────────
# Phase M3-7c: SO(3) Berry coupling integration on the 3D residual
# ─────────────────────────────────────────────────────────────────────
#
# M3-7c extends the M3-7b residual `cholesky_el_residual_3D!` (which
# carried trivial-driven θ_{ab} rows) by activating the SO(3) Berry
# kinetic 1-form
#
#     Θ_rot^{(3D)} = (1/3) Σ_{a<b} (α_a^3 β_b − α_b^3 β_a) · dθ_{ab}
#
# (`src/berry.jl::berry_F_3d`, verified at the stencil level in
# `test/test_M3_prep_3D_berry_verification.jl`, 797 asserts) on the per-
# axis (α_a, β_a) residual rows.
#
# # Berry-modified per-axis Hamilton equations (3D)
#
# The Berry-modified rows of `Ω · X = -dH` per axis a, summed across the
# three pair-rotation generators (a, b) ∈ {(1, 2), (1, 3), (2, 3)},
# give the 3D analog of the 2D M3-3c boxed equations. Because each
# pair-generator J_{ab} commutes with the (α_c, β_c) for c ∉ {a, b}
# (the third axis is fixed under the (a, b) rotation), the per-pair
# Berry contributions add independently across the three pairs. Per
# axis a, summing over pairs in which a participates:
#
#   axis a = 1 (in pairs (1,2) and (1,3); a is the "left" index, sign +):
#     α̇_1 = β_1 − (α_2^3/(3 α_1^2)) θ̇_12 − (α_3^3/(3 α_1^2)) θ̇_13
#     β̇_1 = γ_1²/α_1 − β_2 θ̇_12 − β_3 θ̇_13
#
#   axis a = 2 (in pair (1,2) right, sign −; in pair (2,3) left, sign +):
#     α̇_2 = β_2 + (α_1^3/(3 α_2^2)) θ̇_12 − (α_3^3/(3 α_2^2)) θ̇_23
#     β̇_2 = γ_2²/α_2 + β_1 θ̇_12 − β_3 θ̇_23
#
#   axis a = 3 (in pairs (1,3) and (2,3) right, sign −):
#     α̇_3 = β_3 + (α_1^3/(3 α_3^2)) θ̇_13 + (α_2^3/(3 α_3^2)) θ̇_23
#     β̇_3 = γ_3²/α_3 + β_1 θ̇_13 + β_2 θ̇_23
#
# Sign convention. Each pair (a, b) (with a < b) contributes to the
# axis-a row with the same sign as the 2D pair (1, 2) → axis-1 row
# (sign +) and to the axis-b row with the same sign as the 2D pair
# (1, 2) → axis-2 row (sign −). This follows directly from the 2D
# berry_partials_2d → residual mapping in `cholesky_el_residual_2D_berry!`
# applied per pair.
#
# # Residual rows (per cell)
#
#     F^x_a    = (x_a_np1 − x_a_n)/dt − ū_a
#     F^u_a    = (u_a_np1 − u_a_n)/dt + (P̄_a^hi − P̄_a^lo)/m̄_a
#     F^α_a    = (α_a_np1 − α_a_n)/dt − β̄_a + (Berry α-modification per axis)
#     F^β_a    = (β_a_np1 − β_a_n)/dt + (∂_a u_a) β̄_a − γ̄_a²/ᾱ_a
#                + (Berry β-modification per axis)
#     F^θ_{ab} = (θ_{ab}_np1 − θ_{ab}_n)/dt
#
# The θ_{ab} rows are kinematic-equation drives with `drive = 0` (free-
# flight cut, axis-aligned ICs); identical in form to M3-3c's F^θ_R row
# (no off-diagonal velocity-gradient stencil). Each Euler angle is
# conserved per cell when the Berry α/β-modification block vanishes
# multiplicatively (β̄_b = 0, θ̇_{ab} = 0).
#
# # Dimension-lift gates carry through (CRITICAL)
#
#   • 3D ⊂ 1D (M3-7 design note §7.1a): on the 1D-symmetric slice
#     (α_2 = α_3 = const, β_2 = β_3 = 0, all θ_{ab} = 0), every Berry
#     α-modification term is multiplied by β̄_b = 0 (for b ∉ {a}) or by
#     θ̇_{ab} = 0 (the θ_{ab} rows pin them); every Berry β-modification
#     term is multiplied by β_b = 0 or θ̇_{ab} = 0. The residual reduces
#     byte-equally to M3-7b's no-Berry form on the dimension-lift slice;
#     M3-7b's gate transitively holds for M3-7c.
#
#   • 3D ⊂ 2D (M3-7 design note §7.1b): on the 2D-symmetric slice
#     (α_3 = const, β_3 = 0, θ_13 = θ_23 = 0), Berry contributions
#     involving the (1, 3) and (2, 3) pairs vanish because either β_3 = 0
#     or θ̇_13 = θ̇_23 = 0 (their rows pin them). The (1, 2) pair Berry
#     block matches M3-3c's 2D form byte-equally — a direct restriction
#     of the closed-form `berry_partials_3d` (see CHECK 3b of
#     `notes_M3_prep_3D_berry_verification.md`).
#
# # Off-diagonal β
#
# Off-diagonal β_{ab} is **not** carried on the 3D field set (per M3-3a
# Q3 default + M3-7 design note §4.4); the 3D D.1 KH (M3-9) work will
# lift to 19-dof when the 3D off-diagonal sector is calibrated.
#
# # H_rot solvability
#
# The closed-form `h_rot_partial_dtheta_3d(α, β, γ²; pair)` per pair
# `(a, b)` is the 3D analog of M3-3c's `h_rot_partial_dtheta`. The
# discrete EL residual rows do NOT consume this closed form directly
# (the per-axis Berry-modification terms encode the rows of `Ω · X = -dH`
# from `src/berry.jl::berry_partials_3d`); the helper exists as a
# verification-gate artefact for §7.4.

"""
    cholesky_el_residual_3D_berry!(F, y_np1, y_n, aux, dt)

Native HG-side 3D Cholesky-sector EL residual **with SO(3) Berry
coupling and (θ_12, θ_13, θ_23) as Newton unknowns**. Writes the
residual `F` for the 15-dof-per-cell flat unknown vector `y_np1`
against the reference state `y_n`. Both vectors share the M3-7b
packing layout (15 dof per leaf):

    y[15(i-1) +  1..3]  = (x_1, x_2, x_3)
    y[15(i-1) +  4..6]  = (u_1, u_2, u_3)
    y[15(i-1) +  7..9]  = (α_1, α_2, α_3)
    y[15(i-1) + 10..12] = (β_1, β_2, β_3)
    y[15(i-1) + 13..15] = (θ_12, θ_13, θ_23)

Auxiliary data carried in `aux::NamedTuple` is identical to
`cholesky_el_residual_3D!` (M3-7b) — see that docstring. The 3D analog
of `cholesky_el_residual_2D_berry!`.

# Verification

  • The §Dimension-lift gate (M3-7 design note §7.1a + §7.1b) holds at
    0.0 absolute on the 1D-symmetric and 2D-symmetric slices (per
    file-level comment block above).
  • The Berry partials match `berry_partials_3d` up to the (1/dt)
    factor and midpoint averaging — verified in
    `test_M3_7c_berry_3d_residual.jl`.
  • The H_rot solvability constraint per pair `(a, b)` evaluates to
    machine zero at the converged Newton iterate — verified in
    `test_M3_7c_h_rot_solvability_3d.jl`.

See `reference/notes_M3_7c_3d_berry_integration.md` for the full
integration write-up + sub-test inventory.
"""
function cholesky_el_residual_3D_berry!(F::AbstractVector,
                                          y_np1::AbstractVector,
                                          y_n::AbstractVector,
                                          aux::NamedTuple,
                                          dt::Real)
    s_vec       = aux.s_vec
    Δm_per_axis = aux.Δm_per_axis
    face_lo     = aux.face_lo_idx
    face_hi     = aux.face_hi_idx
    M_vv_over   = aux.M_vv_override
    ρ_ref       = aux.ρ_ref
    wrap_lo     = haskey(aux, :wrap_lo_idx) ? aux.wrap_lo_idx : nothing
    wrap_hi     = haskey(aux, :wrap_hi_idx) ? aux.wrap_hi_idx : nothing

    N = length(s_vec)
    @assert length(y_np1) == 15 * N "y_np1 length $(length(y_np1)) does not match 15 * N = $(15 * N)"
    @assert length(y_n)   == 15 * N
    @assert length(F)     == 15 * N
    @assert length(Δm_per_axis[1]) == N
    @assert length(Δm_per_axis[2]) == N
    @assert length(Δm_per_axis[3]) == N

    Tres = promote_type(eltype(y_np1), eltype(y_n), eltype(s_vec), typeof(dt))

    # 15-dof per-cell index helpers (mirror M3-7b's residual).
    @inline get_x(y, a, i) = y[15 * (i - 1) + a]              # 1..3
    @inline get_u(y, a, i) = y[15 * (i - 1) + 3 + a]          # 4..6
    @inline get_α(y, a, i) = y[15 * (i - 1) + 6 + a]          # 7..9
    @inline get_β(y, a, i) = y[15 * (i - 1) + 9 + a]          # 10..12
    @inline get_θ(y, ab, i) = y[15 * (i - 1) + 12 + ab]       # 13..15

    @inbounds for i in 1:N
        s_i = s_vec[i]

        # Per-cell θ_{ab} midpoints + finite-difference rates (shared by
        # all three axes via the Berry α/β-modification blocks).
        θ12_n   = get_θ(y_n,   1, i)
        θ12_np1 = get_θ(y_np1, 1, i)
        θ13_n   = get_θ(y_n,   2, i)
        θ13_np1 = get_θ(y_np1, 2, i)
        θ23_n   = get_θ(y_n,   3, i)
        θ23_np1 = get_θ(y_np1, 3, i)
        dθ12_dt = (θ12_np1 - θ12_n) / Tres(dt)
        dθ13_dt = (θ13_np1 - θ13_n) / Tres(dt)
        dθ23_dt = (θ23_np1 - θ23_n) / Tres(dt)

        # Cache per-axis self midpoints (needed for Berry cross-axis terms).
        α1_n   = get_α(y_n,   1, i); α1_np1 = get_α(y_np1, 1, i)
        α2_n   = get_α(y_n,   2, i); α2_np1 = get_α(y_np1, 2, i)
        α3_n   = get_α(y_n,   3, i); α3_np1 = get_α(y_np1, 3, i)
        β1_n   = get_β(y_n,   1, i); β1_np1 = get_β(y_np1, 1, i)
        β2_n   = get_β(y_n,   2, i); β2_np1 = get_β(y_np1, 2, i)
        β3_n   = get_β(y_n,   3, i); β3_np1 = get_β(y_np1, 3, i)
        ᾱ_self_1 = (α1_n + α1_np1) / 2
        ᾱ_self_2 = (α2_n + α2_np1) / 2
        ᾱ_self_3 = (α3_n + α3_np1) / 2
        β̄_self_1 = (β1_n + β1_np1) / 2
        β̄_self_2 = (β2_n + β2_np1) / 2
        β̄_self_3 = (β3_n + β3_np1) / 2

        for a in 1:3
            # Self midpoints.
            x_n   = get_x(y_n,   a, i)
            x_np1 = get_x(y_np1, a, i)
            u_n   = get_u(y_n,   a, i)
            u_np1 = get_u(y_np1, a, i)
            α_n   = get_α(y_n,   a, i)
            α_np1 = get_α(y_np1, a, i)
            β_n   = get_β(y_n,   a, i)
            β_np1 = get_β(y_np1, a, i)

            x̄ = (x_n   + x_np1)   / 2
            ū = (u_n   + u_np1)   / 2
            ᾱ = (α_n   + α_np1)   / 2
            β̄ = (β_n   + β_np1)   / 2

            # Neighbor along axis a (lo, hi).
            ilo = face_lo[a][i]
            ihi = face_hi[a][i]

            # Periodic-coordinate wrap (M3-7b 3-axis generalisation).
            wrap_lo_off = wrap_lo === nothing ? zero(Tres) : Tres(wrap_lo[a][i])
            wrap_hi_off = wrap_hi === nothing ? zero(Tres) : Tres(wrap_hi[a][i])

            # Lo-face neighbor data; mirror-self when out-of-domain.
            if ilo == 0
                x_lo_n   = x_n
                x_lo_np1 = x_np1
                u_lo_n   = u_n
                u_lo_np1 = u_np1
                s_lo     = s_i
            else
                x_lo_n   = get_x(y_n,   a, ilo) + wrap_lo_off
                x_lo_np1 = get_x(y_np1, a, ilo) + wrap_lo_off
                u_lo_n   = get_u(y_n,   a, ilo)
                u_lo_np1 = get_u(y_np1, a, ilo)
                s_lo     = s_vec[ilo]
            end
            x̄_lo = (x_lo_n + x_lo_np1) / 2
            ū_lo = (u_lo_n + u_lo_np1) / 2

            # Hi-face neighbor data; mirror-self when out-of-domain.
            if ihi == 0
                x_hi_n   = x_n
                x_hi_np1 = x_np1
                u_hi_n   = u_n
                u_hi_np1 = u_np1
                s_hi     = s_i
            else
                x_hi_n   = get_x(y_n,   a, ihi) + wrap_hi_off
                x_hi_np1 = get_x(y_np1, a, ihi) + wrap_hi_off
                u_hi_n   = get_u(y_n,   a, ihi)
                u_hi_np1 = get_u(y_np1, a, ihi)
                s_hi     = s_vec[ihi]
            end
            x̄_hi = (x_hi_n + x_hi_np1) / 2
            ū_hi = (u_hi_n + u_hi_np1) / 2

            Δm_i = Δm_per_axis[a][i]

            # Per-axis M_vv (override or EOS branch); mirrors M3-7b.
            M̄vv_a = if M_vv_over !== nothing
                Tres(M_vv_over[a])
            else
                Δx_avg = if ilo == 0 && ihi == 0
                    zero(Tres)
                elseif ilo == 0
                    2 * (x̄_hi - x̄)
                elseif ihi == 0
                    2 * (x̄ - x̄_lo)
                else
                    x̄_hi - x̄_lo
                end
                J̄_self = Δx_avg > 0 ? Δx_avg / (2 * Δm_i) : zero(Tres)
                J̄_self > 0 ? Mvv(J̄_self, s_i) : zero(Tres)
            end

            # Pressure stencil (mirrors M3-7b).
            P_self = Tres(ρ_ref) * M̄vv_a
            P_lo_neighbor = if ilo == 0
                P_self
            else
                Mvv_lo = if M_vv_over !== nothing
                    Tres(M_vv_over[a])
                else
                    Tres(ρ_ref) * Mvv(one(Tres) / Tres(ρ_ref), s_lo)
                end
                Tres(ρ_ref) * Mvv_lo
            end
            P_hi_neighbor = if ihi == 0
                P_self
            else
                Mvv_hi = if M_vv_over !== nothing
                    Tres(M_vv_over[a])
                else
                    Tres(ρ_ref) * Mvv(one(Tres) / Tres(ρ_ref), s_hi)
                end
                Tres(ρ_ref) * Mvv_hi
            end
            P̄_lo = (P_self + P_lo_neighbor) / 2
            P̄_hi = (P_self + P_hi_neighbor) / 2

            Δx̄_full = x̄_hi - x̄_lo
            div̄u_a = if Δx̄_full > 0
                (ū_hi - ū_lo) / Δx̄_full
            else
                zero(Tres)
            end

            m̄_a = Δm_i
            γ²_a = M̄vv_a - β̄^2

            # Berry α/β-modifications per axis a (signs from rows of
            # Ω · X = -dH summed over the three pair-generators in
            # which a participates; see file-level comment block above).
            #
            # axis 1: pairs (1,2) [+] and (1,3) [+]
            # axis 2: pairs (1,2) [-] and (2,3) [+]
            # axis 3: pairs (1,3) [-] and (2,3) [-]
            berry_α_term, berry_β_term = if a == 1
                ( (ᾱ_self_2^3) / (3 * ᾱ_self_1^2) * dθ12_dt +
                  (ᾱ_self_3^3) / (3 * ᾱ_self_1^2) * dθ13_dt,
                  β̄_self_2 * dθ12_dt +
                  β̄_self_3 * dθ13_dt
                )
            elseif a == 2
                ( -(ᾱ_self_1^3) / (3 * ᾱ_self_2^2) * dθ12_dt +
                   (ᾱ_self_3^3) / (3 * ᾱ_self_2^2) * dθ23_dt,
                  -β̄_self_1 * dθ12_dt +
                   β̄_self_3 * dθ23_dt
                )
            else
                ( -(ᾱ_self_1^3) / (3 * ᾱ_self_3^2) * dθ13_dt +
                  -(ᾱ_self_2^3) / (3 * ᾱ_self_3^2) * dθ23_dt,
                  -β̄_self_1 * dθ13_dt +
                  -β̄_self_2 * dθ23_dt
                )
            end

            # Residual rows.
            base = 15 * (i - 1)
            F[base + a]      = (x_np1 - x_n) / dt - ū                          # F^x_a
            F[base + 3 + a]  = (u_np1 - u_n) / dt + (P̄_hi - P̄_lo) / m̄_a        # F^u_a
            F[base + 6 + a]  = (α_np1 - α_n) / dt - β̄ + berry_α_term           # F^α_a
            F[base + 9 + a]  = (β_np1 - β_n) / dt + div̄u_a * β̄ -
                               (ᾱ != 0 ? γ²_a / ᾱ : zero(Tres)) + berry_β_term # F^β_a
        end

        # F^θ_{ab}: kinematic-drive form (3D analog of M3-3c's F^θ_R).
        # In M3-7c first cut (no off-diagonal velocity-gradient stencil
        # yet), drive = 0 ⇒ each Euler angle is conserved per cell across
        # the Newton step. The Berry α/β-modifications above already
        # encode the Hamilton equations of `Ω · X = -dH`, so the H_rot
        # solvability identity is structurally guaranteed (per-pair
        # check in `test_M3_7c_h_rot_solvability_3d.jl`).
        base = 15 * (i - 1)
        F[base + 13] = (θ12_np1 - θ12_n) / dt
        F[base + 14] = (θ13_np1 - θ13_n) / dt
        F[base + 15] = (θ23_np1 - θ23_n) / dt
    end
    return F
end

"""
    cholesky_el_residual_3D_berry(y_np1, y_n, aux, dt)

Allocating wrapper around `cholesky_el_residual_3D_berry!`. Returns a
fresh residual vector. Used in tests where allocation cost is irrelevant.
"""
function cholesky_el_residual_3D_berry(y_np1::AbstractVector,
                                         y_n::AbstractVector,
                                         aux::NamedTuple,
                                         dt::Real)
    Tres = promote_type(eltype(y_np1), eltype(y_n), eltype(aux.s_vec), typeof(dt))
    F = similar(y_np1, Tres, length(y_np1))
    cholesky_el_residual_3D_berry!(F, y_np1, y_n, aux, dt)
    return F
end
