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
                x_lo_n   = get_x(y_n,   a, ilo)
                x_lo_np1 = get_x(y_np1, a, ilo)
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
                x_hi_n   = get_x(y_n,   a, ihi)
                x_hi_np1 = get_x(y_np1, a, ihi)
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
    return (
        s_vec       = s_vec,
        θR_vec      = θR_vec,
        Δm_per_axis = (Δm_1, Δm_2),
        face_lo_idx = face_lo_idx,
        face_hi_idx = face_hi_idx,
        M_vv_override = M_vv_override,
        ρ_ref       = ρ_t,
    )
end
