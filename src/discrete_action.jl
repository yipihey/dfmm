# discrete_action.jl
#
# Phase 2: per-step discrete action sum and per-segment Lagrangian
# density evaluations for the full deterministic action
#
#     L_det = ½ ẋ² + L_Ch(α, β, β̇; γ),
#
# with γ derived from the EOS (γ² = M_vv(J, s) − β²) and the
# deviatoric / heat-flux pieces deferred to Phases 5/7. See
# `reference/notes_phase2_discretization.md` for the full derivation
# of the discrete EL system; this file implements only the action sum
# itself, used for the symplecticity diagnostic and for the 2-segment
# hand-checks during Phase 2 development. The Newton step in
# `newton_step.jl` calls `det_el_residual` directly (also defined in
# `cholesky_sector.jl` extension).
#
# Discretisation choice: midpoint-rule quadrature on every term, the
# strict superset of Phase 1's `cholesky_one_step_action`. With Phase 1
# initial conditions (single segment, zero velocity, frozen J & s) the
# Phase-2 action reduces to Δt · L_Ch evaluated by `cholesky_one_step_action`
# bit-for-bit. See the discretisation note §"Phase-1 reduction check".

using StaticArrays: SVector

"""
    _mvv_ad(J, s; Gamma=5/3, cv=1.0)

AD-friendly variant of the Track-C EOS adiabat
`M_vv = J^(1-Γ) exp(s/cv)`. Phase-2 ForwardDiff Jacobians on the EL
residual carry `Dual` numbers through `J`, but Track-C's
`Mvv(J::Real, s::Real)` declares `::Float64` return type which would
strip the dual tape. We therefore inline the same expression here
without forcing a `Float64` conversion. Numerical agreement with
`Mvv(J, s)` is bit-identical for `Float64` inputs.

Cold-limit underflow guard mirrors `eos.jl`: when
`(1-Γ) log(J) + s/cv < -700` the result is zero (exp underflow).
"""
@inline function _mvv_ad(J, s;
                         Gamma::Real = 5.0/3.0,
                         cv::Real = 1.0)
    log_Mvv = (1 - Gamma) * log(J) + s / cv
    return log_Mvv < -700 ? zero(log_Mvv) : exp(log_Mvv)
end

"""
    midpoint_strain(u_n, u_np1, x_n, x_np1)

Cell-centered midpoint strain rate `(∂_x u)_j^{n+1/2}` for one segment,
using the staggered layout of Phase 2: positions and velocities at
vertices, evaluated at the midpoint of the timestep.

`u_n, u_np1` are 2-tuples `(u_left, u_right)` of vertex velocities at
the segment's left and right vertex at times `t_n` and `t_{n+1}`.
`x_n, x_np1` are the corresponding 2-tuples of vertex positions.

The midpoint Eulerian strain is
`(ū_right − ū_left) / (x̄_right − x̄_left)` with `_bar` the half-step
average. Returns a scalar.
"""
function midpoint_strain(u_left_n, u_right_n, u_left_np1, u_right_np1,
                         x_left_n, x_right_n, x_left_np1, x_right_np1)
    ū_left = (u_left_n + u_left_np1) / 2
    ū_right = (u_right_n + u_right_np1) / 2
    x̄_left = (x_left_n + x_left_np1) / 2
    x̄_right = (x_right_n + x_right_np1) / 2
    return (ū_right - ū_left) / (x̄_right - x̄_left)
end

"""
    midpoint_J(x_left_n, x_right_n, x_left_np1, x_right_np1, Δm)

Midpoint specific volume `J_j^{n+1/2} = (x̄_right − x̄_left)/Δm_j`.
"""
function midpoint_J(x_left_n, x_right_n, x_left_np1, x_right_np1, Δm)
    x̄_left = (x_left_n + x_left_np1) / 2
    x̄_right = (x_right_n + x_right_np1) / 2
    return (x̄_right - x̄_left) / Δm
end

"""
    segment_action(seg_n, seg_np1, vert_n, vert_np1, Δm, dt)

Per-segment discrete action contribution `Δm_j · Δt · L_Ch_j|_{n+1/2}`
plus the half-shared kinetic. Used by `discrete_action_sum` only for
diagnostics; the Newton residual is implemented directly.

Arguments are unpacked tuples to keep the function AD-friendly.
"""
function segment_cholesky_action(α_n, β_n, α_np1, β_np1,
                                 Mvv_mid, divu_mid, dt)
    ᾱ = (α_n + α_np1) / 2
    β̄ = (β_n + β_np1) / 2
    return -((ᾱ^3) / 3) * (β_np1 - β_n) -
           dt * ((ᾱ^3) / 3) * divu_mid * β̄ +
           (dt / 2) * (ᾱ^2) * (Mvv_mid - β̄^2)
end

"""
    discrete_action_sum(mesh_state_n, mesh_state_np1, Δm_vec, dt)

Total per-step discrete action `ΔS_n = Σ_i kinetic_i + Σ_j Cholesky_j`
on a periodic Phase-2 mesh. `mesh_state_n` and `mesh_state_np1` are
length-N vectors of `DetField` (segment-indexed); `Δm_vec` are the
fixed segment masses; `dt` the timestep.

The kinetic contribution per *vertex* `i` is
`(m̄_i / 2) (ū_i)² · Δt` where `ū_i = (u_i^n + u_i^{n+1})/2`. Vertex
mass `m̄_i = (Δm_{i-1} + Δm_i)/2` is assembled cyclically from the
adjacent segment masses.

The Cholesky contribution per segment `j` uses the midpoint strain
and midpoint `M_vv`. With Phase-1 initial conditions (single
segment, all velocities zero, J frozen) the kinetic vanishes,
the strain is zero, `M_vv` reduces to its Phase-1 constant, and
`discrete_action_sum` returns Δt · `cholesky_one_step_action`
bit-for-bit.
"""
function discrete_action_sum(mesh_state_n::Vector{DetField{T}},
                             mesh_state_np1::Vector{DetField{T}},
                             Δm_vec::AbstractVector,
                             L_box::T,
                             dt::Real) where {T<:Real}
    N = length(mesh_state_n)
    @assert length(mesh_state_np1) == N
    @assert length(Δm_vec) == N

    S = zero(T)

    # Kinetic per vertex
    for i in 1:N
        i_left = i == 1 ? N : i - 1
        m̄ = (T(Δm_vec[i_left]) + T(Δm_vec[i])) / 2
        ū = (mesh_state_n[i].u + mesh_state_np1[i].u) / 2
        S += T(0.5) * m̄ * ū^2 * T(dt)
    end

    # Cholesky per segment
    for j in 1:N
        j_right = j == N ? 1 : j + 1
        # Periodic offset for the right vertex of the last segment.
        x_right_n  = j == N ? mesh_state_n[j_right].x  + L_box : mesh_state_n[j_right].x
        x_right_np1 = j == N ? mesh_state_np1[j_right].x + L_box : mesh_state_np1[j_right].x
        x_left_n   = mesh_state_n[j].x
        x_left_np1 = mesh_state_np1[j].x

        u_left_n   = mesh_state_n[j].u
        u_left_np1 = mesh_state_np1[j].u
        u_right_n  = mesh_state_n[j_right].u
        u_right_np1 = mesh_state_np1[j_right].u

        divu_mid = midpoint_strain(u_left_n, u_right_n, u_left_np1, u_right_np1,
                                   x_left_n, x_right_n, x_left_np1, x_right_np1)
        J_mid = midpoint_J(x_left_n, x_right_n, x_left_np1, x_right_np1, T(Δm_vec[j]))
        # Use Track-C EOS at midpoint J, segment-frozen entropy.
        s_j = mesh_state_n[j].s   # frozen
        Mvv_mid = _mvv_ad(J_mid, s_j)

        ΔS_Ch = segment_cholesky_action(mesh_state_n[j].α, mesh_state_n[j].β,
                                        mesh_state_np1[j].α, mesh_state_np1[j].β,
                                        Mvv_mid, divu_mid, T(dt))
        S += T(Δm_vec[j]) * ΔS_Ch
    end

    return S
end
