# cholesky_sector.jl
#
# Hamilton–Pontryagin Lagrangian, discrete one-step action, and discrete
# Euler–Lagrange residual for the Cholesky sector (α, β) of the dfmm-2d
# variational scheme. 1D specialization, Phase 1 scope.
#
# ──────────────────────────────────────────────────────────────────────
# Continuous side (v2 §2.1 and §3.1; methods paper §3).
# ──────────────────────────────────────────────────────────────────────
#
# The boxed Hamilton equations (v2 eq:dfmm-cholesky-cov) are
#
#     D_t^{(0)} α = β,           D_t^{(1)} β = γ²/α,
#
# with γ² = M_vv(J,s) − β². The strain coupling (∂_x u) lives in the
# covariant derivative D_t^{(q)} (charge q), not in the Hamiltonian
# H_Ch = −½ α² γ² (v2 §3.1, eq:H-Ch). The weighted symplectic 2-form is
# ω = α² dα ∧ dβ (v2 eq:symp-form).
#
# Hamilton–Pontryagin Lagrangian (v2 eq:L-Ch, with the symplectic
# potential θ = −(α³/3) dβ chosen so the EL system reproduces the
# boxed equations with the standard EL convention; see "sign note"
# below for the relationship to v2's stated form).
#
#     L_Ch = −(α³/3) D_t^{(1)} β − H_Ch
#          = −(α³/3) D_t^{(1)} β + ½ α² (M_vv − β²).
#
# The continuous EL equations are:
#   δβ:  ∂L/∂β − d/dt (∂L/∂β̇) = 0
#         → −(α³/3)(∂_x u) − αβ·sign − (… α²α̇ …) = 0
#         → α² α̇ = α² β  → α̇ = β    (recovers D_t^{(0)} α = β with q=0).
#   δα:  ∂L/∂α = 0  (no α̇ term in L)
#         → −α² D_t^{(1)} β + α (M_vv − β²) = 0
#         → D_t^{(1)} β = γ²/α  (recovers the second boxed equation).
#
# Sign note. v2 writes L_Ch = +(α³/3) D_t^{(1)} β − H_Ch. The standard
# EL convention with that sign produces the boxed equations only if one
# also flips the sign of the symplectic potential, equivalently swaps
# the order of the wedge in ω. The two conventions describe the same
# physics; we use the form above so that EL on L_Ch with standard
# Lagrangian-mechanics signs yields the boxed Hamilton equations
# directly. See `reference/notes_phaseA0.md` for the algebraic check.
#
# ──────────────────────────────────────────────────────────────────────
# Discrete side (methods paper §9.4 discrete action, §9.5 discrete EL
# and parallel transport).
# ──────────────────────────────────────────────────────────────────────
#
# Discrete one-step action with midpoint quadrature:
#
#     ΔS_n = Δt · L_Ch|_{n+1/2}
#          = −(ᾱ³/3)(β_{n+1} − β_n)
#            − Δt (ᾱ³/3) (∂_x u)_{n+1/2} β̄
#            + (Δt/2) ᾱ² (M_vv − β̄²),
#
# with midpoint averages ᾱ = (α_n+α_{n+1})/2, β̄ = (β_n+β_{n+1})/2,
# and (∂_x u)_{n+1/2} the externally-supplied midpoint strain rate
# (Phase 1: a fixed scalar).
#
# The discrete EL system at the midpoint, equivalent to varying ΔS_n
# directly with respect to (α_{n+1}, β_{n+1}) and reading off the
# Hamilton equations at the midpoint, is the 2-equation residual
#
#     F₁ = (α_{n+1} − α_n)/Δt − β̄                                = 0,
#     F₂ = (β_{n+1} − β_n)/Δt + (∂_x u)_{n+1/2} β̄ − (M_vv − β̄²)/ᾱ = 0,
#
# i.e. the implicit midpoint discretization of the boxed Hamilton
# equations with the discrete D_t^{(q)} of methods-paper §9.5
# implementing the parallel-transport stencil. Symplectic to 2nd order;
# coincides with the variational integrator obtained from the midpoint
# discrete Lagrangian (Marsden–West 2001 §1.7 and §2.5).
#
# This is the residual evaluated by `cholesky_el_residual!` below and
# fed to the Newton solver in `newton_step.jl`.

using StaticArrays: SVector

"""
    cholesky_one_step_action(α_n, β_n, α_np1, β_np1, M_vv, divu_half, dt)

Discrete one-step action `ΔS_n` for the Cholesky sector with midpoint
quadrature (methods paper §9.4). Used for diagnostics and for the
symplecticity check in `test_phase1_symplectic.jl`. Not used by the
Newton step itself — the EL residual is implemented directly in
`cholesky_el_residual` from the boxed Hamilton equations at the
midpoint, which is algebraically equivalent.
"""
function cholesky_one_step_action(α_n, β_n, α_np1, β_np1, M_vv, divu_half, dt)
    ᾱ = (α_n + α_np1) / 2
    β̄ = (β_n + β_np1) / 2
    return -((ᾱ^3) / 3) * (β_np1 - β_n) -
           dt * ((ᾱ^3) / 3) * divu_half * β̄ +
           (dt / 2) * (ᾱ^2) * (M_vv - β̄^2)
end

"""
    cholesky_el_residual(q_np1, q_n, M_vv, divu_half, dt)

Discrete Euler–Lagrange residual for the Cholesky sector at one
midpoint-rule step. Returns the 2-vector `(F₁, F₂)` defined above.
The Newton solver in `newton_step.jl` finds `q_np1 = (α_{n+1}, β_{n+1})`
such that this residual vanishes.

Arguments:
- `q_np1::SVector{2}` — unknown `(α_{n+1}, β_{n+1})`.
- `q_n::SVector{2}`   — known `(α_n, β_n)`.
- `M_vv`              — externally-supplied second velocity moment
                        (constant in Phase 1 — fixed J, s).
- `divu_half`         — externally-supplied midpoint strain rate
                        `(∂_x u)_{n+1/2}` (a scalar in Phase 1).
- `dt`                — timestep `Δt`.

The residual is equation-by-equation:

    F₁ = (α_{n+1} − α_n)/Δt − β̄,                                 (D_t^{(0)} α = β)
    F₂ = (β_{n+1} − β_n)/Δt + (∂_x u)_{n+1/2} β̄ − (M_vv − β̄²)/ᾱ.  (D_t^{(1)} β = γ²/α)

Both lines are the implicit-midpoint discretization of the boxed
Hamilton equations (v2 eq:dfmm-cholesky-cov), with the connection
contribution provided by `D_t_q` from `discrete_transport.jl`.
"""
function cholesky_el_residual(q_np1, q_n, M_vv, divu_half, dt)
    α_n, β_n = q_n[1], q_n[2]
    α_np1, β_np1 = q_np1[1], q_np1[2]
    ᾱ = (α_n + α_np1) / 2
    β̄ = (β_n + β_np1) / 2
    γ²_bar = M_vv - β̄^2
    # F₁: D_t^{(0)} α = β   (charge 0, no strain coupling)
    F1 = D_t_q(α_n, α_np1, divu_half, 0, dt) - β̄
    # F₂: D_t^{(1)} β = γ²/α  (charge 1, strain coupling via D_t)
    F2 = D_t_q(β_n, β_np1, divu_half, 1, dt) - γ²_bar / ᾱ
    return SVector{2}(F1, F2)
end

"""
    cholesky_hamiltonian(α, β, M_vv)

Cholesky-sector Hamiltonian density `H_Ch = −½ α² γ² = −½ α² (M_vv − β²)`
(v2 eq:H-Ch). Conserved by the continuous flow when `(∂_x u) = 0` and
`M_vv` constant; used for the symplecticity diagnostic in Phase 1.
"""
cholesky_hamiltonian(α, β, M_vv) = -0.5 * α^2 * (M_vv - β^2)

# ──────────────────────────────────────────────────────────────────────
# Phase 2: full deterministic Euler–Lagrange residual on a multi-segment
# periodic Lagrangian-mass mesh.
# ──────────────────────────────────────────────────────────────────────
#
# The Phase-2 EL system (see `reference/notes_phase2_discretization.md`)
# has 4N unknowns per timestep on a periodic N-segment mesh:
#
#     y = (x_1, u_1, α_1, β_1, x_2, u_2, α_2, β_2, …, x_N, u_N, α_N, β_N)
#
# packed in segment-major order. Per segment j (cyclic), four
# residuals per *vertex i* / *segment j*:
#
#     F^x_i = (x_i^{n+1} − x_i^n)/Δt − (u_i^n + u_i^{n+1})/2
#     F^u_i = (u_i^{n+1} − u_i^n)/Δt
#             + (P̄_xx,i − P̄_xx,i-1) / m̄_i
#     F^α_j = (α_j^{n+1} − α_j^n)/Δt − (β_j^n + β_j^{n+1})/2
#     F^β_j = (β_j^{n+1} − β_j^n)/Δt
#             + (∂_x u)_j^{n+1/2} · β̄_j
#             − (M̄_vv,j − β̄_j²) / ᾱ_j
#
# with `m̄_i = (Δm_{i-1} + Δm_i)/2` cyclically, midpoint-J segment
# pressure `P̄_xx,j = M̄_vv,j / J̄_j` (i.e. ρ̄_j · M̄_vv,j), midpoint
# `J̄_j = (x̄_{j+1} − x̄_j)/Δm_j`, midpoint Eulerian strain
# `(∂_x u)_j = (ū_{j+1} − ū_j)/(x̄_{j+1} − x̄_j)`, and entropy frozen
# (`s_j^{n+1} = s_j^n`). Periodic wrap: x_{N+1} ≡ x_1 + L_box,
# Δm_0 ≡ Δm_N.
#
# Sparsity: each row depends on at most three vertex blocks (left,
# self, right), giving a banded Jacobian of bandwidth 2·4 = 8.

"""
    det_el_residual(y_np1::AbstractVector, y_n::AbstractVector,
                    Δm::AbstractVector, s::AbstractVector,
                    L_box, dt)

Residual vector of the Phase-2 deterministic discrete EL system on a
periodic mesh.

Arguments:
- `y_np1` — flat unknown vector of length `4N`, packed segment-major
  as `(x_1, u_1, α_1, β_1, x_2, u_2, α_2, β_2, …)`.
- `y_n`   — known flat vector at time `t_n`, same packing.
- `Δm`    — vector of segment masses, length `N`.
- `s`     — vector of cell-centered entropies, length `N` (frozen
  across the step in Phase 2).
- `L_box` — Lagrangian box length for periodic wrap.
- `dt`    — timestep.

Returns the residual `F` as an `AbstractVector` of the same length
and element type as `y_np1` (so it remains AD-friendly with
ForwardDiff dual numbers).

Verified by hand on a 2-segment uniform-state mesh (segment lengths
equal, all `u = 0`, `α = α_0`, `β = 0`): all four residuals per
segment vanish identically, since pressures telescope to zero and
the Cholesky residuals reduce to Phase 1's zero-strain form which is
satisfied at fixed point `(α_0, 0)`.
"""
function det_el_residual(y_np1::AbstractVector, y_n::AbstractVector,
                         Δm::AbstractVector, s::AbstractVector,
                         L_box, dt)
    N = length(Δm)
    @assert length(y_np1) == 4N
    @assert length(y_n) == 4N
    @assert length(s) == N

    Tres = promote_type(eltype(y_np1), eltype(y_n), eltype(Δm), eltype(s),
                        typeof(L_box), typeof(dt))
    F = similar(y_np1, Tres, 4N)

    # Helpers: index → (x, u, α, β) per segment j (1-based).
    @inline get_x(y, j) = y[4*(j-1) + 1]
    @inline get_u(y, j) = y[4*(j-1) + 2]
    @inline get_α(y, j) = y[4*(j-1) + 3]
    @inline get_β(y, j) = y[4*(j-1) + 4]

    # Per-segment midpoint quantities. Compute once per segment:
    # midpoint J_j, M_vv_j, P_xx_j, strain_j. Then assemble residuals
    # row-by-row.
    J̄ = similar(y_np1, Tres, N)
    M̄vv = similar(y_np1, Tres, N)
    P̄xx = similar(y_np1, Tres, N)
    div̄u = similar(y_np1, Tres, N)

    @inbounds for j in 1:N
        j_right = j == N ? 1 : j + 1
        x_left_n   = get_x(y_n, j)
        x_left_np1 = get_x(y_np1, j)
        # Periodic wrap on the right vertex of segment N.
        wrap = (j == N) ? L_box : zero(L_box)
        x_right_n   = get_x(y_n, j_right)   + wrap
        x_right_np1 = get_x(y_np1, j_right) + wrap
        u_left_n   = get_u(y_n, j)
        u_left_np1 = get_u(y_np1, j)
        u_right_n   = get_u(y_n, j_right)
        u_right_np1 = get_u(y_np1, j_right)

        x̄_left  = (x_left_n  + x_left_np1)  / 2
        x̄_right = (x_right_n + x_right_np1) / 2
        ū_left  = (u_left_n  + u_left_np1)  / 2
        ū_right = (u_right_n + u_right_np1) / 2

        Δx̄ = x̄_right - x̄_left
        J̄[j] = Δx̄ / Δm[j]
        # EOS midpoint M_vv at frozen entropy.
        # Use the AD-friendly variant (Track-C's `Mvv(J, s)` declares
        # `::Float64` and would strip ForwardDiff `Dual` tags).
        M̄vv[j] = _mvv_ad(J̄[j], s[j])
        # Pressure ρ M_vv = M_vv / J.
        P̄xx[j] = M̄vv[j] / J̄[j]
        # Eulerian strain at midpoint.
        div̄u[j] = (ū_right - ū_left) / Δx̄
    end

    # Per-vertex / per-segment residuals.
    @inbounds for i in 1:N
        x_i_n   = get_x(y_n, i)
        x_i_np1 = get_x(y_np1, i)
        u_i_n   = get_u(y_n, i)
        u_i_np1 = get_u(y_np1, i)

        ū_i = (u_i_n + u_i_np1) / 2

        # F^x: x_dot = u
        F[4*(i-1) + 1] = (x_i_np1 - x_i_n) / dt - ū_i

        # F^u: m̄_i u_dot = -(P_i - P_{i-1})
        i_left = i == 1 ? N : i - 1
        m̄_i = (Δm[i_left] + Δm[i]) / 2
        F[4*(i-1) + 2] =
            (u_i_np1 - u_i_n) / dt +
            (P̄xx[i] - P̄xx[i_left]) / m̄_i

        # F^α and F^β at segment i (in the segment-major packing,
        # row i corresponds to segment i).
        α_n   = get_α(y_n, i)
        α_np1 = get_α(y_np1, i)
        β_n   = get_β(y_n, i)
        β_np1 = get_β(y_np1, i)
        ᾱ = (α_n + α_np1) / 2
        β̄ = (β_n + β_np1) / 2
        γ²_bar = M̄vv[i] - β̄^2

        F[4*(i-1) + 3] = (α_np1 - α_n) / dt - β̄
        F[4*(i-1) + 4] =
            (β_np1 - β_n) / dt + div̄u[i] * β̄ - γ²_bar / ᾱ
    end

    return F
end

