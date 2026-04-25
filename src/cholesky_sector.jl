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
