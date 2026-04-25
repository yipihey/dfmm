# deviatoric.jl
#
# Phase 5: deviatoric stress sector. 1D specialization of the symmetric
# pressure tensor decomposes into the parallel kinetic moment
# `P_xx = ρ M_vv` and the perpendicular pressure `P_⊥`; the scalar
# `Π = P_xx − P_⊥` is the anisotropy. py-1d carries `P_⊥` as an
# advected field and applies BGK relaxation each step that drives
# both `P_xx` and `P_⊥` toward the isotropic mean
# `P_iso = (P_xx + 2 P_⊥)/3`.
#
# The variational analogue (methods paper §3.3, v2 eq. 36):
#
#     D_t Π = -Π/τ - 2 η S^dev + Q_Π
#
# In 1D the only deviatoric strain is `S^dev = ∂_x u`. The Phase-5
# brief (`reference/MILESTONE_1_PLAN.md` Phase 5) recommends the
# **hard-constraint** discretization: integrate the BGK ODE for `Π`
# *outside* the Newton system after the implicit step. With η = 0 the
# scheme matches py-1d's discrete BGK exactly (an exponential decay
# of `Π` toward zero).
#
# We track `P_⊥` per segment as a charge-1 (in mass-density sense)
# Lagrangian field. Per step:
#
#   1. Hyperbolic transport (no BGK): `(P_⊥/ρ)^{n+1} = (P_⊥/ρ)^n`
#      (i.e. `P_⊥^{n+1, transport} = P_⊥^n · ρ^{n+1}/ρ^n`); equivalent
#      to py-1d's flux `∂_x(u P_⊥) = 0` when re-expressed in
#      Lagrangian-mass coordinates with `dx = J dm` and Δm fixed.
#   2. BGK relax: relax both `P_xx` and `P_⊥` toward the isotropic
#      mean with the closed-form exponential
#          `Π^{n+1} = Π^{transport} · exp(-Δt/τ)`
#      (η = 0 case). With `P_xx = ρ M_vv(J, s)` already determined
#      by the variational integrator, `P_⊥^{n+1} = P_xx^{n+1} -
#      Π^{n+1}` closes the update.
#
# This is the operator-split implicit-midpoint discretization of
# `Π̇ = -Π/τ` to second order in `Δt`.
#
# Functions exported here are pure on a per-segment basis to keep
# them unit-testable. The mesh-level driver lives in `newton_step.jl`
# (`det_step!` post-implicit hook) so the BGK update follows
# immediately after the Newton solve.
#
# References:
#   methods paper §3.3 "Deviatoric stress as a dynamical variable"
#   v2 eq. 36 (Hamilton-Pontryagin form of the BGK constraint)
#   py-1d/dfmm/schemes/cholesky.py lines 153-179 (discrete BGK update)

"""
    deviatoric_bgk_step(Π_n, divu_half, τ, η, dt) -> Π_np1

Hard-constraint implicit-midpoint update for the deviatoric scalar `Π`
under the BGK relaxation `Π̇ + Π/τ = -2 η (∂_x u)` (methods paper §3.3,
v2 eq. 36 1D specialization). The midpoint formula is

    Π^{n+1} = (Π^n - 2 η · (∂_x u)_{n+1/2} · Δt) / (1 + Δt/τ).

For η = 0 (the Phase-5 default that matches py-1d), this reduces to
implicit-midpoint exponential decay; the explicit-exponential form
`Π^{n+1} = Π^n · exp(-Δt/τ)` is also correct (and is what py-1d
uses). Both are second-order; we use the bilinear form here so
finite-η problems work with no code change.

Stable for any `Δt/τ > 0`.
"""
function deviatoric_bgk_step(Π_n::Real, divu_half::Real,
                             τ::Real, η::Real, dt::Real)
    # Implicit-midpoint of Π̇ + Π/τ = -2 η div(u):
    #   (Π_np1 - Π_n)/Δt + (Π_np1 + Π_n)/(2τ) = -2 η div(u)
    # Rearrange:
    #   Π_np1 (1/Δt + 1/(2τ)) = Π_n (1/Δt - 1/(2τ)) - 2 η div(u)
    # Multiply by Δt:
    #   Π_np1 (1 + Δt/(2τ)) = Π_n (1 - Δt/(2τ)) - 2 η div(u) Δt
    half_ratio = dt / (2 * τ)
    num = Π_n * (1 - half_ratio) - 2 * η * divu_half * dt
    den = 1 + half_ratio
    return num / den
end

"""
    deviatoric_bgk_step_exponential(Π_n, dt, τ) -> Π_np1

Exact-exponential BGK update for `Π` with `η = 0`: `Π^{n+1} = Π^n exp(-Δt/τ)`.
This is what py-1d uses (`cholesky.py` line 154) and is what the
Phase-5 regression test compares against. Bit-equality with py-1d's
`decay = np.exp(-dt/tau)` requires this form (the bilinear form in
`deviatoric_bgk_step` is correct to second order but differs from
py-1d's exact-exponential in `O(Δt²/τ²)` terms).
"""
deviatoric_bgk_step_exponential(Π_n::Real, dt::Real, τ::Real) =
    Π_n * exp(-dt / τ)

"""
    pperp_advect_lagrangian(Pp_n, ρ_n, ρ_np1) -> Pp_np1_transport

Transport of the perpendicular pressure `P_⊥` under the hyperbolic
step in Lagrangian coordinates. The Eulerian conservation law
`∂_t P_⊥ + ∂_x(u P_⊥) = 0` rewrites in Lagrangian-mass coordinates
as `D_t P_⊥ = -P_⊥ ∂_x u`, equivalently `D_t (P_⊥/ρ) = 0` (since
`D_t ρ = -ρ ∂_x u` from continuity). So
`(P_⊥/ρ)^{n+1, transport} = (P_⊥/ρ)^n`, hence

    P_⊥^{n+1, transport} = P_⊥^n · ρ^{n+1}/ρ^n.

This is the post-Newton hyperbolic-transport update for `P_⊥`,
applied before the BGK relaxation closes the step. The implicit-
midpoint Newton solve already advances ρ self-consistently with the
parallel pressure `P_xx = ρ M_vv(J, s)`; we use the new ρ here.
"""
pperp_advect_lagrangian(Pp_n::Real, ρ_n::Real, ρ_np1::Real) =
    Pp_n * ρ_np1 / ρ_n

"""
    bgk_relax_pressures(Pxx_n, Pp_n, dt, τ) -> (Pxx_np1, Pp_np1)

Apply the joint BGK relaxation to `(P_xx, P_⊥)` toward their isotropic
mean `P_iso = (P_xx + 2 P_⊥)/3`. Matches py-1d's update exactly
(`cholesky.py` lines 165-167):

    decay = exp(-Δt/τ)
    P_iso = (P_xx + 2 P_⊥)/3
    P_xx^{n+1} = P_iso + (P_xx - P_iso) · decay
    P_⊥^{n+1}  = P_iso + (P_⊥  - P_iso) · decay

Equivalent to `Π^{n+1} = Π^n · decay` where `Π = P_xx - P_⊥`, since
the relaxation conserves `P_iso` (no isotropic-mean change).

The variational integrator's parallel pressure `P_xx = ρ M_vv` is
already determined by the (α, β, J, s) state at the new time;
this function's role is to relax the **anisotropy**, leaving the
total `(P_xx + 2 P_⊥)/3` unchanged. After this step `Π = P_xx - P_⊥`
has decayed by factor `exp(-Δt/τ)`.
"""
function bgk_relax_pressures(Pxx_n::Real, Pp_n::Real, dt::Real, τ::Real)
    decay = exp(-dt / τ)
    P_iso = (Pxx_n + 2 * Pp_n) / 3
    Pxx_new = P_iso + (Pxx_n - P_iso) * decay
    Pp_new  = P_iso + (Pp_n  - P_iso) * decay
    return (Pxx_new, Pp_new)
end

"""
    pperp_step(Pp_n, ρ_n, ρ_np1, Pxx_np1, dt, τ) -> Pp_np1

One operator-split update for `P_⊥`:

  1. Lagrangian transport: `Pp_transport = Pp_n · ρ_np1/ρ_n`.
  2. BGK relaxation toward `P_iso` with `P_xx = Pxx_np1`.

Returns the new `P_⊥` after both substeps. The variational integrator
provides `Pxx_np1 = ρ_np1 · M_vv(J_np1, s)`; this routine then
relaxes the anisotropy `Π = Pxx_np1 - Pp_transport` by a factor
`exp(-Δt/τ)`. Matches py-1d's split-step physics applied to the
Lagrangian frame.

In the τ → 0 limit the relaxation is instantaneous and `P_⊥ → P_xx`
(Euler-isotropic). In the τ → ∞ limit BGK is off and `P_⊥` simply
advects, retaining whatever anisotropy the transport produces (the
collisionless limit).
"""
function pperp_step(Pp_n::Real, ρ_n::Real, ρ_np1::Real,
                    Pxx_np1::Real, dt::Real, τ::Real)
    Pp_transport = pperp_advect_lagrangian(Pp_n, ρ_n, ρ_np1)
    _, Pp_new = bgk_relax_pressures(Pxx_np1, Pp_transport, dt, τ)
    return Pp_new
end
