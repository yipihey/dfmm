# enzyme_adjoint.jl
#
# Reverse-mode end-to-end gradient for the Phase-1 Cholesky-sector
# variational integrator (`cholesky_step` / `cholesky_run`) via the
# implicit function theorem.
#
# At the converged `q* = cholesky_step(q_n, M_vv, divu_half, dt)` the
# discrete EL residual satisfies `F(q*, q_n, M_vv, divu_half, dt) = 0`.
# Differentiating that constraint at fixed `q*`:
#
#     J_q dq_np1 + ∂F/∂q_n dq_n + ∂F/∂M_vv dM_vv
#                + ∂F/∂divu_half ddivu_half + ∂F/∂dt ddt = 0,
#
# with `J_q := ∂F/∂q_np1`. In reverse mode, given the output cotangent
# `q̄_np1` we solve `J_q^T λ = q̄_np1` and read off the parameter
# cotangents as `-λ^T ∂F/∂p`. Implementing this rule by hand means we
# never differentiate through the Newton iterations themselves — the
# gradient is exact and independent of solver tolerance.
#
# The vector-Jacobian product `λ^T ∂F/∂p` is computed with Enzyme reverse
# mode against `cholesky_el_residual`; `J_q` is built from two reverse
# passes against the same residual with unit seeds. `q*` is treated as
# `Const` everywhere so Enzyme never tries to trace the solver.
#
# See `specs/05_julia_ecosystem_survey.md` §4.1 for the design intent.

using Enzyme: Enzyme, Reverse, Forward, Active, Const, Duplicated
using StaticArrays: SVector, SMatrix

# Scalar residual dot-product `λ · F(q_np1, q_n, M_vv, divu, dt)`.
# We unpack the SVector inputs into scalar arguments so Enzyme's
# Active/Const annotations are unambiguous on each component.
@inline function _residual_dot(λ1::Real, λ2::Real,
                               α_np1::Real, β_np1::Real,
                               α_n::Real, β_n::Real,
                               M_vv::Real, divu::Real, dt::Real)
    F = cholesky_el_residual(SVector{2}(α_np1, β_np1),
                             SVector{2}(α_n, β_n),
                             M_vv, divu, dt)
    return λ1 * F[1] + λ2 * F[2]
end

# Compute `λ^T · ∂F/∂(α_np1, β_np1, α_n, β_n, M_vv, divu, dt)` via a
# single Enzyme reverse pass. Returns the seven scalar cotangents in
# argument order (the first two are the q_np1 entries, the remaining
# five are the parameter entries).
@inline function _residual_vjp_full(λ1::Float64, λ2::Float64,
                                    α_np1::Float64, β_np1::Float64,
                                    α_n::Float64, β_n::Float64,
                                    M_vv::Float64, divu::Float64, dt::Float64)
    (grads,) = Enzyme.autodiff(
        Reverse, _residual_dot, Active,
        Const(λ1), Const(λ2),
        Active(α_np1), Active(β_np1),
        Active(α_n), Active(β_n),
        Active(M_vv), Active(divu), Active(dt),
    )
    # grads: (nothing, nothing, ∂/∂α_np1, ∂/∂β_np1, ∂/∂α_n, ∂/∂β_n,
    #         ∂/∂M_vv, ∂/∂divu, ∂/∂dt)
    return (grads[3], grads[4], grads[5], grads[6],
            grads[7], grads[8], grads[9])
end

# Build the 2×2 Jacobian `J_q = ∂F/∂q_np1` at the converged state by
# two reverse passes with the canonical λ-seeds (1,0) and (0,1).
@inline function _jacobian_q(q_np1::SVector{2,Float64},
                             q_n::SVector{2,Float64},
                             M_vv::Float64, divu::Float64, dt::Float64)
    α_np1, β_np1 = q_np1[1], q_np1[2]
    α_n,  β_n  = q_n[1],  q_n[2]
    # Row 1 of J = (∂F1/∂α_np1, ∂F1/∂β_np1): seed λ = (1, 0).
    J11, J12, _, _, _, _, _ = _residual_vjp_full(
        1.0, 0.0, α_np1, β_np1, α_n, β_n, M_vv, divu, dt)
    # Row 2 of J = (∂F2/∂α_np1, ∂F2/∂β_np1): seed λ = (0, 1).
    J21, J22, _, _, _, _, _ = _residual_vjp_full(
        0.0, 1.0, α_np1, β_np1, α_n, β_n, M_vv, divu, dt)
    return SMatrix{2,2,Float64}(J11, J21, J12, J22)
end

"""
    cholesky_step_pullback(q_n, M_vv, divu_half, dt; kwargs...)
        -> (q_np1, pullback)

Run one `cholesky_step` and return the converged state together with
its reverse-mode pullback closure. Calling `pullback(q̄_np1)` returns
the cotangents `(q̄_n, M̄_vv, divū_half, d̄t)` of the four inputs given
an output cotangent `q̄_np1`.

The pullback uses the implicit function theorem at the converged state
and does **not** trace through the Newton iterations: `q_np1` is held
`Const` for Enzyme and the linear adjoint solve is on the dense 2×2
`J_q^T`. Gradients are therefore exact up to the residual norm of the
forward solve (driven below `10·abstol` by the existing tolerance
check in `cholesky_step`).

Designed to chain across `cholesky_run` — see `cholesky_run_gradient`.
"""
function cholesky_step_pullback(q_n::SVector{2,Float64},
                                M_vv::Real, divu_half::Real, dt::Real;
                                kwargs...)
    M_vv64 = Float64(M_vv); divu64 = Float64(divu_half); dt64 = Float64(dt)
    q_np1 = cholesky_step(q_n, M_vv64, divu64, dt64; kwargs...)
    pullback = let q_np1=q_np1, q_n=q_n,
                   M_vv=M_vv64, divu=divu64, dt=dt64
        function (q̄_np1::SVector{2,Float64})
            # Solve J_q^T λ = q̄_np1 for the adjoint multiplier.
            J = _jacobian_q(q_np1, q_n, M_vv, divu, dt)
            λ = transpose(J) \ q̄_np1
            # Cotangents on (q_n, M_vv, divu, dt) are -λ^T ∂F/∂p.
            _, _, dα_n, dβ_n, dM_vv, ddivu, ddt = _residual_vjp_full(
                λ[1], λ[2],
                q_np1[1], q_np1[2],
                q_n[1],  q_n[2],
                M_vv, divu, dt,
            )
            return (SVector{2,Float64}(-dα_n, -dβ_n),
                    -dM_vv, -ddivu, -ddt)
        end
    end
    return q_np1, pullback
end

"""
    cholesky_run_gradient(loss, q_0, M_vv, divu_half, dt, N; kwargs...)
        -> (loss_value, q̄_0, M̄_vv, divū_half, d̄t)

End-to-end reverse-mode gradient of the scalar functional
`loss(q_N)` evaluated on the terminal state of an N-step Cholesky-sector
trajectory `cholesky_run(q_0, M_vv, divu_half, dt, N; kwargs...)`.

`loss` must be a callable `SVector{2,Float64} -> Real`. Its gradient at
`q_N` is computed with Enzyme reverse mode. The trajectory is replayed
backwards step-by-step through `cholesky_step_pullback`, accumulating
cotangents on the time-constant parameters `M_vv`, `divu_half`, `dt`.

Returns the loss value, the initial-condition cotangent `q̄_0`, and the
three scalar parameter cotangents.
"""
# Scalar form of cholesky_el_residual returning one component, used as
# the function differentiated by Enzyme forward when assembling the
# parameter-side JVP into the 2-vector RHS of the IFT linear system.
@inline function _residual_component(component::Int,
                                     α_np1::Real, β_np1::Real,
                                     α_n::Real, β_n::Real,
                                     M_vv::Real, divu::Real, dt::Real)
    F = cholesky_el_residual(SVector{2}(α_np1, β_np1),
                             SVector{2}(α_n, β_n),
                             M_vv, divu, dt)
    return F[component]
end

# Apply `∂F/∂(q_n, M_vv, divu, dt)` to the input-tangent bundle
# `(q̇_n, Ṁ_vv, divu̇, ṫ)` in one forward Enzyme pass per output
# component. Returns the 2-vector `∂F/∂p · ṗ` at fixed `q_np1`.
@inline function _residual_jvp_params(α_np1::Float64, β_np1::Float64,
                                       α_n::Float64, β_n::Float64,
                                       M_vv::Float64, divu::Float64, dt::Float64,
                                       q̇_n::SVector{2,Float64},
                                       Ṁ_vv::Float64, divu̇::Float64, ṫ::Float64)
    # Two scalar forward calls — one per output component — each with
    # the full parameter tangent bundle threaded through Duplicated.
    (dF1,) = Enzyme.autodiff(
        Forward, _residual_component, Duplicated,
        Const(1),
        Const(α_np1), Const(β_np1),
        Duplicated(α_n, q̇_n[1]), Duplicated(β_n, q̇_n[2]),
        Duplicated(M_vv, Ṁ_vv),
        Duplicated(divu, divu̇),
        Duplicated(dt, ṫ),
    )
    (dF2,) = Enzyme.autodiff(
        Forward, _residual_component, Duplicated,
        Const(2),
        Const(α_np1), Const(β_np1),
        Duplicated(α_n, q̇_n[1]), Duplicated(β_n, q̇_n[2]),
        Duplicated(M_vv, Ṁ_vv),
        Duplicated(divu, divu̇),
        Duplicated(dt, ṫ),
    )
    return SVector{2,Float64}(dF1, dF2)
end

"""
    cholesky_step_jvp(q_n, M_vv, divu_half, dt, q̇_n, Ṁ_vv, divu̇, ṫ; kwargs...)
        -> (q_np1, q̇_np1)

Forward-mode pushforward (JVP) of `cholesky_step` via the implicit
function theorem. Given input tangents `(q̇_n, Ṁ_vv, divu̇, ṫ)`, returns
the converged state `q_np1` together with the output tangent `q̇_np1`
that satisfies the linearized constraint at `q*`:

    J_q q̇_np1 = -[∂F/∂q_n q̇_n + ∂F/∂M_vv Ṁ_vv
                  + ∂F/∂divu_half divu̇ + ∂F/∂dt ṫ].

The Jacobian `J_q` is built from two Enzyme reverse passes with unit
seeds; the parameter-side directional derivative on the right-hand side
is one Enzyme forward pass per output component with `q_np1` held
`Const`. The same caveats as `cholesky_step_pullback` apply (gradient
is exact up to forward residual norm; never differentiates through
Newton iterations).

Useful as a cheap pushforward for small-input parameter sweeps and as a
cross-check against the reverse-mode pullback via the
forward-reverse duality `⟨q̄, J·q̇⟩ = ⟨J^T·q̄, q̇⟩`.
"""
function cholesky_step_jvp(q_n::SVector{2,Float64},
                            M_vv::Real, divu_half::Real, dt::Real,
                            q̇_n::SVector{2,Float64},
                            Ṁ_vv::Real, divu̇::Real, ṫ::Real;
                            kwargs...)
    M_vv64 = Float64(M_vv); divu64 = Float64(divu_half); dt64 = Float64(dt)
    Ṁ64  = Float64(Ṁ_vv);  divu̇64 = Float64(divu̇);     ṫ64  = Float64(ṫ)
    q_np1 = cholesky_step(q_n, M_vv64, divu64, dt64; kwargs...)
    J = _jacobian_q(q_np1, q_n, M_vv64, divu64, dt64)
    rhs = _residual_jvp_params(q_np1[1], q_np1[2],
                                q_n[1],  q_n[2],
                                M_vv64,  divu64, dt64,
                                q̇_n, Ṁ64, divu̇64, ṫ64)
    q̇_np1 = J \ (-rhs)
    return q_np1, q̇_np1
end

"""
    cholesky_run_jvp(q_0, M_vv, divu_half, dt, N,
                     q̇_0, Ṁ_vv, divu̇, ṫ; kwargs...)
        -> (traj, traj_dot)

Multi-step forward-mode JVP. Returns the trajectory
`[q_0, q_1, …, q_N]` together with its tangent trajectory
`[q̇_0, q̇_1, …, q̇_N]` propagated step-by-step under fixed time-constant
parameter tangents `(Ṁ_vv, divu̇, ṫ)`.
"""
function cholesky_run_jvp(q_0::SVector{2,Float64},
                           M_vv::Real, divu_half::Real, dt::Real,
                           N::Integer,
                           q̇_0::SVector{2,Float64},
                           Ṁ_vv::Real, divu̇::Real, ṫ::Real;
                           kwargs...)
    traj   = Vector{SVector{2,Float64}}(undef, N + 1)
    tdot   = Vector{SVector{2,Float64}}(undef, N + 1)
    traj[1] = q_0
    tdot[1] = q̇_0
    @inbounds for n in 1:N
        traj[n+1], tdot[n+1] = cholesky_step_jvp(
            traj[n], M_vv, divu_half, dt,
            tdot[n], Ṁ_vv, divu̇, ṫ;
            kwargs...,
        )
    end
    return traj, tdot
end

function cholesky_run_gradient(loss::F,
                               q_0::SVector{2,Float64},
                               M_vv::Real, divu_half::Real, dt::Real,
                               N::Integer;
                               kwargs...) where {F}
    M_vv64 = Float64(M_vv); divu64 = Float64(divu_half); dt64 = Float64(dt)

    # Forward pass: store full trajectory.
    traj = Vector{SVector{2,Float64}}(undef, N + 1)
    traj[1] = q_0
    @inbounds for n in 1:N
        traj[n+1] = cholesky_step(traj[n], M_vv64, divu64, dt64; kwargs...)
    end
    L = loss(traj[N+1])

    # Cotangent seed on the terminal state from the loss gradient.
    # Enzyme reverse on a scalar function of two scalars.
    # Mark the loss closure `Const` so Enzyme doesn't try to
    # differentiate through captures (e.g. target Vectors). Inputs are
    # the two scalars of the terminal state.
    _loss_pair(α, β) = loss(SVector{2,Float64}(α, β))
    (lgrads,) = Enzyme.autodiff(
        Reverse, Const(_loss_pair), Active,
        Active(traj[N+1][1]), Active(traj[N+1][2]),
    )
    q̄ = SVector{2,Float64}(lgrads[1], lgrads[2])

    # Reverse pass: step n maps q̄_{n+1} → q̄_n with parameter accruals.
    M̄_vv = 0.0; divū = 0.0; d̄t = 0.0
    @inbounds for n in N:-1:1
        q_np1 = traj[n+1]
        q_n   = traj[n]
        J = _jacobian_q(q_np1, q_n, M_vv64, divu64, dt64)
        λ = transpose(J) \ q̄
        _, _, dα_n, dβ_n, dM, dd, dt_grad = _residual_vjp_full(
            λ[1], λ[2],
            q_np1[1], q_np1[2],
            q_n[1],  q_n[2],
            M_vv64,  divu64, dt64,
        )
        q̄    = SVector{2,Float64}(-dα_n, -dβ_n)
        M̄_vv += -dM
        divū += -dd
        d̄t   += -dt_grad
    end
    return L, q̄, M̄_vv, divū, d̄t
end
