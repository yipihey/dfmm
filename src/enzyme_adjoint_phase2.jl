# enzyme_adjoint_phase2.jl
#
# Reverse-mode end-to-end gradient for the Phase-2 multi-segment
# deterministic variational integrator (`det_el_residual` Newton
# solve) via the implicit function theorem. Scope: periodic boundary
# conditions, no artificial viscosity (`q_kind = :none`) — the minimum
# Phase-2 surface that closes against the existing regression tests.
# Inflow/outflow and `:vNR_linear_quadratic` are routine extensions; we
# leave them for follow-on PRs.
#
# Pipeline at each step, given y_n, parameters p = (Δm, s, L_box, dt):
#
#   forward:  y_np1 = solve  F(y_np1; y_n, p) = 0       (NewtonRaphson + AutoForwardDiff)
#   J_q    :  J = ∂F/∂y_np1 |_{y_np1*}                  (ForwardDiff.jacobian, dense)
#   adjoint:  J_q^T λ = ȳ_np1                           (LinearAlgebra dense solve)
#   VJP    :  (ȳ_n, Δm̄, s̄, L̄_box, d̄t) = -λ^T ∂F/∂p     (Enzyme reverse)
#
# The Jacobian is built dense via ForwardDiff because the existing
# Newton driver already drives this same path internally; sparse
# coloring (det_jac_sparsity + DifferentiationInterface) is a routine
# performance upgrade once the dense adjoint is validated.

using ForwardDiff: ForwardDiff
using NonlinearSolve: NonlinearProblem, NonlinearFunction, solve,
                      NewtonRaphson, AutoForwardDiff, ReturnCode
using LinearAlgebra: LinearAlgebra

# Pure-functional one-step Newton solve: no mesh mutation, no
# post-Newton entropy / P_⊥ branches. Returns a fresh y_np1 Vector.
# Mirrors `det_step!`'s Newton inner loop for the
# `bc=:periodic, q_kind=:none, tau=nothing` path.
function det_step_functional(y_n::AbstractVector{Float64},
                              Δm::AbstractVector{Float64},
                              s::AbstractVector{Float64},
                              L_box::Real, dt::Real;
                              abstol::Real = 1e-13,
                              reltol::Real = 1e-13,
                              maxiters::Int = 50)
    N = length(Δm)
    @assert length(y_n) == 4N
    @assert length(s) == N
    Δm64 = collect(Float64, Δm)
    s64  = collect(Float64, s)
    L64  = Float64(L_box)
    dt64 = Float64(dt)

    y0 = explicit_euler_guess(collect(Float64, y_n), Δm64, s64, L64, dt64)
    p = (collect(Float64, y_n), Δm64, s64, L64, dt64)
    f = (u, p_in) -> begin
        y_n_in, Δm_in, s_in, L_box_in, dt_in = p_in
        return det_el_residual(u, y_n_in, Δm_in, s_in, L_box_in, dt_in;
                                q_kind = :none, bc = :periodic)
    end

    prob = NonlinearProblem(f, y0, p)
    sol = solve(prob, NewtonRaphson(; autodiff = AutoForwardDiff());
                abstol = abstol, reltol = reltol, maxiters = maxiters)
    res = f(sol.u, p)
    if maximum(abs, res) > 1e6 * abstol && sol.retcode != ReturnCode.Success
        error("det_step_functional Newton solve failed: retcode = $(sol.retcode), " *
              "‖residual‖∞ = $(maximum(abs, res))")
    end
    return collect(Float64, sol.u)
end

# Scalar reduction `λ · F(y_np1, y_n, Δm, s, L_box, dt)` used by the
# Enzyme reverse pass to compute the parameter-side VJP.
@inline function _det_residual_dot(λ::AbstractVector{Float64},
                                    y_np1::AbstractVector{Float64},
                                    y_n::AbstractVector{Float64},
                                    Δm::AbstractVector{Float64},
                                    s::AbstractVector{Float64},
                                    L_box::Float64, dt::Float64)
    F = det_el_residual(y_np1, y_n, Δm, s, L_box, dt;
                        q_kind = :none, bc = :periodic)
    acc = 0.0
    @inbounds for i in eachindex(F)
        acc += λ[i] * F[i]
    end
    return acc
end

"""
    det_step_pullback(y_n, Δm, s, L_box, dt; kwargs...) -> (y_np1, pullback)

Phase-2 one-step IFT pullback (periodic BC, no q-viscosity). Runs the
deterministic Newton solve via `det_step_functional`, then returns a
pullback closure that, given an output cotangent `ȳ_np1`, returns the
five input cotangents `(ȳ_n, Δm̄, s̄, L̄_box, d̄t)`.

The dense `J_q^T λ = ȳ_np1` adjoint solve uses LinearAlgebra; the
parameter VJP `-λ^T ∂F/∂p` is one Enzyme reverse pass on
`det_el_residual` with the converged `y_np1*` held `Const`. Gradients
are exact up to the forward residual norm.
"""
function det_step_pullback(y_n::AbstractVector{Float64},
                            Δm::AbstractVector{Float64},
                            s::AbstractVector{Float64},
                            L_box::Real, dt::Real;
                            abstol::Real = 1e-13,
                            reltol::Real = 1e-13,
                            maxiters::Int = 50)
    L64  = Float64(L_box)
    dt64 = Float64(dt)
    y_n64 = collect(Float64, y_n)
    Δm64  = collect(Float64, Δm)
    s64   = collect(Float64, s)
    y_np1 = det_step_functional(y_n64, Δm64, s64, L64, dt64;
                                 abstol = abstol, reltol = reltol,
                                 maxiters = maxiters)

    # Dense Jacobian J = ∂F/∂y_np1 at the converged state.
    J = ForwardDiff.jacobian(
        u -> det_el_residual(u, y_n64, Δm64, s64, L64, dt64;
                              q_kind = :none, bc = :periodic),
        y_np1,
    )
    J_T = transpose(J)

    pullback = let y_np1=y_np1, y_n64=y_n64, Δm64=Δm64, s64=s64,
                   L64=L64, dt64=dt64, J_T=J_T
        function (ȳ_np1::AbstractVector{Float64})
            @assert length(ȳ_np1) == length(y_np1)
            λ = J_T \ collect(Float64, ȳ_np1)

            # VJP: -λ^T ∂F/∂p via Enzyme reverse on the scalar
            # _det_residual_dot. Vector inputs use `Duplicated(x, dx)`;
            # Enzyme accumulates into `dx`. Scalar inputs use `Active`.
            dy_n = zero(y_n64)
            dΔm  = zero(Δm64)
            ds   = zero(s64)
            (grads,) = Enzyme.autodiff(
                Reverse, _det_residual_dot, Active,
                Const(λ),
                Const(y_np1),
                Duplicated(y_n64, dy_n),
                Duplicated(Δm64,  dΔm),
                Duplicated(s64,   ds),
                Active(L64),
                Active(dt64),
            )
            # grads = (nothing, nothing, nothing, nothing, nothing, dL_box, ddt)
            dL_box = grads[6]
            ddt    = grads[7]
            # Output cotangent sign: q̇_np1 = -J^{-1} ∂F/∂p · ṗ in JVP,
            # so the pullback emits -(λ^T ∂F/∂p). Negate accumulated
            # arrays + scalars.
            return (-dy_n, -dΔm, -ds, -dL_box, -ddt)
        end
    end
    return y_np1, pullback
end

"""
    det_run_gradient(loss, y_0, Δm, s, L_box, dt, N; kwargs...)
        -> (loss_value, ȳ_0, Δm̄, s̄, L̄_box, d̄t)

End-to-end reverse-mode gradient of the scalar functional `loss(y_N)`
through an N-step Phase-2 deterministic trajectory.

`loss` is `Vector{Float64} -> Real`. Its gradient at `y_N` is built
with Enzyme reverse mode (one pass; the loss is treated as a generic
scalar function of the flat state). The trajectory is replayed
backwards via `det_step_pullback`, accumulating cotangents on
`(Δm, s, L_box, dt)`.

Periodic BC, no q-viscosity — same scope as `det_step_pullback`.
"""
function det_run_gradient(loss::F,
                          y_0::AbstractVector{Float64},
                          Δm::AbstractVector{Float64},
                          s::AbstractVector{Float64},
                          L_box::Real, dt::Real,
                          N::Integer;
                          abstol::Real = 1e-13,
                          reltol::Real = 1e-13,
                          maxiters::Int = 50) where {F}
    Δm64 = collect(Float64, Δm)
    s64  = collect(Float64, s)
    L64  = Float64(L_box)
    dt64 = Float64(dt)
    y_0_64 = collect(Float64, y_0)

    # Forward pass: store trajectory + per-step pullbacks.
    traj  = Vector{Vector{Float64}}(undef, N + 1)
    pulls = Vector{Function}(undef, N)
    traj[1] = y_0_64
    @inbounds for n in 1:N
        traj[n+1], pulls[n] = det_step_pullback(
            traj[n], Δm64, s64, L64, dt64;
            abstol = abstol, reltol = reltol, maxiters = maxiters,
        )
    end
    L = loss(traj[N+1])

    # Loss-gradient seed at y_N via Enzyme reverse. The loss is marked
    # `Const` to tolerate closures over (immutable but Enzyme-opaque)
    # captured arrays — see EnzymeMutabilityException FAQ.
    ȳ = zero(traj[N+1])
    Enzyme.autodiff(Reverse, Const(loss), Active, Duplicated(traj[N+1], ȳ))

    # Reverse-mode sweep through pullbacks; accumulate parameter
    # cotangents.
    Δm̄ = zero(Δm64); s̄ = zero(s64); L̄_box = 0.0; d̄t = 0.0
    @inbounds for n in N:-1:1
        ȳ_n, ΔΔm, Δs, ΔL, Δdt = pulls[n](ȳ)
        ȳ      = ȳ_n
        Δm̄    .+= ΔΔm
        s̄    .+= Δs
        L̄_box += ΔL
        d̄t    += Δdt
    end
    return L, ȳ, Δm̄, s̄, L̄_box, d̄t
end
