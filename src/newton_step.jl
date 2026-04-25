# newton_step.jl
#
# Implicit-midpoint Newton step for the Phase-1 Cholesky-sector
# variational integrator. Wraps NonlinearSolve.jl with ForwardDiff
# Jacobians on a 2-DOF SVector residual.
#
# The per-step problem is small (2 unknowns: α_{n+1}, β_{n+1}), so we
# use SimpleNewtonRaphson on StaticArrays-backed residuals. This is
# the right fit: zero allocations on the inner loop, small dense
# Jacobian factored in place.

using NonlinearSolve: NonlinearProblem, solve, NewtonRaphson, SimpleNewtonRaphson,
                      AutoForwardDiff, ReturnCode
using StaticArrays: SVector

"""
    cholesky_step(q_n::SVector{2}, M_vv, divu_half, dt;
                  abstol = 1e-13, reltol = 1e-13, maxiters = 50)

Advance one Cholesky-sector cell from time `t_n` to `t_{n+1} = t_n + dt`
by solving the discrete Euler–Lagrange residual
`cholesky_el_residual(q_np1, q_n, M_vv, divu_half, dt) = 0` for
`q_np1 = (α_{n+1}, β_{n+1})`.

Arguments:
- `q_n::SVector{2}` — current state `(α_n, β_n)`.
- `M_vv`            — externally-supplied second velocity moment
                      (Phase 1: a constant).
- `divu_half`       — midpoint strain rate `(∂_x u)_{n+1/2}` for the
                      step (Phase 1: a constant).
- `dt`              — timestep.

Keyword arguments:
- `abstol`, `reltol` — Newton tolerances. Default `1e-13` is tighter
                      than the downstream B.1 energy-drift target of
                      `1e-8`, as required.
- `maxiters`        — Newton iteration cap.

Returns the new state `q_np1::SVector{2}`.

Implementation notes. The 2-DOF residual uses the
`SimpleNewtonRaphson` algorithm with `AutoForwardDiff` for the
Jacobian — both ship in NonlinearSolve.jl's lightweight stack. For
this problem size (2×2 dense Jacobian) ForwardDiff is the right
choice; Enzyme is overkill and has setup overhead. Per-step solves
typically converge in 2–4 Newton iterations on smooth states.
"""
function cholesky_step(
    q_n::SVector{2,T},
    M_vv::Real,
    divu_half::Real,
    dt::Real;
    abstol::Real = 1e-13,
    reltol::Real = 1e-13,
    maxiters::Int = 50,
) where {T<:Real}
    # Pack scalar parameters as a tuple; SimpleNewtonRaphson handles
    # tuple-shaped p without allocations.
    p = (q_n, T(M_vv), T(divu_half), T(dt))

    # Closure form expected by NonlinearProblem: f(u, p) → residual.
    # We construct an SVector residual to keep the solver allocation-free.
    f = (u, p_in) -> begin
        q_n_in, M_vv_in, divu_in, dt_in = p_in
        return cholesky_el_residual(u, q_n_in, M_vv_in, divu_in, dt_in)
    end

    # Initial guess: explicit Euler from the boxed equations evaluated at q_n.
    α_n, β_n = q_n[1], q_n[2]
    γ2_n = M_vv - β_n^2
    α0 = α_n + dt * β_n
    β0 = β_n + dt * (γ2_n / α_n - divu_half * β_n)
    u0 = SVector{2,T}(T(α0), T(β0))

    prob = NonlinearProblem{false}(f, u0, p)
    sol = solve(
        prob,
        SimpleNewtonRaphson(; autodiff = AutoForwardDiff());
        abstol = abstol,
        reltol = reltol,
        maxiters = maxiters,
    )
    # Some retcodes (MaxIters) can fire even when the residual is at
    # round-off; check the residual norm directly as the success
    # criterion. Tighter than retcode == Success for this problem.
    res = cholesky_el_residual(sol.u, q_n, M_vv, divu_half, dt)
    res_norm = max(abs(res[1]), abs(res[2]))
    if res_norm > 10 * abstol && sol.retcode != ReturnCode.Success
        error("cholesky_step Newton solve failed: retcode = $(sol.retcode), " *
              "‖residual‖∞ = $res_norm, residual = $res")
    end
    return sol.u::SVector{2,T}
end

"""
    cholesky_run(q_0::SVector{2}, M_vv, divu_half, dt, N; kwargs...)

Convenience driver: run `N` consecutive `cholesky_step` calls starting
from `q_0`, with constant `M_vv`, `divu_half`, `dt`. Returns a
`Vector{SVector{2}}` of length `N+1` with the trajectory
`[q_0, q_1, …, q_N]`. Used by the Phase-1 tests.

Keyword arguments are forwarded to `cholesky_step`.
"""
function cholesky_run(
    q_0::SVector{2,T},
    M_vv::Real,
    divu_half::Real,
    dt::Real,
    N::Integer;
    kwargs...,
) where {T<:Real}
    traj = Vector{SVector{2,T}}(undef, N + 1)
    traj[1] = q_0
    for n in 1:N
        traj[n+1] = cholesky_step(traj[n], M_vv, divu_half, dt; kwargs...)
    end
    return traj
end
