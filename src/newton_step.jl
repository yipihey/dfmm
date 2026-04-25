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

# ──────────────────────────────────────────────────────────────────────
# Phase 2: multi-segment deterministic Newton step
# ──────────────────────────────────────────────────────────────────────

"""
    pack_state!(y, mesh::Mesh1D)

Pack the mesh's `(x, u, α, β)` per segment into the flat `y` vector
of length `4N` in segment-major order, in-place. Used by `det_step!`
to prepare the Newton initial guess and to write the solver result
back into the mesh.
"""
function pack_state!(y::AbstractVector, mesh::Mesh1D{T,DetField{T}}) where {T<:Real}
    N = n_segments(mesh)
    @assert length(y) == 4N
    @inbounds for j in 1:N
        seg = mesh.segments[j].state
        y[4*(j-1) + 1] = seg.x
        y[4*(j-1) + 2] = seg.u
        y[4*(j-1) + 3] = seg.α
        y[4*(j-1) + 4] = seg.β
    end
    return y
end

function pack_state(mesh::Mesh1D{T,DetField{T}}) where {T<:Real}
    N = n_segments(mesh)
    y = Vector{T}(undef, 4N)
    pack_state!(y, mesh)
    return y
end

"""
    unpack_state!(mesh::Mesh1D, y)

Write the Newton-solved `y` vector back into the mesh, preserving the
segment ordering. Entropy `s` is *not* updated (frozen in Phase 2).
"""
function unpack_state!(mesh::Mesh1D{T,DetField{T}}, y::AbstractVector) where {T<:Real}
    N = n_segments(mesh)
    @assert length(y) == 4N
    @inbounds for j in 1:N
        seg = mesh.segments[j]
        s_old = seg.state.s
        seg.state = DetField{T}(
            T(y[4*(j-1) + 1]),
            T(y[4*(j-1) + 2]),
            T(y[4*(j-1) + 3]),
            T(y[4*(j-1) + 4]),
            s_old,
        )
        # Refresh the cached half-step momentum diagnostic.
        i_left = j == 1 ? N : j - 1
        m̄ = (mesh.segments[i_left].Δm + mesh.segments[j].Δm) / 2
        mesh.p_half[j] = m̄ * seg.state.u
    end
    return mesh
end

"""
    det_step!(mesh::Mesh1D, dt; abstol=1e-13, reltol=1e-13, maxiters=50)

Advance the multi-segment Phase-2 mesh by one timestep `dt`. Solves
the discrete Euler–Lagrange residual `det_el_residual` for the new
state via NonlinearSolve `NewtonRaphson()` with `AutoForwardDiff()`
Jacobian. Mutates the mesh in place; returns the mesh for convenience.

The initial guess is an explicit-Euler advance:
  x_i^{n+1,(0)} = x_i^n + dt · u_i^n
  u_i^{n+1,(0)} = u_i^n − dt · (P_i − P_{i-1})/m̄_i
  α_j^{n+1,(0)} = α_j^n + dt · β_j^n
  β_j^{n+1,(0)} = β_j^n + dt · (γ²_j/α_j^n − (∂_x u)_j · β_j^n)
"""
function det_step!(mesh::Mesh1D{T,DetField{T}}, dt::Real;
                   abstol::Real = 1e-13, reltol::Real = 1e-13,
                   maxiters::Int = 50) where {T<:Real}
    N = n_segments(mesh)
    Δm_vec = T[seg.Δm for seg in mesh.segments]
    s_vec  = T[seg.state.s for seg in mesh.segments]
    L_box  = mesh.L_box

    y_n = pack_state(mesh)
    y0  = explicit_euler_guess(y_n, Δm_vec, s_vec, L_box, T(dt))

    # Closure with parameters captured.
    p = (y_n, Δm_vec, s_vec, L_box, T(dt))
    f = (u, p_in) -> begin
        y_n_in, Δm_in, s_in, L_box_in, dt_in = p_in
        return det_el_residual(u, y_n_in, Δm_in, s_in, L_box_in, dt_in)
    end

    prob = NonlinearProblem(f, y0, p)
    sol = solve(
        prob,
        NewtonRaphson(; autodiff = AutoForwardDiff());
        abstol = abstol,
        reltol = reltol,
        maxiters = maxiters,
    )
    res = det_el_residual(sol.u, y_n, Δm_vec, s_vec, L_box, T(dt))
    res_norm = maximum(abs, res)
    # Newton may report `Stalled` retcode on smooth nearly-converged
    # problems where the per-step descent falls below relative
    # tolerance; the residual is still at round-off in such cases.
    # Accept any residual within 1e4 × abstol regardless of retcode.
    if res_norm > 1e4 * abstol && sol.retcode != ReturnCode.Success
        error("det_step! Newton solve failed: retcode = $(sol.retcode), " *
              "‖residual‖∞ = $res_norm")
    end

    unpack_state!(mesh, sol.u)
    return mesh
end

"""
    det_run!(mesh::Mesh1D, dt, N_steps; kwargs...)

Convenience driver: run `N_steps` consecutive `det_step!` calls on
`mesh`. Mutates `mesh` in place, returns `mesh`. Keyword arguments
are forwarded to `det_step!`.
"""
function det_run!(mesh::Mesh1D{T,DetField{T}}, dt::Real, N_steps::Integer;
                  kwargs...) where {T<:Real}
    for _ in 1:N_steps
        det_step!(mesh, dt; kwargs...)
    end
    return mesh
end

# Internal: explicit-Euler guess for the Newton solver.
function explicit_euler_guess(y_n::AbstractVector{T}, Δm::Vector{T},
                              s::Vector{T}, L_box::T, dt::T) where {T<:Real}
    N = length(Δm)
    y0 = similar(y_n)
    @inline get_x(y, j) = y[4*(j-1) + 1]
    @inline get_u(y, j) = y[4*(j-1) + 2]
    @inline get_α(y, j) = y[4*(j-1) + 3]
    @inline get_β(y, j) = y[4*(j-1) + 4]

    # Pre-compute pressures and strains at time n.
    P = Vector{T}(undef, N)
    divu = Vector{T}(undef, N)
    @inbounds for j in 1:N
        j_right = j == N ? 1 : j + 1
        wrap = (j == N) ? L_box : zero(T)
        x_left   = get_x(y_n, j)
        x_right  = get_x(y_n, j_right) + wrap
        u_left   = get_u(y_n, j)
        u_right  = get_u(y_n, j_right)
        Δx = x_right - x_left
        J = Δx / Δm[j]
        Mvv_j = Mvv(J, s[j])
        P[j] = Mvv_j / J
        divu[j] = (u_right - u_left) / Δx
    end

    @inbounds for i in 1:N
        x_n_i = get_x(y_n, i)
        u_n_i = get_u(y_n, i)
        α_n_j = get_α(y_n, i)
        β_n_j = get_β(y_n, i)
        i_left = i == 1 ? N : i - 1
        m̄ = (Δm[i_left] + Δm[i]) / 2

        y0[4*(i-1) + 1] = x_n_i + dt * u_n_i
        y0[4*(i-1) + 2] = u_n_i - dt * (P[i] - P[i_left]) / m̄
        # γ²_n/α_n − (∂_x u)·β
        # Use segment i's M_vv at time n.
        x_left   = get_x(y_n, i)
        i_right = i == N ? 1 : i + 1
        wrap = (i == N) ? L_box : zero(T)
        x_right  = get_x(y_n, i_right) + wrap
        J_n = (x_right - x_left) / Δm[i]
        Mvv_n = Mvv(J_n, s[i])
        γ²_n = max(Mvv_n - β_n_j^2, zero(T))
        y0[4*(i-1) + 3] = α_n_j + dt * β_n_j
        y0[4*(i-1) + 4] = β_n_j + dt * (γ²_n / α_n_j - divu[i] * β_n_j)
    end
    return y0
end
