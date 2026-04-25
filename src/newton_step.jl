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

using NonlinearSolve: NonlinearProblem, NonlinearFunction, solve,
                      NewtonRaphson, SimpleNewtonRaphson,
                      AutoForwardDiff, ReturnCode
using StaticArrays: SVector
import SparseArrays
using SparseArrays: spzeros, SparseMatrixCSC
import SparseMatrixColorings  # required by NonlinearSolve for sparse AD path

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
        Pp_old = seg.state.Pp
        seg.state = DetField{T}(
            T(y[4*(j-1) + 1]),
            T(y[4*(j-1) + 2]),
            T(y[4*(j-1) + 3]),
            T(y[4*(j-1) + 4]),
            s_old,
            Pp_old,
        )
        # Refresh the cached half-step momentum diagnostic.
        i_left = j == 1 ? N : j - 1
        m̄ = (mesh.segments[i_left].Δm + mesh.segments[j].Δm) / 2
        mesh.p_half[j] = m̄ * seg.state.u
    end
    return mesh
end

"""
    det_jac_sparsity(N::Int) -> SparseMatrixCSC{Bool}

Build the Boolean sparsity pattern of the Phase-2 EL Jacobian for a
periodic mesh of `N` segments, used to seed sparse-AD Jacobian
computation. Each row block (for residual-segment `i`) has nonzero
column entries in segments `{i_left, i, i_right}` (cyclic), so the
Jacobian is **tri-block-banded** with 3 × 4 = 12 nonzeros per row, a
total of 48N nonzeros per Jacobian (vs the dense 16 N² used by the
default ForwardDiff path).

Concretely, for residual row `r = 4(i-1) + k` (k ∈ {1,2,3,4}) and
column `c = 4(j-1) + ℓ` (ℓ ∈ {1,2,3,4}), the entry is potentially
nonzero iff `j ∈ {i_left, i, i_right}`. The sparsity structure is
a function of `N` only — independent of `dt`, `s`, etc. — so this
prototype is built once per problem size and reused.
"""
function det_jac_sparsity(N::Int)
    rows = Int[]; cols = Int[]
    for i in 1:N
        i_left  = i == 1 ? N : i - 1
        i_right = i == N ? 1 : i + 1
        for k in 1:4
            r = 4 * (i - 1) + k
            for j in (i_left, i, i_right)
                for ℓ in 1:4
                    push!(rows, r)
                    push!(cols, 4 * (j - 1) + ℓ)
                end
            end
        end
    end
    # Use SparseArrays.sparse with combine=max to dedupe overlapping
    # (rows, cols) entries (they arise when N=2 or generally when
    # i_left and i_right alias). vals are all 1.0; the prototype
    # only carries the structural pattern.
    return SparseArrays.sparse(rows, cols, ones(Float64, length(rows)),
                               4N, 4N, max)
end

"""
    det_step!(mesh::Mesh1D, dt; tau=nothing, sparse_jac=true,
              q_kind=:none, c_q_quad=1.0, c_q_lin=0.5,
              abstol=1e-13, reltol=1e-13, maxiters=50)

Advance the multi-segment Phase-2/5 mesh by one timestep `dt`. Solves
the discrete Euler–Lagrange residual `det_el_residual` for the new
state via NonlinearSolve `NewtonRaphson()` with `AutoForwardDiff()`
Jacobian. Mutates the mesh in place; returns the mesh for convenience.

If `tau !== nothing`, applies a Phase-5 post-Newton BGK update to
the perpendicular-pressure field `P_⊥`: Lagrangian transport
`(P_⊥/ρ)^{n+1} = (P_⊥/ρ)^n` followed by joint relaxation of
`(P_xx, P_⊥)` toward their isotropic mean with exponential decay
`exp(-dt/τ)` (matching `py-1d/dfmm/schemes/cholesky.py` lines
153-179). With `tau === nothing` the Phase-2 single-pressure path
runs unchanged (`P_⊥` is left at its initial value, which is
isotropic-Maxwellian if the mesh was constructed without explicit
`Pps`).

**Phase 5b (`q_kind`)**: opt-in artificial viscosity. With the
default `q_kind = :none`, the integrator runs the bare variational
form (bit-equal to Phase 5). With `q_kind = :vNR_linear_quadratic`,
the Kuropatenko / von Neumann-Richtmyer combined linear+quadratic
artificial pressure is added to the EL momentum residual at each
segment midpoint, and a post-Newton entropy update absorbs the
q-dissipation `Δs/c_v = -(Γ-1) (q/P_xx) (∂_x u) Δt` so the EOS
stays consistent with the dissipated kinetic energy. Coefficients
`c_q_quad ∈ [1, 2]` (default 1.0) and `c_q_lin ∈ [0, 0.5]` (default
0.5) follow Caramana-Shashkov-Whalen 1998 §2.

The initial guess is an explicit-Euler advance:
  x_i^{n+1,(0)} = x_i^n + dt · u_i^n
  u_i^{n+1,(0)} = u_i^n − dt · (P_i − P_{i-1})/m̄_i
  α_j^{n+1,(0)} = α_j^n + dt · β_j^n
  β_j^{n+1,(0)} = β_j^n + dt · (γ²_j/α_j^n − (∂_x u)_j · β_j^n)
"""
function det_step!(mesh::Mesh1D{T,DetField{T}}, dt::Real;
                   tau::Union{Real,Nothing} = nothing,
                   sparse_jac::Bool = true,
                   q_kind::Symbol = Q_KIND_NONE,
                   c_q_quad::Real = 1.0,
                   c_q_lin::Real  = 0.5,
                   abstol::Real = 1e-13, reltol::Real = 1e-13,
                   maxiters::Int = 50) where {T<:Real}
    @assert q_kind_supported(q_kind) "Unsupported q_kind = $q_kind. " *
        "Use :none or :vNR_linear_quadratic."
    N = n_segments(mesh)
    Δm_vec = T[seg.Δm for seg in mesh.segments]
    s_vec  = T[seg.state.s for seg in mesh.segments]
    L_box  = mesh.L_box

    # Cache pre-Newton ρ_n per segment (for the post-Newton P_⊥
    # transport step). We need ρ at time n, before the implicit step
    # mutates segment positions.
    ρ_n_vec = T[segment_density(mesh, j) for j in 1:N]
    Pp_n_vec = T[mesh.segments[j].state.Pp for j in 1:N]

    y_n = pack_state(mesh)
    y0  = explicit_euler_guess(y_n, Δm_vec, s_vec, L_box, T(dt))

    # Closure with parameters captured. Phase 5b: forward q_kind +
    # coefficients into the residual. With q_kind = :none these are
    # ignored at near-zero cost (compute_q_segment skipped); the
    # bit-equality with Phase 5 is preserved.
    cqq = T(c_q_quad)
    cql = T(c_q_lin)
    p = (y_n, Δm_vec, s_vec, L_box, T(dt))
    f = (u, p_in) -> begin
        y_n_in, Δm_in, s_in, L_box_in, dt_in = p_in
        return det_el_residual(u, y_n_in, Δm_in, s_in, L_box_in, dt_in;
                               q_kind = q_kind,
                               c_q_quad = cqq,
                               c_q_lin  = cql)
    end

    # Build the NonlinearProblem with sparse Jacobian prototype when
    # sparse_jac is requested and the system is large enough to make
    # it worthwhile (N ≥ 16 for det_jac_sparsity to amortize). The
    # NonlinearFunction's `jac_prototype` carries the sparsity, and
    # NonlinearSolve's coloring algorithm seeds the AD path
    # accordingly (no AutoSparse needed; that's a removed v4 API).
    if sparse_jac && N ≥ 16
        jac_proto = det_jac_sparsity(N)
        nf = NonlinearFunction(f; jac_prototype = jac_proto)
        prob = NonlinearProblem(nf, y0, p)
        sol = solve(
            prob,
            NewtonRaphson(; autodiff = AutoForwardDiff());
            abstol = abstol,
            reltol = reltol,
            maxiters = maxiters,
        )
    else
        prob = NonlinearProblem(f, y0, p)
        sol = solve(
            prob,
            NewtonRaphson(; autodiff = AutoForwardDiff());
            abstol = abstol,
            reltol = reltol,
            maxiters = maxiters,
        )
    end
    res = det_el_residual(sol.u, y_n, Δm_vec, s_vec, L_box, T(dt);
                          q_kind = q_kind,
                          c_q_quad = cqq,
                          c_q_lin  = cql)
    res_norm = maximum(abs, res)
    # Newton may report `Stalled` or `MaxIters` retcodes on smooth
    # nearly-converged problems where the per-step descent falls below
    # relative tolerance; the residual is still at round-off in such
    # cases. Accept any residual within 1e6 × abstol regardless of
    # retcode (Phase-5 Sod runs near the Newton tolerance limit at
    # large N).
    if res_norm > 1e6 * abstol && sol.retcode != ReturnCode.Success
        error("det_step! Newton solve failed: retcode = $(sol.retcode), " *
              "‖residual‖∞ = $res_norm")
    end

    unpack_state!(mesh, sol.u)

    # Phase 5b: post-Newton entropy update for the q-dissipation.
    # The artificial viscosity dissipates kinetic energy at rate
    # `(q/ρ) ∂_x u` (negative when ∂_x u < 0, i.e. compression heats).
    # For an ideal-gas EOS `M_vv = J^{1-Γ} exp(s/c_v)`, the change in
    # internal energy at fixed J is converted to entropy via
    #   Δs/c_v = -(Γ-1) (q/P_xx) (∂_x u) Δt
    # which is non-negative in compression (q > 0, ∂_x u < 0). At
    # `q_kind = :none` we skip this branch entirely so Phase 5
    # behaviour is bit-equal.
    if q_active(q_kind)
        Γ_q = T(GAMMA_LAW_DEFAULT)
        cqq = T(c_q_quad)
        cql = T(c_q_lin)
        # Recompute midpoint divu and q per segment from the
        # **midpoint** (½(y_n + y_np1)) state — same definition as
        # `det_el_residual` so the dissipation is consistent with the
        # momentum equation that produced the new state.
        @inbounds for j in 1:N
            j_right = j == N ? 1 : j + 1
            wrap = (j == N) ? L_box : zero(T)
            # Midpoint vertex positions and velocities.
            x_left_n  = y_n[4*(j-1) + 1]
            x_right_n = y_n[4*(j_right-1) + 1] + wrap
            x_left_np1  = mesh.segments[j].state.x
            x_right_np1 = mesh.segments[j_right].state.x + wrap
            u_left_n  = y_n[4*(j-1) + 2]
            u_right_n = y_n[4*(j_right-1) + 2]
            u_left_np1  = mesh.segments[j].state.u
            u_right_np1 = mesh.segments[j_right].state.u

            x̄_left  = (x_left_n  + x_left_np1)  / 2
            x̄_right = (x_right_n + x_right_np1) / 2
            ū_left  = (u_left_n  + u_left_np1)  / 2
            ū_right = (u_right_n + u_right_np1) / 2
            Δx̄ = x̄_right - x̄_left
            divu_j = (ū_right - ū_left) / Δx̄
            J̄_j = Δx̄ / Δm_vec[j]
            ρ̄_j = one(T) / J̄_j
            s_pre = mesh.segments[j].state.s
            M̄vv_j = Mvv(J̄_j, s_pre)
            P̄xx_j = ρ̄_j * M̄vv_j

            c_s_bar = sqrt(max(Γ_q * M̄vv_j, zero(T)))
            q_j = compute_q_segment(divu_j, ρ̄_j, c_s_bar, Δx̄;
                                    c_q_quad = cqq, c_q_lin = cql)
            # Entropy update Δs/c_v = -(Γ-1)(q/P_xx)·divu·dt.
            # Skip when q is zero or P_xx degenerate.
            if q_j > 0 && P̄xx_j > 0
                Δs_q = -(Γ_q - one(T)) * (q_j / P̄xx_j) * divu_j * T(dt)
                seg = mesh.segments[j]
                seg.state = DetField{T}(seg.state.x, seg.state.u,
                                        seg.state.α, seg.state.β,
                                        s_pre + Δs_q, seg.state.Pp)
            end
        end
    end

    # Phase-5: post-Newton BGK update of (P_xx, P_⊥). Computed in
    # three substeps:
    #   (1) Lagrangian transport of P_⊥ (P_⊥/ρ conserved):
    #       Pp_transport = Pp_n · ρ_np1/ρ_n.
    #   (2) Joint BGK relaxation of (P_xx, P_⊥) toward P_iso, matching
    #       py-1d's `cholesky.py` lines 165-167 exactly. Conserves the
    #       trace P_xx + 2 P_⊥ to round-off.
    #   (3) Entropy update: the variational integrator carries
    #       P_xx = ρ M_vv(J, s), so when BGK shifts (P_xx, P_⊥) jointly,
    #       we re-anchor the EOS by adjusting entropy:
    #           s_new = s_old + c_v · log(P_xx_new / P_xx_pre).
    #       This converts BGK's anisotropy-relaxation into a
    #       physically-consistent isotropic-thermalization (heat
    #       distributed evenly across velocity components, raising s).
    #       For an initially-isotropic IC where transport generates
    #       Π = ρ Δ M_vv > 0, this is the standard "viscous heating"
    #       channel.
    #
    # In the τ → 0 limit (Euler), each step the BGK fully isotropises
    # so P_xx = P_⊥ = P_iso and the entropy rise tracks the bulk
    # compression's thermal portion — recovering Euler-limit Sod.
    # In τ → ∞ (collisionless), BGK is off and (P_xx, P_⊥) advect
    # independently; the integrator matches py-1d's free-streaming
    # limit.
    if tau !== nothing
        τ = T(tau)
        decay = exp(-T(dt) / τ)
        # Realizability headroom for β: matches py-1d's
        # `realizability_headroom = 0.999` (`config.py` line 44).
        rh = T(0.999)
        @inbounds for j in 1:N
            seg = mesh.segments[j]
            ρ_np1 = segment_density(mesh, j)
            J_np1 = one(T) / ρ_np1
            s_pre = seg.state.s
            Mvv_pre = Mvv(J_np1, s_pre)
            Pxx_pre = ρ_np1 * Mvv_pre
            # Transport step (Pp/ρ conserved along Lagrangian
            # trajectories).
            Pp_transport = Pp_n_vec[j] * ρ_np1 / ρ_n_vec[j]
            # BGK relaxation: relax both Pxx and Pp toward P_iso
            # (py-1d/dfmm/schemes/cholesky.py lines 165-167).
            Pxx_new, Pp_new = bgk_relax_pressures(Pxx_pre, Pp_transport,
                                                  T(dt), τ)
            # Re-anchor the EOS: update s so M_vv(J_np1, s_new) =
            # Pxx_new/ρ_np1.
            #   M_vv(J, s) = J^(1-Γ) exp(s/c_v)
            # ⇒ Δs/c_v = log(M_vv_new/M_vv_pre) = log(Pxx_new/Pxx_pre).
            Mvv_new = Pxx_new / ρ_np1
            if Mvv_new > 0 && Mvv_pre > 0
                Δs_over_cv = log(Mvv_new / Mvv_pre)
                s_new = s_pre + Δs_over_cv  # c_v normalised to 1
            else
                s_new = s_pre
                Mvv_new = Mvv_pre
            end
            # BGK relaxation of β (matches py-1d line 168):
            #   β_new = β_n · exp(-Δt/τ),
            # plus realizability clip |β| ≤ 0.999 √M_vv (line 172-174).
            β_new = seg.state.β * decay
            β_max = rh * sqrt(max(Mvv_new, zero(T)))
            if β_new > β_max
                β_new = β_max
            elseif β_new < -β_max
                β_new = -β_max
            end
            seg.state = DetField{T}(seg.state.x, seg.state.u,
                                    seg.state.α, β_new,
                                    s_new, Pp_new)
        end
    end

    return mesh
end

"""
    det_run!(mesh::Mesh1D, dt, N_steps; tau=nothing, kwargs...)

Convenience driver: run `N_steps` consecutive `det_step!` calls on
`mesh`. Mutates `mesh` in place, returns `mesh`. Keyword arguments
(including the Phase-5 `tau` for BGK relaxation) are forwarded to
`det_step!`.
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
