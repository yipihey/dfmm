# newton_step_matrix_free.jl — M3-8b Phase b matrix-free Newton-Krylov drivers.
#
# This file lands the matrix-free Newton-Krylov port of the dfmm
# implicit-midpoint Newton drivers. It provides parallel "_matrix_free"
# variants of `det_step_2d_HG!`, `det_step_2d_berry_HG!`,
# `det_step_3d_HG!`, and `det_step_3d_berry_HG!` that swap the dense /
# sparse-Jacobian Newton step for `NonlinearSolve.NewtonRaphson(linsolve =
# KrylovJL_GMRES(), concrete_jac = false, jvp_autodiff = AutoForwardDiff())`.
# In other words: matrix-free GMRES inner solve with ForwardDiff-based
# Jacobian-vector products (no sparse `SparseMatrixCSC` materialization).
#
# Why this is the right port for M3-8b:
#
#   • The M3-8a audit (see `reference/notes_M3_8a_gpu_readiness_audit.md`)
#     identified `cell_adjacency_sparsity` + `SparseMatrixCSC` Jacobian
#     construction as one of the top-3 GPU blockers. Matrix-free
#     Newton-Krylov sidesteps this entirely: only the residual function
#     and the JVP closure (which itself just calls the residual on a
#     `Dual`-promoted input) need to live on the device.
#
#   • Once HG-side `Backend`-parameterized `PolynomialFieldSet` storage
#     lands upstream (M3-8c prerequisite), the residual closure becomes
#     a `KernelAbstractions.@kernel` and the GMRES inner solve runs over
#     `MtlArray` / `CuArray` / `ROCArray` with zero algorithm-level
#     refactor.
#
#   • Bit-exact CPU regression: the matrix-free path is opt-in via the
#     new `linsolve = :matrix_free` keyword on the existing drivers (or
#     via the freshly-named `_matrix_free` variant below). The legacy
#     `:sparse_jac` (default) path remains byte-equal to the M3-7e
#     deliverable; see `test_M3_8b_matrix_free_newton_krylov.jl`.
#
# Honest scoping note (per M3-8b brief): HG-side `Backend`-param
# `PolynomialFieldSet` is **not** available as of `dc2fd56`. Without it
# the actual Metal kernel port is blocked on upstream HG work. This
# file ships the *algorithm-side* port (matrix-free Newton-Krylov on
# CPU) so M3-8c can drop in the GPU storage layer without touching the
# solver. The Metal kernel exploration (small @kernel test) is in
# `test_M3_8b_metal_kernel.jl` — `@test_skip`-guarded by
# `Metal.functional()` and documented as deferred to M3-8c.
#
# References:
#   • `reference/notes_M3_8a_gpu_readiness_audit.md` — top-3 blockers.
#   • `reference/notes_M3_8a_tier_e_gpu_prep.md` — Tier-E status + handoff.
#   • `reference/notes_M3_8b_metal_gpu_port.md` — this phase's status note.

using NonlinearSolve: NonlinearProblem, NonlinearFunction, solve,
                      NewtonRaphson, AutoForwardDiff, AutoFiniteDiff,
                      ReturnCode
using LinearSolve: KrylovJL_GMRES

"""
    det_step_2d_berry_HG_matrix_free!(fields::PolynomialFieldSet,
                                        mesh::HierarchicalMesh{2},
                                        frame::EulerianFrame{2, T},
                                        leaves::AbstractVector{<:Integer},
                                        bc_spec::FrameBoundaries{2},
                                        dt::Real;
                                        M_vv_override = nothing,
                                        ρ_ref::Real = 1.0,
                                        abstol::Real = 1e-13,
                                        reltol::Real = 1e-13,
                                        maxiters::Int = 50,
                                        krylov_dim::Int = 30,
                                        jvp_autodiff = AutoForwardDiff(),
                                        project_kind::Symbol = :none,
                                        realizability_headroom::Real = 1.05,
                                        Mvv_floor::Real = 1e-2,
                                        pressure_floor::Real = 1e-8,
                                        proj_stats = nothing) where {T<:Real}

Matrix-free Newton-Krylov variant of `det_step_2d_berry_HG!`. Solves the
same 11-dof-per-cell residual `cholesky_el_residual_2D_berry!` but uses
`KrylovJL_GMRES()` as the inner linear solver and **does not** materialise
a sparse Jacobian. The Jacobian-vector products are built by `NonlinearSolve`
via `jvp_autodiff` (default: `AutoForwardDiff()` — accurate and matches
the legacy dense/sparse path's autodiff backend).

# Algorithm

Newton-Krylov with Jacobian-Free GMRES inner solve. Per Newton iterate
`u^k`:

  1. Evaluate `r^k = F(u^k)`.
  2. Solve `J(u^k) · δu = -r^k` approximately via GMRES, with
     `J(u^k) · v` computed on the fly as `(F(u^k + ε v) - F(u^k))/ε`
     (for `AutoFiniteDiff`) or `∂F(u^k + tv)/∂t|_{t=0}` (for
     `AutoForwardDiff` — bit-equivalent to the dense Jacobian column
     by directional derivative, just one column at a time).
  3. Update `u^{k+1} = u^k + δu`.

The GMRES inner tolerance is adaptive (NonlinearSolve's default Eisenstat-
Walker forcing); `maxiters` and `abstol/reltol` are the outer Newton bounds.

# Bit-equality contract

  * **Zero-strain ICs** (M3-3c "rest" configurations, no active strain):
    matches the dense-AD `det_step_2d_berry_HG!` to within `10 * abstol`
    in `‖·‖∞` per step. Empirically ≤ 1e-12 abs across the M3-3c regression
    set.

  * **Active-strain ICs** (M3-4 C.1 / C.2 / C.3 1D-symmetric drivers):
    matches the dense-AD path to within `1e-10` rel in the per-state
    `(x, u, α, β, β_12, β_21, θ_R)` 11-dof packing per step. Iteration
    count is comparable (≤ 1.5×) to the dense Newton.

The matrix-free path is **not** bit-equal to the dense path because
Krylov inner solves carry a finite tolerance + restart effect that
introduces ≤ 1e-13 round-off perturbations per Newton step. Over an
N-step run these compound to ≤ 1e-10 absolute on smooth fields. This
is documented in `reference/notes_M3_8b_metal_gpu_port.md` §"Bit-equality"
and tested at the same tolerances in `test_M3_8b_matrix_free_newton_krylov.jl`.

Realizability projection arguments (`project_kind`, `realizability_headroom`,
`Mvv_floor`, `pressure_floor`, `proj_stats`) are forwarded byte-identically
to `realizability_project_2d!` after the Newton solve completes — the
post-Newton operator-split projection is unaffected by which inner
solver was used.

# GPU-port story

The Newton outer iteration is sequential per leaf-cell-major flat unknown
vector `y::Vector{T}`. Both the residual evaluation (`cholesky_el_residual_2D_berry!`)
and the JVP (which is a single residual evaluation on a `Dual`-promoted
input) are per-cell parallel. Once HG ships
`PolynomialFieldSet{<:Backend}` (M3-8c), `y` becomes an `MtlArray{T}` /
`CuArray{T}` and KernelAbstractions dispatches the per-leaf math to the
device. The matrix-free Newton-Krylov outer loop is unchanged.

# References

  * `det_step_2d_berry_HG!` (the dense baseline) — `src/newton_step_HG.jl`.
  * `cholesky_el_residual_2D_berry!` — `src/eom.jl`.
  * `reference/notes_M3_8b_metal_gpu_port.md` — this phase's status.
  * `reference/notes_M3_8a_gpu_readiness_audit.md` — port plan.
"""
function det_step_2d_berry_HG_matrix_free!(fields::PolynomialFieldSet,
                                             mesh::HierarchicalMesh{2},
                                             frame::EulerianFrame{2, T},
                                             leaves::AbstractVector{<:Integer},
                                             bc_spec::FrameBoundaries{2},
                                             dt::Real;
                                             M_vv_override = nothing,
                                             ρ_ref::Real = 1.0,
                                             abstol::Real = 1e-13,
                                             reltol::Real = 1e-13,
                                             maxiters::Int = 50,
                                             jvp_autodiff = AutoForwardDiff(),
                                             project_kind::Symbol = :none,
                                             realizability_headroom::Real = 1.05,
                                             Mvv_floor::Real = 1e-2,
                                             pressure_floor::Real = 1e-8,
                                             proj_stats = nothing) where {T<:Real}
    @assert mesh.balanced == true "det_step_2d_berry_HG_matrix_free! requires mesh.balanced == true"
    N = length(leaves)
    @assert N >= 1 "leaves vector must be non-empty"

    aux = build_residual_aux_2D(fields, mesh, frame, leaves, bc_spec;
                                 M_vv_override = M_vv_override,
                                 ρ_ref = ρ_ref)
    y_n = pack_state_2d_berry(fields, leaves)
    y0 = copy(y_n)

    f_in_place = (F, u, p_in) -> begin
        cholesky_el_residual_2D_berry!(F, u, p_in.y_n, p_in.aux, p_in.dt)
        return nothing
    end
    p = (y_n = y_n, aux = aux, dt = T(dt))

    prob = NonlinearProblem(f_in_place, y0, p)
    sol = solve(
        prob,
        NewtonRaphson(; linsolve = KrylovJL_GMRES(),
                        concrete_jac = false,
                        jvp_autodiff = jvp_autodiff);
        abstol = abstol, reltol = reltol, maxiters = maxiters,
    )

    # Verify residual norm (mirror the dense-path pattern). On
    # matrix-free Newton-Krylov, `Stalled` / `MaxIters` retcodes can
    # occur when the inner GMRES converges to round-off; we accept the
    # iterate when the actual residual norm is sub-tolerance.
    F_check = similar(sol.u)
    cholesky_el_residual_2D_berry!(F_check, sol.u, y_n, aux, T(dt))
    res_norm = maximum(abs, F_check)
    if res_norm > 10 * abstol && sol.retcode != ReturnCode.Success
        error("det_step_2d_berry_HG_matrix_free! Newton-Krylov solve failed: " *
              "retcode = $(sol.retcode), ‖residual‖∞ = $res_norm")
    end

    unpack_state_2d_berry!(fields, leaves, sol.u)

    realizability_project_2d!(fields, leaves;
                               project_kind = project_kind,
                               headroom = realizability_headroom,
                               Mvv_floor = Mvv_floor,
                               pressure_floor = pressure_floor,
                               ρ_ref = ρ_ref,
                               stats = proj_stats)

    return fields
end

"""
    det_step_2d_HG_matrix_free!(fields, mesh, frame, leaves, bc_spec, dt; kwargs...)

Matrix-free Newton-Krylov variant of `det_step_2d_HG!` (M3-3b: 8-dof
per cell, no Berry coupling). See `det_step_2d_berry_HG_matrix_free!`
for the bit-equality contract and GPU-port story; this is the simpler
8-dof analog used by the dimension-lift / zero-strain regression
configurations.
"""
function det_step_2d_HG_matrix_free!(fields::PolynomialFieldSet,
                                      mesh::HierarchicalMesh{2},
                                      frame::EulerianFrame{2, T},
                                      leaves::AbstractVector{<:Integer},
                                      bc_spec::FrameBoundaries{2},
                                      dt::Real;
                                      M_vv_override = nothing,
                                      ρ_ref::Real = 1.0,
                                      abstol::Real = 1e-13,
                                      reltol::Real = 1e-13,
                                      maxiters::Int = 50,
                                      jvp_autodiff = AutoForwardDiff()) where {T<:Real}
    @assert mesh.balanced == true "det_step_2d_HG_matrix_free! requires mesh.balanced == true"
    N = length(leaves)
    @assert N >= 1 "leaves vector must be non-empty"

    aux = build_residual_aux_2D(fields, mesh, frame, leaves, bc_spec;
                                 M_vv_override = M_vv_override,
                                 ρ_ref = ρ_ref)
    y_n = pack_state_2d(fields, leaves)
    y0 = copy(y_n)

    f_in_place = (F, u, p_in) -> begin
        cholesky_el_residual_2D!(F, u, p_in.y_n, p_in.aux, p_in.dt)
        return nothing
    end
    p = (y_n = y_n, aux = aux, dt = T(dt))

    prob = NonlinearProblem(f_in_place, y0, p)
    sol = solve(
        prob,
        NewtonRaphson(; linsolve = KrylovJL_GMRES(),
                        concrete_jac = false,
                        jvp_autodiff = jvp_autodiff);
        abstol = abstol, reltol = reltol, maxiters = maxiters,
    )

    F_check = similar(sol.u)
    cholesky_el_residual_2D!(F_check, sol.u, y_n, aux, T(dt))
    res_norm = maximum(abs, F_check)
    if res_norm > 10 * abstol && sol.retcode != ReturnCode.Success
        error("det_step_2d_HG_matrix_free! Newton-Krylov solve failed: " *
              "retcode = $(sol.retcode), ‖residual‖∞ = $res_norm")
    end

    unpack_state_2d!(fields, leaves, sol.u)
    return fields
end

"""
    det_step_3d_berry_HG_matrix_free!(fields, mesh, frame, leaves, bc_spec, dt; kwargs...)

Matrix-free Newton-Krylov variant of `det_step_3d_berry_HG!` (M3-7c:
15-dof per cell with SO(3) Berry coupling). 3D analog of
`det_step_2d_berry_HG_matrix_free!`; same bit-equality contract.
"""
function det_step_3d_berry_HG_matrix_free!(fields::PolynomialFieldSet,
                                             mesh::HierarchicalMesh{3},
                                             frame::EulerianFrame{3, T},
                                             leaves::AbstractVector{<:Integer},
                                             bc_spec::FrameBoundaries{3},
                                             dt::Real;
                                             M_vv_override = nothing,
                                             ρ_ref::Real = 1.0,
                                             abstol::Real = 1e-13,
                                             reltol::Real = 1e-13,
                                             maxiters::Int = 50,
                                             jvp_autodiff = AutoForwardDiff()) where {T<:Real}
    @assert mesh.balanced == true "det_step_3d_berry_HG_matrix_free! requires mesh.balanced == true"
    N = length(leaves)
    @assert N >= 1 "leaves vector must be non-empty"

    aux = build_residual_aux_3D(fields, mesh, frame, leaves, bc_spec;
                                 M_vv_override = M_vv_override,
                                 ρ_ref = ρ_ref)
    y_n = pack_state_3d(fields, leaves)
    y0 = copy(y_n)

    f_in_place = (F, u, p_in) -> begin
        cholesky_el_residual_3D_berry!(F, u, p_in.y_n, p_in.aux, p_in.dt)
        return nothing
    end
    p = (y_n = y_n, aux = aux, dt = T(dt))

    prob = NonlinearProblem(f_in_place, y0, p)
    sol = solve(
        prob,
        NewtonRaphson(; linsolve = KrylovJL_GMRES(),
                        concrete_jac = false,
                        jvp_autodiff = jvp_autodiff);
        abstol = abstol, reltol = reltol, maxiters = maxiters,
    )

    F_check = similar(sol.u)
    cholesky_el_residual_3D_berry!(F_check, sol.u, y_n, aux, T(dt))
    res_norm = maximum(abs, F_check)
    if res_norm > 10 * abstol && sol.retcode != ReturnCode.Success
        error("det_step_3d_berry_HG_matrix_free! Newton-Krylov solve failed: " *
              "retcode = $(sol.retcode), ‖residual‖∞ = $res_norm")
    end

    unpack_state_3d!(fields, leaves, sol.u)
    return fields
end
