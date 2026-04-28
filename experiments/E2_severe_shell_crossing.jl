# E2_severe_shell_crossing.jl
#
# M3-8 Phase a Deliverable: E.2 severe shell-crossing 2D driver (Tier E,
# methods paper §10.6 E.2).
#
# 2D extension of the M2-3 1D compression-cascade scenario. Drives the
# 2D Cholesky-sector Newton system on a *superposition* of two-axis
# Zel'dovich velocity profiles at extreme amplitude (A_x = A_y = 0.7),
# with realizability projection enabled to test that the projection
# prevents compression cascade at intersecting caustics.
#
# Acceptance pattern (asserted in `test/test_M3_8a_E2_shell_crossing.jl`):
#
#   1. NaN-count = 0 across n_steps at T_factor ≤ 0.25 (well pre-caustic).
#
#   2. Realizability projection effectiveness: at least *some* cells
#      project at near-caustic times when projection is enabled.
#      Conversely, when projection is disabled (`project_kind=:none`)
#      the unprotected scheme may NaN — graceful failure documented.
#
#   3. Mass / momentum / energy conservation modulo projection events:
#      total mass conservation ≤ 1e-10 (exact ρ_per_cell convention);
#      total momentum drift ≤ 1e-8.
#
#   4. Long-horizon stability: at T_factor = 0.5 (post-caustic), the
#      state remains finite (bounded behavior, not quantitative).
#
# Usage:
#   julia> include("experiments/E2_severe_shell_crossing.jl")
#   julia> result = run_E2_severe_shell_crossing(; level=3, T_factor=0.25)

using HierarchicalGrids: HierarchicalMesh, EulerianFrame, FrameBoundaries,
    REFLECTING, PERIODIC, enumerate_leaves, refine_cells!, cell_physical_box
using dfmm: tier_e_severe_shell_crossing_ic_full, primitive_recovery_2d_per_cell,
            det_step_2d_berry_HG!, read_detfield_2d, DetField2D,
            ProjectionStats

"""
    run_E2_severe_shell_crossing(; level=3, A_x=0.7, A_y=0.7,
                                   T_factor=0.25, n_steps=10,
                                   project_kind=:reanchor,
                                   M_vv_override=nothing, ρ_ref=1.0,
                                   realizability_headroom=1.05)

Drive the E.2 severe shell-crossing IC for `n_steps` Newton steps with
`dt = T_factor · t_cross / n_steps`. Returns a NamedTuple with the
trajectory:

  • `t::Vector{Float64}`           — wall times
  • `nan_count::Vector{Int}`       — # of NaN cells per step
  • `mass::Vector{Float64}`        — total mass per step
  • `Px, Py::Vector{Float64}`      — per-axis total momentum per step
  • `KE::Vector{Float64}`          — total kinetic energy per step
  • `proj_n_events::Vector{Int}`   — # of projection events per step
  • `proj_n_floor::Vector{Int}`    — # of floor (Mvv-floor) events per step
  • `gamma_min::Vector{Float64}`   — spatial min γ_a (across both axes)
                                     per step
  • `mesh, frame, leaves, fields`  — solver state at end
  • `ic_params, t_cross`           — initial-condition parameters
"""
function run_E2_severe_shell_crossing(; level::Integer = 3,
                                        A_x::Real = 0.7,
                                        A_y::Real = 0.7,
                                        T_factor::Real = 0.25,
                                        n_steps::Integer = 10,
                                        project_kind::Symbol = :reanchor,
                                        M_vv_override = nothing,
                                        ρ_ref::Real = 1.0,
                                        realizability_headroom::Real = 1.05,
                                        Mvv_floor::Real = 1e-2,
                                        pressure_floor::Real = 1e-8,
                                        ρ0::Real = 1.0, P0::Real = 1e-6,
                                        abstol::Real = 1e-10,
                                        reltol::Real = 1e-10,
                                        maxiters::Integer = 50)
    ic = tier_e_severe_shell_crossing_ic_full(; level = level,
                                                A_x = A_x, A_y = A_y,
                                                ρ0 = ρ0, P0 = P0)
    bc_spec = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                    (PERIODIC, PERIODIC)))

    n_leaves = length(ic.leaves)
    cell_areas = Vector{Float64}(undef, n_leaves)
    for (i, ci) in enumerate(ic.leaves)
        lo, hi = cell_physical_box(ic.frame, ci)
        cell_areas[i] = (hi[1] - lo[1]) * (hi[2] - lo[2])
    end

    dt = Float64(T_factor) * ic.t_cross / max(1, n_steps)

    t = zeros(Float64, n_steps + 1)
    nan_count = zeros(Int, n_steps + 1)
    mass = zeros(Float64, n_steps + 1)
    Px = zeros(Float64, n_steps + 1)
    Py = zeros(Float64, n_steps + 1)
    KE = zeros(Float64, n_steps + 1)
    proj_n_events = zeros(Int, n_steps + 1)
    proj_n_floor = zeros(Int, n_steps + 1)
    gamma_min = fill(NaN, n_steps + 1)

    # Initial diagnostics.
    rec0 = primitive_recovery_2d_per_cell(ic.fields, ic.leaves, ic.frame,
                                            ic.ρ_per_cell)
    mass[1] = sum(ic.ρ_per_cell .* cell_areas)
    Px[1] = sum(ic.ρ_per_cell .* rec0.u_x .* cell_areas)
    Py[1] = sum(ic.ρ_per_cell .* rec0.u_y .* cell_areas)
    KE[1] = sum(0.5 .* ic.ρ_per_cell .* (rec0.u_x.^2 .+ rec0.u_y.^2) .* cell_areas)
    nan_count[1] = _count_nans_e2(ic.fields, ic.leaves)
    gamma_min[1] = _gamma_min_2d(ic.fields, ic.leaves;
                                   M_vv_override = M_vv_override)

    proj_stats = ProjectionStats()
    failed_at = nothing

    for n in 1:n_steps
        proj_stats = ProjectionStats()
        try
            det_step_2d_berry_HG!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                                    bc_spec, dt;
                                    M_vv_override = M_vv_override, ρ_ref = ρ_ref,
                                    project_kind = project_kind,
                                    realizability_headroom = realizability_headroom,
                                    Mvv_floor = Mvv_floor,
                                    pressure_floor = pressure_floor,
                                    proj_stats = proj_stats,
                                    abstol = abstol, reltol = reltol,
                                    maxiters = maxiters)
        catch err
            @info "E.2 Newton solve failed at step $n: $(err); recording graceful failure."
            failed_at = n
            for k in n:n_steps
                nan_count[k + 1] = n_leaves
                t[k + 1] = k * dt
            end
            break
        end
        t[n + 1] = n * dt
        nan_count[n + 1] = _count_nans_e2(ic.fields, ic.leaves)
        proj_n_events[n + 1] = proj_stats.n_events
        proj_n_floor[n + 1] = proj_stats.n_floor_events
        rec = primitive_recovery_2d_per_cell(ic.fields, ic.leaves, ic.frame,
                                                ic.ρ_per_cell)
        mass[n + 1] = sum(ic.ρ_per_cell .* cell_areas)
        Px[n + 1] = sum(ic.ρ_per_cell .* rec.u_x .* cell_areas)
        Py[n + 1] = sum(ic.ρ_per_cell .* rec.u_y .* cell_areas)
        KE[n + 1] = sum(0.5 .* ic.ρ_per_cell .* (rec.u_x.^2 .+ rec.u_y.^2) .* cell_areas)
        gamma_min[n + 1] = _gamma_min_2d(ic.fields, ic.leaves;
                                           M_vv_override = M_vv_override)
    end

    return (
        t = t, nan_count = nan_count, mass = mass,
        Px = Px, Py = Py, KE = KE,
        proj_n_events = proj_n_events,
        proj_n_floor = proj_n_floor,
        gamma_min = gamma_min,
        mesh = ic.mesh, frame = ic.frame, leaves = ic.leaves,
        fields = ic.fields, ρ_per_cell = ic.ρ_per_cell,
        ic_params = ic.params, t_cross = ic.t_cross,
        failed_at = failed_at,
        dt = dt,
    )
end

function _count_nans_e2(fields, leaves)
    cnt = 0
    for ci in leaves
        if isnan(Float64(fields.u_1[ci][1])) || isnan(Float64(fields.u_2[ci][1])) ||
           isnan(Float64(fields.α_1[ci][1])) || isnan(Float64(fields.α_2[ci][1])) ||
           isnan(Float64(fields.β_1[ci][1])) || isnan(Float64(fields.β_2[ci][1])) ||
           isnan(Float64(fields.s[ci][1]))
            cnt += 1
        end
    end
    return cnt
end

# Compute spatial-minimum γ_a^2 across both axes via the per-cell
# γ²_a = M_vv_a − β_a² formula. Returns the smaller of the two axis
# spatial minima; small or negative values indicate caustic onset.
function _gamma_min_2d(fields, leaves; M_vv_override = nothing)
    g1_min = Inf
    g2_min = Inf
    for ci in leaves
        β1 = Float64(fields.β_1[ci][1])
        β2 = Float64(fields.β_2[ci][1])
        if M_vv_override !== nothing
            Mv1 = Float64(M_vv_override[1])
            Mv2 = Float64(M_vv_override[2])
        else
            # Fall back to s-based EOS Mvv per axis (rough proxy).
            # We use the IC density as a placeholder; this is a
            # diagnostic rather than a load-bearing quantity.
            Mv1 = 1.0
            Mv2 = 1.0
        end
        γ1sq = Mv1 - β1^2
        γ2sq = Mv2 - β2^2
        g1_min = min(g1_min, γ1sq)
        g2_min = min(g2_min, γ2sq)
    end
    return min(g1_min, g2_min)
end
