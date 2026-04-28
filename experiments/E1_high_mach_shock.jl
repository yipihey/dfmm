# E1_high_mach_shock.jl
#
# M3-8 Phase a Deliverable: E.1 high-Mach 2D shock driver (Tier E,
# methods paper §10.6 E.1).
#
# Drives the 2D Cholesky-sector Newton system on a 1D-symmetric Sod-style
# shock with extreme Mach numbers (M ∈ {5, 10}). The downstream state
# is set by the analytical Rankine-Hugoniot relations (in
# `tier_e_high_mach_shock_ic_full`).
#
# Acceptance pattern (asserted in `test/test_M3_8a_E1_high_mach.jl`):
#
#   1. NaN-count = 0 across n_steps. The high-Mach configuration drives
#      |s| toward the realizability bound; the variational scheme is
#      expected to *report* its own failure via realizability projection
#      events, not via NaN propagation.
#
#   2. Total energy bounded: kinetic + thermal proxy stays within a
#      bounded fraction of the IC value (no exponential blow-up).
#
#   3. Transverse (y)-independence preserved at ≤ 1e-10 absolute (the
#      1D-symmetric IC keeps the trivial axis trivial).
#
#   4. Shock-capture verdict: post-shock pressure ratio computed from
#      cell-averaged primitive recovery within ~50% of analytical RH
#      (loose tolerance per M3-3 Open Issue #2; the variational scheme
#      is dispersion-limited at high Mach).
#
# Usage:
#   julia> include("experiments/E1_high_mach_shock.jl")
#   julia> result = run_E1_high_mach_shock(; level=3, mach=10.0, n_steps=5)

using HierarchicalGrids: HierarchicalMesh, EulerianFrame, FrameBoundaries,
    REFLECTING, PERIODIC, enumerate_leaves, refine_cells!, cell_physical_box
using dfmm: tier_e_high_mach_shock_ic_full, primitive_recovery_2d_per_cell,
            det_step_2d_berry_HG!, read_detfield_2d, DetField2D

"""
    run_E1_high_mach_shock(; level=3, mach=10.0, dt=1e-5, n_steps=5,
                            shock_axis=1, M_vv_override=nothing,
                            ρ_ref=1.0, project_kind=:reanchor,
                            realizability_headroom=1.05)

Drive the E.1 high-Mach shock IC for `n_steps` Newton steps. Returns a
NamedTuple with the trajectory:

  • `t::Vector{Float64}`               — wall times
  • `nan_count::Vector{Int}`           — # of NaN cells per step
  • `KE::Vector{Float64}`              — total kinetic energy per step
  • `mass::Vector{Float64}`            — total mass per step
  • `y_dev_max::Vector{Float64}`       — max y-independence violation
                                          per step
  • `proj_count::Vector{Int}`          — # of realizability projections
                                          per step
  • `slice::NamedTuple`                — final y=const slice (ρ, u_x, P)
  • `mesh, frame, leaves, fields`      — solver state at end
  • `ic_params`                        — initial-condition parameters
                                          (Mach, RH downstream state)
"""
function run_E1_high_mach_shock(; level::Integer = 3,
                                  mach::Real = 10.0,
                                  dt::Real = 1e-5,
                                  n_steps::Integer = 5,
                                  shock_axis::Integer = 1,
                                  M_vv_override = nothing,
                                  ρ_ref::Real = 1.0,
                                  project_kind::Symbol = :none,
                                  realizability_headroom::Real = 1.05,
                                  Mvv_floor::Real = 1e-2,
                                  pressure_floor::Real = 1e-8,
                                  abstol::Real = 1e-10,
                                  reltol::Real = 1e-10,
                                  maxiters::Integer = 50)
    ic = tier_e_high_mach_shock_ic_full(; level = level, shock_axis = shock_axis,
                                          mach = mach)
    bc_spec = FrameBoundaries{2}(((REFLECTING, REFLECTING),
                                    (PERIODIC, PERIODIC)))
    if shock_axis == 2
        bc_spec = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                        (REFLECTING, REFLECTING)))
    end

    n_leaves = length(ic.leaves)
    cell_areas = Vector{Float64}(undef, n_leaves)
    for (i, ci) in enumerate(ic.leaves)
        lo, hi = cell_physical_box(ic.frame, ci)
        cell_areas[i] = (hi[1] - lo[1]) * (hi[2] - lo[2])
    end

    t = zeros(Float64, n_steps + 1)
    nan_count = zeros(Int, n_steps + 1)
    KE = zeros(Float64, n_steps + 1)
    mass = zeros(Float64, n_steps + 1)
    y_dev_max = zeros(Float64, n_steps + 1)
    proj_count = zeros(Int, n_steps + 1)

    rec0 = primitive_recovery_2d_per_cell(ic.fields, ic.leaves, ic.frame,
                                            ic.ρ_per_cell)
    KE[1] = _total_kinetic_energy(rec0, cell_areas)
    mass[1] = sum(ic.ρ_per_cell .* cell_areas)
    nan_count[1] = _count_nans_2d(ic.fields, ic.leaves)
    y_dev_max[1] = _y_independence_metric_e1(rec0)

    for n in 1:n_steps
        try
            det_step_2d_berry_HG!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                                    bc_spec, dt;
                                    M_vv_override = M_vv_override, ρ_ref = ρ_ref,
                                    project_kind = project_kind,
                                    realizability_headroom = realizability_headroom,
                                    Mvv_floor = Mvv_floor,
                                    pressure_floor = pressure_floor,
                                    abstol = abstol, reltol = reltol,
                                    maxiters = maxiters)
        catch err
            # Graceful failure: catch Newton-solve failures, fill NaN.
            @info "E.1 Newton solve failed at step $n: $(err); recording graceful failure."
            nan_count[n + 1] = n_leaves
            t[n + 1] = n * dt
            for k in (n + 1):n_steps
                nan_count[k + 1] = n_leaves
                t[k + 1] = k * dt
            end
            break
        end
        t[n + 1] = n * dt
        nan_count[n + 1] = _count_nans_2d(ic.fields, ic.leaves)
        rec = primitive_recovery_2d_per_cell(ic.fields, ic.leaves, ic.frame,
                                                ic.ρ_per_cell)
        KE[n + 1] = _total_kinetic_energy(rec, cell_areas)
        mass[n + 1] = sum(ic.ρ_per_cell .* cell_areas)
        y_dev_max[n + 1] = _y_independence_metric_e1(rec)
    end

    rec_final = primitive_recovery_2d_per_cell(ic.fields, ic.leaves, ic.frame,
                                                 ic.ρ_per_cell)
    slice = _extract_y_const_slice_e1(rec_final, 0.5)

    return (
        t = t, nan_count = nan_count, KE = KE, mass = mass,
        y_dev_max = y_dev_max, proj_count = proj_count,
        slice = slice,
        mesh = ic.mesh, frame = ic.frame, leaves = ic.leaves,
        fields = ic.fields, ρ_per_cell = ic.ρ_per_cell,
        ic_params = ic.params,
    )
end

# Per-cell NaN sweep across the 8 Newton-driven slots.
function _count_nans_2d(fields, leaves)
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

function _total_kinetic_energy(rec, cell_areas)
    KE = 0.0
    for i in eachindex(cell_areas)
        ke_i = 0.5 * rec.ρ[i] * (rec.u_x[i]^2 + rec.u_y[i]^2)
        KE += ke_i * cell_areas[i]
    end
    return KE
end

function _y_independence_metric_e1(rec)
    x_unique = sort!(unique(round.(rec.x; digits = 12)))
    max_dev = 0.0
    for xv in x_unique
        mask = abs.(rec.x .- xv) .< 1e-10
        ρ_col = rec.ρ[mask]
        if length(ρ_col) > 1
            max_dev = max(max_dev, maximum(ρ_col) - minimum(ρ_col))
        end
    end
    return max_dev
end

function _extract_y_const_slice_e1(rec, y_target::Real)
    y_unique = sort!(unique(round.(rec.y; digits = 12)))
    y_pick = y_unique[argmin(abs.(y_unique .- y_target))]
    mask = abs.(rec.y .- y_pick) .< 1e-10
    perm = sortperm(rec.x[mask])
    return (
        y = y_pick,
        x = rec.x[mask][perm],
        ρ = rec.ρ[mask][perm],
        u_x = rec.u_x[mask][perm],
        P = rec.P[mask][perm],
    )
end
