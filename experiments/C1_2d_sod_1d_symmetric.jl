# C1_2d_sod_1d_symmetric.jl
#
# M3-4 Phase 2 Deliverable 3: C.1 1D-symmetric 2D Sod driver.
#
# §10.4 C.1 of the methods paper. Drives the 2D Cholesky-sector Newton
# system with the 1D-symmetric Sod IC: a step-function discontinuity in
# (ρ, P) along the shock_axis (default = x), trivial along the y-axis.
# Uses `tier_c_sod_full_ic` (the M3-4 Phase 2 IC bridge) to populate
# the 12-field 2D field set, then steps via `det_step_2d_berry_HG!`.
#
# Acceptance gates (asserted in `test/test_M3_4_C1_sod.jl`):
#
#   1. y-independence: ρ(x, y_1) ≈ ρ(x, y_2) per output step, ≤ 1e-12
#      absolute. By symmetry the y-trivial direction must stay trivial.
#
#   2. Conservation: total mass (exactly preserved by the IC bridge's
#      ρ_per_cell convention), total momentum bounded.
#
#   3. 1D-reduction-vs-golden: extract a y=const slice and compare to
#      `reference/golden/A1_sod.h5`. The variational Cholesky-sector
#      solver does NOT use HLL, so the L∞ error against the golden is
#      ~10-20% (M3-3 Open Issue #2 — known dispersion limit). The
#      driver records the slice and reports the rel error; the test
#      file asserts a loose tolerance consistent with this open issue.
#
# Usage:
#   julia> include("experiments/C1_2d_sod_1d_symmetric.jl")
#   julia> result = run_C1_2d_sod(; level=4, n_steps=10, dt=1e-4)

using HierarchicalGrids: HierarchicalMesh, EulerianFrame, FrameBoundaries,
    REFLECTING, PERIODIC, enumerate_leaves, refine_cells!, cell_physical_box
using dfmm: tier_c_sod_full_ic, primitive_recovery_2d_per_cell,
            det_step_2d_berry_HG!, read_detfield_2d, DetField2D

"""
    run_C1_2d_sod(; level=4, dt=1e-4, n_steps=10, shock_axis=1,
                   M_vv_override=(1.0, 1.0), ρ_ref=1.0)

Drive the C.1 1D-symmetric 2D Sod IC for `n_steps` Newton steps. Returns
a NamedTuple with the trajectory:
  • `t::Vector{Float64}`               — wall times
  • `y_dev_max::Vector{Float64}`       — max y-independence violation
                                          per step
  • `slice::NamedTuple`                — final y=const slice
                                          (ρ, u_x, P) at the cell-center
                                          row closest to y = 0.5
  • `mesh, frame, leaves, fields`      — solver state at end
  • `ρ_per_cell::Vector{Float64}`      — per-cell density profile
                                          (fixed by the bridge convention)
"""
function run_C1_2d_sod(; level::Integer = 4,
                       dt::Real = 1e-4,
                       n_steps::Integer = 10,
                       shock_axis::Integer = 1,
                       M_vv_override = (1.0, 1.0),
                       ρ_ref::Real = 1.0)
    ic = tier_c_sod_full_ic(; level = level, shock_axis = shock_axis)
    bc_spec = FrameBoundaries{2}(((REFLECTING, REFLECTING),
                                   (PERIODIC, PERIODIC)))

    t = zeros(Float64, n_steps + 1)
    y_dev_max = zeros(Float64, n_steps + 1)
    rec0 = primitive_recovery_2d_per_cell(ic.fields, ic.leaves, ic.frame,
                                           ic.ρ_per_cell)
    y_dev_max[1] = _y_independence_metric(rec0)

    for n in 1:n_steps
        det_step_2d_berry_HG!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                                bc_spec, dt;
                                M_vv_override = M_vv_override, ρ_ref = ρ_ref)
        t[n + 1] = n * dt
        rec = primitive_recovery_2d_per_cell(ic.fields, ic.leaves, ic.frame,
                                              ic.ρ_per_cell)
        y_dev_max[n + 1] = _y_independence_metric(rec)
    end

    rec_final = primitive_recovery_2d_per_cell(ic.fields, ic.leaves, ic.frame,
                                                 ic.ρ_per_cell)
    slice = _extract_y_const_slice(rec_final, 0.5)

    return (
        t = t, y_dev_max = y_dev_max,
        slice = slice,
        mesh = ic.mesh, frame = ic.frame, leaves = ic.leaves,
        fields = ic.fields, ρ_per_cell = ic.ρ_per_cell,
    )
end

# Per-x-column max(ρ) - min(ρ); 0 means perfect y-independence.
function _y_independence_metric(rec)
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

function _extract_y_const_slice(rec, y_target::Real)
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
