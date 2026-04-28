# C1_3d_sod.jl
#
# M3-7e: C.1 1D-symmetric 3D Sod driver. The 3D analog of
# `experiments/C1_2d_sod_1d_symmetric.jl`.
#
# Drives the 3D Cholesky-sector Newton system (`det_step_3d_berry_HG!`)
# with the 1D-symmetric 3D Sod IC: a step-function discontinuity in
# `(ρ, P)` along `shock_axis ∈ {1, 2, 3}`, trivial along the other two
# axes. Uses `tier_c_sod_3d_full_ic` (M3-7e Phase a).
#
# Acceptance gates (asserted in `test/test_M3_7e_C1_3d_sod.jl`):
#
#   1. Transverse (y, z)-independence: ρ(x, y_1, z_1) ≈ ρ(x, y_2, z_2)
#      per output step, ≤ 1e-12 absolute.
#   2. Conservation: total mass exact (fixed by ρ_per_cell convention),
#      transverse momenta P_y, P_z ≡ 0 throughout.
#   3. Bridge round-trip: per-cell primitive recovery matches the IC
#      profile to round-off at t = 0.
#
# Usage:
#   julia> include("experiments/C1_3d_sod.jl")
#   julia> result = run_C1_3d_sod(; level=2, n_steps=3, dt=1e-4)

using HierarchicalGrids: HierarchicalMesh, EulerianFrame, FrameBoundaries,
    REFLECTING, PERIODIC, enumerate_leaves, refine_cells!, cell_physical_box
using dfmm: tier_c_sod_3d_full_ic, primitive_recovery_3d_per_cell,
            det_step_3d_berry_HG!, read_detfield_3d, DetField3D

"""
    run_C1_3d_sod(; level=2, dt=1e-4, n_steps=3, shock_axis=1,
                    M_vv_override=(1.0, 1.0, 1.0), ρ_ref=1.0)

Drive the C.1 1D-symmetric 3D Sod IC for `n_steps` Newton steps.
Returns a NamedTuple with the trajectory: `t::Vector{Float64}`,
`transverse_dev_max::Vector{Float64}` (max ρ deviation across the two
trivial axes per output step), `slice` (a 1D x-axis profile at the
mid-point of the trivial axes), `mesh, frame, leaves, fields,
ρ_per_cell`.

`level = 2` ⇒ 4×4×4 = 64 leaves; `level = 3` ⇒ 512 leaves.
"""
function run_C1_3d_sod(; level::Integer = 2,
                         dt::Real = 1e-4,
                         n_steps::Integer = 3,
                         shock_axis::Integer = 1,
                         M_vv_override = (1.0, 1.0, 1.0),
                         ρ_ref::Real = 1.0)
    ic = tier_c_sod_3d_full_ic(; level = level, shock_axis = shock_axis)
    # PERIODIC across trivial axes; REFLECTING along the shock axis (Sod
    # has solid endpoints in the shock direction). The mix below assumes
    # `shock_axis = 1`; for other axes we permute.
    bc_axes = ntuple(3) do a
        a == shock_axis ? (REFLECTING, REFLECTING) : (PERIODIC, PERIODIC)
    end
    bc_spec = FrameBoundaries{3}(bc_axes)

    t = zeros(Float64, n_steps + 1)
    transverse_dev_max = zeros(Float64, n_steps + 1)

    rec0 = primitive_recovery_3d_per_cell(ic.fields, ic.leaves, ic.frame,
                                            ic.ρ_per_cell)
    transverse_dev_max[1] = _transverse_independence_metric(rec0, shock_axis)

    for n in 1:n_steps
        det_step_3d_berry_HG!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                               bc_spec, dt;
                               M_vv_override = M_vv_override, ρ_ref = ρ_ref)
        t[n + 1] = n * dt
        rec = primitive_recovery_3d_per_cell(ic.fields, ic.leaves, ic.frame,
                                              ic.ρ_per_cell)
        transverse_dev_max[n + 1] = _transverse_independence_metric(rec,
                                                                       shock_axis)
    end

    rec_final = primitive_recovery_3d_per_cell(ic.fields, ic.leaves, ic.frame,
                                                ic.ρ_per_cell)
    slice = _extract_axis_slice(rec_final, shock_axis)

    return (
        t = t,
        transverse_dev_max = transverse_dev_max,
        slice = slice,
        mesh = ic.mesh, frame = ic.frame, leaves = ic.leaves,
        fields = ic.fields, ρ_per_cell = ic.ρ_per_cell,
        params = ic.params,
    )
end

# Per-shock-axis-coordinate column max(ρ) - min(ρ) across the two trivial
# axes; 0 means perfect (y, z)-independence.
function _transverse_independence_metric(rec, shock_axis::Integer)
    coord = shock_axis == 1 ? rec.x : (shock_axis == 2 ? rec.y : rec.z)
    coord_unique = sort!(unique(round.(coord; digits = 12)))
    max_dev = 0.0
    for cv in coord_unique
        mask = abs.(coord .- cv) .< 1e-10
        ρ_col = rec.ρ[mask]
        if length(ρ_col) > 1
            max_dev = max(max_dev, maximum(ρ_col) - minimum(ρ_col))
        end
    end
    return max_dev
end

# Extract a 1D profile along the shock axis at the trivial-axes-midpoint
# cells (closest cell to (0.5, 0.5) in the trivial axes).
function _extract_axis_slice(rec, shock_axis::Integer)
    if shock_axis == 1
        u_par = rec.u_x
    elseif shock_axis == 2
        u_par = rec.u_y
    else
        u_par = rec.u_z
    end
    coord = shock_axis == 1 ? rec.x : (shock_axis == 2 ? rec.y : rec.z)
    # Trivial axes' mid coordinates.
    triv_a, triv_b = if shock_axis == 1
        rec.y, rec.z
    elseif shock_axis == 2
        rec.x, rec.z
    else
        rec.x, rec.y
    end
    triv_a_unique = sort!(unique(round.(triv_a; digits = 12)))
    triv_b_unique = sort!(unique(round.(triv_b; digits = 12)))
    a_pick = triv_a_unique[max(1, length(triv_a_unique) ÷ 2)]
    b_pick = triv_b_unique[max(1, length(triv_b_unique) ÷ 2)]
    mask = (abs.(triv_a .- a_pick) .< 1e-10) .& (abs.(triv_b .- b_pick) .< 1e-10)
    perm = sortperm(coord[mask])
    return (
        coord = coord[mask][perm],
        ρ = rec.ρ[mask][perm],
        u_par = u_par[mask][perm],
        P = rec.P[mask][perm],
        triv_a_at = a_pick, triv_b_at = b_pick,
    )
end
