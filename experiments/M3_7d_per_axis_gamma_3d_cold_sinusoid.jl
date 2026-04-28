# M3_7d_per_axis_gamma_3d_cold_sinusoid.jl
#
# §7.5 Per-axis γ selectivity test in 3D (M3-7 design note
# `reference/notes_M3_7_3d_extension.md`):
#
# Cold sinusoid IC with per-axis wave numbers (kx, ky, kz):
#   ρ uniform, u_a = A · sin(2π * k_a · x_a / L_a) for a ∈ {1,2,3}.
# Run on a 3D Cholesky-sector field set for a short window. The
# per-axis γ diagnostic must demonstrate per-axis selectivity:
#
#   1D-symmetric (k = (1, 0, 0)): γ_1 collapses spatially; γ_2, γ_3
#                                  uniform.
#   2D-symmetric (k = (1, 1, 0)): γ_1, γ_2 collapse correlated;
#                                  γ_3 uniform.
#   Full 3D       (k = (1, 1, 1)): all three γ_a non-trivial
#                                  simultaneously.
#
# The 3D analog of `experiments/M3_3d_per_axis_gamma_cold_sinusoid.jl`.
#
# Usage (REPL):
#   julia> include("experiments/M3_7d_per_axis_gamma_3d_cold_sinusoid.jl")
#   julia> result = run_M3_7d_per_axis_gamma_selectivity_3d()

using HierarchicalGrids: HierarchicalMesh, EulerianFrame,
    refine_cells!, enumerate_leaves, FrameBoundaries, REFLECTING, PERIODIC,
    cell_physical_box
using dfmm: allocate_cholesky_3d_fields, write_detfield_3d!,
            read_detfield_3d, DetField3D,
            det_step_3d_berry_HG!, gamma_per_axis_3d_field

"""
    init_cold_sinusoid_3d!(fields, mesh, frame, leaves;
                            A=0.3, kx=1, ky=0, kz=0,
                            α0=1.0, β0=0.0, s0=0.0)

Initialize a 3D Cholesky-sector field set with the C.2 cold-sinusoid
IC, generalised to 3D:

    ρ uniform (per cell),
    u_a(x_a) = A * sin(2π * k_a (x_a - lo_a) / L_a)    for a = 1, 2, 3
            (= 0 if k_a == 0).

α_a = α0, β_a = β0, θ_ab = 0, s = s0 (uniform). Cold limit (β_a = 0)
is the natural starting state.
"""
function init_cold_sinusoid_3d!(fields, mesh, frame, leaves;
                                  A::Real = 0.3,
                                  kx::Integer = 1, ky::Integer = 0, kz::Integer = 0,
                                  α0::Real = 1.0, β0::Real = 0.0,
                                  s0::Real = 0.0)
    lo_box = (frame.lo[1], frame.lo[2], frame.lo[3])
    L = (frame.hi[1] - frame.lo[1],
         frame.hi[2] - frame.lo[2],
         frame.hi[3] - frame.lo[3])
    for ci in leaves
        lo, hi = cell_physical_box(frame, ci)
        cx = 0.5 * (lo[1] + hi[1])
        cy = 0.5 * (lo[2] + hi[2])
        cz = 0.5 * (lo[3] + hi[3])
        u1 = kx == 0 ? 0.0 : A * sin(2π * kx * (cx - lo_box[1]) / L[1])
        u2 = ky == 0 ? 0.0 : A * sin(2π * ky * (cy - lo_box[2]) / L[2])
        u3 = kz == 0 ? 0.0 : A * sin(2π * kz * (cz - lo_box[3]) / L[3])
        v = DetField3D((cx, cy, cz), (u1, u2, u3),
                        (α0, α0, α0), (β0, β0, β0),
                        0.0, 0.0, 0.0, s0)
        write_detfield_3d!(fields, ci, v)
    end
    return fields
end

"""
    run_M3_7d_per_axis_gamma_selectivity_3d(; level=2, A=0.3,
                                              kx=1, ky=0, kz=0,
                                              dt=2e-3, n_steps=40,
                                              M_vv_override=(1.0,1.0,1.0),
                                              ρ_ref=1.0) -> NamedTuple

Drive the §7.5 per-axis γ selectivity test in 3D. Builds a 3D
`HierarchicalMesh{3}` of `(2^level)^3` leaves, initializes the cold
sinusoid IC with `(kx, ky, kz)`, and runs `n_steps` of
`det_step_3d_berry_HG!`. At each step records per-axis γ trajectory
(min/max across cells per axis) and returns a NamedTuple with the
final per-axis γ field.

# Mesh sizing

`level = 2` ⇒ 4×4×4 = 64 leaves; `level = 3` ⇒ 8×8×8 = 512 leaves
(the latter is the headline-figure resolution; `level = 2` is the
fast unit-test default).

# Wave-number setups (M3-7 §5)

| (kx, ky, kz) | Active axes | Trivial axes | Expected γ pattern |
|---|---|---|---|
| (1, 0, 0) | 1 | 2, 3 | γ_1 collapses; γ_2, γ_3 uniform |
| (1, 1, 0) | 1, 2 | 3 | γ_1, γ_2 collapse correlated; γ_3 uniform |
| (1, 1, 1) | 1, 2, 3 | none | all three γ_a non-trivial |
"""
function run_M3_7d_per_axis_gamma_selectivity_3d(; level::Integer = 2,
                                                   A::Real = 0.3,
                                                   kx::Integer = 1,
                                                   ky::Integer = 0,
                                                   kz::Integer = 0,
                                                   dt::Real = 2e-3,
                                                   n_steps::Integer = 40,
                                                   M_vv_override = (1.0, 1.0, 1.0),
                                                   ρ_ref::Real = 1.0)
    mesh = HierarchicalMesh{3}(; balanced = true)
    for _ in 1:level
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    leaves = enumerate_leaves(mesh)
    frame = EulerianFrame(mesh, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    fields = allocate_cholesky_3d_fields(mesh)
    init_cold_sinusoid_3d!(fields, mesh, frame, leaves;
                             A = A, kx = kx, ky = ky, kz = kz)

    # REFLECTING BCs across all three axes (sidesteps the periodic-x
    # coordinate-wrap issue documented in M3-3c handoff and inherited
    # by 3D — a cold sinusoid with kx integer and u(0) = u(L) = 0 has
    # naturally compatible reflecting boundaries).
    bc_spec = FrameBoundaries{3}(((REFLECTING, REFLECTING),
                                    (REFLECTING, REFLECTING),
                                    (REFLECTING, REFLECTING)))

    γ1_max = zeros(Float64, n_steps + 1); γ1_min = zeros(Float64, n_steps + 1)
    γ2_max = zeros(Float64, n_steps + 1); γ2_min = zeros(Float64, n_steps + 1)
    γ3_max = zeros(Float64, n_steps + 1); γ3_min = zeros(Float64, n_steps + 1)

    γ_init = gamma_per_axis_3d_field(fields, leaves;
                                      M_vv_override = M_vv_override,
                                      ρ_ref = ρ_ref)
    γ1_max[1] = maximum(γ_init[1, :]); γ1_min[1] = minimum(γ_init[1, :])
    γ2_max[1] = maximum(γ_init[2, :]); γ2_min[1] = minimum(γ_init[2, :])
    γ3_max[1] = maximum(γ_init[3, :]); γ3_min[1] = minimum(γ_init[3, :])

    t = zeros(Float64, n_steps + 1)
    for n in 1:n_steps
        det_step_3d_berry_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                               M_vv_override = M_vv_override,
                               ρ_ref = ρ_ref)
        t[n + 1] = n * dt
        γ_now = gamma_per_axis_3d_field(fields, leaves;
                                         M_vv_override = M_vv_override,
                                         ρ_ref = ρ_ref)
        γ1_max[n + 1] = maximum(γ_now[1, :]); γ1_min[n + 1] = minimum(γ_now[1, :])
        γ2_max[n + 1] = maximum(γ_now[2, :]); γ2_min[n + 1] = minimum(γ_now[2, :])
        γ3_max[n + 1] = maximum(γ_now[3, :]); γ3_min[n + 1] = minimum(γ_now[3, :])
    end

    γ_final = gamma_per_axis_3d_field(fields, leaves;
                                       M_vv_override = M_vv_override,
                                       ρ_ref = ρ_ref)

    return (
        t = t,
        γ1_max = γ1_max, γ1_min = γ1_min,
        γ2_max = γ2_max, γ2_min = γ2_min,
        γ3_max = γ3_max, γ3_min = γ3_min,
        γ_final = γ_final,
        mesh = mesh, frame = frame, leaves = leaves,
        fields = fields,
        params = (level = level, A = A,
                  kx = kx, ky = ky, kz = kz,
                  dt = dt, n_steps = n_steps,
                  M_vv_override = M_vv_override, ρ_ref = ρ_ref),
    )
end
