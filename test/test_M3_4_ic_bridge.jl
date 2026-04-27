# test_M3_4_ic_bridge.jl
#
# M3-4 Phase 2 Deliverable 1: Tier-C IC bridge + primitive recovery.
#
# Bridges primitive `(ρ, u_x, u_y, P)` cell-averages onto the M3-3
# 12-field Cholesky-sector state consumed by `det_step_2d_berry_HG!`.
# Cold-limit, isotropic IC convention: α = 1, β = 0, θ_R = 0, Pp = Q = 0;
# s solved from EOS `Mvv(1/ρ, s) = P/ρ`.
#
# Acceptance gates (this test file):
#
#   1. `s_from_pressure_density` round-trip: `Mvv(1/ρ, s_from_pressure_density(ρ, P))
#      ≈ P/ρ` to ≤ 1e-12 relative across `(ρ, P) ∈ {(0.125, 0.1), (1, 1), (10, 100)}`.
#
#   2. `cholesky_sector_state_from_primitive` field placement: returned
#      DetField2D has α = (1,1), β = (0,0), θ_R = 0, Pp = Q = 0, with
#      x and u from the input. The s solves the EOS round-trip.
#
#   3. `primitive_recovery_2d` round-trip on a uniform IC: for `(ρ_0, u_0, P_0)`
#      uniform across all cells, `(ρ̂, û_x, û_y, P̂) ≈ (ρ_0, u_x, u_y, P_0)`
#      to ≤ 1e-12 relative.
#
#   4. `primitive_recovery_2d_per_cell` round-trip on a step-function IC
#      (Sod-style): per-cell ρ matches the IC; per-cell P matches via
#      ρ · M_vv(1/ρ, s).
#
#   5. Full-IC factory smoke: `tier_c_sod_full_ic`, `tier_c_cold_sinusoid_full_ic`,
#      `tier_c_plane_wave_full_ic` produce field sets with the expected
#      shape (12 named fields, mesh.balanced == true, `length(leaves) ==
#      (2^level)^2`) and the per-leaf state matches the cholesky bridge
#      applied to the primitive IC analytically.

using Test
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, cell_physical_box
using dfmm: DetField2D, allocate_cholesky_2d_fields,
            read_detfield_2d, write_detfield_2d!,
            s_from_pressure_density,
            cholesky_sector_state_from_primitive,
            primitive_recovery_2d, primitive_recovery_2d_per_cell,
            tier_c_sod_full_ic, tier_c_cold_sinusoid_full_ic,
            tier_c_plane_wave_full_ic,
            Mvv, GAMMA_LAW_DEFAULT, CV_DEFAULT
using dfmm: pack_state_2d_berry, unpack_state_2d_berry!

const M3_4_BRIDGE_TOL = 1e-12

@testset "M3-4 Phase 2: IC bridge — s_from_pressure_density round-trip" begin
    for (ρ, P) in [(0.125, 0.1), (1.0, 1.0), (10.0, 100.0), (0.5, 0.3)]
        s = s_from_pressure_density(ρ, P)
        Mvv_check = Mvv(1.0 / ρ, s)
        ratio = Mvv_check / (P / ρ)
        @test abs(ratio - 1.0) ≤ M3_4_BRIDGE_TOL
        # And via the equivalent (ρ, P) → s identity.
        @test isfinite(s)
    end

    # Errors on non-positive inputs.
    @test_throws ArgumentError s_from_pressure_density(0.0, 1.0)
    @test_throws ArgumentError s_from_pressure_density(1.0, 0.0)
    @test_throws ArgumentError s_from_pressure_density(-1.0, 1.0)
end

@testset "M3-4 Phase 2: cholesky_sector_state_from_primitive — field placement" begin
    ρ = 1.0; ux = 0.5; uy = -0.3; P = 0.7
    cx, cy = 0.25, 0.75
    v = cholesky_sector_state_from_primitive(ρ, ux, uy, P, (cx, cy))
    @test v.x[1] ≈ cx
    @test v.x[2] ≈ cy
    @test v.u[1] ≈ ux
    @test v.u[2] ≈ uy
    @test v.alphas == (1.0, 1.0)
    @test v.betas == (0.0, 0.0)
    @test v.θ_R == 0.0
    @test v.Pp == 0.0
    @test v.Q == 0.0
    # s satisfies the EOS round-trip.
    @test abs(Mvv(1.0 / ρ, v.s) - P / ρ) ≤ M3_4_BRIDGE_TOL
end

@testset "M3-4 Phase 2: primitive_recovery_2d — uniform IC round-trip" begin
    mesh = HierarchicalMesh{2}(; balanced = true)
    for _ in 1:3
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    leaves = enumerate_leaves(mesh)
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
    fields = allocate_cholesky_2d_fields(mesh)
    ρ0, ux0, uy0, P0 = 1.0, 0.3, -0.2, 0.5
    for ci in leaves
        lo, hi = cell_physical_box(frame, ci)
        cx = 0.5 * (lo[1] + hi[1])
        cy = 0.5 * (lo[2] + hi[2])
        v = cholesky_sector_state_from_primitive(ρ0, ux0, uy0, P0, (cx, cy))
        write_detfield_2d!(fields, ci, v)
    end
    rec = primitive_recovery_2d(fields, leaves, frame; ρ_ref = ρ0)
    @test all(abs.(rec.ρ .- ρ0) .≤ M3_4_BRIDGE_TOL)
    @test all(abs.(rec.u_x .- ux0) .≤ M3_4_BRIDGE_TOL)
    @test all(abs.(rec.u_y .- uy0) .≤ M3_4_BRIDGE_TOL)
    @test all(abs.(rec.P .- P0) ./ P0 .≤ M3_4_BRIDGE_TOL)
    @test length(rec.x) == length(leaves)
    @test length(rec.y) == length(leaves)
    # Cell centers in [0,1].
    @test all(0.0 .≤ rec.x .≤ 1.0)
    @test all(0.0 .≤ rec.y .≤ 1.0)
end

@testset "M3-4 Phase 2: primitive_recovery_2d_per_cell — step-function IC" begin
    mesh = HierarchicalMesh{2}(; balanced = true)
    for _ in 1:3
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    leaves = enumerate_leaves(mesh)
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
    fields = allocate_cholesky_2d_fields(mesh)
    ρ_per_cell = Float64[]
    for ci in leaves
        lo, hi = cell_physical_box(frame, ci)
        cx = 0.5 * (lo[1] + hi[1])
        cy = 0.5 * (lo[2] + hi[2])
        ρ_i = cx < 0.5 ? 1.0 : 0.125
        P_i = cx < 0.5 ? 1.0 : 0.1
        v = cholesky_sector_state_from_primitive(ρ_i, 0.0, 0.0, P_i, (cx, cy))
        write_detfield_2d!(fields, ci, v)
        push!(ρ_per_cell, ρ_i)
    end
    rec = primitive_recovery_2d_per_cell(fields, leaves, frame, ρ_per_cell)
    @test all(abs.(rec.ρ .- ρ_per_cell) .≤ M3_4_BRIDGE_TOL)
    # Reconstructed pressure matches the per-cell IC value to round-off.
    for (i, ci) in enumerate(leaves)
        lo, hi = cell_physical_box(frame, ci)
        cx = 0.5 * (lo[1] + hi[1])
        P_expect = cx < 0.5 ? 1.0 : 0.1
        @test abs(rec.P[i] - P_expect) / P_expect ≤ M3_4_BRIDGE_TOL
    end
end

@testset "M3-4 Phase 2: tier_c_sod_full_ic — shape + per-leaf state" begin
    ic = tier_c_sod_full_ic(; level = 4, shock_axis = 1)
    @test ic.name == "tier_c_sod_full"
    @test ic.mesh.balanced == true
    @test length(ic.leaves) == 256                         # 2^(2*4)
    @test length(ic.ρ_per_cell) == 256
    # Per-leaf state: read DetField2D and verify bridge invariants.
    for ci in ic.leaves
        v = read_detfield_2d(ic.fields, ci)
        @test v.alphas == (1.0, 1.0)
        @test v.betas  == (0.0, 0.0)
        @test v.θ_R    == 0.0
        @test v.Pp     == 0.0
        @test v.Q      == 0.0
        # Velocity = 0 (default uL = uR = 0).
        @test v.u == (0.0, 0.0)
    end
    # Sod-step density profile along x: cells with center < 0.5 have ρ = 1,
    # cells with center ≥ 0.5 have ρ = 0.125.
    n_left = count(ρ -> ρ == 1.0, ic.ρ_per_cell)
    n_right = count(ρ -> ρ == 0.125, ic.ρ_per_cell)
    @test n_left == 128         # half of 256, x_split = 0.5 on level-4 mesh
    @test n_right == 128
    # Pressure recovery matches the IC.
    rec = primitive_recovery_2d_per_cell(ic.fields, ic.leaves, ic.frame, ic.ρ_per_cell)
    for i in eachindex(ic.leaves)
        P_expect = ic.ρ_per_cell[i] == 1.0 ? 1.0 : 0.1
        @test abs(rec.P[i] - P_expect) / P_expect ≤ M3_4_BRIDGE_TOL
    end
end

@testset "M3-4 Phase 2: tier_c_cold_sinusoid_full_ic — shape + per-leaf state" begin
    ic = tier_c_cold_sinusoid_full_ic(; level = 3, A = 0.5, k = (1, 0))
    @test ic.name == "tier_c_cold_sinusoid_full"
    @test ic.mesh.balanced == true
    @test length(ic.leaves) == 64
    # Density uniform = ρ0 = 1.0.
    @test all(ic.ρ_per_cell .== 1.0)
    # u_y = 0 because k = (1, 0).
    for ci in ic.leaves
        v = read_detfield_2d(ic.fields, ci)
        @test v.u[2] == 0.0
        @test v.alphas == (1.0, 1.0)
        @test v.betas == (0.0, 0.0)
    end
    # u_x analytical match.
    for ci in ic.leaves
        lo, hi = cell_physical_box(ic.frame, ci)
        cx = 0.5 * (lo[1] + hi[1])
        v = read_detfield_2d(ic.fields, ci)
        u_expect = 0.5 * sin(2π * cx)
        @test abs(v.u[1] - u_expect) ≤ M3_4_BRIDGE_TOL
    end
end

@testset "M3-4 Phase 2: tier_c_plane_wave_full_ic — rotational consistency" begin
    A = 1e-3
    ρ0 = 1.0; P0 = 1.0
    ic_x  = tier_c_plane_wave_full_ic(; level = 3, A = A, k_mag = 1, angle = 0.0,
                                         ρ0 = ρ0, P0 = P0)
    ic_y  = tier_c_plane_wave_full_ic(; level = 3, A = A, k_mag = 1, angle = π/2,
                                         ρ0 = ρ0, P0 = P0)
    @test ic_x.name == "tier_c_plane_wave_full"
    @test length(ic_x.leaves) == 64
    @test length(ic_y.leaves) == 64

    # At θ = 0: u_y = 0, u_x = (c/ρ0) δρ.
    for ci in ic_x.leaves
        v = read_detfield_2d(ic_x.fields, ci)
        @test abs(v.u[2]) ≤ M3_4_BRIDGE_TOL
    end
    # At θ = π/2: u_x = 0.
    for ci in ic_y.leaves
        v = read_detfield_2d(ic_y.fields, ci)
        @test abs(v.u[1]) ≤ 1e-15        # cos(π/2) ≈ round-off
    end

    # ρ_per_cell ≈ ρ0 ± A (plane wave amplitude).
    @test maximum(abs.(ic_x.ρ_per_cell .- ρ0)) ≤ A + M3_4_BRIDGE_TOL
    @test maximum(abs.(ic_y.ρ_per_cell .- ρ0)) ≤ A + M3_4_BRIDGE_TOL
end

@testset "M3-4 Phase 2: pack_state_2d_berry round-trip preserves bridge state" begin
    ic = tier_c_cold_sinusoid_full_ic(; level = 3, A = 0.5, k = (1, 0))
    y = pack_state_2d_berry(ic.fields, ic.leaves)
    # M3-6 Phase 0: 11-dof per cell (was 9 in M3-3c); two new fields
    # `β_12, β_21` joined the residual.
    @test length(y) == 11 * length(ic.leaves)

    # Mutate a copy and unpack — cell content should round-trip.
    fields2 = allocate_cholesky_2d_fields(ic.mesh)
    unpack_state_2d_berry!(fields2, ic.leaves, y)
    for ci in ic.leaves
        v1 = read_detfield_2d(ic.fields, ci)
        v2 = read_detfield_2d(fields2, ci)
        # Newton-unknown slots round-trip exactly.
        @test v2.x == v1.x
        @test v2.u == v1.u
        @test v2.alphas == v1.alphas
        @test v2.betas == v1.betas
        @test v2.betas_off == v1.betas_off    # M3-6 Phase 0
        @test v2.θ_R == v1.θ_R
    end
end
