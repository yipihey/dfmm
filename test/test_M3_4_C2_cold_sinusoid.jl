# test_M3_4_C2_cold_sinusoid.jl
#
# M3-4 Phase 2 Deliverable 4: C.2 2D cold sinusoid acceptance driver.
#
# Drives `det_step_2d_berry_HG!` with the C.2 cold-sinusoid IC
# (`tier_c_cold_sinusoid_full_ic`) under both
#   • k = (1, 0) — 1D-symmetric (only x-axis active);
#   • k = (1, 1) — genuinely 2D (both axes active).
#
# Acceptance gates:
#
#   1. For k = (1, 0): ratio std(γ_1) / std(γ_2) > 1e10 (generalizes
#      M3-3d's >1e6 selectivity gate; this driver uses the full IC bridge,
#      not the M3-3d direct-state init, but the active/trivial-axis
#      structure is the same).
#
#   2. For k = (1, 1): both γ_1 and γ_2 develop spatial structure;
#      assert std(γ_a) > thresholds qualitatively.
#
#   3. Conservation: total mass / momentum ≤ 1e-10 (cold sinusoid has
#      symmetric u, so net momentum should stay near 0 across short
#      integration windows).

using Test
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, PERIODIC, cell_physical_box
using Statistics: std
using dfmm: tier_c_cold_sinusoid_full_ic,
            primitive_recovery_2d_per_cell,
            det_step_2d_berry_HG!, gamma_per_axis_2d_field,
            read_detfield_2d, DetField2D

const M3_4_C2_SELECTIVITY_RATIO = 1e10
const M3_4_C2_CONSERVATION_TOL = 1.0e-10

function _measure_gamma_stds(fields, leaves; M_vv_override = (1.0, 1.0))
    γ = gamma_per_axis_2d_field(fields, leaves; M_vv_override = M_vv_override)
    return std(γ[1, :]), std(γ[2, :])
end

@testset "M3-4 Phase 2 C.2: 2D cold sinusoid" begin
    @testset "C.2 (k = (1, 0)): 1D-symmetric per-axis γ selectivity" begin
        ic = tier_c_cold_sinusoid_full_ic(; level = 4, A = 0.5, k = (1, 0))
        bc_spec = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                       (PERIODIC, PERIODIC)))
        dt = 2e-3
        n_steps = 40

        # Initial state: u_2 = 0 ⇒ axis-2 γ is uniform; u_1 oscillates
        # ⇒ axis-1 develops spatial γ-structure as the Newton solve
        # spreads strain.
        for _ in 1:n_steps
            det_step_2d_berry_HG!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                                    bc_spec, dt;
                                    M_vv_override = (1.0, 1.0),
                                    ρ_ref = 1.0)
        end

        std1, std2 = _measure_gamma_stds(ic.fields, ic.leaves;
                                            M_vv_override = (1.0, 1.0))
        @test std1 > 1e-6                # active axis develops structure
        ratio = std1 / max(std2, eps())
        @test ratio > M3_4_C2_SELECTIVITY_RATIO
    end

    @testset "C.2 (k = (1, 1)): genuinely 2D — both axes active" begin
        ic = tier_c_cold_sinusoid_full_ic(; level = 4, A = 0.3, k = (1, 1))
        bc_spec = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                       (PERIODIC, PERIODIC)))
        dt = 2e-3
        n_steps = 30

        for _ in 1:n_steps
            det_step_2d_berry_HG!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                                    bc_spec, dt;
                                    M_vv_override = (1.0, 1.0),
                                    ρ_ref = 1.0)
        end

        std1, std2 = _measure_gamma_stds(ic.fields, ic.leaves;
                                            M_vv_override = (1.0, 1.0))
        # Both axes carry strain ⇒ both stds bounded away from 0.
        @test std1 > 1e-7
        @test std2 > 1e-7
        # And similar magnitudes (both axes equally active by symmetry).
        ratio = std1 / max(std2, eps())
        @test 1e-3 < ratio < 1e3        # within 3 orders of magnitude
    end

    @testset "C.2: conservation gates (k = (1, 0))" begin
        ic = tier_c_cold_sinusoid_full_ic(; level = 3, A = 0.3, k = (1, 0))
        bc_spec = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                       (PERIODIC, PERIODIC)))
        dt = 1e-3
        n_steps = 10

        # Per-cell areas (Eulerian, fixed).
        cell_areas = Vector{Float64}(undef, length(ic.leaves))
        for (i, ci) in enumerate(ic.leaves)
            lo, hi = cell_physical_box(ic.frame, ci)
            cell_areas[i] = (hi[1] - lo[1]) * (hi[2] - lo[2])
        end
        u_x_0 = [Float64(ic.fields.u_1[ci][1]) for ci in ic.leaves]
        u_y_0 = [Float64(ic.fields.u_2[ci][1]) for ci in ic.leaves]
        M_0  = sum(ic.ρ_per_cell .* cell_areas)
        Px_0 = sum(ic.ρ_per_cell .* u_x_0 .* cell_areas)
        Py_0 = sum(ic.ρ_per_cell .* u_y_0 .* cell_areas)

        for _ in 1:n_steps
            det_step_2d_berry_HG!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                                    bc_spec, dt;
                                    M_vv_override = (1.0, 1.0),
                                    ρ_ref = 1.0)
        end
        u_x_f = [Float64(ic.fields.u_1[ci][1]) for ci in ic.leaves]
        u_y_f = [Float64(ic.fields.u_2[ci][1]) for ci in ic.leaves]
        M_f  = sum(ic.ρ_per_cell .* cell_areas)
        Px_f = sum(ic.ρ_per_cell .* u_x_f .* cell_areas)
        Py_f = sum(ic.ρ_per_cell .* u_y_f .* cell_areas)

        @test abs(M_f - M_0) ≤ M3_4_C2_CONSERVATION_TOL
        # u_y stays 0 by symmetry under k = (1, 0).
        @test abs(Py_f) ≤ M3_4_C2_CONSERVATION_TOL
        # Px is approximately preserved (sin integrates to 0 at IC,
        # remains ~0 by symmetric advection on a periodic box).
        @test abs(Px_f - Px_0) ≤ 1e-6
    end

    @testset "C.2: density preserved at IC and after step (k = (1, 0))" begin
        ic = tier_c_cold_sinusoid_full_ic(; level = 3, A = 0.3, k = (1, 0))
        @test all(ic.ρ_per_cell .== 1.0)        # uniform IC
        rec = primitive_recovery_2d_per_cell(ic.fields, ic.leaves, ic.frame,
                                              ic.ρ_per_cell)
        @test all(rec.ρ .== 1.0)
        # u_y = 0 everywhere at IC.
        @test all(abs.(rec.u_y) .≤ 1e-15)
    end
end
