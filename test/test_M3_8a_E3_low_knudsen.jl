# test_M3_8a_E3_low_knudsen.jl
#
# M3-8 Phase a: E.3 very low Knudsen 2D acceptance gates
# (methods paper §10.6 E.3).
#
# Tests the implicit Newton solver in the small-τ BGK relaxation regime
# (Navier-Stokes limit). The 2D Cholesky-sector substrate is exercised
# on a smooth low-amplitude strain mode; the test asserts that the
# state remains in the near-equilibrium manifold (β small, γ²_a ≈ M_vv)
# without timestep blowup.
#
# Note on the deterministic-only substrate: the M3-3 deterministic
# Newton step does not include explicit BGK relaxation (the M2-3 Pp/Q
# sector is operator-split outside the deterministic step). For E.3 we
# verify that the *deterministic* Newton handles the stiff-τ smooth
# strain mode without numerical instability, and that the Cholesky
# state remains close to local equilibrium across n_steps. This is the
# variational analog of the Navier-Stokes-limit verification.
#
# References:
#   • methods paper §10.6 E.3 (Tier E.3 spec)
#   • reference/notes_M3_8a_tier_e_gpu_prep.md
#   • experiments/E3_low_knudsen.jl

using Test
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, PERIODIC, cell_physical_box
using dfmm: tier_e_low_knudsen_ic_full, primitive_recovery_2d_per_cell,
            det_step_2d_berry_HG!

include(joinpath(@__DIR__, "..", "experiments", "E3_low_knudsen.jl"))

@testset "M3-8a E.3: very low Knudsen 2D" begin
    @testset "E.3 IC bridge round-trip (k=(1,0), level=2)" begin
        ic = tier_e_low_knudsen_ic_full(; level = 2, k = (1, 0), A_u = 1e-2)
        rec = primitive_recovery_2d_per_cell(ic.fields, ic.leaves, ic.frame,
                                                ic.ρ_per_cell)
        L1 = ic.params.L1
        for i in eachindex(ic.leaves)
            x_i = rec.x[i]
            u_x_expect = 1e-2 * sin(2π * (x_i - ic.params.lo[1]) / L1)
            @test abs(rec.u_x[i] - u_x_expect) ≤ 1e-12
            # Trivial axis (k_2=0): u_y=0.
            @test abs(rec.u_y[i]) ≤ 1e-14
            @test abs(rec.ρ[i] - 1.0) ≤ 1e-14
        end
    end

    @testset "E.3 Knudsen number characterization" begin
        ic1 = tier_e_low_knudsen_ic_full(; level = 2, τ = 1e-6)
        ic2 = tier_e_low_knudsen_ic_full(; level = 2, τ = 1e-3)
        @test ic1.params.Kn < ic2.params.Kn
        @test ic1.params.Kn < 1e-3   # Stiff regime.
        @test ic2.params.Kn > 1e-4   # Less stiff.
        # τ_dyn is set by the sound-crossing time L/c_s.
        @test ic1.params.τ_dyn > 0.0
        @test ic1.params.τ_dyn < 100.0
    end

    @testset "E.3 Navier-Stokes limit verification (k=(1,0), n=5 steps)" begin
        result = run_E3_low_knudsen(; level = 2, k = (1, 0), A_u = 1e-2,
                                      τ = 1e-6, n_steps = 5, dt = 1e-5,
                                      M_vv_override = (1.0, 1.0))
        # No NaN.
        for k in eachindex(result.nan_count)
            @test result.nan_count[k] == 0
        end
        # β_max grows linearly (kinematic strain accumulation in the
        # cold-limit regime); should remain small.
        @test result.beta_max[end] < 1e-2   # Bounded by the IC amplitude scale.
        # γ²_a stays close to M_vv (near-equilibrium preservation).
        @test maximum(result.gamma_dev_max) < 1e-2
        # Mass conservation.
        @test result.mass[end] ≈ result.mass[1] atol=1e-12
        # Trivial-axis momentum (k_2=0) stays at 0.
        for Py_t in result.Py
            @test abs(Py_t) ≤ 1e-12
        end
        # Non-trivial-axis momentum drift bounded.
        Px_drift = abs(result.Px[end] - result.Px[1])
        @test Px_drift ≤ 1e-10
    end

    @testset "E.3 axis-swap symmetry (k=(1,0) vs k=(0,1))" begin
        # Swap k from (1,0) to (0,1): u_x ↔ u_y.
        ic_x = tier_e_low_knudsen_ic_full(; level = 2, k = (1, 0), A_u = 1e-2)
        ic_y = tier_e_low_knudsen_ic_full(; level = 2, k = (0, 1), A_u = 1e-2)
        rec_x = primitive_recovery_2d_per_cell(ic_x.fields, ic_x.leaves, ic_x.frame,
                                                  ic_x.ρ_per_cell)
        rec_y = primitive_recovery_2d_per_cell(ic_y.fields, ic_y.leaves, ic_y.frame,
                                                  ic_y.ρ_per_cell)
        # Per-cell: at (x_i, y_i), u_x of k=(1,0) IC should equal u_y of
        # k=(0,1) IC at the swapped position (y_i, x_i).
        for i in 1:min(4, length(ic_x.leaves))
            xi, yi = rec_x.x[i], rec_x.y[i]
            ux_i = rec_x.u_x[i]
            j = findfirst(k -> abs(rec_y.x[k] - yi) < 1e-10 && abs(rec_y.y[k] - xi) < 1e-10,
                          eachindex(ic_y.leaves))
            if j !== nothing
                @test abs(rec_y.u_y[j] - ux_i) ≤ 1e-12
            end
        end
    end

    @testset "E.3 stiffer τ regression (τ=1e-8 vs τ=1e-6)" begin
        # Even stiffer τ should not destabilize the Newton solver.
        # The deterministic step ignores τ explicitly (no Pp/Q sector
        # in the M3-3 path), so this is a regression that the smooth
        # strain mode handling is τ-independent.
        result_stiff = run_E3_low_knudsen(; level = 2, k = (1, 0), A_u = 1e-2,
                                            τ = 1e-8, n_steps = 3,
                                            dt = 1e-5,
                                            M_vv_override = (1.0, 1.0))
        result_less = run_E3_low_knudsen(; level = 2, k = (1, 0), A_u = 1e-2,
                                           τ = 1e-6, n_steps = 3,
                                           dt = 1e-5,
                                           M_vv_override = (1.0, 1.0))
        for k in eachindex(result_stiff.nan_count)
            @test result_stiff.nan_count[k] == 0
            @test result_less.nan_count[k] == 0
        end
        # Same dynamics across τ (deterministic Cholesky-sector path
        # is τ-independent).
        for k in eachindex(result_stiff.beta_max)
            @test abs(result_stiff.beta_max[k] - result_less.beta_max[k]) ≤ 1e-12
        end
    end

    @testset "E.3 amplitude scaling (A_u = 1e-3 vs A_u = 1e-2)" begin
        result_small = run_E3_low_knudsen(; level = 2, k = (1, 0), A_u = 1e-3,
                                            τ = 1e-6, n_steps = 3,
                                            dt = 1e-5,
                                            M_vv_override = (1.0, 1.0))
        result_large = run_E3_low_knudsen(; level = 2, k = (1, 0), A_u = 1e-2,
                                            τ = 1e-6, n_steps = 3,
                                            dt = 1e-5,
                                            M_vv_override = (1.0, 1.0))
        # Both stable.
        for k in eachindex(result_small.nan_count)
            @test result_small.nan_count[k] == 0
            @test result_large.nan_count[k] == 0
        end
        # Large amplitude should drive larger β.
        @test result_large.beta_max[end] > result_small.beta_max[end]
        # γ deviation should grow with amplitude.
        @test result_large.gamma_dev_max[end] >= result_small.gamma_dev_max[end]
    end
end
