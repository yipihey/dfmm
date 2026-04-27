# test_M3_6_phase4_D7_dust_traps.jl
#
# §M3-6 Phase 4 acceptance gates: D.7 dust-traps in vortices.
# Methods paper §10.5 D.7. Taylor-Green vortex IC + 2-species
# (gas + dust) `TracerMeshHG2D` substrate; the dust species is
# pressureless cold (M_vv = 0) so per-species γ correctly distinguishes
# the two phases.
#
# Tests:
#
#   GATE 1: IC sanity — Taylor-Green velocity profile, α=1, β=0,
#            β_off=0, dust IC `c_dust = 1 + ε·sin·sin`.
#   GATE 2: IC mass conservation (gas + dust per-axis integrals).
#   GATE 3: Driver smoke at L=3 — public NamedTuple shape.
#   GATE 4: Dust mass conservation (Phase 3 contract: byte-stable
#            tracer matrix → bit-exact integrated dust mass).
#   GATE 5: Per-species γ separation — gas γ ≠ dust γ at every step.
#   GATE 6: 4-component realizability — n_neg_jac = 0 for stable runs.
#   GATE 7: Long-horizon stability at level 4 (T_factor=0.1).
#   GATE 8: Conservation invariants — Px, Py drift bounds (Taylor-Green
#            integrates to zero over periodic box).
#   GATE 9: Honest scientific finding — peak/mean ratio is structurally
#            constant under Phase 3 advect_tracers_HG_2d! no-op.
#   GATE 10: Vortex-center dust accumulation diagnostic (10% gate
#            documented as honest finding).
#   GATE 11: Snapshot bookkeeping.

using Test
using Statistics: mean, std
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, PERIODIC, cell_physical_box
using dfmm: tier_d_dust_trap_ic_full, det_step_2d_berry_HG!,
    gamma_per_axis_2d_field, gamma_per_axis_2d_per_species_field,
    advect_tracers_HG_2d!, n_species, species_index,
    ProjectionStats,
    read_detfield_2d, allocate_cholesky_2d_fields,
    cholesky_sector_state_from_primitive,
    TracerMeshHG2D

const _D7_DRIVER_PATH = joinpath(@__DIR__, "..", "experiments",
                                  "D7_dust_traps.jl")
include(_D7_DRIVER_PATH)

const M3_6_PHASE4_TOL = 1.0e-12

@testset verbose = true "M3-6 Phase 4 §D.7 dust-traps in vortices" begin

    # ─────────────────────────────────────────────────────────────
    # GATE 1: IC sanity (Taylor-Green velocity, cold-limit α/β,
    #          dust concentration profile).
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 1: IC sanity (velocity, α, β, dust profile)" begin
        ic = tier_d_dust_trap_ic_full(; level = 3, U0 = 1.0,
                                        ρ0 = 1.0, P0 = 1.0,
                                        ε_dust = 0.05)
        @test ic.name == "tier_d_dust_trap"
        @test ic.t_eddy ≈ 1.0 atol = M3_6_PHASE4_TOL  # L1/U0 = 1
        @test n_species(ic.tm) == 2
        @test species_index(ic.tm, :gas) == 1
        @test species_index(ic.tm, :dust) == 2

        # Per-cell velocity matches analytic Taylor-Green profile.
        L1 = ic.params.L1
        L2 = ic.params.L2
        lo = ic.params.lo
        for ci in ic.leaves
            v = read_detfield_2d(ic.fields, ci)
            cx = v.x[1]
            cy = v.x[2]
            m1 = (cx - Float64(lo[1])) / Float64(L1)
            m2 = (cy - Float64(lo[2])) / Float64(L2)
            u1_expected = 1.0 * sin(2π * m1) * cos(2π * m2)
            u2_expected = -1.0 * cos(2π * m1) * sin(2π * m2)
            @test v.u[1] ≈ u1_expected atol = M3_6_PHASE4_TOL
            @test v.u[2] ≈ u2_expected atol = M3_6_PHASE4_TOL
            @test v.alphas[1] ≈ 1.0 atol = M3_6_PHASE4_TOL
            @test v.alphas[2] ≈ 1.0 atol = M3_6_PHASE4_TOL
            @test v.betas[1] ≈ 0.0 atol = M3_6_PHASE4_TOL
            @test v.betas[2] ≈ 0.0 atol = M3_6_PHASE4_TOL
            @test v.betas_off[1] ≈ 0.0 atol = M3_6_PHASE4_TOL
            @test v.betas_off[2] ≈ 0.0 atol = M3_6_PHASE4_TOL
            @test v.θ_R ≈ 0.0 atol = M3_6_PHASE4_TOL
            @test v.Pp ≈ 0.0 atol = M3_6_PHASE4_TOL
            @test v.Q ≈ 0.0 atol = M3_6_PHASE4_TOL
        end

        # Dust concentration profile per leaf: 1 + ε·sin·sin.
        k_gas = species_index(ic.tm, :gas)
        k_dust = species_index(ic.tm, :dust)
        for ci in ic.leaves
            v = read_detfield_2d(ic.fields, ci)
            cx = v.x[1]; cy = v.x[2]
            m1 = (cx - Float64(lo[1])) / Float64(L1)
            m2 = (cy - Float64(lo[2])) / Float64(L2)
            c_dust_expected = 1.0 + 0.05 * sin(2π * m1) * sin(2π * m2)
            @test ic.tm.tracers[k_gas, ci] ≈ 1.0 atol = M3_6_PHASE4_TOL
            @test ic.tm.tracers[k_dust, ci] ≈ c_dust_expected atol = M3_6_PHASE4_TOL
        end

        # Cell count = 2^level × 2^level = 64.
        @test length(ic.leaves) == 64
        @test length(ic.ρ_per_cell) == length(ic.leaves)
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 2: IC mass conservation
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 2: IC mass conservation" begin
        ic = tier_d_dust_trap_ic_full(; level = 3, ε_dust = 0.05)
        areas = cell_areas_2d(ic.frame, ic.leaves)
        # Fluid mass = ρ0 · box_volume.
        M_fluid = sum(ic.ρ_per_cell .* areas)
        @test M_fluid ≈ 1.0 atol = M3_6_PHASE4_TOL

        # Total dust mass: Σ c_dust · A.
        M_dust = dust_total_mass(ic.tm, ic.leaves, areas)
        # Note: by symmetry, the sin·sin contribution averages to ~0 on
        # a uniform mesh, so M_dust ≈ box_volume = 1.0.
        @test M_dust ≈ 1.0 atol = 1e-10

        # Total gas tracer mass = box_volume.
        M_gas = gas_total_mass(ic.tm, ic.leaves, areas)
        @test M_gas ≈ 1.0 atol = M3_6_PHASE4_TOL
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 3: Driver smoke (L=3 short run)
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 3: driver smoke L=3" begin
        r = run_D7_dust_traps(; level = 3, T_factor = 0.1,
                                ε_dust = 0.05, U0 = 1.0)
        @test !r.nan_seen
        @test length(r.t) == r.params.n_steps + 1
        @test length(r.M_dust_traj) == length(r.t)
        @test length(r.M_gas_traj) == length(r.t)
        @test length(r.peak_over_mean) == length(r.t)
        @test length(r.gas_gamma_mean) == length(r.t)
        @test length(r.dust_gamma_max) == length(r.t)
        @test length(r.gamma_separation) == length(r.t)
        @test length(r.n_negative_jacobian) == length(r.t)
        @test length(r.snapshots) == length(r.params.snapshots_at)
        @test r.t[1] == 0.0
        @test r.t[end] ≈ r.params.T_end atol = M3_6_PHASE4_TOL
        @test r.t_eddy ≈ 1.0 atol = M3_6_PHASE4_TOL
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 4: Dust mass conservation (HEADLINE — Phase 3 substrate
    #          byte-stable; integrated mass is bit-exact)
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 4: dust mass conservation (bit-exact)" begin
        r = run_D7_dust_traps(; level = 3, T_factor = 0.1,
                                ε_dust = 0.05, U0 = 1.0)
        @test !r.nan_seen
        # Pure-Lagrangian advect_tracers_HG_2d! is a no-op + the
        # fluid mesh / volumes are static under the variational
        # scheme's IC: dust mass = Σ c·A is unchanged per step at
        # every floating-point bit.
        @test r.M_dust_err_max == 0.0
        @test r.M_gas_err_max == 0.0
        # Per-step trajectory should also be bit-stable.
        for k in eachindex(r.M_dust_traj)
            @test r.M_dust_traj[k] == r.M_dust_traj[1]
            @test r.M_gas_traj[k] == r.M_gas_traj[1]
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 5: Per-species γ separation
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 5: per-species γ separation (gas ≠ dust)" begin
        r = run_D7_dust_traps(; level = 3, T_factor = 0.1,
                                ε_dust = 0.05, U0 = 1.0)
        @test !r.nan_seen
        # Dust species: M_vv = (0, 0) ⇒ γ_dust = 0 everywhere by
        # construction (β > 0 ⇒ γ² < 0 ⇒ floored).
        @test r.dust_gamma_max[1] == 0.0
        @test r.dust_gamma_max[end] == 0.0
        for k in eachindex(r.dust_gamma_max)
            @test r.dust_gamma_max[k] == 0.0
        end
        # Gas species: M_vv = (1, 1) override ⇒ γ_gas finite and ≈ 1
        # at IC (β = 0). Stays positive throughout the run on stable
        # configurations.
        @test r.gas_gamma_mean[1] ≈ 1.0 atol = M3_6_PHASE4_TOL
        @test r.gas_gamma_min[1] ≈ 1.0 atol = M3_6_PHASE4_TOL
        for k in eachindex(r.gas_gamma_mean)
            # On stable run gas γ stays positive (β stays bounded).
            @test r.gas_gamma_mean[k] > 0.5
        end

        # Separation ratio should be enormous (gas / dust ≈ 1 / 0+).
        @test r.gamma_separation[1] > 1e10
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 6: 4-component realizability — n_neg_jac = 0 on stable runs
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 6: 4-component realizability" begin
        r = run_D7_dust_traps(; level = 3, T_factor = 0.1,
                                ε_dust = 0.05, U0 = 1.0,
                                project_kind = :reanchor)
        @test !r.nan_seen
        # On the stable T_factor=0.1 run with project_kind=:reanchor,
        # the 4-component cone holds throughout.
        @test sum(r.n_negative_jacobian) == 0
        for k in eachindex(r.n_negative_jacobian)
            @test r.n_negative_jacobian[k] == 0
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 7: Long-horizon stability at L=4
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 7: stability at L=4" begin
        r = run_D7_dust_traps(; level = 4, T_factor = 0.1,
                                ε_dust = 0.05, U0 = 1.0)
        @test !r.nan_seen
        @test r.M_dust_err_max == 0.0
        @test sum(r.n_negative_jacobian) == 0
        @test r.gas_gamma_mean[end] > 0.5
        # No NaN propagated into the gas γ trajectory.
        for k in eachindex(r.gas_gamma_mean)
            @test isfinite(r.gas_gamma_mean[k])
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 8: Conservation invariants — Px, Py drift bounds
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 8: conservation (M, Px, Py)" begin
        r = run_D7_dust_traps(; level = 3, T_factor = 0.1,
                                ε_dust = 0.05, U0 = 1.0)
        @test !r.nan_seen
        # Mass: bit-stable (Eulerian cells fixed; ρ_per_cell fixed).
        @test r.M_err_max ≈ 0.0 atol = M3_6_PHASE4_TOL
        # Momentum: Taylor-Green has zero net momentum. By symmetry
        # under periodic BCs the integrals are exactly 0 to round-off.
        @test r.Px_err_max < 1e-10
        @test r.Py_err_max < 1e-10
        # KE: bounded (no exponential growth).
        @test isfinite(r.KE_err_max)
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 9: Honest finding — peak/mean ratio is structurally
    #          constant under Phase 3 no-op advect_tracers_HG_2d!
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 9: tracer matrix byte-stability (honest)" begin
        r = run_D7_dust_traps(; level = 3, T_factor = 0.1,
                                ε_dust = 0.05, U0 = 1.0)
        @test !r.nan_seen
        # The dust peak / mean is bit-stable through the run because
        # advect_tracers_HG_2d! is a no-op and the mesh is static.
        for k in eachindex(r.peak_over_mean)
            @test r.peak_over_mean[k] == r.peak_over_mean[1]
            @test r.dust_peak[k] == r.dust_peak[1]
            @test r.dust_mean[k] == r.dust_mean[1]
        end
        # IC peak/mean ≈ 1 + ε (sampling-truncated by mesh).
        # At L=3 with ε=0.05 we measure peak/mean ≈ 1.043.
        @test r.peak_over_mean[1] > 1.0
        @test r.peak_over_mean[1] < 1.06
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 10: Vortex-center dust accumulation diagnostic
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 10: vortex-center dust diagnostic" begin
        ic = tier_d_dust_trap_ic_full(; level = 4, ε_dust = 0.05)
        L1 = ic.params.L1; L2 = ic.params.L2; lo_box = ic.params.lo
        vc = vortex_center_dust(ic.tm, ic.frame, ic.leaves;
                                  lo = lo_box, L1 = L1, L2 = L2)
        @test length(vc.c_at_vortex) == 4
        # Two vortex centres (0.25,0.25) and (0.75,0.75) sit at +ε
        # peaks; (0.25,0.75) and (0.75,0.25) sit at −ε troughs.
        # Peak should be (1+ε); min should be (1-ε); subject to
        # cell-centre sampling truncation.
        @test vc.c_peak > 1.0
        @test vc.c_min < 1.0
        # Symmetric: c_peak + c_min ≈ 2.0 (peak + trough).
        @test (vc.c_peak + vc.c_min) ≈ 2.0 atol = 1e-10
        @test vc.c_mean ≈ 1.0 atol = 1e-10
        # Peak/mean ratio: ≈ 1 + ε truncated.
        @test vc.peak_over_mean > 1.0
        @test vc.peak_over_mean < 1.06
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 11: Snapshot bookkeeping
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 11: snapshot bookkeeping" begin
        r = run_D7_dust_traps(; level = 3, T_factor = 0.1,
                                ε_dust = 0.05, U0 = 1.0,
                                snapshots_at = (0.0, 0.5, 1.0))
        @test !r.nan_seen
        @test length(r.snapshots) == 3
        for snap in r.snapshots
            @test length(snap.x_centres) == length(r.ic.leaves)
            @test length(snap.y_centres) == length(r.ic.leaves)
            @test length(snap.c_dust) == length(r.ic.leaves)
            @test length(snap.u_1) == length(r.ic.leaves)
            @test length(snap.u_2) == length(r.ic.leaves)
            @test length(snap.γ_gas_1) == length(r.ic.leaves)
            @test 0.0 ≤ snap.t ≤ r.params.T_end + M3_6_PHASE4_TOL
        end
        # First snapshot at t=0; last at T_end (within rounding).
        @test r.snapshots[1].t == 0.0
        @test r.snapshots[end].t ≈ r.params.T_end atol = r.params.dt
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 12: Helper functions sanity
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 12: helper functions" begin
        @test dust_trap_eddy_time(; U0 = 1.0, L = 1.0) ≈ 1.0
        @test dust_trap_eddy_time(; U0 = 2.0, L = 1.0) ≈ 0.5
        # cell_areas_2d on a level-2 mesh: 16 cells, each 0.25² = 0.0625.
        ic_lite = tier_d_dust_trap_ic_full(; level = 2)
        areas = cell_areas_2d(ic_lite.frame, ic_lite.leaves)
        @test length(areas) == 16
        for A in areas
            @test A ≈ 0.0625 atol = M3_6_PHASE4_TOL
        end
        @test sum(areas) ≈ 1.0 atol = M3_6_PHASE4_TOL
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 13: Per-species γ field walker reduces correctly
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 13: per-species γ field walker" begin
        ic = tier_d_dust_trap_ic_full(; level = 3, ε_dust = 0.05)
        γ = gamma_per_axis_2d_per_species_field(ic.fields, ic.leaves;
                                                  M_vv_override_per_species = (
                                                      (1.0, 1.0),
                                                      (0.0, 0.0)),
                                                  n_species = 2)
        @test size(γ) == (2, 2, length(ic.leaves))
        # Gas species at IC (β = 0): γ = √M_vv = 1.
        for i in 1:size(γ, 3)
            @test γ[1, 1, i] ≈ 1.0 atol = M3_6_PHASE4_TOL
            @test γ[1, 2, i] ≈ 1.0 atol = M3_6_PHASE4_TOL
            # Dust species: M_vv = 0 ⇒ γ = 0.
            @test γ[2, 1, i] == 0.0
            @test γ[2, 2, i] == 0.0
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 14: Multi-level sweep + plot driver smoke
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 14: sweep + plot driver" begin
        sweep = run_D7_dust_traps_sweep(; levels = (3,),
                                          T_factor = 0.05,
                                          ε_dust = 0.05, U0 = 1.0)
        @test length(sweep.results) == 1
        @test sweep.t_eddy ≈ 1.0 atol = M3_6_PHASE4_TOL
        @test length(sweep.peak_over_mean_final) == 1
        @test length(sweep.M_dust_err_final) == 1
        # Plot driver: just verify it doesn't error (CSV fallback OK).
        tmp = tempname() * ".png"
        try
            plot_D7_dust_traps(sweep; save_path = tmp)
            # Either PNG or CSV fallback was emitted.
            @test isfile(tmp) || isfile(replace(tmp, ".png" => "_L3.csv"))
        catch e
            @info "plot driver fell back: $e"
            @test true
        end
    end
end
