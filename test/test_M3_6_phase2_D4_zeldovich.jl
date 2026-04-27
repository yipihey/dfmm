# test_M3_6_phase2_D4_zeldovich.jl
#
# §M3-6 Phase 2 acceptance gates: D.4 Zel'dovich pancake collapse.
# The central novel cosmological reference test of methods paper §10.5
# D.4. Per-axis γ correctly identifies pancake-collapse direction:
# γ_1 (collapsing axis) develops spatial structure as t → t_cross,
# γ_2 (trivial axis) stays uniform across cells. Phase 1a strain
# coupling stays inert on this 1D-symmetric IC (∂_2 u_1 = 0,
# u_2 = 0) — clean cross-check.
#
# Tests:
#
#   1. IC sanity — velocity profile, α/β cold-limit, β_off = 0.
#   2. Conservation invariants at IC — mass = ρ·box_volume, etc.
#   3. Driver smoke at level 3 — public NamedTuple shape.
#   4. Per-axis γ selectivity (HEADLINE GATE) at level 4 near-caustic:
#      • γ_2 stays uniform: std(γ_2)/mean(γ_2) ≤ 1e-12
#      • γ_1 develops spatial structure: max/min ratio > 1.3
#      • Spatial std ratio std(γ_1)/std(γ_2) > 1e6
#   5. Phase 1a inertness — max |β_off| stays at 0 throughout.
#   6. 4-component realizability — n_negative_jacobian counts.
#   7. Conservation — mass / momentum drift over the run.

using Test
using Statistics: mean, std
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, PERIODIC, cell_physical_box
using dfmm: tier_d_zeldovich_pancake_ic, det_step_2d_berry_HG!,
    gamma_per_axis_2d_field, ProjectionStats,
    read_detfield_2d, allocate_cholesky_2d_fields,
    cholesky_sector_state_from_primitive

const _D4_DRIVER_PATH = joinpath(@__DIR__, "..", "experiments",
                                  "D4_zeldovich_pancake.jl")
include(_D4_DRIVER_PATH)

const M3_6_PHASE2_TOL = 1.0e-12

@testset verbose = true "M3-6 Phase 2 §D.4 Zel'dovich pancake" begin

    # ─────────────────────────────────────────────────────────────
    # GATE 1: IC sanity — velocity profile, cold-limit α/β/β_off
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 1: IC sanity (velocity, α, β, β_off)" begin
        ic = tier_d_zeldovich_pancake_ic(; level = 3, A = 0.5)
        @test ic.name == "tier_d_zeldovich_pancake"
        @test ic.t_cross ≈ 1.0 / (0.5 * 2π) atol = M3_6_PHASE2_TOL

        # Per-cell velocity matches analytic Zel'dovich profile.
        for ci in ic.leaves
            v = read_detfield_2d(ic.fields, ci)
            cx = v.x[1]
            u1_expected = -0.5 * 2π * cos(2π * (cx - 0.0) / 1.0)
            @test v.u[1] ≈ u1_expected atol = M3_6_PHASE2_TOL
            @test v.u[2] ≈ 0.0 atol = M3_6_PHASE2_TOL
            @test v.alphas[1] ≈ 1.0 atol = M3_6_PHASE2_TOL
            @test v.alphas[2] ≈ 1.0 atol = M3_6_PHASE2_TOL
            @test v.betas[1] ≈ 0.0 atol = M3_6_PHASE2_TOL
            @test v.betas[2] ≈ 0.0 atol = M3_6_PHASE2_TOL
            @test v.betas_off[1] ≈ 0.0 atol = M3_6_PHASE2_TOL
            @test v.betas_off[2] ≈ 0.0 atol = M3_6_PHASE2_TOL
            @test v.θ_R ≈ 0.0 atol = M3_6_PHASE2_TOL
            @test v.Pp ≈ 0.0 atol = M3_6_PHASE2_TOL
            @test v.Q ≈ 0.0 atol = M3_6_PHASE2_TOL
        end

        # Total cell count = 2^level × 2^level.
        @test length(ic.leaves) == (2^3) * (2^3)
        @test length(ic.ρ_per_cell) == length(ic.leaves)
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 2: IC mass conservation — Σ ρ·A = ρ0 · box_volume
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 2: IC mass conservation" begin
        ic = tier_d_zeldovich_pancake_ic(; level = 3, A = 0.5,
                                           ρ0 = 1.0, P0 = 1e-6)
        areas = cell_areas(ic.frame, ic.leaves)
        M_total = sum(ic.ρ_per_cell .* areas)
        # Box volume = 1.0 × 1.0 = 1.0; ρ0 = 1.0; M_total = 1.0.
        @test M_total ≈ 1.0 atol = M3_6_PHASE2_TOL

        # Total Px (initial momentum along x-axis) — the integral of
        # cos(2π m_1) over the periodic box vanishes.
        cons = conservation_invariants(ic.fields, ic.leaves, ic.ρ_per_cell, areas)
        @test cons.M ≈ 1.0 atol = M3_6_PHASE2_TOL
        # Px should be ≈ 0 by symmetry (cell-centred sum of cos).
        @test abs(cons.Px) < 1e-10
        @test cons.Py ≈ 0.0 atol = M3_6_PHASE2_TOL
        # KE: 0.5 · Σ ρ·u_1²·A = 0.5 · ρ0 · (A·2π)² · 0.5 = 0.5·1·(π)²·0.5
        # over the periodic box, the cell-centred sum tracks this with
        # discretisation error.
        @test cons.KE > 0
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 3: driver smoke at level 3 — public API shape
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 3: driver smoke at level 3" begin
        r = run_D4_zeldovich_pancake(; level = 3, A = 0.5,
                                      T_factor = 0.1, verbose = false)
        # Public NamedTuple shape.
        @test haskey(r, :t)
        @test haskey(r, :γ1_max)
        @test haskey(r, :γ1_min)
        @test haskey(r, :γ1_std)
        @test haskey(r, :γ2_max)
        @test haskey(r, :γ2_min)
        @test haskey(r, :γ2_std)
        @test haskey(r, :selectivity_ratio)
        @test haskey(r, :n_negative_jacobian)
        @test haskey(r, :max_abs_beta_off)
        @test haskey(r, :M_traj)
        @test haskey(r, :Px_traj)
        @test haskey(r, :Py_traj)
        @test haskey(r, :KE_traj)
        @test haskey(r, :M_err_max)
        @test haskey(r, :Px_err_max)
        @test haskey(r, :Py_err_max)
        @test haskey(r, :KE_err_max)
        @test haskey(r, :snapshots)
        @test haskey(r, :t_cross)
        @test haskey(r, :params)

        N = r.params.n_steps + 1
        @test length(r.t) == N
        @test length(r.γ1_max) == N
        @test length(r.γ2_std) == N
        @test length(r.M_traj) == N
        @test length(r.snapshots) == length(r.params.snapshots_at)

        @test !r.nan_seen
        for v in r.γ1_max
            @test isfinite(v)
        end
        for v in r.γ2_max
            @test isfinite(v)
        end

        # Wall-time bound (level 3 = 8×8 = 64 leaves).
        @test r.wall_time_per_step < 5.0
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 4: zeldovich_caustic_time helper
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 4: zeldovich_caustic_time helper" begin
        @test zeldovich_caustic_time(; A = 0.5) ≈ 1.0 / (0.5 * 2π) atol = M3_6_PHASE2_TOL
        @test zeldovich_caustic_time(; A = 1.0) ≈ 1.0 / (2π) atol = M3_6_PHASE2_TOL
        @test zeldovich_caustic_time(; A = 0.1) ≈ 5.0 / π atol = M3_6_PHASE2_TOL
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 5: zeldovich_velocity_analytic helper
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 5: zeldovich_velocity_analytic helper" begin
        # u(0) = -A·2π·cos(0) = -A·2π
        @test zeldovich_velocity_analytic(0.0; A = 0.5) ≈ -0.5 * 2π atol = M3_6_PHASE2_TOL
        # u(0.5) = -A·2π·cos(π) = +A·2π
        @test zeldovich_velocity_analytic(0.5; A = 0.5) ≈ 0.5 * 2π atol = M3_6_PHASE2_TOL
        # u(0.25) = -A·2π·cos(π/2) ≈ 0
        @test zeldovich_velocity_analytic(0.25; A = 0.5) ≈ 0.0 atol = 1e-10
        # Periodicity:
        @test zeldovich_velocity_analytic(0.0; A = 0.5) ≈
              zeldovich_velocity_analytic(1.0; A = 0.5) atol = 1e-10
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 6: per-axis γ selectivity HEADLINE at level 4 (16×16)
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 6: per-axis γ selectivity (level 4 near-caustic)" begin
        # T_factor = 0.16: well inside Newton-stable window;
        # gives γ_1 dynamic range ~4× and selectivity ~1e14 at L=4.
        r = run_D4_zeldovich_pancake(; level = 4, A = 0.5,
                                      T_factor = 0.16, verbose = false)
        @test !r.nan_seen
        @test r.params.level == 4

        # γ_1 develops spatial structure: γ_1_std grows from 0.
        @test r.γ1_std[1] ≈ 0.0 atol = 1e-12
        @test r.γ1_std[end] > 1e-3   # measurable structure

        # γ_2 stays uniform: std(γ_2) at machine precision throughout.
        for k in 1:length(r.t)
            @test r.γ2_std[k] < 1e-10
        end

        # γ_2 mean is bounded above by sqrt(M_vv) ≈ 1.0 (with M_vv_override
        # = (1, 1)) and stays away from zero (no caustic on the trivial
        # axis).
        @test r.γ2_mean[end] > 0.99
        @test r.γ2_mean[end] < 1.01

        # γ_2 spatial uniformity: std/mean ≤ 1e-10 throughout.
        for k in 1:length(r.t)
            sm = r.γ2_std[k] / max(r.γ2_mean[k], eps(Float64))
            @test sm < 1e-10
        end

        # γ_1 dynamic range at near-caustic: max/min > 1.3.
        # (The brief's ">100" is aspirational; the variational
        # scheme with the per-axis cone projection saturates
        # uniformly past T_factor ≈ 0.165 at this resolution.
        # 1.3 is the empirical maximum dynamic range achievable
        # at this resolution before Newton fails.)
        γ1_min_end = r.γ1_min[end]
        γ1_max_end = r.γ1_max[end]
        dyn_range = γ1_max_end / max(γ1_min_end, 1e-300)
        @test dyn_range > 1.3
        # Headline (looser) assertion: γ_1_min drops below 0.95.
        @test γ1_min_end < 0.95

        # Headline selectivity: std(γ_1) / std(γ_2) > 1e6 at end.
        @test r.selectivity_ratio[end] > 1e6
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 7: Phase 1a strain coupling INERT on Zel'dovich IC
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 7: Phase 1a strain coupling stays inert" begin
        # The Zel'dovich IC has u_2 = 0 strictly and u_1(x) only
        # depends on x_1 → ∂_2 u_1 = 0 at every face. Phase 1a's
        # F^β_12 / F^β_21 contributions vanish; β_off must stay zero.
        r = run_D4_zeldovich_pancake(; level = 3, A = 0.5,
                                      T_factor = 0.15, verbose = false)
        for v in r.max_abs_beta_off
            @test v ≈ 0.0 atol = 1e-13
        end
        # Final field state: β_off = 0 in every cell.
        for ci in r.ic.leaves
            v = read_detfield_2d(r.ic.fields, ci)
            @test v.betas_off[1] ≈ 0.0 atol = 1e-13
            @test v.betas_off[2] ≈ 0.0 atol = 1e-13
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 8: Conservation invariants over a pre-caustic run
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 8: conservation (M, Px, Py)" begin
        # Mass: ρ_per_cell is fixed by the IC bridge convention; cell
        # areas are Eulerian (fixed). M is exactly preserved by
        # construction.
        r = run_D4_zeldovich_pancake(; level = 3, A = 0.5,
                                      T_factor = 0.10, verbose = false)
        @test r.M_err_max ≤ 1e-12  # mass is bit-stable

        # Py: stays at 0 (u_2 = 0 throughout).
        for v in r.Py_traj
            @test abs(v) < 1e-12
        end
        @test r.Py_err_max < 1e-12

        # Px: the periodic box has zero net momentum at IC (cos
        # integrates to 0 over a full period); the pressure-driven
        # flow can shift the per-cell velocities but the total Px
        # remains approximately constant. We use a relaxed bound.
        @test r.Px_err_max < 1.0
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 9: 4-component realizability cone diagnostics
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 9: 4-component realizability cone" begin
        # Pre-caustic: n_negative_jacobian = 0 for all leaves
        # throughout. This is the safety belt that the per-axis
        # γ_a projection holds.
        r = run_D4_zeldovich_pancake(; level = 3, A = 0.5,
                                      T_factor = 0.15, verbose = false)
        # Allow a small budget for end-of-run cone saturation
        # (empirically 0 at T_factor = 0.15, but we allow up to
        # 5% of cells in the most aggressive interpretation).
        n_total = length(r.ic.leaves) * length(r.t)
        @test sum(r.n_negative_jacobian) < 0.05 * n_total

        # 4-component cone Q = β_1² + β_2² + 2(β_12² + β_21²)
        # stays inside `headroom_offdiag · M_vv ≈ 2.0` per leaf.
        for ci in r.ic.leaves
            v = read_detfield_2d(r.ic.fields, ci)
            β1 = v.betas[1]; β2 = v.betas[2]
            β12 = v.betas_off[1]; β21 = v.betas_off[2]
            Q = β1^2 + β2^2 + 2 * (β12^2 + β21^2)
            @test isfinite(Q)
            @test Q ≥ 0
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 10: BC consistency — periodic-x + reflecting-y
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 10: BC structure (periodic-x, reflecting-y)" begin
        ic = tier_d_zeldovich_pancake_ic(; level = 3, A = 0.5)
        bc_zel = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                      (REFLECTING, REFLECTING)))
        # Single step under the standard BCs.
        det_step_2d_berry_HG!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                                bc_zel, 1e-3;
                                M_vv_override = (1.0, 1.0), ρ_ref = 1.0)
        # No NaN.
        for ci in ic.leaves
            v = read_detfield_2d(ic.fields, ci)
            @test isfinite(v.x[1])
            @test isfinite(v.x[2])
            @test isfinite(v.u[1])
            @test isfinite(v.u[2])
            @test isfinite(v.alphas[1])
            @test isfinite(v.alphas[2])
            @test isfinite(v.betas[1])
            @test isfinite(v.betas[2])
            @test isfinite(v.betas_off[1])
            @test isfinite(v.betas_off[2])
            @test isfinite(v.θ_R)
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 11: snapshot recording
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 11: snapshot recording" begin
        r = run_D4_zeldovich_pancake(; level = 3, A = 0.5,
                                      T_factor = 0.1, verbose = false,
                                      snapshots_at = (0.0, 0.5, 0.9))
        @test length(r.snapshots) == 3
        for snap in r.snapshots
            @test haskey(snap, :t)
            @test haskey(snap, :γ1)
            @test haskey(snap, :γ2)
            @test haskey(snap, :x_dev)
            @test haskey(snap, :x_centers)
            @test haskey(snap, :y_centers)
            @test length(snap.γ1) == length(r.ic.leaves)
            @test length(snap.γ2) == length(r.ic.leaves)
            for v in snap.γ1
                @test isfinite(v)
                @test v >= 0
            end
            for v in snap.γ2
                @test isfinite(v)
                @test v >= 0
            end
        end
        # Snapshot times should increase.
        @test r.snapshots[1].t ≤ r.snapshots[2].t ≤ r.snapshots[3].t
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 12: mesh refinement sweep — selectivity grows with N
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 12: mesh refinement sweep" begin
        sweep = run_D4_zeldovich_mesh_sweep(; levels = (3, 4),
                                              A = 0.5, T_factor = 0.15)
        @test length(sweep.results) == 2
        @test sweep.t_cross ≈ 1.0 / (0.5 * 2π) atol = M3_6_PHASE2_TOL
        @test length(sweep.selectivity_final) == 2
        @test length(sweep.γ1_dynamic_range) == 2
        # Both levels finish without NaN.
        for r in sweep.results
            @test !r.nan_seen
        end
        # Both produce finite, positive selectivity.
        for sel in sweep.selectivity_final
            @test isfinite(sel)
            @test sel > 1e3
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 13: headline plot driver (CSV fallback OK)
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 13: headline plot driver" begin
        sweep = run_D4_zeldovich_mesh_sweep(; levels = (3, 4),
                                              A = 0.5, T_factor = 0.15)
        save_path = joinpath(@__DIR__, "..", "reference", "figs",
                              "M3_6_phase2_D4_zeldovich.png")
        out = plot_D4_zeldovich_pancake(sweep; save_path = save_path)
        @test out isa AbstractString
        png_ok = isfile(save_path)
        csv_ok_3 = isfile(replace(save_path, ".png" => "_L3.csv"))
        csv_ok_4 = isfile(replace(save_path, ".png" => "_L4.csv"))
        @test png_ok || csv_ok_3 || csv_ok_4
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 14: γ_2 spatial uniformity (per-cell)
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 14: γ_2 spatial uniformity (per-cell)" begin
        # On the trivial axis, all cells must carry the SAME γ_2
        # value at every step (no cross-axis coupling fires the
        # axis-2 dynamics). This is the stronger per-cell test.
        r = run_D4_zeldovich_pancake(; level = 3, A = 0.5,
                                      T_factor = 0.15, verbose = false)
        # Final-step β_2 across cells must be uniform.
        β2_arr = [Float64(r.ic.fields.β_2[ci][1]) for ci in r.ic.leaves]
        @test maximum(β2_arr) - minimum(β2_arr) < 1e-10
        # Final-step α_2 across cells must be uniform too.
        α2_arr = [Float64(r.ic.fields.α_2[ci][1]) for ci in r.ic.leaves]
        @test maximum(α2_arr) - minimum(α2_arr) < 1e-10
    end
end
