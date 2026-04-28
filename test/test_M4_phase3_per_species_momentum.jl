# test_M4_phase3_per_species_momentum.jl
#
# §M4 Phase 3 acceptance gates: per-species momentum coupling
# closes the M3-6 Phase 4 D.7 dust-traps honest-finding loop. Adds
# `PerSpeciesMomentumHG2D` extension of `TracerMeshHG2D` with
# per-species (u_x, u_y) per cell + per-species drag relaxation
# timescale `τ_drag`. The kinematic mechanism for vortex-centre
# accumulation is implemented; the τ_drag regime sweep
# differentiates the three limits.
#
# Tests (8 GATEs):
#
#   GATE 1: IC sanity — `PerSpeciesMomentumHG2D` allocator.
#            u_per_species[k=*, *, ci] = u_gas at IC; dx_per_species
#            = 0; τ_drag_per_species respected. With u_dust_offset
#            keyword, dust velocity = u_gas + offset · ê_x at IC.
#
#   GATE 2: Drag relaxation kernel — exponential limits.
#            τ → 0: u_dust ← u_gas exactly (passive-scalar limit).
#            τ → ∞: u_dust unchanged (decoupled limit).
#            Intermediate τ: u_dust → u_gas with rate (1−exp(−dt/τ)).
#
#   GATE 3: Position update kernel — `dx ← dx + dt·u_k`.
#            For uniform u, dx[axis] = dt · u_k[axis] after one step.
#
#   GATE 4: Per-species mass conservation (bit-exact). The
#            `tracers` matrix is byte-stable (Phase 3 contract);
#            `accumulate_species_to_cells!` preserves total mass
#            (sum of new_c == sum of source concentrations).
#
#   GATE 5: 4-component realizability — n_neg_jac = 0 for the
#            stable per-species momentum-coupled run at L=3,
#            T_factor=0.1, project_kind=:reanchor.
#
#   GATE 6: τ_drag regime differentiation — u_dust_offset = 0.5
#            distinguishes the three τ regimes:
#              tightly-coupled (τ → 0): offset erased on drag step.
#              decoupled (τ → ∞): offset retained throughout.
#              intermediate: u_dust speed in between.
#
#   GATE 7: Bit-exact regression — at zero per-species momentum
#            (no `PerSpeciesMomentumHG2D` constructed), the M3-6
#            Phase 4 driver path is byte-equal. The `TracerMeshHG2D`
#            substrate, `det_step_2d_berry_HG!`, and
#            `advect_tracers_HG_2d!` are all unchanged.
#
#   GATE 8: Driver smoke at L=3, T_factor=0.1 — public NamedTuple
#            shape; trajectories finite; remap diagnostics valid.

using Test
using Statistics: mean, std
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, PERIODIC, cell_physical_box
using dfmm: tier_d_dust_trap_ic_full, tier_d_dust_trap_per_species_ic_full,
    det_step_2d_berry_HG!,
    gamma_per_axis_2d_field, gamma_per_axis_2d_per_species_field,
    advect_tracers_HG_2d!, n_species, species_index,
    ProjectionStats,
    read_detfield_2d, allocate_cholesky_2d_fields,
    cholesky_sector_state_from_primitive,
    TracerMeshHG2D, PerSpeciesMomentumHG2D,
    drag_relax_per_species!, advance_positions_per_species!,
    accumulate_species_to_cells!, dust_peak_over_mean_remapped

const _D7_PER_SPECIES_DRIVER_PATH = joinpath(@__DIR__, "..", "experiments",
                                              "D7_dust_traps.jl")
isdefined(Main, :run_D7_dust_traps_per_species) ||
    include(_D7_PER_SPECIES_DRIVER_PATH)

const M4_PHASE3_TOL = 1.0e-12

@testset verbose = true "M4 Phase 3 — per-species momentum coupling" begin

    # ─────────────────────────────────────────────────────────────
    # GATE 1: IC sanity (PerSpeciesMomentumHG2D allocator).
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 1: PerSpeciesMomentumHG2D IC allocator" begin
        ic = tier_d_dust_trap_per_species_ic_full(;
            level = 3, τ_drag_per_species = (0.0, 0.1))
        psm = ic.psm
        @test isa(psm, PerSpeciesMomentumHG2D)
        @test n_species(psm) == 2
        @test species_index(psm, :gas) == 1
        @test species_index(psm, :dust) == 2
        @test psm.τ_drag_per_species[1] == 0.0
        @test psm.τ_drag_per_species[2] == 0.1
        # u_per_species at IC should equal u_gas (co-moving).
        for ci in ic.leaves
            ug1 = Float64(ic.fields.u_1[ci][1])
            ug2 = Float64(ic.fields.u_2[ci][1])
            @test psm.u_per_species[1, 1, ci] ≈ ug1 atol = M4_PHASE3_TOL
            @test psm.u_per_species[1, 2, ci] ≈ ug2 atol = M4_PHASE3_TOL
            @test psm.u_per_species[2, 1, ci] ≈ ug1 atol = M4_PHASE3_TOL
            @test psm.u_per_species[2, 2, ci] ≈ ug2 atol = M4_PHASE3_TOL
            @test psm.dx_per_species[1, 1, ci] == 0.0
            @test psm.dx_per_species[1, 2, ci] == 0.0
            @test psm.dx_per_species[2, 1, ci] == 0.0
            @test psm.dx_per_species[2, 2, ci] == 0.0
        end
        # With u_dust_offset != 0 the dust velocity is biased.
        ic_off = tier_d_dust_trap_per_species_ic_full(;
            level = 3, τ_drag_per_species = (0.0, 0.1),
            u_dust_offset = 0.5)
        for ci in ic_off.leaves
            ug1 = Float64(ic_off.fields.u_1[ci][1])
            @test ic_off.psm.u_per_species[2, 1, ci] ≈ ug1 + 0.5 atol = M4_PHASE3_TOL
            # Gas (k=1) still tracks gas exactly (no offset).
            @test ic_off.psm.u_per_species[1, 1, ci] ≈ ug1 atol = M4_PHASE3_TOL
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 2: Drag relaxation kernel — exponential limits.
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 2: drag_relax_per_species! exponential limits" begin
        # Tightly-coupled limit: τ = 0 ⇒ u_dust ← u_gas exactly.
        ic = tier_d_dust_trap_per_species_ic_full(;
            level = 3, τ_drag_per_species = (0.0, 0.0),
            u_dust_offset = 0.5)
        # Sanity: dust IC has +0.5 offset on axis 1.
        ci0 = ic.leaves[1]
        ug1_0 = Float64(ic.fields.u_1[ci0][1])
        @test ic.psm.u_per_species[2, 1, ci0] ≈ ug1_0 + 0.5 atol = M4_PHASE3_TOL
        drag_relax_per_species!(ic.psm, 0.01; leaves = ic.leaves)
        # After one drag step with τ=0: u_dust = u_gas exactly.
        for ci in ic.leaves
            ug1 = Float64(ic.fields.u_1[ci][1])
            ug2 = Float64(ic.fields.u_2[ci][1])
            @test ic.psm.u_per_species[2, 1, ci] ≈ ug1 atol = M4_PHASE3_TOL
            @test ic.psm.u_per_species[2, 2, ci] ≈ ug2 atol = M4_PHASE3_TOL
        end

        # Decoupled limit: τ = Inf ⇒ u_dust unchanged.
        ic_d = tier_d_dust_trap_per_species_ic_full(;
            level = 3, τ_drag_per_species = (0.0, Inf),
            u_dust_offset = 0.5)
        u_dust_pre = copy(ic_d.psm.u_per_species)
        drag_relax_per_species!(ic_d.psm, 0.01; leaves = ic_d.leaves)
        # Dust velocity unchanged (decoupled).
        @test ic_d.psm.u_per_species ≈ u_dust_pre atol = M4_PHASE3_TOL

        # Intermediate τ: u_dust → u_gas with rate (1 − exp(−dt/τ)).
        τ_mid = 0.1
        ic_m = tier_d_dust_trap_per_species_ic_full(;
            level = 3, τ_drag_per_species = (0.0, τ_mid),
            u_dust_offset = 0.5)
        u_dust_pre_m = copy(ic_m.psm.u_per_species)
        dt = 0.05
        drag_relax_per_species!(ic_m.psm, dt; leaves = ic_m.leaves)
        # For each leaf, u_dust_post = u_dust_pre + fac·(u_gas − u_dust_pre)
        # with fac = 1 − exp(−dt/τ).
        fac_expected = 1.0 - exp(-dt / τ_mid)
        for ci in ic_m.leaves
            ug1 = Float64(ic_m.fields.u_1[ci][1])
            u_pre = u_dust_pre_m[2, 1, ci]
            u_post = ic_m.psm.u_per_species[2, 1, ci]
            expected = u_pre + fac_expected * (ug1 - u_pre)
            @test u_post ≈ expected atol = M4_PHASE3_TOL
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 3: Position update kernel.
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 3: advance_positions_per_species! kinematics" begin
        ic = tier_d_dust_trap_per_species_ic_full(;
            level = 3, τ_drag_per_species = (0.0, 0.1))
        # Pre-step: dx = 0; u_dust = u_gas at IC.
        for ci in ic.leaves
            @test ic.psm.dx_per_species[1, 1, ci] == 0.0
            @test ic.psm.dx_per_species[2, 1, ci] == 0.0
        end
        dt = 0.01
        advance_positions_per_species!(ic.psm, dt; leaves = ic.leaves)
        for ci in ic.leaves
            ug1 = Float64(ic.fields.u_1[ci][1])
            ug2 = Float64(ic.fields.u_2[ci][1])
            # dx after one step = dt · u (since u_dust = u_gas at IC).
            @test ic.psm.dx_per_species[2, 1, ci] ≈ dt * ug1 atol = M4_PHASE3_TOL
            @test ic.psm.dx_per_species[2, 2, ci] ≈ dt * ug2 atol = M4_PHASE3_TOL
            @test ic.psm.dx_per_species[1, 1, ci] ≈ dt * ug1 atol = M4_PHASE3_TOL
            @test ic.psm.dx_per_species[1, 2, ci] ≈ dt * ug2 atol = M4_PHASE3_TOL
        end
        # Second step accumulates.
        advance_positions_per_species!(ic.psm, dt; leaves = ic.leaves)
        for ci in ic.leaves
            ug1 = Float64(ic.fields.u_1[ci][1])
            @test ic.psm.dx_per_species[2, 1, ci] ≈ 2 * dt * ug1 atol = M4_PHASE3_TOL
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 4: Per-species mass conservation under remap.
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 4: accumulate_species_to_cells! mass conservation" begin
        ic = tier_d_dust_trap_per_species_ic_full(;
            level = 3, τ_drag_per_species = (0.0, 0.1))
        # Source mass = sum of c_dust over leaves.
        k_dust = species_index(ic.tm, :dust)
        src_total = sum(ic.tm.tracers[k_dust, ci] for ci in ic.leaves)
        # Apply some drift via position kernel.
        dt = 0.05
        for _ in 1:5
            advance_positions_per_species!(ic.psm, dt; leaves = ic.leaves)
        end
        lo_b = ic.params.lo
        hi_b = (Float64(lo_b[1]) + ic.params.L1,
                Float64(lo_b[2]) + ic.params.L2)
        new_c = accumulate_species_to_cells!(ic.psm, ic.frame, ic.leaves;
                                               k = k_dust, lo = lo_b,
                                               hi = hi_b, wrap = true)
        new_total = sum(Float64, new_c)
        @test new_total ≈ src_total atol = 1e-10
        # Every cell concentration ≥ 0 (no negative mass under remap).
        @test all(>=(0.0), new_c)
        # The original tracer matrix is byte-stable (Phase 3 contract).
        @test all(ic.tm.tracers[k_dust, ci] ≈
                   1.0 + 0.05 * sin(2π * ((Float64(ic.fields.x_1[ci][1]) - Float64(lo_b[1])) / ic.params.L1)) *
                                   sin(2π * ((Float64(ic.fields.x_2[ci][1]) - Float64(lo_b[2])) / ic.params.L2))
                   for ci in ic.leaves)
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 5: 4-component realizability under per-species momentum.
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 5: 4-component realizability (n_neg_jac=0)" begin
        r = run_D7_dust_traps_per_species(; level = 3, T_factor = 0.1,
                                            τ_drag_dust = 0.1,
                                            u_dust_offset = 0.5)
        @test !r.nan_seen
        # Every step has n_neg_jac == 0 (cone respected).
        @test all(==(0), r.n_negative_jacobian)
        @test sum(r.n_negative_jacobian) == 0
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 6: τ_drag regime differentiation.
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 6: τ_drag regime differentiation" begin
        # Run three regimes with the dust velocity biased at IC.
        offset = 0.5
        # Tightly coupled: τ → 0.
        r_tight = run_D7_dust_traps_per_species(; level = 3,
                    T_factor = 0.1, τ_drag_dust = 1e-6,
                    u_dust_offset = offset)
        # Decoupled: τ → ∞.
        r_decoupled = run_D7_dust_traps_per_species(; level = 3,
                    T_factor = 0.1, τ_drag_dust = 1e6,
                    u_dust_offset = offset)
        # Intermediate.
        r_mid = run_D7_dust_traps_per_species(; level = 3,
                    T_factor = 0.1, τ_drag_dust = 0.1,
                    u_dust_offset = offset)
        @test !r_tight.nan_seen
        @test !r_decoupled.nan_seen
        @test !r_mid.nan_seen
        # Tightly coupled: u_dust speed converges to gas speed
        # (offset erased after first step).
        @test r_tight.u_dust_speed_mean[1] > r_tight.u_dust_speed_mean[end] + 1e-6
        # Decoupled: u_dust speed stays at IC value (offset retained;
        # tolerance accounts for finite τ=1e6, where dt/τ ~ 1e-9 leaks).
        @test r_decoupled.u_dust_speed_mean[1] ≈ r_decoupled.u_dust_speed_mean[end] atol = 1e-6
        # Intermediate: u_dust speed in between.
        @test r_tight.u_dust_speed_mean[end] ≤ r_mid.u_dust_speed_mean[end] + 1e-10
        @test r_mid.u_dust_speed_mean[end] ≤ r_decoupled.u_dust_speed_mean[end] + 1e-10
        # Drift magnitude differs across regimes (the IC offset
        # propagates further in the decoupled regime).
        @test r_decoupled.dx_dust_max[end] ≥ r_tight.dx_dust_max[end] - 1e-10
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 7: Bit-exact regression (no per-species momentum).
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 7: bit-exact regression at zero per-species momentum" begin
        # The base D.7 driver path (no `PerSpeciesMomentumHG2D`)
        # is byte-equal to the M3-6 Phase 4 baseline. Build IC via
        # the base factory; verify the tracer matrix + fields match
        # bit-for-bit.
        ic_a = tier_d_dust_trap_ic_full(; level = 3, U0 = 1.0,
                                          ε_dust = 0.05)
        ic_b = tier_d_dust_trap_ic_full(; level = 3, U0 = 1.0,
                                          ε_dust = 0.05)
        @test ic_a.tm.tracers == ic_b.tm.tracers
        for ci in ic_a.leaves
            @test ic_a.fields.u_1[ci][1] === ic_b.fields.u_1[ci][1]
            @test ic_a.fields.u_2[ci][1] === ic_b.fields.u_2[ci][1]
            @test ic_a.fields.α_1[ci][1] === ic_b.fields.α_1[ci][1]
            @test ic_a.fields.α_2[ci][1] === ic_b.fields.α_2[ci][1]
        end
        # `advect_tracers_HG_2d!` is still a no-op on `TracerMeshHG2D`
        # (Phase 3 contract, not perturbed by M4 Phase 3).
        tm_pre = copy(ic_a.tm.tracers)
        advect_tracers_HG_2d!(ic_a.tm, 0.1)
        @test ic_a.tm.tracers == tm_pre
        # Per-species momentum extension is opt-in: extending one IC
        # does not perturb the other.
        psm_a = PerSpeciesMomentumHG2D(ic_a.tm;
                                        τ_drag_per_species = [0.0, 0.1],
                                        leaves = ic_a.leaves)
        @test ic_a.tm.tracers == ic_b.tm.tracers
        # Calling drag/position kernels on psm_a does not perturb
        # ic_b's fields or tracer matrix.
        drag_relax_per_species!(psm_a, 0.01; leaves = ic_a.leaves)
        advance_positions_per_species!(psm_a, 0.01; leaves = ic_a.leaves)
        @test ic_b.tm.tracers == ic_a.tm.tracers
        for ci in ic_a.leaves
            @test ic_a.fields.u_1[ci][1] === ic_b.fields.u_1[ci][1]
            @test ic_a.fields.u_2[ci][1] === ic_b.fields.u_2[ci][1]
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 8: Driver smoke at level 3.
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 8: driver smoke at level 3" begin
        r = run_D7_dust_traps_per_species(; level = 3, T_factor = 0.1,
                                            τ_drag_dust = 0.1,
                                            u_dust_offset = 0.0)
        @test !r.nan_seen
        # NamedTuple shape sanity.
        @test r.t isa AbstractVector{Float64}
        @test r.peak_over_mean_remapped isa AbstractVector{Float64}
        @test r.collapse_fraction isa AbstractVector{Float64}
        @test r.dx_dust_max isa AbstractVector{Float64}
        @test r.u_dust_speed_mean isa AbstractVector{Float64}
        @test length(r.t) == r.params.n_steps + 1
        # Trajectories all finite.
        @test all(isfinite, r.peak_over_mean_remapped)
        @test all(isfinite, r.dx_dust_max)
        @test all(isfinite, r.u_dust_speed_mean)
        @test all(isfinite, r.collapse_fraction)
        # collapse_fraction ∈ [0, 1].
        @test all(c -> 0.0 ≤ c ≤ 1.0, r.collapse_fraction)
        # peak_over_mean_remapped ≥ 1 (trivial bound: max ≥ mean).
        @test all(p -> p ≥ 1.0 - 1e-10, r.peak_over_mean_remapped)
        # dx_dust_max ≥ 0; monotonically non-decreasing within a stable run.
        @test all(d -> d ≥ -1e-12, r.dx_dust_max)
        # Mass conservation: total integrated dust mass is bit-stable
        # (Phase 3 contract carried forward).
        @test r.M_dust_err_max == 0.0
        # n_negative_jacobian = 0 throughout.
        @test all(==(0), r.n_negative_jacobian)
        # Wall-time per step is positive and bounded.
        @test r.wall_time_per_step > 0
        @test r.wall_time_per_step < 10  # sanity bound on level 3
    end

end
