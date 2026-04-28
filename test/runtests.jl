using dfmm
using Test

@testset verbose = true "dfmm" begin
    @testset verbose = true "Phase 1: zero-strain" begin
        include("test_phase1_zero_strain.jl")
    end
    @testset verbose = true "Phase 1: uniform-strain" begin
        include("test_phase1_uniform_strain.jl")
    end
    @testset verbose = true "Phase 1: symplectic" begin
        include("test_phase1_symplectic.jl")
    end
    @testset verbose = true "Phase 2" begin
        @testset "Phase 2: mass" begin
            include("test_phase2_mass.jl")
        end
        @testset "Phase 2: momentum" begin
            include("test_phase2_momentum.jl")
        end
        @testset "Phase 2: free streaming" begin
            include("test_phase2_free_streaming.jl")
        end
        @testset "Phase 2: acoustic" begin
            include("test_phase2_acoustic.jl")
        end
    end
    @testset verbose = true "Phase 3" begin
        # Tier B.2 cold-limit reduction — the central unification test
        # of the methods paper. Order: Zel'dovich pre-crossing match,
        # then Hessian-degeneracy diagnostic (also writes the
        # `reference/figs/phase3_hessian_degen.png` plot), then energy /
        # mass / momentum conservation.
        @testset "Phase 3: Zel'dovich pre-crossing" begin
            include("test_phase3_zeldovich.jl")
        end
        @testset "Phase 3: Hessian degeneracy" begin
            include("test_phase3_hessian_degen.jl")
        end
        @testset "Phase 3: energy / mass / momentum drift" begin
            include("test_phase3_energy_drift.jl")
        end
    end
    @testset verbose = true "Phase 4: energy drift" begin
        include("test_phase4_energy_drift.jl")
    end
    @testset verbose = true "Phase 5: Sod" begin
        # Tier A.1 Sod regression vs py-1d golden. Adds the deviatoric
        # P_⊥ sector with BGK relaxation and asserts L∞ rel error < 0.05
        # on (rho, u, Pxx, Pp) at t_end = 0.2, τ = 1e-3 (warm regime).
        include("test_phase5_sod_regression.jl")
    end
    @testset verbose = true "Phase 5b: tensor-q" begin
        # Opt-in artificial viscosity (Kuropatenko / vNR) added to the
        # variational integrator. Default `q_kind = :none` reproduces
        # Phase 5 bit-equally; with `q_kind = :vNR_linear_quadratic`
        # the Sod L∞ rel error drops below 0.10 on (ρ, Pxx, Pp).
        include("test_phase5b_artificial_viscosity.jl")
    end
    @testset verbose = true "Phase 6: cold-sinusoid τ-scan" begin
        # Tier A.2 cold-sinusoid τ-scan. Generalizes the Phase-3
        # single-τ Zel'dovich match across six τ decades (10⁻³…10⁷),
        # asserts pre-crossing density vs Zel'dovich to L∞ rel < 1e-3
        # per τ, γ stays at machine-noise scale, conservation invariants
        # hold, and renders the headline 6-panel τ-scan plot
        # (`reference/figs/A2_cold_sinusoid_tauscan.png`). The post-
        # crossing golden match is `@test_skip`'d pending Phase-5b
        # shock-capturing — see `reference/notes_phase6_cold_sinusoid.md`.
        include("test_phase6_cold_sinusoid_scan.jl")
    end
    @testset verbose = true "Phase 7: steady shock" begin
        # Tier A.3 steady-shock Mach scan + heat-flux Lagrange
        # multiplier. Asserts:
        #  (a) The closed-form heat-flux BGK primitives
        #      (`heat_flux_bgk_step`, `heat_flux_step`) match the
        #      bit-equal exponential decay py-1d uses.
        #  (b) `setup_steady_shock` builds an IC whose downstream
        #      state matches analytical Rankine-Hugoniot to 3
        #      decimal places (methods paper §10.2 A.3) for
        #      M1 ∈ {1.5, 2, 3, 5, 10}.
        #  (c) `Mesh1D` and `DetField` accept the new `bc` and `Q`
        #      fields without breaking Phase 1–6 paths.
        #  (d) `det_step!` advances `Q` via BGK (exponential decay
        #      to <1e-10 over 10 steps on a smooth mesh).
        #  (e) Short-horizon (t = 0.005, ~30 steps at N = 80)
        #      preservation of the downstream R-H plateau by the
        #      variational integrator + Phase-5b artificial
        #      viscosity for M1 = 3. The long-horizon golden match
        #      is gated on a future shock-capturing extension —
        #      see `reference/notes_phase7_steady_shock.md`.
        include("test_phase7_steady_shock.jl")
    end
    @testset "eos" begin
        include("test_eos.jl")
    end
    @testset "diagnostics" begin
        include("test_diagnostics.jl")
    end
    @testset "io" begin
        include("test_io.jl")
    end
    @testset "calibration" begin
        include("test_calibration.jl")
    end
    @testset "plotting" begin
        include("test_plotting.jl")
    end
    @testset "Track D — Stochastic dressing primitives" begin
        include("test_vg_sampler.jl")
        include("test_burst_stats.jl")
        include("test_self_consistency.jl")
    end
    @testset verbose=true "Phase 8: stochastic injection" begin
        # Phase 8: variance-gamma noise post-Newton injection +
        # Phase 9 / Tier B.4 burst-stats self-consistency monitor.
        # See `reference/notes_phase8_stochastic_injection.md` and
        # `experiments/B4_compression_bursts.jl`.
        include("test_phase8_stochastic_injection.jl")
    end
    @testset verbose=true "Phase M2-3: realizability projection" begin
        # M2-3: closes M1 Open #4 — the long-time stochastic
        # realizability instability. Adds `realizability_project!` to
        # the post-Newton injection so wave-pool runs reach 10⁴+ steps
        # under production calibration. See
        # `reference/notes_M2_3_realizability.md` and
        # `experiments/M2_3_long_time_stochastic.jl`.
        include("test_phase_M2_3_realizability.jl")
    end
    @testset "setups" begin
        include("test_setups.jl")
    end
    @testset "regression scaffold" begin
        include("test_regression_scaffold.jl")
    end
    @testset verbose = true "Phase 11: passive tracer" begin
        # Tier B.5 — passive scalar advection: bit-exact tracer
        # preservation in pure-Lagrangian regions, multi-tracer
        # support, and decade-level fidelity vs a reference Eulerian
        # upwind scheme. See `reference/notes_phase11_passive_tracer.md`.
        include("test_phase11_tracer_advection.jl")
    end
    @testset verbose=true "Phase M2-2: multi-tracer wave-pool" begin
        # Tier B.6 — multi-tracer fidelity in 1D wave-pool turbulence
        # with Phase-8 stochastic injection enabled. Verifies that
        # Phase-11's bit-exact tracer-preservation property holds
        # even when stochastic noise is active (the noise mutates
        # ρu, P_xx, P_⊥, s — never the tracer matrix). See
        # `reference/notes_M2_2_multitracer.md`.
        include("test_phase_M2_2_multitracer.jl")
    end
    @testset verbose = true "Phase M2-1: action-based AMR" begin
        # Tier B.3 — 1D action-based AMR vs gradient-based AMR on
        # the off-center blast. Verifies refine/coarsen primitives
        # (mass / momentum / tracer conservation), the action-error
        # and gradient indicators, and the headline 20-50% cell-
        # count reduction at matched L² accuracy.
        # See `reference/notes_M2_1_amr.md` and
        # `experiments/B3_action_amr.jl`.
        include("test_phase_M2_1_amr.jl")
    end
    @testset verbose = true "Cross-phase smoke (defensive integration)" begin
        # Catches inter-phase regressions that per-phase tests miss
        # — e.g. struct extension without updating mutation call sites.
        # See test/test_integration_all_phases.jl docstring.
        include("test_integration_all_phases.jl")
    end
    @testset verbose = true "Phase M3-0: HG + R3D integration" begin
        # Foundation phase for the M3 (multi-D) refactor. Confirms HG
        # and R3D import cleanly and the HG-substrate-based Phase-1
        # Cholesky-sector path is bit-exactly equivalent to M1's
        # `Mesh1D`+`Segment` path. See
        # `reference/notes_M3_0_hg_integration.md` and
        # `reference/MILESTONE_3_PLAN.md` Phase M3-0.
        include("test_M3_0_smoke.jl")
        include("test_M3_0_parity_1D.jl")
    end
    @testset verbose = true "M3-prep: Tier-C IC factories" begin
        # Setup-only tests for the Tier-C dimension-generic IC
        # factories `tier_c_sod_ic`, `tier_c_cold_sinusoid_ic`,
        # `tier_c_plane_wave_ic`. These run in advance of the M3-4
        # solver-coupled C.1/C.2/C.3 consistency tests; they only
        # verify the IC field-set values (mass conservation,
        # y-direction independence, rotational invariance, sample
        # primitive points). See
        # `reference/notes_M3_prep_tierC_ic_factories.md`.
        include("test_M3_prep_setups_tierC.jl")
    end
    @testset verbose = true "Phase M3-1: Phase 2 + 5 + 5b on HG" begin
        # Phase-2 (bulk + entropy + multi-segment periodic mesh),
        # Phase-5 (deviatoric P_⊥ via post-Newton BGK), and Phase-5b
        # (opt-in Kuropatenko / vNR tensor-q) ports to the HG
        # substrate. The HG-side driver delegates to M1's `det_step!`
        # for bit-exact parity. See
        # `reference/notes_M3_1_phase2_5_5b_port.md` and
        # `reference/MILESTONE_3_PLAN.md` Phase M3-1.
        @testset "Phase 2 (HG): mass" begin
            include("test_M3_1_phase2_mass_HG.jl")
        end
        @testset "Phase 2 (HG): momentum" begin
            include("test_M3_1_phase2_momentum_HG.jl")
        end
        @testset "Phase 2 (HG): free streaming" begin
            include("test_M3_1_phase2_free_streaming_HG.jl")
        end
        @testset "Phase 2 (HG): acoustic" begin
            include("test_M3_1_phase2_acoustic_HG.jl")
        end
        @testset "Phase 5 (HG): Sod regression" begin
            include("test_M3_1_phase5_sod_HG.jl")
        end
        @testset "Phase 5b (HG): q_kind=:none bit-equality" begin
            include("test_M3_1_phase5b_qnone_bit_equal.jl")
        end
        @testset "Phase 5b (HG): q_kind=:vNR Sod" begin
            include("test_M3_1_phase5b_qvnr_sod.jl")
        end
        @testset "Phase 5b (HG): compute_q formula" begin
            include("test_M3_1_phase5b_qformula_unit.jl")
        end
    end
    @testset verbose = true "Phase M3-prep: Berry-connection stencils" begin
        # Tier-1 building blocks for M3-3 (2D Cholesky + Berry connection).
        # Pure-functional pre-compute of the verified symbolic forms
        #   Θ_rot^(2D) = (1/3)(α_1³ β_2 − α_2³ β_1) dθ_R
        #   Θ_rot^(3D) = (1/3) Σ_{a<b}(α_a³ β_b − α_b³ β_a) dθ_{ab}
        #   θ_offdiag  = -(1/2)(α_1² α_2 dβ_{21} + α_1 α_2² dβ_{12}).
        # Cross-checked against `scripts/verify_berry_connection*.py`.
        # See `reference/notes_M3_prep_berry_stencil.md` for the API
        # and `src/berry.jl` for the implementation.
        include("test_M3_prep_berry_stencil.jl")
    end
    @testset verbose = true "Phase M3-prep: 3D Berry verification (M3-7 pre-flight)" begin
        # Pre-flight gate for Milestone M3-7 (3D extension): the 3D
        # SO(3) Berry-connection stencils in `src/berry.jl`
        # (`berry_F_3d`, `berry_term_3d`, `berry_partials_3d`,
        # `BerryStencil3D`) are pinned numerically against the SymPy
        # authority `scripts/verify_berry_connection_3D.py` at random
        # sample points to round-off precision (most asserts at 0.0
        # absolute since only polynomial arithmetic is involved).
        # Reproduces all 8 SymPy CHECKs:
        #   CHECK 1: closedness dΩ = 0 (84 cyclic triples in 9D);
        #   CHECK 2: per-axis Hamilton eqs on slice θ̇_ab = 0;
        #   CHECK 3a: full iso pullback (Berry blocks vanish);
        #   CHECK 3b: 5×5 sub-block matches 2D Ω;
        #   CHECK 4: rank-8 + closed-form 1D kernel direction;
        #   CHECK 5: F_ab antisymmetry under axis swap;
        #   CHECK 6: degeneracy boundary α_a = α_b;
        #   CHECK 7: Ω = dΘ exactness (no monopole/Chern class);
        #   CHECK 8: cyclic-Bianchi-like polynomial relation.
        # Plus a stencil-internal-consistency smoke test for M3-7's
        # eventual 3D EL residual integration. See
        # `reference/notes_M3_prep_3D_berry_verification.md`.
        include("test_M3_prep_3D_berry_verification.jl")
    end
    @testset verbose = true "Phase M3-3a: HaloView smoke + 2D field set + Cholesky DD" begin
        # M3-3a sub-phase of M3-3 (2D Cholesky + Berry connection).
        # Three tests:
        #   • HaloView coefficient access for an order-0 MonomialBasis{2, 0}
        #     field on an 8×8 balanced 2D HierarchicalMesh.
        #   • Per-axis Cholesky decomposition driver in `src/cholesky_DD.jl`
        #     (decompose ↔ recompose round-trip; per-axis γ diagnostic).
        #   • 2D 10-dof field-set allocation + read/write round-trip.
        # See `reference/notes_M3_3_2d_cholesky_berry.md` §9 (sub-phase
        # split) and `reference/notes_M3_3a_field_set_cholesky.md`
        # (this sub-phase's status note).
        include("test_M3_3a_halo_smoke.jl")
        include("test_M3_3a_cholesky_DD.jl")
        include("test_M3_3a_field_set_2d.jl")
    end
    @testset verbose = true "Phase M3-3b: native 2D EL residual (no Berry; θ_R fixed)" begin
        # M3-3b sub-phase of M3-3 (2D Cholesky + Berry connection).
        # First native HG-side EL residual on the 2D substrate. The
        # residual is the per-axis lift of M1's 1D `det_el_residual`
        # WITHOUT Berry coupling and WITHOUT θ_R as a Newton unknown
        # (M3-3c lands those). Two test files:
        #   • Zero-strain regression: cold-limit fixed-point IC,
        #     residual = 0 to machine precision; one Newton step
        #     preserves the state byte-equally; pack/unpack round-trip.
        #   • Dimension-lift parity gate (§6.1 — the critical M3-3b
        #     acceptance criterion): 2D 1D-symmetric configuration
        #     reproduces M1's Phase-1 zero-strain trajectory to ≤ 1e-12.
        # See `reference/notes_M3_3_2d_cholesky_berry.md` §3 + §6.1
        # and `reference/notes_M3_3b_native_residual.md`.
        include("test_M3_3b_2d_zero_strain.jl")
        include("test_M3_3b_dimension_lift_zero_strain.jl")
    end
    @testset verbose = true "Phase M3-3c: Berry coupling + θ_R Newton unknown" begin
        # M3-3c sub-phase of M3-3 (2D Cholesky + Berry connection).
        # Promotes θ_R from a fixed IC value to a Newton unknown
        # (8N → 9N rows). Adds the closed-form Berry partials from
        # `src/berry.jl` into the per-axis residual rows; F^θ_R is
        # the kinematic equation (trivial drive in M3-3c — off-diagonal
        # velocity gradients enter at M3-3d/M3-6). Four tests:
        #   • Dimension-lift gate re-verification (§6.1): Berry vanishes
        #     on the 1D-symmetric slice; M1 parity ≤ 1e-12.
        #   • Berry verification reproduction (§6.2): residual partials
        #     match `berry_partials_2d` at 5–10 random samples.
        #   • Iso-pullback ε-expansion (§6.3): Berry contribution is
        #     O(ε²) along the iso-submanifold tangent direction.
        #   • H_rot solvability (§6.4): closed-form ∂H_rot/∂θ_R from
        #     §6.6 satisfies kernel-orthogonality at 5 generic samples;
        #     Newton converges ≤ 5 iterations on non-isotropic 2D ICs.
        # See `reference/notes_M3_3_2d_cholesky_berry.md` §4 + §6 and
        # `reference/notes_M3_3c_berry_integration.md` (this sub-phase's
        # status note).
        include("test_M3_3c_dimension_lift_with_berry.jl")
        include("test_M3_3c_berry_residual.jl")
        include("test_M3_3c_iso_pullback.jl")
        include("test_M3_3c_h_rot_solvability.jl")
    end
    @testset verbose = true "Phase M3-3d: per-axis γ + AMR/realizability per-axis" begin
        # M3-3d sub-phase of M3-3 (2D Cholesky + Berry connection).
        # Wires the per-axis γ diagnostic (`gamma_per_axis_2d_field` /
        # `gamma_per_axis_2d_diag`) through the diagnostics + I/O layer,
        # adds the per-axis action-AMR indicator on `HierarchicalMesh{2}`
        # plus the HG `register_refinement_listener!` field-set listener
        # (closing M3-2b's deferred Swaps 2+3 for the 2D scope), and
        # extends the M2-3 realizability projection to per-axis 2D.
        # The headline scientific gate is §6.5 per-axis γ selectivity:
        # cold sinusoid IC with k_y = 0 ⇒ γ_1 collapses spatially while
        # γ_2 stays uniform (per-axis decomposition correctly identifies
        # the collapsing axis). See
        # `reference/notes_M3_3_2d_cholesky_berry.md` §4.3 + §6.5 and
        # `reference/notes_M3_3d_per_axis_gamma_amr.md` (this sub-phase's
        # status note).
        include("test_M3_3d_gamma_per_axis_diag.jl")
        include("test_M3_3d_realizability_per_axis.jl")
        include("test_M3_3d_amr_per_axis.jl")
        include("test_M3_3d_selectivity.jl")
    end
    @testset verbose = true "Phase M3-2: Phase 7/8/11 + M2 on HG" begin
        # Phase 7 (heat-flux Q + steady shock + inflow/outflow), Phase 8
        # (variance-gamma stochastic injection), Phase 11 (passive
        # tracers), and M2-1 (action-AMR) + M2-3 (realizability
        # projection) sub-phases on the HG substrate. M3-3e retired the
        # `cache_mesh::Mesh1D` shim across all paths; these tests now
        # gate the native HG-side implementations. See
        # `reference/notes_M3_2_phase7811_m2_port.md`,
        # `reference/notes_M3_3e_5_cache_mesh_dropped.md`, and
        # `reference/MILESTONE_3_PLAN.md` Phase M3-2 / M3-3.
        @testset "Phase 7 (HG): steady-shock + heat flux Q" begin
            include("test_M3_2_phase7_steady_shock_HG.jl")
        end
        @testset "Phase 8 (HG): stochastic injection" begin
            include("test_M3_2_phase8_stochastic_HG.jl")
        end
        @testset "Phase 11 (HG): tracer advection" begin
            include("test_M3_2_phase11_tracer_HG.jl")
        end
        @testset "M2-2 (HG): multi-tracer wave-pool" begin
            include("test_M3_2_M2_2_multitracer_HG.jl")
        end
        @testset "M2-1 (HG): action-based AMR" begin
            include("test_M3_2_M2_1_amr_HG.jl")
        end
        @testset "M2-3 (HG): realizability projection" begin
            include("test_M3_2_M2_3_realizability_HG.jl")
        end
    end
    @testset verbose = true "Phase M3-2b: HG swap-in (sparsity + BCKind)" begin
        # M3-2b Swap 6 — `det_jac_sparsity_HG` against M1's hand-rolled
        # sparsity (parity gate on N = 16 periodic mesh).
        # M3-2b Swap 8 — `BCKind` + `FrameBoundaries{1}` attached to
        # `DetMeshHG`, retiring the `bc, inflow_state, outflow_state, n_pin`
        # kwargs on `det_step_HG!`. See
        # `reference/notes_M3_2b_swaps68_sparsity_bc.md`.
        include("test_M3_2b_swap6_sparsity_HG.jl")
        include("test_M3_2b_swap8_bckind_HG.jl")
    end
    @testset verbose = true "Phase M3-3e-1: native det_step_HG! vs cache_mesh" begin
        # M3-3e-1 defensive cross-check: the native deterministic-Newton
        # path must produce byte-equal state to running M1's `det_step!`
        # on the cached `Mesh1D` shim. After M3-3e-5 drops the cache_mesh
        # field this test will be retired (it relies on directly running
        # the M1 baseline alongside). See
        # `reference/notes_M3_3e_1_det_step_native.md`.
        include("test_M3_3e_1_native_vs_cache.jl")
    end
    @testset verbose = true "Phase M3-3e-2: native stochastic injection vs cache_mesh" begin
        # M3-3e-2 defensive cross-check: native `inject_vg_noise_HG!`
        # and `det_run_stochastic_HG!` must produce byte-equal state to
        # M1's `inject_vg_noise!` / `det_run_stochastic!` on a parallel
        # `Mesh1D`. RNG sequencing gate at K = 10 steps. After M3-3e-5
        # drops the cache_mesh field this test will be retired. See
        # `reference/notes_M3_3e_2_stochastic_native.md`.
        include("test_M3_3e_2_stochastic_native_vs_cache.jl")
    end
    @testset verbose = true "Phase M3-3e-3: native AMR + TracerMeshHG vs cache_mesh" begin
        # M3-3e-3 defensive cross-check: native `refine_segment_HG!`,
        # `coarsen_segment_pair_HG!`, `amr_step_HG!`, and the standalone
        # `TracerMeshHG` storage must produce byte-equal state to running
        # M1's AMR primitives + `TracerMesh` on a parallel `Mesh1D`. K =
        # 10 refine + K = 10 coarsen + 5 (det_step + amr_step) cycles +
        # 6 mixed refine/coarsen with 3 tracers. After M3-3e-5 drops the
        # cache_mesh field this test will be retired. See
        # `reference/notes_M3_3e_3_amr_tracers_native.md`.
        include("test_M3_3e_3_amr_tracer_native_vs_cache.jl")
    end
    @testset verbose = true "Phase M3-3e-4: native realizability_project_HG! vs cache_mesh" begin
        # M3-3e-4 defensive cross-check: native `realizability_project_HG!`
        # must produce byte-equal state to running M1's
        # `realizability_project!` on a parallel `Mesh1D`. Per-cell projection
        # has no inter-cell coupling so the lift is mechanical; this test
        # captures the assumption explicitly. After M3-3e-5 drops the
        # cache_mesh field this test will be retired. See
        # `reference/notes_M3_3e_4_realizability_native.md`.
        include("test_M3_3e_4_realizability_native_vs_cache.jl")
    end
    @testset verbose = true "Phase M3-4: periodic-x wrap on 2D EL residual" begin
        # M3-4 prerequisite: closes the M3-3c handoff item "the periodic-x
        # coordinate wrap for active strain is a noted M3-3c handoff
        # item". Adds `build_periodic_wrap_tables` and threads per-axis-
        # per-cell wrap offsets into both `cholesky_el_residual_2D!` and
        # `cholesky_el_residual_2D_berry!`. The 2D residual now correctly
        # handles periodic boundaries on active / advecting flows,
        # mirroring the 1D `+L_box` wrap in
        # `cholesky_sector.jl::det_el_residual`. Tests verify cell-extent
        # positivity at the seam, REFLECTING/PERIODIC mix correctness,
        # and translation equivariance. See
        # `reference/notes_M3_4_tier_c_consistency.md`.
        include("test_M3_4_periodic_wrap.jl")
    end
    @testset verbose = true "Phase M3-5: Bayesian L↔E remap" begin
        # M3-5 wires HG's `compute_overlap` + polynomial-remap kernels
        # into a dfmm-side per-step driver via `BayesianRemapState{D, T}`,
        # `bayesian_remap_l_to_e!` / `bayesian_remap_e_to_l!`,
        # `remap_round_trip!`, and the
        # `liouville_monotone_increase_diagnostic` helper.
        # Three test blocks:
        #   • Conservation regression: L→E→L round trip preserves
        #     mass-weighted totals (mass / momentum / energy /
        #     tracer mass) to 1e-12 on `:float`; the `:exact`
        #     backend matches to documented HG-caveat magnitudes
        #     (16-bit lattice; ≤ 5 % drift on dfmm benign configs;
        #     ~0 drift on uniform fields with axis-aligned
        #     triangulation).
        #   • Liouville monotone-increase diagnostic: HG's
        #     `RemapDiagnostics` proxy (overlap volume / source
        #     physical volume) is positive, balanced, and zero
        #     `n_negative_jacobian_cells`; partition-of-unity holds
        #     across deformation cycles to 1e-12.
        #   • IntExact audit harness: HG's `audit_overlap` canonical
        #     2D + 3D polytope battery (5 + 4 = 9 polytopes) passes
        #     at atol = 1e-12; `:float` vs `:exact` `compute_overlap`
        #     on the identity-overlap dfmm 4×4 setup matches box
        #     volume to ~1e-3 on the default 16-bit lattice.
        # See `reference/notes_M3_5_bayesian_remap.md`.
        include("test_M3_5_remap_conservation.jl")
        include("test_M3_5_liouville_monotone.jl")
        include("test_M3_5_intexact_audit.jl")
    end
    @testset verbose = true "Phase M3-4 Phase 2: Tier-C IC bridge" begin
        # M3-4 Phase 2 (a): IC bridge from primitive (ρ, u_x, u_y, P) onto
        # the M3-3 12-field Cholesky-sector state, plus the inverse
        # primitive-recovery diagnostic. See
        # `reference/notes_M3_4_tier_c_consistency.md`
        # §"Pre-Tier-C handoff items" deliverables 1+2.
        include("test_M3_4_ic_bridge.jl")
    end
    @testset verbose = true "Phase M3-4 Phase 2 (b): C.1 1D-symmetric 2D Sod" begin
        # M3-4 Phase 2 (b): C.1 1D-symmetric 2D Sod acceptance driver.
        # y-independence ≤ 1e-12 at every step + conservation gates.
        # 1D-reduction-vs-golden gate is captured at the loose tolerance
        # documented in MILESTONE_3_STATUS Open Issue #2 (Sod L∞ ~10-20%).
        include("test_M3_4_C1_sod.jl")
    end
    @testset verbose = true "Phase M3-4 Phase 2 (c): C.2 2D cold sinusoid" begin
        # M3-4 Phase 2 (c): C.2 2D cold sinusoid acceptance driver.
        # Per-axis γ selectivity for k = (1, 0); genuine 2D structure for
        # k = (1, 1); conservation gates.
        include("test_M3_4_C2_cold_sinusoid.jl")
    end
    @testset verbose = true "Phase M3-4 Phase 2 (d): C.3 2D plane wave" begin
        # M3-4 Phase 2 (d): C.3 2D plane-wave acceptance driver.
        # u parallel to k̂ at IC; rotational invariance under π/2; bounded
        # mode amplitude under linear-acoustic evolution; conservation.
        include("test_M3_4_C3_plane_wave.jl")
    end

    @testset verbose = true "Phase M3-6 Phase 0: off-diagonal β reactivation" begin
        # M3-6 Phase 0 reactivates the off-diagonal Cholesky pair
        # `β_12, β_21` in the 2D residual. The Newton system grows from
        # 9 dof per cell (M3-3c) to 11 dof per cell. At β_12=β_21=0
        # IC (the M3-3c regression configuration), the new residual
        # rows are trivial-drive `(β_*_np1 − β_*_n)/dt` and the
        # 11-dof system factorises to the 9-dof M3-3c sub-system byte-
        # equally. M3-6 Phase 1 (D.1 KH falsifier) will activate the
        # off-diagonal strain coupling drive that breaks the
        # triviality. See `reference/notes_M3_6_phase0_offdiag_beta.md`
        # and `scripts/verify_berry_connection_offdiag.py` (9 SymPy
        # CHECKs reproduced numerically at the residual level).
        include("test_M3_6_phase0_offdiag_residual.jl")
        include("test_M3_6_phase0_offdiag_dimension_lift.jl")
        include("test_M3_6_phase0_offdiag_realizability.jl")
    end

    @testset verbose = true "Phase M3-6 Phase 1a: off-diagonal strain coupling" begin
        # M3-6 Phase 1a wires the off-diagonal Hamiltonian
        #   H_rot^off = G̃_12 · (α_1·β_21 + α_2·β_12) / 2
        # into the 2D EL residual. The cross-axis velocity-gradient
        # stencil (∂_2 u_1, ∂_1 u_2) feeds G̃_12 (symmetric strain →
        # F^β_12, F^β_21 drives) and W_12 (antisymmetric / vorticity →
        # F^θ_R drive). At axis-aligned ICs (every M3-3c regression and
        # M3-4 driver) both strain components vanish identically and
        # the residual reduces byte-equal to M3-6 Phase 0 — preserving
        # the §Dimension-lift gate at bit-exact 0.0 absolute. The KH
        # IC factory + Drazin-Reid calibration are M3-6 Phase 1b/1c.
        # See `reference/notes_M3_6_phase1a_strain_coupling.md`.
        include("test_M3_6_phase1a_strain_coupling.jl")
    end

    @testset verbose = true "Phase M3-6 Phase 1b: KH IC factory + 4-comp realizability" begin
        # M3-6 Phase 1b adds the `tier_d_kh_ic` / `tier_d_kh_ic_full`
        # factories (sheared base flow + antisymmetric tilt-mode
        # perturbation) and extends `realizability_project_2d!` from a
        # 2-component (β_1, β_2) s-raise projection to a 4-component
        # (β_1, β_2, β_12, β_21) cone with a post-hoc β-scaling step.
        # The 4-component check uses headroom_offdiag = 2.0 by default;
        # at β_12 = β_21 = 0 the projection is byte-equal to the M3-3d
        # 2-component output. The KH IC + cone are the prerequisites
        # for M3-6 Phase 1c (Drazin-Reid γ_KH calibration). See
        # `reference/notes_M3_6_phase1b_kh_ic_realizability.md`.
        include("test_M3_6_phase1b_kh_ic.jl")
        include("test_M3_6_phase1b_realizability_4comp.jl")
    end

    @testset verbose = true "Phase M3-6 Phase 1c: D.1 KH falsifier" begin
        # M3-6 Phase 1c closes M3-6 Phase 1 (D.1 Kelvin-Helmholtz
        # falsifier — methods paper §10.5 D.1). The
        # `experiments/D1_KH_growth_rate.jl` driver runs the Phase 1b
        # KH IC through `det_step_2d_berry_HG!` (Phase 1a strain
        # coupling + Phase 1b 4-component realizability cone) at
        # multiple refinement levels, fits the linear growth rate
        # γ_measured to the antisymmetric tilt-mode amplitude, and
        # compares to the classical Drazin-Reid prediction
        # γ_DR = U / (2 w). The c_off² calibration value is the
        # methods paper's prediction. Acceptance gates:
        #   • γ_measured / γ_DR ∈ [0.5, 2.0] at level 4 (16×16 mesh).
        #   • Mesh refinement convergence: |γ(L=5) − γ(L=4)| / |γ(L=4)|
        #     ≤ 0.2 across the level 4 → 5 → 6 sweep.
        #   • 4-component realizability: total n_negative_jacobian == 0
        #     across all leaves throughout the run.
        #   • Long-horizon stability: 3·T_KH NaN-free at level 3.
        # The level-5 mesh-sweep run is in this test file (it adds
        # ~5 minutes; the full level-6 sweep is left to the driver
        # for the headline figure). See
        # `reference/notes_M3_6_phase1c_D1_kh_falsifier.md`.
        include("test_M3_6_phase1c_D1_kh_growth_rate.jl")
    end

    @testset verbose = true "Phase M3-6 Phase 2: D.4 Zel'dovich pancake" begin
        # M3-6 Phase 2 tests the D.4 Zel'dovich pancake collapse —
        # the central novel cosmological reference test of methods
        # paper §10.5 D.4. Drives `tier_d_zeldovich_pancake_ic` (a
        # 1D-symmetric sinusoidal velocity perturbation along axis 1,
        # u_2 = 0 trivial) through `det_step_2d_berry_HG!` (with
        # M3-6 Phase 1a strain coupling and Phase 1b 4-component
        # realizability cone). The headline scientific gate is
        # **per-axis γ selectivity**:
        #   • γ_2 (trivial axis) stays uniform across cells — std/mean
        #     stays ≤ 1e-10 throughout.
        #   • γ_1 (collapsing axis) develops measurable spatial
        #     structure as t → t_cross — max/min > 1.3 at near-caustic
        #     time (L=4, T_factor=0.16).
        #   • Spatial std ratio std(γ_1)/std(γ_2) > 1e6.
        # Plus cross-checks:
        #   • Phase 1a strain coupling stays inert on the axis-aligned
        #     IC: max |β_off| = 0 throughout (∂_2 u_1 = 0 stencil).
        #   • Conservation: M, Px, Py preserved.
        #   • 4-component cone Q stays non-negative.
        # See `reference/notes_M3_6_phase2_D4_zeldovich.md`.
        include("test_M3_6_phase2_D4_zeldovich.jl")
    end

    @testset verbose = true "Phase M3-7 prep: 3D field set + per-axis Cholesky DD" begin
        # M3-7 prep scaffolding for the 3D extension milestone (the
        # full M3-7a-e phases land scientifically later). Scope of
        # this prep:
        #   • `DetField3D{T}` working struct — 13 dof per leaf cell
        #     `(x_a, u_a, α_a, β_a)_{a=1,2,3} + (θ_12, θ_13, θ_23) + s`.
        #     Off-diagonal β + post-Newton (Pp/Q) deferred per M3-3a
        #     Q3 / M3-7 design note §4.4.
        #   • Per-axis Cholesky decomposition driver in
        #     `src/cholesky_DD_3d.jl` (NEW FILE — separate from
        #     `src/cholesky_DD.jl` for parallel-safety with M3-6
        #     Phase 3): `cholesky_decompose_3d`, `cholesky_recompose_3d`,
        #     `gamma_per_axis_3d` (matrix + diagonal forms),
        #     `rotation_matrix_3d`. Intrinsic Cardan ZYX Euler-angle
        #     convention (`R = R_12 · R_13 · R_23`) — pinned in the
        #     top-of-file docstring; matches the SymPy authority in
        #     `scripts/verify_berry_connection_3D.py`.
        #   • Five test blocks: round-trip (50 random samples to ≤
        #     1e-12); iso-pullback (α_1 = α_2 = α_3 well-defined; γ
        #     equal); 2D reduction (θ_13 = θ_23 = 0 reproduces 2D byte-
        #     equal on top-left 2×2 block); per-axis γ on anisotropic
        #     M_vv (matrix + M1-form signatures); zero allocations on
        #     hot path.
        # See `reference/notes_M3_7_3d_extension.md` (full M3-7 design
        # note) and `reference/notes_M3_7_prep_3d_scaffolding.md` (this
        # sub-phase's status note).
        include("test_M3_7_prep_3d_scaffolding.jl")
    end
    @testset verbose = true "Phase M3-6 Phase 3: 2D substrate" begin
        # M3-6 Phase 3 (a/b/c) lifts three M3-3 / M3-3e 1D substrates
        # to the 2D path so the M3-6 Phase 4 (D.7 dust traps) and
        # Phase 5 (D.10 ISM tracers) drivers can plug in:
        #
        #   (a) `TracerMeshHG2D` — per-species per-cell passive
        #       scalars on `HierarchicalMesh{2}` + 14-named-field 2D
        #       Cholesky-sector field set. Pure-Lagrangian byte-exact
        #       preservation (Phase 11 + M2-2 invariants on the 2D
        #       path); refine/coarsen mass conservation via
        #       `register_tracers_on_refine_2d!`.
        #
        #   (b) `inject_vg_noise_HG_2d!` — per-axis VG stochastic
        #       injection on the 2D field set with `axes` selectivity
        #       (axis-1 injection leaves axis-2 fields byte-equal).
        #       Honours the M3-6 Phase 1b 4-component β-cone.
        #
        #   (c) `gamma_per_axis_2d_per_species_field` — per-species
        #       wrapper over `gamma_per_axis_2d_field` for D.7 dust
        #       and D.10 ISM-tracer per-species γ diagnostics.
        # See `reference/notes_M3_6_phase3_2d_substrate.md`.
        include("test_M3_6_phase3_tracer_2d.jl")
        include("test_M3_6_phase3_stochastic_2d.jl")
        include("test_M3_6_phase3_gamma_per_species.jl")
    end
    @testset verbose = true "Phase M3-6 Phase 4: D.7 dust-traps in vortices" begin
        # M3-6 Phase 4 lifts the M3-6 Phase 3 2D substrate to a Tier-D
        # falsifier driver: methods paper §10.5 D.7 dust-trapping in
        # vortices. A Taylor-Green vortex IC + 2-species (gas + dust)
        # `TracerMeshHG2D`. The dust species is pressureless cold
        # (M_vv = 0); per-species γ correctly distinguishes the two
        # phases. Driver: `experiments/D7_dust_traps.jl`.
        #
        # Honest scientific finding: the dfmm 2D variational scheme's
        # `advect_tracers_HG_2d!` is a no-op (pure-Lagrangian frame;
        # Phase 3 contract — tracer matrix byte-stable per step).
        # Sub-cell centrifugal-drift dust accumulation is *not* captured
        # by the current substrate; the headline diagnostic is per-
        # species γ separation + bit-exact dust mass conservation.
        # See `reference/notes_M3_6_phase4_D7_dust_traps.md`.
        include("test_M3_6_phase4_D7_dust_traps.jl")
    end
    @testset verbose = true "Phase M3-6 Phase 5: D.10 ISM multi-tracer fidelity" begin
        # M3-6 Phase 5 closes M3-6 entire. Methods paper §10.5 D.10
        # community-impact test: multi-tracer 2D shocked turbulence
        # with metallicity-tracking-style fidelity. Uses
        # `tier_d_ism_tracers_ic_full` (KH-style sheared base flow + N=3
        # species `TracerMeshHG2D` `[:cold, :warm, :hot]`) driven through
        # `det_step_2d_berry_HG!` + `inject_vg_noise_HG_2d!` (axes=(1,2),
        # project_kind=:reanchor) for K iterations. Driver:
        # `experiments/D10_ism_multi_tracer.jl`.
        #
        # Headline result: tracer matrix byte-equal to IC at end-time
        # (the 2D analog of M2-2's 1D structural bit-exactness
        # argument). The bit-exactness derives from inspection of the
        # write sets — neither `det_step_2d_berry_HG!` nor
        # `inject_vg_noise_HG_2d!` writes `tm.tracers`. The test serves
        # as a defensive regression guard against future refactors.
        # See `reference/notes_M3_6_phase5_D10_ism_tracers.md`.
        include("test_M3_6_phase5_D10_ism_tracers.jl")
    end
    @testset verbose = true "Phase M3-7a: 3D HaloView smoke + field set" begin
        # M3-7a (3D extension, sub-phase a):
        #   (a) 3D HaloView smoke test on a 4×4×4 balanced
        #       `HierarchicalMesh{3}` (`test_M3_7a_halo_smoke.jl`).
        #       Verifies the depth=1 contract for D=3 (interior 6-face
        #       access; corner-leaf out-of-domain returns nothing; BC-
        #       aware wrap via `face_neighbors_with_bcs`; allocation-
        #       free fast path); characterises depth=2 (smoke-only —
        #       Q1/Q4 of the M3-7 design note's §11 open questions;
        #       depth=2 accepts 2-hop offsets, throws on 3-hop).
        #   (b) 3D field-set allocator + read/write helpers
        #       (`test_M3_7a_field_set_3d.jl`). Verifies the 16-named-
        #       field `PolynomialFieldSet` (13 Newton + entropy)
        #       layout, bit-exact round-trip across all 16 dof, write-
        #       order independence, single-leaf write isolation, and
        #       the T-parameterised allocator (Float32 sanity).
        # M3-7b will land the 3D EL residual; M3-7c the SO(3) Berry
        # coupling; M3-7d per-axis γ AMR; M3-7e Tier-C/D 3D drivers.
        # See `reference/notes_M3_7_3d_extension.md` §3 + §11 Q1/Q4
        # and `reference/notes_M3_7a_3d_halo_allocator.md`.
        include("test_M3_7a_halo_smoke.jl")
        include("test_M3_7a_field_set_3d.jl")
    end
    @testset verbose = true "Phase M3-7b: native 3D EL residual (no Berry; θ_ab trivial)" begin
        # M3-7b sub-phase of M3-7 (3D extension). First native HG-side
        # 3D EL residual on the M3-7a 3D Cholesky-sector substrate.
        # Direct dimension-lift of M3-3b's 2D residual: per-axis sums
        # over `a ∈ {1, 2, 3}`, 6-face stencil (vs 4 in 2D), 15 Newton-
        # driven rows per leaf (vs 8 in M3-3b). The three θ_ab rows are
        # TRIVIAL-DRIVEN — Berry coupling lands in M3-7c. Two test
        # files:
        #   • Zero-strain regression: cold-limit fixed-point IC on a
        #     4×4×4 mesh; residual = 0 to machine precision; one Newton
        #     step preserves the state byte-equally; pack/unpack 15-dof
        #     round-trip; face-neighbor table sanity (REFLECTING +
        #     PERIODIC); EOS-driven cold-limit reduction; triply-
        #     periodic regression.
        #   • Dimension-lift parity gates (§7.1a + §7.1b — the load-
        #     bearing M3-7b acceptance criterion):
        #       §7.1a 3D ⊂ 1D — 1D-symmetric 3D config matches M1's
        #       Phase-1 zero-strain trajectory byte-equal (≤ 1e-12;
        #       achieved 0.0 absolute);
        #       §7.1b 3D ⊂ 2D — 2D-symmetric 3D config matches M3-3b's
        #       2D `det_step_2d_HG!` byte-equal (≤ 1e-12; achieved 0.0
        #       absolute). 3D ⊂ 1D run on 4×4×4 + 8×8×8 meshes; axis-
        #       swap symmetry across all three principal axes.
        # See `reference/notes_M3_7_3d_extension.md` §3 + §7 and
        # `reference/notes_M3_7b_native_3d_residual.md`.
        include("test_M3_7b_3d_zero_strain.jl")
        include("test_M3_7b_dimension_lift_3d.jl")
    end
    @testset verbose = true "Phase M3-7c: SO(3) Berry coupling integration (3D)" begin
        # M3-7c sub-phase of M3-7 (3D extension). Promotes the trivial-
        # driven θ_{ab} rows of M3-7b's 3D EL residual to Newton-active
        # rows coupled via the verified SO(3) Berry kinetic 1-form
        # `Θ_rot^{(3D)} = (1/3) Σ_{a<b} (α_a^3 β_b - α_b^3 β_a) dθ_{ab}`
        # from `src/berry.jl`. Each pair (a, b) contributes Berry α/β-
        # modifications to the per-axis residual rows, summed across
        # the three pair-generators in which axis a participates. The
        # F^θ_{ab} rows remain kinematic-drive form (drive = 0;
        # off-diagonal velocity-gradient stencil deferred to M3-9).
        # Four test files:
        #   • Berry verification reproduction (§7.2): 6 random samples
        #     × per-pair θ probes; FD-vs-closed-form match to 1e-9 +
        #     cross-check against `berry_partials_3d`.
        #   • Iso-pullback ε-expansion (§7.3): F_{ab} = 0 on iso slice;
        #     ε-extrapolation slope ≈ 1 ± 0.1; residual-level Berry
        #     contribution scales linearly in ε.
        #   • H_rot solvability (§7.4): closed-form ∂H_rot/∂θ_{ab} per
        #     pair satisfies kernel-orthogonality at 5 random (α, β,
        #     γ²) × 3 (θ̇)_test × 3 pairs (45 contractions); Newton
        #     converges in ≤ 7 iter on non-isotropic 3D IC; post-Newton
        #     residual ≤ 1e-10.
        #   • Dimension-lift parity with Berry (§7.1a + §7.1b — the
        #     load-bearing M3-7c acceptance criterion):
        #       §7.1a 3D-Berry ⊂ 1D matches M1 byte-equal (0.0 abs);
        #       §7.1b 3D-Berry ⊂ 2D-Berry matches M3-3c's
        #       `det_step_2d_berry_HG!` byte-equal (0.0 abs) on both
        #       1D-symmetric and non-trivial Berry ICs (β + θ_12).
        # See `reference/notes_M3_7_3d_extension.md` §4 + §7 and
        # `reference/notes_M3_7c_3d_berry_integration.md`.
        include("test_M3_7c_berry_3d_residual.jl")
        include("test_M3_7c_iso_pullback_3d.jl")
        include("test_M3_7c_h_rot_solvability_3d.jl")
        include("test_M3_7c_dimension_lift_3d_with_berry.jl")
    end

    @testset verbose = true "Phase M3-7d: per-axis γ + AMR/realizability per-axis (3D)" begin
        # M3-7d sub-phase of M3-7 (3D extension). 3D analog of M3-3d's
        # per-axis γ + AMR/realizability per-axis wiring. Three deliverables:
        #
        #   (a) `gamma_per_axis_3d_field` — wraps `gamma_per_axis_3d` over
        #       a 3D Cholesky-sector field set for diagnostic + AMR
        #       consumption (3D analog of M3-3d's `gamma_per_axis_2d_field`).
        #
        #   (b) 3D action-AMR — `action_error_indicator_3d_per_axis` +
        #       `register_field_set_on_refine_3d!` + `step_with_amr_3d!`.
        #       The 3D action-error indicator is the per-axis ΔS_cell
        #       evaluated in 3D (analog of M3-3d's 2D version, with
        #       `max_a` aggregation across all three axes).
        #
        #   (c) `realizability_project_3d!` + `ProjectionStats3D` — extends
        #       M3-3d's 2D version to project (β_1, β_2, β_3) onto the
        #       3-component cone. Off-diagonal β still omitted per M3-3a
        #       Q3 default + M3-7 design note §4.4.
        #
        # Four test files:
        #   • Per-axis γ diagnostic (3D): math primitive + field walker +
        #     EOS path + 1D-symmetric reduction (γ_2 = γ_3 byte-equal).
        #   • 3D AMR refine/coarsen: indicator selectivity (1D/2D/full),
        #     refine + coarsen round-trip byte-equal, listener walks all
        #     16 named fields.
        #   • 3D realizability: no-op / fires / floor branch / 2D-symmetric
        #     reduction at β_3 = 0 byte-equal / 1D-symmetric reduction.
        #   • §7.5 Per-axis γ selectivity gate (the headline scientific
        #     gate): 1D-symmetric ratio > 1e10, 2D-symmetric ratio > 1e6,
        #     full 3D all-axes-fire.
        # See `reference/notes_M3_7_3d_extension.md` §5 + §7.5 and
        # `reference/notes_M3_7d_3d_per_axis_gamma_amr.md`.
        include("test_M3_7d_gamma_per_axis_diag_3d.jl")
        include("test_M3_7d_amr_3d.jl")
        include("test_M3_7d_realizability_3d.jl")
        include("test_M3_7d_selectivity.jl")
    end

    @testset verbose = true "Phase M3-7e: 3D Tier-C/D drivers (closes M3-7)" begin
        # M3-7e (final M3-7 sub-phase). 3D analogs of the M3-4 Phase 2
        # 2D Tier-C drivers + the M3-6 Phase 2 D.4 Zel'dovich pancake.
        #
        # Four test files (one per Tier-C/D driver):
        #   • C.1 1D-symmetric 3D Sod (`test_M3_7e_C1_3d_sod.jl`):
        #     IC bridge round-trip; transverse (y, z)-independence
        #     ≤ 1e-12 per output step; conservation; axis-swap symmetry.
        #   • C.2 3D cold sinusoid (`test_M3_7e_C2_3d_cold_sinusoid.jl`):
        #     §7.5 per-axis γ selectivity reproduction across three k
        #     symmetry classes (1D-sym k=(1,0,0): ratio>1e10;
        #     2D-sym k=(1,1,0): ratio>1e6; full-3D k=(1,1,1): all axes
        #     fire).
        #   • C.3 3D plane wave (`test_M3_7e_C3_3d_plane_wave.jl`):
        #     IC bridge round-trip; linear-acoustic stability across
        #     n_steps = 5; trivial-axis velocities = 0 to round-off;
        #     mesh-resolution sanity (levels 2, 3).
        #   • D.4 3D Zel'dovich pancake (`test_M3_7e_D4_zeldovich_3d.jl`):
        #     the headline scientific test of M3-7. Per-axis γ
        #     selectivity at near-caustic > 1e10 (M3-7d gate); γ_2 = γ_3
        #     byte-equal by symmetry; mass / momentum / energy
        #     conservation ≤ 1e-8; 1D-symmetry preservation
        #     (u_2 = u_3 = 0 throughout).
        # See `reference/notes_M3_7e_3d_tier_cd_drivers.md`.
        include("test_M3_7e_C1_3d_sod.jl")
        include("test_M3_7e_C2_3d_cold_sinusoid.jl")
        include("test_M3_7e_C3_3d_plane_wave.jl")
        include("test_M3_7e_D4_zeldovich_3d.jl")
    end
    @testset verbose = true "Phase M3-8a: Tier-E stress tests + GPU prep" begin
        # M3-8 Phase a (first sub-phase of M3-8). Three Tier-E stress-test
        # drivers per methods paper §10.6:
        #
        #   • E.1 high-Mach 2D shocks (`test_M3_8a_E1_high_mach.jl`):
        #     Mach 5/10 Sod-style; graceful failure (no NaN, KE bounded);
        #     analytical Rankine-Hugoniot downstream state at IC.
        #   • E.2 severe shell-crossing (`test_M3_8a_E2_shell_crossing.jl`):
        #     2D extension of M2-3 compression cascade; superposition of
        #     two-axis Zel'dovich at A=0.7; realizability projection
        #     prevents γ_min from going negative pre-caustic.
        #   • E.3 very low Knudsen (`test_M3_8a_E3_low_knudsen.jl`):
        #     stiff-τ BGK relaxation regime; tests Newton stability;
        #     near-equilibrium preservation (γ²_a ≈ M_vv).
        #
        # GPU readiness audit + Apple Metal probe deferred to
        # `reference/notes_M3_8a_gpu_readiness_audit.md`. The actual
        # Metal kernel port is M3-8b. See
        # `reference/notes_M3_8a_tier_e_gpu_prep.md`.
        include("test_M3_8a_E1_high_mach.jl")
        include("test_M3_8a_E2_shell_crossing.jl")
        include("test_M3_8a_E3_low_knudsen.jl")
    end
    @testset verbose = true "Phase M3-8b: matrix-free Newton-Krylov + Metal kernel exploration" begin
        # M3-8 Phase b. Two deliverables under the actual GPU port phase:
        #
        #   • Matrix-free Newton-Krylov (`test_M3_8b_matrix_free_newton_krylov.jl`):
        #     bit-exact regression vs the existing dense / sparse-Jacobian
        #     `det_step_2d_berry_HG!` + `det_step_3d_berry_HG!` baseline on
        #     M3-3c zero-strain configs and M3-4 C.1/C.2 active-strain
        #     configs. Drops `SparseMatrixCSC` Jacobian construction +
        #     ForwardDiff coloring (M3-8a audit blockers #1+#2). The
        #     algorithm-side prerequisite for the M3-8c Metal port: once
        #     HG ships `PolynomialFieldSet{<:KA.Backend}` storage, the
        #     same Newton-Krylov outer loop runs on `MtlArray` /
        #     `CuArray` / `ROCArray` with no algorithm refactor.
        #
        #   • Metal kernel exploration (`test_M3_8b_metal_kernel.jl`):
        #     deferred to M3-8c. Without HG-side `Backend` parameterization
        #     there is no clean GPU lift for the residual function. The
        #     fixture detects `Metal.functional()` and runs an elementwise-
        #     add smoke kernel; the per-leaf residual kernel is `@test_skip`
        #     guarded until M3-8c lands the upstream HG storage substrate.
        #
        # See `reference/notes_M3_8b_metal_gpu_port.md` for the phase
        # status note + handoff items to M3-8c.
        include("test_M3_8b_matrix_free_newton_krylov.jl")
        include("test_M3_8b_metal_kernel.jl")
    end
end
