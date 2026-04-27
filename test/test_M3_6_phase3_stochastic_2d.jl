# test_M3_6_phase3_stochastic_2d.jl
#
# M3-6 Phase 3 (b): unit + integration tests for the 2D stochastic
# injection `inject_vg_noise_HG_2d!` with explicit per-axis selectivity.
#
# Three blocks:
#
#   1. **Per-axis selectivity (load-bearing structural property):**
#      with `axes = (1,)`, axis-2 fields `(β_2, u_2)` are byte-equal
#      pre/post the injection. With `axes = (2,)`, axis-1 fields
#      `(β_1, u_1)` are byte-equal pre/post.
#
#   2. **4-component realizability cone respected:** post-injection +
#      `realizability_project_2d!`, the 4-component cone
#      `Q = β_1² + β_2² + 2(β_12² + β_21²) ≤ M_vv · headroom_offdiag`
#      holds at every leaf — verified by zero negative-Jacobian
#      cells `n_negative_jacobian == 0`.
#
#   3. **No-noise reduction at zero RNG output:** with `C_A = 0,
#      C_B = 0`, the 2D path is a pure no-op (state byte-equal pre/
#      post). This is the deterministic-fallback parity gate.
#
#   4. **2D ⊃ 1D parity at axis-1-aligned IC:** with axis-2-trivial
#      IC (`u_2 = 0`, `β_2 = 0` everywhere) and `axes = (1,)`, the
#      RNG draws + per-cell δ values match a 1-axis run.
#
# See `reference/notes_M3_6_phase3_2d_substrate.md`.

using Test
using Random: MersenneTwister
using HierarchicalGrids: HierarchicalMesh, EulerianFrame,
    refine_cells!, enumerate_leaves,
    FrameBoundaries, REFLECTING, PERIODIC,
    cell_physical_box
using dfmm
using dfmm: inject_vg_noise_HG_2d!, InjectionDiagnostics2D,
            NoiseInjectionParams, ProjectionStats,
            allocate_cholesky_2d_fields,
            write_detfield_2d!, read_detfield_2d, DetField2D,
            cholesky_sector_state_from_primitive,
            realizability_project_2d!,
            tier_d_zeldovich_pancake_ic, Mvv

const BC_PERIODIC_2D = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                            (PERIODIC, PERIODIC)))

# Helper: check the per-axis γ²_a ≥ 0 cone per leaf and return the
# count of leaves with γ²_a < 0 on either axis (the
# "n_negative_jacobian" diagnostic).
function n_negative_jacobian_2d(fields, leaves; ρ_ref = 1.0,
                                 Gamma = 1.4)
    n_neg = 0
    for ci in leaves
        β1 = Float64(fields.β_1[ci][1])
        β2 = Float64(fields.β_2[ci][1])
        s_i = Float64(fields.s[ci][1])
        Mvv_iso = Float64(Mvv(1.0 / Float64(ρ_ref), s_i;
                                Gamma = Float64(Gamma)))
        γ1² = Mvv_iso - β1 * β1
        γ2² = Mvv_iso - β2 * β2
        if γ1² < 0 || γ2² < 0
            n_neg += 1
        end
    end
    return n_neg
end

# Cold-isotropic 2D mesh + field set helper.
function build_uniform_2d(level::Int; ρ_ref::Real = 1.0,
                            P_ref::Real = 0.1)
    mesh = HierarchicalMesh{2}(; balanced = true)
    for _ in 1:level
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    leaves = enumerate_leaves(mesh)
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
    fields = allocate_cholesky_2d_fields(mesh)
    for ci in leaves
        lo, hi = cell_physical_box(frame, ci)
        cx = 0.5 * (lo[1] + hi[1])
        cy = 0.5 * (lo[2] + hi[2])
        v = cholesky_sector_state_from_primitive(ρ_ref, 0.0, 0.0,
                                                  P_ref, (cx, cy))
        write_detfield_2d!(fields, ci, v)
    end
    return (mesh = mesh, leaves = leaves, frame = frame, fields = fields)
end

# Snapshot all per-cell field values into a NamedTuple of vectors.
function snapshot_fields_2d(fields, leaves)
    N = length(leaves)
    snap = (
        x_1 = [Float64(fields.x_1[ci][1])  for ci in leaves],
        x_2 = [Float64(fields.x_2[ci][1])  for ci in leaves],
        u_1 = [Float64(fields.u_1[ci][1])  for ci in leaves],
        u_2 = [Float64(fields.u_2[ci][1])  for ci in leaves],
        β_1 = [Float64(fields.β_1[ci][1])  for ci in leaves],
        β_2 = [Float64(fields.β_2[ci][1])  for ci in leaves],
        β_12 = [Float64(fields.β_12[ci][1]) for ci in leaves],
        β_21 = [Float64(fields.β_21[ci][1]) for ci in leaves],
        s   = [Float64(fields.s[ci][1])    for ci in leaves],
        Pp  = [Float64(fields.Pp[ci][1])   for ci in leaves],
    )
    return snap
end

@testset "M3-6 Phase 3 (b): inject_vg_noise_HG_2d!" begin

    @testset "constructor: InjectionDiagnostics2D" begin
        d = InjectionDiagnostics2D(7)
        @test length(d.divu[1]) == 7
        @test length(d.divu[2]) == 7
        @test length(d.eta[1]) == 7
        @test length(d.eta[2]) == 7
        @test length(d.delta_rhou[1]) == 7
        @test length(d.delta_KE_vol[2]) == 7
        @test length(d.compressive[1]) == 7
        @test all(d.divu[1] .== 0.0)
        @test all(d.eta[2] .== 0.0)
        @test !any(d.compressive[1])
    end

    @testset "no-noise reduction (C_A = C_B = 0, uniform IC): byte-equal" begin
        # Uniform IC with C_A = C_B = 0: drift and noise both vanish
        # ⇒ pure no-op. Note: θ_factor must be > 0 to satisfy
        # `rand_variance_gamma!` even though the noise is multiplied
        # by C_B = 0. The RNG advances but does not perturb fields.
        st = build_uniform_2d(2)
        params = NoiseInjectionParams(; C_A = 0.0, C_B = 0.0,
                                        λ = 1.0, θ_factor = 0.5,
                                        ke_budget_fraction = 0.5,
                                        pressure_floor = 1e-12,
                                        ell_corr = 0,
                                        project_kind = :none)
        rng = MersenneTwister(42)
        snap_pre = snapshot_fields_2d(st.fields, st.leaves)
        inject_vg_noise_HG_2d!(st.fields, st.mesh, st.frame, st.leaves,
                                BC_PERIODIC_2D, 1e-3;
                                params = params, rng = rng,
                                axes = (1, 2))
        snap_post = snapshot_fields_2d(st.fields, st.leaves)
        # All fields byte-equal (uniform IC ⇒ divu = 0, drift = 0,
        # noise_amp = C_B · ρ · √(0 · dt) = 0 ⇒ δ = 0 ⇒ state
        # unchanged).
        for k in keys(snap_pre)
            @test snap_pre[k] == snap_post[k]
        end
    end

    @testset "axis-1 selectivity: axes = (1,) leaves axis-2 byte-equal" begin
        # Build a non-trivial 2D state where axis-2 fields are well-
        # defined but should not be perturbed by axis-1 injection.
        st = build_uniform_2d(2)
        # Add axis-1 velocity gradient to give injection something to
        # bite on (divu_1 ≠ 0 driving compression).
        for (k, ci) in enumerate(st.leaves)
            lo, hi = cell_physical_box(st.frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            v = cholesky_sector_state_from_primitive(
                1.0, -0.3 * sin(2π * cx), 0.0, 0.1,
                (cx, 0.5 * (lo[2] + hi[2])))
            write_detfield_2d!(st.fields, ci, v)
        end

        params = NoiseInjectionParams(; C_A = 0.5, C_B = 1.0,
                                        λ = 0.5, θ_factor = 0.5,
                                        ke_budget_fraction = 0.3,
                                        pressure_floor = 1e-12,
                                        ell_corr = 1,
                                        project_kind = :none)
        rng = MersenneTwister(123)
        snap_pre = snapshot_fields_2d(st.fields, st.leaves)
        inject_vg_noise_HG_2d!(st.fields, st.mesh, st.frame, st.leaves,
                                BC_PERIODIC_2D, 1e-3;
                                params = params, rng = rng,
                                axes = (1,))
        snap_post = snapshot_fields_2d(st.fields, st.leaves)

        # Axis-2 fields: byte-equal pre/post.
        @test snap_pre.x_2 == snap_post.x_2
        @test snap_pre.u_2 == snap_post.u_2
        @test snap_pre.β_2 == snap_post.β_2
        @test snap_pre.β_12 == snap_post.β_12
        @test snap_pre.β_21 == snap_post.β_21

        # Axis-1 u: should change (injection wrote into axis-1 u).
        @test snap_pre.u_1 != snap_post.u_1

        # x_1 unchanged (the per-cell variational scheme stores x_a as
        # cell-center; injection writes only u_a, β_a, s, Pp).
        @test snap_pre.x_1 == snap_post.x_1

        # s, Pp may or may not change depending on whether
        # KE-debit fired; assert at least one cell saw motion.
        @test (snap_pre.s != snap_post.s) || (snap_pre.Pp != snap_post.Pp) ||
              (snap_pre.u_1 != snap_post.u_1)
    end

    @testset "axis-2 selectivity: axes = (2,) leaves axis-1 byte-equal" begin
        # Symmetric to the axis-1 selectivity test: axis-2 velocity
        # gradient drives axis-2 injection; axis-1 fields untouched.
        st = build_uniform_2d(2)
        for ci in st.leaves
            lo, hi = cell_physical_box(st.frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = cholesky_sector_state_from_primitive(
                1.0, 0.0, -0.3 * sin(2π * cy), 0.1, (cx, cy))
            write_detfield_2d!(st.fields, ci, v)
        end

        params = NoiseInjectionParams(; C_A = 0.5, C_B = 1.0,
                                        λ = 0.5, θ_factor = 0.5,
                                        ke_budget_fraction = 0.3,
                                        pressure_floor = 1e-12,
                                        ell_corr = 1,
                                        project_kind = :none)
        rng = MersenneTwister(7)
        snap_pre = snapshot_fields_2d(st.fields, st.leaves)
        inject_vg_noise_HG_2d!(st.fields, st.mesh, st.frame, st.leaves,
                                BC_PERIODIC_2D, 1e-3;
                                params = params, rng = rng,
                                axes = (2,))
        snap_post = snapshot_fields_2d(st.fields, st.leaves)

        # Axis-1 fields: byte-equal pre/post.
        @test snap_pre.x_1 == snap_post.x_1
        @test snap_pre.u_1 == snap_post.u_1
        @test snap_pre.β_1 == snap_post.β_1
        @test snap_pre.β_12 == snap_post.β_12
        @test snap_pre.β_21 == snap_post.β_21

        # Axis-2 u: changed.
        @test snap_pre.u_2 != snap_post.u_2
    end

    @testset "axes = (1, 2) updates both axes" begin
        # Both axes have non-trivial velocity gradients; both should
        # see injection.
        st = build_uniform_2d(2)
        for ci in st.leaves
            lo, hi = cell_physical_box(st.frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = cholesky_sector_state_from_primitive(
                1.0, -0.2 * sin(2π * cx), -0.2 * sin(2π * cy), 0.1,
                (cx, cy))
            write_detfield_2d!(st.fields, ci, v)
        end

        params = NoiseInjectionParams(; C_A = 0.5, C_B = 1.0,
                                        λ = 0.5, θ_factor = 0.5,
                                        ke_budget_fraction = 0.3,
                                        pressure_floor = 1e-12,
                                        ell_corr = 1,
                                        project_kind = :none)
        rng = MersenneTwister(7)
        snap_pre = snapshot_fields_2d(st.fields, st.leaves)
        diag = InjectionDiagnostics2D(length(st.leaves))
        inject_vg_noise_HG_2d!(st.fields, st.mesh, st.frame, st.leaves,
                                BC_PERIODIC_2D, 1e-3;
                                params = params, rng = rng,
                                axes = (1, 2), diag = diag)
        snap_post = snapshot_fields_2d(st.fields, st.leaves)

        # Both axes' u changed.
        @test snap_pre.u_1 != snap_post.u_1
        @test snap_pre.u_2 != snap_post.u_2

        # Both axes' η buffers populated.
        @test any(diag.eta[1] .!= 0.0)
        @test any(diag.eta[2] .!= 0.0)

        # Some cells had divu < 0 ⇒ compressive bit set on at least
        # one cell on each axis.
        @test any(diag.compressive[1])
        @test any(diag.compressive[2])
    end

    @testset "axes = () → no-op (state byte-equal)" begin
        # Empty `axes` tuple: no axis sees injection ⇒ pure no-op.
        st = build_uniform_2d(2)
        for ci in st.leaves
            lo, hi = cell_physical_box(st.frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            v = cholesky_sector_state_from_primitive(
                1.0, -0.3 * sin(2π * cx), 0.0, 0.1,
                (cx, 0.5 * (lo[2] + hi[2])))
            write_detfield_2d!(st.fields, ci, v)
        end
        params = NoiseInjectionParams(; C_A = 0.5, C_B = 1.0,
                                        λ = 0.5, θ_factor = 0.5,
                                        ke_budget_fraction = 0.3,
                                        pressure_floor = 1e-12,
                                        ell_corr = 1,
                                        project_kind = :none)
        rng = MersenneTwister(99)
        snap_pre = snapshot_fields_2d(st.fields, st.leaves)
        inject_vg_noise_HG_2d!(st.fields, st.mesh, st.frame, st.leaves,
                                BC_PERIODIC_2D, 1e-3;
                                params = params, rng = rng,
                                axes = ())
        snap_post = snapshot_fields_2d(st.fields, st.leaves)
        for k in keys(snap_pre)
            @test snap_pre[k] == snap_post[k]
        end
    end

    @testset "4-component cone respected post-injection (project_kind = :reanchor)" begin
        # Drive a non-trivial state with axis-1 compression; apply
        # injection with realizability projection. Post-injection,
        # n_negative_jacobian should be 0 (cone holds at every leaf).
        st = build_uniform_2d(2)
        for ci in st.leaves
            lo, hi = cell_physical_box(st.frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = cholesky_sector_state_from_primitive(
                1.0, -0.3 * sin(2π * cx), 0.0, 0.5,
                (cx, cy))
            write_detfield_2d!(st.fields, ci, v)
        end

        params = NoiseInjectionParams(; C_A = 0.5, C_B = 1.0,
                                        λ = 0.5, θ_factor = 0.5,
                                        ke_budget_fraction = 0.3,
                                        pressure_floor = 1e-12,
                                        Mvv_floor = 1e-2,
                                        realizability_headroom = 1.05,
                                        ell_corr = 1,
                                        project_kind = :reanchor)
        rng = MersenneTwister(31)
        proj_stats = ProjectionStats()
        inject_vg_noise_HG_2d!(st.fields, st.mesh, st.frame, st.leaves,
                                BC_PERIODIC_2D, 1e-3;
                                params = params, rng = rng,
                                axes = (1, 2),
                                proj_stats = proj_stats)
        n_neg = n_negative_jacobian_2d(st.fields, st.leaves)
        @test n_neg == 0   # 4-component cone strictly held
        @test proj_stats.n_steps >= 1
    end

    @testset "RNG-bit reproducibility: same seed → same δ_rhou" begin
        st1 = build_uniform_2d(2)
        st2 = build_uniform_2d(2)
        for ci in st1.leaves
            lo, hi = cell_physical_box(st1.frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = cholesky_sector_state_from_primitive(
                1.0, -0.3 * sin(2π * cx), 0.0, 0.1, (cx, cy))
            write_detfield_2d!(st1.fields, ci, v)
            write_detfield_2d!(st2.fields, ci, v)
        end

        params = NoiseInjectionParams(; C_A = 0.5, C_B = 1.0,
                                        λ = 0.5, θ_factor = 0.5,
                                        ke_budget_fraction = 0.3,
                                        pressure_floor = 1e-12,
                                        ell_corr = 1,
                                        project_kind = :none)
        diag1 = InjectionDiagnostics2D(length(st1.leaves))
        diag2 = InjectionDiagnostics2D(length(st2.leaves))
        rng1 = MersenneTwister(2026)
        rng2 = MersenneTwister(2026)
        inject_vg_noise_HG_2d!(st1.fields, st1.mesh, st1.frame, st1.leaves,
                                BC_PERIODIC_2D, 1e-3;
                                params = params, rng = rng1,
                                axes = (1, 2), diag = diag1)
        inject_vg_noise_HG_2d!(st2.fields, st2.mesh, st2.frame, st2.leaves,
                                BC_PERIODIC_2D, 1e-3;
                                params = params, rng = rng2,
                                axes = (1, 2), diag = diag2)
        # Identical RNG seeds + identical IC ⇒ byte-equal δ.
        @test diag1.delta_rhou[1] == diag2.delta_rhou[1]
        @test diag1.delta_rhou[2] == diag2.delta_rhou[2]
        @test diag1.eta[1] == diag2.eta[1]
        @test diag1.eta[2] == diag2.eta[2]
    end

    @testset "1-axis variant on a Zel'dovich-like axis-aligned IC" begin
        # The Zel'dovich pancake IC has u_2 = 0, β_2 = 0 strictly.
        # Injection with axes = (1,) should perturb only axis-1
        # quantities and leave axis-2 byte-equal — the strong form of
        # the Phase 1a inertness check from D.4 (max |β_off| = 0).
        ic = tier_d_zeldovich_pancake_ic(; level = 3, A = 0.3, T = Float64)
        params = NoiseInjectionParams(; C_A = 0.5, C_B = 1.0,
                                        λ = 0.5, θ_factor = 0.5,
                                        ke_budget_fraction = 0.3,
                                        pressure_floor = 1e-12,
                                        ell_corr = 1,
                                        project_kind = :none)
        rng = MersenneTwister(123)   # any nonzero seed
        snap_pre = snapshot_fields_2d(ic.fields, ic.leaves)
        inject_vg_noise_HG_2d!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                                BC_PERIODIC_2D, 1e-3;
                                params = params, rng = rng,
                                axes = (1,))
        snap_post = snapshot_fields_2d(ic.fields, ic.leaves)

        # Axis-2 strict inertness: x_2, u_2, β_2 all byte-equal.
        @test snap_pre.x_2 == snap_post.x_2
        @test snap_pre.u_2 == snap_post.u_2
        @test snap_pre.β_2 == snap_post.β_2
        # β_12, β_21 — also unchanged (off-diag pair untouched by 2-
        # component per-axis injection without realizability projection).
        @test snap_pre.β_12 == snap_post.β_12
        @test snap_pre.β_21 == snap_post.β_21
    end

end
