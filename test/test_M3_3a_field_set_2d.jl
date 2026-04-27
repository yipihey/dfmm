# test_M3_3a_field_set_2d.jl
#
# M3-3a 2D Cholesky-sector field-set tests
# (`src/types.jl::DetField2D`, `src/setups_2d.jl::allocate_cholesky_2d_fields`).
#
# Verifies:
#   1. `DetField2D` carries the 10 Newton unknowns + 2 post-Newton
#      sectors expected by M3-3b.
#   2. `allocate_cholesky_2d_fields` builds a 12-named-field
#      `PolynomialFieldSet` over `n_cells(mesh)` of a 4×4 balanced 2D
#      HierarchicalMesh.
#   3. `read_detfield_2d ∘ write_detfield_2d! ≡ id` round-trips
#      bit-exactly.
#   4. Mass-conservation read/write: write the same `DetField2D` to
#      every leaf, sum the per-leaf reads, recover the input value
#      times leaf count to round-off.
#
# See `reference/notes_M3_3_2d_cholesky_berry.md` §3.1 + §10 (Q3 — off-
# diagonal β fields are intentionally omitted from the M3-3a layout).

using Test
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    n_cells, is_leaf
using dfmm: DetField2D, n_dof_newton, spatial_dimension,
    allocate_cholesky_2d_fields, read_detfield_2d, write_detfield_2d!

@testset "M3-3a 2D Cholesky-sector field set" begin

    # ─────────────────────────────────────────────────────────────
    # Block 1: DetField2D structural contract
    # ─────────────────────────────────────────────────────────────
    @testset "DetField2D: 12 Newton fields + Pp/Q post-Newton (M3-6 Phase 0)" begin
        # M3-6 Phase 0: Newton-named fields grew from 10 to 12 with
        # the addition of `betas_off = (β_12, β_21)`. The 8-arg
        # compatibility constructor defaults `betas_off` to (0, 0).
        v = DetField2D((1.0, 2.0), (0.5, -0.3),
                        (1.2, 0.8), (0.0, 0.1),
                        0.314, 0.95, 1.0, 0.0)
        @test v isa DetField2D{Float64}
        @test spatial_dimension(v) == 2
        @test n_dof_newton(v) == 12
        @test n_dof_newton(DetField2D) == 12

        # Newton-unknown entries.
        @test v.x      == (1.0, 2.0)
        @test v.u      == (0.5, -0.3)
        @test v.alphas == (1.2, 0.8)
        @test v.betas  == (0.0, 0.1)
        # M3-6 Phase 0: backward-compat constructor defaults off-diag β to zero.
        @test v.betas_off == (0.0, 0.0)
        @test v.θ_R    == 0.314
        @test v.s      == 0.95

        # Post-Newton entries.
        @test v.Pp == 1.0
        @test v.Q  == 0.0
    end

    @testset "DetField2D: 9-arg constructor carries betas_off (M3-6 Phase 0)" begin
        v = DetField2D((1.0, 2.0), (0.5, -0.3),
                        (1.2, 0.8), (0.0, 0.1),
                        (0.07, -0.03),
                        0.314, 0.95, 1.0, 0.0)
        @test v isa DetField2D{Float64}
        @test v.betas_off == (0.07, -0.03)
        @test v.θ_R    == 0.314
        @test v.Pp == 1.0
    end

    @testset "DetField2D: 6-arg constructor defaults Pp=Q=0" begin
        v = DetField2D((0.0, 0.0), (0.0, 0.0),
                        (1.0, 1.0), (0.0, 0.0),
                        0.0, 1.0)
        @test v.Pp == 0.0
        @test v.Q  == 0.0
    end

    @testset "DetField2D: type promotion" begin
        # Mix Float64 and Int — should promote to Float64.
        v = DetField2D((1, 2), (0, 0),
                        (1.0, 1.0), (0.0, 0.0),
                        0, 1, 0, 0)
        @test v isa DetField2D{Float64}
    end

    # ─────────────────────────────────────────────────────────────
    # Block 2: allocate_cholesky_2d_fields
    # ─────────────────────────────────────────────────────────────
    # Build a 4×4 balanced 2D HierarchicalMesh (level 2 → 16 leaves).
    mesh = HierarchicalMesh{2}(; balanced = true)
    for _ in 1:2
        leaves = enumerate_leaves(mesh)
        refine_cells!(mesh, leaves)
    end
    leaves_after = enumerate_leaves(mesh)

    @testset "4×4 balanced mesh has 16 leaves and 21 cells" begin
        @test length(leaves_after) == 16
        # Total cell count: 1 (root) + 4 + 16 = 21.
        @test n_cells(mesh) == 21
    end

    fields = allocate_cholesky_2d_fields(mesh)

    @testset "field set has all 14 named scalar fields (M3-6 Phase 0)" begin
        # The 12 Newton-named fields plus Pp + Q (post-Newton) = 14.
        # M3-6 Phase 0 added :β_12, :β_21 to the 10 prior Newton fields.
        for name in (:x_1, :x_2, :u_1, :u_2,
                     :α_1, :α_2, :β_1, :β_2,
                     :β_12, :β_21,
                     :θ_R, :s, :Pp, :Q)
            view = getproperty(fields, name)
            @test length(view) == n_cells(mesh)
        end
    end

    # ─────────────────────────────────────────────────────────────
    # Block 3: write/read round-trip
    # ─────────────────────────────────────────────────────────────
    @testset "write_detfield_2d! / read_detfield_2d round-trip" begin
        # Per-leaf deterministic state: leaf k → DetField2D with
        # tagged values so a wrong index shows up as a tagged mismatch.
        for (j, ci) in enumerate(leaves_after)
            v = DetField2D(
                (Float64(j), Float64(j) + 0.5),       # x_1, x_2
                (Float64(j) * 2,  -Float64(j)),       # u_1, u_2
                (1.0 + 0.01 * j,  0.5 + 0.01 * j),    # α_1, α_2 (positive)
                (0.001 * j,       -0.001 * j),        # β_1, β_2
                0.1 * j,                              # θ_R
                1.0 + 0.0001 * j,                     # s
                Float64(j) * 10.0,                    # Pp
                Float64(j) * 100.0,                   # Q
            )
            write_detfield_2d!(fields, ci, v)
        end

        for (j, ci) in enumerate(leaves_after)
            v = read_detfield_2d(fields, ci)
            @test v.x[1]      == Float64(j)
            @test v.x[2]      == Float64(j) + 0.5
            @test v.u[1]      == Float64(j) * 2
            @test v.u[2]      == -Float64(j)
            @test v.alphas[1] == 1.0 + 0.01 * j
            @test v.alphas[2] == 0.5 + 0.01 * j
            @test v.betas[1]  == 0.001 * j
            @test v.betas[2]  == -0.001 * j
            @test v.θ_R       == 0.1 * j
            @test v.s         == 1.0 + 0.0001 * j
            @test v.Pp        == Float64(j) * 10.0
            @test v.Q         == Float64(j) * 100.0
        end
    end

    # ─────────────────────────────────────────────────────────────
    # Block 4: leaf-iteration sanity / mass-conservation analog
    # ─────────────────────────────────────────────────────────────
    @testset "uniform write: leaf-sum recovers n_leaves * value" begin
        # Reset to a single uniform DetField2D; sum of α_1 over leaves
        # equals n_leaves * value, modulo round-off. This is the read/
        # write conservation analog of the Tier-C `tier_c_total_mass`
        # helper for the new 2D Cholesky-sector field set.
        v_u = DetField2D((0.0, 0.0), (0.0, 0.0),
                          (2.5, 1.7), (0.0, 0.0),
                          0.0, 1.0, 0.0, 0.0)
        for ci in leaves_after
            write_detfield_2d!(fields, ci, v_u)
        end
        sum_α1 = 0.0
        sum_α2 = 0.0
        sum_θR = 0.0
        for ci in leaves_after
            r = read_detfield_2d(fields, ci)
            sum_α1 += r.alphas[1]
            sum_α2 += r.alphas[2]
            sum_θR += r.θ_R
        end
        @test sum_α1 ≈ 2.5 * length(leaves_after)  atol = 1e-12
        @test sum_α2 ≈ 1.7 * length(leaves_after)  atol = 1e-12
        @test sum_θR == 0.0
    end

    # ─────────────────────────────────────────────────────────────
    # Block 5: independence — writing one leaf does not perturb others
    # ─────────────────────────────────────────────────────────────
    @testset "single-leaf write does not touch other leaves" begin
        # Reset to a tagged baseline.
        v_base = DetField2D((1.0, 2.0), (0.0, 0.0),
                             (3.0, 4.0), (0.5, -0.5),
                             0.5, 1.5, 7.0, 9.0)
        for ci in leaves_after
            write_detfield_2d!(fields, ci, v_base)
        end
        # Overwrite only leaf 1's α_1 via a fresh DetField2D.
        target = leaves_after[1]
        v_new = DetField2D((1.0, 2.0), (0.0, 0.0),
                            (33.0, 4.0), (0.5, -0.5),
                            0.5, 1.5, 7.0, 9.0)
        write_detfield_2d!(fields, target, v_new)

        for (j, ci) in enumerate(leaves_after)
            r = read_detfield_2d(fields, ci)
            if ci == target
                @test r.alphas[1] == 33.0
            else
                @test r.alphas[1] == 3.0
            end
            # Other entries unchanged for every leaf.
            @test r.x[1] == 1.0
            @test r.x[2] == 2.0
            @test r.alphas[2] == 4.0
        end
    end
end
