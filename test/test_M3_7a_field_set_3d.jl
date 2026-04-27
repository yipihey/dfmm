# test_M3_7a_field_set_3d.jl
#
# M3-7a 3D Cholesky-sector field-set tests
# (`src/types.jl::DetField3D`,
#  `src/setups_2d.jl::allocate_cholesky_3d_fields`).
#
# This is the 3D analog of `test/test_M3_3a_field_set_2d.jl`. Verifies:
#
#   1. `DetField3D` carries the 13 Newton unknowns (12 per-axis +
#      3 angles + 1 entropy = 16 named scalars total) expected by
#      M3-7b.
#   2. `allocate_cholesky_3d_fields` builds a 16-named-field
#      `PolynomialFieldSet` over `n_cells(mesh)` of a 4×4×4 balanced
#      3D HierarchicalMesh.
#   3. `read_detfield_3d ∘ write_detfield_3d! ≡ id` round-trips
#      bit-exactly across all 16 dof.
#   4. Multi-leaf write in any order produces same final state as a
#      single-pass write (write-set isolation per leaf).
#   5. Mass-conservation read/write: write the same `DetField3D` to
#      every leaf, sum the per-leaf reads, recover the input value
#      times leaf count to round-off.
#   6. Single-leaf write does not perturb other leaves.
#
# See `reference/notes_M3_7_3d_extension.md` §3 (3D EL residual handoff)
# and `reference/notes_M3_7_prep_3d_scaffolding.md` (DetField3D + 3D
# Cholesky decomposition already in place; this test pins the field
# set + read/write contract on top).

using Test
using Random
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    n_cells, is_leaf
using dfmm: DetField3D, n_dof_newton, spatial_dimension,
    allocate_cholesky_3d_fields, read_detfield_3d, write_detfield_3d!

@testset "M3-7a 3D Cholesky-sector field set" begin

    # ─────────────────────────────────────────────────────────────
    # Block 1: DetField3D structural contract
    # ─────────────────────────────────────────────────────────────
    @testset "DetField3D: 13 Newton unknowns + entropy" begin
        v = DetField3D((1.0, 2.0, 3.0),         # x
                        (0.5, -0.3, 0.1),       # u
                        (1.2, 0.8, 0.6),        # alphas
                        (0.0, 0.1, -0.2),       # betas
                        0.314, -0.157, 0.785,   # θ_12, θ_13, θ_23
                        0.95)                   # s
        @test v isa DetField3D{Float64}
        @test spatial_dimension(v) == 3
        @test n_dof_newton(v) == 13
        @test n_dof_newton(DetField3D) == 13

        # Newton-unknown entries.
        @test v.x      == (1.0, 2.0, 3.0)
        @test v.u      == (0.5, -0.3, 0.1)
        @test v.alphas == (1.2, 0.8, 0.6)
        @test v.betas  == (0.0, 0.1, -0.2)
        @test v.θ_12   == 0.314
        @test v.θ_13   == -0.157
        @test v.θ_23   == 0.785
        @test v.s      == 0.95
    end

    @testset "DetField3D: type promotion (Int / Float64 mix)" begin
        # Mix Float64 and Int — should promote to Float64.
        v = DetField3D((1, 2, 3),
                        (0, 0, 0),
                        (1.0, 1.0, 1.0),
                        (0.0, 0.0, 0.0),
                        0, 1, 0,
                        1)
        @test v isa DetField3D{Float64}
    end

    # ─────────────────────────────────────────────────────────────
    # Block 2: allocate_cholesky_3d_fields
    # ─────────────────────────────────────────────────────────────
    # Build a 4×4×4 balanced 3D HierarchicalMesh (level 2 → 64 leaves).
    mesh = HierarchicalMesh{3}(; balanced = true)
    for _ in 1:2
        leaves_iter = enumerate_leaves(mesh)
        refine_cells!(mesh, leaves_iter)
    end
    leaves_after = enumerate_leaves(mesh)

    @testset "4×4×4 balanced 3D mesh has 64 leaves and 73 cells" begin
        @test length(leaves_after) == 64
        # Cumulative cell count: 1 (root) + 8 + 64 = 73.
        @test n_cells(mesh) == 73
    end

    fields = allocate_cholesky_3d_fields(mesh)

    @testset "field set has all 16 named scalar fields (M3-7a)" begin
        # 13 Newton-named fields (12 per-axis + 3 angles) + 1 entropy.
        # M3-7a: no off-diagonal β, no post-Newton Pp / Q (deferred per
        # M3-7 design note §4.4).
        for name in (:x_1, :x_2, :x_3,
                     :u_1, :u_2, :u_3,
                     :α_1, :α_2, :α_3,
                     :β_1, :β_2, :β_3,
                     :θ_12, :θ_13, :θ_23,
                     :s)
            view = getproperty(fields, name)
            @test length(view) == n_cells(mesh)
        end
    end

    # ─────────────────────────────────────────────────────────────
    # Block 3: write/read round-trip — all 16 dof byte-equal
    # ─────────────────────────────────────────────────────────────
    @testset "write_detfield_3d! / read_detfield_3d round-trip (byte-equal)" begin
        # Per-leaf deterministic state: leaf k → DetField3D with
        # tagged values so a wrong index shows up as a tagged mismatch.
        for (j, ci) in enumerate(leaves_after)
            v = DetField3D(
                (Float64(j), Float64(j) + 0.5, Float64(j) + 0.25),  # x_1, x_2, x_3
                (Float64(j) * 2,  -Float64(j), Float64(j) * 3),     # u
                (1.0 + 0.01 * j,  0.5 + 0.01 * j, 0.7 + 0.01 * j),  # α (positive)
                (0.001 * j,       -0.001 * j, 0.0005 * j),          # β
                0.1 * j,                                             # θ_12
                -0.05 * j,                                           # θ_13
                0.2 * j,                                             # θ_23
                1.0 + 0.0001 * j,                                    # s
            )
            write_detfield_3d!(fields, ci, v)
        end

        for (j, ci) in enumerate(leaves_after)
            v = read_detfield_3d(fields, ci)
            # All 16 dof byte-equal to the input.
            @test v.x[1]      === Float64(j)
            @test v.x[2]      === Float64(j) + 0.5
            @test v.x[3]      === Float64(j) + 0.25
            @test v.u[1]      === Float64(j) * 2
            @test v.u[2]      === -Float64(j)
            @test v.u[3]      === Float64(j) * 3
            @test v.alphas[1] === 1.0 + 0.01 * j
            @test v.alphas[2] === 0.5 + 0.01 * j
            @test v.alphas[3] === 0.7 + 0.01 * j
            @test v.betas[1]  === 0.001 * j
            @test v.betas[2]  === -0.001 * j
            @test v.betas[3]  === 0.0005 * j
            @test v.θ_12      === 0.1 * j
            @test v.θ_13      === -0.05 * j
            @test v.θ_23      === 0.2 * j
            @test v.s         === 1.0 + 0.0001 * j
        end
    end

    # ─────────────────────────────────────────────────────────────
    # Block 4: leaf-iteration sanity / mass-conservation analog
    # ─────────────────────────────────────────────────────────────
    @testset "uniform write: leaf-sum recovers n_leaves * value" begin
        # Reset to a single uniform DetField3D; sum of α_a over leaves
        # equals n_leaves * value, modulo round-off.
        v_u = DetField3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                          (2.5, 1.7, 0.9), (0.0, 0.0, 0.0),
                          0.0, 0.0, 0.0, 1.0)
        for ci in leaves_after
            write_detfield_3d!(fields, ci, v_u)
        end
        sum_α1 = 0.0
        sum_α2 = 0.0
        sum_α3 = 0.0
        sum_θ12 = 0.0
        sum_θ13 = 0.0
        sum_θ23 = 0.0
        sum_s   = 0.0
        for ci in leaves_after
            r = read_detfield_3d(fields, ci)
            sum_α1 += r.alphas[1]
            sum_α2 += r.alphas[2]
            sum_α3 += r.alphas[3]
            sum_θ12 += r.θ_12
            sum_θ13 += r.θ_13
            sum_θ23 += r.θ_23
            sum_s   += r.s
        end
        @test sum_α1 ≈ 2.5 * length(leaves_after)  atol = 1e-12
        @test sum_α2 ≈ 1.7 * length(leaves_after)  atol = 1e-12
        @test sum_α3 ≈ 0.9 * length(leaves_after)  atol = 1e-12
        @test sum_θ12 == 0.0
        @test sum_θ13 == 0.0
        @test sum_θ23 == 0.0
        @test sum_s   ≈ 1.0 * length(leaves_after)  atol = 1e-12
    end

    # ─────────────────────────────────────────────────────────────
    # Block 5: independence — writing one leaf does not perturb others
    # ─────────────────────────────────────────────────────────────
    @testset "single-leaf write does not touch other leaves" begin
        # Reset to a tagged baseline.
        v_base = DetField3D((1.0, 2.0, 3.0), (0.0, 0.0, 0.0),
                             (3.0, 4.0, 5.0), (0.5, -0.5, 0.25),
                             0.5, -0.25, 0.1, 1.5)
        for ci in leaves_after
            write_detfield_3d!(fields, ci, v_base)
        end
        # Overwrite only leaf 1's α_1 via a fresh DetField3D.
        target = leaves_after[1]
        v_new = DetField3D((1.0, 2.0, 3.0), (0.0, 0.0, 0.0),
                            (33.0, 4.0, 5.0), (0.5, -0.5, 0.25),
                            0.5, -0.25, 0.1, 1.5)
        write_detfield_3d!(fields, target, v_new)

        for ci in leaves_after
            r = read_detfield_3d(fields, ci)
            if ci == target
                @test r.alphas[1] == 33.0
            else
                @test r.alphas[1] == 3.0
            end
            # Other entries unchanged for every leaf.
            @test r.x[1] == 1.0
            @test r.x[2] == 2.0
            @test r.x[3] == 3.0
            @test r.alphas[2] == 4.0
            @test r.alphas[3] == 5.0
            @test r.θ_12 == 0.5
            @test r.θ_13 == -0.25
            @test r.θ_23 == 0.1
        end
    end

    # ─────────────────────────────────────────────────────────────
    # Block 6: write-order independence — multi-leaf write in random
    # order produces same final state as single-pass forward write
    # ─────────────────────────────────────────────────────────────
    @testset "write order independence (random vs forward sweep)" begin
        # Generate a deterministic per-leaf state.
        rng = MersenneTwister(7919)
        per_leaf = Dict{Int, DetField3D{Float64}}()
        for (j, ci) in enumerate(leaves_after)
            per_leaf[Int(ci)] = DetField3D(
                (Float64(j), Float64(j) + 1.0, Float64(j) + 2.0),
                (rand(rng), rand(rng), rand(rng)),
                (1.0 + 0.1 * rand(rng), 1.0 + 0.1 * rand(rng), 1.0 + 0.1 * rand(rng)),
                (rand(rng) - 0.5, rand(rng) - 0.5, rand(rng) - 0.5),
                rand(rng), rand(rng), rand(rng),
                1.0 + 0.01 * rand(rng),
            )
        end

        # Forward-sweep write into one allocator.
        fields_fwd = allocate_cholesky_3d_fields(mesh)
        for ci in leaves_after
            write_detfield_3d!(fields_fwd, ci, per_leaf[Int(ci)])
        end

        # Random-order write into a fresh allocator.
        fields_rand = allocate_cholesky_3d_fields(mesh)
        order = collect(leaves_after)
        Random.shuffle!(rng, order)
        for ci in order
            write_detfield_3d!(fields_rand, ci, per_leaf[Int(ci)])
        end

        # Per-leaf read must agree byte-equal across the two writes.
        for ci in leaves_after
            r_fwd  = read_detfield_3d(fields_fwd,  ci)
            r_rand = read_detfield_3d(fields_rand, ci)
            @test r_fwd.x      === r_rand.x
            @test r_fwd.u      === r_rand.u
            @test r_fwd.alphas === r_rand.alphas
            @test r_fwd.betas  === r_rand.betas
            @test r_fwd.θ_12   === r_rand.θ_12
            @test r_fwd.θ_13   === r_rand.θ_13
            @test r_fwd.θ_23   === r_rand.θ_23
            @test r_fwd.s      === r_rand.s
        end
    end

    # ─────────────────────────────────────────────────────────────
    # Block 7: 2D byte-equal regression sentinel — the 3D allocator
    # does not affect any 2D / 1D path. We don't repeat the M3-3a
    # field_set tests here; the regression suite already covers them.
    # This block just documents the contract.
    # ─────────────────────────────────────────────────────────────
    @testset "T-parameterised allocator (Float32 sanity)" begin
        # Float32 mesh allocator: the T parameter threads through.
        fields_f32 = allocate_cholesky_3d_fields(mesh; T = Float32)
        # Round-trip with explicit Float32 inputs verifies the allocator
        # honours T = Float32 (the underlying SoA Vector{Float32} backing
        # is opaque at the PolynomialFieldView eltype but the round-trip
        # returns Float32 values byte-equal to input).
        v32 = DetField3D((1.0f0, 2.0f0, 3.0f0),
                          (0.0f0, 0.0f0, 0.0f0),
                          (1.0f0, 1.0f0, 1.0f0),
                          (0.0f0, 0.0f0, 0.0f0),
                          0.5f0, 0.25f0, 0.1f0,
                          1.0f0)
        write_detfield_3d!(fields_f32, leaves_after[1], v32)
        r32 = read_detfield_3d(fields_f32, leaves_after[1])
        @test r32 isa DetField3D{Float32}
        @test r32.x      === (1.0f0, 2.0f0, 3.0f0)
        @test r32.alphas === (1.0f0, 1.0f0, 1.0f0)
        @test r32.θ_12   === 0.5f0
        @test r32.s      === 1.0f0
    end
end
