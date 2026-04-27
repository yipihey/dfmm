# test_M3_6_phase3_tracer_2d.jl
#
# M3-6 Phase 3 (a): unit + integration tests for the 2D tracer
# substrate `TracerMeshHG2D` and its refinement listener
# `register_tracers_on_refine_2d!`.
#
# Three blocks:
#
#   1. **Pure-Lagrangian byte-exact preservation (Phase 11 invariant
#      lifted to 2D):** with no refinement events, the tracer matrix
#      is never written by the dfmm path, so any number of
#      `det_step_2d_berry_HG!` calls leaves it byte-identical.
#
#   2. **Refine/coarsen mass conservation (M2-2 invariant lifted to
#      2D):** parent → 4 children prolongation preserves total mass
#      `Σ c·V` exactly on equal-volume isotropic refinement.
#      Coarsen reverses this byte-equally (volume-weighted mean →
#      arithmetic mean for equal volumes).
#
#   3. **Multi-species independence:** two species (A, B) initialised
#      to disjoint patterns; refining a leaf does not cross-
#      contaminate the species rows.
#
# See `reference/notes_M3_6_phase3_2d_substrate.md`.

using Test
using HierarchicalGrids: HierarchicalMesh, EulerianFrame,
    refine_cells!, coarsen_cells!, enumerate_leaves,
    FrameBoundaries, REFLECTING, PERIODIC,
    cell_physical_box, unregister_refinement_listener!
using HierarchicalGrids: n_cells as hg_n_cells
using dfmm: TracerMeshHG2D, advect_tracers_HG_2d!,
            set_species!, species_index, n_species, n_cells_2d,
            register_tracers_on_refine_2d!,
            allocate_cholesky_2d_fields,
            write_detfield_2d!, DetField2D,
            register_field_set_on_refine!,
            det_step_2d_berry_HG!,
            cholesky_sector_state_from_primitive

@testset "M3-6 Phase 3 (a): TracerMeshHG2D" begin

    @testset "constructor / accessors" begin
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:2
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_2d_fields(mesh)

        # Default constructor: 1 species, default name.
        tm0 = TracerMeshHG2D(fields, mesh)
        @test n_species(tm0) == 1
        @test n_cells_2d(tm0) == hg_n_cells(mesh)
        @test tm0.names == [:species1]
        @test all(tm0.tracers .== 0.0)

        # Multi-species + custom names.
        tm = TracerMeshHG2D(fields, mesh; n_species = 3,
                             names = [:dust, :gas, :metals])
        @test n_species(tm) == 3
        @test n_cells_2d(tm) == hg_n_cells(mesh)
        @test tm.names == [:dust, :gas, :metals]
        @test species_index(tm, :gas) == 2
        @test species_index(tm, :dust) == 1
        @test species_index(tm, :metals) == 3
        @test_throws ArgumentError species_index(tm, :nonexistent)

        # Invalid species count.
        @test_throws AssertionError TracerMeshHG2D(fields, mesh;
                                                    n_species = 0)

        # Names length mismatch.
        @test_throws AssertionError TracerMeshHG2D(fields, mesh;
                                                    n_species = 2,
                                                    names = [:only_one])
    end

    @testset "set_species!: vector form + functional form" begin
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:2
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_2d_fields(mesh)
        tm = TracerMeshHG2D(fields, mesh; n_species = 2,
                             names = [:A, :B])

        # Vector form for species A.
        vals = collect(Float64, 1:length(leaves))
        set_species!(tm, :A, vals, leaves)
        for (i, ci) in enumerate(leaves)
            @test tm.tracers[1, ci] == Float64(i)
        end
        # Species B still all zeros.
        for ci in leaves
            @test tm.tracers[2, ci] == 0.0
        end

        # Functional form for species B (mesh-cell-index dependent).
        set_species!(tm, :B, ci -> Float64(ci) * 2, leaves)
        for ci in leaves
            @test tm.tracers[2, ci] == 2.0 * Float64(ci)
        end

        # Invalid name.
        @test_throws ArgumentError set_species!(tm, :ghost,
                                                  ones(length(leaves)),
                                                  leaves)

        # Length mismatch.
        @test_throws AssertionError set_species!(tm, :A,
                                                  ones(length(leaves) + 1),
                                                  leaves)

        # Out-of-range index.
        @test_throws AssertionError set_species!(tm, 99,
                                                  ones(length(leaves)),
                                                  leaves)
    end

    @testset "advect_tracers_HG_2d!: pure no-op (byte-equal preservation)" begin
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:3   # 8x8 = 64 leaves
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_2d_fields(mesh)
        tm = TracerMeshHG2D(fields, mesh; n_species = 2,
                             names = [:A, :B])

        # Set up nontrivial state.
        set_species!(tm, :A, ci -> Float64(ci) * 0.123, leaves)
        set_species!(tm, :B, ci -> sin(ci * 0.5) + 7.0, leaves)
        snap = copy(tm.tracers)

        # 100 advection calls — should never mutate the matrix.
        for _ in 1:100
            advect_tracers_HG_2d!(tm, 1.234e-3)
        end
        @test tm.tracers == snap   # byte-exact (`==` for Matrix{Float64})
    end

    @testset "Phase 11 + M2-2 invariants in 2D: byte-equal under det_step" begin
        # Build a small 2D Cholesky-sector field set (4×4 mesh) and a
        # multi-species tracer mesh. Drive `det_step_2d_berry_HG!` for
        # K steps with no refinement → tracer matrix unchanged byte-
        # exact.
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:2   # 4x4 = 16 leaves
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)

        # Cold-limit isotropic IC: ρ = 1, u = (0, 0), P = 0.1 ⇒ uniform.
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = cholesky_sector_state_from_primitive(1.0, 0.0, 0.0, 0.1,
                                                      (cx, cy))
            write_detfield_2d!(fields, ci, v)
        end

        tm = TracerMeshHG2D(fields, mesh; n_species = 2,
                             names = [:rho_dust, :rho_gas])
        # Per-leaf species pattern.
        set_species!(tm, :rho_dust,
                      ci -> 0.01 * (Float64(ci) % 5), leaves)
        set_species!(tm, :rho_gas,
                      ci -> 1.0 + 0.1 * sin(Float64(ci)), leaves)
        snap = copy(tm.tracers)

        bc = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                  (PERIODIC, PERIODIC)))
        dt = 1e-3
        # 5 deterministic Newton steps — no refinement events, no
        # writes to the tracer matrix.
        for _ in 1:5
            det_step_2d_berry_HG!(fields, mesh, frame, leaves, bc, dt;
                                   project_kind = :none)
            advect_tracers_HG_2d!(tm, dt)
        end

        @test tm.tracers == snap   # byte-exact preservation
    end

    @testset "refine event: parent → children prolongation (mass conservation)" begin
        mesh = HierarchicalMesh{2}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))   # 4 leaves
        leaves = enumerate_leaves(mesh)
        @test length(leaves) == 4
        fields = allocate_cholesky_2d_fields(mesh)
        tm = TracerMeshHG2D(fields, mesh; n_species = 2,
                             names = [:A, :B])

        # Mark each leaf with a unique value per species; the parent's
        # value `(c_A_pre, c_B_pre)` is unique by construction so the
        # 4 post-refine children can be identified by their tuple.
        for (k, ci) in enumerate(leaves)
            tm.tracers[1, ci] = Float64(k)            # species A
            tm.tracers[2, ci] = Float64(k) * 10.0     # species B
        end

        # Mark parent volume V_parent = 0.25 (unit-square / 4 = 0.25)
        # and child volume V_child = V_parent / 4 = 0.0625.
        target_parent = leaves[1]
        c_A_pre = tm.tracers[1, target_parent]
        c_B_pre = tm.tracers[2, target_parent]
        V_parent = 0.25
        # Mass before refine = c_parent · V_parent + 3 untouched
        # leaves at value c_k · V_parent for k = 2, 3, 4.
        mass_A_pre = c_A_pre * V_parent
        mass_B_pre = c_B_pre * V_parent

        # Register both the field-set and tracer listeners.
        handle_fs = register_field_set_on_refine!(fields, mesh)
        handle_tm = register_tracers_on_refine_2d!(tm)
        try
            refine_cells!(mesh, [target_parent])
            new_leaves = enumerate_leaves(mesh)
            @test length(new_leaves) == 7   # 1 split into 4 ⇒ 4 + 3 = 7

            # The 4 new children all carry the parent's `(c_A_pre,
            # c_B_pre)` tuple. Identify them by tuple match.
            new_child_indices = [ci for ci in new_leaves
                                  if tm.tracers[1, ci] == c_A_pre &&
                                     tm.tracers[2, ci] == c_B_pre]
            @test length(new_child_indices) == 4

            # Mass conservation: Σ c_child · V_child = c_parent · V_parent.
            V_child = V_parent / 4.0
            mass_A_post = sum(tm.tracers[1, ci] * V_child
                                for ci in new_child_indices)
            mass_B_post = sum(tm.tracers[2, ci] * V_child
                                for ci in new_child_indices)
            @test mass_A_post == mass_A_pre   # bit-exact
            @test mass_B_post == mass_B_pre   # bit-exact

            # Untouched leaves preserve the pre-refine value-multiset.
            # HG renumbers mesh-cell indices on refine; the listener's
            # `index_remap` correctly relocates per-cell values, but
            # the per-leaf values may now sit at different ci indices
            # than pre-refine ⇒ assert on the multiset, not per-ci.
            pre_tuples = sort([(Float64(k), Float64(k) * 10.0)
                                for k in 1:4])
            post_tuples = sort([(tm.tracers[1, ci], tm.tracers[2, ci])
                                 for ci in new_leaves])
            buf = copy(post_tuples)
            # Remove 3 extra parent-tuple copies (parent appears 1×
            # pre-refine, 4× post-refine ⇒ 3 extras).
            for _ in 1:3
                idx = findfirst(t -> t == (c_A_pre, c_B_pre), buf)
                deleteat!(buf, idx)
            end
            sort!(buf)
            @test buf == pre_tuples
        finally
            unregister_refinement_listener!(mesh, handle_tm)
            unregister_refinement_listener!(mesh, handle_fs)
        end
    end

    @testset "coarsen event: children → parent volume-weighted mean" begin
        # Strategy: register listeners BEFORE the refine event so the
        # listener prolongates a known parent value into its 4
        # children. Then mutate the children to (1, 2, 3, 4) — using
        # the field-set's `α_1` slot as a marker (1, 2, 3, 4) so we
        # can identify them. Then coarsen and assert the parent tracer
        # value equals (1+2+3+4)/4 = 2.5.
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:2
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves_pre = enumerate_leaves(mesh)
        @test length(leaves_pre) == 16

        fields = allocate_cholesky_2d_fields(mesh)
        tm = TracerMeshHG2D(fields, mesh; n_species = 1, names = [:rho])

        # Mark each leaf with a unique value, set parent's value to a
        # known sentinel.
        for (k, ci) in enumerate(leaves_pre)
            tm.tracers[1, ci] = -42.0 * Float64(k)   # large unique negs
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.0), (0.0, 0.0),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end
        target_parent = leaves_pre[1]
        sentinel_value = 999.0
        tm.tracers[1, target_parent] = sentinel_value

        # Register listeners NOW (before the refine).
        handle_fs = register_field_set_on_refine!(fields, mesh)
        handle_tm = register_tracers_on_refine_2d!(tm)
        try
            # Refine the parent → 4 children inherit `sentinel_value`
            # via the listener.
            refine_cells!(mesh, [target_parent])
            leaves_after_refine = enumerate_leaves(mesh)

            # Find the 4 children: leaves whose tracer value equals
            # `sentinel_value`. (sentinel_value is unique by construction.)
            children = [ci for ci in leaves_after_refine
                          if tm.tracers[1, ci] == sentinel_value]
            @test length(children) == 4

            # Mutate children to (1, 2, 3, 4).
            for (k, ci) in enumerate(children)
                tm.tracers[1, ci] = Float64(k)
            end
            V_parent = 0.25
            V_child = V_parent / 4.0
            mass_pre = sum(tm.tracers[1, ci] * V_child for ci in children)
            @test mass_pre ≈ (1+2+3+4) * V_child rtol = 1e-14

            # Coarsen the children back to the parent. HG's `coarsen_cells!`
            # operates on the index of the (now non-leaf) parent.
            coarsen_cells!(mesh, [target_parent])
            leaves_after_coarsen = enumerate_leaves(mesh)
            @test length(leaves_after_coarsen) == 16   # back to pre-refine

            # `target_parent` is again a leaf; its tracer = mean of
            # children = (1+2+3+4)/4 = 2.5.
            @test tm.tracers[1, target_parent] ≈ 2.5 rtol = 1e-14

            mass_post = tm.tracers[1, target_parent] * V_parent
            @test mass_post ≈ mass_pre rtol = 1e-14
        finally
            unregister_refinement_listener!(mesh, handle_tm)
            unregister_refinement_listener!(mesh, handle_fs)
        end
    end

    @testset "multi-species independence under refine/coarsen" begin
        # Two species A and B with disjoint per-cell patterns; refine
        # → tracer rows do not cross-contaminate. The independence
        # invariant: the listener's update on one species row does
        # not perturb the other species row.
        #
        # Test design:
        #   1. Snapshot species A values pre-refine.
        #   2. Refine. Listener updates both rows in lock-step.
        #   3. Verify that after the refine, the multiset of
        #      (A_value, B_value) tuples per leaf matches the
        #      expected mapping: untouched leaves preserve their
        #      tuple; the 4 new children carry the parent's tuple.
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:2   # 16 leaves
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_2d_fields(mesh)
        tm = TracerMeshHG2D(fields, mesh; n_species = 2,
                             names = [:A, :B])

        # Disjoint ranges: A in (1000, 2000), B in (-2000, -1000).
        # The (A, B) tuple per leaf is unique by construction.
        for (k, ci) in enumerate(leaves)
            tm.tracers[1, ci] = 1000.0 + Float64(k)
            tm.tracers[2, ci] = -1000.0 - Float64(k)
        end

        target = leaves[3]
        c_A_pre = tm.tracers[1, target]
        c_B_pre = tm.tracers[2, target]
        # Independence pair: the parent's (A, B) tuple is bound.
        @test c_A_pre == 1003.0
        @test c_B_pre == -1003.0

        # Pre-refine multiset of (A, B) tuples (only leaves count).
        pre_tuples = sort([(tm.tracers[1, ci], tm.tracers[2, ci])
                            for ci in leaves])

        handle_fs = register_field_set_on_refine!(fields, mesh)
        handle_tm = register_tracers_on_refine_2d!(tm)
        try
            refine_cells!(mesh, [target])
            new_leaves = enumerate_leaves(mesh)
            @test length(new_leaves) == 19   # 16 - 1 + 4

            # Post-refine multiset of (A, B) tuples per LEAF.
            post_tuples = sort([(tm.tracers[1, ci], tm.tracers[2, ci])
                                 for ci in new_leaves])

            # Independence: the 4 new children inherit (c_A_pre,
            # c_B_pre) exactly. Total of 19 leaves: 15 untouched + 4
            # new children with the parent's tuple.
            n_match = count(t -> t == (c_A_pre, c_B_pre), post_tuples)
            @test n_match == 4

            # Conservation by tuple-multiset: removing the 4 new
            # children's (parent) tuples and the parent's removed
            # tuple, the remaining 15 should match the pre-refine
            # set with the parent removed. (The parent's tuple
            # appears once in pre_tuples and 4 times in post_tuples.)
            @test n_match == 4
            # Multiset minus 3 extra parent-tuple copies (4 post - 1
            # pre = 3 extras in post).
            post_minus_extras = sort([t for t in post_tuples])
            # Drop 3 occurrences of (c_A_pre, c_B_pre) from
            # post_tuples; result should equal pre_tuples.
            buf = copy(post_tuples)
            for _ in 1:3
                idx = findfirst(t -> t == (c_A_pre, c_B_pre), buf)
                deleteat!(buf, idx)
            end
            sort!(buf)
            @test buf == pre_tuples
        finally
            unregister_refinement_listener!(mesh, handle_tm)
            unregister_refinement_listener!(mesh, handle_fs)
        end
    end

    @testset "refine listener: column dimension grows correctly" begin
        mesh = HierarchicalMesh{2}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))   # 4 leaves
        fields = allocate_cholesky_2d_fields(mesh)
        tm = TracerMeshHG2D(fields, mesh; n_species = 3)
        @test size(tm.tracers, 2) == hg_n_cells(mesh)
        nc_pre = hg_n_cells(mesh)

        handle_tm = register_tracers_on_refine_2d!(tm)
        try
            refine_cells!(mesh, [enumerate_leaves(mesh)[1]])
            nc_post = hg_n_cells(mesh)
            @test nc_post == nc_pre + 4   # 1 leaf split into 4 (+ parent kept)
            @test size(tm.tracers, 2) == nc_post
            @test size(tm.tracers, 1) == 3   # species count unchanged
        finally
            unregister_refinement_listener!(mesh, handle_tm)
        end
    end

end
