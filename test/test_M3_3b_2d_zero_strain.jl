# test_M3_3b_2d_zero_strain.jl
#
# M3-3b 2D zero-strain regression on the native HG-side EL residual
# (`src/eom.jl::cholesky_el_residual_2D!`) and Newton driver
# (`src/newton_step_HG.jl::det_step_2d_HG!`).
#
# Coverage:
#
#   1. Cold-limit zero-strain fixed point:
#        α=1, β=0, u=0, M_vv=(0, 0), uniform 4×4 mesh
#      (a) Residual evaluates to the machine-precision zero vector when
#          y_n == y_np1 (true fixed point of the per-axis Cholesky-sector
#          EL system).
#      (b) One Newton step preserves the state byte-equally.
#      (c) Multiple Newton steps preserve the state byte-equally over a
#          longer trajectory (10 steps).
#
#   2. Pack/unpack round-trip:
#        `pack_state_2d ∘ unpack_state_2d! ≡ id` over the 8-dof Newton
#        unknowns.
#
#   3. Face-neighbor table sanity:
#        On a 4×4 balanced mesh with REFLECTING BCs, lo-axis-side leaves
#        get 0 (no neighbour, mirror-self downstream), interior leaves
#        get matching neighbour indices.
#
# This file does NOT cover the dimension-lift gate (§6.1 of the design
# note); that's `test_M3_3b_dimension_lift_zero_strain.jl`.
#
# See `reference/notes_M3_3_2d_cholesky_berry.md` §3 (M3-3b chapter)
# and `reference/notes_M3_3b_native_residual.md` for the status note.

using Test
using StaticArrays
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, PERIODIC, cell_physical_box,
    n_cells
using dfmm: DetField2D, allocate_cholesky_2d_fields,
    read_detfield_2d, write_detfield_2d!,
    cholesky_el_residual_2D!, cholesky_el_residual_2D,
    pack_state_2d, unpack_state_2d!,
    build_face_neighbor_tables, build_residual_aux_2D,
    det_step_2d_HG!

@testset "M3-3b 2D zero-strain (cold-limit fixed point)" begin

    # Build a 4×4 balanced 2D mesh (level 2 → 16 leaves, 21 cells).
    function build_mesh(level::Int = 2)
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:level
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        return mesh, enumerate_leaves(mesh)
    end

    mesh, leaves = build_mesh(2)
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
    bc_spec = FrameBoundaries{2}(((REFLECTING, REFLECTING),
                                   (REFLECTING, REFLECTING)))

    @testset "mesh sanity: 4×4 → 16 leaves, 21 total cells" begin
        @test length(leaves) == 16
        @test n_cells(mesh) == 21
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 1: zero-strain (cold-limit) fixed-point IC
    # ─────────────────────────────────────────────────────────────────
    function init_cold_zero_strain!(fields, leaves, frame; α=1.0, β=0.0)
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = DetField2D((cx, cy), (0.0, 0.0),
                            (α, α), (β, β),
                            0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end
        return fields
    end

    @testset "cold-limit residual = 0 at static IC (4×4 mesh)" begin
        fields = allocate_cholesky_2d_fields(mesh)
        init_cold_zero_strain!(fields, leaves, frame)

        aux = build_residual_aux_2D(fields, mesh, frame, leaves, bc_spec;
                                     M_vv_override = (0.0, 0.0),
                                     ρ_ref = 1.0)
        y = pack_state_2d(fields, leaves)
        F = similar(y)
        cholesky_el_residual_2D!(F, y, y, aux, 1e-3)

        # The residual should evaluate to identically zero at the
        # cold-limit fixed point (M_vv = 0 everywhere, β = 0, u = 0,
        # uniform IC).
        @test maximum(abs, F) == 0.0
        @test length(F) == 8 * length(leaves)
    end

    @testset "cold-limit one-step preservation: y_{n+1} == y_n" begin
        fields = allocate_cholesky_2d_fields(mesh)
        init_cold_zero_strain!(fields, leaves, frame)

        y_before = pack_state_2d(fields, leaves)
        det_step_2d_HG!(fields, mesh, frame, leaves, bc_spec, 1e-3;
                         M_vv_override = (0.0, 0.0),
                         ρ_ref = 1.0)
        y_after = pack_state_2d(fields, leaves)

        # Newton on a true fixed point should converge in 0 or 1
        # iterations to the IC byte-equally.
        @test maximum(abs, y_after .- y_before) == 0.0
    end

    @testset "cold-limit 10-step run preserves the state byte-equally" begin
        fields = allocate_cholesky_2d_fields(mesh)
        init_cold_zero_strain!(fields, leaves, frame)

        y_initial = pack_state_2d(fields, leaves)
        for _ in 1:10
            det_step_2d_HG!(fields, mesh, frame, leaves, bc_spec, 1e-3;
                             M_vv_override = (0.0, 0.0),
                             ρ_ref = 1.0)
        end
        y_final = pack_state_2d(fields, leaves)

        @test maximum(abs, y_final .- y_initial) == 0.0
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 2: residual structural contract
    # ─────────────────────────────────────────────────────────────────
    @testset "residual length: 8 dof per cell × N cells" begin
        fields = allocate_cholesky_2d_fields(mesh)
        init_cold_zero_strain!(fields, leaves, frame)
        aux = build_residual_aux_2D(fields, mesh, frame, leaves, bc_spec;
                                     M_vv_override = (0.0, 0.0),
                                     ρ_ref = 1.0)
        y = pack_state_2d(fields, leaves)
        F = cholesky_el_residual_2D(y, y, aux, 1e-3)
        @test length(F) == 8 * length(leaves)
        @test maximum(abs, F) == 0.0
    end

    @testset "residual is zero on EOS-driven cold-limit (M_vv via Mvv)" begin
        # Set entropy to a value that gives M_vv ≈ 0 (cold limit).
        # With Γ = 5/3 and J = 0.25 (cell extent = 0.25, ρ_ref = 1
        # → J = 1/ρ = 1.0; per-axis cell extent gives Δm_a = 0.25,
        # cell extent / Δm = 1, so J = 1), `Mvv(1, s) = exp(s)`.
        # We pick s = -800 so `Mvv` underflows to exact 0.0.
        fields = allocate_cholesky_2d_fields(mesh)
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = DetField2D((cx, cy), (0.0, 0.0),
                            (1.0, 1.0), (0.0, 0.0),
                            0.0, -800.0, 0.0, 0.0)  # s = -800 → cold
            write_detfield_2d!(fields, ci, v)
        end
        aux = build_residual_aux_2D(fields, mesh, frame, leaves, bc_spec;
                                     M_vv_override = nothing,   # use EOS
                                     ρ_ref = 1.0)
        y = pack_state_2d(fields, leaves)
        F = similar(y)
        cholesky_el_residual_2D!(F, y, y, aux, 1e-3)
        @test maximum(abs, F) == 0.0
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 3: pack / unpack round-trip
    # ─────────────────────────────────────────────────────────────────
    @testset "pack_state_2d / unpack_state_2d! round-trip" begin
        fields = allocate_cholesky_2d_fields(mesh)
        # Tag each leaf with a unique pattern so a wrong index shows up.
        for (k, ci) in enumerate(leaves)
            v = DetField2D((Float64(k), Float64(k) + 0.5),
                            (Float64(k) * 2, -Float64(k)),
                            (1.0 + 0.01 * k, 0.5 + 0.01 * k),
                            (0.001 * k, -0.001 * k),
                            0.1 * k, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        y = pack_state_2d(fields, leaves)
        @test length(y) == 8 * length(leaves)

        # Round-trip: zero out fields, unpack, re-read.
        for ci in leaves
            v_zero = DetField2D((0.0, 0.0), (0.0, 0.0),
                                 (0.0, 0.0), (0.0, 0.0),
                                 0.0, 0.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v_zero)
        end
        unpack_state_2d!(fields, leaves, y)
        for (k, ci) in enumerate(leaves)
            v = read_detfield_2d(fields, ci)
            @test v.x[1]      == Float64(k)
            @test v.x[2]      == Float64(k) + 0.5
            @test v.u[1]      == Float64(k) * 2
            @test v.u[2]      == -Float64(k)
            @test v.alphas[1] == 1.0 + 0.01 * k
            @test v.alphas[2] == 0.5 + 0.01 * k
            @test v.betas[1]  == 0.001 * k
            @test v.betas[2]  == -0.001 * k
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 4: face-neighbor table sanity
    # ─────────────────────────────────────────────────────────────────
    @testset "face_lo_idx / face_hi_idx contracts on REFLECTING BCs" begin
        face_lo_idx, face_hi_idx = build_face_neighbor_tables(mesh, leaves, bc_spec)
        # Two axes.
        @test length(face_lo_idx) == 2
        @test length(face_hi_idx) == 2
        @test length(face_lo_idx[1]) == length(leaves)
        @test length(face_hi_idx[1]) == length(leaves)
        @test length(face_lo_idx[2]) == length(leaves)
        @test length(face_hi_idx[2]) == length(leaves)

        # On a 4×4 mesh with REFLECTING BCs: 4 leaves are on the lo-x
        # boundary (axis 1 lo neighbour = 0), 4 on hi-x, etc.
        n_lo_axis1_boundary = count(==(0), face_lo_idx[1])
        n_hi_axis1_boundary = count(==(0), face_hi_idx[1])
        n_lo_axis2_boundary = count(==(0), face_lo_idx[2])
        n_hi_axis2_boundary = count(==(0), face_hi_idx[2])
        @test n_lo_axis1_boundary == 4
        @test n_hi_axis1_boundary == 4
        @test n_lo_axis2_boundary == 4
        @test n_hi_axis2_boundary == 4

        # Symmetry: if i has hi-neighbour j, then j has lo-neighbour i.
        for i in 1:length(leaves)
            j = face_hi_idx[1][i]
            if j > 0
                @test face_lo_idx[1][j] == i
            end
            j = face_hi_idx[2][i]
            if j > 0
                @test face_lo_idx[2][j] == i
            end
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 5: PERIODIC BCs — residual stays zero at uniform IC
    # ─────────────────────────────────────────────────────────────────
    # The 2D Eulerian periodic-wrap stencil does not yet handle the
    # x-coordinate wrap-around (M3-3c handles non-trivial periodic
    # advection). For the M3-3b zero-strain regression we restrict to
    # cases where M_vv_override is supplied, so the residual ignores
    # `x` and the periodic wrap reduces to the same uniform IC the
    # REFLECTING case sees.
    @testset "PERIODIC BCs cold-limit residual = 0 at static IC" begin
        bc_per = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                       (PERIODIC, PERIODIC)))
        fields = allocate_cholesky_2d_fields(mesh)
        init_cold_zero_strain!(fields, leaves, frame)
        aux = build_residual_aux_2D(fields, mesh, frame, leaves, bc_per;
                                     M_vv_override = (0.0, 0.0),
                                     ρ_ref = 1.0)
        y = pack_state_2d(fields, leaves)
        F = similar(y)
        cholesky_el_residual_2D!(F, y, y, aux, 1e-3)
        @test maximum(abs, F) == 0.0
    end
end
