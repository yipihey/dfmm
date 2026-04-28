# test_M3_7b_3d_zero_strain.jl
#
# M3-7b 3D zero-strain regression on the native HG-side EL residual
# (`src/eom.jl::cholesky_el_residual_3D!`) and Newton driver
# (`src/newton_step_HG.jl::det_step_3d_HG!`).
#
# Coverage:
#
#   1. Cold-limit zero-strain fixed point on a 4×4×4 mesh:
#        α_a=1 (a=1,2,3), β_a=0, u_a=0, θ_ab=0, M_vv=(0, 0, 0), uniform
#      (a) Residual evaluates to the machine-precision zero vector when
#          y_n == y_np1 (true fixed point of the per-axis Cholesky-sector
#          EL system in 3D).
#      (b) One Newton step preserves the state byte-equally.
#      (c) Multiple Newton steps preserve the state byte-equally over a
#          longer trajectory (10 steps).
#
#   2. Pack/unpack round-trip:
#        `pack_state_3d ∘ unpack_state_3d! ≡ id` over the 15-dof Newton
#        unknowns.
#
#   3. Face-neighbor table sanity:
#        On a 4×4×4 balanced mesh with REFLECTING BCs in all 3 axes,
#        face-boundary leaves get 0; interior leaves get matching
#        neighbour indices.
#
#   4. EOS-driven cold limit: with `s = -800` (so `Mvv → 0` via
#      underflow), the residual at the IC is identically zero with
#      `M_vv_override = nothing` (EOS branch).
#
# This file does NOT cover the dimension-lift gates (M3-7 design note
# §7.1a + §7.1b); those are `test_M3_7b_dimension_lift_3d.jl`.
#
# See `reference/notes_M3_7_3d_extension.md` §3 + §7 and
# `reference/notes_M3_7b_native_3d_residual.md`.

using Test
using StaticArrays
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, PERIODIC, cell_physical_box,
    n_cells
using dfmm: DetField3D, allocate_cholesky_3d_fields,
    read_detfield_3d, write_detfield_3d!,
    cholesky_el_residual_3D!, cholesky_el_residual_3D,
    pack_state_3d, unpack_state_3d!,
    build_face_neighbor_tables_3d, build_periodic_wrap_tables_3d,
    build_residual_aux_3D,
    det_step_3d_HG!

@testset "M3-7b 3D zero-strain (cold-limit fixed point)" begin

    # Build a 4×4×4 balanced 3D mesh (level 2 → 64 leaves, 73 cells).
    function build_mesh(level::Int = 2)
        mesh = HierarchicalMesh{3}(; balanced = true)
        for _ in 1:level
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        return mesh, enumerate_leaves(mesh)
    end

    mesh, leaves = build_mesh(2)
    frame = EulerianFrame(mesh, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    bc_spec = FrameBoundaries{3}(((REFLECTING, REFLECTING),
                                   (REFLECTING, REFLECTING),
                                   (REFLECTING, REFLECTING)))

    @testset "mesh sanity: 4×4×4 → 64 leaves, 73 total cells" begin
        @test length(leaves) == 64
        @test n_cells(mesh) == 73
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 1: zero-strain (cold-limit) fixed-point IC
    # ─────────────────────────────────────────────────────────────────
    function init_cold_zero_strain!(fields, leaves, frame; α=1.0, β=0.0)
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            cz = 0.5 * (lo[3] + hi[3])
            v = DetField3D((cx, cy, cz), (0.0, 0.0, 0.0),
                            (α, α, α), (β, β, β),
                            0.0, 0.0, 0.0, 1.0)
            write_detfield_3d!(fields, ci, v)
        end
        return fields
    end

    @testset "cold-limit residual = 0 at static IC (4×4×4 mesh)" begin
        fields = allocate_cholesky_3d_fields(mesh)
        init_cold_zero_strain!(fields, leaves, frame)

        aux = build_residual_aux_3D(fields, mesh, frame, leaves, bc_spec;
                                     M_vv_override = (0.0, 0.0, 0.0),
                                     ρ_ref = 1.0)
        y = pack_state_3d(fields, leaves)
        F = similar(y)
        cholesky_el_residual_3D!(F, y, y, aux, 1e-3)

        @test maximum(abs, F) == 0.0
        @test length(F) == 15 * length(leaves)
    end

    @testset "cold-limit one-step preservation: y_{n+1} == y_n" begin
        fields = allocate_cholesky_3d_fields(mesh)
        init_cold_zero_strain!(fields, leaves, frame)

        y_before = pack_state_3d(fields, leaves)
        det_step_3d_HG!(fields, mesh, frame, leaves, bc_spec, 1e-3;
                         M_vv_override = (0.0, 0.0, 0.0),
                         ρ_ref = 1.0)
        y_after = pack_state_3d(fields, leaves)

        @test maximum(abs, y_after .- y_before) == 0.0
    end

    @testset "cold-limit 10-step run preserves the state byte-equally" begin
        fields = allocate_cholesky_3d_fields(mesh)
        init_cold_zero_strain!(fields, leaves, frame)

        y_initial = pack_state_3d(fields, leaves)
        for _ in 1:10
            det_step_3d_HG!(fields, mesh, frame, leaves, bc_spec, 1e-3;
                             M_vv_override = (0.0, 0.0, 0.0),
                             ρ_ref = 1.0)
        end
        y_final = pack_state_3d(fields, leaves)

        @test maximum(abs, y_final .- y_initial) == 0.0
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 2: residual structural contract
    # ─────────────────────────────────────────────────────────────────
    @testset "residual length: 15 dof per cell × N cells" begin
        fields = allocate_cholesky_3d_fields(mesh)
        init_cold_zero_strain!(fields, leaves, frame)
        aux = build_residual_aux_3D(fields, mesh, frame, leaves, bc_spec;
                                     M_vv_override = (0.0, 0.0, 0.0),
                                     ρ_ref = 1.0)
        y = pack_state_3d(fields, leaves)
        F = cholesky_el_residual_3D(y, y, aux, 1e-3)
        @test length(F) == 15 * length(leaves)
        @test maximum(abs, F) == 0.0
    end

    @testset "residual is zero on EOS-driven cold-limit (M_vv via Mvv)" begin
        # `s = -800` makes Mvv underflow to exact 0.0.
        fields = allocate_cholesky_3d_fields(mesh)
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            cz = 0.5 * (lo[3] + hi[3])
            v = DetField3D((cx, cy, cz), (0.0, 0.0, 0.0),
                            (1.0, 1.0, 1.0), (0.0, 0.0, 0.0),
                            0.0, 0.0, 0.0, -800.0)
            write_detfield_3d!(fields, ci, v)
        end
        aux = build_residual_aux_3D(fields, mesh, frame, leaves, bc_spec;
                                     M_vv_override = nothing,
                                     ρ_ref = 1.0)
        y = pack_state_3d(fields, leaves)
        F = similar(y)
        cholesky_el_residual_3D!(F, y, y, aux, 1e-3)
        @test maximum(abs, F) == 0.0
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 3: pack / unpack round-trip
    # ─────────────────────────────────────────────────────────────────
    @testset "pack_state_3d / unpack_state_3d! round-trip" begin
        fields = allocate_cholesky_3d_fields(mesh)
        # Tag each leaf with a unique pattern so a wrong index shows up.
        for (k, ci) in enumerate(leaves)
            v = DetField3D((Float64(k),    Float64(k) + 0.5, Float64(k) + 0.25),
                           (Float64(k) * 2, -Float64(k),     Float64(k) * 0.3),
                           (1.0 + 0.01 * k, 0.5 + 0.01 * k,  0.7 + 0.01 * k),
                           (0.001 * k,     -0.001 * k,       0.002 * k),
                           0.1 * k, 0.05 * k, -0.07 * k, 1.0)
            write_detfield_3d!(fields, ci, v)
        end

        y = pack_state_3d(fields, leaves)
        @test length(y) == 15 * length(leaves)

        # Round-trip: zero out fields, unpack, re-read.
        for ci in leaves
            v_zero = DetField3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                                 (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                                 0.0, 0.0, 0.0, 0.0)
            write_detfield_3d!(fields, ci, v_zero)
        end
        unpack_state_3d!(fields, leaves, y)
        for (k, ci) in enumerate(leaves)
            v = read_detfield_3d(fields, ci)
            @test v.x[1]      == Float64(k)
            @test v.x[2]      == Float64(k) + 0.5
            @test v.x[3]      == Float64(k) + 0.25
            @test v.u[1]      == Float64(k) * 2
            @test v.u[2]      == -Float64(k)
            @test v.u[3]      == Float64(k) * 0.3
            @test v.alphas[1] == 1.0 + 0.01 * k
            @test v.alphas[2] == 0.5 + 0.01 * k
            @test v.alphas[3] == 0.7 + 0.01 * k
            @test v.betas[1]  == 0.001 * k
            @test v.betas[2]  == -0.001 * k
            @test v.betas[3]  == 0.002 * k
            @test v.θ_12      == 0.1 * k
            @test v.θ_13      == 0.05 * k
            @test v.θ_23      == -0.07 * k
            # Entropy was untouched by unpack — still 0 from the zero-out.
            @test v.s         == 0.0
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 4: face-neighbor table sanity
    # ─────────────────────────────────────────────────────────────────
    @testset "face-neighbor table: REFLECTING BCs leave boundaries at 0" begin
        face_lo, face_hi = build_face_neighbor_tables_3d(mesh, leaves, bc_spec)
        @test length(face_lo) == 3
        @test length(face_hi) == 3
        @test length(face_lo[1]) == length(leaves)
        @test length(face_lo[2]) == length(leaves)
        @test length(face_lo[3]) == length(leaves)
        # Every entry must be 0 (boundary) or a valid leaf-major index.
        for a in 1:3
            for i in 1:length(leaves)
                @test 0 <= face_lo[a][i] <= length(leaves)
                @test 0 <= face_hi[a][i] <= length(leaves)
            end
        end
        # On a 4×4×4 grid (16 leaves per face), each axis has 16 leaves
        # on each wall of that axis; their lo (or hi) face neighbour
        # along that axis is 0 (REFLECTING).
        for a in 1:3
            n_lo_zero = count(==(0), face_lo[a])
            n_hi_zero = count(==(0), face_hi[a])
            @test n_lo_zero == 16
            @test n_hi_zero == 16
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 5: triply-periodic configuration sanity
    # ─────────────────────────────────────────────────────────────────
    @testset "triply-periodic BCs: every face has a neighbour" begin
        bc_periodic = FrameBoundaries{3}(((PERIODIC, PERIODIC),
                                            (PERIODIC, PERIODIC),
                                            (PERIODIC, PERIODIC)))
        face_lo, face_hi = build_face_neighbor_tables_3d(mesh, leaves, bc_periodic)
        # Triply-periodic: every face should map to a real leaf.
        for a in 1:3
            @test count(==(0), face_lo[a]) == 0
            @test count(==(0), face_hi[a]) == 0
        end
        # The wrap tables should fire on the boundary leaves: 16 per
        # face × 6 faces, but each leaf has wraps only at the walls
        # along the axis. On a 4×4×4 mesh, axis a has 16 leaves on its
        # lo wall and 16 on its hi wall; their wrap offsets are -L_a
        # and +L_a respectively.
        wrap_lo, wrap_hi = build_periodic_wrap_tables_3d(mesh, frame, leaves,
                                                          face_lo, face_hi)
        for a in 1:3
            @test count(<(-0.5), wrap_lo[a]) == 16
            @test count(>(0.5),  wrap_hi[a]) == 16
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 6: zero-strain residual under triply-periodic BCs
    # ─────────────────────────────────────────────────────────────────
    @testset "zero-strain residual = 0 under triply-periodic BCs" begin
        bc_periodic = FrameBoundaries{3}(((PERIODIC, PERIODIC),
                                            (PERIODIC, PERIODIC),
                                            (PERIODIC, PERIODIC)))
        fields = allocate_cholesky_3d_fields(mesh)
        init_cold_zero_strain!(fields, leaves, frame)
        aux = build_residual_aux_3D(fields, mesh, frame, leaves, bc_periodic;
                                     M_vv_override = (0.0, 0.0, 0.0),
                                     ρ_ref = 1.0)
        y = pack_state_3d(fields, leaves)
        F = similar(y)
        cholesky_el_residual_3D!(F, y, y, aux, 1e-3)
        @test maximum(abs, F) == 0.0
    end
end
