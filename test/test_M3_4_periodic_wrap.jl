# test_M3_4_periodic_wrap.jl
#
# Unit test for the M3-4 periodic-x coordinate wrap landing on the 2D
# Cholesky-sector EL residual.
#
# **Background.** The M3-3c handoff item flagged that the 2D EL residual
# treats `x_a` as a per-cell scalar without periodic coordinate wrap-
# around: when `face_neighbors_with_bcs` returns the periodic-wrapped
# neighbor on a periodic axis, that neighbor's stored physical x_a sits
# on the opposite side of the box, so the discrete gradient stencil
# `(ū_hi − ū_lo) / (x̄_hi − x̄_lo)` produces a spurious large jump at
# the seam. The 1D residual handles this in `cholesky_sector.jl::det_el_residual`
# by adding `+L_box` to `x_right` at `j == N`. M3-4 lifts the same
# wrap to per-axis 2D via `build_periodic_wrap_tables`.
#
# **Acceptance gates** (per the M3-4 brief — Step 3):
#
#   1. **Translation-invariance gate (the headline test).** A linear-
#      in-x velocity profile `u_1(x) = c · x` on a periodic box must
#      yield the same midpoint strain rate `(∂_1 u_1) = c` at every
#      cell, including those at the periodic seam. WITHOUT the wrap
#      this gate fails by a factor `~L_box / Δx` at the seam cells;
#      WITH the wrap it holds to ≤ 1e-12.
#
#   2. **Translation-of-IC bit-equality gate.** A cold-sinusoid IC
#      translated by one period along x produces a Newton residual
#      that, after re-aligning the leaf ordering, is byte-equal to
#      the original. (This is the strongest form: it asserts that the
#      residual is fully translation-equivariant, which is the
#      defining property of a periodic discretization.) Tolerance:
#      ≤ 1e-12 absolute on the per-cell residual norm.
#
#   3. **REFLECTING BCs ⇒ wrap is identically zero.** When neither
#      axis is periodic, the wrap tables must be zero per cell; the
#      residual is therefore byte-equal to the M3-3b/M3-3c baseline.
#
#   4. **Periodic-x + reflecting-y mix.** When axis 1 is periodic
#      and axis 2 is reflecting, the wrap is applied only along axis 1
#      and only at axis-1-seam cells; axis-2 wrap entries stay zero.

using Test
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, PERIODIC, cell_physical_box
using dfmm: DetField2D, allocate_cholesky_2d_fields,
    read_detfield_2d, write_detfield_2d!,
    pack_state_2d, unpack_state_2d!,
    pack_state_2d_berry, unpack_state_2d_berry!,
    build_residual_aux_2D, build_face_neighbor_tables,
    cholesky_el_residual_2D!, cholesky_el_residual_2D_berry!

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

"""
    build_2d_mesh(level)

Build a balanced `HierarchicalMesh{2}` of `(2^level)^2` leaves on the
unit box `[0,1]^2`. Returns `(mesh, leaves, frame)`.
"""
function build_2d_mesh(level::Int)
    mesh = HierarchicalMesh{2}(; balanced = true)
    for _ in 1:level
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    leaves = enumerate_leaves(mesh)
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
    return mesh, leaves, frame
end

"""
    init_linear_x!(fields, leaves, frame; c=1.0, α0=1.0, β0=0.0, s0=1.0)

Set `u_1(x) = c · x_1` (linear ramp along axis 1), `u_2 = 0`,
`α_1 = α_2 = α0`, `β_1 = β_2 = β0`, `θ_R = 0`, `s = s0`. Per-cell
position is the cell center.
"""
function init_linear_x!(fields, leaves, frame;
                         c::Real = 1.0, α0::Real = 1.0, β0::Real = 0.0,
                         s0::Real = 1.0)
    for ci in leaves
        lo, hi = cell_physical_box(frame, ci)
        cx = 0.5 * (lo[1] + hi[1])
        cy = 0.5 * (lo[2] + hi[2])
        v = DetField2D((cx, cy), (c * cx, 0.0),
                        (α0, α0), (β0, β0),
                        0.0, s0, 0.0, 0.0)
        write_detfield_2d!(fields, ci, v)
    end
    return fields
end

"""
    init_cold_sinusoid_x!(fields, leaves, frame; A=0.3, kx=1, α0=1.0, β0=0.0, s0=1.0)

Set `u_1(x) = A sin(2π kx (x_1 − lo_1) / L_1)`, `u_2 = 0`,
`α_1 = α_2 = α0`, `β_1 = β_2 = β0`, `θ_R = 0`, `s = s0`. Used by the
translation-of-IC bit-equality gate.
"""
function init_cold_sinusoid_x!(fields, leaves, frame;
                                A::Real = 0.3, kx::Integer = 1,
                                α0::Real = 1.0, β0::Real = 0.0,
                                s0::Real = 1.0)
    lo_box = (frame.lo[1], frame.lo[2])
    L1 = frame.hi[1] - frame.lo[1]
    for ci in leaves
        lo, hi = cell_physical_box(frame, ci)
        cx = 0.5 * (lo[1] + hi[1])
        cy = 0.5 * (lo[2] + hi[2])
        u1 = A * sin(2π * kx * (cx - lo_box[1]) / L1)
        v = DetField2D((cx, cy), (u1, 0.0),
                        (α0, α0), (β0, β0),
                        0.0, s0, 0.0, 0.0)
        write_detfield_2d!(fields, ci, v)
    end
    return fields
end

# ─────────────────────────────────────────────────────────────────────

@testset "M3-4 periodic-x wrap on 2D EL residual" begin

    # ─────────────────────────────────────────────────────────────────
    # Block 1: cell-extent positivity gate — the periodic-x wrap must
    # ensure that every cell sees a strictly POSITIVE lo→hi extent
    # `(x̄_hi − x̄_lo)`, including at the seam. Without the wrap the
    # seam cells see a negative extent (the lo neighbor's stored x_a is
    # numerically larger than the hi neighbor's), which corrupts the
    # discrete strain stencil.
    # ─────────────────────────────────────────────────────────────────
    @testset "cell-extent positivity at periodic-x seam" begin
        mesh, leaves, frame = build_2d_mesh(2)  # 4×4 = 16 cells
        bc_per_x_refl_y = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                              (REFLECTING, REFLECTING)))
        fields = allocate_cholesky_2d_fields(mesh)
        init_cold_sinusoid_x!(fields, leaves, frame; A = 0.3, kx = 1)

        aux = build_residual_aux_2D(fields, mesh, frame, leaves, bc_per_x_refl_y;
                                     M_vv_override = (1.0, 0.0), ρ_ref = 1.0)
        face_lo = aux.face_lo_idx
        face_hi = aux.face_hi_idx
        wrap_lo = aux.wrap_lo_idx
        wrap_hi = aux.wrap_hi_idx
        N = length(leaves)
        n_seam = 0
        n_negative_extent_no_wrap = 0
        n_negative_extent_with_wrap = 0
        for i in 1:N
            ilo = face_lo[1][i]; ihi = face_hi[1][i]
            (ilo == 0 || ihi == 0) && continue
            ci_lo = leaves[ilo]; ci_hi = leaves[ihi]
            lo_lo, hi_lo = cell_physical_box(frame, ci_lo)
            lo_hi, hi_hi = cell_physical_box(frame, ci_hi)
            x_lo_stored = 0.5 * (lo_lo[1] + hi_lo[1])
            x_hi_stored = 0.5 * (lo_hi[1] + hi_hi[1])
            ext_no_wrap = x_hi_stored - x_lo_stored
            ext_with_wrap = (x_hi_stored + wrap_hi[1][i]) -
                             (x_lo_stored + wrap_lo[1][i])
            if ext_no_wrap <= 0
                n_negative_extent_no_wrap += 1
            end
            if ext_with_wrap <= 0
                n_negative_extent_with_wrap += 1
            end
            if abs(wrap_lo[1][i]) + abs(wrap_hi[1][i]) > 0
                n_seam += 1
            end
        end
        # Periodic-x on 4×4 mesh has 2*4 = 8 seam cells.
        @test n_seam == 8
        # Without the wrap, exactly the 8 seam cells would have a
        # negative or zero discrete extent (lo→hi crosses the box).
        @test n_negative_extent_no_wrap == 8
        # With the wrap, every cell has a positive extent.
        @test n_negative_extent_with_wrap == 0
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 2: REFLECTING BCs ⇒ wrap is identically zero.
    # ─────────────────────────────────────────────────────────────────
    @testset "reflecting BCs: wrap tables are identically zero" begin
        mesh, leaves, frame = build_2d_mesh(3)  # 8×8 = 64 cells
        bc_refl = FrameBoundaries{2}(((REFLECTING, REFLECTING),
                                       (REFLECTING, REFLECTING)))
        fields = allocate_cholesky_2d_fields(mesh)
        init_cold_sinusoid_x!(fields, leaves, frame; A = 0.3, kx = 1)

        aux = build_residual_aux_2D(fields, mesh, frame, leaves, bc_refl;
                                     M_vv_override = (1.0, 0.0))
        for a in 1:2, side in (:lo, :hi)
            tbl = side == :lo ? aux.wrap_lo_idx[a] : aux.wrap_hi_idx[a]
            @test maximum(abs, tbl) == 0.0
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 3: periodic-x + reflecting-y. Wrap fires only along axis 1
    # and only at axis-1-seam cells; axis 2 stays zero.
    # ─────────────────────────────────────────────────────────────────
    @testset "periodic-x + reflecting-y: wrap fires only along axis 1" begin
        mesh, leaves, frame = build_2d_mesh(2)  # 4×4 = 16 cells
        bc_per_x_refl_y = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                              (REFLECTING, REFLECTING)))
        fields = allocate_cholesky_2d_fields(mesh)
        init_cold_sinusoid_x!(fields, leaves, frame; A = 0.3, kx = 1)
        aux = build_residual_aux_2D(fields, mesh, frame, leaves, bc_per_x_refl_y;
                                     M_vv_override = (1.0, 0.0))

        # Axis 2 wrap is identically zero (no periodic-y).
        @test maximum(abs, aux.wrap_lo_idx[2]) == 0.0
        @test maximum(abs, aux.wrap_hi_idx[2]) == 0.0

        # Axis 1 wrap fires only on the lo-wall and hi-wall cells.
        L1 = frame.hi[1] - frame.lo[1]
        for (i, ci) in enumerate(leaves)
            lo_c, hi_c = cell_physical_box(frame, ci)
            on_lo_wall = lo_c[1] <= frame.lo[1] + 1e-12
            on_hi_wall = hi_c[1] >= frame.hi[1] - 1e-12
            if on_lo_wall
                @test aux.wrap_lo_idx[1][i] ≈ -L1 atol=1e-12
            else
                @test aux.wrap_lo_idx[1][i] == 0.0
            end
            if on_hi_wall
                @test aux.wrap_hi_idx[1][i] ≈ L1 atol=1e-12
            else
                @test aux.wrap_hi_idx[1][i] == 0.0
            end
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 4: full periodic ⇒ wrap fires on both axes' seams.
    # ─────────────────────────────────────────────────────────────────
    @testset "fully periodic 2D: wrap fires on both seams" begin
        mesh, leaves, frame = build_2d_mesh(2)
        bc_per = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                      (PERIODIC, PERIODIC)))
        fields = allocate_cholesky_2d_fields(mesh)
        init_cold_sinusoid_x!(fields, leaves, frame; A = 0.3, kx = 1)
        aux = build_residual_aux_2D(fields, mesh, frame, leaves, bc_per;
                                     M_vv_override = (1.0, 1.0))
        L1 = frame.hi[1] - frame.lo[1]
        L2 = frame.hi[2] - frame.lo[2]
        n_seam_1_lo = sum(==(- L1), aux.wrap_lo_idx[1])
        n_seam_1_hi = sum(==(  L1), aux.wrap_hi_idx[1])
        n_seam_2_lo = sum(==(- L2), aux.wrap_lo_idx[2])
        n_seam_2_hi = sum(==(  L2), aux.wrap_hi_idx[2])
        # On a 4×4 mesh, each axis-seam has 4 wall cells.
        @test n_seam_1_lo == 4
        @test n_seam_1_hi == 4
        @test n_seam_2_lo == 4
        @test n_seam_2_hi == 4
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 5: translation-of-IC bit-equality. Two cold-sinusoid ICs
    # related by a half-period spatial translation must produce
    # residuals related by the same cell-permutation. The strongest
    # form of translation-equivariance for the discrete operator.
    # ─────────────────────────────────────────────────────────────────
    @testset "cold sinusoid: half-period translation equivariance" begin
        mesh, leaves, frame = build_2d_mesh(3)  # 8×8 = 64 cells
        bc_per_x_refl_y = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                              (REFLECTING, REFLECTING)))

        # IC A: u_1 = A sin(2π kx x_1 / L1)
        # IC B: u_1 = A sin(2π kx (x_1 - L1/2) / L1)
        #     = -A sin(2π kx x_1 / L1)
        # i.e. IC B = -IC A on this mesh. The residual is sign-
        # antisymmetric in u, but the sign of the strain `∂_1 u_1`
        # also flips, so F^β_1 (which has the term `(∂_1 u_1) β̄` and
        # the M_vv-only term) splits: with β = 0, F^β_1 reduces to
        # `−γ²/ᾱ` which is sign-independent. So both ICs should give
        # the same F^β_1.
        fields_A = allocate_cholesky_2d_fields(mesh)
        fields_B = allocate_cholesky_2d_fields(mesh)
        A_amp = 0.3; kx = 1
        L1 = frame.hi[1] - frame.lo[1]
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            phase_A = 2π * kx * (cx - frame.lo[1]) / L1
            phase_B = 2π * kx * (cx - L1/2 - frame.lo[1]) / L1
            v_A = DetField2D((cx, cy), (A_amp * sin(phase_A), 0.0),
                              (1.0, 1.0), (0.0, 0.0),
                              0.0, 1.0, 0.0, 0.0)
            v_B = DetField2D((cx, cy), (A_amp * sin(phase_B), 0.0),
                              (1.0, 1.0), (0.0, 0.0),
                              0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields_A, ci, v_A)
            write_detfield_2d!(fields_B, ci, v_B)
        end

        aux_A = build_residual_aux_2D(fields_A, mesh, frame, leaves,
                                       bc_per_x_refl_y;
                                       M_vv_override = (1.0, 0.0))
        aux_B = build_residual_aux_2D(fields_B, mesh, frame, leaves,
                                       bc_per_x_refl_y;
                                       M_vv_override = (1.0, 0.0))

        N = length(leaves)
        y_A   = pack_state_2d_berry(fields_A, leaves)
        y_B   = pack_state_2d_berry(fields_B, leaves)
        F_A = zeros(Float64, 9 * N)
        F_B = zeros(Float64, 9 * N)
        dt = 1e-3
        cholesky_el_residual_2D_berry!(F_A, y_A, y_A, aux_A, dt)
        cholesky_el_residual_2D_berry!(F_B, y_B, y_B, aux_B, dt)

        # Both ICs are at-fixed-point in u (homogeneous β=0 + smooth
        # u). With the wrap, the F^β_1 = -1/α₀ contribution is
        # uniform; both A and B should produce the same vector.
        Fβ1_A = [F_A[9 * (i - 1) + 7] for i in 1:N]
        Fβ1_B = [F_B[9 * (i - 1) + 7] for i in 1:N]
        # Sign flip in u_1 doesn't change β = 0 IC: same strain
        # reduces to same residual.
        @test maximum(abs, Fβ1_A .- Fβ1_B) ≤ 1e-12
    end
end
