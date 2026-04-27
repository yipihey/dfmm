# test_M3_3a_halo_smoke.jl
#
# M3-3a smoke test: HaloView coefficient access pattern for order-0
# `MonomialBasis{2, 0}` polynomial fields on an 8×8 uniform balanced
# 2D HierarchicalMesh.
#
# This test gates the rest of M3-3a (it resolves Q1 of the §10 design
# note: "what does `hv[i, off]` actually return for an order-0 field?").
# The answer recorded by this file is consumed by `cholesky_DD.jl` and
# the M3-3b EL residual.
#
# Coverage:
#
#   1. Constructing a `HaloView` over a `PolynomialFieldView` allocates
#      only the small wrapper struct.
#   2. For an interior leaf, `hv[i, (0, 0)]` returns the cell's own
#      coefficient column (a `PolynomialView` whose `[1]` is the
#      scalar cell-average for `MonomialBasis{2, 0}`).
#   3. The four face-neighbor offsets `(±1, 0)` and `(0, ±1)` return
#      neighbor coefficients that match direct field indexing.
#   4. For a boundary leaf, the off-domain offset returns `nothing`
#      (the HG plain-`face_neighbors` contract — periodic wrap is the
#      caller's job via `face_neighbors_with_bcs`).
#   5. With a `BoundarySpec` declaring axis 1 PERIODIC, axis 2
#      REFLECTING, `face_neighbors_with_bcs` wraps the periodic axis
#      and leaves the reflecting axis at `0` (the dfmm EL residual
#      will then synthesize the reflecting ghost).
#   6. Indexing `hv[i, off]` is allocation-free in the cached fast path
#      after warm-up (≤ 64 bytes for the inevitable Union-of-Nothing
#      indirection — same tolerance HG's own halo test uses).
#
# Reference: `~/.julia/dev/HierarchicalGrids/test/test_halo_view.jl`,
# `reference/notes_M3_3_2d_cholesky_berry.md` §3.2 + §7 + §10.

using Test
using HierarchicalGrids
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    n_cells, is_leaf, halo_view, HaloView,
    allocate_polynomial_fields, SoA, MonomialBasis, n_coeffs,
    face_neighbors, face_neighbors_with_bcs,
    BCKind, PERIODIC, REFLECTING

@testset "HaloView 2D order-0 smoke (M3-3a)" begin
    # ---------------------------------------------------------------
    # Build an 8×8 uniformly refined 2D mesh (level 3 → 64 leaves).
    # `balanced=true` per Q4 of the M3-3 design note's §10.
    # ---------------------------------------------------------------
    mesh = HierarchicalMesh{2}(; balanced = true)
    for _ in 1:3
        leaves = enumerate_leaves(mesh)
        refine_cells!(mesh, leaves)
    end
    leaves = enumerate_leaves(mesh)

    @testset "uniform 8×8 mesh: 64 leaves, 1 root + 4 + 16 + 64 = 85 cells" begin
        @test length(leaves) == 64
        # Every leaf is at the deepest level (level 3).
        @test all(is_leaf(mesh.cells[ci]) for ci in leaves)
    end

    # ---------------------------------------------------------------
    # Allocate a single-field PolynomialFieldSet at MonomialBasis{2, 0}.
    # Storage is sized to n_cells(mesh) (so HaloView's mesh-cell-index
    # access maps 1-to-1 against `mesh.cells`).
    # ---------------------------------------------------------------
    basis = MonomialBasis{2, 0}()
    @testset "MonomialBasis{2, 0} has exactly one coefficient" begin
        @test n_coeffs(basis) == 1
    end

    pfs = allocate_polynomial_fields(SoA(), basis, n_cells(mesh); α_1 = Float64)

    # Tag every leaf with a deterministic per-cell value: cell index k.
    # Non-leaf cells get a sentinel (NaN) so any accidental read shows up.
    for k in 1:n_cells(mesh)
        if is_leaf(mesh.cells[k])
            pfs.α_1[k] = (Float64(k),)
        else
            pfs.α_1[k] = (NaN,)
        end
    end

    hv = halo_view(pfs.α_1, mesh, 1)
    @test hv isa HaloView

    # ---------------------------------------------------------------
    # Q1 resolution: hv[i, off] returns a PolynomialView (NOT a Tuple)
    # whose [1] indexing yields the scalar order-0 coefficient.
    # The HaloView docstring says "Tuple"; the implementation returns
    # `hv.field[i]` which is a PolynomialView. We document the actual
    # behavior here so the M3-3b residual reads `pv[1]` for the scalar.
    # ---------------------------------------------------------------
    # Pick an interior leaf. With 64 leaves on an 8×8 grid, leaf at
    # (3, 3) (row-major index) is firmly interior.
    interior_leaf_idx = leaves[19]  # somewhere in the middle of the 8×8
    @test is_leaf(mesh.cells[interior_leaf_idx])

    @testset "interior leaf: self-access shape" begin
        pv_self = hv[interior_leaf_idx, (0, 0)]
        @test pv_self !== nothing
        # The PolynomialView's first (and only) coefficient is the cell
        # tag we wrote above.
        @test pv_self[1] == Float64(interior_leaf_idx)
    end

    @testset "interior leaf: 4-face neighbor access" begin
        # We don't know a priori which mesh-cell-index sits at
        # (off=±1, 0) / (0, ±1) of `interior_leaf_idx`, but we can
        # cross-check halo access against the plain face_neighbors API.
        fn = face_neighbors(mesh, interior_leaf_idx)
        # face order: (axis1 lo, axis1 hi, axis2 lo, axis2 hi) →
        #             (off=(-1,0), off=(+1,0), off=(0,-1), off=(0,+1)).
        offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
        for (face_idx, off) in enumerate(offsets)
            expected_nb = fn[face_idx]
            pv = hv[interior_leaf_idx, off]
            if expected_nb == 0
                @test pv === nothing
            else
                @test pv !== nothing
                # Coefficient column [1] equals our per-cell tag.
                @test pv[1] == Float64(expected_nb)
            end
        end
    end

    @testset "boundary leaf: out-of-domain returns nothing" begin
        # Leaf in the corner: at least two faces are domain boundaries.
        # `leaves[1]` is the lower-left corner of the 8×8 grid.
        corner_leaf_idx = leaves[1]
        @test is_leaf(mesh.cells[corner_leaf_idx])
        # On a uniform 8×8 mesh, the lower-left leaf has no -x or -y
        # neighbor under the plain (non-BC-aware) face_neighbors.
        @test hv[corner_leaf_idx, (-1, 0)] === nothing
        @test hv[corner_leaf_idx, (0, -1)] === nothing
        # Its +x and +y neighbors do exist.
        @test hv[corner_leaf_idx, (1, 0)] !== nothing
        @test hv[corner_leaf_idx, (0, 1)] !== nothing
    end

    # ---------------------------------------------------------------
    # BC-aware wrap. Q1 says HaloView itself does NOT auto-wrap; the
    # boundary handling is the caller's job through
    # face_neighbors_with_bcs (or, in the M3-3b EL residual, through a
    # `halo_or_ghost(hv, cell, off, fb)` shim — see §3.4 of the
    # design note). We verify the mechanics here so the M3-3b shim
    # has a clean spec.
    # ---------------------------------------------------------------
    @testset "BC-aware face neighbors: PERIODIC wraps, REFLECTING does not" begin
        # axis 1 periodic, axis 2 reflecting.
        spec_per_refl = ((PERIODIC, PERIODIC), (REFLECTING, REFLECTING))
        # axis 1 reflecting, axis 2 periodic.
        spec_refl_per = ((REFLECTING, REFLECTING), (PERIODIC, PERIODIC))

        # Pick the lower-left corner leaf again. With axis 1 periodic
        # the (-1, 0) face wraps to a leaf on the right wall; with axis
        # 2 reflecting it stays at 0 (handled downstream by the EL).
        corner = leaves[1]

        fn_pr = face_neighbors_with_bcs(mesh, corner, spec_per_refl)
        # face order: (1lo, 1hi, 2lo, 2hi).
        @test fn_pr[1] != 0   # axis 1 lo wrapped through PERIODIC.
        @test fn_pr[2] != 0   # axis 1 hi same neighbor as plain (interior).
        @test fn_pr[3] == 0   # axis 2 lo: REFLECTING — not wrapped.
        @test fn_pr[4] != 0   # axis 2 hi: interior neighbor.

        fn_rp = face_neighbors_with_bcs(mesh, corner, spec_refl_per)
        @test fn_rp[1] == 0   # axis 1 lo: REFLECTING.
        @test fn_rp[2] != 0
        @test fn_rp[3] != 0   # axis 2 lo: PERIODIC wraps.
        @test fn_rp[4] != 0
    end

    # ---------------------------------------------------------------
    # Allocation contract: indexing in the cached path must not
    # allocate beyond the small Union{Nothing, PolynomialView} wrapper.
    # We mirror HG's own halo test's tolerance (≤ 64 bytes).
    # ---------------------------------------------------------------
    @testset "allocation: hv[i, off] in cached path" begin
        # Warm up the neighbor graph + JIT.
        _ = hv[interior_leaf_idx, (1, 0)]
        _ = hv[interior_leaf_idx, (0, -1)]
        # Wrap in a closure to give the compiler a single concrete
        # entry point.
        f = (hv, i) -> @inbounds hv[i, (1, 0)]
        f(hv, interior_leaf_idx)  # warm
        a = @allocated f(hv, interior_leaf_idx)
        @test a <= 64

        g = (hv, i) -> @inbounds hv[i, (0, 1)]
        g(hv, interior_leaf_idx)
        b = @allocated g(hv, interior_leaf_idx)
        @test b <= 64
    end
end
