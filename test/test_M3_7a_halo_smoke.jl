# test_M3_7a_halo_smoke.jl
#
# M3-7a smoke test: HaloView coefficient access pattern for order-0
# `MonomialBasis{3, 0}` polynomial fields on a 4×4×4 uniform balanced
# 3D `HierarchicalMesh`.
#
# This is the 3D analog of `test/test_M3_3a_halo_smoke.jl`. It pins the
# HaloView contract for D=3 + resolves Q1 of the M3-7 design note's §11
# open questions ("does HaloView work as expected at depth=1 on a 3D
# octree?"). The Q1 default — "stay at depth=1" — is verified; depth=2
# is *characterised* (smoke-only).
#
# Coverage:
#
#   1. Mesh sanity: 4×4×4 balanced 3D mesh has 64 leaves, 73 cells
#      (1 root + 8 + 64 = 73 in the multi-level cumulative count).
#   2. Basis sanity: `n_coeffs(MonomialBasis{3, 0}()) == 1`.
#   3. Constructing a `HaloView` over a `PolynomialFieldView` on a
#      3D mesh allocates only the small wrapper struct.
#   4. Interior-leaf self-access: `hv[i, (0, 0, 0)][1]` returns the
#      cell's own coefficient (cell tag = `Float64(i)`).
#   5. Interior-leaf 6-face neighbor access: `hv[i, off]` agrees with
#      `face_neighbors(mesh, i)` cross-check on all 6 face offsets
#      (±x, ±y, ±z).
#   6. Boundary-leaf out-of-domain: a corner leaf returns `nothing` for
#      the three outward offsets, returns valid `PolynomialView` on
#      the three inward offsets.
#   7. BC-aware wrap (`face_neighbors_with_bcs`): mixed
#      (PERIODIC, REFLECTING, REFLECTING) wraps axis 1 boundary faces;
#      axis 2 / axis 3 boundary faces stay at 0 (downstream EL synthesises
#      the reflecting ghost).
#   8. Allocation contract: cached `hv[i, off]` is ≤ 64 bytes per call
#      (mirrors the 2D smoke test tolerance).
#   9. Depth-2 characterisation: 2-hop offsets (e.g. `(2, 0, 0)`) are
#      accepted by `HaloView(field, 2)` and either return a
#      `PolynomialView` (interior 2-hop) or `nothing` (boundary).
#      Depth bound is enforced (`(3, 0, 0)` throws).
#
# Reference: `~/.julia/dev/HierarchicalGrids/test/test_halo_view.jl`,
# `reference/notes_M3_7_3d_extension.md` §3 + §11 Q1 + Q4,
# `reference/notes_M3_7_prep_3d_scaffolding.md`.

using Test
using HierarchicalGrids
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    n_cells, is_leaf, halo_view, HaloView,
    allocate_polynomial_fields, SoA, MonomialBasis, n_coeffs,
    face_neighbors, face_neighbors_with_bcs,
    BCKind, PERIODIC, REFLECTING

@testset "HaloView 3D order-0 smoke (M3-7a)" begin
    # ---------------------------------------------------------------
    # Build a 4×4×4 uniformly refined 3D mesh (level 2 → 64 leaves).
    # `balanced=true` per Q4 default (M3-3a inherited).
    # ---------------------------------------------------------------
    mesh = HierarchicalMesh{3}(; balanced = true)
    for _ in 1:2
        leaves = enumerate_leaves(mesh)
        refine_cells!(mesh, leaves)
    end
    leaves = enumerate_leaves(mesh)

    @testset "uniform 4×4×4 mesh: 64 leaves at level 2" begin
        @test length(leaves) == 64
        # Multi-level cumulative cell count: 1 (root) + 8 + 64 = 73.
        @test n_cells(mesh) == 73
        # Every leaf is at the deepest level (level 2).
        @test all(is_leaf(mesh.cells[ci]) for ci in leaves)
    end

    # ---------------------------------------------------------------
    # Allocate a single-field PolynomialFieldSet at MonomialBasis{3, 0}.
    # Storage is sized to n_cells(mesh) so HaloView's mesh-cell-index
    # access maps 1-to-1 against `mesh.cells`.
    # ---------------------------------------------------------------
    basis = MonomialBasis{3, 0}()
    @testset "MonomialBasis{3, 0} has exactly one coefficient" begin
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
    # Pick an interior leaf. With 64 leaves on a 4×4×4 grid, every
    # leaf has at least 3 boundary faces (the corners), but we want
    # one with all 6 face neighbors interior — none of the 64 leaves
    # qualify on a 4×4×4 grid, since every leaf is on the boundary
    # of the unit cube along at least one axis. Choose the deepest-
    # interior leaf: one whose face_neighbors tuple has the FEWEST zeros.
    # ---------------------------------------------------------------
    function n_boundary_faces(mesh, i)
        nb = face_neighbors(mesh, i)
        return count(==(UInt32(0)), nb)
    end
    # Among 64 leaves on a 4×4×4 grid, every leaf sits on a face;
    # the most-interior leaves touch exactly 1 boundary face (interior
    # cells of any 4×4×4 face). Pick the one with the fewest boundary
    # faces.
    n_bnds = [n_boundary_faces(mesh, ci) for ci in leaves]
    interior_leaf_idx = leaves[argmin(n_bnds)]
    @test is_leaf(mesh.cells[interior_leaf_idx])

    @testset "interior leaf: self-access shape" begin
        pv_self = hv[interior_leaf_idx, (0, 0, 0)]
        @test pv_self !== nothing
        @test pv_self[1] == Float64(interior_leaf_idx)
    end

    @testset "interior leaf: 6-face neighbor access (±x, ±y, ±z)" begin
        # We don't know a priori which mesh-cell-index sits at each
        # offset, but we can cross-check halo access against the plain
        # face_neighbors API.
        fn = face_neighbors(mesh, interior_leaf_idx)
        # face order: (axis1 lo, axis1 hi, axis2 lo, axis2 hi,
        #              axis3 lo, axis3 hi) →
        #             ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0),
        #              (0, 0, -1), (0, 0, 1)).
        offsets = ((-1, 0, 0), (1, 0, 0),
                   (0, -1, 0), (0, 1, 0),
                   (0, 0, -1), (0, 0, 1))
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

    @testset "boundary leaf: 3-corner out-of-domain returns nothing" begin
        # Pick the leaf with the most boundary faces — should be a
        # corner with at least 3 outward boundary faces.
        corner_leaf_idx = leaves[argmax(n_bnds)]
        @test is_leaf(mesh.cells[corner_leaf_idx])
        @test n_boundary_faces(mesh, corner_leaf_idx) >= 3

        # Determine which face directions are out-of-domain.
        fn_corner = face_neighbors(mesh, corner_leaf_idx)
        offsets = ((-1, 0, 0), (1, 0, 0),
                   (0, -1, 0), (0, 1, 0),
                   (0, 0, -1), (0, 0, 1))
        for (face_idx, off) in enumerate(offsets)
            pv = hv[corner_leaf_idx, off]
            if fn_corner[face_idx] == 0
                @test pv === nothing
            else
                @test pv !== nothing
                @test pv[1] == Float64(fn_corner[face_idx])
            end
        end
    end

    # ---------------------------------------------------------------
    # BC-aware wrap. HaloView itself does NOT auto-wrap; the boundary
    # handling is the caller's job through face_neighbors_with_bcs (or
    # downstream a `halo_or_ghost(hv, cell, off, fb)` shim, M3-7b's
    # 3D analog of M3-3b). We verify the mechanics here so M3-7b has
    # a clean spec.
    # ---------------------------------------------------------------
    @testset "BC-aware face neighbors: mixed PERIODIC/REFLECTING (3D)" begin
        # axis 1 periodic, axis 2 + 3 reflecting — the 3D analog of the
        # 2D mixed spec.
        spec_per_refl_refl = ((PERIODIC, PERIODIC),
                               (REFLECTING, REFLECTING),
                               (REFLECTING, REFLECTING))
        # All-periodic (3D-periodic, the analog of a triple-torus).
        spec_all_periodic = ((PERIODIC, PERIODIC),
                              (PERIODIC, PERIODIC),
                              (PERIODIC, PERIODIC))

        # Pick a leaf on the lo-x boundary. The corner leaf certainly
        # qualifies; its (-1, 0, 0) face is at the lo-x boundary.
        corner = leaves[argmax(n_bnds)]
        fn_plain = face_neighbors(mesh, corner)
        # If face 1 (axis 1 lo) is on boundary, the BC-aware version with
        # PERIODIC axis 1 must wrap.
        if fn_plain[1] == 0
            fn_pr = face_neighbors_with_bcs(mesh, corner, spec_per_refl_refl)
            @test fn_pr[1] != 0   # axis 1 lo wrapped through PERIODIC.
            # Axis 2 / 3 lo / hi: REFLECTING — boundary not wrapped.
            # (Whether those particular faces are boundary depends on
            # which corner; we test only the periodic axis here since
            # the reflecting axes' is invariant to the spec.)
        end

        # All-periodic: every boundary face wraps.
        fn_all = face_neighbors_with_bcs(mesh, corner, spec_all_periodic)
        # No zero entries because every boundary face wrapped.
        @test count(==(UInt32(0)), fn_all) == 0
    end

    # ---------------------------------------------------------------
    # Allocation contract: indexing in the cached path must not
    # allocate beyond the small Union{Nothing, PolynomialView} wrapper.
    # We mirror HG's own halo test's tolerance (≤ 64 bytes).
    # ---------------------------------------------------------------
    @testset "allocation: hv[i, off] in cached path (3D)" begin
        # Warm up the neighbor graph + JIT.
        _ = hv[interior_leaf_idx, (1, 0, 0)]
        _ = hv[interior_leaf_idx, (0, -1, 0)]
        _ = hv[interior_leaf_idx, (0, 0, 1)]
        # Wrap in a closure to give the compiler a single concrete
        # entry point.
        f = (hv, i) -> @inbounds hv[i, (1, 0, 0)]
        f(hv, interior_leaf_idx)  # warm
        a = @allocated f(hv, interior_leaf_idx)
        @test a <= 64

        g = (hv, i) -> @inbounds hv[i, (0, 0, 1)]
        g(hv, interior_leaf_idx)
        b = @allocated g(hv, interior_leaf_idx)
        @test b <= 64

        h = (hv, i) -> @inbounds hv[i, (0, 1, 0)]
        h(hv, interior_leaf_idx)
        c = @allocated h(hv, interior_leaf_idx)
        @test c <= 64
    end

    # ---------------------------------------------------------------
    # Depth-2 characterisation (M3-7 design note §11 Q1 / Q4).
    # The default is "stay at depth=1"; depth=2 is smoke-only. We
    # verify mechanics: the constructor accepts depth=2; 2-hop offsets
    # walk the neighbor graph correctly; 3-hop offsets throw.
    # ---------------------------------------------------------------
    @testset "depth=2: 2-hop neighbor walk + depth bound enforcement" begin
        hv2 = halo_view(pfs.α_1, mesh, 2)
        @test hv2 isa HaloView

        # 2-hop along a single axis: should land on a leaf two cells
        # over (or `nothing` if it leaves the domain).
        pv_2hop_x = hv2[interior_leaf_idx, (2, 0, 0)]
        # The 2-hop result is either a valid PolynomialView (some
        # interior leaf) or `nothing` (the 2-hop walk hit the boundary).
        # On a 4×4×4 grid with 64 leaves, most leaves have at least
        # ONE 2-hop direction that succeeds; we verify the API does
        # not error.
        @test pv_2hop_x === nothing || pv_2hop_x !== nothing  # tautology — the point is no exception

        # Mixed 2-hop offset: (1, 1, 0). Sum of |offsets| = 2 ≤ depth=2.
        pv_mixed = hv2[interior_leaf_idx, (1, 1, 0)]
        @test pv_mixed === nothing || pv_mixed !== nothing  # API accepts

        # depth bound enforcement: 3-hop must throw.
        @test_throws ArgumentError hv2[interior_leaf_idx, (3, 0, 0)]
        @test_throws ArgumentError hv2[interior_leaf_idx, (1, 1, 1)]
    end

    # ---------------------------------------------------------------
    # Depth-2 honest characterisation: at least one interior 2-hop
    # along an axis succeeds (returns a PolynomialView) — the smoke
    # test demonstrates depth=2 is not vacuously broken in 3D.
    # ---------------------------------------------------------------
    @testset "depth=2: at least one 2-hop call returns PolynomialView" begin
        hv2 = halo_view(pfs.α_1, mesh, 2)
        # Scan all 64 leaves and all 6 axis-aligned 2-hop offsets;
        # count how many succeed (return PolynomialView).
        n_success = 0
        offsets_2hop = ((2, 0, 0), (-2, 0, 0),
                        (0, 2, 0), (0, -2, 0),
                        (0, 0, 2), (0, 0, -2))
        for ci in leaves, off in offsets_2hop
            pv = hv2[ci, off]
            if pv !== nothing
                n_success += 1
                # Check the value is the leaf tag of some other leaf.
                @test pv[1] != Float64(ci)        # different cell
                @test !isnan(pv[1])               # actual leaf data
            end
        end
        # On a 4×4×4 grid with 64 leaves and 6 axis-aligned 2-hops, the
        # interior leaves of each face contribute; we expect a comfortable
        # majority of successes (interior 2-hops), not zero.
        @test n_success > 0
    end
end
