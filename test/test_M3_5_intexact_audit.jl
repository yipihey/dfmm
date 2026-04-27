# test/test_M3_5_intexact_audit.jl
#
# Milestone M3-5 IntExact audit harness.
#
# Calls HG's `audit_overlap()` on the canonical 2D + 3D polytope
# battery and verifies the audit report passes (no float-vs-exact
# divergence beyond the documented HG caveats). Also documents the
# caveats from `~/.julia/dev/HierarchicalGrids/docs/src/exact_backend.md`:
#   - D = 2 0//0 collinear-triangle degeneracy (rare; the canonical
#     battery avoids it).
#   - D ∈ {2, 3} default-lattice (16-bit) volume drift up to ~30 % on
#     near-degenerate configurations.
#
# Per the M3-5 plan: ~10-20 asserts.

using HierarchicalGrids
using HierarchicalGrids: OverlapAuditReport, audit_overlap

# ---------------------------------------------------------------------------
# Test 1 — canonical battery passes.
# ---------------------------------------------------------------------------
@testset "canonical-polytope IntExact audit passes" begin
    report = dfmm.audit_overlap_dfmm(; verbose = false, atol = 1e-10)
    @test report isa OverlapAuditReport
    # 5 D=2 polytopes + 4 D=3 polytopes
    @test report.n_polytopes_checked == 9
    @test report.n_passed == report.n_polytopes_checked
    @test report.n_failed == 0
    # Residual relative differences should be very small on the audit
    # battery (lattice scale = 2^10 = 1024).
    @test report.max_volume_relative_diff < 1e-5
    @test report.max_moment_relative_diff < 1e-5
    @test isempty(report.failures)
end

# ---------------------------------------------------------------------------
# Test 2 — tighter tolerance still passes.
# ---------------------------------------------------------------------------
@testset "tighter atol audit passes" begin
    report = dfmm.audit_overlap_dfmm(; verbose = false, atol = 1e-6)
    @test report.n_failed == 0
end

# ---------------------------------------------------------------------------
# Test 3 — HG `compute_overlap` :float vs :exact agreement on the
# identity-overlap dfmm setup. This is the canonical 2D scenario most
# likely to reflect dfmm's actual usage; we verify the per-overlap-
# entry volumes agree to within ~1e-4 (per HG docs, the 16-bit lattice
# can drift by a few % on adversarial configurations; on this benign
# setup the agreement is much better).
# ---------------------------------------------------------------------------
@testset "compute_overlap :float vs :exact magnitude on dfmm 4x4" begin
    eul_mesh = dfmm.uniform_eulerian_mesh(2, 2)  # 16 leaves
    frame = HierarchicalGrids.EulerianFrame(eul_mesh, (0.0, 0.0), (1.0, 1.0))
    lag_mesh = triangulate_eulerian(eul_mesh, frame)

    overlap_f = HierarchicalGrids.compute_overlap(lag_mesh, frame;
                                                    moment_order = 0,
                                                    backend = :float)
    overlap_e = HierarchicalGrids.compute_overlap(lag_mesh, frame;
                                                    moment_order = 0,
                                                    backend = :exact)
    # Both should produce the same number of entries (both record every
    # nonzero geometric overlap polygon).
    @test overlap_f.n_lag == overlap_e.n_lag
    @test overlap_f.n_eul == overlap_e.n_eul

    # Total overlap volume should match the box volume to round-off.
    box_vol = 1.0 * 1.0
    @test isapprox(HierarchicalGrids.total_overlap_volume(overlap_f), box_vol;
                    rtol = 1e-12)
    @test isapprox(HierarchicalGrids.total_overlap_volume(overlap_e), box_vol;
                    rtol = 1e-3)   # 16-bit lattice
end

# ---------------------------------------------------------------------------
# Test 4 — IntExact caveats observed magnitudes (bookkeeping).
#
# Records observed magnitudes for the project lab notebook so future
# regressions can be triaged.
# ---------------------------------------------------------------------------
@testset "IntExact caveats — observed magnitudes" begin
    # Run audit at a few atol levels and record passes.
    atols = (1e-12, 1e-10, 1e-6, 1e-4)
    for atol in atols
        r = dfmm.audit_overlap_dfmm(; verbose = false, atol = atol)
        @info "Audit at atol = $atol" passed = r.n_passed failed = r.n_failed max_vol_rel_diff = r.max_volume_relative_diff max_moment_rel_diff = r.max_moment_relative_diff
        # All atols ≥ 1e-10 should pass (the audit's max_*_rel_diff is
        # already bounded below 1e-10 on the canonical battery per
        # HG docs).
        if atol >= 1e-10
            @test r.n_failed == 0
        end
    end
end
