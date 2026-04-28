# test_M3_7c_berry_3d_residual.jl
#
# §7.2 Berry verification reproduction (M3-7c) — residual level.
#
# Sample 5–10 random `(α, β, θ_12, θ_13, θ_23)` configurations and
# verify that the Berry-coupling contribution to the 15-dof 3D residual
# matches the closed-form `berry_partials_3d` from `src/berry.jl`. The
# residual rows under test are F^α_a, F^β_a (per axis a); the Berry part
# of each row is the linear `θ̇_{ab}`-coefficient. We isolate it by
# perturbing a single θ_{ab}^{n+1} per probe while holding all other
# state fixed and reading off the residual change.
#
# Expected derivatives (3D analog of M3-3c §6.2). Per axis a, summing
# over the pair-generators in which a participates (signs from the
# rows of Ω · X = -dH per pair):
#
#   axis 1: F^α_1 += +(α_2³/(3α_1²)) dθ̇_12 + (α_3³/(3α_1²)) dθ̇_13
#           F^β_1 += +β_2 dθ̇_12 + β_3 dθ̇_13
#   axis 2: F^α_2 += -(α_1³/(3α_2²)) dθ̇_12 + (α_3³/(3α_2²)) dθ̇_23
#           F^β_2 += -β_1 dθ̇_12 + β_3 dθ̇_23
#   axis 3: F^α_3 += -(α_1³/(3α_3²)) dθ̇_13 - (α_2³/(3α_3²)) dθ̇_23
#           F^β_3 += -β_1 dθ̇_13 - β_2 dθ̇_23
#
# Plus F^θ_{ab} = 1/dt (kinematic-drive row).
#
# Cross-check against `berry_partials_3d`: the closed-form Berry
# partials in `src/berry.jl` return (∂Θ/∂α_a, ∂Θ/∂β_a) per axis with
# Θ = Σ_{a<b} F_{ab} · dθ_{ab}; dividing by α_a² (the symplectic
# weight) maps back to the Hamilton-equation form used in the residual.
# This gate also demonstrates SymPy CHECK 7 (∂Θ/∂θ_{ab} = F_{ab}) at
# the residual level: the F^θ_{ab}-row finite difference recovers
# 1/dt regardless of (α, β) (free-flight kinematic drive).

using Test
using StaticArrays
using Random
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, cell_physical_box
using dfmm: DetField3D, allocate_cholesky_3d_fields,
    read_detfield_3d, write_detfield_3d!,
    pack_state_3d, unpack_state_3d!,
    cholesky_el_residual_3D_berry!, cholesky_el_residual_3D_berry,
    build_residual_aux_3D, berry_partials_3d, berry_F_3d

const M3_7C_BERRY_TOL = 1.0e-9

@testset "M3-7c §7.2 Berry verification reproduction (residual level)" begin
    mesh = HierarchicalMesh{3}(; balanced = true)
    for _ in 1:1
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    leaves = enumerate_leaves(mesh)
    frame = EulerianFrame(mesh, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    bc_spec = FrameBoundaries{3}(((REFLECTING, REFLECTING),
                                   (REFLECTING, REFLECTING),
                                   (REFLECTING, REFLECTING)))

    rng = MersenneTwister(20260427)

    # 6 random samples (target: 5–10).
    for sample in 1:6
        α1 = 0.5 + rand(rng) * 1.5
        α2 = 0.5 + rand(rng) * 1.5
        α3 = 0.5 + rand(rng) * 1.5
        # Stay off the iso-degeneracy boundaries so the per-pair Berry
        # partials are in the generic regime.
        if abs(α1 - α2) < 0.1; α2 += 0.2; end
        if abs(α1 - α3) < 0.1; α3 += 0.25; end
        if abs(α2 - α3) < 0.1; α3 += 0.3; end
        β1 = 2 * rand(rng) - 1
        β2 = 2 * rand(rng) - 1
        β3 = 2 * rand(rng) - 1
        θ12 = 0.3 * (2 * rand(rng) - 1)
        θ13 = 0.3 * (2 * rand(rng) - 1)
        θ23 = 0.3 * (2 * rand(rng) - 1)

        # Initialize fields uniformly across all leaves (no spatial
        # gradient ⇒ pressure stencil contributes 0 ⇒ residual reads
        # the per-axis Cholesky-sector + Berry block alone).
        fields = allocate_cholesky_3d_fields(mesh)
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            cz = 0.5 * (lo[3] + hi[3])
            v = DetField3D((cx, cy, cz), (0.0, 0.0, 0.0),
                            (α1, α2, α3), (β1, β2, β3),
                            θ12, θ13, θ23, 1.0)
            write_detfield_3d!(fields, ci, v)
        end

        # M_vv_override = constants ⇒ residual decoupled from EOS.
        M_vv = (1.0, 1.0, 1.0)
        aux = build_residual_aux_3D(fields, mesh, frame, leaves, bc_spec;
                                     M_vv_override = M_vv, ρ_ref = 1.0)

        y_n = pack_state_3d(fields, leaves)
        dt = 1e-3
        Δθ = 1e-7

        F_base = cholesky_el_residual_3D_berry(y_n, y_n, aux, dt)

        # Mid-point quantities for expected partials. With y_np1 ≈ y_n
        # (single-θ perturbation only at the linear-Δθ level), midpoints
        # equal the IC values per cell.
        ᾱ1 = α1; ᾱ2 = α2; ᾱ3 = α3
        β̄1 = β1; β̄2 = β2; β̄3 = β3

        # Probe each θ_{ab} independently.
        for ab_pair in (:θ12, :θ13, :θ23)
            θ_idx = ab_pair === :θ12 ? 13 : (ab_pair === :θ13 ? 14 : 15)
            y_np1_perturbed = copy(y_n)
            for (i_leaf, _) in enumerate(leaves)
                y_np1_perturbed[15 * (i_leaf - 1) + θ_idx] += Δθ
            end
            F_perturb = cholesky_el_residual_3D_berry(y_np1_perturbed, y_n, aux, dt)

            # Expected coefficients depend on which θ pair is perturbed.
            # Layout: rows 7..9 = F^α_{1,2,3}; rows 10..12 = F^β_{1,2,3};
            # rows 13..15 = F^θ_{12,13,23}.
            if ab_pair === :θ12
                expected_dα1 =  (ᾱ2^3) / (3 * ᾱ1^2) / dt
                expected_dα2 = -(ᾱ1^3) / (3 * ᾱ2^2) / dt
                expected_dα3 =  0.0
                expected_dβ1 =  β̄2 / dt
                expected_dβ2 = -β̄1 / dt
                expected_dβ3 =  0.0
                expected_dθ12 = 1 / dt
                expected_dθ13 = 0.0
                expected_dθ23 = 0.0
            elseif ab_pair === :θ13
                expected_dα1 =  (ᾱ3^3) / (3 * ᾱ1^2) / dt
                expected_dα2 =  0.0
                expected_dα3 = -(ᾱ1^3) / (3 * ᾱ3^2) / dt
                expected_dβ1 =  β̄3 / dt
                expected_dβ2 =  0.0
                expected_dβ3 = -β̄1 / dt
                expected_dθ12 = 0.0
                expected_dθ13 = 1 / dt
                expected_dθ23 = 0.0
            else  # :θ23
                expected_dα1 =  0.0
                expected_dα2 =  (ᾱ3^3) / (3 * ᾱ2^2) / dt
                expected_dα3 = -(ᾱ2^3) / (3 * ᾱ3^2) / dt
                expected_dβ1 =  0.0
                expected_dβ2 =  β̄3 / dt
                expected_dβ3 = -β̄2 / dt
                expected_dθ12 = 0.0
                expected_dθ13 = 0.0
                expected_dθ23 = 1 / dt
            end

            max_dα1_err = 0.0; max_dα2_err = 0.0; max_dα3_err = 0.0
            max_dβ1_err = 0.0; max_dβ2_err = 0.0; max_dβ3_err = 0.0
            max_dθ12_err = 0.0; max_dθ13_err = 0.0; max_dθ23_err = 0.0
            for (i_leaf, _) in enumerate(leaves)
                base = 15 * (i_leaf - 1)
                num_dα1 = (F_perturb[base +  7] - F_base[base +  7]) / Δθ
                num_dα2 = (F_perturb[base +  8] - F_base[base +  8]) / Δθ
                num_dα3 = (F_perturb[base +  9] - F_base[base +  9]) / Δθ
                num_dβ1 = (F_perturb[base + 10] - F_base[base + 10]) / Δθ
                num_dβ2 = (F_perturb[base + 11] - F_base[base + 11]) / Δθ
                num_dβ3 = (F_perturb[base + 12] - F_base[base + 12]) / Δθ
                num_dθ12 = (F_perturb[base + 13] - F_base[base + 13]) / Δθ
                num_dθ13 = (F_perturb[base + 14] - F_base[base + 14]) / Δθ
                num_dθ23 = (F_perturb[base + 15] - F_base[base + 15]) / Δθ
                max_dα1_err = max(max_dα1_err, abs(num_dα1 - expected_dα1))
                max_dα2_err = max(max_dα2_err, abs(num_dα2 - expected_dα2))
                max_dα3_err = max(max_dα3_err, abs(num_dα3 - expected_dα3))
                max_dβ1_err = max(max_dβ1_err, abs(num_dβ1 - expected_dβ1))
                max_dβ2_err = max(max_dβ2_err, abs(num_dβ2 - expected_dβ2))
                max_dβ3_err = max(max_dβ3_err, abs(num_dβ3 - expected_dβ3))
                max_dθ12_err = max(max_dθ12_err, abs(num_dθ12 - expected_dθ12))
                max_dθ13_err = max(max_dθ13_err, abs(num_dθ13 - expected_dθ13))
                max_dθ23_err = max(max_dθ23_err, abs(num_dθ23 - expected_dθ23))
            end

            # Tolerance accounts for FD truncation at Δθ = 1e-7 plus
            # division by `dt = 1e-3`.
            tol = M3_7C_BERRY_TOL * (1 / dt)
            @test max_dα1_err ≤ tol
            @test max_dα2_err ≤ tol
            @test max_dα3_err ≤ tol
            @test max_dβ1_err ≤ tol
            @test max_dβ2_err ≤ tol
            @test max_dβ3_err ≤ tol
            @test max_dθ12_err ≤ tol
            @test max_dθ13_err ≤ tol
            @test max_dθ23_err ≤ tol
        end

        # Cross-check that the residual-implied Berry partials reconcile
        # with `berry_partials_3d` from `src/berry.jl` (CHECK 7 of the 3D
        # SymPy verification, residual-level reproduction). Specifically,
        # multiplying the residual coefficient on θ̇_{ab} by α_a² recovers
        # the Berry stencil entries `dF/dβ_a` (per pair); the residual
        # coefficient on β_b matches `dF/dα_a / α_a²`.
        αv = SVector(α1, α2, α3)
        βv = SVector(β1, β2, β3)
        # Probe with dθ matrix = identity-on-(1,2) for pair-12 partials etc.
        # But since berry_partials_3d returns scalars summed over all
        # active dθ entries, we pick dθ matrices isolating each pair.
        z = 0.0
        dθ_12 = SMatrix{3,3,Float64,9}(z, -1.0,  z,
                                        1.0,  z,  z,
                                        z,    z,  z)  # only dθ[1,2]=1, dθ[2,1]=-1
        dθ_13 = SMatrix{3,3,Float64,9}(z,  z, -1.0,
                                        z,  z,  z,
                                        1.0, z,  z)
        dθ_23 = SMatrix{3,3,Float64,9}(z, z, z,
                                        z, z, -1.0,
                                        z, 1.0, z)
        # Pair (1, 2) — only dθ[1,2] = 1 contributes; using
        # berry_partials_3d to read off F_12 and per-axis partials.
        (gαβ_12, F_12_vec) = berry_partials_3d(αv, βv, dθ_12)
        # gαβ_12[1] = ∂F_12/∂α_1 · 1 = α_1² β_2; etc.
        @test gαβ_12[1] ≈ α1^2 * β2 rtol=1e-14
        @test gαβ_12[2] ≈ -α2^2 * β1 rtol=1e-14
        @test gαβ_12[3] ≈ 0.0 atol=1e-14
        @test gαβ_12[4] ≈ -(α2^3) / 3 rtol=1e-14
        @test gαβ_12[5] ≈  (α1^3) / 3 rtol=1e-14
        @test gαβ_12[6] ≈ 0.0 atol=1e-14

        # Pair (1, 3)
        (gαβ_13, _) = berry_partials_3d(αv, βv, dθ_13)
        @test gαβ_13[1] ≈ α1^2 * β3 rtol=1e-14
        @test gαβ_13[2] ≈ 0.0 atol=1e-14
        @test gαβ_13[3] ≈ -α3^2 * β1 rtol=1e-14
        @test gαβ_13[4] ≈ -(α3^3) / 3 rtol=1e-14
        @test gαβ_13[5] ≈ 0.0 atol=1e-14
        @test gαβ_13[6] ≈  (α1^3) / 3 rtol=1e-14

        # Pair (2, 3)
        (gαβ_23, _) = berry_partials_3d(αv, βv, dθ_23)
        @test gαβ_23[1] ≈ 0.0 atol=1e-14
        @test gαβ_23[2] ≈ α2^2 * β3 rtol=1e-14
        @test gαβ_23[3] ≈ -α3^2 * β2 rtol=1e-14
        @test gαβ_23[4] ≈ 0.0 atol=1e-14
        @test gαβ_23[5] ≈ -(α3^3) / 3 rtol=1e-14
        @test gαβ_23[6] ≈  (α2^3) / 3 rtol=1e-14

        # F values themselves match `berry_F_3d`.
        F_3d = berry_F_3d(αv, βv)
        @test F_3d[1] ≈ (α1^3 * β2 - α2^3 * β1) / 3 rtol=1e-14
        @test F_3d[2] ≈ (α1^3 * β3 - α3^3 * β1) / 3 rtol=1e-14
        @test F_3d[3] ≈ (α2^3 * β3 - α3^3 * β2) / 3 rtol=1e-14
    end
end
