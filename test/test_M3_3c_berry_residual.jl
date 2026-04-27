# test_M3_3c_berry_residual.jl
#
# §6.2 Berry verification reproduction (M3-3c).
#
# Sample 5–10 random `(α, β, θ_R)` configurations and verify that the
# Berry-coupling contribution to the 9-dof residual matches the closed-
# form `berry_partials_2d` from `src/berry.jl`. The residual rows under
# test are F^α_a, F^β_a; the Berry part of each row is the
# `θ̇_R`-coefficient. We isolate it by taking finite differences in
# `θ_R^{n+1}` while holding (x, u, α, β, θ_R^n) fixed and reading off
# the residual change.
#
# The expected derivatives, from rows of `Ω · X = -dH` at midpoints
# (see file-level comment in `src/eom.jl`):
#
#     ∂F^α_1 / ∂(θ̇_R) = + ᾱ_2³ / (3 ᾱ_1²)
#     ∂F^α_2 / ∂(θ̇_R) = - ᾱ_1³ / (3 ᾱ_2²)
#     ∂F^β_1 / ∂(θ̇_R) = + β̄_2
#     ∂F^β_2 / ∂(θ̇_R) = - β̄_1
#
# The connection to `berry_partials_2d`: the Berry kinetic 1-form is
# `Θ = F · dθ_R` with `F = (α_1³β_2 − α_2³β_1)/3`. Closed-form
# `berry_partials_2d` returns `(∂F/∂α_1, ∂F/∂α_2, ∂F/∂β_1, ∂F/∂β_2,
# ∂F/∂θ_R) · dθ_R`. The 5×5 Ω entries derived from `dΘ` give the
# residual contributions above; specifically Ω[α_a, θ_R] = ∂F/∂α_a
# and Ω[β_a, θ_R] = ∂F/∂β_a, and dividing by `α_a²` (the Ω[α_a, β_a]
# weight) maps back to the Hamilton-equation form used in the residual.
#
# Specifically for axis 1:  Ω[α_1, β_1] · β̇_1 + Ω[α_1, θ_R] · θ̇_R + ∂H/∂α_1 = 0
#   ⇒  α_1² β̇_1 + α_1² β_2 · θ̇_R = α_1 γ_1²
#   ⇒  β̇_1 = γ_1²/α_1 − β_2 · θ̇_R     (so ∂F^β_1/∂θ̇_R = +β̄_2)
# And: Ω[β_1, α_1] · α̇_1 + Ω[β_1, θ_R] · θ̇_R + ∂H/∂β_1 = 0
#   ⇒  −α_1² α̇_1 − (α_2³/3) · θ̇_R = α_1² β_1
#   ⇒  α̇_1 = β_1 − (α_2³/(3α_1²)) · θ̇_R    (so ∂F^α_1/∂θ̇_R = +ᾱ_2³/(3ᾱ_1²))

using Test
using StaticArrays
using Random
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, cell_physical_box
using dfmm: DetField2D, allocate_cholesky_2d_fields,
    read_detfield_2d, write_detfield_2d!,
    pack_state_2d_berry, unpack_state_2d_berry!,
    cholesky_el_residual_2D_berry!, cholesky_el_residual_2D_berry,
    build_residual_aux_2D, berry_partials_2d

const M3_3C_BERRY_TOL = 1.0e-9   # finite-difference probe tolerance

@testset "M3-3c §6.2 Berry verification reproduction" begin
    mesh = HierarchicalMesh{2}(; balanced = true)
    for _ in 1:2
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    leaves = enumerate_leaves(mesh)
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
    bc_spec = FrameBoundaries{2}(((REFLECTING, REFLECTING),
                                   (REFLECTING, REFLECTING)))

    rng = MersenneTwister(20260426)

    # 8 random samples (target: 5–10).
    for sample in 1:8
        # Sample non-degenerate (α, β, θ_R), with α positive and bounded
        # away from zero, plus a small θ_R offset.
        α1 = 0.5 + rand(rng) * 1.5
        α2 = 0.5 + rand(rng) * 1.5
        # Stay off the iso boundary so the per-axis Berry partials are
        # in the generic regime.
        if abs(α1 - α2) < 0.1
            α2 += 0.2
        end
        β1 = 2 * rand(rng) - 1
        β2 = 2 * rand(rng) - 1
        θR = 0.3 * (2 * rand(rng) - 1)

        # Initialize fields uniformly across all leaves (no spatial
        # gradient ⇒ pressure stencil contributes 0 ⇒ residual reads
        # the per-axis Cholesky-sector + Berry block alone).
        fields = allocate_cholesky_2d_fields(mesh)
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = DetField2D((cx, cy), (0.0, 0.0),
                            (α1, α2), (β1, β2), θR, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        # M_vv_override = constants ⇒ residual decoupled from EOS.
        M_vv = (1.0, 1.0)  # generic non-zero per-axis
        aux = build_residual_aux_2D(fields, mesh, frame, leaves, bc_spec;
                                     M_vv_override = M_vv, ρ_ref = 1.0)

        y_n = pack_state_2d_berry(fields, leaves)
        dt = 1e-3
        # Probe: take y_np1 = y_n (so midpoint = self for everything
        # except θ_R — we add a small Δθ_R ≪ 1 to extract the linear
        # response). The base residual at y_np1 = y_n + (θ_R: Δθ_R)
        # has Berry contribution ≈ partial · Δθ_R / dt + O(Δθ_R²).
        Δθ_R = 1e-7

        F_base = cholesky_el_residual_2D_berry(y_n, y_n, aux, dt)

        y_np1_perturbed = copy(y_n)
        # M3-6 Phase 0: 11-dof packing, θ_R is at index 11 (was 9 in M3-3c).
        for (i_leaf, _) in enumerate(leaves)
            y_np1_perturbed[11 * (i_leaf - 1) + 11] += Δθ_R
        end
        F_perturb = cholesky_el_residual_2D_berry(y_np1_perturbed, y_n, aux, dt)

        # Numerical partials per cell, `(F_perturb - F_base) / Δθ_R`
        # gives ∂F/∂θ̇_R · (1/dt) since the residual is in (D_t)
        # form. Multiplying by `dt` gives `∂F/∂(Δθ_R)` directly.
        # But the residual rows include `(θR_np1 − θR_n)/dt` only in
        # the new F^θ_R row; for F^α_a, F^β_a the Berry term is a
        # function of `θ̇_R = (θ_R_np1 − θ_R_n)/dt`. So
        # ∂F^α_a/∂θ_R_np1 = (∂F^α_a/∂θ̇_R) / dt.
        max_dα1_err = 0.0; max_dα2_err = 0.0
        max_dβ1_err = 0.0; max_dβ2_err = 0.0
        max_dθR_err = 0.0

        # Mid-point quantities for the expected partials. With y_np1 ≈
        # y_n (θ_R perturbation only at the linear-Δθ_R level), the
        # midpoint α_a ≈ α_a, β_a ≈ β_a per cell.
        ᾱ1 = α1; ᾱ2 = α2; β̄1 = β1; β̄2 = β2

        # Expected ∂F^*/∂θ_R_np1.
        expected_dα1 = (ᾱ2^3) / (3 * ᾱ1^2) / dt
        expected_dα2 = -(ᾱ1^3) / (3 * ᾱ2^2) / dt
        expected_dβ1 = β̄2 / dt
        expected_dβ2 = -β̄1 / dt
        expected_dθR = 1 / dt   # F^θ_R = (θR_np1 − θR_n)/dt → ∂/∂θR_np1 = 1/dt

        for (i_leaf, _) in enumerate(leaves)
            # M3-6 Phase 0: 11-dof packing.
            #   row 5 = F^α_1, 6 = F^α_2, 7 = F^β_1, 8 = F^β_2,
            #   row 9 = F^β_12, 10 = F^β_21, 11 = F^θ_R.
            base = 11 * (i_leaf - 1)
            num_dα1 = (F_perturb[base + 5]  - F_base[base + 5])  / Δθ_R
            num_dα2 = (F_perturb[base + 6]  - F_base[base + 6])  / Δθ_R
            num_dβ1 = (F_perturb[base + 7]  - F_base[base + 7])  / Δθ_R
            num_dβ2 = (F_perturb[base + 8]  - F_base[base + 8])  / Δθ_R
            num_dθR = (F_perturb[base + 11] - F_base[base + 11]) / Δθ_R

            max_dα1_err = max(max_dα1_err, abs(num_dα1 - expected_dα1))
            max_dα2_err = max(max_dα2_err, abs(num_dα2 - expected_dα2))
            max_dβ1_err = max(max_dβ1_err, abs(num_dβ1 - expected_dβ1))
            max_dβ2_err = max(max_dβ2_err, abs(num_dβ2 - expected_dβ2))
            max_dθR_err = max(max_dθR_err, abs(num_dθR - expected_dθR))
        end

        # Tolerance accounts for FD truncation at Δθ_R = 1e-7 plus
        # division by `dt = 1e-3`: each FD error term is bounded by
        # (Δθ_R / dt)² × |second derivative| ≈ 1e-8.
        @test max_dα1_err ≤ M3_3C_BERRY_TOL * (1 / dt)
        @test max_dα2_err ≤ M3_3C_BERRY_TOL * (1 / dt)
        @test max_dβ1_err ≤ M3_3C_BERRY_TOL * (1 / dt)
        @test max_dβ2_err ≤ M3_3C_BERRY_TOL * (1 / dt)
        @test max_dθR_err ≤ M3_3C_BERRY_TOL * (1 / dt)

        # Cross-check that the Berry partials from `src/berry.jl` agree
        # with the residual-implied partials structurally. The closed
        # form is:
        #     berry_partials_2d(α, β, dθ_R) returns
        #         ((∂F/∂α_1) dθ_R, (∂F/∂α_2) dθ_R,
        #          (∂F/∂β_1) dθ_R, (∂F/∂β_2) dθ_R,
        #          F)   (= (α_1²β_2, -α_2²β_1, -α_2³/3, α_1³/3, F))
        αv = SVector(α1, α2)
        βv = SVector(β1, β2)
        bp = berry_partials_2d(αv, βv, 1.0)  # dθ_R = 1, so coefficients only
        # Identities: (residual coefficient on θ̇_R) · α_a²
        #   (∂F^α_1/∂θ̇_R) · α_1² = ᾱ_2³ / (3 ᾱ_1²) · α_1² = α_2³ / 3 = -bp[3]
        #   (∂F^α_2/∂θ̇_R) · α_2² = -α_1³ / 3 = -bp[4]
        # i.e., the Cholesky-symplectic weight α_a² recovers the
        # `dF/dβ_a` partials of Berry.
        @test (ᾱ2^3) / (3 * ᾱ1^2) * α1^2 ≈ -bp[3] rtol=1e-14
        @test -(ᾱ1^3) / (3 * ᾱ2^2) * α2^2 ≈ -bp[4] rtol=1e-14
        # And: (∂F^β_a/∂θ̇_R) · α_a² = bp[a]:
        #   β̄_2 · α_1² = α_1² β_2 = bp[1]
        #   −β̄_1 · α_2² = -α_2² β_1 = bp[2]
        @test β̄2 * α1^2 ≈ bp[1] rtol=1e-14
        @test -β̄1 * α2^2 ≈ bp[2] rtol=1e-14
    end
end
