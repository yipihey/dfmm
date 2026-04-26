# test_M3_prep_berry_stencil.jl
#
# Tests for `src/berry.jl` (M3-prep, Tier-1 building blocks for M3-3).
#
# Coverage:
#  • Symbolic verification points reproduced exactly from
#    `scripts/verify_berry_connection*.py`.
#  • Closed-form partial derivatives consistent with finite differences
#    (1e-6 step, relative error ≤ 1e-7).
#  • Axis-swap antisymmetry of the 2D Berry function.
#  • Cyclic-permutation behavior of the 3D pair-functions F_{ab}.
#  • Iso-limit (α_1 = α_2 = α): F vanishes — matches the §6 ε-derivation.
#  • 7D-to-5D reduction: off-diagonal kinetic contributions vanish
#    when α-only inputs are paired with β_off = 0.
#  • Allocation count = 0 after warm-up.

using Test
using StaticArrays
using Random

using dfmm

# ──────────────────────────────────────────────────────────────────────
# 1. Specific verification points (reproduce SymPy results exactly)
# ──────────────────────────────────────────────────────────────────────

@testset "Berry 2D — explicit verification points" begin
    # Iso-limit: F = 0 for any β at α_1 = α_2.
    α = SVector{2,Float64}(1.0, 1.0)
    β = SVector{2,Float64}(0.0, 0.0)
    @test berry_F_2d(α, β) == 0.0
    @test berry_term_2d(α, β, 1.0) == 0.0

    # Generic point P1: α=(2,1), β=(0.5,1), dθ=1
    α = SVector{2,Float64}(2.0, 1.0)
    β = SVector{2,Float64}(0.5, 1.0)
    # F = (8*1 - 1*0.5)/3 = 7.5/3 = 2.5
    @test berry_F_2d(α, β) ≈ 2.5 rtol=1e-15
    @test berry_term_2d(α, β, 1.0) ≈ 2.5 rtol=1e-15

    # Axis-swap of P1: α=(1,2), β=(1,0.5), dθ=1
    αs = SVector{2,Float64}(1.0, 2.0)
    βs = SVector{2,Float64}(1.0, 0.5)
    @test berry_F_2d(αs, βs) ≈ -2.5 rtol=1e-15

    # Generic point P2: α=(3,2), β=(0.5,1.5), dθ=0.7
    α = SVector{2,Float64}(3.0, 2.0)
    β = SVector{2,Float64}(0.5, 1.5)
    # F = (27*1.5 - 8*0.5)/3 = (40.5 - 4)/3 = 36.5/3
    @test berry_F_2d(α, β) ≈ 36.5/3 rtol=1e-15
    @test berry_term_2d(α, β, 0.7) ≈ (36.5/3) * 0.7 rtol=1e-15

    # Generic point P3: α=(1.5, 0.8), β=(-0.3, 0.6), dθ=1.2
    α = SVector{2,Float64}(1.5, 0.8)
    β = SVector{2,Float64}(-0.3, 0.6)
    # F = (3.375*0.6 - 0.512*(-0.3))/3 = (2.025 + 0.1536)/3 = 2.1786/3
    expected = (1.5^3 * 0.6 - 0.8^3 * (-0.3)) / 3
    @test berry_F_2d(α, β) ≈ expected rtol=1e-15
    @test berry_term_2d(α, β, 1.2) ≈ expected * 1.2 rtol=1e-15
end

@testset "Berry 2D — closed-form partial derivatives" begin
    α = SVector{2,Float64}(2.0, 1.0)
    β = SVector{2,Float64}(0.5, 1.0)
    dθ = 1.0
    p = berry_partials_2d(α, β, dθ)
    # Expected from the doc:
    @test p[1] ≈ α[1]^2 * β[2] * dθ            rtol=1e-15  # 4.0
    @test p[2] ≈ -α[2]^2 * β[1] * dθ           rtol=1e-15  # -0.5
    @test p[3] ≈ -(α[2]^3) * dθ / 3            rtol=1e-15  # -1/3
    @test p[4] ≈  (α[1]^3) * dθ / 3            rtol=1e-15  # 8/3
    @test p[5] ≈ (α[1]^3 * β[2] - α[2]^3 * β[1]) / 3       rtol=1e-15  # F = 2.5
end

@testset "Berry 2D — partials match finite differences" begin
    rng = MersenneTwister(20260426)
    h = 1e-6
    for _ in 1:8
        α = SVector{2,Float64}(0.5 + rand(rng), 0.5 + rand(rng))
        β = SVector{2,Float64}(2 * rand(rng) - 1, 2 * rand(rng) - 1)
        dθ = 0.5 + rand(rng)
        p = berry_partials_2d(α, β, dθ)

        # ∂Θ/∂α_1
        αp = SVector{2,Float64}(α[1] + h, α[2])
        αm = SVector{2,Float64}(α[1] - h, α[2])
        fd_α1 = (berry_term_2d(αp, β, dθ) - berry_term_2d(αm, β, dθ)) / (2h)
        @test isapprox(p[1], fd_α1; rtol=1e-7, atol=1e-9)

        # ∂Θ/∂α_2
        αp = SVector{2,Float64}(α[1], α[2] + h)
        αm = SVector{2,Float64}(α[1], α[2] - h)
        fd_α2 = (berry_term_2d(αp, β, dθ) - berry_term_2d(αm, β, dθ)) / (2h)
        @test isapprox(p[2], fd_α2; rtol=1e-7, atol=1e-9)

        # ∂Θ/∂β_1
        βp = SVector{2,Float64}(β[1] + h, β[2])
        βm = SVector{2,Float64}(β[1] - h, β[2])
        fd_β1 = (berry_term_2d(α, βp, dθ) - berry_term_2d(α, βm, dθ)) / (2h)
        @test isapprox(p[3], fd_β1; rtol=1e-7, atol=1e-9)

        # ∂Θ/∂β_2
        βp = SVector{2,Float64}(β[1], β[2] + h)
        βm = SVector{2,Float64}(β[1], β[2] - h)
        fd_β2 = (berry_term_2d(α, βp, dθ) - berry_term_2d(α, βm, dθ)) / (2h)
        @test isapprox(p[4], fd_β2; rtol=1e-7, atol=1e-9)

        # ∂Θ/∂θ_R: linear in dθ_R, so derivative is F (independent of dθ_R).
        fd_θ = (berry_term_2d(α, β, dθ + h) - berry_term_2d(α, β, dθ - h)) / (2h)
        @test isapprox(p[5], fd_θ; rtol=1e-7, atol=1e-9)
    end
end

@testset "Berry 2D — axis-swap antisymmetry" begin
    rng = MersenneTwister(0xb12)
    for _ in 1:6
        α = SVector{2,Float64}(0.4 + rand(rng), 0.4 + rand(rng))
        β = SVector{2,Float64}(2 * rand(rng) - 1, 2 * rand(rng) - 1)
        αs = SVector{2,Float64}(α[2], α[1])
        βs = SVector{2,Float64}(β[2], β[1])
        @test berry_F_2d(αs, βs) ≈ -berry_F_2d(α, β) atol=1e-14
    end
end

@testset "Berry 2D — iso reduction (α_1 = α_2)" begin
    # At α_1 = α_2, F = (α³ β_2 - α³ β_1)/3 = (α³/3)(β_2 - β_1).
    # In the symmetric β state (β_1 = β_2), F = 0; this is the
    # iso-pullback "Berry vanishes" check (CHECK 6 in
    # verify_berry_connection.py §6).
    rng = MersenneTwister(0xa1a2)
    for _ in 1:5
        a = 0.5 + rand(rng)
        b = 2 * rand(rng) - 1
        α = SVector{2,Float64}(a, a)
        β = SVector{2,Float64}(b, b)
        @test berry_F_2d(α, β) == 0.0
    end
    # And on the wider iso-α slice with β_1 ≠ β_2, F = (α³/3)(β_2 - β_1).
    α = SVector{2,Float64}(2.0, 2.0)
    β = SVector{2,Float64}(0.5, 1.5)
    @test berry_F_2d(α, β) ≈ (2.0^3 / 3) * (1.5 - 0.5) atol=1e-14
end

@testset "Berry 2D — BerryStencil2D pre-compute" begin
    α = SVector{2,Float64}(2.0, 1.0)
    β = SVector{2,Float64}(0.5, 1.0)
    s = BerryStencil2D(α, β)
    @test s.F ≈ 2.5 rtol=1e-15
    @test s.dF_dα ≈ SVector{2,Float64}(α[1]^2 * β[2], -α[2]^2 * β[1]) rtol=1e-15
    @test s.dF_dβ ≈ SVector{2,Float64}(-(α[2]^3)/3, (α[1]^3)/3) rtol=1e-15

    # apply matches direct evaluation
    for dθ in (0.0, 1.0, -0.7, 3.14)
        @test dfmm.apply(s, dθ) ≈ berry_term_2d(α, β, dθ) rtol=1e-15
    end
end

# ──────────────────────────────────────────────────────────────────────
# 2. 3D SO(3) Berry verification
# ──────────────────────────────────────────────────────────────────────

@testset "Berry 3D — explicit verification points" begin
    # Iso-limit α=(1,1,1), β=(0.5,0.5,0.5): all F_ab = 0.
    α = SVector{3,Float64}(1.0, 1.0, 1.0)
    β = SVector{3,Float64}(0.5, 0.5, 0.5)
    F = berry_F_3d(α, β)
    @test F[1] == 0.0
    @test F[2] == 0.0
    @test F[3] == 0.0

    # Generic point: α=(2,1,0.5), β=(0.5,1,2), dθ=[[0,1,0.5],[-1,0,0.3],...]
    α = SVector{3,Float64}(2.0, 1.0, 0.5)
    β = SVector{3,Float64}(0.5, 1.0, 2.0)
    F = berry_F_3d(α, β)
    # F_12 = (8*1 - 1*0.5)/3 = 7.5/3 = 2.5
    # F_13 = (8*2 - 0.125*0.5)/3 = (16 - 0.0625)/3 = 15.9375/3 = 5.3125
    # F_23 = (1*2 - 0.125*1)/3 = (2 - 0.125)/3 = 1.875/3 = 0.625
    @test F[1] ≈ 2.5 rtol=1e-15
    @test F[2] ≈ 5.3125 rtol=1e-15
    @test F[3] ≈ 0.625 rtol=1e-15

    dθ = SMatrix{3,3,Float64,9}(
        # col 1
        0.0, -1.0, -0.5,
        # col 2
        1.0,  0.0, -0.3,
        # col 3
        0.5,  0.3,  0.0,
    )
    # Θ = F_12 * dθ_12 + F_13 * dθ_13 + F_23 * dθ_23
    expected = 2.5 * 1.0 + 5.3125 * 0.5 + 0.625 * 0.3
    @test berry_term_3d(α, β, dθ) ≈ expected rtol=1e-15

    # Sanity: matches the SymPy-helper Python computation exactly.
    @test berry_term_3d(α, β, dθ) ≈ 5.34375 rtol=1e-15
end

@testset "Berry 3D — closed-form partials match finite differences" begin
    rng = MersenneTwister(20260426 + 3)
    h = 1e-6
    for _ in 1:5
        α = SVector{3,Float64}(0.5 + rand(rng), 0.5 + rand(rng), 0.5 + rand(rng))
        β = SVector{3,Float64}(2 * rand(rng) - 1, 2 * rand(rng) - 1, 2 * rand(rng) - 1)
        d12 = 2 * rand(rng) - 1
        d13 = 2 * rand(rng) - 1
        d23 = 2 * rand(rng) - 1
        dθ = SMatrix{3,3,Float64,9}(
            0.0, -d12, -d13,
            d12,  0.0, -d23,
            d13,  d23,  0.0,
        )
        (grad, F) = berry_partials_3d(α, β, dθ)

        # α-partials
        for k in 1:3
            αp = setindex(α, α[k] + h, k)
            αm = setindex(α, α[k] - h, k)
            fd = (berry_term_3d(αp, β, dθ) - berry_term_3d(αm, β, dθ)) / (2h)
            @test isapprox(grad[k], fd; rtol=1e-7, atol=1e-9)
        end
        # β-partials
        for k in 1:3
            βp = setindex(β, β[k] + h, k)
            βm = setindex(β, β[k] - h, k)
            fd = (berry_term_3d(α, βp, dθ) - berry_term_3d(α, βm, dθ)) / (2h)
            @test isapprox(grad[3 + k], fd; rtol=1e-7, atol=1e-9)
        end
        # F itself: equals the partial w.r.t. each dθ_{ab} (linear).
        # ∂Θ/∂dθ_12:
        dθp = SMatrix{3,3,Float64,9}(0.0, -(d12+h), -d13, d12+h, 0.0, -d23, d13, d23, 0.0)
        dθm = SMatrix{3,3,Float64,9}(0.0, -(d12-h), -d13, d12-h, 0.0, -d23, d13, d23, 0.0)
        fd12 = (berry_term_3d(α, β, dθp) - berry_term_3d(α, β, dθm)) / (2h)
        @test isapprox(F[1], fd12; rtol=1e-7, atol=1e-9)
    end
end

@testset "Berry 3D — pair antisymmetry under axis swap" begin
    # CHECK 5 in verify_berry_connection_3D.py: F_{ab}(α_a, α_b, β_a, β_b)
    # is antisymmetric under (a, b) swap.
    rng = MersenneTwister(0x33d)
    for _ in 1:4
        α = SVector{3,Float64}(0.4 + rand(rng), 0.4 + rand(rng), 0.4 + rand(rng))
        β = SVector{3,Float64}(2 * rand(rng) - 1, 2 * rand(rng) - 1, 2 * rand(rng) - 1)
        F = berry_F_3d(α, β)
        # Swap axis 1<->2: F_12 changes sign.
        αs = SVector{3,Float64}(α[2], α[1], α[3])
        βs = SVector{3,Float64}(β[2], β[1], β[3])
        Fs = berry_F_3d(αs, βs)
        @test Fs[1] ≈ -F[1] atol=1e-14
        # Swap 1<->3: F_13 (the (1,3) pair) changes sign.
        αs = SVector{3,Float64}(α[3], α[2], α[1])
        βs = SVector{3,Float64}(β[3], β[2], β[1])
        Fs = berry_F_3d(αs, βs)
        @test Fs[2] ≈ -F[2] atol=1e-14
        # Swap 2<->3: F_23 changes sign.
        αs = SVector{3,Float64}(α[1], α[3], α[2])
        βs = SVector{3,Float64}(β[1], β[3], β[2])
        Fs = berry_F_3d(αs, βs)
        @test Fs[3] ≈ -F[3] atol=1e-14
    end
end

@testset "Berry 3D — cyclic axis permutation" begin
    # Under cyclic permutation σ = (1→2→3→1), the pair (1,2) maps to
    # (2,3) and the pair (1,3) maps to (2,1) i.e. -F_12, and (2,3) maps
    # to (3,1) i.e. -F_13. So under cyclic σ:
    #   F_12 ↦ F_23 (after relabeling)
    #   F_13 ↦ -F_12
    #   F_23 ↦ -F_13
    # i.e. F'(σα, σβ) = (F_23, -F_12, -F_13)  where σα = (α_3, α_1, α_2).
    rng = MersenneTwister(0xc7c)
    for _ in 1:4
        α = SVector{3,Float64}(0.4 + rand(rng), 0.4 + rand(rng), 0.4 + rand(rng))
        β = SVector{3,Float64}(2 * rand(rng) - 1, 2 * rand(rng) - 1, 2 * rand(rng) - 1)
        F = berry_F_3d(α, β)
        # σ: 1→2, 2→3, 3→1 ⇒ new axis 1 = old axis 3, etc.
        # so σ(α) = (α_3, α_1, α_2)
        ασ = SVector{3,Float64}(α[3], α[1], α[2])
        βσ = SVector{3,Float64}(β[3], β[1], β[2])
        Fσ = berry_F_3d(ασ, βσ)
        # Fσ_12 = (1/3)(α_3³ β_1 - α_1³ β_3) = -F_13(α, β)
        @test Fσ[1] ≈ -F[2] atol=1e-14
        # Fσ_13 = (1/3)(α_3³ β_2 - α_2³ β_3) = -F_23
        @test Fσ[2] ≈ -F[3] atol=1e-14
        # Fσ_23 = (1/3)(α_1³ β_2 - α_2³ β_1) =  F_12
        @test Fσ[3] ≈  F[1] atol=1e-14
    end
end

@testset "Berry 3D — iso reduction" begin
    # Full iso (CHECK 3a): all F_ab vanish.
    rng = MersenneTwister(0x3033)
    for _ in 1:5
        a = 0.5 + rand(rng)
        b = 2 * rand(rng) - 1
        α = SVector{3,Float64}(a, a, a)
        β = SVector{3,Float64}(b, b, b)
        F = berry_F_3d(α, β)
        @test F[1] == 0.0
        @test F[2] == 0.0
        @test F[3] == 0.0
    end
end

@testset "Berry 3D — 2D reduction (CHECK 3b of SymPy script)" begin
    # When α_3 is constant, β_3 = 0, and dθ_13 = dθ_23 = 0, the 3D
    # Berry term reduces exactly to the 2D Berry term on the (1,2)
    # sector.
    rng = MersenneTwister(0x32d)
    for _ in 1:4
        α12 = SVector{2,Float64}(0.5 + rand(rng), 0.5 + rand(rng))
        β12 = SVector{2,Float64}(2 * rand(rng) - 1, 2 * rand(rng) - 1)
        dθR = 2 * rand(rng) - 1

        α3 = SVector{3,Float64}(α12[1], α12[2], 1.7)
        β3 = SVector{3,Float64}(β12[1], β12[2], 0.0)
        dθ_3d = SMatrix{3,3,Float64,9}(
            0.0,    -dθR,  0.0,
            dθR,     0.0,  0.0,
            0.0,     0.0,  0.0,
        )
        @test berry_term_3d(α3, β3, dθ_3d) ≈ berry_term_2d(α12, β12, dθR) atol=1e-14
    end
end

@testset "Berry 3D — BerryStencil3D pre-compute" begin
    α = SVector{3,Float64}(2.0, 1.0, 0.5)
    β = SVector{3,Float64}(0.5, 1.0, 2.0)
    s = BerryStencil3D(α, β)
    F_ref = berry_F_3d(α, β)
    @test s.F ≈ F_ref rtol=1e-14
    # Spot-check stencil entries.
    @test s.dF_dα[1, 1] ≈ α[1]^2 * β[2]   rtol=1e-15  # ∂F_12/∂α_1
    @test s.dF_dα[1, 2] ≈ -α[2]^2 * β[1]  rtol=1e-15  # ∂F_12/∂α_2
    @test s.dF_dβ[1, 1] ≈ -(α[2]^3) / 3   rtol=1e-15  # ∂F_12/∂β_1
    @test s.dF_dβ[1, 2] ≈  (α[1]^3) / 3   rtol=1e-15  # ∂F_12/∂β_2
    @test s.dF_dα[2, 3] ≈ -α[3]^2 * β[1]  rtol=1e-15  # ∂F_13/∂α_3
    @test s.dF_dβ[3, 3] ≈  (α[2]^3) / 3   rtol=1e-15  # ∂F_23/∂β_3

    # apply against a probe rotation matrix
    dθ = SMatrix{3,3,Float64,9}(
        0.0, -1.0, -0.5,
        1.0,  0.0, -0.3,
        0.5,  0.3,  0.0,
    )
    @test dfmm.apply(s, dθ) ≈ berry_term_3d(α, β, dθ) rtol=1e-14
end

# ──────────────────────────────────────────────────────────────────────
# 3. Off-diagonal-L₂ kinetic 1-form
# ──────────────────────────────────────────────────────────────────────

@testset "Off-diagonal kinetic 1-form — coefficients" begin
    # CHECK 2 of verify_berry_connection_offdiag.py:
    #   coefficient of dβ_{12} in θ_offdiag is -(1/2) α_1 α_2².
    #   coefficient of dβ_{21} in θ_offdiag is -(1/2) α_1² α_2.
    α = SVector{2,Float64}(2.0, 1.0)
    c = kinetic_offdiag_coeffs_2d(α)
    @test c[1] ≈ -1.0   atol=1e-15  # coeff of dβ_{12} = -(α_1 α_2²)/2 = -1
    @test c[2] ≈ -2.0   atol=1e-15  # coeff of dβ_{21} = -(α_1² α_2)/2 = -2

    # Spot check at α=(1.5, 2.5):
    α = SVector{2,Float64}(1.5, 2.5)
    c = kinetic_offdiag_coeffs_2d(α)
    @test c[1] ≈ -(1.5 * 2.5^2) / 2  rtol=1e-15
    @test c[2] ≈ -(1.5^2 * 2.5) / 2  rtol=1e-15
end

@testset "Off-diagonal kinetic — bilinear evaluation" begin
    α = SVector{2,Float64}(2.0, 1.0)
    # β as a 2x2 matrix with off-diagonals β_{12}, β_{21}.
    # SMatrix is column-major: arg order is (β[1,1], β[2,1], β[1,2], β[2,2]).
    β = SMatrix{2,2,Float64,4}(0.0, 0.7, 0.3, 0.0)  # β_{21}=0.7, β_{12}=0.3
    @test β[1, 2] == 0.3
    @test β[2, 1] == 0.7

    val = kinetic_offdiag_2d(α, β)
    # = -(1/2)(α_1² α_2 · β[2,1] + α_1 α_2² · β[1,2])
    # = -(1/2)(4 * 1 * 0.7 + 2 * 1 * 0.3) = -(1/2)(2.8 + 0.6) = -1.7
    @test val ≈ -1.7  atol=1e-15

    # Reduces to 0 when β_off = 0.
    β0 = SMatrix{2,2,Float64,4}(0.0, 0.0, 0.0, 0.0)
    @test kinetic_offdiag_2d(α, β0) == 0.0
end

# ──────────────────────────────────────────────────────────────────────
# 4. Allocation count for hot-path call (M3-3 will call this every
#    Newton iter per cell).
# ──────────────────────────────────────────────────────────────────────

@testset "Berry 2D — zero allocations on hot path" begin
    α = SVector{2,Float64}(2.0, 1.0)
    β = SVector{2,Float64}(0.5, 1.0)
    dθ = 0.7

    # Warm up to compile.
    _ = berry_term_2d(α, β, dθ)
    _ = berry_partials_2d(α, β, dθ)
    _ = BerryStencil2D(α, β)

    # Wrap in zero-arg closures so @allocated sees a true call.
    f1() = berry_term_2d(α, β, dθ)
    f2() = berry_partials_2d(α, β, dθ)
    f3() = BerryStencil2D(α, β)

    @test (@allocated f1()) == 0
    @test (@allocated f2()) == 0
    @test (@allocated f3()) == 0
end

@testset "Berry 3D — zero allocations on hot path" begin
    α = SVector{3,Float64}(2.0, 1.0, 0.5)
    β = SVector{3,Float64}(0.5, 1.0, 2.0)
    dθ = SMatrix{3,3,Float64,9}(
        0.0, -1.0, -0.5,
        1.0,  0.0, -0.3,
        0.5,  0.3,  0.0,
    )

    _ = berry_term_3d(α, β, dθ)
    _ = berry_partials_3d(α, β, dθ)
    _ = BerryStencil3D(α, β)

    g1() = berry_term_3d(α, β, dθ)
    g2() = berry_partials_3d(α, β, dθ)
    g3() = BerryStencil3D(α, β)

    @test (@allocated g1()) == 0
    @test (@allocated g2()) == 0
    @test (@allocated g3()) == 0
end
