# test_M3_7_prep_3d_scaffolding.jl
#
# M3-7 prep tests for the 3D scaffolding:
#   • `DetField3D{T}` structural contract (13-dof Newton unknown set
#     per leaf cell; mirrors `DetField2D{T}`'s 12-dof structure).
#   • `cholesky_decompose_3d ∘ cholesky_recompose_3d ≡ id` on 50
#     random `(α, θ)` inputs (round-trip; ≤ 1e-12 absolute).
#   • Iso-pullback on the slice α_1 = α_2 = α_3: γ entries equal;
#     decomposition well-defined.
#   • 2D reduction: at θ_13 = θ_23 = 0 with arbitrary `α_3`, the
#     3D decomposition of `L_3D` reduces to the 2D
#     `cholesky_decompose_2d` byte-equal on the top-left 2×2 block.
#   • Per-axis γ diagnostic: known-input test where `M_vv` has
#     anisotropic diagonal entries; verify γ correctly reports
#     per-axis values; allocation-free hot path.
#   • Allocation: zero allocations on hot path after warm-up for
#     all four primitives (decompose, recompose, gamma_per_axis_3d
#     in both signatures).
#
# See `reference/notes_M3_7_3d_extension.md` §2 (13-dof field set),
# §11 (eigendecomposition route), `reference/notes_M3_3a_field_set_cholesky.md`
# (2D pattern), and `reference/notes_M3_7_prep_3d_scaffolding.md` (this
# sub-phase's status note).

using Test
using Random
using StaticArrays
using LinearAlgebra: det, I
using dfmm: cholesky_decompose_3d, cholesky_recompose_3d, gamma_per_axis_3d
using dfmm: cholesky_decompose_2d, cholesky_recompose_2d
using dfmm: rotation_matrix_3d
using dfmm: DetField3D, n_dof_newton, spatial_dimension

const _M3_7_PREP_TOL = 1e-12

@testset "M3-7 prep: 3D field set + per-axis Cholesky decomposition" begin

    # ─────────────────────────────────────────────────────────────
    # Block 1: DetField3D structural contract
    # ─────────────────────────────────────────────────────────────
    @testset "DetField3D structural contract" begin
        x = (1.0, 2.0, 3.0)
        u = (0.1, 0.2, 0.3)
        α = (2.5, 1.2, 0.7)
        β = (0.05, 0.07, 0.09)
        v = DetField3D(x, u, α, β, 0.31, -0.18, 0.42, 1.5)

        @test v isa DetField3D{Float64}
        @test v.x == x
        @test v.u == u
        @test v.alphas == α
        @test v.betas == β
        @test v.θ_12 == 0.31
        @test v.θ_13 == -0.18
        @test v.θ_23 == 0.42
        @test v.s == 1.5
        @test spatial_dimension(v) == 3
        @test n_dof_newton(v) == 13
        @test n_dof_newton(DetField3D{Float64}) == 13

        # Type promotion across mixed-eltype tuples.
        v2 = DetField3D((1.0f0, 2.0f0, 3.0f0), (0, 0, 0),
                        (1, 1, 1), (0.0, 0.0, 0.0),
                        0, 0, 0, 1)
        @test v2 isa DetField3D{Float64}
    end

    # ─────────────────────────────────────────────────────────────
    # Block 2: rotation_matrix_3d sanity
    # ─────────────────────────────────────────────────────────────
    @testset "rotation_matrix_3d: proper rotation in SO(3)" begin
        # At θ = (0, 0, 0): identity.
        R0 = rotation_matrix_3d(0.0, 0.0, 0.0)
        @test R0[1, 1] ≈ 1.0  atol = _M3_7_PREP_TOL
        @test R0[2, 2] ≈ 1.0  atol = _M3_7_PREP_TOL
        @test R0[3, 3] ≈ 1.0  atol = _M3_7_PREP_TOL
        @test R0[1, 2] ≈ 0.0  atol = _M3_7_PREP_TOL
        @test R0[1, 3] ≈ 0.0  atol = _M3_7_PREP_TOL
        @test R0[2, 3] ≈ 0.0  atol = _M3_7_PREP_TOL

        # Random rotation: orthogonal (R Rᵀ = I) + det = +1.
        rng = MersenneTwister(0xC3D7)
        for _ in 1:5
            θ12 = (rand(rng) - 0.5) * π / 2
            θ13 = (rand(rng) - 0.5) * π / 2
            θ23 = (rand(rng) - 0.5) * π / 2
            R = rotation_matrix_3d(θ12, θ13, θ23)
            @test maximum(abs.(R * transpose(R) - I)) ≤ _M3_7_PREP_TOL
            @test det(R) ≈ 1.0  atol = _M3_7_PREP_TOL
        end

        # 2D reduction: at θ_13 = θ_23 = 0, R = R_12(θ_12) — the 2D
        # rotation around axis 3. Top-left 2×2 block must match 2D.
        for θR in (0.0, 0.31, -0.5, 1.1)
            R = rotation_matrix_3d(θR, 0.0, 0.0)
            c, s = cos(θR), sin(θR)
            @test R[1, 1] ≈  c
            @test R[1, 2] ≈ -s
            @test R[2, 1] ≈  s
            @test R[2, 2] ≈  c
            @test R[3, 3] ≈  1.0
            @test R[3, 1] ≈  0.0  atol = _M3_7_PREP_TOL
            @test R[3, 2] ≈  0.0  atol = _M3_7_PREP_TOL
            @test R[1, 3] ≈  0.0  atol = _M3_7_PREP_TOL
            @test R[2, 3] ≈  0.0  atol = _M3_7_PREP_TOL
        end
    end

    # ─────────────────────────────────────────────────────────────
    # Block 3: round-trip on 50 random inputs
    # ─────────────────────────────────────────────────────────────
    @testset "round-trip: decompose ∘ recompose ≡ id (50 random in canonical hemisphere)" begin
        # Canonical round-trip range (see cholesky_DD_3d.jl docstring):
        #   • α_1 ≥ α_2 ≥ α_3 > 0 (descending sort)
        #   • θ_12, θ_13, θ_23 ∈ (-π/2, π/2) (canonical hemisphere)
        # The 3D analog of the 2D `θ_R ∈ (-π/2, π/2]` round-trip range.
        rng = MersenneTwister(0xC3D0)
        n_pass = 0
        max_err = 0.0
        for _ in 1:50
            # Sort α descending: α_1 ≥ α_2 ≥ α_3 > 0.
            a_unsorted = [0.1 + 5 * rand(rng) for _ in 1:3]
            a = sort(a_unsorted, rev = true)
            α = SVector{3, Float64}(a[1], a[2], a[3])
            # Sample inside the canonical hemisphere (open interval to
            # avoid the boundary).
            θ12 = (rand(rng) - 0.5) * π * 0.95
            θ13 = (rand(rng) - 0.5) * π * 0.95
            θ23 = (rand(rng) - 0.5) * π * 0.95
            θ = SVector{3, Float64}(θ12, θ13, θ23)

            L = cholesky_recompose_3d(α, θ)
            α′, θ′ = cholesky_decompose_3d(L)

            err_α = maximum(abs.(α .- α′))
            err_θ = maximum(abs.(θ .- θ′))
            max_err = max(max_err, err_α, err_θ)

            @test err_α ≤ _M3_7_PREP_TOL
            @test err_θ ≤ _M3_7_PREP_TOL
            n_pass += 1
        end
        @test n_pass == 50
        @info "M3-7 prep round-trip max_err: $max_err"
    end

    # ─────────────────────────────────────────────────────────────
    # Block 4: iso-pullback (α_1 = α_2 = α_3)
    # ─────────────────────────────────────────────────────────────
    @testset "iso-pullback: α_1 = α_2 = α_3 (decomposition well-defined)" begin
        # On the iso slice the rotation is genuinely gauge-degenerate
        # (M = α² I commutes with every rotation — any orthogonal R
        # is an eigenvector basis). The eigendecomposition returns a
        # numerically-noisy R; the canonical gauge fix in
        # `cholesky_decompose_3d` makes the (α, θ) result deterministic
        # but **not** equal to the input θ (the gauge fixes pin the
        # representative, but on the iso slice the input θ is gauge-
        # equivalent to many different angle triples). We only assert:
        #   (a) α-eigenvalues are recovered (all equal to α₀),
        #   (b) the result is finite (no NaN / Inf),
        #   (c) the underlying M = L Lᵀ is preserved (the physics is
        #       gauge-invariant — only M matters, not the (α, θ)
        #       parameterization).
        for α0 in (0.5, 1.0, 2.7, 3.3)
            α = SVector{3, Float64}(α0, α0, α0)
            for θ_in in (
                SVector{3, Float64}(0.0, 0.0, 0.0),
                SVector{3, Float64}(0.3, 0.0, 0.0),
                SVector{3, Float64}(0.0, 0.3, 0.0),
                SVector{3, Float64}(0.0, 0.0, 0.3),
                SVector{3, Float64}(0.2, -0.1, 0.4),
            )
                L = cholesky_recompose_3d(α, θ_in)
                α′, θ′ = cholesky_decompose_3d(L)

                # (a) α-eigenvalues recovered.
                @test α′[1] ≈ α0  atol = _M3_7_PREP_TOL
                @test α′[2] ≈ α0  atol = _M3_7_PREP_TOL
                @test α′[3] ≈ α0  atol = _M3_7_PREP_TOL

                # (b) finite.
                @test all(isfinite, α′)
                @test all(isfinite, θ′)

                # (c) M is gauge-invariant: recomposing with the
                # decomposed (α′, θ′) reproduces the original M to
                # machine precision (even though θ′ may differ from
                # θ_in due to the iso-slice gauge degeneracy).
                L2 = cholesky_recompose_3d(α′, θ′)
                M_in = L * transpose(L)
                M_out = L2 * transpose(L2)
                @test maximum(abs.(M_in .- M_out)) ≤ _M3_7_PREP_TOL

                # M = L Lᵀ = α² I on the iso slice (gauge-invariant).
                @test M_in[1, 1] ≈ α0^2  atol = _M3_7_PREP_TOL
                @test M_in[2, 2] ≈ α0^2  atol = _M3_7_PREP_TOL
                @test M_in[3, 3] ≈ α0^2  atol = _M3_7_PREP_TOL
                @test abs(M_in[1, 2]) ≤ _M3_7_PREP_TOL
                @test abs(M_in[1, 3]) ≤ _M3_7_PREP_TOL
                @test abs(M_in[2, 3]) ≤ _M3_7_PREP_TOL
            end
        end
    end

    @testset "iso-pullback: γ_a are all equal at α-iso, M_vv-iso" begin
        # γ at α_1 = α_2 = α_3, M_vv,11 = M_vv,22 = M_vv,33 → γ all equal.
        α = SVector{3, Float64}(1.5, 1.5, 1.5)
        Mvv_diag = SVector{3, Float64}(2.0, 2.0, 2.0)
        β = SVector{3, Float64}(0.1, 0.1, 0.1)
        γ = gamma_per_axis_3d(β, Mvv_diag)
        @test γ[1] ≈ γ[2]  atol = _M3_7_PREP_TOL
        @test γ[2] ≈ γ[3]  atol = _M3_7_PREP_TOL
        # Concrete value: √(2.0 - 0.01) = √1.99
        @test γ[1] ≈ sqrt(1.99)  atol = _M3_7_PREP_TOL
    end

    # ─────────────────────────────────────────────────────────────
    # Block 5: 2D reduction (3D → 2D dimension lift)
    # ─────────────────────────────────────────────────────────────
    @testset "2D reduction: θ_13 = θ_23 = 0 reproduces 2D byte-equal" begin
        # At θ_13 = θ_23 = 0, R(θ_12, 0, 0) = R_12(θ_12) which is the
        # standard 2D rotation R(θ_R) padded with axis-3 = identity.
        # So `cholesky_recompose_3d((α_1, α_2, α_3), (θ_R, 0, 0))`
        # yields a 3×3 Cholesky factor whose top-left 2×2 block equals
        # `cholesky_recompose_2d((α_1, α_2), θ_R)` byte-equally, with
        # the third row/column being (0, 0, α_3).

        for (α_2d_a, α_2d_b, α_3d_extra) in (
            (2.5, 1.2, 0.5),     # standard sort
            (3.0, 1.0, 0.5),
            (1.5, 1.4, 0.3),
            (2.0, 0.5, 0.1),
        )
            for θR in (0.0, 0.31, -0.5, 1.0, -1.2)
                α2 = SVector{2, Float64}(α_2d_a, α_2d_b)
                L2 = cholesky_recompose_2d(α2, θR)

                α3 = SVector{3, Float64}(α_2d_a, α_2d_b, α_3d_extra)
                θ3 = SVector{3, Float64}(θR, 0.0, 0.0)
                L3 = cholesky_recompose_3d(α3, θ3)

                # Top-left 2×2 block of L3 byte-equals L2.
                @test L3[1, 1] == L2[1, 1]
                @test L3[1, 2] == L2[1, 2]
                @test L3[2, 1] == L2[2, 1]
                @test L3[2, 2] == L2[2, 2]
                # Third row/column zero except diagonal.
                @test abs(L3[3, 1]) ≤ _M3_7_PREP_TOL
                @test abs(L3[3, 2]) ≤ _M3_7_PREP_TOL
                @test L3[3, 3] ≈ α_3d_extra  atol = _M3_7_PREP_TOL
                @test abs(L3[1, 3]) ≤ _M3_7_PREP_TOL
                @test abs(L3[2, 3]) ≤ _M3_7_PREP_TOL

                # Decomposition recovers the 2D angle as θ_12 with
                # θ_13 = θ_23 = 0 (provided the sort places the 2D
                # eigenvalues into the top two slots, which the input
                # ensures by construction with α_3d_extra < min(α2)).
                α3′, θ3′ = cholesky_decompose_3d(L3)
                @test α3′[1] ≈ α_2d_a       atol = _M3_7_PREP_TOL
                @test α3′[2] ≈ α_2d_b       atol = _M3_7_PREP_TOL
                @test α3′[3] ≈ α_3d_extra   atol = _M3_7_PREP_TOL
                @test θ3′[1] ≈ θR           atol = _M3_7_PREP_TOL
                @test abs(θ3′[2]) ≤ _M3_7_PREP_TOL
                @test abs(θ3′[3]) ≤ _M3_7_PREP_TOL
            end
        end
    end

    # ─────────────────────────────────────────────────────────────
    # Block 6: per-axis γ on anisotropic M_vv
    # ─────────────────────────────────────────────────────────────
    @testset "per-axis γ on anisotropic M_vv (3D principal-axis frame)" begin
        # In the principal-axis frame, M_vv = diag(M_vv,11, M_vv,22, M_vv,33)
        # gives γ_a = √(M_vv,aa / α_a²).
        α = SVector{3, Float64}(2.0, 0.5, 1.0)
        # Construct M_vv with diagonal (9, 4, 4) and zero off-diagonal.
        M_vv = SMatrix{3, 3, Float64, 9}(9.0, 0.0, 0.0,
                                          0.0, 4.0, 0.0,
                                          0.0, 0.0, 4.0)
        γ = gamma_per_axis_3d(α, M_vv)
        # γ_1 = √(9/4) = 1.5
        # γ_2 = √(4/0.25) = 4.0
        # γ_3 = √(4/1) = 2.0
        @test γ[1] ≈ 1.5  atol = _M3_7_PREP_TOL
        @test γ[2] ≈ 4.0  atol = _M3_7_PREP_TOL
        @test γ[3] ≈ 2.0  atol = _M3_7_PREP_TOL

        # Realizability floor: M_vv,aa = 0 → γ_a = 0 (not NaN, Inf).
        M_vv_floor = SMatrix{3, 3, Float64, 9}(0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0)
        γ0 = gamma_per_axis_3d(α, M_vv_floor)
        @test γ0[1] == 0.0
        @test γ0[2] == 0.0
        @test γ0[3] == 0.0
        @test all(isfinite, γ0)
    end

    @testset "per-axis γ M1-form: γ²_a = M_vv,aa − β_a²" begin
        # By-hand calculation for β = (0.10, 0.05, 0.03), Mvv_diag =
        # (0.40, 0.30, 0.20):
        #   γ_1² = 0.40 - 0.01 = 0.39 → γ_1 = √0.39
        #   γ_2² = 0.30 - 0.0025 = 0.2975 → γ_2 = √0.2975
        #   γ_3² = 0.20 - 0.0009 = 0.1991 → γ_3 = √0.1991
        β = SVector{3, Float64}(0.10, 0.05, 0.03)
        Mvv_diag = SVector{3, Float64}(0.40, 0.30, 0.20)
        γ = gamma_per_axis_3d(β, Mvv_diag)
        @test γ[1] ≈ sqrt(0.39)    atol = _M3_7_PREP_TOL
        @test γ[2] ≈ sqrt(0.2975)  atol = _M3_7_PREP_TOL
        @test γ[3] ≈ sqrt(0.1991)  atol = _M3_7_PREP_TOL

        # Realizability floor in the M1 form: M_vv − β² < 0 → γ = 0.
        β_huge = SVector{3, Float64}(2.0, 2.0, 2.0)
        γ_floor = gamma_per_axis_3d(β_huge, Mvv_diag)
        @test γ_floor[1] == 0.0
        @test γ_floor[2] == 0.0
        @test γ_floor[3] == 0.0
        @test all(isfinite, γ_floor)
    end

    # ─────────────────────────────────────────────────────────────
    # Block 7: hot-path allocation
    # ─────────────────────────────────────────────────────────────
    @testset "hot-path allocation: SVector / SMatrix inputs" begin
        α  = SVector{3, Float64}(2.0, 1.5, 0.8)
        θ  = SVector{3, Float64}(0.31, -0.18, 0.42)
        L  = cholesky_recompose_3d(α, θ)   # warm
        Mv = SMatrix{3, 3, Float64, 9}(3.0, 0.0, 0.0,
                                        0.0, 1.0, 0.0,
                                        0.0, 0.0, 0.5)
        Mvd = SVector{3, Float64}(3.0, 1.0, 0.5)
        β   = SVector{3, Float64}(0.1, 0.05, 0.02)
        _   = cholesky_decompose_3d(L)
        _   = gamma_per_axis_3d(α, Mv)
        _   = gamma_per_axis_3d(β, Mvd)

        # Closures pin the type-stable entry points.
        f_recomp = (α, θ) -> cholesky_recompose_3d(α, θ)
        f_decomp = (L,)    -> cholesky_decompose_3d(L)
        f_gamma1 = (α, Mv) -> gamma_per_axis_3d(α, Mv)
        f_gamma2 = (β, Md) -> gamma_per_axis_3d(β, Md)

        # Warm.
        f_recomp(α, θ); f_decomp(L); f_gamma1(α, Mv); f_gamma2(β, Mvd)

        a1 = @allocated f_recomp(α, θ)
        a2 = @allocated f_decomp(L)
        a3 = @allocated f_gamma1(α, Mv)
        a4 = @allocated f_gamma2(β, Mvd)

        @test a1 == 0
        @test a2 == 0
        @test a3 == 0
        @test a4 == 0
    end

    # ─────────────────────────────────────────────────────────────
    # Block 8: identity recomposition (θ = 0)
    # ─────────────────────────────────────────────────────────────
    @testset "θ = (0, 0, 0): canonical L is diagonal" begin
        α = SVector{3, Float64}(1.5, 0.7, 0.3)
        θ = SVector{3, Float64}(0.0, 0.0, 0.0)
        L = cholesky_recompose_3d(α, θ)
        @test L[1, 1] == α[1]
        @test L[2, 2] == α[2]
        @test L[3, 3] == α[3]
        @test L[1, 2] == 0.0
        @test L[1, 3] == 0.0
        @test L[2, 1] == 0.0
        @test L[2, 3] == 0.0
        @test L[3, 1] == 0.0
        @test L[3, 2] == 0.0
    end
end
