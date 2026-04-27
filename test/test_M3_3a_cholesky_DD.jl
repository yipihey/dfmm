# test_M3_3a_cholesky_DD.jl
#
# M3-3a per-axis Cholesky decomposition driver tests
# (`src/cholesky_DD.jl`). Three blocks:
#
#   1. `cholesky_decompose_2d ∘ cholesky_recompose_2d ≡ id` to ≤ 1e-14
#      absolute on 50 random `(α, θ_R)` inputs (round-trip).
#   2. Iso slice `α_1 = α_2`: per-axis γ entries equal; θ_R from
#      `cholesky_decompose_2d` is well-defined (we return 0 on the iso
#      boundary as a deterministic gauge — the Berry kinetic term
#      vanishes anyway).
#   3. Per-axis γ diagnostic: known-input test where `M_vv` has
#      anisotropic diagonal entries; verify γ correctly reports per-axis
#      values; allocation-free hot path.
#
# See `reference/notes_M3_3_2d_cholesky_berry.md` §4.1, §4.3, §6.3.

using Test
using Random
using StaticArrays
using dfmm: cholesky_decompose_2d, cholesky_recompose_2d, gamma_per_axis_2d

const _M3_3A_CDD_TOL = 1e-13

@testset "M3-3a per-axis Cholesky decomposition driver" begin

    # ─────────────────────────────────────────────────────────────
    # Block 1: round-trip on random inputs
    # ─────────────────────────────────────────────────────────────
    @testset "round-trip: decompose ∘ recompose ≡ id (50 random inputs)" begin
        rng = MersenneTwister(0xCD20)
        n_pass = 0
        for _ in 1:50
            # Sample α_1 ≥ α_2 > 0 (the canonical sort convention).
            a = 0.1 + 5 * rand(rng)
            b = 0.05 + (a - 0.05) * rand(rng)   # 0.05 ≤ b ≤ a
            α  = SVector{2, Float64}(a, b)
            θR = (rand(rng) - 0.5) * π          # θ_R ∈ (-π/2, π/2]; sample (-π/2, π/2)

            L = cholesky_recompose_2d(α, θR)
            α′, θR′ = cholesky_decompose_2d(L)

            @test α′[1] ≈ α[1]  atol = _M3_3A_CDD_TOL
            @test α′[2] ≈ α[2]  atol = _M3_3A_CDD_TOL
            @test θR′  ≈ θR     atol = _M3_3A_CDD_TOL
            n_pass += 1
        end
        @test n_pass == 50
    end

    # ─────────────────────────────────────────────────────────────
    # Block 2: iso-pullback gauge (α_1 = α_2)
    # ─────────────────────────────────────────────────────────────
    @testset "iso slice α_1 = α_2: deterministic gauge" begin
        # On the iso slice the rotation angle is gauge-equivalent to any
        # value (M = α² · I commutes with every rotation). We pick the
        # convention θ_R = 0 (i.e. atan(0, 0) = 0).
        for α0 in (0.5, 1.0, 2.7)
            α = SVector{2, Float64}(α0, α0)
            for θR_in in (0.0, 0.3, -0.7)
                L = cholesky_recompose_2d(α, θR_in)
                α′, θR′ = cholesky_decompose_2d(L)
                # α-eigenvalues equal up to round-off.
                @test α′[1] ≈ α0  atol = _M3_3A_CDD_TOL
                @test α′[2] ≈ α0  atol = _M3_3A_CDD_TOL
                # Gauge: re-decomposed angle is zero (the canonical
                # representative on the iso slice). The point of this
                # test is determinism, not bit-exactness vs the input
                # angle.
                @test θR′ ≈ 0.0  atol = _M3_3A_CDD_TOL
            end
        end
    end

    # ─────────────────────────────────────────────────────────────
    # Block 3: per-axis γ diagnostic
    # ─────────────────────────────────────────────────────────────
    @testset "per-axis γ on anisotropic M_vv (principal-axis frame)" begin
        # In the principal-axis frame, M_vv = diag(M_vv,11, M_vv,22)
        # gives γ_a = √(M_vv,aa / α_a²).
        α = SVector{2, Float64}(2.0, 0.5)
        M_vv = SMatrix{2, 2, Float64, 4}(9.0, 0.0,
                                          0.0, 4.0)
        γ = gamma_per_axis_2d(α, M_vv)
        # γ_1 = √(9/4) = 1.5
        # γ_2 = √(4/0.25) = 4.0
        @test γ[1] ≈ 1.5  atol = _M3_3A_CDD_TOL
        @test γ[2] ≈ 4.0  atol = _M3_3A_CDD_TOL

        # Realizability floor: M_vv,aa = 0 → γ_a = 0 (not NaN, not
        # Inf).
        M_vv_floor = SMatrix{2, 2, Float64, 4}(0.0, 0.0,
                                                0.0, 0.0)
        γ0 = gamma_per_axis_2d(α, M_vv_floor)
        @test γ0[1] == 0.0
        @test γ0[2] == 0.0
        @test !isnan(γ0[1]) && !isnan(γ0[2])
    end

    @testset "per-axis γ: iso slice gives equal γ entries" begin
        α  = SVector{2, Float64}(1.3, 1.3)
        Mv = SMatrix{2, 2, Float64, 4}(2.5, 0.0, 0.0, 2.5)
        γ  = gamma_per_axis_2d(α, Mv)
        @test γ[1] ≈ γ[2]
        # Concrete value: √(2.5 / 1.69) = √(1.479…)
        @test γ[1] ≈ sqrt(2.5 / (1.3^2))  atol = _M3_3A_CDD_TOL
    end

    # ─────────────────────────────────────────────────────────────
    # Block 4: hot-path allocation
    # ─────────────────────────────────────────────────────────────
    @testset "hot-path allocation: SVector / SMatrix inputs" begin
        α    = SVector{2, Float64}(1.7, 0.6)
        θR   = 0.42
        L    = cholesky_recompose_2d(α, θR)  # warm
        Mv   = SMatrix{2, 2, Float64, 4}(3.0, 0.0, 0.0, 1.0)
        _    = cholesky_decompose_2d(L)
        _    = gamma_per_axis_2d(α, Mv)

        # Three closures pin the type-stable entry points.
        f_recomp = (α, θR) -> cholesky_recompose_2d(α, θR)
        f_decomp = (L,)    -> cholesky_decompose_2d(L)
        f_gamma  = (α, Mv) -> gamma_per_axis_2d(α, Mv)

        # Warm.
        f_recomp(α, θR); f_decomp(L); f_gamma(α, Mv)

        a1 = @allocated f_recomp(α, θR)
        a2 = @allocated f_decomp(L)
        a3 = @allocated f_gamma(α, Mv)

        @test a1 == 0
        @test a2 == 0
        @test a3 == 0
    end

    # ─────────────────────────────────────────────────────────────
    # Block 5: identity recomposition
    # ─────────────────────────────────────────────────────────────
    @testset "θ_R = 0 axis-aligned recomposition is diagonal" begin
        α = SVector{2, Float64}(1.5, 0.4)
        L = cholesky_recompose_2d(α, 0.0)
        # Canonical L at θ_R = 0 is exactly diag(α).
        @test L[1, 1] == α[1]
        @test L[2, 2] == α[2]
        @test L[1, 2] == 0.0
        @test L[2, 1] == 0.0
    end

    @testset "θ_R = π/2 swaps the columns up to sign" begin
        α = SVector{2, Float64}(2.0, 0.7)
        L = cholesky_recompose_2d(α, π / 2)
        # R(π/2) = [[0, -1], [1, 0]] so L = R · diag(α) yields
        # column 1 = (0, α_1), column 2 = (-α_2, 0).
        @test L[1, 1] ≈ 0.0     atol = _M3_3A_CDD_TOL
        @test L[2, 1] ≈ α[1]    atol = _M3_3A_CDD_TOL
        @test L[1, 2] ≈ -α[2]   atol = _M3_3A_CDD_TOL
        @test L[2, 2] ≈ 0.0     atol = _M3_3A_CDD_TOL
        # Round-trip still works (modulo the angle gauge: θ_R = π/2
        # is on the boundary of the half-open range (-π/2, π/2], so
        # decompose may return either +π/2 or its equivalent gauge).
        α′, θR′ = cholesky_decompose_2d(L)
        @test α′[1] ≈ α[1] atol = _M3_3A_CDD_TOL
        @test α′[2] ≈ α[2] atol = _M3_3A_CDD_TOL
        @test abs(abs(θR′) - π/2) < _M3_3A_CDD_TOL
    end
end
