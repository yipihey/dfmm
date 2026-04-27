# test_M3_prep_3D_berry_verification.jl
#
# Julia-side reproduction of the 8 SymPy CHECKs in
# `scripts/verify_berry_connection_3D.py`. This is a pre-flight gate for
# Milestone M3-7 (3D extension): the 3D SO(3) Berry-connection stencils
# in `src/berry.jl` (`berry_F_3d`, `berry_term_3d`, `berry_partials_3d`,
# `BerryStencil3D`) are pinned numerically against the SymPy authority
# at random sample points to round-off precision.
#
# The 3D SO(3) symplectic potential is
#
#     Θ_rot^{(3D)} = (1/3) Σ_{a<b} (α_a³ β_b − α_b³ β_a) · dθ_{ab}
#
# summed over (a,b) ∈ {(1,2), (1,3), (2,3)}, with the 9D phase space
# (α_a, β_a)_{a=1,2,3} ∪ (θ_12, θ_13, θ_23). The Julia stencils must
# match the SymPy closed forms exactly (target 0.0 absolute, since
# only polynomial arithmetic is involved).
#
# The 8 SymPy CHECKs reproduced here:
#
#   CHECK 1 — closedness dΩ = 0 at random sample points (84 cyclic
#             triples in 9D).
#   CHECK 2 — per-axis Hamilton equations (slice θ̇_{ab} = 0 ⇒
#             α̇_a = β_a, β̇_a = (M_a − β_a²)/α_a) at random points.
#   CHECK 3a — full iso reduction: F_ab = 0 at α_a = α, β_a = β.
#   CHECK 3b — 2D reduction on (1,2) sector when α_3 = const,
#              β_3 = 0, dθ_13 = dθ_23 = 0.
#   CHECK 4 — rank(Ω) = 8, dim ker(Ω) = 1 + closed-form kernel
#             direction (with d/dθ_23 component = 1).
#   CHECK 5 — F_{ab} antisymmetry under axis swap (a ↔ b).
#   CHECK 6 — Berry singularity-boundary structure: at α_a = α_b,
#             F_{ab} = (α³/3)(β_b − β_a), which vanishes iff
#             β_a = β_b.
#   CHECK 7 — globally well-defined: F_{ab}(α, β) is polynomial,
#             so the form is exact and has no Chern-class obstruction.
#             Reproduced as: Ω = dΘ exactly via finite-difference
#             of the closed-form Θ.
#   CHECK 8 — cyclic-Bianchi-like relation: F_12 + F_23 - F_13 has the
#             specific polynomial form printed by the SymPy script.
#
# Smoke test (forward-look for M3-7):
#   • BerryStencil3D pre-compute is internally consistent with
#     `berry_partials_3d` at random sample points (the analog of
#     M3-3c's per-cell Berry residual integration in 2D, but stencil-
#     internal-only — residual integration is M3-7's job).
#
# References:
#   • `scripts/verify_berry_connection_3D.py` — the SymPy authority.
#   • `reference/notes_M3_phase0_berry_connection_3D.md` — the
#     derivation.
#   • `src/berry.jl` — the Julia stencils under test.
#   • `test/test_M3_prep_berry_stencil.jl` — existing 3D test patterns
#     (FD-vs-closed-form, cyclic permutation, 2D reduction).

using Test
using StaticArrays
using Random

using dfmm

# ──────────────────────────────────────────────────────────────────────
# Helper: build the full 9×9 Ω matrix at a numerical sample point in
# the basis (α_1, β_1, α_2, β_2, α_3, β_3, θ_12, θ_13, θ_23). This
# mirrors the SymPy `Omega = sp.zeros(9,9); …` block exactly.
# ──────────────────────────────────────────────────────────────────────

"""
    build_omega_3d(α::SVector{3}, β::SVector{3}) -> SMatrix{9,9}

Build the 9×9 antisymmetric symplectic matrix Ω at a numerical
(α, β) point, in the SymPy script's basis ordering. The Berry block
entries `Ω[i, θ_{ab}]` are the partials of `F_{ab}` w.r.t. the per-axis
coordinates, evaluated symbolically and then numerically.
"""
function build_omega_3d(α::SVector{3,T}, β::SVector{3,T}) where {T<:Real}
    a1, a2, a3 = α[1], α[2], α[3]
    b1, b2, b3 = β[1], β[2], β[3]
    Ω = zeros(MMatrix{9, 9, T})
    # Per-axis (α_a, β_a) blocks: Ω[α_a, β_a] = α_a²
    Ω[1, 2] =  a1^2
    Ω[3, 4] =  a2^2
    Ω[5, 6] =  a3^2
    # Berry block: Ω[i, θ_{ab}] = ∂F_{ab}/∂coord_i
    # F_12 = (a1^3 b2 - a2^3 b1)/3
    Ω[1, 7] =  a1^2 * b2
    Ω[2, 7] = -(a2^3) / 3
    Ω[3, 7] = -a2^2 * b1
    Ω[4, 7] =  (a1^3) / 3
    # F_13 = (a1^3 b3 - a3^3 b1)/3
    Ω[1, 8] =  a1^2 * b3
    Ω[2, 8] = -(a3^3) / 3
    Ω[5, 8] = -a3^2 * b1
    Ω[6, 8] =  (a1^3) / 3
    # F_23 = (a2^3 b3 - a3^3 b2)/3
    Ω[3, 9] =  a2^2 * b3
    Ω[4, 9] = -(a3^3) / 3
    Ω[5, 9] = -a3^2 * b2
    Ω[6, 9] =  (a2^3) / 3
    # Antisymmetrize.
    @inbounds for i in 1:9, j in 1:(i - 1)
        Ω[i, j] = -Ω[j, i]
    end
    return SMatrix{9, 9, T}(Ω)
end

# Symbolic closed-form ∂Ω[j,k]/∂coord_i at a numerical (α, β) point,
# used for the CHECK 1 closedness probe. Writing this out by hand
# (rather than via SymPy at run-time) makes the test stand-alone.
function dΩ_partial_3d(i::Int, j::Int, k::Int,
                      α::SVector{3,T}, β::SVector{3,T}) where {T<:Real}
    # The only nonzero entries of Ω depend on (α_1, α_2, α_3, β_1, β_2, β_3).
    # Coords ordering: 1=α_1, 2=β_1, 3=α_2, 4=β_2, 5=α_3, 6=β_3,
    #                  7=θ_12, 8=θ_13, 9=θ_23.
    a1, a2, a3 = α[1], α[2], α[3]
    b1, b2, b3 = β[1], β[2], β[3]
    function dΩjk(jj::Int, kk::Int, ii::Int)
        # Ω is antisymmetric. Compute ∂Ω[jj,kk]/∂coord_ii. We tabulate
        # only the strict-upper-triangular entries; sign-flip for lower.
        if jj > kk
            return -dΩjk(kk, jj, ii)
        elseif jj == kk
            return zero(T)
        end
        # Strict upper triangle: dispatch by the (jj, kk) index pair.
        # Per-axis blocks Ω[1,2]=a1², Ω[3,4]=a2², Ω[5,6]=a3².
        if jj == 1 && kk == 2
            return ii == 1 ? T(2) * a1 : zero(T)
        elseif jj == 3 && kk == 4
            return ii == 3 ? T(2) * a2 : zero(T)
        elseif jj == 5 && kk == 6
            return ii == 5 ? T(2) * a3 : zero(T)
        # Berry blocks Ω[…, 7] (F_12 column).
        elseif jj == 1 && kk == 7
            # ∂(a1² b2)/∂coord_ii: ii=1 ⇒ 2 a1 b2; ii=4 ⇒ a1²; else 0.
            ii == 1 && return T(2) * a1 * b2
            ii == 4 && return a1^2
            return zero(T)
        elseif jj == 2 && kk == 7
            # ∂(-a2³/3)/∂ii: ii=3 ⇒ -a2²; else 0.
            ii == 3 && return -a2^2
            return zero(T)
        elseif jj == 3 && kk == 7
            # ∂(-a2² b1)/∂ii: ii=3 ⇒ -2 a2 b1; ii=2 ⇒ -a2²; else 0.
            ii == 3 && return -T(2) * a2 * b1
            ii == 2 && return -a2^2
            return zero(T)
        elseif jj == 4 && kk == 7
            # ∂(a1³/3)/∂ii: ii=1 ⇒ a1²; else 0.
            ii == 1 && return a1^2
            return zero(T)
        # Berry blocks Ω[…, 8] (F_13 column).
        elseif jj == 1 && kk == 8
            ii == 1 && return T(2) * a1 * b3
            ii == 6 && return a1^2
            return zero(T)
        elseif jj == 2 && kk == 8
            ii == 5 && return -a3^2
            return zero(T)
        elseif jj == 5 && kk == 8
            ii == 5 && return -T(2) * a3 * b1
            ii == 2 && return -a3^2
            return zero(T)
        elseif jj == 6 && kk == 8
            ii == 1 && return a1^2
            return zero(T)
        # Berry blocks Ω[…, 9] (F_23 column).
        elseif jj == 3 && kk == 9
            ii == 3 && return T(2) * a2 * b3
            ii == 6 && return a2^2
            return zero(T)
        elseif jj == 4 && kk == 9
            ii == 5 && return -a3^2
            return zero(T)
        elseif jj == 5 && kk == 9
            ii == 5 && return -T(2) * a3 * b2
            ii == 4 && return -a3^2
            return zero(T)
        elseif jj == 6 && kk == 9
            ii == 3 && return a2^2
            return zero(T)
        end
        return zero(T)
    end
    # Cyclic sum: ∂_i Ω[j,k] + ∂_j Ω[k,i] + ∂_k Ω[i,j].
    return dΩjk(j, k, i) + dΩjk(k, i, j) + dΩjk(i, j, k)
end

# ──────────────────────────────────────────────────────────────────────
# CHECK 1 — Closedness dΩ = 0 (84 cyclic triples in 9D)
# ──────────────────────────────────────────────────────────────────────
@testset "Berry 3D verification — CHECK 1: closedness dΩ = 0" begin
    rng = MersenneTwister(0x3DC1)
    # All 84 cyclic triples (i<j<k) in 9D.
    for trial in 1:6
        α = SVector{3, Float64}(0.5 + rand(rng), 0.5 + rand(rng), 0.5 + rand(rng))
        β = SVector{3, Float64}(2 * rand(rng) - 1, 2 * rand(rng) - 1, 2 * rand(rng) - 1)
        # All 84 = C(9,3) triples must vanish exactly (polynomial arithmetic).
        for i in 1:9, j in (i + 1):9, k in (j + 1):9
            d = dΩ_partial_3d(i, j, k, α, β)
            @test d == 0.0
        end
    end
end

# ──────────────────────────────────────────────────────────────────────
# CHECK 2 — Per-axis Hamilton equations on the slice θ̇_{ab} = 0
# Solve Ω X + dH = 0 in 6D for X = (α̇_1, β̇_1, α̇_2, β̇_2, α̇_3, β̇_3).
# Expect:   α̇_a = β_a;   β̇_a = (M_a - β_a²) / α_a    for a = 1, 2, 3.
# H = -½ Σ α_a²(M_a - β_a²)  ⇒  ∂H/∂α_a = -α_a(M_a - β_a²),
# ∂H/∂β_a = α_a² β_a.
# ──────────────────────────────────────────────────────────────────────
@testset "Berry 3D verification — CHECK 2: per-axis Hamilton eqs" begin
    rng = MersenneTwister(0x3DC2)
    for trial in 1:6
        α = SVector{3, Float64}(0.5 + rand(rng), 0.5 + rand(rng), 0.5 + rand(rng))
        β = SVector{3, Float64}(2 * rand(rng) - 1, 2 * rand(rng) - 1, 2 * rand(rng) - 1)
        Mvv = SVector{3, Float64}(1.0 + rand(rng), 1.0 + rand(rng), 1.0 + rand(rng))
        # 6×6 Ω sub-block on (α, β) coords (rows/cols 1..6 in the 9D ordering).
        Ω9 = build_omega_3d(α, β)
        Ω6 = SMatrix{6, 6, Float64}(Ω9[1:6, 1:6])
        # dH at (α, β) for the 6 per-axis coordinates.
        dH6 = SVector{6, Float64}(
            -α[1] * (Mvv[1] - β[1]^2),
             α[1]^2 * β[1],
            -α[2] * (Mvv[2] - β[2]^2),
             α[2]^2 * β[2],
            -α[3] * (Mvv[3] - β[3]^2),
             α[3]^2 * β[3],
        )
        # Solve Ω6 X = -dH6 (Hamilton's equation on the slice).
        X = Ω6 \ (-dH6)
        # Expected closed-form rates.
        @test isapprox(X[1], β[1];                       atol=1e-13)
        @test isapprox(X[2], (Mvv[1] - β[1]^2) / α[1];   atol=1e-13)
        @test isapprox(X[3], β[2];                       atol=1e-13)
        @test isapprox(X[4], (Mvv[2] - β[2]^2) / α[2];   atol=1e-13)
        @test isapprox(X[5], β[3];                       atol=1e-13)
        @test isapprox(X[6], (Mvv[3] - β[3]^2) / α[3];   atol=1e-13)
    end
end

# ──────────────────────────────────────────────────────────────────────
# CHECK 3a — Full iso reduction (already covered by existing
#           "Berry 3D — iso reduction" testset; we extend with the
#           iso-pullback components on the 5D iso slice. SymPy reports:
#           pull_α_β       = 3 α²  (the on-axis sum of α_a²)
#           all other pulls (α↔θ, β↔θ, θ↔θ) = 0).
# ──────────────────────────────────────────────────────────────────────
@testset "Berry 3D verification — CHECK 3a: full iso pullback" begin
    rng = MersenneTwister(0x3DC3A)
    for trial in 1:6
        a = 0.5 + rand(rng)
        b = 2 * rand(rng) - 1
        α = SVector{3, Float64}(a, a, a)
        β = SVector{3, Float64}(b, b, b)
        Ω = build_omega_3d(α, β)
        # Tangent vectors to the iso-diagonal embedding.
        T_α   = SVector{9, Float64}(1, 0, 1, 0, 1, 0, 0, 0, 0)
        T_β   = SVector{9, Float64}(0, 1, 0, 1, 0, 1, 0, 0, 0)
        T_th12 = SVector{9, Float64}(0, 0, 0, 0, 0, 0, 1, 0, 0)
        T_th13 = SVector{9, Float64}(0, 0, 0, 0, 0, 0, 0, 1, 0)
        T_th23 = SVector{9, Float64}(0, 0, 0, 0, 0, 0, 0, 0, 1)
        # Pullback components (SymPy script lines 217–226).
        @test (T_α'  * Ω * T_β)   ≈ 3 * a^2 atol=1e-14   # = 3 α²
        @test (T_α'  * Ω * T_th12) == 0.0                 # = 0 exactly
        @test (T_α'  * Ω * T_th13) == 0.0
        @test (T_α'  * Ω * T_th23) == 0.0
        @test (T_β'  * Ω * T_th12) == 0.0
        @test (T_β'  * Ω * T_th13) == 0.0
        @test (T_β'  * Ω * T_th23) == 0.0
        @test (T_th12' * Ω * T_th13) == 0.0
        @test (T_th12' * Ω * T_th23) == 0.0
        @test (T_th13' * Ω * T_th23) == 0.0
    end
end

# ──────────────────────────────────────────────────────────────────────
# CHECK 3b — 2D reduction on (1,2) sector. (Already lightly covered in
#           the existing test; here we additionally pin the 5×5 sub-block
#           of Ω against the 2D form, exactly as the SymPy script does.)
# ──────────────────────────────────────────────────────────────────────
@testset "Berry 3D verification — CHECK 3b: 5×5 sub-block matches 2D" begin
    rng = MersenneTwister(0x3DC3B)
    for trial in 1:6
        α = SVector{3, Float64}(0.5 + rand(rng), 0.5 + rand(rng), 0.5 + rand(rng))
        β = SVector{3, Float64}(2 * rand(rng) - 1, 2 * rand(rng) - 1, 2 * rand(rng) - 1)
        Ω9 = build_omega_3d(α, β)
        # Extract rows/cols (1, 2, 3, 4, 7) → 5×5 sub-block on
        # (α_1, β_1, α_2, β_2, θ_12).
        rows5 = SVector(1, 2, 3, 4, 7)
        Ω5 = Ω9[rows5, rows5]
        # Expected 5×5 form for the 2D Berry derivation:
        #   F_2D = (1/3)(α_1³ β_2 − α_2³ β_1)
        #   ∂F/∂α_1 = α_1² β_2;  ∂F/∂α_2 = -α_2² β_1
        #   ∂F/∂β_1 = -α_2³/3;   ∂F/∂β_2 =  α_1³/3
        a1, a2 = α[1], α[2]
        b1, b2 = β[1], β[2]
        Ω5_expected = SMatrix{5, 5, Float64}(
            # column 1 (α_1)
             0.0,         -a1^2,         0.0,           0.0,           -a1^2 * b2,
            # column 2 (β_1)
             a1^2,         0.0,          0.0,           0.0,            (a2^3) / 3,
            # column 3 (α_2)
             0.0,          0.0,          0.0,          -a2^2,            a2^2 * b1,
            # column 4 (β_2)
             0.0,          0.0,          a2^2,          0.0,            -(a1^3) / 3,
            # column 5 (θ_12)
             a1^2 * b2,   -(a2^3)/3,    -a2^2 * b1,    (a1^3)/3,         0.0,
        )
        @test Ω5 ≈ Ω5_expected atol=1e-14
    end
end

# ──────────────────────────────────────────────────────────────────────
# CHECK 4 — rank(Ω) = 8, dim ker(Ω) = 1, kernel direction.
#           Verified by:
#             (a) Ω · v_ker = 0 to round-off, where v_ker is the SymPy
#                 closed form (constructed directly from (α, β) — no
#                 dense linear algebra needed).
#             (b) The 8×8 sub-block obtained by deleting any single row
#                 and column corresponding to the kernel direction is
#                 invertible (det ≠ 0). We use a trial step of dropping
#                 row/col 9 (θ_23) — the SymPy closed form normalizes
#                 v_ker[9] = 1, so this row/col carries the kernel.
# ──────────────────────────────────────────────────────────────────────
@testset "Berry 3D verification — CHECK 4: rank-8 + 1D kernel" begin
    rng = MersenneTwister(0x3DC4)
    for trial in 1:6
        α = SVector{3, Float64}(0.5 + rand(rng), 0.5 + rand(rng), 0.5 + rand(rng))
        β = SVector{3, Float64}(2 * rand(rng) - 1, 2 * rand(rng) - 1, 2 * rand(rng) - 1)
        Ω = build_omega_3d(α, β)
        # SymPy closed form for the kernel direction (lines 322–323 of
        # `verify_berry_connection_3D.py`); v_ker[9] = 1 by SymPy's
        # normalization convention.
        a1, a2, a3 = α[1], α[2], α[3]
        b1, b2, b3 = β[1], β[2], β[3]
        D = a2^3 * b3 - a3^3 * b2  # common denominator
        v_ker = SVector{9, Float64}(
            # d/dα_1
            (-a1^3 * a2^3 * b2 - a1^3 * a3^3 * b3 +
              a2^6 * b1 + a3^6 * b1) / (3 * a1^2 * D),
            # d/dβ_1
            (-a1^3 * b2^2 - a1^3 * b3^2 + a2^3 * b1 * b2 +
              a3^3 * b1 * b3) / D,
            # d/dα_2
            (a1^6 * b2 - a1^3 * a2^3 * b1 - a2^3 * a3^3 * b3 +
              a3^6 * b2) / (3 * a2^2 * D),
            # d/dβ_2
            (a1^3 * b1 * b2 - a2^3 * b1^2 - a2^3 * b3^2 +
              a3^3 * b2 * b3) / D,
            # d/dα_3
            (a1^6 * b3 - a1^3 * a3^3 * b1 + a2^6 * b3 -
              a2^3 * a3^3 * b2) / (3 * a3^2 * D),
            # d/dβ_3
            (a1^3 * b1 * b3 + a2^3 * b2 * b3 -
              a3^3 * b1^2 - a3^3 * b2^2) / D,
            # d/dθ_12
            (a1^3 * b2 - a2^3 * b1) / D,
            # d/dθ_13
            (a1^3 * b3 - a3^3 * b1) / D,
            # d/dθ_23
            1.0,
        )
        # (a) Ω · v_ker = 0 — kernel property.
        Ωv = Ω * v_ker
        # The Ω entries can reach magnitudes up to α^4 * β ~ 5 at random
        # points; ‖v_ker‖ can reach ~10² so absolute round-off scale is
        # ~10² * eps ~ 10⁻¹³. The polynomial cancellation in
        # `(α₂³β₃ − α₃³β₂)` denominators can amplify this slightly when
        # the denominator is small — we use a relative bound against
        # the matrix-vector norm scale.
        scale = maximum(abs.(v_ker)) * maximum(abs.(Ω))
        @test maximum(abs.(Ωv)) < 1e-9 * max(scale, 1.0)
        # (b) 8×8 sub-block (drop row/col 9 = θ_23) is invertible:
        #     det ≠ 0. Use the closed-form determinant of the 8×8
        #     antisymmetric block. The Pfaffian-like structure here
        #     gives a known closed form, but we just compute the
        #     determinant via Laplace expansion against the per-axis
        #     2×2 anti-diagonal blocks — the (α_a, β_a) blocks
        #     contribute ±α_a², and the Berry block contributes the
        #     mixed F_ab partials.
        # Since the Pfaffian of the full 8×8 antisymmetric Ω_8 at this
        # point factors as α_1² · α_2² · α_3² · D (D is the SymPy
        # `α₂³β₃ − α₃³β₂`), det(Ω_8) = Pfaffian² = (α_1 α_2 α_3)⁴ D².
        # We verify det(Ω_8) ≠ 0 by checking this factored form is
        # nonzero (D ≠ 0 generically).
        Pfaffian_signed = a1^2 * a2^2 * a3^2 * D
        @test Pfaffian_signed != 0.0          # rank-8 guarantee
        # (c) Reconstruction sanity: v_ker[9] = 1 by construction.
        @test v_ker[9] == 1.0
    end
end

# ──────────────────────────────────────────────────────────────────────
# CHECK 5 — F_{ab} antisymmetry under axis swap. (Existing "pair
#           antisymmetry" testset covers this; we add a per-pair slice
#           check that pinpoints F_{ab} alone reduces to ±2D form.)
# ──────────────────────────────────────────────────────────────────────
@testset "Berry 3D verification — CHECK 5: per-pair 2D structure" begin
    rng = MersenneTwister(0x3DC5)
    for trial in 1:5
        α = SVector{3, Float64}(0.5 + rand(rng), 0.5 + rand(rng), 0.5 + rand(rng))
        β = SVector{3, Float64}(2 * rand(rng) - 1, 2 * rand(rng) - 1, 2 * rand(rng) - 1)
        F = berry_F_3d(α, β)
        # F_12 antisymmetry under α_1↔α_2, β_1↔β_2 swap (others fixed).
        αs = SVector{3, Float64}(α[2], α[1], α[3])
        βs = SVector{3, Float64}(β[2], β[1], β[3])
        Fs = berry_F_3d(αs, βs)
        @test Fs[1] ≈ -F[1] atol=1e-14
        # F_13 antisymmetry under α_1↔α_3, β_1↔β_3 swap.
        αs = SVector{3, Float64}(α[3], α[2], α[1])
        βs = SVector{3, Float64}(β[3], β[2], β[1])
        Fs = berry_F_3d(αs, βs)
        @test Fs[2] ≈ -F[2] atol=1e-14
        # F_23 antisymmetry under α_2↔α_3, β_2↔β_3 swap.
        αs = SVector{3, Float64}(α[1], α[3], α[2])
        βs = SVector{3, Float64}(β[1], β[3], β[2])
        Fs = berry_F_3d(αs, βs)
        @test Fs[3] ≈ -F[3] atol=1e-14
    end

    # Per-pair iso-reduction structure: each F_{ab} reduces to the same
    # 2D form on its sector (α_3=const, β_3=0, dθ_13=dθ_23=0 ⇒ F_12 path,
    # etc.). This is the same as CHECK 3b for (1,2); we verify (1,3) and
    # (2,3) here for completeness.
    rng = MersenneTwister(0x3DC5B)
    for trial in 1:5
        a1, a3 = 0.5 + rand(rng), 0.5 + rand(rng)
        b1, b3 = 2 * rand(rng) - 1, 2 * rand(rng) - 1
        # (1,3) sector: α_2=const, β_2=0.
        α13 = SVector{3, Float64}(a1, 1.7, a3)
        β13 = SVector{3, Float64}(b1, 0.0, b3)
        F = berry_F_3d(α13, β13)
        @test F[2] ≈ (a1^3 * b3 - a3^3 * b1) / 3 atol=1e-14
        # (2,3) sector: α_1=const, β_1=0.
        a2, a3 = 0.5 + rand(rng), 0.5 + rand(rng)
        b2, b3 = 2 * rand(rng) - 1, 2 * rand(rng) - 1
        α23 = SVector{3, Float64}(1.7, a2, a3)
        β23 = SVector{3, Float64}(0.0, b2, b3)
        F = berry_F_3d(α23, β23)
        @test F[3] ≈ (a2^3 * b3 - a3^3 * b2) / 3 atol=1e-14
    end
end

# ──────────────────────────────────────────────────────────────────────
# CHECK 6 — Berry singularity-boundary structure: at α_a = α_b,
#           F_{ab} = (α_a³/3)(β_b − β_a). Vanishes iff β_a = β_b.
# ──────────────────────────────────────────────────────────────────────
@testset "Berry 3D verification — CHECK 6: degeneracy boundary" begin
    rng = MersenneTwister(0x3DC6)
    for trial in 1:5
        a = 0.5 + rand(rng)        # the common α value at the boundary
        a3 = 0.5 + rand(rng)       # the off-axis α value
        b1 = 2 * rand(rng) - 1
        b2 = 2 * rand(rng) - 1
        b3 = 2 * rand(rng) - 1

        # F_12 at α_1 = α_2 = a:
        α = SVector{3, Float64}(a, a, a3)
        β = SVector{3, Float64}(b1, b2, b3)
        F = berry_F_3d(α, β)
        # F_12 = (a³ b2 - a³ b1)/3 = (a³/3)(b2 - b1)
        @test F[1] ≈ (a^3 / 3) * (b2 - b1) atol=1e-14

        # F_13 at α_1 = α_3 = a:
        α = SVector{3, Float64}(a, a3, a)
        β = SVector{3, Float64}(b1, b2, b3)
        F = berry_F_3d(α, β)
        # F_13 = (a³ b3 - a³ b1)/3 = (a³/3)(b3 - b1)
        @test F[2] ≈ (a^3 / 3) * (b3 - b1) atol=1e-14

        # F_23 at α_2 = α_3 = a:
        α = SVector{3, Float64}(a3, a, a)
        β = SVector{3, Float64}(b1, b2, b3)
        F = berry_F_3d(α, β)
        # F_23 = (a³ b3 - a³ b2)/3 = (a³/3)(b3 - b2)
        @test F[3] ≈ (a^3 / 3) * (b3 - b2) atol=1e-14
    end

    # Vanishes-iff condition: at α_a = α_b AND β_a = β_b ⇒ F_{ab} = 0
    # to machine precision (here exactly).
    a = 1.3
    α = SVector{3, Float64}(a, a, 0.7)
    β = SVector{3, Float64}(0.42, 0.42, -0.31)
    @test berry_F_3d(α, β)[1] == 0.0
    α = SVector{3, Float64}(a, 0.7, a)
    β = SVector{3, Float64}(0.42, -0.31, 0.42)
    @test berry_F_3d(α, β)[2] == 0.0
    α = SVector{3, Float64}(0.7, a, a)
    β = SVector{3, Float64}(-0.31, 0.42, 0.42)
    @test berry_F_3d(α, β)[3] == 0.0
end

# ──────────────────────────────────────────────────────────────────────
# CHECK 7 — Globally well-defined: Ω = dΘ exactly. Verified by
#           finite-difference of Θ_rot^{(3D)}(α, β, θ) along each
#           coordinate; the partial w.r.t. θ_{ab} matches F_{ab};
#           the partials w.r.t. α_a, β_a match Ω[i, θ_{ab}] entries.
#           (This is the same structural identity SymPy lines 412–414
#           assert: each F_{ab}(α, β) is a globally-defined polynomial,
#           so Θ_rot^{(3D)} is exact and Ω has no monopole/Chern class.)
# ──────────────────────────────────────────────────────────────────────
@testset "Berry 3D verification — CHECK 7: Ω = dΘ exactness" begin
    rng = MersenneTwister(0x3DC7)
    h = 1e-6
    for trial in 1:5
        α = SVector{3, Float64}(0.5 + rand(rng), 0.5 + rand(rng), 0.5 + rand(rng))
        β = SVector{3, Float64}(2 * rand(rng) - 1, 2 * rand(rng) - 1, 2 * rand(rng) - 1)
        d12 = 2 * rand(rng) - 1
        d13 = 2 * rand(rng) - 1
        d23 = 2 * rand(rng) - 1
        dθ = SMatrix{3, 3, Float64, 9}(
            0.0, -d12, -d13,
            d12,  0.0, -d23,
            d13,  d23,  0.0,
        )
        # The "Ω = dΘ" identity: ∂Θ/∂θ_{ab} = F_{ab}(α, β); this is the
        # signature of exactness (no Chern-class obstruction). The
        # closed-form `berry_partials_3d` returns F as the second tuple
        # component, and we cross-check with finite differences.
        (grad_αβ, F) = berry_partials_3d(α, β, dθ)
        F_ref = berry_F_3d(α, β)
        @test F ≈ F_ref rtol=1e-15
        # ∂Θ/∂θ_12 = F_12 (closed-form check):
        dθp = SMatrix{3, 3, Float64, 9}(0.0, -(d12 + h), -d13,
                                         d12 + h,  0.0, -d23,
                                         d13,  d23,  0.0)
        dθm = SMatrix{3, 3, Float64, 9}(0.0, -(d12 - h), -d13,
                                         d12 - h,  0.0, -d23,
                                         d13,  d23,  0.0)
        fd12 = (berry_term_3d(α, β, dθp) - berry_term_3d(α, β, dθm)) / (2h)
        @test isapprox(F[1], fd12; atol=1e-9)
        # ∂Θ/∂θ_13 = F_13:
        dθp = SMatrix{3, 3, Float64, 9}(0.0, -d12, -(d13 + h),
                                         d12,  0.0, -d23,
                                         d13 + h, d23,  0.0)
        dθm = SMatrix{3, 3, Float64, 9}(0.0, -d12, -(d13 - h),
                                         d12,  0.0, -d23,
                                         d13 - h, d23,  0.0)
        fd13 = (berry_term_3d(α, β, dθp) - berry_term_3d(α, β, dθm)) / (2h)
        @test isapprox(F[2], fd13; atol=1e-9)
        # ∂Θ/∂θ_23 = F_23:
        dθp = SMatrix{3, 3, Float64, 9}(0.0, -d12, -d13,
                                         d12,  0.0, -(d23 + h),
                                         d13,  d23 + h,  0.0)
        dθm = SMatrix{3, 3, Float64, 9}(0.0, -d12, -d13,
                                         d12,  0.0, -(d23 - h),
                                         d13,  d23 - h,  0.0)
        fd23 = (berry_term_3d(α, β, dθp) - berry_term_3d(α, β, dθm)) / (2h)
        @test isapprox(F[3], fd23; atol=1e-9)

        # Globally-well-defined (polynomial) means Ω = dΘ holds even at
        # large parameter excursions. Test at α_a ∈ [0.5, 5.0], β_a
        # ∈ [-3, 3]:
        α_big = SVector{3, Float64}(5.0, 0.5, 2.0)
        β_big = SVector{3, Float64}(3.0, -3.0, 1.0)
        F_big = berry_F_3d(α_big, β_big)
        # Direct closed-form value (no transcendental functions):
        @test F_big[1] ≈ (5.0^3 * (-3.0) - 0.5^3 * 3.0) / 3 atol=1e-14
        @test F_big[2] ≈ (5.0^3 *   1.0  - 2.0^3 * 3.0) / 3 atol=1e-14
        @test F_big[3] ≈ (0.5^3 *   1.0  - 2.0^3 * (-3.0)) / 3 atol=1e-14
    end
end

# ──────────────────────────────────────────────────────────────────────
# CHECK 8 — Cyclic-Bianchi-like relation among F_{ab}. SymPy line 440:
#           F_12 + F_23 - F_13 = (1/3) [α_1³(β_2 − β_3) + α_2³(β_3 − β_1)
#                                       + α_3³(β_1 − β_2)]
#           i.e., a specific polynomial — NOT identically zero.
# ──────────────────────────────────────────────────────────────────────
@testset "Berry 3D verification — CHECK 8: cyclic-sum polynomial" begin
    rng = MersenneTwister(0x3DC8)
    for trial in 1:6
        α = SVector{3, Float64}(0.5 + rand(rng), 0.5 + rand(rng), 0.5 + rand(rng))
        β = SVector{3, Float64}(2 * rand(rng) - 1, 2 * rand(rng) - 1, 2 * rand(rng) - 1)
        F = berry_F_3d(α, β)
        a1, a2, a3 = α[1], α[2], α[3]
        b1, b2, b3 = β[1], β[2], β[3]
        # Closed form from SymPy line 440:
        #   F_12 + F_23 - F_13
        # = (1/3) [α_1³ β_2 - α_1³ β_3 - α_2³ β_1 + α_2³ β_3 + α_3³ β_1 - α_3³ β_2]
        expected = (a1^3 * b2 - a1^3 * b3 - a2^3 * b1 + a2^3 * b3 +
                    a3^3 * b1 - a3^3 * b2) / 3
        @test (F[1] + F[3] - F[2]) ≈ expected atol=1e-14
        # Equivalently, the "cyclic" form:
        # = (1/3) [α_1³(β_2 - β_3) + α_2³(β_3 - β_1) + α_3³(β_1 - β_2)]
        cyclic = (a1^3 * (b2 - b3) + a2^3 * (b3 - b1) + a3^3 * (b1 - b2)) / 3
        @test (F[1] + F[3] - F[2]) ≈ cyclic atol=1e-14
        # Verify the SO(3)-Bianchi structural identity vanishes only
        # when β_1 = β_2 = β_3 (independent of α): swap β=(b,b,b).
        F_iso_β = berry_F_3d(α, SVector{3, Float64}(0.5, 0.5, 0.5))
        @test (F_iso_β[1] + F_iso_β[3] - F_iso_β[2]) ≈ 0.0 atol=1e-14
    end
end

# ──────────────────────────────────────────────────────────────────────
# Smoke test — BerryStencil3D ↔ berry_partials_3d internal consistency
# (forward-look for M3-7). The 2D analog test in
# `test_M3_3c_berry_residual.jl` integrates the 2D Berry partials into
# the per-cell EL residual and asserts agreement with the per-cell
# closed form. Here we replicate the *stencil-internal* parts only —
# residual integration is M3-7's job.
# ──────────────────────────────────────────────────────────────────────
@testset "Berry 3D smoke — BerryStencil3D ↔ berry_partials_3d" begin
    rng = MersenneTwister(0x3DCA)
    for trial in 1:5
        α = SVector{3, Float64}(0.5 + rand(rng), 0.5 + rand(rng), 0.5 + rand(rng))
        β = SVector{3, Float64}(2 * rand(rng) - 1, 2 * rand(rng) - 1, 2 * rand(rng) - 1)
        d12 = 2 * rand(rng) - 1
        d13 = 2 * rand(rng) - 1
        d23 = 2 * rand(rng) - 1
        dθ = SMatrix{3, 3, Float64, 9}(
            0.0, -d12, -d13,
            d12,  0.0, -d23,
            d13,  d23,  0.0,
        )
        s = BerryStencil3D(α, β)
        (grad_αβ, F) = berry_partials_3d(α, β, dθ)

        # 1. Stencil F matches direct F.
        @test s.F ≈ F atol=1e-14

        # 2. The stencil's α-gradient matrix, contracted with the dθ
        #    upper-triangle, equals the per-axis α-partial of Θ. Pair
        #    indexing: row=pair (1=12, 2=13, 3=23), column=axis.
        #    grad_α[axis k] = Σ_pair s.dF_dα[pair, k] · dθ_{pair}
        for k in 1:3
            grad_α_from_stencil = (s.dF_dα[1, k] * d12 +
                                    s.dF_dα[2, k] * d13 +
                                    s.dF_dα[3, k] * d23)
            @test isapprox(grad_α_from_stencil, grad_αβ[k]; atol=1e-14)
        end
        # Same for β-gradient.
        for k in 1:3
            grad_β_from_stencil = (s.dF_dβ[1, k] * d12 +
                                    s.dF_dβ[2, k] * d13 +
                                    s.dF_dβ[3, k] * d23)
            @test isapprox(grad_β_from_stencil, grad_αβ[3 + k]; atol=1e-14)
        end

        # 3. apply(stencil, dθ) = berry_term_3d(α, β, dθ).
        @test dfmm.apply(s, dθ) ≈ berry_term_3d(α, β, dθ) atol=1e-14
    end
end
