# test_M3_8b_metal_kernel.jl
#
# M3-8b Phase b: Metal GPU kernel exploration test stub.
#
# **Status**: deferred to M3-8c. The full Metal kernel port is blocked
# on HG-side `Backend`-parameterized `PolynomialFieldSet` storage,
# which has NOT landed upstream as of dfmm `dc2fd56`.
#
# Per the M3-8a audit (`reference/notes_M3_8a_gpu_readiness_audit.md`)
# and the M3-8b brief's "Honest scoping" instruction: when HG-side
# `Backend` is not ready, ship the matrix-free Newton-Krylov port (the
# algorithm-side prerequisite) and document the hard dependency. This
# test fixture is the placeholder for the per-leaf residual kernel; it
# uses `@test_skip` until M3-8c lands the GPU storage substrate.
#
# What this fixture WILL do once M3-8c is unblocked:
#
#   1. Detect Metal availability via `Metal.functional()`.
#   2. Allocate per-leaf 11-dof unknown vector as `MtlArray{Float32}`
#      (or `Float64` if the M2 Max kernel supports it).
#   3. Launch a `@kernel` that evaluates `cholesky_el_residual_2D_berry`-
#      shaped per-cell math over a 4×4 (level-2) mesh.
#   4. Compare the GPU residual vector to the CPU one; require bit-
#      equal agreement modulo Float32 round-off (≤ 1e-6 abs in
#      mixed-precision; exactly 0.0 in matched precision).
#
# References:
#   • `reference/notes_M3_8a_gpu_readiness_audit.md` — Metal probe + plan.
#   • `reference/notes_M3_8b_metal_gpu_port.md` — phase status.
#   • `~/.julia/dev/HierarchicalGrids/` — verify Backend support
#     before lifting the @test_skip's.

using Test

@testset "M3-8b: Metal kernel exploration (deferred to M3-8c)" begin
    @testset "M3-8b: Metal.jl availability check" begin
        # This testset checks whether `Metal.jl` is loadable on the
        # current host. If yes, we can exercise a small smoke kernel.
        # If no (CI / non-Apple-Silicon hosts), we skip with a
        # documented reason.
        metal_available = false
        try
            # Metal is NOT a hard dep of dfmm (per M3-8a Project.toml
            # `[weakdeps]` plan). On hosts where it's installed in the
            # global env it can still be `using`-ed for testing.
            @eval Main using Metal
            if @isdefined Metal
                # Even if loaded, Metal.functional() may be false on
                # non-Apple-Silicon (e.g. x86_64 macOS, Linux).
                metal_available = Metal.functional()
            end
        catch _
            metal_available = false
        end

        if metal_available
            @info "Metal is functional; running smoke kernel."
            # Smoke test: elementwise add on 1024 Float32. This
            # mirrors the M3-8a Metal probe headline number (warm
            # 0.83 ms / 1.76 ms per op).
            @eval Main begin
                a = MtlArray(Float32.(collect(1:1024)))
                b = MtlArray(Float32.(collect(1024:-1:1)))
                c = a .+ b
                c_cpu = Array(c)
                @test all(c_cpu .== Float32(1025))
            end
        else
            # Metal not available (most CI configs, all Linux/x86 hosts).
            # `@test_skip` produces a Pass annotation in the test summary,
            # making the deferred status visible without failing the run.
            @test_skip metal_available
        end
    end

    @testset "M3-8b: per-leaf residual kernel (deferred to M3-8c)" begin
        # Placeholder for the future kernel-equality test:
        #
        #   1. Build a level-2 (4×4 = 16 cells) 2D mesh + 11-dof IC.
        #   2. Pack to MtlArray{Float64}.
        #   3. Launch `@kernel function cholesky_residual_2d_berry_kernel!(F, ...)`
        #      and synchronize.
        #   4. Compare to the CPU `cholesky_el_residual_2D_berry!` output.
        #
        # Blocker: HG-side `Backend`-parameterized `PolynomialFieldSet`
        # storage does NOT exist as of HG `81914c2`. The face-neighbor
        # tables, Δm_per_axis arrays, and `aux::NamedTuple` are all
        # CPU-side `Vector{T}`. A KA `@kernel` would need to read
        # these from device memory. The clean lift is M3-8c after
        # HG ships `PolynomialFieldSet{<:KA.Backend}`.
        #
        # Until then: explicitly @test_skip with a reason.
        hg_backend_available = false  # UPDATE when HG ships Backend
        if hg_backend_available
            error("M3-8c: implement the per-leaf residual kernel here.")
        else
            @test_skip hg_backend_available
        end
    end
end  # @testset "M3-8b: Metal kernel exploration (deferred to M3-8c)"
