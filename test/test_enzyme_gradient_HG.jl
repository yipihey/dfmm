# test_enzyme_gradient_HG.jl
#
# Validates the HG Phase-1 reverse-mode gradient driver
# (`cholesky_run_HG_gradient`) against central finite differences on a
# small HG simplicial mesh. Mirrors the Phase-1 Cholesky validation
# but exercises the HG field-set storage layer.

using dfmm
using HierarchicalGrids: SimplicialMesh
using StaticArrays: SVector
using Test

@testset "Enzyme HG Phase-1 gradient (cholesky_run_HG_gradient)" begin

    # Build a 4-cell HG mesh + Cholesky field set with non-trivial ICs.
    function build_mesh_fields(N::Int)
        mesh = uniform_simplicial_mesh_1D(N, 1.0)
        fields = allocate_chfield_HG(N; T = Float64)
        # Per-cell ICs spanning a smooth profile.
        for j in 1:N
            α0 = 1.0 + 0.05 * sin(2π * (j - 0.5) / N)
            β0 = 0.03 * cos(2π * (j - 0.5) / N)
            write_alphabeta!(fields, j, SVector{2,Float64}(α0, β0))
        end
        return mesh, fields
    end

    # Loss: mean squared deviation from a target profile, applied
    # component-wise across cells.
    function make_loss(N)
        α_target = [0.95 + 0.02 * (j - 1) for j in 1:N]
        β_target = [-0.01 * j for j in 1:N]
        return function (αs, βs)
            acc = 0.0
            for j in eachindex(αs)
                acc += (αs[j] - α_target[j])^2 + (βs[j] - β_target[j])^2
            end
            return 0.5 * acc / length(αs)
        end
    end

    @testset "Single step, N_cells = 4: gradient vs central FD" begin
        N = 4
        mesh, fields = build_mesh_fields(N)
        M_vv, divu, dt, N_steps = 1.0, -0.1, 0.02, 1
        loss = make_loss(N)

        # Snapshot ICs for FD.
        α_init = Float64[fields.alpha[j][1] for j in 1:N]
        β_init = Float64[fields.beta[j][1]  for j in 1:N]

        L, ᾱ_0, β̄_0, M̄, d̄, t̄ =
            cholesky_run_HG_gradient(loss, mesh, fields, M_vv, divu, dt, N_steps)

        function run_and_loss(α0, β0, Mvv_, d_, dt_)
            mesh2, fields2 = build_mesh_fields(N)
            for j in 1:N
                write_alphabeta!(fields2, j, SVector{2,Float64}(α0[j], β0[j]))
            end
            cholesky_run_HG!(mesh2, fields2, Mvv_, d_, dt_, N_steps)
            αs = Float64[fields2.alpha[j][1] for j in 1:N]
            βs = Float64[fields2.beta[j][1]  for j in 1:N]
            return loss(αs, βs)
        end

        # Reference loss matches.
        @test L ≈ run_and_loss(α_init, β_init, M_vv, divu, dt) rtol = 1e-14

        # FD on every per-cell α and β.
        h = 1e-6
        for j in 1:N
            α_pl = copy(α_init); α_pl[j] += h
            α_mn = copy(α_init); α_mn[j] -= h
            β_pl = copy(β_init); β_pl[j] += h
            β_mn = copy(β_init); β_mn[j] -= h
            dα_fd = (run_and_loss(α_pl, β_init, M_vv, divu, dt) -
                     run_and_loss(α_mn, β_init, M_vv, divu, dt)) / (2h)
            dβ_fd = (run_and_loss(α_init, β_pl, M_vv, divu, dt) -
                     run_and_loss(α_init, β_mn, M_vv, divu, dt)) / (2h)
            @test ᾱ_0[j] ≈ dα_fd atol = 1e-7
            @test β̄_0[j] ≈ dβ_fd atol = 1e-7
        end
        # FD on scalars.
        M̄_fd = (run_and_loss(α_init, β_init, M_vv + h, divu, dt) -
                run_and_loss(α_init, β_init, M_vv - h, divu, dt)) / (2h)
        d̄_fd = (run_and_loss(α_init, β_init, M_vv, divu + h, dt) -
                run_and_loss(α_init, β_init, M_vv, divu - h, dt)) / (2h)
        t̄_fd = (run_and_loss(α_init, β_init, M_vv, divu, dt + h) -
                run_and_loss(α_init, β_init, M_vv, divu, dt - h)) / (2h)
        @test M̄ ≈ M̄_fd atol = 1e-7
        @test d̄ ≈ d̄_fd atol = 1e-7
        @test t̄ ≈ t̄_fd atol = 1e-7
    end

    @testset "Multi-step (N_cells = 4, N_steps = 5) vs central FD" begin
        N, N_steps = 4, 5
        mesh, fields = build_mesh_fields(N)
        M_vv, divu, dt = 1.05, -0.2, 0.01
        loss = make_loss(N)

        α_init = Float64[fields.alpha[j][1] for j in 1:N]
        β_init = Float64[fields.beta[j][1]  for j in 1:N]

        L, ᾱ_0, β̄_0, M̄, d̄, t̄ =
            cholesky_run_HG_gradient(loss, mesh, fields, M_vv, divu, dt, N_steps)

        function run_and_loss(α0, β0, Mvv_, d_, dt_)
            mesh2, fields2 = build_mesh_fields(N)
            for j in 1:N
                write_alphabeta!(fields2, j, SVector{2,Float64}(α0[j], β0[j]))
            end
            cholesky_run_HG!(mesh2, fields2, Mvv_, d_, dt_, N_steps)
            αs = Float64[fields2.alpha[j][1] for j in 1:N]
            βs = Float64[fields2.beta[j][1]  for j in 1:N]
            return loss(αs, βs)
        end

        h = 1e-6
        for j in 1:N
            α_pl = copy(α_init); α_pl[j] += h
            α_mn = copy(α_init); α_mn[j] -= h
            β_pl = copy(β_init); β_pl[j] += h
            β_mn = copy(β_init); β_mn[j] -= h
            dα_fd = (run_and_loss(α_pl, β_init, M_vv, divu, dt) -
                     run_and_loss(α_mn, β_init, M_vv, divu, dt)) / (2h)
            dβ_fd = (run_and_loss(α_init, β_pl, M_vv, divu, dt) -
                     run_and_loss(α_init, β_mn, M_vv, divu, dt)) / (2h)
            @test ᾱ_0[j] ≈ dα_fd atol = 1e-7
            @test β̄_0[j] ≈ dβ_fd atol = 1e-7
        end
        M̄_fd = (run_and_loss(α_init, β_init, M_vv + h, divu, dt) -
                run_and_loss(α_init, β_init, M_vv - h, divu, dt)) / (2h)
        d̄_fd = (run_and_loss(α_init, β_init, M_vv, divu + h, dt) -
                run_and_loss(α_init, β_init, M_vv, divu - h, dt)) / (2h)
        t̄_fd = (run_and_loss(α_init, β_init, M_vv, divu, dt + h) -
                run_and_loss(α_init, β_init, M_vv, divu, dt - h)) / (2h)
        @test M̄ ≈ M̄_fd atol = 1e-7
        @test d̄ ≈ d̄_fd atol = 1e-7
        @test t̄ ≈ t̄_fd atol = 1e-7
    end
end
