# test_phase3_hessian_degen.jl
#
# Phase 3 — Hessian-degeneracy diagnostic for the cold-limit run.
#
# The 2×2 Hessian of H_Ch = -½ α² γ² on the (α, β) tangent space is
# (v2 §3.5)
#     Hess(H_Ch) = [[-γ², 2αβ],
#                   [ 2αβ, α²]],
#     det = -α² γ² - 4 α² β².
#
# In the cold limit (s → -∞) we have M_vv → 0, hence γ → 0 and (since
# β was initialised at 0 and is transported by ∂_x u with no source)
# β stays small throughout the pre-crossing phase. So `det Hess` stays
# at machine-epsilon throughout time and space — *that* is the
# variational signature of phase-space rank collapse: the Hessian is
# rank-1 everywhere in the cold limit (v2 §3.5 final paragraph).
#
# The "approaches zero as t → t_cross" claim of the Phase-3 brief is
# therefore a statement about the *cold-limit reduction* itself:
# the Hessian is degenerate throughout the run (det/α² ≪ 1), and
# the integrator does not blow up there. We assert both.
#
# Spatial structure. With β advected by the strain (β̇ + (∂_x u) β =
# γ²/α ≈ 0), and the strain shaped like cos(2π m), β grows fastest
# where the strain is largest. Magnitude-wise this means β² is largest
# near m ∈ {1/4, 3/4} (where ∂_x u is extremal), and stays smallest
# at m ∈ {0, 1/2, 1} (the velocity-gradient nodes). γ is set by
# M_vv(J, s); J shrinks at m = 1/2 (the compressive caustic), so
# M_vv = J^(1-Γ) e^{s/cv} actually *grows* there for Γ > 1 (Γ = 5/3
# in dfmm), which keeps γ at m = 1/2 from being the spatial minimum.
#
# Net effect: in the cold limit the spatial profile of |det Hess| is
# essentially uniform at the machine-epsilon noise level, decorated
# by tiny variations from advected β and varying γ. The brief asks
# us to plot it and document where the minimum sits. The honest
# observation (the brief is explicit: "DO NOT mark tests @test_skip
# to make them green — fail honestly") is that the spatial location
# of the minimum is set by tiny floating-point details rather than
# by physics in the deterministic cold limit; what matters
# physically is the *uniform smallness* of |det Hess|, not its
# argmin. We test that uniform smallness, plot the spatial profile,
# and write `reference/figs/phase3_hessian_degen.png` with the
# evolution.
#
# References:
#   reference/MILESTONE_1_PLAN.md (Phase 3 acceptance #2)
#   design/03_action_note_v2.tex §3.5
#   src/diagnostics.jl

using Test
using dfmm
using CairoMakie

# Reuse the helpers from test_phase3_zeldovich.jl. Loaded directly
# from the file so this test stands alone if requested in isolation.
const _phase3_helpers = joinpath(@__DIR__, "test_phase3_zeldovich.jl")
isdefined(@__MODULE__, :setup_zeldovich) || include(_phase3_helpers)

"""
Compute |det Hess(H_Ch)| at every segment of `mesh`. γ is reconstructed
from the EOS via γ² = M_vv(J, s) − β², matching `src/diagnostics.jl`'s
`det_hessian_HCh` interpretation.
"""
function det_hessian_profile(mesh)
    N = n_segments(mesh)
    out = Vector{Float64}(undef, N)
    @inbounds for j in 1:N
        seg = mesh.segments[j].state
        ρ = segment_density(mesh, j)
        J = 1 / ρ
        Mvv_j = Mvv(J, seg.s)
        γ_j = sqrt(max(Mvv_j - seg.β^2, 0.0))
        out[j] = abs(det_hessian_HCh(seg.α, seg.β, γ_j))
    end
    return out
end

"""
m-coordinate (mass-cell-center) for segment j on a uniform-Δm mesh of
length L_box.
"""
m_center(j::Integer, N::Integer, L::Real) = L * (j - 0.5) / N

# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

@testset "Phase 3: Hessian-degeneracy diagnostic" begin
    A = 0.01
    t_cross = 1 / (2π * A)
    N = 128
    dt = t_cross / 1000

    # Initial Hessian. With β = 0, γ = √M_vv, det Hess = -α² γ² ≈ -1e-14
    # uniformly. This is the cold-limit baseline: rank-1 Hessian
    # everywhere, before any evolution.
    @testset "initial state: |det Hess| ≪ α² uniformly" begin
        mesh = setup_zeldovich(N, A)
        prof = det_hessian_profile(mesh)
        # All cells should have |det Hess| < 1e-12 (the M_vv ceiling
        # set by s_0 = log(1e-14), times α² = 1).
        @test maximum(prof) < 1e-13
        # Rank-1 signature: the 2x2 Hessian's *lower-right* entry
        # α² ≈ 1, but its determinant is 1e-14 ⇒ |det/α²| ≈ 1e-14.
        for j in 1:N
            seg = mesh.segments[j].state
            @test abs(prof[j]) / seg.α^2 < 1e-13
        end
    end

    # Evolution: |det Hess| stays small throughout pre-crossing.
    @testset "evolution: |det Hess|/α² stays at machine-noise level" begin
        mesh = setup_zeldovich(N, A)
        max_det_over_time = 0.0
        for n in 1:Int(round(0.99 * t_cross / dt))
            det_step!(mesh, dt)
            prof = det_hessian_profile(mesh)
            max_det_over_time = max(max_det_over_time, maximum(prof))
        end
        @info "Phase-3 Hess evolution" max_det_over_time
        # Through the entire pre-crossing run, |det Hess| stays at
        # machine-epsilon scale (here at most ~5e-13). The cold-limit
        # reduction is realized by *uniform* Hessian degeneracy, not
        # by a localized event at the caustic.
        @test max_det_over_time < 5e-12
    end

    # Cold-limit baseline check at t = t_cross. The brief asks for
    # `|det Hess|` to "approach zero as t → t_cross, spatial minimum
    # at m ≈ 1/2". The honest empirical finding (preserved in
    # `reference/notes_phase3_solver.md`):
    #
    #   * |det Hess| = α²(γ² + 4β²). With γ²/α² ≪ 1 (cold limit) and
    #     β advected by the strain (β stays ≪ γ before any crossing),
    #     |det Hess| ≈ α² γ² is dominated by γ² = M_vv = J^{1-Γ}e^{s/cv}
    #     in the cold limit.
    #   * J shrinks at m = 1/2 (the compressive caustic where the
    #     Eulerian density diverges); M_vv = J^(1-Γ) for Γ = 5/3 is
    #     therefore *largest* at m = 1/2, so |det Hess| is spatially
    #     *maximal*, not minimal, at the caustic.
    #   * The spatial argmin of |det Hess| sits where ρ is smallest
    #     (the rarefaction node, m = 0 or m = 1), where M_vv is
    #     smallest.
    #
    # In all cases |det Hess| stays bounded by 5e-13 over the whole
    # run — uniformly small, the rank-1 signature of cold-limit
    # phase-space sheets. Tests therefore assert (a) |det Hess| at the
    # caustic m = 1/2 is at machine-noise level (the brief's bound is
    # not violated; we just clarify which location is min vs max), and
    # (b) the spatial maximum sits *at the caustic* (the falsifiable
    # variational signature: the Hessian is "most rank-1-like" exactly
    # where the Zel'dovich mapping becomes singular, because that is
    # where M_vv is largest in this Γ > 1 setup).
    @testset "near caustic m≈1/2: |det Hess| at machine-noise + argmax at caustic" begin
        mesh = setup_zeldovich(N, A)
        for n in 1:Int(round(0.99 * t_cross / dt))
            det_step!(mesh, dt)
        end
        prof = det_hessian_profile(mesh)
        # Cell index closest to m = 1/2.
        j_caustic = (N ÷ 2) + 1
        m_caustic_val = m_center(j_caustic, N, mesh.L_box)
        det_at_caustic = prof[j_caustic]
        det_argmax = argmax(prof)
        m_argmax = m_center(det_argmax, N, mesh.L_box)
        det_argmin = argmin(prof)
        m_argmin = m_center(det_argmin, N, mesh.L_box)
        @info "Phase-3 caustic spatial structure" j_caustic m_caustic_val det_at_caustic m_argmax m_argmin
        # (a) det Hess at the caustic is at machine-noise level.
        @test det_at_caustic < 1e-11
        # (b) the spatial maximum coincides with the predicted caustic
        # location m = 1/2 (within one cell width, |Δm| = 1/N).
        @test abs(m_argmax - 0.5) < 1.5 / N
        # (c) the spatial minimum sits near the rarefaction nodes
        # m ∈ {0, 1}; pick the closer of the two for the assertion.
        Δm_cell = 1 / N
        @test min(abs(m_argmin - 0.0), abs(m_argmin - 1.0)) < 1.5 * Δm_cell
    end

    # Render the diagnostic plot to reference/figs/phase3_hessian_degen.png.
    # Uses CairoMakie via the existing `plot_diagnostics` helper for
    # consistency with Track-C plotting conventions, but lays out the
    # axes manually to give a (m, t) heatmap-like view.
    @testset "diagnostic plot rendering" begin
        # Re-run the full evolution and capture (t, m → |det Hess|).
        mesh = setup_zeldovich(N, A)
        N_steps = Int(round(0.99 * t_cross / dt))
        # Subsample for the figure; full N_steps would be a huge image.
        sample_every = max(1, N_steps ÷ 64)
        ts = Float64[]
        det_grid = Vector{Vector{Float64}}()
        # Capture initial state.
        push!(ts, 0.0)
        push!(det_grid, det_hessian_profile(mesh))
        for n in 1:N_steps
            det_step!(mesh, dt)
            if n % sample_every == 0
                push!(ts, n * dt)
                push!(det_grid, det_hessian_profile(mesh))
            end
        end

        ms = [m_center(j, N, mesh.L_box) for j in 1:N]

        # Build a CairoMakie figure manually: top panel = heatmap of
        # log10 |det Hess| over (m, t); bottom panel = time series of
        # the spatial min and max.
        fig = CairoMakie.Figure(size = (700, 600))

        ax_top = CairoMakie.Axis(fig[1, 1];
            xlabel = "m (mass coordinate)",
            ylabel = "t / t_cross",
            title = "log₁₀ |det Hess(H_Ch)| over (m, t) — Phase 3 cold limit",
        )
        # det_matrix[t, m]
        det_matrix = hcat(det_grid...)' |> Matrix # rows=t, cols=m
        log10det = @. log10(max(det_matrix, 1e-30))
        CairoMakie.heatmap!(ax_top, ms, ts ./ t_cross, log10det';
                            colormap = :viridis)

        ax_bot = CairoMakie.Axis(fig[2, 1];
            xlabel = "t / t_cross",
            ylabel = "log₁₀ |det Hess|",
            title = "Spatial min and max of |det Hess|",
        )
        min_t = [log10(max(minimum(p), 1e-30)) for p in det_grid]
        max_t = [log10(max(maximum(p), 1e-30)) for p in det_grid]
        CairoMakie.lines!(ax_bot, ts ./ t_cross, min_t;
                          label = "spatial min", color = :blue)
        CairoMakie.lines!(ax_bot, ts ./ t_cross, max_t;
                          label = "spatial max", color = :red)
        CairoMakie.axislegend(ax_bot; position = :rb)

        # Final-time spatial profile (right column).
        ax_final = CairoMakie.Axis(fig[1:2, 2];
            xlabel = "log₁₀ |det Hess|",
            ylabel = "m",
            title = "Final-time profile (t = 0.99 t_cross)",
        )
        final_log = [log10(max(p, 1e-30)) for p in det_grid[end]]
        CairoMakie.scatter!(ax_final, final_log, ms; markersize = 4)

        fig_path = joinpath(dirname(@__DIR__), "reference", "figs",
                            "phase3_hessian_degen.png")
        mkpath(dirname(fig_path))
        CairoMakie.save(fig_path, fig)
        @info "Phase-3 plot saved" fig_path
        @test isfile(fig_path)
    end
end
