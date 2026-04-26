# Shared helpers for the M3-1 Phase-5/5b Sod tests on the HG path.

function build_sod_mesh_HG(ic; mirror::Bool = true)
    N0 = length(ic.rho)
    Γ = 5.0 / 3.0
    if mirror
        rho = vcat(ic.rho, reverse(ic.rho))
        u   = vcat(ic.u,   -reverse(ic.u))
        P   = vcat(ic.P,   reverse(ic.P))
        Pp  = vcat(ic.Pp,  reverse(ic.Pp))
        αs  = vcat(ic.alpha_init, reverse(ic.alpha_init))
        βs  = vcat(ic.beta_init,  -reverse(ic.beta_init))
        N = 2 * N0
        dx = 2.0 / N
        L_box = 2.0
    else
        rho = copy(ic.rho); u = copy(ic.u); P = copy(ic.P); Pp = copy(ic.Pp)
        αs = copy(ic.alpha_init); βs = copy(ic.beta_init)
        N = N0
        dx = 1.0 / N
        L_box = 1.0
    end
    Δm = rho .* dx
    ss = log.(P ./ rho .^ Γ)
    positions = collect((0:N-1) .* dx)
    velocities = u
    return DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                  Δm = Δm, Pps = Pp, L_box = L_box,
                                  bc = :periodic)
end

function extract_eulerian_profiles_HG(mesh_HG::DetMeshHG)
    dfmm.sync_cache_from_HG!(mesh_HG)
    return extract_eulerian_profiles(mesh_HG.cache_mesh)
end
