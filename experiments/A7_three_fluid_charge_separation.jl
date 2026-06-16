using dfmm
using Printf
using CairoMakie
using LinearAlgebra
using StaticArrays
using HierarchicalGrids

# experiments/A7_three_fluid_charge_separation.jl
# 3-Fluid (H, p, e) Charge Separation in a Primordial Shock.

# ==============================================================================
# 1. PHYSICAL PARAMETERS & CONFIGURATION
# ==============================================================================

const T1_PHYS  = 50.0       # Pre-shock temperature [K]
const N1_PHYS  = 1e-3       # Pre-shock number density [cm^-3]
const X_ION    = 1e-4       # Ionization fraction (n_p / n_H)
const U_INFALL = 6.64e5     # Inflow velocity [cm/s] (Mach ~8 at 50K)

const T_END      = 0.4      # Total simulation time [code units]
const RESOLUTION = 1000     # Number of grid cells
const CFL        = 0.03      # Courant-Friedrichs-Lewy stability factor

# Physical Constants
const KB       = 1.380649e-16
const M_P_G    = 1.6726219e-24
const M_E_G    = 9.10938356e-28
const E_CHARGE = 1.60217663e-19
const CM_PC    = 3.086e18
const GAMMA    = 5.0 / 3.0

# Cross Sections [cm^2]
const SIGMA_CX      = 5.0e-15  
const SIGMA_EH      = 1.0e-15  
const SIGMA_COULOMB = 1.0e-13  

# Species Metadata
const M_H = 1.0
const M_P = 1.0
const M_E = M_E_G / M_P_G

# Derived Scaling Units
const u0   = sqrt(KB * T1_PHYS / M_P_G)
const rho0 = N1_PHYS * M_P_G
const L0   = 1.0 / (N1_PHYS * 1e-15)
const t0   = L0 / u0
const L0_PC = L0 / CM_PC

# Dimensionless Inflow
const M1 = U_INFALL / (u0 * sqrt(GAMMA))
const u1_d = U_INFALL / u0
const rho1_d = 1.0
const P1_d = 1.0 / GAMMA

# ==============================================================================
# 2. SOLVER CORE: HLL FLUX & ADVECTION
# ==============================================================================

function _primitives(U)
    r = max(U[1], 1e-15); v = U[2] / r
    pxx = max(U[3] - r*v^2, 1e-15); pp = max(U[4], 1e-15)
    return r, v, pxx, pp
end

@inline function hll_flux(UL, UR, csL, csR)
    rL, vL, pxxL, ppL = _primitives(UL)
    rR, vR, pxxR, ppR = _primitives(UR)
    
    sL = min(vL - csL, vR - csR)
    sR = max(vL + csL, vR + csR)
    
    # State: [rho, rho*u, E_long, P_perp, rho*L1, rho*alpha, rho*beta, M3]
    # E_long = rho*u^2 + pxx. (No 2*pp term!).
    uL_v = @SVector [rL, rL*vL, rL*vL^2 + pxxL, ppL, UL[5], UL[6], UL[7], UL[8]]
    uR_v = @SVector [rR, rR*vR, rR*vR^2 + pxxR, ppR, UR[5], UR[6], UR[7], UR[8]]
    
    # Flux of E_long is precisely M3.
    fL = @SVector [rL*vL, rL*vL^2 + pxxL, UL[8], vL*ppL, vL*UL[5], vL*UL[6], vL*UL[7], vL*UL[8]]
    fR = @SVector [rR*vR, rR*vR^2 + pxxR, UR[8], vR*ppR, vR*UR[5], vR*UR[6], vR*UR[7], vR*UR[8]]
    
    if sL >= 0 return fL end
    if sR <= 0 return fR end
    return (sR .* fL .- sL .* fR .+ sL .* sR .* (uR_v .- uL_v)) ./ (sR - sL)
end

function step_species!(U, dx, dt, tau)
    N = size(U, 2)
    cs = [sqrt(3.0 * _primitives(U[:,i])[3] / _primitives(U[:,i])[1]) for i in 1:N]
    F = zeros(8, N+1)
    for i in 2:N; F[:, i] = hll_flux(U[:, i-1], U[:, i], cs[i-1], cs[i]); end
    F[:, 1] = F[:, 2]; F[:, N+1] = F[:, N]
    for i in 1:N; U[:, i] .-= (dt/dx) .* (F[:, i+1] .- F[:, i]); end
    
    decay = exp(-dt/tau)
    for i in 1:N
        r, v, pxx, pp = _primitives(U[:, i])
        p_iso = (pxx + 2pp)/3.0
        
        pxx_n = p_iso + (pxx - p_iso)*decay
        pp_n = p_iso + (pp - p_iso)*decay
        
        # M3 = rho*u^3 + 3*u*pxx + Q
        q_old = U[8, i] - r*v^3 - 3*v*pxx
        q_new = q_old * decay
        
        U[3, i] = r*v^2 + pxx_n
        U[4, i] = pp_n
        U[7, i] *= decay # beta decays
        U[8, i] = r*v^3 + 3*v*pxx_n + q_new # rebuild M3
    end
end

# ==============================================================================
# 3. MULTI-FLUID COUPLING: DIFFERENTIAL DRAG
# ==============================================================================

function couple_3_fluids!(U_H, U_p, U_e, dt)
    N = size(U_H, 2)
    for i in 1:N
        rH, vH, pxxH, ppH = _primitives(U_H[:,i])
        rp, vp, pxxp, ppp = _primitives(U_p[:,i])
        re, ve, pxxe, ppe = _primitives(U_e[:,i])
        
        TH_p = (pxxH + 2ppH)/(3rH) * M_H * T1_PHYS
        Tp_p = (pxxp + 2ppp)/(3rp) * M_P * T1_PHYS
        Te_p = (pxxe + 2ppe)/(3re) * M_E * T1_PHYS
        
        vrel_Hp = sqrt(8.0*KB/π * (TH_p/M_P_G + Tp_p/M_P_G) + ((vH - vp)*u0)^2)
        vrel_He = sqrt(8.0*KB/π * (TH_p/M_P_G + Te_p/M_E_G) + ((vH - ve)*u0)^2)
        vrel_pe = sqrt(8.0*KB/π * (Tp_p/M_P_G + Te_p/M_E_G) + ((vp - ve)*u0)^2)
        
        # Rates (nu_AB = frequency at which A is dragged by B)
        nu_Hp = (rp * N1_PHYS / M_P) * SIGMA_CX * vrel_Hp * t0
        nu_pH = (rH * N1_PHYS / M_H) * SIGMA_CX * vrel_Hp * t0
        
        nu_He = (re * N1_PHYS / M_E) * SIGMA_EH * vrel_He * t0
        nu_eH = (rH * N1_PHYS / M_H) * SIGMA_EH * vrel_He * t0
        
        nu_pe = (re * N1_PHYS / M_E) * SIGMA_COULOMB * vrel_pe * t0
        nu_ep = (rp * N1_PHYS / M_P) * SIGMA_COULOMB * vrel_pe * t0
        
        # 3x3 Backward Euler for v
        M11 = 1.0 + dt*(nu_Hp + nu_He)
        M12 = -dt*nu_Hp
        M13 = -dt*nu_He
        
        M21 = -dt*nu_pH
        M22 = 1.0 + dt*(nu_pH + nu_pe)
        M23 = -dt*nu_pe
        
        M31 = -dt*nu_eH
        M32 = -dt*nu_ep
        M33 = 1.0 + dt*(nu_eH + nu_ep)
        
        M = @SMatrix [M11 M12 M13; M21 M22 M23; M31 M32 M33]
        b = @SVector [vH, vp, ve]
        v_new = M \ b
        
        # Energy conservation (frictional heating)
        dKE_H = 0.5*rH*(vH^2 - v_new[1]^2)
        dKE_p = 0.5*rp*(vp^2 - v_new[2]^2)
        dKE_e = 0.5*re*(ve^2 - v_new[3]^2)
        dE_tot = max(0.0, dKE_H + dKE_p + dKE_e)
        
        total_mass = rH + rp + re
        
        U_H[2,i] = rH * v_new[1]
        U_p[2,i] = rp * v_new[2]
        U_e[2,i] = re * v_new[3]
        
        # Add frictional heat (longitudinal part only in E_xx)
        # Assuming isotropic heating, 1/3 goes to E_xx.
        heat_H = dE_tot * (rH/total_mass)
        heat_p = dE_tot * (rp/total_mass)
        heat_e = dE_tot * (re/total_mass)
        
        U_H[3,i] += rH*(v_new[1]^2 - vH^2) + heat_H / 3.0
        U_p[3,i] += rp*(v_new[2]^2 - vp^2) + heat_p / 3.0
        U_e[3,i] += re*(v_new[3]^2 - ve^2) + heat_e / 3.0
        
        U_H[4,i] += heat_H / 3.0
        U_p[4,i] += heat_p / 3.0
        U_e[4,i] += heat_e / 3.0
    end
end

# ==============================================================================
# 4. MAIN EXPERIMENT LOOP
# ==============================================================================

function run_charge_separation()
    N = RESOLUTION
    x_min_pc, x_max_pc = -15.0, 15.0
    x_min_d = x_min_pc / L0_PC; x_max_d = x_max_pc / L0_PC
    dx = (x_max_d - x_min_d) / N
    x = range(x_min_d + dx/2, x_max_d - dx/2, length=N)
    
    r_ratio = (GAMMA+1)*M1^2 / ((GAMMA-1)*M1^2 + 2)
    rho2_d = rho1_d * r_ratio; u2_d = u1_d / r_ratio; P2_d = P1_d * (2*GAMMA*M1^2 - (GAMMA-1))/(GAMMA+1)
    
    U_H = zeros(8, N); U_p = zeros(8, N); U_e = zeros(8, N)
    for i in 1:N
        f = 0.5 * (1.0 - tanh(x[i]/1.0))
        rd = rho1_d*f + rho2_d*(1-f); ud = u1_d*f + u2_d*(1-f); pd = P1_d*f + P2_d*(1-f)
        U_H[:,i] = [rd, rd*ud, rd*ud^2 + pd, pd, rd*x[i], rd*0.02, 0, rd*ud^3 + 3ud*pd]
        U_p[:,i] = [rd*X_ION, rd*X_ION*ud, rd*X_ION*ud^2 + pd*X_ION, pd*X_ION, rd*X_ION*x[i], rd*X_ION*0.02, 0, rd*X_ION*ud^3 + 3ud*(pd*X_ION)]
        U_e[:,i] = [rd*X_ION*M_E, rd*X_ION*M_E*ud, rd*X_ION*M_E*ud^2 + pd*X_ION, pd*X_ION, rd*X_ION*M_E*x[i], rd*X_ION*M_E*0.02, 0, rd*X_ION*M_E*ud^3 + 3ud*(pd*X_ION)]
    end
    
    obs_x  = x .* L0_PC
    obs_nH = Observable(U_H[1,:] .* N1_PHYS)
    obs_np = Observable(U_p[1,:] ./ M_P .* (N1_PHYS * 1e4))
    obs_ne = Observable(U_e[1,:] ./ M_E .* (N1_PHYS * 1e4))
    obs_TH = Observable([(U_H[3,i]-U_H[2,i]^2/U_H[1,i] + 2U_H[4,i])/(3U_H[1,i]) * M_H * T1_PHYS for i in 1:N])
    obs_Te = Observable([(U_e[3,i]-U_e[2,i]^2/U_e[1,i] + 2U_e[4,i])/(3U_e[1,i]) * M_E * T1_PHYS for i in 1:N])
    obs_uH = Observable(U_H[2,:] ./ U_H[1,:] .* (u0/1e5))
    obs_ue = Observable(U_e[2,:] ./ U_e[1,:] .* (u0/1e5))
    obs_j  = Observable( (U_p[1,:]./M_P .* N1_PHYS) .* (U_p[2,:]./U_p[1,:] .- U_e[2,:]./U_e[1,:]) .* (E_CHARGE * u0) )

    fig = Figure(size=(1400, 1000))
    ax1 = Axis(fig[1,1], ylabel="n [cm^-3]", title="Density (p,e scaled x1e4)")
    ax2 = Axis(fig[1,2], ylabel="T [K]", title="Temperature", yscale=log10)
    ax3 = Axis(fig[2,1], ylabel="u [km/s]", title="Velocity")
    ax4 = Axis(fig[3,1], ylabel="J [A/cm^2]", title="Electric Current", xlabel="Position [pc]")
    
    lines!(ax1, obs_x, obs_nH, color=:black, label="H")
    lines!(ax1, obs_x, obs_np, color=:orange, label="p")
    lines!(ax1, obs_x, obs_ne, color=:cyan, linestyle=:dash, label="e")
    axislegend(ax1, position=:rb)
    
    lines!(ax2, obs_x, obs_TH, color=:black)
    lines!(ax2, obs_x, obs_Te, color=:cyan, linestyle=:dash)
    ylims!(ax2, 10.0, 2000.0)
    
    lines!(ax3, obs_x, obs_uH, color=:black)
    lines!(ax3, obs_x, obs_ue, color=:cyan, linestyle=:dash)
    
    lines!(ax4, obs_x, obs_j, color=:red)

    t = 0.0; steps = 0
    
    println("Recording Physical 3-Fluid Shock...")
    record(fig, "reference/figs/three_fluid_physical.mp4", 1:120; framerate=20) do frame
        target_t = frame * (T_END / 120.0)
        while t < target_t
            # Correct Dynamic CFL
            smax = 0.0
            for i in 1:N
                r, v, pxx, _ = _primitives(U_H[:,i]); smax = max(smax, abs(v) + sqrt(3.0*max(pxx, 1e-15)/max(r, 1e-15)))
                r, v, pxx, _ = _primitives(U_p[:,i]); smax = max(smax, abs(v) + sqrt(3.0*max(pxx, 1e-15)/max(r, 1e-15)))
                r, v, pxx, _ = _primitives(U_e[:,i]); smax = max(smax, abs(v) + sqrt(3.0*max(pxx, 1e-15)/max(r, 1e-15)))
            end
            dt = CFL * dx / smax
            
            step_species!(U_H, dx, dt, 1e-1); step_species!(U_p, dx, dt, 1e-1); step_species!(U_e, dx, dt, 1e-2)
            
            # BCs
            for c in 1:2
                U_H[1:4, c] = [rho1_d, rho1_d*u1_d, P1_d + rho1_d*u1_d^2, P1_d]
                U_H[5, c] = rho1_d*(x[c] - u1_d*(t+dt)); U_H[6:7, c] = [rho1_d*0.02, 0]; U_H[8, c] = rho1_d*u1_d^3 + 3u1_d*P1_d
                
                U_p[1:4, c] = [rho1_d*X_ION, rho1_d*X_ION*u1_d, (P1_d+rho1_d*u1_d^2)*X_ION, P1_d*X_ION]
                U_p[5, c] = U_p[1,c]*(x[c] - u1_d*(t+dt)); U_p[6:7, c] = [U_p[1,c]*0.02, 0]; U_p[8, c] = U_p[1,c]*u1_d^3 + 3u1_d*(P1_d*X_ION)
                
                ve_ext = U_e[2,3]/U_e[1,3]
                U_e[1:4, c] = [rho1_d*X_ION*M_E, rho1_d*X_ION*M_E*ve_ext, rho1_d*X_ION*M_E*ve_ext^2 + P1_d*X_ION, P1_d*X_ION]
                U_e[5, c] = U_e[1,c]*(x[c] - ve_ext*(t+dt)); U_e[6:7, c] = [U_e[1,c]*0.02, 0]; U_e[8, c] = U_e[1,c]*ve_ext^3 + 3ve_ext*(P1_d*X_ION)
            end
            U_H[:, N] = U_H[:, N-1]; U_p[:, N] = U_p[:, N-1]; U_e[:, N] = U_e[:, N-1]
            
            couple_3_fluids!(U_H, U_p, U_e, dt)
            t += dt; steps += 1
        end
        
        obs_nH[] = U_H[1,:] .* N1_PHYS
        obs_np[] = U_p[1,:] ./ M_P .* (N1_PHYS * 1e4)
        obs_ne[] = U_e[1,:] ./ M_E .* (N1_PHYS * 1e4)
        obs_uH[] = U_H[2,:] ./ U_H[1,:] .* (u0/1e5)
        obs_ue[] = U_e[2,:] ./ U_e[1,:] .* (u0/1e5)
        obs_TH[] = [max((U_H[3,i]-U_H[2,i]^2/U_H[1,i] + 2U_H[4,i])/(3U_H[1,i]) * T1_PHYS, 1.0) for i in 1:N]
        obs_Te[] = [max((U_e[3,i]-U_e[2,i]^2/U_e[1,i] + 2U_e[4,i])/(3U_e[1,i]) * M_E * T1_PHYS, 1.0) for i in 1:N]
        obs_j[]  = (U_p[1,:]./M_P .* N1_PHYS) .* (U_p[2,:]./U_p[1,:] .- U_e[2,:]./U_e[1,:]) .* (E_CHARGE * u0)
        autolimits!(ax3); autolimits!(ax4)
    end
    save("reference/figs/three_fluid_physical_final.png", fig)
    println("Saved PNG and Movie. Completed in $steps steps.")
end

run_charge_separation()
