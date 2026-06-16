using dfmm
using Printf
using CairoMakie
using LinearAlgebra
using StaticArrays
using HierarchicalGrids

# This script runs the low-mach shock and plots the final state in physical units, 
# using a direct port of the explicit Python HLL solver.

# --- Physical Constants ---
const KB = 1.380649e-16  # erg/K
const M_P_G = 1.6726219e-24 # g
const M_E_G = 9.10938356e-28 # g
const E_CHARGE = 1.60217663e-19 # Coulombs
const CM_PER_PC = 3.086e18

# --- Simulation Parameters (Dimensionless) ---
const M_H = 1.0; const M_P = 1.0; const M_E = M_E_G/M_P_G
const SIGMA_HH = 1.0e-15 # cm^2, physical cross section
const SIGMA_CX = 5.0e-15
const SIGMA_EH = 1.0e-15
const SIGMA_COULOMB = 1.0e-13
const GAMMA_EOS = 5.0/3.0; const CSCOEF = 3.0

function _species_primitives(U)
    rho=U[1]; u=U[2]/max(rho,1e-30); Pxx=U[3]-rho*u*u; Pp=U[4]
    return rho, u, Pxx, Pp
end

function _species_step_explicit(U, dx, dt, tau_self, N, L0, u0, n0)
    rho=zeros(N); u=zeros(N); Pxx=zeros(N); Pp=zeros(N)
    L1=zeros(N); alpha=zeros(N); beta=zeros(N); Q=zeros(N); cs=zeros(N)
    for i in 1:N
        r=U[1,i]; v=U[2,i]/max(r,1e-30); pxx=U[3,i]-r*v*v; pp=U[4,i]
        rho[i]=r; u[i]=v; Pxx[i]=pxx; Pp[i]=pp; L1[i]=U[5,i]; alpha[i]=U[6,i]; beta[i]=U[7,i]; Q[i]=U[8,i]
        cs[i]=sqrt(CSCOEF*max(pxx,1e-30)/max(r,1e-30))
    end
    fluxes=zeros(8,N+1)
    for i in 1:N
        # Left state for the interface i+1/2 is cell i
        rL=rho[i]; uL=u[i]; pxxL=Pxx[i]; ppL=Pp[i]; lL=L1[i]; aL=alpha[i]; bL=beta[i]; qL=Q[i]; csL=cs[i]
        # Right state for the interface i+1/2 is cell i+1 (or cell N for the last interface)
        rR= i<N ? rho[i+1] : rho[i]; uR= i<N ? u[i+1] : u[i]; pxxR= i<N ? Pxx[i+1] : Pxx[i]; ppR= i<N ? Pp[i+1] : Pp[i]
        lR= i<N ? L1[i+1] : L1[i]; aR= i<N ? alpha[i+1] : alpha[i]; bR= i<N ? beta[i+1] : beta[i]; qR= i<N ? Q[i+1] : Q[i]; csR= i<N ? cs[i+1] : cs[i]
        
        sL=min(uL-csL,uR-csR); sR=max(uL+csL,uR+csR)
        if sL>=0.0; f1=rL*uL;f2=rL*uL^2+pxxL;f3=uL*(rL*uL^2+3*pxxL+2*ppL);f4=uL*ppL;f5=uL*lL;f6=uL*aL;f7=uL*bL;f8=uL*qL
        elseif sR<=0.0; f1=rR*uR;f2=rR*uR^2+pxxR;f3=uR*(rR*uR^2+3*pxxR+2*ppR);f4=uR*ppR;f5=rR*lR;f6=rR*aR;f7=rR*bR;f8=rR*qR
        else
            f1L=rL*uL;f2L=rL*uL^2+pxxL;f3L=uL*(rL*uL^2+3*pxxL+2*ppL);f4L=uL*ppL;f5L=uL*lL;f6L=uL*aL;f7L=uL*bL;f8L=uL*qL
            f1R=rR*uR;f2R=rR*uR^2+pxxR;f3R=uR*(rR*uR^2+3*pxxR+2*ppR);f4R=uR*ppR;f5R=rR*lR;f6R=rR*aR;f7R=rR*bR;f8R=uR*qR
            u1L=rL;u2L=rL*uL;u3L=rL*uL^2+pxxL+2*ppL;u4L=ppL;u5L=lL;u6L=aL;u7L=bL;u8L=qL
            u1R=rR;u2R=rR*uR;u3R=rR*uR^2+pxxR+2*ppR;u4R=ppR;u5R=lR;u6R=aR;u7R=bR;u8R=qR
            ds=sR-sL
            f1=(sR*f1L-sL*f1R+sL*sR*(u1R-u1L))/ds;f2=(sR*f2L-sL*f2R+sL*sR*(u2R-u2L))/ds
            f3=(sR*f3L-sL*f3R+sL*sR*(u3R-u3L))/ds;f4=(sR*f4L-sL*f4R+sL*sR*(u4R-u4L))/ds
            f5=(sR*f5L-sL*f5R+sL*sR*(u5R-u5L))/ds;f6=(sR*f6L-sL*f6R+sL*sR*(u6R-u6L))/ds
            f7=(sR*f7L-sL*f7R+sL*sR*(u7R-u7L))/ds;f8=(sR*f8L-sL*f8R+sL*sR*(u8R-u8L))/ds
        end; fluxes[:,i]=[f1,f2,f3,f4,f5,f6,f7,f8] # Flux at cell i+1/2 edge
    end
    # HLL boundary conditions: Copy flux from interior to ghost cells
    fluxes[:,1]=fluxes[:,2]
    for k in 1:8
        fluxes[k,N+1] = fluxes[k,N]
    end

    # Update U
    for i in 1:N; for k in 1:8; U[k,i]-=(dt/dx)*(fluxes[k,i+1]-fluxes[k,i]); end; end
    for i in 1:N
        r=U[1,i]; v=U[2,i]/max(r,1e-30); pxx=U[3,i]-r*v*v; pp=U[4,i]
        p_iso=(pxx+2*pp)/3.0; relax=exp(-dt/tau_self)
        pxx_new=p_iso+(pxx-p_iso)*relax; pp_new=p_iso+(pp-p_iso)*relax
        U[3,i]=pxx_new+r*v*v; U[4,i]=pp_new; U[7,i]*=relax; U[8,i]*=relax
    end
end

function couple_3_fluids(U_H,U_p,U_e,dt,N, u0, T0, n0, rho0)
    for i in 1:N
        rH_d,vH_d,pxxH_d,ppH_d=_species_primitives(U_H[:,i])
        rp_d,vp_d,pxxp_d,ppp_d=_species_primitives(U_p[:,i])
        re_d,ve_d,pxxe_d,ppe_d=_species_primitives(U_e[:,i])
        
        TH=(pxxH_d+2.0*ppH_d)/(3.0*rH_d)*M_H*T0
        Tp=(pxxp_d+2.0*ppp_d)/(3.0*rp_d)*M_P*T0
        Te=(pxxe_d+2.0*ppe_d)/(3.0*re_d)*M_E*T0
        
        if TH<0||Tp<0||Te<0; continue; end
        
        vrel_Hp=sqrt(8.0*KB/π*(TH/M_P_G+Tp/M_P_G)+((vH_d-vp_d)*u0)^2)
        vrel_He=sqrt(8.0*KB/π*(TH/M_P_G+Te/M_E_G)+((vH_d-ve_d)*u0)^2)
        vrel_pe=sqrt(8.0*KB/π*(Tp/M_P_G+Te/M_E_G)+((vp_d-ve_d)*u0)^2)
        
        D_Hp = (rH_d*n0) * (rp_d*n0) * SIGMA_CX * vrel_Hp
        D_He = (rH_d*n0) * (re_d*n0) * SIGMA_EH * vrel_He
        D_pe = (rp_d*n0) * (re_d*n0) * SIGMA_COULOMB * vrel_pe
        
        M=@SMatrix[(rH_d*n0*M_P_G+dt*(D_Hp+D_He)) (-dt*D_Hp) (-dt*D_He); (-dt*D_Hp) (rp_d*n0*M_P_G+dt*(D_Hp+D_pe)) (-dt*D_pe); (-dt*D_He) (-dt*D_pe) (re_d*n0*M_E_G+dt*(D_He+D_pe))]
        b_vec=SVector(rH_d*n0*M_P_G*vH_d*u0, rp_d*n0*M_P_G*vp_d*u0, re_d*n0*M_E_G*ve_d*u0)
        v_new_physical = M \ b_vec
        
        vH_new=v_new_physical[1]/u0; vp_new=v_new_physical[2]/u0; ve_new=v_new_physical[3]/u0
        
        E_kin_old=0.5*(rH_d*n0*M_P_G)*(vH_d*u0)^2 + 0.5*(rp_d*n0*M_P_G)*(vp_d*u0)^2 + 0.5*(re_d*n0*M_E_G)*(ve_d*u0)^2
        E_kin_new=0.5*(rH_d*n0*M_P_G)*v_new_physical[1]^2 + 0.5*(rp_d*n0*M_P_G)*v_new_physical[2]^2 + 0.5*(re_d*n0*M_E_G)*v_new_physical[3]^2
        dE=max(0.0,E_kin_old-E_kin_new)
        total_mass=rH_d*n0*M_P_G+rp_d*n0*M_P_G+re_d*n0*M_E_G
        
        U_H[2,i]=U_H[1,i]*vH_new; U_p[2,i]=U_p[1,i]*vp_new; U_e[2,i]=U_e[1,i]*ve_new
        
        U_H[3,i]+=U_H[1,i]*(vH_new^2-vH_d^2) + (dE*(rH_d*n0*M_P_G/total_mass))/(rho0*u0^2)
        U_p[3,i]+=U_p[1,i]*(vp_new^2-vp_d^2) + (dE*(rp_d*n0*M_P_G/total_mass))/(rho0*u0^2)
        U_e[3,i]+=U_e[1,i]*(ve_new^2-ve_d^2) + (dE*(re_d*n0*M_E_G/total_mass))/(rho0*u0^2)
        
        relax_H=dt*(D_Hp+D_He)/(rH_d*n0*M_P_G); relax_p=dt*(D_Hp+D_pe)/(rp_d*n0*M_P_G); relax_e=dt*(D_He+D_pe)/(re_d*n0*M_E_G)
        
        U_H[7,i]*=exp(-relax_H); U_H[8,i]*=exp(-relax_H); U_p[7,i]*=exp(-relax_p); U_p[8,i]*=exp(-relax_p); U_e[7,i]*=exp(-relax_e); U_e[8,i]*=exp(-relax_e)
    end
end

function rankine_hugoniot(rho1,u1,P1,gamma_eos)
    c_s1=sqrt(gamma_eos*P1/rho1); M1=u1/c_s1
    rho_ratio=(gamma_eos+1)*M1^2/((gamma_eos-1)*M1^2+2)
    P_ratio=(2*gamma_eos*M1^2-(gamma_eos-1))/(gamma_eos+1)
    return rho1*rho_ratio, u1/rho_ratio, P1*P_ratio
end

function main()
    T1_phys=50.0; n1_phys=1e-3; u1_phys=1e5
    u0=sqrt(KB*T1_phys/M_P_G); rho0=n1_phys*M_P_G; L0=1/(n1_phys*SIGMA_HH); P0=rho0*u0^2; J0=E_CHARGE*n1_phys*u0/10.0
    M1=u1_phys/sqrt(GAMMA_EOS*KB*T1_phys/M_P_G)
    rho1_d=1.0; P1_d=(KB*T1_phys*n1_phys)/P0
    u1_d=u1_phys/u0
    rho2_d,u2_d,P2_d=rankine_hugoniot(rho1_d,u1_d,P1_d,GAMMA_EOS)
    
    N=1200; x_min_d,x_max_d=-20.0,20.0; dx_d=(x_max_d-x_min_d)/N
    x_coords_d=range(x_min_d+dx_d/2,x_max_d-dx_d/2,length=N)
    
    U_H=zeros(8,N); U_p=zeros(8,N); U_e=zeros(8,N)
    w=4.0
    for i in 1:N
        f=0.5*(1.0-tanh((x_coords_d[i]-0.0)/w))
        rH=rho1_d*f+rho2_d*(1-f); uH=u1_d*f+u2_d*(1-f); pH=P1_d*f+P2_d*(1-f)
        U_H[1,i]=rH;U_H[2,i]=rH*uH;U_H[3,i]=pH+rH*uH^2;U_H[4,i]=pH;U_H[5,i]=rH*x_coords_d[i];U_H[6,i]=rH*0.02
        U_p[1,i]=rH*1e-4;U_p[2,i]=U_p[1,i]*uH;U_p[3,i]=pH*1e-4+U_p[1,i]*uH^2;U_p[4,i]=pH*1e-4
        U_p[5,i]=U_p[1,i]*x_coords_d[i];U_p[6,i]=U_p[1,i]*0.02
        U_e[1,i]=rH*1e-4*M_E;U_e[2,i]=U_e[1,i]*uH;U_e[3,i]=pH*1e-4+U_e[1,i]*uH^2;U_e[4,i]=pH*1e-4
        U_e[5,i]=U_e[1,i]*x_coords_d[i];U_e[6,i]=U_e[1,i]*0.02
    end
    
    tau_H=1e-1; tau_p=1e-1; tau_e=1e-2
    t=0.0; t_end=3.0; cfl=0.3; nsteps=0
    println("Running Low-Mach Julia Simulation...")
    while t<t_end
        smax=0.0
        for i in 1:N;_,v,pxx,_=_species_primitives(U_H[:,i]);smax=max(smax,abs(v)+sqrt(3*max(pxx,1e-15)));end
        for i in 1:N;_,v,pxx,_=_species_primitives(U_p[:,i]);smax=max(smax,abs(v)+sqrt(3*max(pxx,1e-15)));end
        for i in 1:N;_,v,pxx,_=_species_primitives(U_e[:,i]);smax=max(smax,abs(v)+sqrt(3*max(pxx,1e-15)));end
        dt=min(cfl*dx_d/smax,t_end-t); if dt<=0; break; end
        
        _species_step_explicit(U_H,dx_d,dt,tau_H,N,L0,u0,n1_phys); _species_step_explicit(U_p,dx_d,dt,tau_p,N,L0,u0,n1_phys); _species_step_explicit(U_e,dx_d,dt,tau_e,N,L0,u0,n1_phys)
        couple_3_fluids(U_H,U_p,U_e,dt,N,u0,T1_phys,n1_phys,rho0)

        for c in 1:2
            U_H[1:4,c]=[rho1_d,rho1_d*u1_d,P1_d+rho1_d*u1_d^2,P1_d]
            U_p[1:4,c]=[rho1_d*1e-4,rho1_d*1e-4*u1_d,P1_d*1e-4+rho1_d*1e-4*u1_d^2,P1_d*1e-4]
            U_e[1:4,c]=[rho1_d*1e-4*M_E,rho1_d*1e-4*M_E*u1_d,P1_d*1e-4+rho1_d*1e-4*M_E*u1_d^2,P1_d*1e-4]
        end
        U_H[:,N]=U_H[:,N-1]; U_p[:,N]=U_p[:,N-1]; U_e[:,N]=U_e[:,N-1]
        t+=dt; nsteps+=1
    end
    println("Finished in $nsteps steps.")
    
    x_pc = x_coords_d .* (L0 / CM_PER_PC)
    u_H_kms=(U_H[2,:]./U_H[1,:]).*(u0/1e5); u_p_kms=(U_p[2,:]./U_p[1,:]).*(u0/1e5); u_e_kms=(U_e[2,:]./U_e[1,:]).*(u0/1e5)
    r_H_phys=U_H[1,:].*n1_phys; r_p_phys=U_p[1,:].*n1_phys; r_e_phys=U_e[1,:].*n1_phys
    T_H=(((U_H[3,:].-U_H[1,:].*u_H_kms.^2)./U_H[1,:]) .* P0)./(KB*r_H_phys)
    T_p=(((U_p[3,:].-U_p[1,:].*u_p_kms.^2)./U_p[1,:]) .* P0)./(KB*r_p_phys)
    T_e=(((U_e[3,:].-U_e[1,:].*u_e_kms.^2)./U_e[1,:]) .* P0)./(KB*r_e_phys)
    P_H=r_H_phys.*KB.*T_H; P_p=r_p_phys.*KB.*T_p; P_e=r_e_phys.*KB.*T_e
    current_j=r_p_phys.*(u_p_kms.-u_e_kms).*(E_CHARGE/10.0).*1e5
    
    fig=Figure(size=(1600,1200))
    ax1=Axis(fig[1,1],ylabel="Num Density [cm^-3]"); ax2=Axis(fig[1,2],ylabel="Temperature [K]")
    ax3=Axis(fig[2,1],ylabel="Pressure [erg/cm^3]"); ax4=Axis(fig[2,2],ylabel="Velocity [km/s]")
    ax5=Axis(fig[3,1],ylabel="Current [A/cm^2]",xlabel="Position [pc]")
    
    lines!(ax1,x_pc,r_H_phys,label="H"); lines!(ax1,x_pc,r_p_phys.*1e4,label="p (x1e4)"); lines!(ax1,x_pc,r_e_phys.*(1/M_E*1e4),label="e (x1e4)",linestyle=:dash); axislegend(ax1,position=:rb)
    lines!(ax2,x_pc,T_H,label="H"); lines!(ax2,x_pc,T_p,label="p"); lines!(ax2,x_pc,T_e,label="e",linestyle=:dash); axislegend(ax2,position=:rt)
    lines!(ax3,x_pc,P_H,label="H"); lines!(ax3,x_pc,P_p,label="p"); lines!(ax3,x_pc,P_e,label="e",linestyle=:dash); axislegend(ax3,position=:rt)
    lines!(ax4,x_pc,u_H_kms,label="H"); lines!(ax4,x_pc,u_p_kms,label="p"); lines!(ax4,x_pc,u_e_kms,label="e",linestyle=:dash); axislegend(ax4,position=:rt)
    lines!(ax5,x_pc,current_j,label="Current J",color=:red); axislegend(ax5,position=:rt); ylims!(ax5,nothing,6e-22)
    
    save("reference/figs/low_mach_shock_physical.png",fig)
    println("Plot saved.")
end

if abspath(PROGRAM_FILE) == @__FILE__; main(); end
