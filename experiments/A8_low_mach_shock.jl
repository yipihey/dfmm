using dfmm
using Printf
using CairoMakie
using LinearAlgebra
using StaticArrays
using HierarchicalGrids

# This script is a copy of A7, modified to run a low-Mach shock as requested.

const M_H = 1.0; const M_P = 1.0; const M_E = 1.0/1836.0
const SIGMA_HH = 100.0; const SIGMA_CX = 500.0; const SIGMA_EH = 100.0
const SIGMA_COULOMB = 1000.0; const GAMMA_EOS = 5.0/3.0; const CSCOEF = 3.0

function _species_primitives(U)
    rho=U[1]; u=U[2]/max(rho,1e-30); Pxx=U[3]-rho*u*u; Pp=U[4]
    return rho, u, Pxx, Pp
end

function _species_step_explicit(U, dx, dt, tau_self, N)
    rho=zeros(N); u=zeros(N); Pxx=zeros(N); Pp=zeros(N)
    L1=zeros(N); alpha=zeros(N); beta=zeros(N); Q=zeros(N); cs=zeros(N)
    for i in 1:N
        r=U[1,i]; v=U[2,i]/max(r,1e-30); pxx=U[3,i]-r*v*v; pp=U[4,i]
        rho[i]=r; u[i]=v; Pxx[i]=pxx; Pp[i]=pp
        L1[i]=U[5,i]; alpha[i]=U[6,i]; beta[i]=U[7,i]; Q[i]=U[8,i]
        cs[i]=sqrt(CSCOEF*max(pxx,1e-30)/max(r,1e-30))
    end
    fluxes=zeros(8,N+1)
    for i in 1:N
        rL=rho[i]; uL=u[i]; pxxL=Pxx[i]; ppL=Pp[i]; lL=L1[i]; aL=alpha[i]; bL=beta[i]; qL=Q[i]; csL=cs[i]
        rR=rho[i]; uR=u[i]; pxxR=Pxx[i]; ppR=Pp[i]; lR=L1[i]; aR=alpha[i]; bR=beta[i]; qR=Q[i]; csR=cs[i] # Incorrectly copying self, should be next cell
        if i < N
            rR=rho[i+1]; uR=u[i+1]; pxxR=Pxx[i+1]; ppR=Pp[i+1]; lR=L1[i+1]; aR=alpha[i+1]; bR=beta[i+1]; qR=Q[i+1]; csR=cs[i+1]
        end
        sL=min(uL-csL,uR-csR); sR=max(uL+csL,uR+csR)
        if sL>=0.0; f1=rL*uL;f2=rL*uL^2+pxxL;f3=uL*(rL*uL^2+3*pxxL+2*ppL);f4=uL*ppL;f5=uL*lL;f6=uL*aL;f7=uL*bL;f8=uL*qL
        elseif sR<=0.0; f1=rR*uR;f2=rR*uR^2+pxxR;f3=uR*(rR*uR^2+3*pxxR+2*ppR);f4=uR*ppR;f5=uR*lR;f6=uR*aR;f7=uR*bR;f8=uR*qR
        else
            f1L=rL*uL;f2L=rL*uL^2+pxxL;f3L=uL*(rL*uL^2+3*pxxL+2*ppL);f4L=uL*ppL;f5L=uL*lL;f6L=uL*aL;f7L=uL*bL;f8L=uL*qL
            f1R=rR*uR;f2R=rR*uR^2+pxxR;f3R=uR*(rR*uR^2+3*pxxR+2*ppR);f4R=uR*ppR;f5R=uR*lR;f6R=uR*aR;f7R=uR*bR;f8R=uR*qR
            u1L=rL;u2L=rL*uL;u3L=rL*uL^2+pxxL+2*ppL;u4L=ppL;u5L=lL;u6L=aL;u7L=bL;u8L=qL
            u1R=rR;u2R=rR*uR;u3R=rR*uR^2+pxxR+2*ppR;u4R=ppR;u5R=lR;u6R=aR;u7R=bR;u8R=qR
            ds=sR-sL
            f1=(sR*f1L-sL*f1R+sL*sR*(u1R-u1L))/ds;f2=(sR*f2L-sL*f2R+sL*sR*(u2R-u2L))/ds
            f3=(sR*f3L-sL*f3R+sL*sR*(u3R-u3L))/ds;f4=(sR*f4L-sL*f4R+sL*sR*(u4R-u4L))/ds
            f5=(sR*f5L-sL*f5R+sL*sR*(u5R-u5L))/ds;f6=(sR*f6L-sL*f6R+sL*sR*(u6R-u6L))/ds
            f7=(sR*f7L-sL*f7R+sL*sR*(u7R-u7L))/ds;f8=(sR*f8L-sL*f8R+sL*sR*(u8R-u8L))/ds
        end; fluxes[:,i+1]=[f1,f2,f3,f4,f5,f6,f7,f8] # Flux at cell i+1/2 edge
    end
    fluxes[:,1]=fluxes[:,2]; fluxes[:,N+1]=fluxes[:,N]
    for i in 1:N; for k in 1:8; U[k,i]-=(dt/dx)*(fluxes[k,i+1]-fluxes[k,i]); end; end
    for i in 1:N
        r=U[1,i]; v=U[2,i]/max(r,1e-30); pxx=U[3,i]-r*v*v; pp=U[4,i]
        p_iso=(pxx+2*pp)/3.0; relax=exp(-dt/tau_self)
        pxx_new=p_iso+(pxx-p_iso)*relax; pp_new=p_iso+(pp-p_iso)*relax
        U[3,i]=pxx_new+r*v*v; U[4,i]=pp_new; U[7,i]*=relax; U[8,i]*=relax
    end
end

function couple_3_fluids(U_H,U_p,U_e,dt,N)
    for i in 1:N
        rH,vH,pxxH,ppH=_species_primitives(U_H[:,i]); rp,vp,pxxp,ppp=_species_primitives(U_p[:,i]); re,ve,pxxe,ppe=_species_primitives(U_e[:,i])
        TH=(pxxH+2.0*ppH)/(3.0*rH)*M_H; Tp=(pxxp+2.0*ppp)/(3.0*rp)*M_P; Te=(pxxe+2.0*ppe)/(3.0*re)*M_E
        if TH<0||Tp<0||Te<0; continue; end
        vrel_Hp=sqrt(8.0/π*(TH/M_H+Tp/M_P)+(vH-vp)^2); vrel_He=sqrt(8.0/π*(TH/M_H+Te/M_E)+(vH-ve)^2); vrel_pe=sqrt(8.0/π*(Tp/M_P+Te/M_E)+(vp-ve)^2)
        D_Hp=rH*rp*SIGMA_CX*vrel_Hp; D_He=rH*re*SIGMA_EH*vrel_He; D_pe=rp*re*SIGMA_COULOMB*vrel_pe
        M=@SMatrix[(rH+dt*(D_Hp+D_He)) (-dt*D_Hp) (-dt*D_He); (-dt*D_Hp) (rp+dt*(D_Hp+D_pe)) (-dt*D_pe); (-dt*D_He) (-dt*D_pe) (re+dt*(D_He+D_pe))]
        b_vec=SVector(rH*vH,rp*vp,re*ve); v_new=M\b_vec
        E_kin_old=0.5*rH*vH^2+0.5*rp*vp^2+0.5*re*ve^2; E_kin_new=0.5*rH*v_new[1]^2+0.5*rp*v_new[2]^2+0.5*re*v_new[3]^2
        dE=max(0.0,E_kin_old-E_kin_new); total_mass=rH+rp+re
        U_H[2,i]=rH*v_new[1]; U_p[2,i]=rp*v_new[2]; U_e[2,i]=re*v_new[3]
        U_H[3,i]+=rH*v_new[1]^2-rH*vH^2+dE*(rH/total_mass); U_p[3,i]+=rp*v_new[2]^2-rp*vp^2+dE*(rp/total_mass); U_e[3,i]+=re*v_new[3]^2-re*ve^2+dE*(re/total_mass)
        relax_H=dt*(D_Hp+D_He)/rH; relax_p=dt*(D_Hp+D_pe)/rp; relax_e=dt*(D_He+D_pe)/re
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
    M1 = 1.2
    N = 600
    x_min, x_max = -10.0, 10.0
    dx = (x_max - x_min) / N
    x_coords = range(x_min + dx/2, x_max - dx/2, length=N)
    
    rho1_H=1.0; P1_H=0.1; c_s1=sqrt(GAMMA_EOS*P1_H/rho1_H); u1=M1*c_s1
    rho1_p=1e-4; P1_p=1e-4*0.1; rho1_e=1e-4*M_E; P1_e=1e-4*0.1
    rho2_H, u2, P2_H = rankine_hugoniot(rho1_H, u1, P1_H, GAMMA_EOS)
    rho2_p=rho2_H*1e-4; P2_p=P2_H*1e-4; rho2_e=rho2_H*1e-4*M_E; P2_e=P2_H*1e-4
    
    U_H=zeros(8,N); U_p=zeros(8,N); U_e=zeros(8,N)
    w=2.0 # Wider shock for weaker Mach
    for i in 1:N
        f=0.5*(1.0-tanh((x_coords[i]-0.0)/w))
        rH=rho1_H*f+rho2_H*(1-f); uH=u1*f+u2*(1-f); pH=P1_H*f+P2_H*(1-f)
        U_H[1,i]=rH;U_H[2,i]=rH*uH;U_H[3,i]=pH+rH*uH^2;U_H[4,i]=pH;U_H[5,i]=rH*x_coords[i];U_H[6,i]=rH*0.02
        rp=rho1_p*f+rho2_p*(1-f); up=u1*f+u2*(1-f); pp=P1_p*f+P2_p*(1-f)
        U_p[1,i]=rp;U_p[2,i]=rp*up;U_p[3,i]=pp+rp*up^2;U_p[4,i]=pp;U_p[5,i]=rp*x_coords[i];U_p[6,i]=rp*0.02
        re=rho1_e*f+rho2_e*(1-f); ue=u1*f+u2*(1-f); pe=P1_e*f+P2_e*(1-f)
        U_e[1,i]=re;U_e[2,i]=re*ue;U_e[3,i]=pe+re*ue^2;U_e[4,i]=pe;U_e[5,i]=re*x_coords[i];U_e[6,i]=re*0.02
    end
    
    tau_H=1e-2; tau_p=1e-2; tau_e=1e-3
    t=0.0; t_end=2.0; cfl=0.3; nsteps=0
    
    println("Running Low-Mach Julia Simulation...")
    
    while t<t_end
        smax=0.0
        for i in 1:N; r,v,pxx,_=_species_primitives(U_H[:,i]); smax=max(smax,abs(v)+sqrt(3*max(pxx,1e-15)/max(r,1e-15))); r,v,pxx,_=_species_primitives(U_p[:,i]); smax=max(smax,abs(v)+sqrt(3*max(pxx,1e-15)/max(r,1e-15))); r,v,pxx,_=_species_primitives(U_e[:,i]); smax=max(smax,abs(v)+sqrt(3*max(pxx,1e-15)/max(r,1e-15))); end
        dt=min(cfl*dx/smax,t_end-t); if dt<=0; break; end
        
        _species_step_explicit(U_H,dx,dt,tau_H,N); _species_step_explicit(U_p,dx,dt,tau_p,N); _species_step_explicit(U_e,dx,dt,tau_e,N)
        couple_3_fluids(U_H,U_p,U_e,dt,N)

        for c in 1:2
            U_H[1,c]=rho1_H;U_H[2,c]=rho1_H*u1;U_H[3,c]=P1_H+rho1_H*u1^2;U_H[4,c]=P1_H;U_H[5,c]=rho1_H*x_coords[c]-rho1_H*u1*(t+dt);U_H[6,c]=rho1_H*0.02
            U_p[1,c]=rho1_p;U_p[2,c]=rho1_p*u1;U_p[3,c]=P1_p+rho1_p*u1^2;U_p[4,c]=P1_p;U_p[5,c]=rho1_p*x_coords[c]-rho1_p*u1*(t+dt);U_p[6,c]=rho1_p*0.02
            U_e[1,c]=rho1_e;U_e[2,c]=rho1_e*u1;U_e[3,c]=P1_e+rho1_e*u1^2;U_e[4,c]=P1_e;U_e[5,c]=rho1_e*x_coords[c]-rho1_e*u1*(t+dt);U_e[6,c]=rho1_e*0.02
        end
        U_H[:,N]=U_H[:,N-1]; U_p[:,N]=U_p[:,N-1]; U_e[:,N]=U_e[:,N-1]
        
        t+=dt; nsteps+=1
    end
    println("Finished in $nsteps steps.")

    u_H = U_H[2,:] ./ U_H[1,:]
    u_p = U_p[2,:] ./ U_p[1,:]
    u_e = U_e[2,:] ./ U_e[1,:]
    
    r_H = U_H[1,:]
    r_p = U_p[1,:]
    r_e = U_e[1,:]
    
    current_j = r_p .* (u_p .- u_e)

    fig=Figure(size=(1600,900))
    ax1=Axis(fig[1,1],ylabel="Density",title="Low-Mach Shock (Final Snapshot)",aspect=AxisAspect(2))
    ax2=Axis(fig[2,1],ylabel="Velocity",aspect=AxisAspect(2))
    ax3=Axis(fig[3,1],ylabel="Electric Current",xlabel="Position",aspect=AxisAspect(2))
    
    lines!(ax1,x_coords,r_H,label="Neutrals",color=:black)
    lines!(ax1,x_coords,r_p.*1e4,label="Protons (x1e4)",color=:orange)
    lines!(ax1,x_coords,r_e.*(1836.0*1e4),label="Electrons (x1e4)",color=:cyan,linestyle=:dash)
    axislegend(ax1,position=:rb)
    ylims!(ax1,0.9,1.4)
    
    lines!(ax2,x_coords,u_H,label="u_H",color=:black)
    lines!(ax2,x_coords,u_p,label="u_p",color=:orange)
    lines!(ax2,x_coords,u_e,label="u_e",color=:cyan)
    axislegend(ax2,position=:rt)
    ylims!(ax2,0.35,0.55)
    
    lines!(ax3,x_coords,current_j,label="Current J",color=:red)
    axislegend(ax3,position=:rt)
    ylims!(ax3,-1e-7,6e-7)

    save("reference/figs/low_mach_shock.png", fig)
    println("Static plot saved to reference/figs/low_mach_shock.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
