import os, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numba as nb

from dfmm.schemes._common import (IDX_RHO, IDX_MOM, IDX_EXX, IDX_PP,
                                  IDX_L1, IDX_ALPHA, IDX_BETA, IDX_M3)
from dfmm.schemes.two_fluid import _species_primitives, _species_step_transmissive
from dfmm.setups.shock import rankine_hugoniot

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")
os.makedirs(FIG_DIR, exist_ok=True)

M_H = 1.0
M_P = 1.0
M_E = 1.0 / 1836.0

SIGMA_HH = 100.0
SIGMA_CX = 500.0
SIGMA_EH = 100.0
SIGMA_COULOMB = 1000.0

@nb.njit(cache=True)
def max_signal_speed_3(U, N):
    smax = 0.0
    for i in range(N):
        rH, vH, pxxH, _, _, _, _, _, _ = _species_primitives(U[0:8, i])
        smax = max(smax, abs(vH) + np.sqrt(3.0*max(pxxH, 1e-15)/max(rH, 1e-15)))
        rp, vp, pxxp, _, _, _, _, _, _ = _species_primitives(U[8:16, i])
        smax = max(smax, abs(vp) + np.sqrt(3.0*max(pxxp, 1e-15)/max(rp, 1e-15)))
        re, ve, pxxe, _, _, _, _, _, _ = _species_primitives(U[16:24, i])
        smax = max(smax, abs(ve) + np.sqrt(3.0*max(pxxe, 1e-15)/max(re, 1e-15)))
    return smax

@nb.njit(cache=True)
def couple_3_fluids(U, dt, N):
    for i in range(N):
        rH, vH, pxxH, ppH, l1H, aH, bH, svvH, qH = _species_primitives(U[0:8, i])
        rp, vp, pxxp, ppp, l1p, ap, bp, svvp, qp = _species_primitives(U[8:16, i])
        re, ve, pxxe, ppe, l1e, ae, be, svve, qe = _species_primitives(U[16:24, i])
        
        TH = (pxxH + 2.0*ppH) / (3.0 * rH) * M_H
        Tp = (pxxp + 2.0*ppp) / (3.0 * rp) * M_P
        Te = (pxxe + 2.0*ppe) / (3.0 * re) * M_E
        
        vrel_Hp = np.sqrt(8.0/np.pi * (TH/M_H + Tp/M_P) + (vH - vp)**2)
        vrel_He = np.sqrt(8.0/np.pi * (TH/M_H + Te/M_E) + (vH - ve)**2)
        vrel_pe = np.sqrt(8.0/np.pi * (Tp/M_P + Te/M_E) + (vp - ve)**2)
        
        D_Hp = rH * rp * SIGMA_CX * vrel_Hp
        D_He = rH * re * SIGMA_EH * vrel_He
        D_pe = rp * re * SIGMA_COULOMB * vrel_pe
        
        M = np.zeros((3,3))
        M[0,0] = rH + dt*(D_Hp + D_He); M[0,1] = -dt*D_Hp; M[0,2] = -dt*D_He
        M[1,0] = -dt*D_Hp; M[1,1] = rp + dt*(D_Hp + D_pe); M[1,2] = -dt*D_pe
        M[2,0] = -dt*D_He; M[2,1] = -dt*D_pe; M[2,2] = re + dt*(D_He + D_pe)
        
        b_vec = np.array([rH*vH, rp*vp, re*ve])
        v_new = np.linalg.solve(M, b_vec)
        
        E_kin_old = 0.5*rH*vH**2 + 0.5*rp*vp**2 + 0.5*re*ve**2
        E_kin_new = 0.5*rH*v_new[0]**2 + 0.5*rp*v_new[1]**2 + 0.5*re*v_new[2]**2
        dE = max(0.0, E_kin_old - E_kin_new)
        
        total_mass = rH + rp + re
        U[IDX_EXX, i] += dE * (rH/total_mass)
        U[IDX_EXX+8, i] += dE * (rp/total_mass)
        U[IDX_EXX+16, i] += dE * (re/total_mass)
        
        U[IDX_MOM, i] = rH * v_new[0]
        U[IDX_MOM+8, i] = rp * v_new[1]
        U[IDX_MOM+16, i] = re * v_new[2]
        
        U[IDX_EXX, i] += rH*v_new[0]**2 - rH*vH**2
        U[IDX_EXX+8, i] += rp*v_new[1]**2 - rp*vp**2
        U[IDX_EXX+16, i] += re*v_new[2]**2 - re*ve**2
        
        relax_H = dt * (D_Hp + D_He) / rH
        relax_p = dt * (D_Hp + D_pe) / rp
        relax_e = dt * (D_He + D_pe) / re
        
        U[IDX_M3, i] *= np.exp(-relax_H); U[IDX_BETA, i] *= np.exp(-relax_H)
        U[IDX_M3+8, i] *= np.exp(-relax_p); U[IDX_BETA+8, i] *= np.exp(-relax_p)
        U[IDX_M3+16, i] *= np.exp(-relax_e); U[IDX_BETA+16, i] *= np.exp(-relax_e)

def run_steady_shock_history(N=600, M1=3.0, t_end=0.6, cfl=0.3, num_frames=120):
    x_min, x_max = -3.0, 2.0
    x = np.linspace(x_min, x_max, N, endpoint=False) + (x_max - x_min)/(2*N)
    dx = (x_max - x_min)/N
    
    rho1_H = 1.0; P1_H = 0.1
    cs1_H = np.sqrt(5.0/3.0 * P1_H / rho1_H)
    u1 = M1 * cs1_H
    rho1_p = 1e-4; P1_p = 1e-4 * 0.1
    rho1_e = 1e-4 * M_E; P1_e = 1e-4 * 0.1
    
    rho2_H, u2, P2_H, _ = rankine_hugoniot(rho1_H, u1, P1_H, 5.0/3.0)
    rho2_p = rho2_H * 1e-4; P2_p = P2_H * 1e-4
    rho2_e = rho2_H * 1e-4 * M_E; P2_e = P2_H * 1e-4
    
    U = np.zeros((24, N))
    for i in range(N):
        if x[i] < 0.0:
            U[0,i]=rho1_H; U[1,i]=rho1_H*u1; U[2,i]=P1_H+rho1_H*u1**2; U[3,i]=P1_H
            U[8,i]=rho1_p; U[9,i]=rho1_p*u1; U[10,i]=P1_p+rho1_p*u1**2; U[11,i]=P1_p
            U[16,i]=rho1_e; U[17,i]=rho1_e*u1; U[18,i]=P1_e+rho1_e*u1**2; U[19,i]=P1_e
        else:
            U[0,i]=rho2_H; U[1,i]=rho2_H*u2; U[2,i]=P2_H+rho2_H*u2**2; U[3,i]=P2_H
            U[8,i]=rho2_p; U[9,i]=rho2_p*u2; U[10,i]=P2_p+rho2_p*u2**2; U[11,i]=P2_p
            U[16,i]=rho2_e; U[17,i]=rho2_e*u2; U[18,i]=P2_e+rho2_e*u2**2; U[19,i]=P2_e
        U[4,i]=U[0,i]*x[i]; U[5,i]=U[0,i]*0.02
        U[12,i]=U[8,i]*x[i]; U[13,i]=U[8,i]*0.02
        U[20,i]=U[16,i]*x[i]; U[21,i]=U[16,i]*0.02

    tau_H = 1e-3; tau_p = 1e-3; tau_e = 1e-4
    t = 0.0; nsteps = 0
    history = []
    dt_frame = t_end / num_frames
    next_frame_time = 0.0
    
    while t < t_end:
        if t >= next_frame_time:
            history.append((t, U.copy()))
            next_frame_time += dt_frame
            
        smax = max_signal_speed_3(U, N)
        dt = min(cfl * dx / smax, t_end - t)
        if dt <= 0: break
        
        # Save boundary L1 gradient state before step
        L1_grad_H = (U[4, 1]/U[0,1] - U[4, 0]/U[0,0])
        
        _species_step_transmissive(U, 0, dx, dt, tau_H, N)
        _species_step_transmissive(U, 8, dx, dt, tau_p, N)
        _species_step_transmissive(U, 16, dx, dt, tau_e, N)
        couple_3_fluids(U, dt, N)
        
        # REFINED BC: Pin mass/mom/energy to inflow reservoir, 
        # but let L1 and higher moments flow in naturally from the ghost cells.
        for c in [0, 1]:
            # Neutrals
            U[0,c]=rho1_H; U[1,c]=rho1_H*u1; U[2,c]=P1_H+rho1_H*u1**2; U[3,c]=P1_H
            # Protons
            U[8,c]=rho1_p; U[9,c]=rho1_p*u1; U[10,c]=P1_p+rho1_p*u1**2; U[11,c]=P1_p
            # Electrons
            U[16,c]=rho1_e; U[17,c]=rho1_e*u1; U[18,c]=P1_e+rho1_e*u1**2; U[19,c]=P1_e
            
        U[:, -1] = U[:, -2]
        t += dt
        nsteps += 1
    return x, history

if __name__ == "__main__":
    print("Running simulation and capturing frames...")
    t0 = time.time()
    x, history = run_steady_shock_history(N=600, M1=3.0, t_end=0.6, num_frames=120)
    print(f"Simulation finished in {time.time()-t0:.2f}s. Generating mp4...")
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    line_H_rho, = axs[0].plot([], [], label="Neutrals (H)", color='k')
    line_p_rho, = axs[0].plot([], [], label="Protons (x 1e4)", color='orange')
    
    line_H_u, = axs[1].plot([], [], label="u_H", color='k')
    line_p_u, = axs[1].plot([], [], label="u_p", color='orange')
    line_e_u, = axs[1].plot([], [], label="u_e", color='cyan')
    
    line_j, = axs[2].plot([], [], color='red', label="Current J")
    
    time_text = axs[0].text(0.02, 0.90, '', transform=axs[0].transAxes, fontsize=12)
    
    axs[0].set_ylabel("Density")
    axs[0].legend(loc="upper right")
    axs[0].set_xlim(-3.0, 2.0)
    axs[0].set_ylim(0.8, 3.5)
    
    axs[1].set_ylabel("Velocity")
    axs[1].legend(loc="upper right")
    axs[1].set_ylim(0.0, 1.5)
    
    axs[2].set_ylabel("Electric Current")
    axs[2].set_xlabel("Position")
    axs[2].legend(loc="upper right")
    axs[2].set_ylim(-1e-5, 1e-5)
    
    plt.suptitle("Evolution of 3-Fluid Charge Separation in Primordial Shock")
    plt.tight_layout()
    
    def init():
        line_H_rho.set_data([], [])
        line_p_rho.set_data([], [])
        line_H_u.set_data([], [])
        line_p_u.set_data([], [])
        line_e_u.set_data([], [])
        line_j.set_data([], [])
        time_text.set_text('')
        return line_H_rho, line_p_rho, line_H_u, line_p_u, line_e_u, line_j, time_text
        
    def update(frame_idx):
        t, U = history[frame_idx]
        u_H = U[1]/U[0]; u_p = U[9]/U[8]; u_e = U[17]/U[16]
        current_j = U[8] * (u_p - u_e)
        
        line_H_rho.set_data(x, U[0]); line_p_rho.set_data(x, U[8]*1e4)
        line_H_u.set_data(x, u_H); line_p_u.set_data(x, u_p); line_e_u.set_data(x, u_e)
        line_j.set_data(x, current_j)
        time_text.set_text(f'Time = {t:.3f}')
        
        max_j = np.max(np.abs(current_j))
        if max_j > 1e-10:
            axs[2].set_ylim(-max_j*1.2, max_j*1.2)
        return line_H_rho, line_p_rho, line_H_u, line_p_u, line_e_u, line_j, time_text

    writer = FFMpegWriter(fps=30, metadata=dict(artist='dfmm'), bitrate=2000)
    anim = FuncAnimation(fig, update, frames=len(history), init_func=init, blit=True)
    
    out_path = os.path.join(FIG_DIR, "three_fluid_shock_evolution.mp4")
    anim.save(out_path, writer=writer)
    print(f"Movie saved to {out_path}")
