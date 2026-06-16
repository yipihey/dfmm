"""
Experiment 16: 3-Fluid Charge Separation in a Primordial Shock.

Species:
  A (Neutrals, H): Mass 1.0, 99.99% of density.
  B (Protons, p):  Mass 1.0, 0.01% of density.
  C (Electrons, e): Mass 1/1836, 0.01% of density.

Cross-Coupling:
  Neutrals drag Protons via Charge Exchange (CX).
  Neutrals drag Electrons via Polarization scattering.
  Protons and Electrons couple via Coulomb drag.
  Because the electron sound speed is ~43x faster, this requires a small dt.
  We use Backward Euler for the 3x3 drag coupling to ensure unconditional stability.

Outputs:
  paper/figs/three_fluid_charge_separation.png
"""
import os, time
import numpy as np
import matplotlib.pyplot as plt
import numba as nb

from dfmm.schemes._common import (IDX_RHO, IDX_MOM, IDX_EXX, IDX_PP,
                                  IDX_L1, IDX_ALPHA, IDX_BETA, IDX_M3)
from dfmm.schemes.two_fluid import _species_primitives, _species_step_transmissive
from dfmm.setups.shock import rankine_hugoniot

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")
os.makedirs(FIG_DIR, exist_ok=True)

# Physical constants for the model
M_H = 1.0
M_P = 1.0
M_E = 1.0 / 1836.0

# Base cross sections (arbitrary scaled units to make the shock fit the domain)
# We want the neutral shock to be ~0.1 units wide.
SIGMA_HH = 100.0
# Charge exchange is ~5x hard sphere
SIGMA_CX = 500.0
# e-H polarization is ~1x
SIGMA_EH = 100.0
# Coulomb is large at low T, let's say 1000.0
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
    """
    3x3 Backward Euler solve for momentum and energy exchange.
    Applied locally in each cell.
    """
    for i in range(N):
        # Extract primitives
        rH, vH, pxxH, ppH, l1H, aH, bH, svvH, qH = _species_primitives(U[0:8, i])
        rp, vp, pxxp, ppp, l1p, ap, bp, svvp, qp = _species_primitives(U[8:16, i])
        re, ve, pxxe, ppe, l1e, ae, be, svve, qe = _species_primitives(U[16:24, i])
        
        TH = (pxxH + 2.0*ppH) / (3.0 * rH) * M_H
        Tp = (pxxp + 2.0*ppp) / (3.0 * rp) * M_P
        Te = (pxxe + 2.0*ppe) / (3.0 * re) * M_E
        
        # 1. Calculate momentum collision frequencies \nu_{AB}^p
        # CX (H-p)
        vrel_Hp = np.sqrt(8.0/np.pi * (TH/M_H + Tp/M_P) + (vH - vp)**2)
        nu_Hp = rp * SIGMA_CX * vrel_Hp # rate at which H is dragged by p
        
        # Polarization (H-e)
        vrel_He = np.sqrt(8.0/np.pi * (TH/M_H + Te/M_E) + (vH - ve)**2)
        nu_He = re * SIGMA_EH * vrel_He
        
        # Coulomb (p-e)
        vrel_pe = np.sqrt(8.0/np.pi * (Tp/M_P + Te/M_E) + (vp - ve)**2)
        nu_pe = re * SIGMA_COULOMB * vrel_pe
        
        # Symmetric rates R_AB = rho_A rho_B / (rho_A + rho_B) * \nu^p_{AB}
        # Actually \nu_{AB} defined above is such that Force = - rho_A \nu_{AB} (vA - vB)
        # Let's use the explicit R_AB form:
        R_Hp = (rH * rp) / (rH + rp) * (SIGMA_CX * vrel_Hp * (rH + rp)/M_H) # Approximation
        
        # Let's simplify to symmetric drag coefficients D_AB:
        # F_{A->B} = - D_AB * (v_A - v_B)
        D_Hp = rH * rp * SIGMA_CX * vrel_Hp
        D_He = rH * re * SIGMA_EH * vrel_He
        D_pe = rp * re * SIGMA_COULOMB * vrel_pe
        
        # Backward Euler for v: M * v^{n+1} = \rho v^n
        # M_11 = rH + dt*(D_Hp + D_He), M_12 = -dt*D_Hp, M_13 = -dt*D_He
        M = np.zeros((3,3))
        M[0,0] = rH + dt*(D_Hp + D_He); M[0,1] = -dt*D_Hp; M[0,2] = -dt*D_He
        M[1,0] = -dt*D_Hp; M[1,1] = rp + dt*(D_Hp + D_pe); M[1,2] = -dt*D_pe
        M[2,0] = -dt*D_He; M[2,1] = -dt*D_pe; M[2,2] = re + dt*(D_He + D_pe)
        
        b_vec = np.array([rH*vH, rp*vp, re*ve])
        
        # Solve M * v_new = b
        v_new = np.linalg.solve(M, b_vec)
        
        # Frictional heating (energy dissipated by drag)
        E_kin_old = 0.5*rH*vH**2 + 0.5*rp*vp**2 + 0.5*re*ve**2
        E_kin_new = 0.5*rH*v_new[0]**2 + 0.5*rp*v_new[1]**2 + 0.5*re*v_new[2]**2
        dE = max(0.0, E_kin_old - E_kin_new)
        
        # Distribute heating to internal energy proportional to mass
        total_mass = rH + rp + re
        U[IDX_EXX, i] += dE * (rH/total_mass)
        U[IDX_EXX+8, i] += dE * (rp/total_mass)
        U[IDX_EXX+16, i] += dE * (re/total_mass)
        
        # Thermal equilibration (simplified explicit Euler, since thermal is usually slower)
        # For CX, nu_T ~ nu_p / 2
        # For H-e, nu_T ~ (m_e/m_H) nu_p
        # Skip detailed thermal equilibration for this demo, charge separation is driven by momentum.
        
        # Update momentum
        U[IDX_MOM, i] = rH * v_new[0]
        U[IDX_MOM+8, i] = rp * v_new[1]
        U[IDX_MOM+16, i] = re * v_new[2]
        
        # Update E_xx to maintain total energy = E_int + E_kin
        U[IDX_EXX, i] += rH*v_new[0]**2 - rH*vH**2
        U[IDX_EXX+8, i] += rp*v_new[1]**2 - rp*vp**2
        U[IDX_EXX+16, i] += re*v_new[2]**2 - re*ve**2
        
        # Relax higher moments (Q, beta, Pxx-Pperp) due to cross collisions
        relax_H = dt * (D_Hp + D_He) / rH
        relax_p = dt * (D_Hp + D_pe) / rp
        relax_e = dt * (D_He + D_pe) / re
        
        U[IDX_M3, i] *= np.exp(-relax_H)
        U[IDX_BETA, i] *= np.exp(-relax_H)
        
        U[IDX_M3+8, i] *= np.exp(-relax_p)
        U[IDX_BETA+8, i] *= np.exp(-relax_p)
        
        U[IDX_M3+16, i] *= np.exp(-relax_e)
        U[IDX_BETA+16, i] *= np.exp(-relax_e)

def run_steady_shock(N=200, M1=3.0, t_end=2.0, cfl=0.3):
    x = np.linspace(0, 1, N, endpoint=False) + 0.5/N
    dx = 1.0/N
    
    # Upstream
    rho1_H = 1.0; P1_H = 0.1
    cs1_H = np.sqrt(5.0/3.0 * P1_H / rho1_H)
    u1 = M1 * cs1_H
    
    rho1_p = 1e-4; P1_p = 1e-4 * 0.1
    rho1_e = 1e-4 * M_E; P1_e = 1e-4 * 0.1
    
    # Downstream (Rankine-Hugoniot for Neutrals)
    rho2_H, u2, P2_H, _ = rankine_hugoniot(rho1_H, u1, P1_H, 5.0/3.0)
    rho2_p = rho2_H * 1e-4; P2_p = P2_H * 1e-4
    rho2_e = rho2_H * 1e-4 * M_E; P2_e = P2_H * 1e-4
    
    U = np.zeros((24, N))
    
    # Initialize Step
    for i in range(N):
        if x[i] < 0.5:
            # H
            U[0,i] = rho1_H; U[1,i] = rho1_H*u1; U[2,i] = P1_H + rho1_H*u1**2; U[3,i] = P1_H
            # p
            U[8,i] = rho1_p; U[9,i] = rho1_p*u1; U[10,i] = P1_p + rho1_p*u1**2; U[11,i] = P1_p
            # e
            U[16,i] = rho1_e; U[17,i] = rho1_e*u1; U[18,i] = P1_e + rho1_e*u1**2; U[19,i] = P1_e
        else:
            # H
            U[0,i] = rho2_H; U[1,i] = rho2_H*u2; U[2,i] = P2_H + rho2_H*u2**2; U[3,i] = P2_H
            # p
            U[8,i] = rho2_p; U[9,i] = rho2_p*u2; U[10,i] = P2_p + rho2_p*u2**2; U[11,i] = P2_p
            # e
            U[16,i] = rho2_e; U[17,i] = rho2_e*u2; U[18,i] = P2_e + rho2_e*u2**2; U[19,i] = P2_e
            
        U[4,i] = U[0,i]*x[i]; U[5,i] = U[0,i]*0.02
        U[12,i] = U[8,i]*x[i]; U[13,i] = U[8,i]*0.02
        U[20,i] = U[16,i]*x[i]; U[21,i] = U[16,i]*0.02

    # Save inflow boundary
    U_inflow = U[:, 0].copy()
    
    # Self-BGK times
    tau_H = 1e-3
    tau_p = 1e-3
    tau_e = 1e-4

    t = 0.0; nsteps = 0
    while t < t_end:
        smax = max_signal_speed_3(U, N)
        dt = min(cfl * dx / smax, t_end - t)
        if dt <= 0: break
        
        # 1. Advect each independently
        _species_step_transmissive(U, 0, dx, dt, tau_H, N)
        _species_step_transmissive(U, 8, dx, dt, tau_p, N)
        _species_step_transmissive(U, 16, dx, dt, tau_e, N)
        
        # 2. Cross-couple
        couple_3_fluids(U, dt, N)
        
        # 3. Apply Dirichlet inflow BC
        U[:, 0] = U_inflow
        U[:, 1] = U_inflow
        
        # 4. Outflow zero-gradient BC
        U[:, -1] = U[:, -2]
        
        t += dt
        nsteps += 1
        if nsteps % 100 == 0:
            print(f"t={t:.4f} / {t_end:.4f}, dt={dt:.2e}, steps={nsteps}")
            
    return x, U, nsteps

if __name__ == "__main__":
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Back to Mach 3 for Neutrals to see the precursor.
    M1 = 3.0
    N = 1200
    # Large domain: x from -10 to 2. Shock at x=0.
    # Information travels upstream at (cs_e - u1) ~ 16.
    # In t=0.5, it travels ~8 units. So x_min = -10 is safe.
    x_min, x_max = -10.0, 2.0
    x = np.linspace(x_min, x_max, N, endpoint=False) + (x_max - x_min)/(2*N)
    dx = (x_max - x_min)/N
    
    print(f"Running N={N} with M=3 in large domain [{x_min}, {x_max}]...")
    
    # Initial conditions
    # Upstream (Left of 0)
    rho1_H = 1.0; P1_H = 0.1
    cs1_H = np.sqrt(5.0/3.0 * P1_H / rho1_H)
    u1 = M1 * cs1_H
    rho1_p = 1e-4; P1_p = 1e-4 * 0.1
    rho1_e = 1e-4 * M_E; P1_e = 1e-4 * 0.1
    
    # Downstream (Right of 0)
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

    U_inflow = U[:, 0].copy()
    tau_H = 1e-3; tau_p = 1e-3; tau_e = 1e-4
    t = 0.0; t_end = 0.5; nsteps = 0; cfl = 0.3
    
    t0 = time.time()
    while t < t_end:
        smax = max_signal_speed_3(U, N)
        dt = min(cfl * dx / smax, t_end - t)
        if dt <= 0: break
        _species_step_transmissive(U, 0, dx, dt, tau_H, N)
        _species_step_transmissive(U, 8, dx, dt, tau_p, N)
        _species_step_transmissive(U, 16, dx, dt, tau_e, N)
        couple_3_fluids(U, dt, N)
        U[:, 0] = U_inflow; U[:, 1] = U_inflow
        U[:, -1] = U[:, -2]
        t += dt
        nsteps += 1
        if nsteps % 1000 == 0:
            print(f"t={t:.4f}, steps={nsteps}")
            
    print(f"Finished in {time.time()-t0:.2f}s ({nsteps} steps).")
    
    u_H = U[1]/U[0]; u_p = U[9]/U[8]; u_e = U[17]/U[16]
    current_j = U[8] * (u_p - u_e)
    
    axs[0].plot(x, U[0], label="Neutrals (H)", color='k')
    axs[0].plot(x, U[8]*1e4, label="Protons (x 1e4)", color='orange')
    axs[0].set_ylabel("Density")
    axs[0].legend()
    
    axs[1].plot(x, u_H, label="u_H", color='k')
    axs[1].plot(x, u_p, label="u_p", color='orange')
    axs[1].plot(x, u_e, label="u_e", color='cyan')
    axs[1].set_ylabel("Velocity")
    axs[1].legend()
    
    axs[2].plot(x, current_j, color='red', label="Current J")
    axs[2].set_ylabel("Electric Current")
    axs[2].set_xlabel("Position")
    axs[2].legend()
    
    plt.suptitle("3-Fluid Charge Separation in Mach 3 Shock (Large Domain)")
    plt.tight_layout()
    filename = os.path.join(FIG_DIR, "three_fluid_large_box.png")
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
