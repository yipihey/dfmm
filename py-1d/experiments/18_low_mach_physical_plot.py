import os, time
import numpy as np
import matplotlib.pyplot as plt
import numba as nb

# Import necessary components from the existing Python dfmm library
from dfmm.schemes._common import (IDX_RHO, IDX_MOM, IDX_EXX, IDX_PP,
                                  IDX_L1, IDX_ALPHA, IDX_BETA, IDX_M3,
                                  _species_primitives, _species_step_transmissive, hll_edge_flux)
from dfmm.setups.shock import rankine_hugoniot

# Directory for figures
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")
os.makedirs(FIG_DIR, exist_ok=True)

# --- Physical Constants ---
KB = 1.380649e-16  # erg/K
M_P_G = 1.6726219e-24 # g (proton mass)
M_E_G = 9.10938356e-28 # g (electron mass)
E_CHARGE = 1.60217663e-19 # Coulombs
CM_PER_PC = 3.086e18 # cm per parsec

# --- Simulation Parameters (Dimensionless) ---
M_H = 1.0; M_P = 1.0; M_E = M_E_G/M_P_G # Relative masses
SIGMA_HH = 1.0e-15 # cm^2, physical cross section (for unit scaling)
SIGMA_CX = 5.0e-15 # p-H charge exchange
SIGMA_EH = 1.0e-15 # e-H polarization scattering
SIGMA_COULOMB = 1.0e-13 # p-e Coulomb
GAMMA_EOS = 5.0/3.0; CSCOEF = 3.0

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
def couple_3_fluids(U, dt, N, u0_unit, T1_phys_k, n0_unit, rho0_mass_density):
    # This function operates on the *full U array* (24 fields)
    for i in range(N):
        # Species H (Neutrals)
        rH_d, vH_d, pxxH_d, ppH_d, _, _, _, _, _ = _species_primitives(U[0:8, i])
        # Species P (Protons)
        rp_d, vp_d, pxxp_d, ppp_d, _, _, _, _, _ = _species_primitives(U[8:16, i])
        # Species E (Electrons)
        re_d, ve_d, pxxe_d, ppe_d, _, _, _, _, _ = _species_primitives(U[16:24, i])
        
        # Calculate physical temperatures from dimensionless pressures
        TH_phys = (pxxH_d + 2.0*ppH_d) / (3.0 * rH_d) * T1_phys_k
        Tp_phys = (pxxp_d + 2.0*ppp_d) / (3.0 * rp_d) * T1_phys_k
        Te_phys = (pxxe_d + 2.0*ppe_d) / (3.0 * re_d) * T1_phys_k
        
        # Guard against negative temperatures (can happen in extreme non-equilibrium)
        TH_phys = max(TH_phys, 1e-3)
        Tp_phys = max(Tp_phys, 1e-3)
        Te_phys = max(Te_phys, 1e-3)
        
        # Calculate physical relative velocities (cm/s)
        vrel_Hp = np.sqrt(8.0*KB/np.pi * (TH_phys/M_P_G + Tp_phys/M_P_G) + ((vH_d-vp_d)*u0_unit)**2)
        vrel_He = np.sqrt(8.0*KB/np.pi * (TH_phys/M_P_G + Te_phys/M_E_G) + ((vH_d-ve_d)*u0_unit)**2)
        vrel_pe = np.sqrt(8.0*KB/np.pi * (Tp_phys/M_P_G + Te_phys/M_E_G) + ((vp_d-ve_d)*u0_unit)**2)
        
        # Physical drag coefficients (mass density * physical cross section * vrel)
        # Note: (rH_d * n0_unit * M_P_G) is the physical mass density of H
        D_Hp = (rH_d * n0_unit * M_P_G) * (rp_d * n0_unit * SIGMA_CX) * vrel_Hp
        D_He = (rH_d * n0_unit * M_P_G) * (re_d * n0_unit * SIGMA_EH) * vrel_He
        D_pe = (rp_d * n0_unit * M_P_G) * (re_d * n0_unit * SIGMA_COULOMB) * vrel_pe
        
        # M matrix for implicit Backward Euler momentum solve (in physical units)
        M = np.zeros((3,3))
        M[0,0] = (rH_d * n0_unit * M_P_G) + dt*(D_Hp + D_He)
        M[0,1] = -dt*D_Hp
        M[0,2] = -dt*D_He
        M[1,0] = -dt*D_Hp
        M[1,1] = (rp_d * n0_unit * M_P_G) + dt*(D_Hp + D_pe)
        M[1,2] = -dt*D_pe
        M[2,0] = -dt*D_He
        M[2,1] = -dt*D_pe
        M[2,2] = (re_d * n0_unit * M_E_G) + dt*(D_He + D_pe)
        
        # b_vec (physical momentum density) at current time n
        b_vec = np.array([
            (rH_d * n0_unit * M_P_G) * (vH_d * u0_unit),
            (rp_d * n0_unit * M_P_G) * (vp_d * u0_unit),
            (re_d * n0_unit * M_E_G) * (ve_d * u0_unit)
        ])
        
        # Solve for new physical velocities (cm/s)
        v_new_physical = np.linalg.solve(M, b_vec)
        
        # Convert back to dimensionless velocities
        vH_new_d = v_new_physical[0] / u0_unit
        vp_new_d = v_new_physical[1] / u0_unit
        ve_new_d = v_new_physical[2] / u0_unit
        
        # Energy dissipation (physical erg/cm^3)
        E_kin_old_phys = 0.5*(rH_d*n0_unit*M_P_G)*(vH_d*u0_unit)**2 + 0.5*(rp_d*n0_unit*M_P_G)*(vp_d*u0_unit)**2 + 0.5*(re_d*n0_unit*M_E_G)*(ve_d*u0_unit)**2
        E_kin_new_phys = 0.5*(rH_d*n0_unit*M_P_G)*v_new_physical[0]**2 + 0.5*(rp_d*n0_unit*M_P_G)*v_new_physical[1]**2 + 0.5*(re_d*n0_unit*M_E_G)*v_new_physical[2]**2
        dE_phys = max(0.0, E_kin_old_phys - E_kin_new_phys)
        
        total_mass_phys = (rH_d*n0_unit*M_P_G) + (rp_d*n0_unit*M_P_G) + (re_d*n0_unit*M_E_G)
        
        # Update momentum (dimensionless)
        U[IDX_MOM, i]    = rH_d * vH_new_d
        U[IDX_MOM+8, i]  = rp_d * vp_new_d
        U[IDX_MOM+16, i] = re_d * ve_new_d
        
        # Update E_xx (dimensionless) to maintain total energy = E_int + E_kin
        # Add frictional heating as internal energy
        # Divide by P0_unit to get dimensionless energy density
        U[IDX_EXX, i]    += rH_d*(vH_new_d**2 - vH_d**2) + (dE_phys * (rH_d*n0_unit*M_P_G/total_mass_phys))/(rho0_mass_density * u0_unit**2)
        U[IDX_EXX+8, i]  += rp_d*(vp_new_d**2 - vp_d**2) + (dE_phys * (rp_d*n0_unit*M_P_G/total_mass_phys))/(rho0_mass_density * u0_unit**2)
        U[IDX_EXX+16, i] += re_d*(ve_new_d**2 - ve_d**2) + (dE_phys * (re_d*n0_unit*M_E_G/total_mass_phys))/(rho0_mass_density * u0_unit**2)
        
        # Higher moment relaxation (decay beta and Q) - use physical rates
        # Divide by physical mass density for dimensionless rate
        relax_H = dt * (D_Hp + D_He) / (rH_d * n0_unit * M_P_G)
        relax_p = dt * (D_Hp + D_pe) / (rp_d * n0_unit * M_P_G)
        relax_e = dt * (D_He + D_pe) / (re_d * n0_unit * M_E_G)
        
        U[IDX_M3, i]     *= np.exp(-relax_H); U[IDX_BETA, i] *= np.exp(-relax_H)
        U[IDX_M3+8, i]   *= np.exp(-relax_p); U[IDX_BETA+8, i] *= np.exp(-relax_p)
        U[IDX_M3+16, i]  *= np.exp(-relax_e); U[IDX_BETA+16, i] *= np.exp(-relax_e)

def run_low_mach_physical_plot(N=1200, M1=1.2, t_end=3.0, cfl=0.3):
    # --- Physical Parameters for this run ---
    T1_phys_k = 50.0       # K
    n1_phys_cm3 = 1e-3     # cm^-3
    u1_phys_cms = 1e5      # 1 km/s in cm/s

    # --- Unit Scaling --- 
    u0_unit = np.sqrt(KB * T1_phys_k / M_P_G) # cm/s
    n0_unit = n1_phys_cm3 # cm^-3
    L0_unit = 1.0 / (n0_unit * SIGMA_HH) # cm
    P0_unit = n0_unit * KB * T1_phys_k # erg/cm^3
    J0_unit = E_CHARGE * n0_unit * u0_unit # A/cm^2 (simple e*n*u scaling)
    rho0_mass_density = n0_unit * M_P_G # g/cm^3


    x_min_d, x_max_d = -20.0, 20.0 # Dimensionless domain
    x_coords_d = np.linspace(x_min_d, x_max_d, N, endpoint=False) + (x_max_d - x_min_d)/(2*N)
    dx_d = (x_max_d - x_min_d)/N
    
    # Dimensionless ICs for a Mach 1.2 shock
    rho1_d = 1.0
    P1_d = (KB * T1_phys_k * n1_phys_cm3) / P0_unit
    u1_d = u1_phys_cms / u0_unit
    
    rho2_d, u2_d, P2_d, _ = rankine_hugoniot(rho1_d, u1_d, P1_d, GAMMA_EOS)
    
    U = np.zeros((24, N)) # Full state array (3 species x 8 fields)
    w_tanh = 4.0 # Tanh smoothing width (dimensionless)
    for i in range(N):
        f = 0.5 * (1.0 - np.tanh((x_coords_d[i] - 0.0) / w_tanh))
        
        rH_d = rho1_d*f + rho2_d*(1-f)
        uH_d = u1_d*f + u2_d*(1-f)
        pH_d = P1_d*f + P2_d*(1-f)

        # Neutrals (H)
        U[0,i] = rH_d
        U[1,i] = rH_d * uH_d
        U[2,i] = pH_d + rH_d*uH_d**2
        U[3,i] = pH_d
        U[4,i] = rH_d * x_coords_d[i] # L1
        U[5,i] = rH_d * 0.02 # alpha
        U[6,i] = rH_d * 0.0  # beta
        U[7,i] = rH_d*uH_d**3 + 3.0*uH_d*pH_d # M3 (Q=0 initially)

        # Protons (P)
        U[8,i]  = rH_d * 1e-4 # Same number density as electrons
        U[9,i]  = U[8,i] * uH_d
        U[10,i] = pH_d * 1e-4 + U[8,i]*uH_d**2
        U[11,i] = pH_d * 1e-4
        U[12,i] = U[8,i] * x_coords_d[i]
        U[13,i] = U[8,i] * 0.02
        U[14,i] = U[8,i] * 0.0
        U[15,i] = U[8,i]*uH_d**3 + 3.0*uH_d*(pH_d*1e-4)

        # Electrons (E)
        U[16,i] = rH_d * 1e-4 * M_E # Number density is same, but mass is M_E
        U[17,i] = U[16,i] * uH_d
        U[18,i] = pH_d * 1e-4 + U[16,i]*uH_d**2 # Pressure is n_e * T_e
        U[19,i] = pH_d * 1e-4
        U[20,i] = U[16,i] * x_coords_d[i]
        U[21,i] = U[16,i] * 0.02
        U[22,i] = U[16,i] * 0.0
        U[23,i] = U[16,i]*uH_d**3 + 3.0*uH_d*(pH_d*1e-4)
    
    tau_H = 1e-1; tau_p = 1e-1; tau_e = 1e-2 # Dimensionless relaxation times
    t_d = 0.0; t_end_d = 3.0; cfl = 0.3; nsteps = 0
    
    # Boundary inflow state (dimensionless)
    U_inflow_H = U[0:8, 0].copy()
    U_inflow_P = U[8:16, 0].copy()
    U_inflow_E = U[16:24, 0].copy()


    print("Running Low-Mach Python Simulation...")
    while t_d < t_end_d:
        smax = max_signal_speed_3(U, N)
        dt_d = min(cfl * dx_d / smax, t_end_d - t_d)
        if dt_d <= 0: break
        
        _species_step_transmissive(U, 0, dx_d, dt_d, tau_H, N) # Neutrals
        _species_step_transmissive(U, 8, dx_d, dt_d, tau_p, N) # Protons
        _species_step_transmissive(U, 16, dx_d, dt_d, tau_e, N) # Electrons
        
        couple_3_fluids(U, dt_d, N, u0_unit, T1_phys_k, n0_unit, rho0_mass_density)
        
        # Boundary conditions (ensure they are 2D slices)
        # Neutrals
        U[0:8, 0] = U_inflow_H # Pin first two cells (fluid primitives)
        U[0:8, 1] = U_inflow_H
        # Protons
        U[8:16, 0] = U_inflow_P
        U[8:16, 1] = U_inflow_P
        # Electrons
        U[16:24, 0] = U_inflow_E
        U[16:24, 1] = U_inflow_E

        # Extrapolate L1 gradient and higher moments from interior
        # This is for the `_species_step_transmissive` internal ghost cells
        for c in [0, 1]: # Apply to the two ghost cells
            # Neutrals
            U[4, c] = U[4, 2] # L1
            U[5, c] = U[5, 2] # alpha
            U[6, c] = U[6, 2] # beta
            U[7, c] = U[7, 2] # M3
            # Protons
            U[12, c] = U[12, 2]
            U[13, c] = U[13, 2]
            U[14, c] = U[14, 2]
            U[15, c] = U[15, 2]
            # Electrons
            U[20, c] = U[20, 2]
            U[21, c] = U[21, 2]
            U[22, c] = U[22, 2]
            U[23, c] = U[23, 2]

        # Outflow (copy last 3 cells from interior)
        U[:, N-2:N] = U[:, N-3:N-1] # This is a 2D slice copy
        
        t_d += dt_d
        nsteps += 1
        if nsteps % 1000 == 0: print(f"t={t_d:.4f} / {t_end_d:.4f}, steps={nsteps}")

    print(f"Finished in {nsteps} steps.")

    # --- Convert to physical units for plotting ---
    x_pc = x_coords_d * (L0_unit / CM_PER_PC)
    u_H_kms=(U[1,:]/U[0,:])*(u0_unit/1e5); u_p_kms=(U[9,:]/U[8,:])*(u0_unit/1e5); u_e_kms=(U[17,:]/U[16,:])*(u0_unit/1e5)
    r_H_phys=U[0,:]*n0_unit; r_p_phys=U[8,:]*n0_unit; r_e_phys=U[16,:]*n0_unit
    
    # Calculate physical temperatures
    T_H_phys = ((U[2,:] - U[0,:]*u_H_kms**2) / U[0,:]) * (P0_unit / (KB * n0_unit))
    T_p_phys = ((U[10,:] - U[8,:]*u_p_kms**2) / U[8,:]) * (P0_unit / (KB * n0_unit))
    T_e_phys = ((U[18,:] - U[16,:]*u_e_kms**2) / U[16,:]) * (P0_unit / (KB * n0_unit))

    # Calculate physical pressures
    P_H_phys = r_H_phys * KB * T_H_phys
    P_p_phys = r_p_phys * KB * T_p_phys
    P_e_phys = r_e_phys * KB * T_e_phys

    current_j_phys = (r_p_phys) * (u_p_kms - u_e_kms) * (E_CHARGE / 10.0) * 1e5 # A/cm^2

    # --- Plotting --- (3x2 Grid)
    fig, axs = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("Low-Mach Primordial Shock (Physical Units)", fontsize=16)

    # Density (Top-Left)
    axs[0,0].plot(x_pc, r_H_phys, label="H", color='k')
    axs[0,0].plot(x_pc, r_p_phys * 1e4, label="p (x1e4)", color='orange')
    axs[0,0].plot(x_pc, r_e_phys * (1/M_E * 1e4), label="e (x1e4)", color='cyan', linestyle='--')
    axs[0,0].set_ylabel("Num Density [cm^-3]")
    axs[0,0].legend(loc="lower right")
    axs[0,0].set_ylim(0.9e-3, 1.4e-3)
    axs[0,0].set_xlim(x_min_pc, x_max_pc)

    # Temperature (Top-Right)
    axs[0,1].plot(x_pc, T_H_phys, label="H", color='k')
    axs[0,1].plot(x_pc, T_p_phys, label="p", color='orange')
    axs[0,1].plot(x_pc, T_e_phys, label="e", color='cyan', linestyle='--')
    axs[0,1].set_ylabel("Temperature [K]")
    axs[0,1].legend(loc="upper right")
    axs[0,1].set_ylim(40, 150)
    axs[0,1].set_xlim(x_min_pc, x_max_pc)

    # Pressure (Middle-Left)
    axs[1,0].plot(x_pc, P_H_phys, label="H", color='k')
    axs[1,0].plot(x_pc, P_p_phys, label="p", color='orange')
    axs[1,0].plot(x_pc, P_e_phys, label="e", color='cyan', linestyle='--')
    axs[1,0].set_ylabel("Pressure [erg/cm^3]")
    axs[1,0].legend(loc="upper right")
    axs[1,0].set_ylim(5e-18, 1.5e-17)
    axs[1,0].set_xlim(x_min_pc, x_max_pc)

    # Velocity (Middle-Right)
    axs[1,1].plot(x_pc, u_H_kms, label="H", color='k')
    axs[1,1].plot(x_pc, u_p_kms, label="p", color='orange')
    axs[1,1].plot(x_pc, u_e_kms, label="e", color='cyan', linestyle='--')
    axs[1,1].set_ylabel("Velocity [km/s]")
    axs[1,1].legend(loc="upper right")
    axs[1,1].set_ylim(0.35, 0.55)
    axs[1,1].set_xlim(x_min_pc, x_max_pc)

    # Electric Current (Bottom-Left)
    axs[2,0].plot(x_pc, current_j_phys, label="Current J", color='red')
    axs[2,0].set_ylabel("Current [A/cm^2]")
    axs[2,0].set_xlabel("Position [pc]")
    axs[2,0].legend(loc="upper right")
    axs[2,0].set_ylim(-1e-22, 6e-22) 
    axs[2,0].set_xlim(x_min_pc, x_max_pc)

    # Remove Bottom-Right empty plot
    fig.delaxes(axs[2,1])
    
    plt.tight_layout()
    filename = os.path.join(FIG_DIR, "low_mach_shock_physical.png")
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

if __name__ == "__main__":
    t0 = time.time()
    run_low_mach_physical_plot()
    print(f"Total runtime: {time.time()-t0:.2f}s")
