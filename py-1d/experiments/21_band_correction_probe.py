"""
Experiment 21 (prototype / probe): approximate deconvolution of the
scheme's own numerical damping, calibrated online from the AMR defect.

Status: exploratory. See reference/notes_amr_semenov_germano.md,
section "Band correction". Usage:  python 21_band_correction_probe.py hll|muscl

1. Measure the one-step damping rate nu(k) per Fourier band from the
   full-coverage two-level AMR defect (dfmm.band_correction).
2. Build an ADM-style band correction C(k) = Q_N(exp(-nu dt)) within a
   wavelength band (hard cutoff or tapered toward lambda_min), funded from
   K first and (optionally) a capped share of the thermal pressure.
3. Iterate nu <- nu + nu_residual with the correction active in the
   coarse prediction (dynamic, self-consistent calibration).
4. A-posteriori test on held-out seeds: coarse-only runs vs filtered DNS,
   spectral ratio T(k) per wavelength band (in coarse cells).
"""
import numpy as np, time, sys
from dfmm.setups.wavepool import make_wave_pool_ic
from dfmm.integrate import run_to, coarse_grain
from dfmm.config import SimulationConfig
from dfmm.amr import TurbModel
from dfmm.band_correction import calibrate_dynamic, run_corrected, RunAborted, BandCorrector, measure_transfer
NC,R=256,8; NF=NC*R
cfg = SimulationConfig() if sys.argv[1]=="hll" else SimulationConfig(reconstruction='muscl',limiter='mc')
m=TurbModel(enabled=True,C_eps=2.9,f_K=0.05)
Uf0,_=make_wave_pool_ic(NF,u0=1.0,P0=0.1,K_max=16,seed=42); Uc0=coarse_grain(Uf0,R)
t0=time.time()
corrs={}
for name,kw in (("hard8 K-only",dict(lambda_min=8,heat_fraction=0.0)),("taper8-32 K-only",dict(lambda_min=8,lambda_full=32,heat_fraction=0.0)),("taper8-32 heat5%",dict(lambda_min=8,lambda_full=32,heat_fraction=0.05))):
    c,h=calibrate_dynamic(Uc0,Uf0,R,cfg,0.3,m,n_iter=3,**kw)
    corrs[name]=(c,h)
    print(name, f"({time.time()-t0:.0f}s)")
    for it,(kc,nu,nres) in enumerate(h):
        sel=(NC/kc>=8)
        print(f"  iter {it}: max|nu_res|/nu in band = {np.max(np.abs(nres[sel])/np.maximum(nu[sel],1e-9)):.2f}; nu_res(lambda~57,24,11) = {nres[3]:.2f} {nres[5]:.2f} {nres[7]:.2f}", flush=True)
ts=[0.05,0.1,0.2,0.3]
def spec(U): u=U[1]/U[0]; return np.abs(np.fft.rfft(u)/u.size)**2
k=np.arange(NC//2+1); lam=NC/np.maximum(k,1)
for seed in (77,101):
    Uf0,_=make_wave_pool_ic(NF,u0=1.0,P0=0.1,K_max=16,seed=seed); Uc0=coarse_grain(Uf0,R)
    dns=[coarse_grain(U,R) for _,U in run_to(Uf0,t_end=ts[-1],save_times=ts,cfg=cfg)[0][1:]]
    Ed=np.mean([spec(U) for U in dns],0)
    print("seed",seed)
    for name,corr in [("plain",None)]+[(n,c[0]) for n,c in corrs.items()]:
        try: sn,th=run_corrected(Uc0,ts[-1],ts,cfg,m if corr else TurbModel(),corr)
        except RunAborted as e: print(f"  {name:18s} ABORTED {e}"); continue
        T=np.mean([spec(U) for _,U,_ in sn[1:]],0)/Ed
        s=" ".join(f"T[{lo}-{hi}]={np.mean(T[(lam>=lo)&(lam<hi)&(k>0)]):.2f}" for lo,hi in [(64,256),(32,64),(16,32),(8,16),(4,8)])
        ke=np.mean([np.mean(0.5*U[1]**2/U[0]) for _,U,_ in sn[1:]])/np.mean([np.mean(0.5*U[1]**2/U[0]) for U in dns])
        print(f"  {name:18s} {s} KE/DNS={ke:.3f} heat debit={th:.2e}", flush=True)
