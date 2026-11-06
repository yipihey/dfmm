"""
Approximate deconvolution of a scheme's OWN numerical dissipation, with
the energy funded from the subgrid reservoir K.

Stolz & Adams' approximate deconvolution model (ADM) recovers the
unfiltered field from a filtered one with the truncated van Cittert
series   u* = Q_N ubar,   Q_N = sum_{nu=0}^{N} (I - G)^nu  ~  G^{-1}.
Here G is not an explicit LES filter but the damping the numerical scheme
itself applies in one step, measured online in Fourier bands from the
two-level AMR defect:

    g(k) = < Re( u_c^{n+1}(k) conj(R u_f^{n+1}(k)) ) > / < |R u_f^{n+1}(k)|^2 >

i.e. the regression coefficient of the coarse one-step prediction on the
restricted fine result, both started from the same projected state. It is
stored as a rate nu(k) = -ln g / dt so it can be applied at any time step.
The correction multiplies the velocity spectrum by

    C(k) = Q_N(g) = (1 - (1 - g)^(N+1)) / g        (bounded by N + 1)

within a band of wavelengths >= lambda_min cells (scales the grid can still
carry). Momentum is conserved exactly. The kinetic energy that the
correction adds is debited from K first and only then from the thermal
pressure, so K acts as the reservoir that numerical dissipation fills
(Semenov source) and the band correction empties (backscatter).
"""
import numpy as np

from .config import SimulationConfig
from .schemes._common import Workspace, IDX_RHO, IDX_MOM, IDX_EXX, IDX_M3
from .schemes.cholesky import max_signal_speed
from .amr import (TwoLevelAMR, TurbModel, coarse_only_step, pad_periodic,
                  restrict)


def band_edges(n_cells):
    """Octave-ish log bands in k = 1 .. n_cells/2."""
    e = np.unique(np.round(np.logspace(0, np.log10(n_cells // 2 + 1), 14)).astype(int))
    return e


class TransferMeter:
    """Accumulates the one-step coarse transfer function per band."""

    def __init__(self, n_cells):
        self.N = n_cells
        self.edges = band_edges(n_cells)
        nb = len(self.edges) - 1
        self.num = np.zeros(nb); self.den = np.zeros(nb); self.dts = []

    def add(self, U_pred, U_true, dt):
        uc = np.fft.rfft(U_pred[IDX_MOM] / U_pred[IDX_RHO])
        uf = np.fft.rfft(U_true[IDX_MOM] / U_true[IDX_RHO])
        cr = np.real(uc * np.conj(uf)); pf = np.abs(uf)**2
        for b in range(len(self.edges) - 1):
            sl = slice(self.edges[b], self.edges[b + 1])
            self.num[b] += cr[sl].sum(); self.den[b] += pf[sl].sum()
        self.dts.append(dt)

    def rate(self):
        """Per-band damping rate nu_b = -ln(g_b)/<dt>, and band centres."""
        g = self.num / np.maximum(self.den, 1e-300)
        dt = np.mean(self.dts)
        nu = -np.log(np.clip(g, 1e-6, None)) / dt
        kc = np.sqrt(self.edges[:-1] * np.maximum(self.edges[1:] - 1, self.edges[:-1]))
        return kc, nu, g


def measure_transfer(Uc0, Uf0, r, cfg, t_end, model=None, dynamic=False):
    """Full-coverage AMR run; returns (k_centres, nu_b, g_b, amr)."""
    Nc = Uc0.shape[1]
    amr = TwoLevelAMR(Uc=Uc0.copy(), Uf=Uf0.copy(), i0=0, i1=Nc, r=r, cfg=cfg,
                      model=model if model is not None else TurbModel(),
                      dynamic_C_eps=dynamic, dynamic_f_K=dynamic)
    meter = TransferMeter(Nc)
    while amr.t < t_end - 1e-12:
        rec = amr.step(t_stop=t_end)
        meter.add(rec["eps"] + amr.Uc, amr.Uc, rec["dt"])  # Uc == R(Uf) now
    kc, nu, g = meter.rate()
    return kc, nu, g, amr


class BandCorrector:
    """ADM-style deconvolution of the measured numerical damping."""

    def __init__(self, kc, nu, n_cells, N=3, lambda_min=8.0, lambda_max=None,
                 shock_mask=None, lambda_full=None, strength=1.0):
        self.N_terms = N
        k = np.arange(n_cells // 2 + 1)
        # interpolate the band rates in log k; no correction outside band
        nu_k = np.interp(np.log(np.maximum(k, 1)), np.log(kc), np.maximum(nu, 0.0))
        lam = n_cells / np.maximum(k, 1e-12)
        band = (k >= 1) & (lam >= lambda_min)
        if lambda_max is not None:
            band &= lam <= lambda_max
        # taper (ADM relaxation analogue): full correction for lambda >=
        # lambda_full, smoothly -> 0 at lambda_min, so the scheme's own
        # dissipation still drains the cutoff region
        w = np.ones_like(lam)
        if lambda_full is not None and lambda_full > lambda_min:
            x = np.clip((np.log(lam) - np.log(lambda_min))
                        / (np.log(lambda_full) - np.log(lambda_min)), 0, 1)
            w = x * x * (3 - 2 * x)
        self.nu_k = strength * np.where(band, w * nu_k, 0.0)
        self.shock_mask = shock_mask   # None or threshold on -du/dx*dx/cs

    def factor(self, dt):
        g = np.exp(-self.nu_k * dt)
        if self.N_terms is None:
            return 1.0 / g
        return (1.0 - (1.0 - g)**(self.N_terms + 1)) / g

    def apply(self, U, K, dt, cfg):
        """Correct velocity in-band; fund KE from K, then thermal.
        Returns (U, K, thermal_debit_total)."""
        rho = U[IDX_RHO]; mom = U[IDX_MOM]; u = mom / rho
        Pxx = U[IDX_EXX] - rho * u * u
        Q = U[IDX_M3] - rho * u**3 - 3 * u * Pxx
        uh = np.fft.rfft(u)
        du = np.fft.irfft(uh * (self.factor(dt) - 1.0), n=u.size)
        if self.shock_mask is not None:
            dx = 1.0 / u.size
            cs = np.sqrt(np.maximum(Pxx, 1e-30) / rho)
            dudx = (np.roll(u, -1) - np.roll(u, 1)) / 2.0
            shock = (-dudx / cs) > self.shock_mask
            shock = shock | np.roll(shock, 1) | np.roll(shock, -1)
            du = np.where(shock, 0.0, du)
        u_new = u + du
        mom_new = rho * u_new
        mom_new += (mom.sum() - mom_new.sum()) / u.size     # exact momentum
        u_new = mom_new / rho
        dKE = 0.5 * rho * (u_new**2 - u**2)
        # energy: debit K first, thermal second; cap so P_th stays >= 50%
        Pth = Pxx - 2 * K
        from_K = np.minimum(np.maximum(dKE, 0.0), K)
        from_th = np.maximum(dKE, 0.0) - from_K
        cap = 0.25 * Pth
        over = from_th > cap
        if np.any(over):
            # scale back the correction in cells that would overdraw heat
            s = np.where(over, (from_K + cap) / np.maximum(dKE, 1e-300), 1.0)
            u_new = u + s * (u_new - u)
            mom_new = rho * u_new
            mom_new += (mom.sum() - mom_new.sum()) / u.size
            u_new = mom_new / rho
            dKE = 0.5 * rho * (u_new**2 - u**2)
            from_K = np.minimum(np.maximum(dKE, 0.0), K)
            from_th = np.maximum(dKE, 0.0) - from_K
        K_new = K - from_K + np.maximum(-dKE, 0.0)   # KE removed -> back to K
        Pxx_new = Pxx - 2 * dKE
        U = U.copy()
        U[IDX_MOM] = rho * u_new
        U[IDX_EXX] = rho * u_new**2 + Pxx_new
        U[IDX_M3] = rho * u_new**3 + 3 * u_new * Pxx_new + Q
        K_new = np.clip(K_new, 0.0, 0.5 * 0.99 * Pxx_new)
        return U, K_new, float(from_th.sum())


def run_corrected(U, t_end, save_times, cfg, model, corrector=None, K=None):
    """Single-level periodic run with K model and optional band correction.
    Returns snapshots [(t, U, K)] and the cumulative thermal debit."""
    N = U.shape[1]; dx = 1.0 / N
    K = np.zeros(N) if K is None else K.copy()
    U = U.copy(); t = 0.0; si = 0; th_debit = 0.0
    snaps = [(0.0, U.copy(), K.copy())]
    ws = Workspace.for_padded_state(np.ascontiguousarray(pad_periodic(U, cfg.n_ghost)),
                                    reconstruction=cfg.reconstruction)
    while si < len(save_times):
        dt = min(cfg.cfl * dx / max_signal_speed(U), save_times[si] - t)
        if dt > 1e-14:
            U, K, _ = coarse_only_step(U, K, dx, dt, cfg.tau, cfg, model, ws)
            if corrector is not None:
                U, K, d = corrector.apply(U, K, dt, cfg)
                th_debit += d * dx
            t += dt
        if t >= save_times[si] - 1e-12:
            snaps.append((t, U.copy(), K.copy())); si += 1
    return snaps, th_debit
