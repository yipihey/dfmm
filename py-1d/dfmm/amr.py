"""
Two-level Berger-Oliger AMR for the 1D Cholesky moment scheme, with an
optional Semenov-style subgrid turbulent-energy field K and online
(in-situ) closure calibration from the coarse-fine defect.

Structure of one coarse step (Enzo-like):

    1. advance the coarse level by dt_c            -> U_c^{n+1}, coarse face fluxes
    2. advance the fine patch r times by dt_c/r    (ghosts: time-interpolated,
                                                    prolonged coarse state),
                                                    accumulating fine fluxes on
                                                    the two patch-boundary faces
    3. record the DEFECT  eps = U_c^{n+1} - R(U_f^{n+1})  on covered cells
       (because U_c^n = R(U_f^n) there after the previous projection, this
       is exactly the "reset" closure-error protocol of experiment 11, with
       interval = one coarse step, obtained for free)
    4. project: U_c <- R(U_f) on covered cells
    5. reflux: correct the two uncovered coarse neighbours with
       (dt_c F_c - sum dt_f F_f) / dx_c

Subgrid turbulent energy K (energy per volume of unresolved x-motions,
K = rho sigma_t^2 / 2, contributing P_t = 2K to P_xx in 1D). K is a
*partition* of P_xx, not extra energy: P_xx = P_th,xx + 2K. Per level:

    d_t K + d_x(u K) = -2 K d_x u                       (1D xx compression, gamma=3)
                       + f_K * max(D_num, 0)            (Semenov 2025 source: KE lost
                                                         to numerical dissipation,
                                                         measured with an advected
                                                         entropy, Semenov 2022)
                       - C_eps rho sigma_t^3 / Delta    (cascade decay into heat)

BGK isotropization acts only on the thermal part P_th,xx, so K stays an
anisotropic (xx-only) pressure until it decays. With K == 0 the step is
identical to `cholesky.hll_step`.

Germano consistency at projection (energy-conserving restriction):

    K_c  <-  <K_f> + K_res,     K_res = 1/2 ( <rho u^2> - <rho u>^2 / <rho> ) >= 0

which leaves the thermal pressure restricting as a plain average,
P_th,c = <P_th,f>. The pre-projection mismatch dK = K_c^pred - K_c^true is
the Germano defect; it calibrates C_eps dynamically (scale similarity:
the same C_eps on both levels).
"""
from dataclasses import dataclass, field
import numpy as np

from .config import SimulationConfig
from .schemes._common import (Workspace, IDX_RHO, IDX_MOM, IDX_EXX, IDX_PP,
                              IDX_BETA, IDX_M3)
from .schemes.cholesky import hll_step, max_signal_speed

_BIG_TAU = 1e300  # disables the BGK block inside hll_step (decay = 1)


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def restrict(A, r):
    """Box-average the last axis by factor r (all fields are additive)."""
    sh = A.shape[:-1] + (A.shape[-1] // r, r)
    return A.reshape(sh).mean(axis=-1)


def resolved_subcell_ke(U_f, r):
    """K_res = 1/2(<rho u^2> - <rho u>^2/<rho>) per coarse cell (>= 0)."""
    rho, mom = U_f[IDX_RHO], U_f[IDX_MOM]
    Rrho = restrict(rho, r)
    Rmom = restrict(mom, r)
    Rke2 = restrict(mom * mom / rho, r)
    return np.maximum(0.5 * (Rke2 - Rmom * Rmom / Rrho), 0.0)


def total_energy(U):
    """Domain-summed total energy density 1/2(E_xx + 2 P_perp) per cell."""
    return 0.5 * (U[IDX_EXX] + 2.0 * U[IDX_PP])


def _minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def prolong_cells(Uc_ext, r, qs):
    """Conservative piecewise-linear (minmod) prolongation.

    Uc_ext : (nf, 3) coarse values at [j-1, j, j+1]
    qs     : sub-cell indices (0..r-1) of coarse cell j to return
    returns (nf, len(qs))
    """
    s = _minmod(Uc_ext[:, 1] - Uc_ext[:, 0], Uc_ext[:, 2] - Uc_ext[:, 1])
    xi = (np.asarray(qs) + 0.5) / r - 0.5
    return Uc_ext[:, 1:2] + s[:, None] * xi[None, :]


# ----------------------------------------------------------------------
# One level step with optional K partition and dissipation measurement
# ----------------------------------------------------------------------
@dataclass
class TurbModel:
    enabled: bool = False
    C_eps: float = 1.0       # decay coefficient (1D: K' = -C_eps rho sigma^3/Delta)
    f_K: float = 1.0         # fraction of numerical dissipation routed into K
    K_max_frac: float = 0.99 # realizability: 2K <= K_max_frac * P_xx


def level_step(Ug, Kg, dx, dt, tau, ng, cfg, model, Delta, ws):
    """Advance a ghost-padded level by dt.

    Ug : (8, N+2ng) padded state with ghosts already filled
    Kg : (N+2ng,) padded K density (ghosts filled), or None
    Returns (U_new_interior, K_new_interior, F, FK, D)
        F  : (8, N+1) hydro face fluxes (faces of the interior)
        FK : (N+1,) K face fluxes
        D  : (N,) measured numerical dissipation rate (energy/vol/time)
    """
    N = Ug.shape[1] - 2 * ng
    sl = slice(ng, ng + N)
    rho0 = Ug[IDX_RHO]
    u0 = Ug[IDX_MOM] / rho0
    Pxx0 = Ug[IDX_EXX] - rho0 * u0 * u0
    Pp0 = Ug[IDX_PP]

    # Hydro + Liouville sources, BGK off (we apply BGK ourselves below)
    Unew = hll_step(Ug, dx, dt, _BIG_TAU, ng, cfg.rho_floor, cfg.alpha_floor,
                    cfg.realizability_headroom, ws.Unew, ws.Fleft)
    F = ws.Fleft[:, ng:ng + N + 1].copy()
    U = Unew[:, sl].copy()

    # Passive upwind transport (by the HLL mass flux) of specific
    # quantities: k = K/rho, s_xx = P_xx/rho^3, s_p = P_perp/rho.
    Frho = F[IDX_RHO]
    def advect(spec_g):
        up = np.where(Frho > 0, spec_g[ng - 1:ng + N], spec_g[ng:ng + N + 1])
        Fs = Frho * up
        dens = rho0[sl] * spec_g[sl] - dt / dx * (Fs[1:] - Fs[:-1])
        return dens, Fs

    rho = U[IDX_RHO]
    u = U[IDX_MOM] / rho
    Pxx = U[IDX_EXX] - rho * u * u
    Pp = U[IDX_PP]

    # Numerical dissipation: actual internal energy minus the reversible
    # (entropy-advected) prediction. 1D: P_xx ~ rho^3, P_perp ~ rho.
    sxx_d, _ = advect(Pxx0 / rho0**3)
    sp_d, _ = advect(Pp0 / rho0)
    Pxx_rev = rho**2 * sxx_d          # rho^3 * (rho s)/rho
    Pp_rev = sp_d                     # rho * (rho s)/rho
    D = 0.5 * ((Pxx - Pxx_rev) + 2.0 * (Pp - Pp_rev)) / dt

    if model.enabled and Kg is not None:
        K, FK = advect(Kg / rho0)
        dudx = (u0[ng + 1:ng + N + 1] - u0[ng - 1:ng + N - 1]) / (2.0 * dx)
        K = K * np.exp(-2.0 * dudx * dt)                # compression (gamma=3)
        K = K + model.f_K * np.maximum(D, 0.0) * dt     # Semenov-2025 source
        # exact decay of K' = -c K^{3/2},  c = C_eps 2^{3/2} rho^{-1/2} / Delta
        c = model.C_eps * 2.0**1.5 / (np.sqrt(rho) * Delta)
        K = np.where(K > 0, 1.0 / (1.0 / np.sqrt(np.maximum(K, 1e-300))
                                   + 0.5 * c * dt)**2, 0.0)
        K = np.clip(K, 0.0, 0.5 * model.K_max_frac * Pxx)
    else:
        K = np.zeros(N); FK = np.zeros(N + 1)

    # BGK on the thermal part only (K == 0 reproduces hll_step exactly)
    decay = np.exp(-dt / tau)
    beta = U[IDX_BETA] / rho
    Q = U[IDX_M3] - rho * u**3 - 3.0 * u * Pxx
    Pth = Pxx - 2.0 * K
    P_iso = (Pth + 2.0 * Pp) / 3.0
    Pth_new = P_iso + (Pth - P_iso) * decay
    Pp_new = P_iso + (Pp - P_iso) * decay
    Pxx_new = Pth_new + 2.0 * K
    b_new = beta * decay
    bmax = cfg.realizability_headroom * np.sqrt(np.maximum(Pxx_new, cfg.rho_floor)
                                                / np.maximum(rho, cfg.rho_floor))
    b_new = np.clip(b_new, -bmax, bmax)
    Q_new = Q * decay
    U[IDX_EXX] = rho * u * u + Pxx_new
    U[IDX_PP] = Pp_new
    U[IDX_BETA] = rho * b_new
    U[IDX_M3] = rho * u**3 + 3.0 * u * Pxx_new + Q_new
    return U, K, F, FK, D


def pad_periodic(A, ng):
    return np.concatenate([A[..., -ng:], A, A[..., :ng]], axis=-1)


def coarse_only_step(U, K, dx, dt, tau, cfg, model, ws=None):
    """Single-level periodic step (used for coarse-only and shadow runs)."""
    ng = cfg.n_ghost
    Ug = np.ascontiguousarray(pad_periodic(U, ng))
    if ws is None:
        ws = Workspace.for_padded_state(Ug)
    Kg = pad_periodic(K, ng) if K is not None else None
    Un, Kn, F, FK, D = level_step(Ug, Kg, dx, dt, tau, ng, cfg, model, dx, ws)
    return Un, Kn, D


def run_single(U, t_end, save_times, cfg, model, K=None, dt_fixed=None):
    """Single-level periodic run with optional K; returns snapshots
    [(t, U, K)]."""
    N = U.shape[1]; dx = 1.0 / N
    K = np.zeros(N) if K is None else K.copy()
    U = U.copy(); t = 0.0
    snaps = [(0.0, U.copy(), K.copy())]
    ws = Workspace.for_padded_state(np.ascontiguousarray(pad_periodic(U, cfg.n_ghost)))
    si = 0
    while si < len(save_times):
        dt = cfg.cfl * dx / max_signal_speed(U) if dt_fixed is None else dt_fixed
        dt = min(dt, save_times[si] - t)
        if dt > 1e-14:
            U, K, _ = coarse_only_step(U, K, dx, dt, cfg.tau, cfg, model, ws)
            t += dt
        if t >= save_times[si] - 1e-12:
            snaps.append((t, U.copy(), K.copy())); si += 1
    return snaps


# ----------------------------------------------------------------------
# Two-level AMR hierarchy (static fine patch)
# ----------------------------------------------------------------------
@dataclass
class TwoLevelAMR:
    Uc: np.ndarray            # (8, Nc) coarse state, periodic domain [0,1)
    Uf: np.ndarray            # (8, (i1-i0)*r) fine patch state
    i0: int                   # first covered coarse cell
    i1: int                   # one past last covered coarse cell
    r: int
    cfg: SimulationConfig = field(default_factory=SimulationConfig)
    model: TurbModel = field(default_factory=TurbModel)
    dynamic_C_eps: bool = False
    dynamic_f_K: bool = False  # also fit the source fraction (2-parameter Germano fit)
    dyn_memory: float = 0.98  # EMA factor for the dynamic C_eps regression
    reflux: bool = True
    t: float = 0.0

    def __post_init__(self):
        self.Nc = self.Uc.shape[1]
        self.dxc = 1.0 / self.Nc
        self.dxf = self.dxc / self.r
        self.full = (self.i0 == 0 and self.i1 == self.Nc)
        assert self.Uf.shape[1] == (self.i1 - self.i0) * self.r
        if not self.full:
            assert self.i0 >= 2 and self.i1 <= self.Nc - 2
        self.Kc = np.zeros(self.Nc)
        self.Kf = np.zeros(self.Uf.shape[1])
        ng = self.cfg.n_ghost
        self.wsc = Workspace.for_padded_state(
            np.ascontiguousarray(pad_periodic(self.Uc, ng)))
        self.wsf = Workspace.for_padded_state(
            np.zeros((8, self.Uf.shape[1] + 2 * ng)))
        self._num = 0.0; self._den = 0.0
        self._A = np.zeros((2, 2)); self._y = np.zeros(2)
        self.C_eps_history = []
        self.f_K_history = []

    # -------- fine ghosts from time-interpolated, prolonged coarse ----
    def _fine_ghosts(self, Uc_t, Kc_t):
        ng, r = self.cfg.n_ghost, self.r
        Nc = self.Nc
        ext = lambda j: np.stack([Uc_t[:, (j - 1) % Nc], Uc_t[:, j],
                                  Uc_t[:, (j + 1) % Nc]], axis=1)
        qL = np.arange(r - ng, r)      # rightmost sub-cells of cell i0-1
        qR = np.arange(0, ng)          # leftmost sub-cells of cell i1
        gL = prolong_cells(ext(self.i0 - 1), r, qL)
        gR = prolong_cells(ext(self.i1), r, qR)
        kL = np.full(ng, Kc_t[self.i0 - 1]); kR = np.full(ng, Kc_t[self.i1])
        return gL, gR, kL, kR

    def step(self, dt_c=None, t_stop=None):
        cfg, r, ng = self.cfg, self.r, self.cfg.n_ghost
        smax_c = max_signal_speed(self.Uc)
        smax_f = max_signal_speed(self.Uf)
        dt = min(cfg.cfl * self.dxc / smax_c, r * cfg.cfl * self.dxf / smax_f)
        if dt_c is not None: dt = min(dt, dt_c)
        if t_stop is not None: dt = min(dt, t_stop - self.t)
        dtf = dt / r
        Uc_n, Kc_n = self.Uc.copy(), self.Kc.copy()

        # 1. coarse advance
        Ucg = np.ascontiguousarray(pad_periodic(self.Uc, ng))
        Kcg = pad_periodic(self.Kc, ng)
        Uc1, Kc1, Fc, FKc, Dc = level_step(Ucg, Kcg, self.dxc, dt, cfg.tau, ng,
                                           cfg, self.model, self.dxc, self.wsc)
        # coarse decay sensitivity dK/dC_eps (for dynamic C_eps), per cell
        rho1 = Uc1[IDX_RHO]
        Kpre = np.maximum(Kc1, 0.0)
        bsens = -(2.0**1.5) * Kpre**1.5 / (np.sqrt(rho1) * self.dxc) * dt

        # 2. fine subcycles
        Fsum = np.zeros((8, 2)); FKsum = np.zeros(2)
        Df_acc = np.zeros(self.Uf.shape[1])
        for m in range(r):
            if self.full:
                Ufg = np.ascontiguousarray(pad_periodic(self.Uf, ng))
                Kfg = pad_periodic(self.Kf, ng)
            else:
                th = m / r
                Uc_t = (1 - th) * Uc_n + th * Uc1
                Kc_t = (1 - th) * Kc_n + th * Kc1
                gL, gR, kL, kR = self._fine_ghosts(Uc_t, Kc_t)
                Ufg = np.ascontiguousarray(np.concatenate([gL, self.Uf, gR], axis=1))
                Kfg = np.concatenate([kL, self.Kf, kR])
            self.Uf, self.Kf, Ff, FKf, Df = level_step(
                Ufg, Kfg, self.dxf, dtf, cfg.tau, ng, cfg, self.model,
                self.dxf, self.wsf)
            Fsum[:, 0] += dtf * Ff[:, 0];  Fsum[:, 1] += dtf * Ff[:, -1]
            FKsum[0] += dtf * FKf[0];      FKsum[1] += dtf * FKf[-1]
            Df_acc += Df * dtf

        # 3. defect on covered cells (pre-projection)
        cov = slice(self.i0, self.i1)
        RUf = restrict(self.Uf, r)
        Kres = resolved_subcell_ke(self.Uf, r)
        Ktrue = restrict(self.Kf, r) + Kres
        rec = dict(t=self.t, dt=dt,
                   eps=Uc1[:, cov] - RUf,
                   Uc_n=Uc_n[:, cov], Kc_n=Kc_n[cov],
                   dudx_n=self._dudx(Uc_n)[cov],
                   Kc_pred=Kc1[cov].copy(), Ktrue=Ktrue, Kres=Kres,
                   Dc=Dc[cov] * dt, Df=restrict(Df_acc, r))

        # dynamic C_eps (Germano-style). Linearised in the parameters p:
        #   K_pred(p) = K_pred(p_used) + J (p - p_used),   d = K_pred(p_used) - K_true
        # Weighted LS over an exponentially-forgotten history of samples,
        # each measured at its own p_used:
        #   (sum J^T J) p* = sum J^T (J p_used - d)
        # 1 parameter: p = C_eps, J = dK/dC (decay sensitivity).
        # 2 parameters: p = (C_eps, f_K), J = (dK/dC, max(D_c,0) dt).
        if self.model.enabled and self.dynamic_C_eps:
            b = bsens[cov]; d = Kc1[cov] - Ktrue
            lam = self.dyn_memory
            Cu = self.model.C_eps
            if not self.dynamic_f_K:
                self._num = lam * self._num + np.sum(b * b * Cu - b * d)
                self._den = lam * self._den + np.sum(b * b)
                if self._den > 0:
                    self.model.C_eps = float(np.clip(self._num / self._den, 0.0, 50.0))
            else:
                J = np.stack([b, np.maximum(Dc[cov], 0.0) * dt], axis=1)
                pu = np.array([Cu, self.model.f_K])
                self._A = lam * self._A + J.T @ J
                self._y = lam * self._y + J.T @ (J @ pu - d)
                try:
                    p = np.linalg.solve(self._A + 1e-30 * np.eye(2), self._y)
                    self.model.C_eps = float(np.clip(p[0], 0.0, 50.0))
                    self.model.f_K = float(np.clip(p[1], 0.0, 1.0))
                except np.linalg.LinAlgError:
                    pass
        self.f_K_history.append((self.t, self.model.f_K))
        self.C_eps_history.append((self.t, self.model.C_eps))

        # 4. project
        Uc1[:, cov] = RUf
        Kc1[cov] = Ktrue
        # 5. reflux on the two uncovered neighbours
        if self.reflux and not self.full:
            jl, jr = self.i0 - 1, self.i1
            Uc1[:, jl] += (dt * Fc[:, self.i0] - Fsum[:, 0]) / self.dxc
            Uc1[:, jr] += (Fsum[:, 1] - dt * Fc[:, self.i1]) / self.dxc
            if self.model.enabled:
                Kc1[jl] += (dt * FKc[self.i0] - FKsum[0]) / self.dxc
                Kc1[jr] += (FKsum[1] - dt * FKc[self.i1]) / self.dxc
        self.Uc, self.Kc = Uc1, np.maximum(Kc1, 0.0)
        self.t += dt
        return rec

    def _dudx(self, U):
        u = U[IDX_MOM] / U[IDX_RHO]
        return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * self.dxc)

    def composite_totals(self):
        """Conserved totals over the composite grid (fine where covered)."""
        mask = np.ones(self.Nc, bool); mask[self.i0:self.i1] = False
        out = {}
        for name, k in (("mass", IDX_RHO), ("mom", IDX_MOM)):
            out[name] = (self.Uc[k, mask].sum() * self.dxc
                         + self.Uf[k].sum() * self.dxf)
        out["energy"] = (total_energy(self.Uc)[mask].sum() * self.dxc
                         + total_energy(self.Uf).sum() * self.dxf)
        return out
