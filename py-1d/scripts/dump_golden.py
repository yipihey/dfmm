"""
Dump py-1d Tier-A regression goldens to HDF5.

This is the single Python script the dfmm-2d Julia driver
(`scripts/make_goldens.jl`) shells out to in order to produce the
six regression-target HDF5 files under `reference/golden/`.

The script is intentionally tiny: it does *no* parameter inference, *no*
default value drift, *no* implicit IO state. All run parameters come in
as a single JSON dict passed on argv (keeps the Julia side as the
single source of truth for default IC parameters; the Julia setup
factories in `src/setups.jl` and the JSON we pass here are kept in
lockstep). This script also is the only edit Track B is permitted to
make under `py-1d/`; flag it explicitly in any handoff.

Schema written (see `reference/golden/SCHEMA.md` for the canonical
description):

    /params              attributes: every numeric/string entry from
                          the CLI JSON (N, t_end, tau, ...). Plus
                          `dfmm_git_sha` and `target` for provenance.
    /grid/x              [N] cell centers
    /timesteps           [n_snaps + 1] times of saved snapshots
    /fields/<name>       [n_snaps + 1, N] one entry per primitive
    /U                   [n_snaps + 1, n_fields, N] full conserved state
                          for round-trip / validation. n_fields = 8 for
                          single-fluid, 16 for two-fluid.

Usage (from the dfmm-2d repo root):

    py-1d/.venv/bin/python py-1d/scripts/dump_golden.py \\
        --target sod \\
        --out reference/golden/sod.h5 \\
        --params '{"N": 400, "t_end": 0.2, "tau": 1e-3, "cfl": 0.3,
                   "sigma_x0": 0.02, "bc": "transmissive"}'

`--target` is one of:
    sod, cold_sinusoid, steady_shock, wavepool, dust_in_gas, eion
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import h5py

# Make sure we resolve `dfmm` (the py-1d package) regardless of CWD.
HERE = Path(__file__).resolve().parent
PY1D_ROOT = HERE.parent
sys.path.insert(0, str(PY1D_ROOT))

import dfmm  # noqa: E402
from dfmm.setups.sod import make_sod_ic  # noqa: E402
from dfmm.setups.sine import make_sine_ic  # noqa: E402
from dfmm.setups.shock import (run_steady_shock as run_steady_shock_pyimpl,  # noqa: E402
                                rankine_hugoniot, GAMMA as SHOCK_GAMMA,
                                build_state_from_primitives,
                                step_inflow_outflow)
from dfmm.setups.wavepool import make_wave_pool_ic  # noqa: E402
from dfmm.integrate import run_to  # noqa: E402
from dfmm.schemes.cholesky import max_signal_speed  # noqa: E402
from dfmm.schemes import two_fluid as tf  # noqa: E402
from dfmm.diagnostics import extract_diagnostics  # noqa: E402


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _attach_params(g: "h5py.Group", params: dict, target: str, py1d_root: Path):
    g.attrs["target"] = target
    g.attrs["dfmm_git_sha"] = _git_sha(py1d_root.parent)
    g.attrs["py1d_dfmm_version"] = getattr(dfmm, "__version__", "unknown")
    for k, v in params.items():
        if isinstance(v, list):
            g.attrs[k] = np.asarray(v)
        elif v is None:
            g.attrs[k] = "None"
        else:
            g.attrs[k] = v


# ---------------------------------------------------------------------
#  Single-fluid drivers
# ---------------------------------------------------------------------


def _run_single_fluid(make_ic, ic_kwargs, run_kwargs, params):
    """Build IC + integrate to t_end with the standard 8-field driver."""
    U0, x = make_ic(**ic_kwargs)
    snaps, ns = run_to(U0, **run_kwargs)
    times = np.array([t for t, _ in snaps], dtype=np.float64)
    U_arr = np.stack([U for _, U in snaps], axis=0)  # [n_snap+1, 8, N]
    return x, times, U_arr, ns


def dump_sod(params: dict, out_path: Path):
    N = params["N"]; t_end = params["t_end"]; tau = params["tau"]
    cfl = params.get("cfl", 0.3); sigma_x0 = params.get("sigma_x0", 0.02)
    bc = params.get("bc", "transmissive")
    save_times = list(params.get("save_times", [t_end]))
    x, times, U, ns = _run_single_fluid(
        make_sod_ic, dict(N=N, sigma_x0=sigma_x0),
        dict(t_end=t_end, save_times=save_times, tau=tau, cfl=cfl, bc=bc),
        params,
    )
    return _write_single_fluid(out_path, "sod", x, times, U, ns, params)


def dump_cold_sinusoid(params: dict, out_path: Path):
    N = params["N"]; t_end = params["t_end"]; tau = params["tau"]
    cfl = params.get("cfl", 0.3); sigma_x0 = params.get("sigma_x0", 0.02)
    A = params.get("A", 1.0); T0 = params.get("T0", 1e-3)
    rho0 = params.get("rho0", 1.0)
    bc = params.get("bc", "periodic")
    save_times = list(params.get("save_times", [t_end]))
    x, times, U, ns = _run_single_fluid(
        make_sine_ic, dict(N=N, A=A, T0=T0, rho0=rho0, sigma_x0=sigma_x0),
        dict(t_end=t_end, save_times=save_times, tau=tau, cfl=cfl, bc=bc),
        params,
    )
    return _write_single_fluid(out_path, "cold_sinusoid", x, times, U, ns, params)


def dump_wavepool(params: dict, out_path: Path):
    N = params["N"]; t_end = params["t_end"]; tau = params["tau"]
    cfl = params.get("cfl", 0.3); sigma_x0 = params.get("sigma_x0", 0.02)
    u0 = params.get("u0", 1.0); P0 = params.get("P0", 0.1)
    K_max = params.get("K_max", 16); alpha_pl = params.get("alpha_pl", 5.0/6.0)
    rho0 = params.get("rho0", 1.0); seed = params.get("seed", 42)
    bc = params.get("bc", "periodic")
    save_times = list(params["save_times"])
    x, times, U, ns = _run_single_fluid(
        make_wave_pool_ic,
        dict(N=N, u0=u0, P0=P0, K_max=K_max, alpha=alpha_pl, rho0=rho0,
             sigma_x0=sigma_x0, seed=seed),
        dict(t_end=t_end, save_times=save_times, tau=tau, cfl=cfl, bc=bc),
        params,
    )
    return _write_single_fluid(out_path, "wavepool", x, times, U, ns, params)


def dump_steady_shock(params: dict, out_path: Path):
    """Steady shock with inflow/outflow BCs; manually orchestrated since
    py-1d's `run_steady_shock` in `setups/shock.py` does not accept
    `save_times`. We reproduce its driver loop with snapshot capture."""
    N = params["N"]; t_end = params["t_end"]; tau = params["tau"]
    cfl = params.get("cfl", 0.3); sigma_x0 = params.get("sigma_x0", 0.02)
    M1 = params["M1"]
    rho1 = params.get("rho1", 1.0); P1 = params.get("P1", 1.0)
    save_times = list(params.get("save_times", [t_end]))

    c_s1 = np.sqrt(SHOCK_GAMMA * P1 / rho1)
    u1 = M1 * c_s1
    rho2, u2, P2, _ = rankine_hugoniot(rho1, u1, P1)
    x = np.linspace(0, 1, N)
    rho = np.where(x < 0.5, rho1, rho2)
    u   = np.where(x < 0.5, u1,   u2)
    P   = np.where(x < 0.5, P1,   P2)
    U = build_state_from_primitives(rho, u, P, sigma_x0=sigma_x0)
    dx = 1.0 / (N - 1)
    U_inflow = np.array([rho1, rho1*u1, rho1*u1*u1 + P1, P1,
                         rho1*0.0, rho1*sigma_x0, 0.0,
                         rho1*u1**3 + 3*u1*P1])
    snaps = [(0.0, U.copy())]
    save_idx = 0
    t = 0.0; ns = 0
    save_arr = np.array(save_times, dtype=np.float64)
    while t < t_end and save_idx < len(save_arr):
        next_save = save_arr[save_idx]
        smax = max_signal_speed(U)
        dt = min(cfl*dx/smax, next_save - t, t_end - t)
        if dt <= 1e-14:
            snaps.append((t, U.copy())); save_idx += 1; continue
        U = step_inflow_outflow(U, dx, dt, tau, U_inflow)
        t += dt; ns += 1
        if t >= next_save - 1e-12:
            snaps.append((t, U.copy())); save_idx += 1
    times = np.array([s[0] for s in snaps], dtype=np.float64)
    U_arr = np.stack([s[1] for s in snaps], axis=0)
    return _write_single_fluid(out_path, "steady_shock", x, times, U_arr, ns,
                                params)


def _write_single_fluid(out_path: Path, target: str, x, times, U_arr, ns,
                        params: dict):
    """Common HDF5 writer for the 8-field single-fluid setups."""
    rho = U_arr[:, 0, :]
    rho_safe = np.maximum(rho, 1e-30)
    u    = U_arr[:, 1, :] / rho_safe
    Pxx  = U_arr[:, 2, :] - rho * u * u
    Pp   = U_arr[:, 3, :]
    L1   = U_arr[:, 4, :] / rho_safe
    alpha = U_arr[:, 5, :] / rho_safe
    beta  = U_arr[:, 6, :] / rho_safe
    M3    = U_arr[:, 7, :]
    Q     = M3 - rho*u**3 - 3.0*u*Pxx
    Svv   = Pxx / rho_safe
    gamma = np.sqrt(np.maximum(Svv - beta**2, 0.0))
    P_iso = (Pxx + 2.0 * Pp) / 3.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        params_grp = f.create_group("params")
        _attach_params(params_grp, params, target, PY1D_ROOT)
        f.attrs["nsteps"] = ns
        f.create_dataset("grid/x", data=np.asarray(x, dtype=np.float64))
        f.create_dataset("timesteps", data=times)
        fld = f.create_group("fields")
        for name, arr in [("rho", rho), ("u", u), ("Pxx", Pxx), ("Pp", Pp),
                          ("L1", L1), ("alpha", alpha), ("beta", beta),
                          ("M3", M3), ("Q", Q), ("gamma", gamma),
                          ("P_iso", P_iso)]:
            fld.create_dataset(name, data=np.asarray(arr, dtype=np.float64),
                               compression="gzip", compression_opts=4)
        f.create_dataset("U", data=U_arr.astype(np.float64),
                         compression="gzip", compression_opts=4)
    return out_path


# ---------------------------------------------------------------------
#  Two-fluid drivers
# ---------------------------------------------------------------------


def _run_two_fluid_loop(U, dx, t_end, tau_dd, tau_gg, kernel_fn, kparams,
                         m_A, m_B, save_times, bc='periodic', cfl=0.3):
    snaps = [(0.0, U.copy())]
    save_idx = 0; t = 0.0; ns = 0
    save_arr = np.asarray(save_times, dtype=np.float64)
    while t < t_end and save_idx < len(save_arr):
        next_save = save_arr[save_idx]
        smax = tf.max_signal_speed_both(U, U.shape[1])
        # In the spatially-uniform plasma case smax is 0; use kernel rate.
        if smax > 1e-30:
            dt_cfl = cfl * dx / smax
        else:
            dt_cfl = next_save - t
        # Cap dt by kernel rate: dt * nu_p < O(0.1)
        nu_p, nu_T = kernel_fn(U, m_A, m_B, *kparams, U.shape[1])
        nu_max = max(float(nu_p.max()), float(nu_T.max()), 1e-30)
        dt_kernel = 0.1 / nu_max
        dt = min(dt_cfl, dt_kernel, next_save - t, t_end - t)
        if dt <= 1e-14:
            snaps.append((t, U.copy())); save_idx += 1; continue
        if smax > 1e-30:
            U = tf.step_two_species(U, dx, dt, tau_dd, tau_gg, kernel_fn,
                                      kparams, m_A, m_B, bc=bc)
        else:
            # No hydro signal — apply cross-coupling only (eion path).
            tf.apply_cross_coupling(U, dt, nu_p, nu_T, m_A, m_B, U.shape[1])
        t += dt; ns += 1
        if t >= next_save - 1e-12:
            snaps.append((t, U.copy())); save_idx += 1
    times = np.array([s[0] for s in snaps], dtype=np.float64)
    U_arr = np.stack([s[1] for s in snaps], axis=0)
    return times, U_arr, ns


def dump_dust_in_gas(params: dict, out_path: Path):
    N = params["N"]; t_end = params["t_end"]
    A = params.get("A", 1.0)
    T0_gas = params.get("T0_gas", 1e-3)
    T0_dust = params.get("T0_dust", 1e-5)
    dust_to_gas = params.get("dust_to_gas", 0.1)
    grain_radius = params.get("grain_radius", 1e-3)
    tau_dd = params.get("tau_dd", 1e3)
    tau_gg = params.get("tau_gg", 1e-5)
    sigma_x0 = params.get("sigma_x0", 0.02)
    cfl = params.get("cfl", 0.3)
    save_times = list(params["save_times"])
    bc = params.get("bc", "periodic")

    x = np.linspace(0, 1, N, endpoint=False) + 0.5/N
    rho_d = np.full(N, dust_to_gas); u_d = A*np.sin(2*np.pi*x); P_d = rho_d*T0_dust
    rho_g = np.ones(N);              u_g = A*np.sin(2*np.pi*x); P_g = rho_g*T0_gas
    U = tf.make_initial_state(N, rho_d, u_d, P_d, rho_g, u_g, P_g,
                                sigma_x0=sigma_x0, x_coords=x)
    dx = 1.0 / N
    times, U_arr, ns = _run_two_fluid_loop(
        U, dx, t_end, tau_dd, tau_gg, tf.kernel_epstein,
        (grain_radius, 1.0), 1.0, 1.0, save_times, bc=bc, cfl=cfl)
    return _write_two_fluid(out_path, "dust_in_gas", x, times, U_arr, ns,
                              params, m_A=1.0, m_B=1.0)


def dump_eion(params: dict, out_path: Path):
    N = params["N"]; t_end = params["t_end"]
    m_e = params.get("m_e", 0.01); m_i = params.get("m_i", 1.0)
    Z_e = params.get("Z_e", 1.0); Z_i = params.get("Z_i", 1.0)
    lnLambda = params.get("lnLambda", 10.0)
    Te0 = params.get("Te0", 2.0); Ti0 = params.get("Ti0", 0.5)
    sigma_x0 = params.get("sigma_x0", 0.02)
    save_times = list(params["save_times"])
    bc = params.get("bc", "periodic")

    x = np.linspace(0, 1, N, endpoint=False) + 0.5/N
    rho_e = np.full(N, m_e); u_e = np.zeros(N); P_e = rho_e * Te0 / m_e
    rho_i = np.full(N, m_i); u_i = np.zeros(N); P_i = rho_i * Ti0 / m_i
    U = tf.make_initial_state(N, rho_e, u_e, P_e, rho_i, u_i, P_i,
                                sigma_x0=sigma_x0, x_coords=x)
    dx = 1.0 / N
    times, U_arr, ns = _run_two_fluid_loop(
        U, dx, t_end, 1e30, 1e30, tf.kernel_coulomb,
        (Z_e, Z_i, lnLambda), m_e, m_i, save_times, bc=bc,
        cfl=params.get("cfl", 0.3))
    return _write_two_fluid(out_path, "eion", x, times, U_arr, ns, params,
                              m_A=m_e, m_B=m_i)


def _write_two_fluid(out_path, target, x, times, U_arr, ns, params,
                     m_A, m_B):
    """16-field two-fluid HDF5 writer; species A in [0:8], B in [8:16]."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        params_grp = f.create_group("params")
        _attach_params(params_grp, params, target, PY1D_ROOT)
        f.attrs["nsteps"] = ns
        f.attrs["m_A"] = m_A; f.attrs["m_B"] = m_B
        f.create_dataset("grid/x", data=np.asarray(x, dtype=np.float64))
        f.create_dataset("timesteps", data=times)
        for label, off in (("A", 0), ("B", 8)):
            grp = f.create_group(f"fields/{label}")
            rho = U_arr[:, off+0, :]
            rho_safe = np.maximum(rho, 1e-30)
            u   = U_arr[:, off+1, :] / rho_safe
            Pxx = U_arr[:, off+2, :] - rho*u*u
            Pp  = U_arr[:, off+3, :]
            L1  = U_arr[:, off+4, :] / rho_safe
            alpha = U_arr[:, off+5, :] / rho_safe
            beta  = U_arr[:, off+6, :] / rho_safe
            M3    = U_arr[:, off+7, :]
            Q     = M3 - rho*u**3 - 3*u*Pxx
            Piso  = (Pxx + 2.0*Pp)/3.0
            m_sp  = m_A if label == "A" else m_B
            T     = Piso * m_sp / rho_safe
            for name, arr in [("rho", rho), ("u", u), ("Pxx", Pxx), ("Pp", Pp),
                              ("L1", L1), ("alpha", alpha), ("beta", beta),
                              ("M3", M3), ("Q", Q), ("Piso", Piso), ("T", T)]:
                grp.create_dataset(name, data=arr.astype(np.float64),
                                    compression="gzip", compression_opts=4)
        f.create_dataset("U", data=U_arr.astype(np.float64),
                         compression="gzip", compression_opts=4)
    return out_path


# ---------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------

DUMPERS = {
    "sod": dump_sod,
    "cold_sinusoid": dump_cold_sinusoid,
    "steady_shock": dump_steady_shock,
    "wavepool": dump_wavepool,
    "dust_in_gas": dump_dust_in_gas,
    "eion": dump_eion,
}


def main(argv=None):
    p = argparse.ArgumentParser(description="Dump py-1d Tier-A HDF5 golden")
    p.add_argument("--target", required=True, choices=sorted(DUMPERS))
    p.add_argument("--out", required=True, type=Path,
                   help="output HDF5 path")
    p.add_argument("--params", required=True,
                   help="JSON-encoded parameter dict")
    args = p.parse_args(argv)
    params = json.loads(args.params)
    DUMPERS[args.target](params, args.out)
    size = args.out.stat().st_size
    print(f"  wrote {args.out} ({size/1024:.1f} KiB)")


if __name__ == "__main__":
    # JIT warmup avoids spuriously slow first run when the script is
    # invoked once per target. The cost is small (~1s) for the few small
    # ICs we've timed.
    main()
