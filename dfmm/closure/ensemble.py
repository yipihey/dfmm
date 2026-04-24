"""
Paired-fixed ensembles and post-processing for LES validation spectra.

Implements the Angulo-Pontzen paired-phase protocol adapted to
compressible-turbulence wave-pool initial conditions:

    For each realization, draw phases {phi_k} from a seeded RNG and
    build two initial conditions -- one with phases phi_k, one with
    phi_k + pi (which flips u(x, 0) sign-for-sign). Evolve both, then
    average their spectra. The paired average cancels odd-order
    nonlinear contaminations to leading order, reducing ensemble
    variance by roughly a factor of 2 in the inertial range.

Further variance reduction comes from time-window averaging around each
focus time (a few snapshots within ~dt_window) and logarithmic k-binning
(see `dfmm.analysis.log_bin_spectrum`).
"""
import numpy as np


def draw_paired_phases(seed, K_max):
    """Return a random phase vector; its paired companion is phases + pi."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2*np.pi, size=K_max)


def build_focus_times(focus_times, n_per_window=5, half_width=0.025):
    """Build save times for time-window averaging.

    Returns
    -------
    save_times : ndarray
        Concatenation of `n_per_window` linearly-spaced times in
        [ft - half_width, ft + half_width] for each focus time ft.
    window_slices : list of slice
        Slice into save_times for each focus time.
    """
    save_times = []
    slices = []
    for ft in focus_times:
        start = len(save_times)
        ts = np.linspace(ft - half_width, ft + half_width, n_per_window)
        save_times.extend(ts)
        slices.append(slice(start, start + n_per_window))
    return np.array(save_times), slices


def ensemble_mean_spectrum(spectra, weights=None):
    """Average spectra across ensemble and time axes.

    Parameters
    ----------
    spectra : ndarray, shape (..., n_k)
        All axes except the last are treated as ensemble dimensions and
        flattened before averaging.
    weights : array-like or None
        Optional weights with the same shape as the ensemble dims.

    Returns
    -------
    mean : ndarray, shape (n_k,)
    std : ndarray, shape (n_k,)
    """
    P = np.asarray(spectra)
    flat = P.reshape(-1, P.shape[-1])
    if weights is None:
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
    else:
        w = np.asarray(weights).reshape(-1)
        w = w / w.sum()
        mean = (flat * w[:, None]).sum(axis=0)
        var = (w[:, None] * (flat - mean)**2).sum(axis=0)
        std = np.sqrt(var)
    return mean, std
