"""
Post-processing analysis utilities.

Functions for computing periodic power spectra, logarithmic k-binning,
and Lagrangian reindexing of saved snapshots. Used by the experiment
scripts to turn simulation output into the figures shown in the paper.
"""
import numpy as np


def compute_spectrum(f, dx=None):
    """One-sided periodic power spectrum of a 1D field.

    Parameters
    ----------
    f : (N,) array
        Signal on a periodic domain of length N*dx.
    dx : float, optional
        Cell spacing. Defaults to 1/N, i.e. unit-length domain.

    Returns
    -------
    k : (N//2 + 1,) array
        Wavenumbers in units of 1/length.
    P : (N//2 + 1,) array
        Power per mode, normalized so that sum(P[1:]) = Var(f) modulo
        the usual Parseval factor.
    """
    N = len(f)
    if dx is None:
        dx = 1.0/N
    f = f - np.mean(f)
    F = np.fft.rfft(f)
    P = np.abs(F)**2 / N**2
    k = np.fft.rfftfreq(N, d=dx)
    return k, P


def log_bin_spectrum(k, P, n_bins=25, k_min=1.0, k_max=None):
    """Average a power spectrum into logarithmic k bins.

    Smooths out the mode-to-mode phase ringing that dominates single-
    realization spectra, making the underlying cascade shape legible.

    Parameters
    ----------
    k : (M,) array
        Wavenumbers.
    P : (M,) or (..., M) array
        Power array. If multi-dimensional, the last axis is averaged
        within bins along with the ensemble dimensions (all flattened).
    n_bins : int
        Number of log bins between k_min and k_max.
    k_min, k_max : float
        Range. Defaults to 1.0 and k.max().

    Returns
    -------
    k_cen : (n_bins,) array
        Geometric-mean bin centers.
    P_binned : (n_bins,) array
        Mean power within each bin (NaN if the bin is empty).
    """
    k = np.asarray(k)
    P = np.asarray(P)
    if k_max is None:
        k_max = float(k.max())
    edges = np.logspace(np.log10(k_min), np.log10(k_max), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    # Flatten all axes except the last (k axis)
    P_flat = P.reshape(-1, len(k))
    out = np.empty(n_bins)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (k >= lo) & (k < hi)
        if mask.sum() == 0:
            out[i] = np.nan
            continue
        out[i] = P_flat[:, mask].mean()
    return centers, out


def log_spectrum_distance(P_a, P_b, k, k_range=(3.0, 50.0)):
    """RMS log-spectrum distance between two spectra over a k band.

    Common metric for how similar two turbulence spectra are: smaller
    is better, zero means identical. Values below ~0.3 log-decades
    correspond to agreement within a factor of 2 per mode on average.
    """
    k = np.asarray(k)
    mask = (k >= k_range[0]) & (k <= k_range[1]) & (P_a > 0) & (P_b > 0)
    if mask.sum() == 0:
        return np.nan
    diff = np.log10(P_a[mask]) - np.log10(P_b[mask])
    return float(np.sqrt(np.mean(diff**2)))


def lagrangian_reindex(U, x):
    """Sort the state by Lagrangian label L_1 (mod 1).

    Parcels carry the label L_1(x, t) advected from their initial
    positions. Reindexing by L_1 recovers a representation ordered by
    particle identity rather than by Eulerian cell, which is useful
    for some diagnostics (e.g. comparing parcel histories).

    Parameters
    ----------
    U : (n_fields, N) array
    x : (N,) array

    Returns
    -------
    L1_sorted : (N,) array
        Sorted label values, wrapped into [0, 1).
    U_sorted : (n_fields, N) array
        State reordered correspondingly.
    """
    rho = U[0]
    L1 = U[4]/rho
    L1_wrap = L1 - np.floor(L1)
    order = np.argsort(L1_wrap)
    return L1_wrap[order], U[:, order]
