"""Unit tests for `dfmm.tracers._locate_many`.

Isolates the JIT scan from the rest of the Tracers class by feeding
it synthetic L1 fields with known structure.
"""
import numpy as np
import pytest

from dfmm.tracers import _locate_many


def _run(label, idx_hint_init, L1_centers, x_centers, dx,
         max_scan=5, periodic=False, period=None, x_min=None):
    """Helper to set up output buffers and invoke the kernel."""
    n = len(label)
    label = np.asarray(label, dtype=float)
    x_out = np.zeros(n)
    idx_out = np.asarray(idx_hint_init, dtype=np.int64).copy()
    frac_out = np.zeros(n)
    if period is None:
        period = (x_centers[-1] - x_centers[0]) + dx
    if x_min is None:
        x_min = x_centers[0] - 0.5 * dx
    _locate_many(label, x_out, idx_out, frac_out,
                 np.asarray(L1_centers, dtype=float),
                 np.asarray(x_centers, dtype=float), float(dx),
                 int(max_scan), bool(periodic), float(period),
                 float(x_min), int(len(x_centers)), int(n))
    return x_out, idx_out, frac_out


def test_monotone_linear_L1_recovers_identity():
    """For L1(x) = x, looking up label=x_centers[k] returns x_centers[k]."""
    N = 64
    x = np.linspace(0, 1, N, endpoint=False) + 0.5 / N
    dx = x[1] - x[0]
    L1 = x.copy()
    labels = x[:5]
    idx0 = np.arange(5, dtype=np.int64)
    x_out, idx_out, frac_out = _run(labels, idx0, L1, x, dx, periodic=False)
    assert np.allclose(x_out, labels, atol=1e-12)


def test_label_between_cells_interpolates_linearly():
    """Label halfway between two cell centers should return the midpoint."""
    N = 8
    x = np.linspace(0, 1, N, endpoint=False) + 0.5 / N
    dx = x[1] - x[0]
    L1 = x.copy()
    # Label half-way between cell 3 and cell 4
    target = 0.5 * (L1[3] + L1[4])
    x_out, idx_out, frac_out = _run(
        np.array([target]), np.array([3]), L1, x, dx, periodic=False)
    assert abs(x_out[0] - 0.5 * (x[3] + x[4])) < 1e-12
    assert idx_out[0] == 3
    assert abs(frac_out[0] - 0.5) < 1e-12


def test_out_of_range_label_returns_nan():
    """A label outside the [min, max] of L1 returns NaN."""
    N = 8
    x = np.linspace(0, 1, N, endpoint=False) + 0.5 / N
    dx = x[1] - x[0]
    L1 = x.copy()
    x_out, _, _ = _run(np.array([-0.5]), np.array([0]),
                       L1, x, dx, periodic=False)
    assert np.isnan(x_out[0])


def test_idx_hint_far_from_truth_fails_gracefully():
    """With `max_scan=2`, a tracer whose true cell is 10 cells from its
    idx_hint cannot be located and should NaN out."""
    N = 32
    x = np.linspace(0, 1, N, endpoint=False) + 0.5 / N
    dx = x[1] - x[0]
    L1 = x.copy()
    # True cell is 20, hint at 0, max_scan=2 → only reaches cells 0-2
    target = L1[20]
    x_out, _, _ = _run(np.array([target]), np.array([0]),
                       L1, x, dx, max_scan=2, periodic=False)
    assert np.isnan(x_out[0])


def test_periodic_wrap_finds_bracket_across_seam():
    """When the label value is continuous across the periodic wrap, the
    wrap-aware bracket test should find it."""
    N = 16
    x = np.linspace(0, 1, N, endpoint=False) + 0.5 / N
    dx = x[1] - x[0]
    L1 = x.copy()
    # Tracer with label just above 0 (on the "left" side of the seam),
    # hint at cell N-1 (on the "right" side).
    target = 0.01  # < L1[0] = 1/(2N) ≈ 0.03 if N=16? Actually 1/32 = 0.03125.
    # Use a target that lives near the seam on the low side.
    # L1[0] = 0.03125 (first cell center). Target 0.01 is less — not in L1 range.
    # Let target be in [L1[N-1] - 1, L1[0]] under wrap: i.e. slightly below L1[0].
    # L1[N-1] = 0.96875. Bracket [N-1, 0] wrap-aware:
    #   L_lo = 0.96875, L_hi = 0.03125 → L_hi = 0.03125 + 1 = 1.03125.
    #   Target 0.01 → target + 1 = 1.01, which is between 0.96875 and 1.03125. ✓
    x_out, idx_out, _ = _run(np.array([target]),
                              np.array([N - 1]), L1, x, dx,
                              max_scan=5, periodic=True)
    # Should find a valid x position (not NaN)
    assert np.isfinite(x_out[0])
    # The reconstructed x is the interpolation inside the wrap bracket
    # (between x_centers[N-1] and x_centers[N-1]+dx, possibly wrapped).
    # It should be finite and within the domain.
    assert 0.0 <= x_out[0] <= 1.0
