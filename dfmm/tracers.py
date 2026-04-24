"""Lagrangian tracers tracked through the passively-advected L1 label.

A tracer is identified (user-facing) by its mass coordinate `q`,
computed once at t=0 from the initial density profile as
`cumsum(rho*dx)`. Internally, a tracer is located each timestep by
root-finding on the label equation

    L1(x, t) = label_k

where `label_k` is the value the L1 field had at the tracer's initial
position at t=0 (the shipped setups initialize L1 = x, so `label_k`
coincides with the tracer's initial x). Because the L1 field is a
conserved-mass-weighted passive scalar (`rho*L1` advects with the
flow), `label_k` is invariant along the tracer's trajectory. The
inverse lookup `x(label)` moves by at most one cell per timestep
(CFL), so a bounded local scan around the tracer's previous cell
index is sufficient.

Storage layout is ndarray-per-field with a `_capacity` >= `_n_active`
so that adaptive refinement (bisecting `q`-cells whose physical
extent exceeds ~1.5 dx) can insert new tracers without reallocating
every step. The same layout translates directly to 2D/3D by widening
scalars to vector components — position `x` becomes `(x, y[, z])`,
`label` becomes a tuple of two or three Lagrangian labels, and
`sample()`'s 1D linear interpolation becomes multilinear.

V1 scope / known limitations
----------------------------
- Assumes L1 remains approximately monotone in x within the local
  scan window around each tracer (pre-shell-crossing, or within one
  stream post-shell-crossing). The cholesky scheme's closure
  diagnostics (gamma -> 0) already flag where this breaks.
- **Periodic seam smearing**. The shipped setups initialize
  `L1 = x` which is NOT itself a periodic field; the L1 values
  across the period boundary jump by one full period. On the first
  step, the HLL scheme treats this jump as a large gradient and
  diffuses it across several cells, making tracer lookup near the
  seam unreliable. The local scan here does handle label-wrap in
  the bracket test (shift L_hi and target by ±period so adjacent
  cells stay comparable), but it cannot recover information the
  underlying scheme has already smeared away. For periodic flows,
  treat tracers within ~O(cs*t/dx) cells of the seam as suspect.
  Transmissive BCs have no seam and track cleanly.
- Single-fluid 8-field state only. Two-fluid tracers would live per
  species (straightforward extension).
"""
import numpy as np
import numba as nb

from .schemes._common import IDX_RHO, IDX_L1

_DEFAULT_MAX_SCAN = 5
_DEFAULT_REFINE_RATIO = 1.5
_GROW_FACTOR = 1.5


@nb.njit(cache=True, fastmath=False)
def _locate_many(label, x_out, idx_out, frac_out,
                 L1_centers, x_centers, dx,
                 max_scan, periodic, period, x_min,
                 n_cells, n_active):
    """Populate x_out, idx_out, frac_out for each active tracer via
    local scan around idx_out's incoming value (the hint).

    Scans cells at offsets [0, +1, -1, +2, -2, ..., +max_scan, -max_scan]
    around the hint. For periodic BCs:
      - cell indices wrap mod n_cells
      - the bracket test shifts L_hi and the target by ±period so
        that the L1 field is treated as continuous across the seam
        (otherwise tracers whose label sits on the far side of the
        wrap could never be bracketed).
      - the returned x is wrapped back into the domain.

    On success writes (x_new, bracket_cell_index, frac_in_cell).
    On failure writes NaN into x_out and leaves idx/frac unchanged.
    """
    for k in range(n_active):
        tgt0 = label[k]
        start = idx_out[k]
        found = False
        best_i = 0
        best_frac = 0.0
        for d in range(max_scan + 1):
            for s_step in range(2):
                if d == 0 and s_step == 1:
                    break
                i = start + d if s_step == 0 else start - d
                if periodic:
                    i = i % n_cells
                    ip = (i + 1) % n_cells
                else:
                    if i < 0 or i + 1 >= n_cells:
                        continue
                    ip = i + 1
                L_lo = L1_centers[i]
                L_hi = L1_centers[ip]
                tgt = tgt0
                if periodic:
                    # Continuize L_hi with respect to L_lo
                    if L_hi - L_lo > 0.5 * period:
                        L_hi -= period
                    elif L_hi - L_lo < -0.5 * period:
                        L_hi += period
                    # Continuize the target with respect to L_lo
                    if tgt - L_lo > 0.5 * period:
                        tgt -= period
                    elif tgt - L_lo < -0.5 * period:
                        tgt += period
                if (tgt >= L_lo and tgt <= L_hi) or (tgt <= L_lo and tgt >= L_hi):
                    denom = L_hi - L_lo
                    if denom > 1e-30 or denom < -1e-30:
                        best_frac = (tgt - L_lo) / denom
                    else:
                        best_frac = 0.5
                    best_i = i
                    found = True
                    break
            if found:
                break
        if found:
            # For periodic wrap (best_i == n_cells - 1, ip == 0), use dx
            # directly since x_centers[0] - x_centers[N-1] is a large
            # negative. Otherwise the cell-center spacing is dx anyway
            # for uniform grids.
            if periodic and best_i == n_cells - 1:
                x_new = x_centers[best_i] + best_frac * dx
            else:
                x_new = x_centers[best_i] + best_frac * (
                    x_centers[best_i + 1] - x_centers[best_i])
            if periodic:
                # Wrap result back into the domain.
                while x_new >= x_min + period:
                    x_new -= period
                while x_new < x_min:
                    x_new += period
            x_out[k] = x_new
            idx_out[k] = best_i
            frac_out[k] = best_frac
        else:
            x_out[k] = np.nan


class Tracers:
    """Container for a set of Lagrangian tracer particles.

    Parameters
    ----------
    q : 1D array
        Mass coordinates (user-facing tracer identities). Monotone
        non-decreasing.
    label : 1D array, same length as q
        L1-label values used for root-finding; must equal L1 at each
        tracer's initial position at t=0.
    x : 1D array, same length as q
        Initial physical positions.
    domain : (float, float)
        `(x_min, x_max)` — the Eulerian simulation domain.
    periodic : bool
        If True, cell indices in the scan wrap modulo n_cells. The
        label value is not wrapped (see module docstring).
    capacity : int, optional
        Pre-allocated buffer size. Defaults to len(q) rounded up by
        the grow factor so one refinement round fits without realloc.
    """

    def __init__(self, q, label, x, domain, periodic=False, capacity=None):
        q = np.asarray(q, dtype=float)
        label = np.asarray(label, dtype=float)
        x = np.asarray(x, dtype=float)
        if not (len(q) == len(label) == len(x)):
            raise ValueError("q, label, x must have the same length")
        n = len(q)
        if capacity is None:
            capacity = max(n, int(np.ceil(n * _GROW_FACTOR)))
        if capacity < n:
            raise ValueError("capacity must be >= len(q)")
        self._capacity = capacity
        self._n = n
        self._q = np.empty(capacity); self._q[:n] = q
        self._label = np.empty(capacity); self._label[:n] = label
        self._x = np.empty(capacity); self._x[:n] = x
        self._idx_hint = np.zeros(capacity, dtype=np.int64)
        self._frac = np.zeros(capacity)
        self.domain = (float(domain[0]), float(domain[1]))
        self.periodic = bool(periodic)

    # ---------- Factory ----------

    @classmethod
    def from_initial_conditions(cls, U0, x_centers, domain=None,
                                periodic=False, capacity=None):
        """Place one tracer per Eulerian cell, at the cell center.

        For a grid of N cells, produces N tracers (both periodic and
        transmissive). Each tracer's mass coordinate `q_i` is the
        cumulative mass up to the cell's center,

            q_i = 0.5 * rho[i] * dx + sum_{j<i} rho[j] * dx,

        which is equivalent to averaging the user's "left edge" and
        "right edge" mass coordinates `cumsum(rho*dx)`. Cell centers
        are the natural tracer positions because the L1 label is only
        known there — a boundary-edge tracer with `label = 0` (or
        `label = q_total`) has no valid bracket in the cell-centered
        L1 field, so the local-scan lookup cannot locate it. Placing
        at cell centers gives every tracer a findable label at t=0
        by construction.
        """
        rho = U0[IDX_RHO]
        L1 = U0[IDX_L1] / np.maximum(rho, 1e-30)
        N = len(x_centers)
        if N < 2:
            raise ValueError("need at least 2 Eulerian cells")
        dx = float(x_centers[1] - x_centers[0])
        if domain is None:
            x_min = float(x_centers[0]) - 0.5 * dx
            x_max = float(x_centers[-1]) + 0.5 * dx
            domain = (x_min, x_max)

        # Mass per cell; cumulative mass at cell centers
        mass_per_cell = rho * dx
        q_centers = np.cumsum(mass_per_cell) - 0.5 * mass_per_cell

        q = q_centers
        x = np.asarray(x_centers, dtype=float).copy()
        label = np.asarray(L1, dtype=float).copy()

        obj = cls(q=q, label=label, x=x, domain=domain,
                  periodic=periodic, capacity=capacity)
        # Seed the idx_hint so each tracer starts at its own cell;
        # otherwise the first update()'s bounded scan only reaches
        # cells within max_scan of cell 0.
        obj._idx_hint[:N] = np.arange(N, dtype=np.int64)
        return obj

    # ---------- Public array views ----------

    @property
    def n(self):
        return self._n

    @property
    def q(self):
        return self._q[:self._n]

    @property
    def label(self):
        return self._label[:self._n]

    @property
    def x(self):
        return self._x[:self._n]

    @property
    def idx_hint(self):
        return self._idx_hint[:self._n]

    @property
    def frac(self):
        return self._frac[:self._n]

    # ---------- Update / sample / snapshot ----------

    def update(self, U, x_centers, max_scan=_DEFAULT_MAX_SCAN):
        """Recompute tracer x positions from the current state.

        Runs a bounded local scan around each tracer's last known
        cell index and linearly interpolates the bracketing cells to
        solve L1(x, t) = label. Updates `x`, `idx_hint`, `frac` in
        place. Tracers that fail the scan are marked NaN in `x` and
        keep their last valid hint (so a later refinement pass can
        re-seed them).
        """
        rho = U[IDX_RHO]
        L1 = U[IDX_L1] / np.maximum(rho, 1e-30)
        dx = float(x_centers[1] - x_centers[0])
        x_min, x_max = self.domain
        period = x_max - x_min
        _locate_many(self._label, self._x, self._idx_hint, self._frac,
                     L1, np.asarray(x_centers, dtype=float), dx,
                     int(max_scan), self.periodic, period, x_min,
                     int(len(x_centers)), int(self._n))

    def sample(self, field_centers):
        """Linear-interpolate a cell-centered field at tracer positions.

        Reuses the (idx_hint, frac) cached by the last `update()`, so
        this is O(n_tracers) with no extra searching.
        """
        field = np.asarray(field_centers)
        idx = self.idx_hint
        frac = self.frac
        ip = (idx + 1) % len(field) if self.periodic else np.minimum(idx + 1,
                                                                     len(field) - 1)
        return field[idx] + frac * (field[ip] - field[idx])

    def sample_primitives(self, U):
        """Convenience: sample rho, u, P_xx, P_perp at tracer positions."""
        rho = U[IDX_RHO]
        u = U[1] / np.maximum(rho, 1e-30)
        Pxx = U[2] - rho * u * u
        Pp = U[3]
        return {
            'rho': self.sample(rho),
            'u': self.sample(u),
            'P_xx': self.sample(Pxx),
            'P_perp': self.sample(Pp),
        }

    def snapshot(self):
        """Return a plain particle-file-style dict at the current time."""
        return {
            'q': self.q.copy(),
            'x': self.x.copy(),
            'label': self.label.copy(),
        }

    # ---------- Adaptive refinement ----------

    def refine(self, x_centers, refine_ratio=_DEFAULT_REFINE_RATIO):
        """Bisect any `q`-cell whose physical extent exceeds
        `refine_ratio * dx`.

        For each neighbouring tracer pair (k, k+1) with
        `x[k+1] - x[k] > refine_ratio * dx`, inserts a new tracer at
        `q_new = 0.5*(q[k] + q[k+1])` and position
        `x_new = 0.5*(x[k] + x[k+1])`. The new tracer's label is
        taken as `0.5*(label[k] + label[k+1])`. This is exact for
        the shipped IC convention (L1 linear in x) and a sensible
        first-order guess otherwise; the next `update()` will snap
        it to the true L1 = label locus.

        Returns the number of tracers added. NaN'd tracers are
        skipped (they'll be ignored by the gap test since NaN
        comparisons fail).
        """
        if self._n < 2:
            return 0
        dx = float(x_centers[1] - x_centers[0])
        threshold = refine_ratio * dx
        x = self.x
        # Gap between consecutive tracers. For periodic, also test the
        # wrap-around gap.
        gaps = np.diff(x)
        insert_indices = np.where(gaps > threshold)[0]
        # Skip any where the gap involves a NaN
        insert_indices = insert_indices[np.isfinite(gaps[insert_indices])]
        n_new = len(insert_indices)
        if n_new == 0:
            return 0

        # Grow buffers if needed
        target_n = self._n + n_new
        if target_n > self._capacity:
            new_cap = max(target_n, int(np.ceil(self._capacity * _GROW_FACTOR)))
            self._grow(new_cap)

        # Insert from right to left so earlier insert positions stay valid
        n = self._n
        q_buf = self._q; lab_buf = self._label; x_buf = self._x
        idx_buf = self._idx_hint; frac_buf = self._frac
        for k in insert_indices[::-1]:
            # Shift [k+1 : n] right by 1, insert at slot k+1
            q_buf[k + 2:n + 1] = q_buf[k + 1:n]
            lab_buf[k + 2:n + 1] = lab_buf[k + 1:n]
            x_buf[k + 2:n + 1] = x_buf[k + 1:n]
            idx_buf[k + 2:n + 1] = idx_buf[k + 1:n]
            frac_buf[k + 2:n + 1] = frac_buf[k + 1:n]
            q_buf[k + 1] = 0.5 * (q_buf[k] + q_buf[k + 2])
            lab_buf[k + 1] = 0.5 * (lab_buf[k] + lab_buf[k + 2])
            x_buf[k + 1] = 0.5 * (x_buf[k] + x_buf[k + 2])
            idx_buf[k + 1] = idx_buf[k]  # seed near the left neighbor
            frac_buf[k + 1] = 0.5
            n += 1
        self._n = n
        return n_new

    def _grow(self, new_capacity):
        if new_capacity <= self._capacity:
            return
        def grow(arr, dtype=None):
            out = np.empty(new_capacity, dtype=arr.dtype if dtype is None else dtype)
            out[:self._n] = arr[:self._n]
            return out
        self._q = grow(self._q)
        self._label = grow(self._label)
        self._x = grow(self._x)
        self._idx_hint = grow(self._idx_hint)
        self._frac = grow(self._frac)
        self._capacity = new_capacity

    def __len__(self):
        return self._n

    def __repr__(self):
        return (f"Tracers(n={self._n}, capacity={self._capacity}, "
                f"domain={self.domain}, periodic={self.periodic})")
