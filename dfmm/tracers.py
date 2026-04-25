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
- **Periodic seam buffer**. The shipped setups initialize `L1 = x`
  which is NOT periodic; the L1 field has a full-period jump at the
  domain boundary, and the HLL scheme diffuses that jump across
  several cells on the first step. For periodic BCs we therefore
  skip the `boundary_buffer` leftmost and rightmost cells at
  construction (default 2) so no tracer's label falls inside the
  damaged zone. This relies on the seam staying stationary — valid
  for flows with zero mean velocity like the wave-pool. The scan
  itself is wrap-aware (cell index mod N, and L_hi/target shifted
  by ±period in the bracket test) so tracers drifting close to
  either edge search on both sides of the seam. Strong net drift in
  periodic would move the seam through the domain and break this
  assumption; that case needs a different approach (e.g. semi-
  Lagrangian velocity integration, or periodic label fields).
  `add_seam_tracer()` opts in to a single tracer placed inside the
  buffer at the steepest-gradient cell — that one tracer rides the
  smeared seam ramp and is exempted from the buffer-zone lost-flag
  test; see its docstring.
- Single-fluid 8-field state only. Two-fluid tracers would live per
  species (straightforward extension).
"""
import numpy as np
import numba as nb

from .schemes._common import IDX_RHO, IDX_L1

_DEFAULT_MAX_SCAN = 5
_DEFAULT_REFINE_RATIO = 1.5
_GROW_FACTOR = 1.5


def _seam_x_parabola(L1, x_centers, period):
    """Sub-cell seam x location from a parabolic x(q) fit.

    Take the cell with the largest L1 value (`imax`), then the two
    cells to its right with periodic wrap (`(imax+1) % N`,
    `(imax+2) % N`). Build three (q, x) points where the first one
    has both its q (L1) and x shifted by `-period` so its q is
    near-zero or slightly negative. Fit `x = a q^2 + b q + c`
    (`numpy.polyfit` deg 2) and return `c = x(q=0)` wrapped into
    `[0, period)`.

    At t=0 with the L1 = x convention this is exact (the three
    points lie on `x = q`, the parabola is degenerate-linear, and
    `x(q=0) = 0`). After HLL diffusion smears the seam ramp the
    fit becomes a true parabola whose root gives a sub-cell
    estimate of the seam position.
    """
    N = len(L1)
    imax = int(np.argmax(L1))
    q1 = L1[imax] - period
    x1 = x_centers[imax] - period
    i2 = (imax + 1) % N
    i3 = (imax + 2) % N
    q2, x2 = L1[i2], x_centers[i2]
    q3, x3 = L1[i3], x_centers[i3]
    # Make x monotone in array order (pull cells across the wrap by
    # +period as needed) so the polyfit sees a continuous curve.
    if x2 < x1: x2 += period
    if x3 < x2: x3 += period
    coeffs = np.polyfit(np.array([q1, q2, q3], dtype=float),
                        np.array([x1, x2, x3], dtype=float), 2)
    x_at_zero = float(coeffs[2])    # x(q=0) is the constant term
    return x_at_zero % period


def _seam_cell_index(L1, buffer=None):
    """Return the cell index at the centre of the L1 wrap-seam ramp.

    Uses the most-negative *raw* forward difference `L1[i+1] - L1[i]`
    (no wrap correction) — at a sharp seam this is ~-period and
    cleanly picks out the discontinuity edge; on a smeared ramp it
    picks the steepest-descent cell. The returned index is the right
    cell of that bracket (`(seam_edge + 1) % N`), which sits closer
    to the centre of the ramp than the left cell does.

    If `buffer` is given, restrict the search to cells inside the
    wraparound buffer band (`[0, buffer-1]` ∪ `[N-buffer, N-1]`) so
    that wavepool-driven local L1 compressions elsewhere can't pull
    the seam tracer off the actual seam ramp. Without the restriction,
    a strong local compression at e.g. cell N/2 can produce a more
    negative dL1/dx than the smeared seam itself.
    """
    diff_fwd = np.roll(L1, -1) - L1
    N = len(L1)
    if buffer is None or buffer * 2 >= N:
        seam_edge = int(np.argmin(diff_fwd))
    else:
        # Mask to the wrap-buffer band; everything else gets +inf so
        # argmin can't pick it.
        mask = np.full(N, np.inf)
        mask[:buffer] = diff_fwd[:buffer]
        mask[N - buffer:] = diff_fwd[N - buffer:]
        seam_edge = int(np.argmin(mask))
    return (seam_edge + 1) % N


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

    def __init__(self, q, label, x, domain, periodic=False, capacity=None,
                 boundary_buffer=0):
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
        self._lost = np.zeros(capacity, dtype=bool)
        self._dx_step = np.zeros(capacity)
        # Per-tracer flag marking seam tracers (placed inside the
        # periodic boundary buffer at the steepest-gradient location).
        # Such tracers are exempt from the buffer-zone lost-flag test
        # in update() — they live in the damaged zone by design.
        self._seam = np.zeros(capacity, dtype=bool)
        self._last_step_stats = None
        self.domain = (float(domain[0]), float(domain[1]))
        self.periodic = bool(periodic)
        # How many outer cells to treat as unreliable (only meaningful
        # in periodic mode; set by the factory, exposed for update()).
        self._buffer = int(boundary_buffer)

    # ---------- Factory ----------

    @classmethod
    def from_initial_conditions(cls, U0, x_centers, domain=None,
                                periodic=False, capacity=None,
                                boundary_buffer=None):
        """Place one tracer per Eulerian cell, at the cell center.

        For a grid of N cells, produces N tracers in the transmissive
        case and `N - 2 * boundary_buffer` tracers in the periodic
        case. Each tracer's mass coordinate `q_i` is the cumulative
        mass up to the cell's center,

            q_i = 0.5 * rho[i] * dx + sum_{j<i} rho[j] * dx,

        which is equivalent to averaging the user's "left edge" and
        "right edge" mass coordinates `cumsum(rho*dx)`. Cell centers
        are the natural tracer positions because the L1 label is only
        known there — a boundary-edge tracer with `label = 0` (or
        `label = q_total`) has no valid bracket in the cell-centered
        L1 field, so the local-scan lookup cannot locate it. Placing
        at cell centers gives every tracer a findable label at t=0
        by construction.

        Parameters
        ----------
        boundary_buffer : int, optional
            For `periodic=True` only: skip the `boundary_buffer`
            leftmost and rightmost cells when placing tracers. The
            HLL scheme diffuses the L1 seam discontinuity across
            a few cells on the first step and `label`-based lookup
            there becomes unreliable; the buffer brackets out the
            damaged zone. Defaults to 2 for periodic, ignored
            (treated as 0) for transmissive. See the module
            docstring for the zero-net-drift assumption this is
            valid under.
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

        if periodic:
            b = 2 if boundary_buffer is None else int(boundary_buffer)
        else:
            b = 0 if boundary_buffer is None else int(boundary_buffer)
        if b < 0 or 2 * b >= N:
            raise ValueError(
                f"boundary_buffer={b} leaves no interior cells (N={N})")

        # Mass per cell; cumulative mass at cell centers
        mass_per_cell = rho * dx
        q_centers = np.cumsum(mass_per_cell) - 0.5 * mass_per_cell

        interior = slice(b, N - b)
        q = q_centers[interior]
        x = np.asarray(x_centers, dtype=float)[interior].copy()
        label = np.asarray(L1, dtype=float)[interior].copy()

        obj = cls(q=q, label=label, x=x, domain=domain,
                  periodic=periodic, capacity=capacity, boundary_buffer=b)
        # Seed idx_hint with the tracer's absolute cell index so the
        # first update()'s bounded scan starts at the right place.
        n_kept = N - 2 * b
        obj._idx_hint[:n_kept] = np.arange(b, N - b, dtype=np.int64)
        return obj

    def add_seam_tracer(self, U, x_centers):
        """Insert one tracer at the seam's `x(q=0)` location.

        For periodic BCs the L1 = x convention has a full-period jump
        at the wrap. We normally skip the `boundary_buffer` cells on
        each side because the HLL scheme smears that discontinuity
        across them — turning the discontinuity into a smooth ramp.
        The standard label-based locate cannot track a particle
        through the no-tracer gap because the original "L1=0" label
        simply ceases to exist once the ramp smears.

        Instead the seam tracer is re-anchored each `update()` to a
        sub-cell estimate of the current seam location: take the
        cell with the largest L1 value (`imax`) and the next two
        cells to the right (with periodic wrap), shift the first
        point's q and x by `-period` so its q is near zero, fit a
        parabola through the three (q, x) pairs, and place the
        tracer at `x(q=0)`. At t=0 (sharp L1 = x ramp) this
        exactly recovers x = 0 = L; on the smeared ramp it
        interpolates between the cells.

        This makes the seam tracer an Eulerian seam-tracking probe
        rather than a Lagrangian particle; its `label` field is set
        to 0 (the q value at which we evaluate). `seam=True` exempts
        it from the buffer-zone lost-flag test in `update()`.

        Returns the index of the inserted tracer (i.e. its slot in
        the public `q`/`x`/`label`/... arrays), or -1 if the call is
        a no-op (transmissive grid, or N < 2).
        """
        if not self.periodic:
            return -1
        rho = U[IDX_RHO]
        L1 = U[IDX_L1] / np.maximum(rho, 1e-30)
        N = len(x_centers)
        if N < 2:
            return -1
        period = self.domain[1] - self.domain[0]
        x_seam = _seam_x_parabola(L1, np.asarray(x_centers, dtype=float),
                                   period)
        # idx_hint is the cell whose center is closest to x_seam — only
        # used to give `sample()` a reasonable interpolation bracket.
        dx = float(x_centers[1] - x_centers[0])
        x_min = self.domain[0]
        seam_idx = int(np.floor((x_seam - x_min) / dx)) % N

        if self._n + 1 > self._capacity:
            self._grow(self._n + 1)
        k = self._n
        self._q[k] = 0.0                # the q value we evaluate the fit at
        self._label[k] = 0.0
        self._x[k] = x_seam
        self._idx_hint[k] = seam_idx
        self._frac[k] = 0.0
        self._lost[k] = False
        self._dx_step[k] = 0.0
        self._seam[k] = True
        self._n += 1
        return k

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

    @property
    def lost(self):
        """Boolean mask: True where the last `update()` left the tracer
        either unresolved (NaN x) or inside the `boundary_buffer`-wide
        damaged band near the periodic seam where the L1 field is
        scheme-smeared and the lookup is physically unreliable.

        Cleared and rewritten each `update()` — a tracer that drifts
        back out of the damaged zone loses its lost flag.
        """
        return self._lost[:self._n]

    @property
    def valid(self):
        """Inverse of `lost` — the tracers whose current x is safe to
        use downstream."""
        return ~self._lost[:self._n]

    @property
    def dx_step(self):
        """Per-tracer signed displacement from the most recent
        `update()`, adjusted for periodic wrap. Length `n`. NaN where
        the tracer was NaN'd by the scan."""
        return self._dx_step[:self._n]

    @property
    def seam(self):
        """Boolean mask: True for tracers placed inside the periodic
        boundary buffer at the seam's steepest-gradient location (see
        `add_seam_tracer`). Such tracers are exempt from the
        damaged-zone lost-flag test in `update()`."""
        return self._seam[:self._n]

    def step_stats(self):
        """Summary statistics of the last `update()`'s Δx distribution.

        Returns a dict with keys:
          n_valid, n_lost, mean, std, median, max_abs
        computed over tracers with `valid=True` and finite Δx. Returns
        None if no `update()` has been called yet. Signed `mean` and
        `median` give net-drift information; `max_abs` is the diagnostic
        for whether `max_scan` is sized appropriately (if `max_abs`
        approaches `max_scan * dx` the scan is near its limit).
        """
        return self._last_step_stats

    # ---------- Update / sample / snapshot ----------

    def update(self, U, x_centers, max_scan=_DEFAULT_MAX_SCAN):
        """Recompute tracer x positions from the current state.

        Runs a bounded local scan around each tracer's last known
        cell index and linearly interpolates the bracketing cells to
        solve L1(x, t) = label. Updates `x`, `idx_hint`, `frac`, and
        `lost` in place. A tracer is marked `lost=True` if either:
          - the scan failed (x is NaN), or
          - the resolved x lies within `boundary_buffer * dx` of
            either periodic domain edge — the zone where HLL
            diffusion of the initial seam discontinuity has
            corrupted the L1 field.
        Tracers that fail the scan keep their last valid idx_hint
        so a later refinement pass can re-seed them.
        """
        rho = U[IDX_RHO]
        L1 = U[IDX_L1] / np.maximum(rho, 1e-30)
        dx = float(x_centers[1] - x_centers[0])
        x_min, x_max = self.domain
        period = x_max - x_min

        # Snapshot x before the scan overwrites it, so we can compute
        # per-tracer Δx below.
        n = self._n
        x_prev = self._x[:n].copy()

        _locate_many(self._label, self._x, self._idx_hint, self._frac,
                     L1, np.asarray(x_centers, dtype=float), dx,
                     int(max_scan), self.periodic, period, x_min,
                     int(len(x_centers)), int(n))

        # Re-anchor seam tracers to the current x(q=0) parabolic
        # fit (see `_seam_x_parabola`). Label-based locate cannot
        # track a seam tracer because the original L1 value at the
        # seam smears away within a few steps; instead we recompute
        # the sub-cell seam x each step from a quadratic x(q) fit
        # through the largest-q cell and its two right-neighbours
        # (periodic wrap).
        if self.periodic and np.any(self._seam[:n]):
            x_arr = np.asarray(x_centers, dtype=float)
            x_seam = _seam_x_parabola(L1, x_arr, period)
            seam_idx = int(np.floor((x_seam - x_min) / dx)) % len(x_centers)
            for k in np.where(self._seam[:n])[0]:
                self._x[k] = x_seam
                self._label[k] = 0.0
                self._idx_hint[k] = seam_idx
                # Cell-fraction for sample(): position within the
                # bracket [seam_idx, seam_idx+1].
                self._frac[k] = ((x_seam - x_arr[seam_idx]) / dx) % 1.0

        # Flag tracers in the damaged zone or NaN'd. Seam tracers are
        # exempt from the buffer-zone test — they're placed there on
        # purpose, riding the smeared seam ramp.
        xs = self._x[:n]
        not_finite = ~np.isfinite(xs)
        if self.periodic and self._buffer > 0:
            lo = x_min + self._buffer * dx
            hi = x_max - self._buffer * dx
            in_buffer = (xs < lo) | (xs > hi)
            in_buffer &= ~self._seam[:n]
            self._lost[:n] = not_finite | in_buffer
        else:
            self._lost[:n] = not_finite

        # Per-step displacement, periodic-wrap-adjusted (minimum-image
        # convention on a tracer that wrapped around the seam).
        d = xs - x_prev
        if self.periodic:
            d = d - period * np.round(d / period)
        self._dx_step[:n] = d
        # Summary stats over the valid, finite-Δx subset.
        good = (~self._lost[:n]) & np.isfinite(d)
        if good.any():
            dg = d[good]
            self._last_step_stats = {
                'n_valid': int(good.sum()),
                'n_lost': int(n - good.sum()),
                'mean': float(dg.mean()),
                'std': float(dg.std()),
                'median': float(np.median(dg)),
                'max_abs': float(np.max(np.abs(dg))),
            }
        else:
            self._last_step_stats = {
                'n_valid': 0, 'n_lost': int(n),
                'mean': float('nan'), 'std': float('nan'),
                'median': float('nan'), 'max_abs': float('nan'),
            }

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
        idx_buf = self._idx_hint; frac_buf = self._frac; lost_buf = self._lost
        dxs_buf = self._dx_step; seam_buf = self._seam
        for k in insert_indices[::-1]:
            # Shift [k+1 : n] right by 1, insert at slot k+1
            q_buf[k + 2:n + 1] = q_buf[k + 1:n]
            lab_buf[k + 2:n + 1] = lab_buf[k + 1:n]
            x_buf[k + 2:n + 1] = x_buf[k + 1:n]
            idx_buf[k + 2:n + 1] = idx_buf[k + 1:n]
            frac_buf[k + 2:n + 1] = frac_buf[k + 1:n]
            lost_buf[k + 2:n + 1] = lost_buf[k + 1:n]
            dxs_buf[k + 2:n + 1] = dxs_buf[k + 1:n]
            seam_buf[k + 2:n + 1] = seam_buf[k + 1:n]
            q_buf[k + 1] = 0.5 * (q_buf[k] + q_buf[k + 2])
            lab_buf[k + 1] = 0.5 * (lab_buf[k] + lab_buf[k + 2])
            x_buf[k + 1] = 0.5 * (x_buf[k] + x_buf[k + 2])
            idx_buf[k + 1] = idx_buf[k]  # seed near the left neighbor
            frac_buf[k + 1] = 0.5
            lost_buf[k + 1] = False       # let the next update() re-classify
            dxs_buf[k + 1] = 0.0
            seam_buf[k + 1] = False
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
        self._lost = grow(self._lost)
        self._dx_step = grow(self._dx_step)
        self._seam = grow(self._seam)
        self._capacity = new_capacity

    def __len__(self):
        return self._n

    def __repr__(self):
        return (f"Tracers(n={self._n}, capacity={self._capacity}, "
                f"domain={self.domain}, periodic={self.periodic})")
