"""Ghost-cell boundary-condition appliers for the 8-field moment state.

Each applier operates on a ghost-padded state `U_ghost` of shape
`(n_fields, N + 2*n_ghost)` — `n_ghost` ghost cells on each side
flanking `N` interior cells — and writes values into the ghost slots
so the unified `hll_step` kernel can compute fluxes at the outer
interior faces without special-casing.

Layout convention
-----------------
    U_ghost[:, 0 ... n_ghost - 1]         -> left ghost cells
    U_ghost[:, n_ghost ... n_ghost+N-1]   -> interior cells
    U_ghost[:, n_ghost+N ... end]         -> right ghost cells

All appliers are `@nb.njit(cache=True)` to be callable from JIT-land.

Supported BCs
-------------
- `apply_periodic`       — copy opposite interior edge.
- `apply_transmissive`   — copy nearest interior cell (zero-gradient
  outflow).
- `apply_reflective`     — mirror-image cell values with velocity-
  like moments sign-flipped (IDX_MOM, IDX_M3 — the odd velocity
  moments).
- `apply_dirichlet`      — fixed state on the boundary (inflow-style).

`apply_mixed(U_ghost, n_ghost, bc_left, bc_right)` dispatches the
two sides independently and is the usual entry point called from
`run_to`.
"""
import numpy as np
import numba as nb

from ._common import IDX_MOM, IDX_M3

# BC type codes — integers for Numba branch-safety.
BC_PERIODIC = 0
BC_TRANSMISSIVE = 1
BC_REFLECTIVE = 2
BC_DIRICHLET = 3

_BC_NAME_TO_CODE = {
    'periodic': BC_PERIODIC,
    'transmissive': BC_TRANSMISSIVE,
    'reflective': BC_REFLECTIVE,
    'dirichlet': BC_DIRICHLET,
}


def bc_code(name):
    """Resolve a BC name string to its integer code; raises for unknown names."""
    key = name.lower()
    if key not in _BC_NAME_TO_CODE:
        raise ValueError(
            f"unknown boundary type {name!r}; "
            f"valid: {sorted(_BC_NAME_TO_CODE)}")
    return _BC_NAME_TO_CODE[key]


@nb.njit(cache=True)
def apply_periodic(U_ghost, n_ghost):
    """Periodic BC: left ghosts copy from the right-most interior cells,
    right ghosts from the left-most interior cells. Independent of
    field content."""
    n_fields, N_tot = U_ghost.shape
    N = N_tot - 2 * n_ghost
    for g in range(n_ghost):
        for k in range(n_fields):
            # Left ghost g <- interior index (N - n_ghost + g)
            U_ghost[k, g] = U_ghost[k, n_ghost + N - n_ghost + g]
            # Right ghost <- interior index g
            U_ghost[k, n_ghost + N + g] = U_ghost[k, n_ghost + g]


@nb.njit(cache=True)
def apply_transmissive(U_ghost, n_ghost):
    """Transmissive (zero-gradient) BC: ghost cells copy the nearest
    interior cell value, creating an outflow boundary with no
    reflection."""
    n_fields, N_tot = U_ghost.shape
    N = N_tot - 2 * n_ghost
    for g in range(n_ghost):
        for k in range(n_fields):
            U_ghost[k, g] = U_ghost[k, n_ghost]               # left -> first interior
            U_ghost[k, n_ghost + N + g] = U_ghost[k, n_ghost + N - 1]  # right -> last


@nb.njit(cache=True)
def apply_reflective(U_ghost, n_ghost):
    """Reflective (wall) BC: mirror the interior cell values into the
    ghost slots, flipping the sign of the odd velocity moments
    (IDX_MOM = rho u, IDX_M3 = rho <v^3>) so a wall's impenetrability
    is enforced. Even moments (rho, E_xx, P_perp, rho*L1, rho*alpha,
    rho*beta) mirror without sign flip."""
    n_fields, N_tot = U_ghost.shape
    N = N_tot - 2 * n_ghost
    for g in range(n_ghost):
        # Left side: ghost g <- interior (n_ghost - 1 - g + 1) = mirror about cell center
        #   with idx n_ghost, which maps interior cell g (0-indexed) to ghost (n_ghost-1-g).
        src_left = n_ghost + (n_ghost - 1 - g)
        for k in range(n_fields):
            v = U_ghost[k, src_left]
            if k == IDX_MOM or k == IDX_M3:
                v = -v
            U_ghost[k, g] = v
        # Right side: ghost (n_ghost + N + g) mirrors interior (n_ghost + N - 1 - g)
        src_right = n_ghost + N - 1 - g
        for k in range(n_fields):
            v = U_ghost[k, src_right]
            if k == IDX_MOM or k == IDX_M3:
                v = -v
            U_ghost[k, n_ghost + N + g] = v


@nb.njit(cache=True)
def apply_dirichlet(U_ghost, n_ghost, state_left, state_right):
    """Dirichlet/inflow BC: copy the supplied 8-field `state_left` into
    all left ghost cells and `state_right` into all right ghost cells.

    `state_left` / `state_right` are 1D length-`n_fields` arrays.
    """
    n_fields, N_tot = U_ghost.shape
    N = N_tot - 2 * n_ghost
    for g in range(n_ghost):
        for k in range(n_fields):
            U_ghost[k, g] = state_left[k]
            U_ghost[k, n_ghost + N + g] = state_right[k]


def apply_mixed(U_ghost, n_ghost, bc_left, bc_right,
                state_left=None, state_right=None):
    """Dispatch per-side. `bc_left` / `bc_right` are BC name strings
    (`'periodic'`, `'transmissive'`, `'reflective'`, `'dirichlet'`).

    If both sides are `'periodic'`, uses the dedicated kernel (avoids
    inconsistency between sides). Otherwise applies each BC in turn to
    its respective ghost cells.

    For `'dirichlet'`, the corresponding `state_left` / `state_right`
    kwarg must be a length-`n_fields` array.
    """
    if bc_left == 'periodic' and bc_right == 'periodic':
        apply_periodic(U_ghost, n_ghost)
        return
    if bc_left == 'periodic' or bc_right == 'periodic':
        raise ValueError("periodic BC must be applied on both sides")

    # Left
    if bc_left == 'transmissive':
        _apply_transmissive_left(U_ghost, n_ghost)
    elif bc_left == 'reflective':
        _apply_reflective_left(U_ghost, n_ghost)
    elif bc_left == 'dirichlet':
        if state_left is None:
            raise ValueError("dirichlet BC requires state_left")
        _apply_dirichlet_left(U_ghost, n_ghost, np.asarray(state_left))
    else:
        raise ValueError(f"unknown bc_left={bc_left!r}")

    # Right
    if bc_right == 'transmissive':
        _apply_transmissive_right(U_ghost, n_ghost)
    elif bc_right == 'reflective':
        _apply_reflective_right(U_ghost, n_ghost)
    elif bc_right == 'dirichlet':
        if state_right is None:
            raise ValueError("dirichlet BC requires state_right")
        _apply_dirichlet_right(U_ghost, n_ghost, np.asarray(state_right))
    else:
        raise ValueError(f"unknown bc_right={bc_right!r}")


# Per-side helpers (used by apply_mixed for non-periodic BCs). Keep them
# small and njit so composing BCs has no Python overhead per step.

@nb.njit(cache=True)
def _apply_transmissive_left(U_ghost, n_ghost):
    n_fields, _ = U_ghost.shape
    for g in range(n_ghost):
        for k in range(n_fields):
            U_ghost[k, g] = U_ghost[k, n_ghost]


@nb.njit(cache=True)
def _apply_transmissive_right(U_ghost, n_ghost):
    n_fields, N_tot = U_ghost.shape
    N = N_tot - 2 * n_ghost
    for g in range(n_ghost):
        for k in range(n_fields):
            U_ghost[k, n_ghost + N + g] = U_ghost[k, n_ghost + N - 1]


@nb.njit(cache=True)
def _apply_reflective_left(U_ghost, n_ghost):
    n_fields, _ = U_ghost.shape
    for g in range(n_ghost):
        src = n_ghost + (n_ghost - 1 - g)
        for k in range(n_fields):
            v = U_ghost[k, src]
            if k == IDX_MOM or k == IDX_M3:
                v = -v
            U_ghost[k, g] = v


@nb.njit(cache=True)
def _apply_reflective_right(U_ghost, n_ghost):
    n_fields, N_tot = U_ghost.shape
    N = N_tot - 2 * n_ghost
    for g in range(n_ghost):
        src = n_ghost + N - 1 - g
        for k in range(n_fields):
            v = U_ghost[k, src]
            if k == IDX_MOM or k == IDX_M3:
                v = -v
            U_ghost[k, n_ghost + N + g] = v


@nb.njit(cache=True)
def _apply_dirichlet_left(U_ghost, n_ghost, state_left):
    n_fields, _ = U_ghost.shape
    for g in range(n_ghost):
        for k in range(n_fields):
            U_ghost[k, g] = state_left[k]


@nb.njit(cache=True)
def _apply_dirichlet_right(U_ghost, n_ghost, state_right):
    n_fields, N_tot = U_ghost.shape
    N = N_tot - 2 * n_ghost
    for g in range(n_ghost):
        for k in range(n_fields):
            U_ghost[k, n_ghost + N + g] = state_right[k]


def pad_with_ghosts(U, n_ghost):
    """Return a new array `(n_fields, N + 2*n_ghost)` with the interior
    slice filled from `U` and ghost slots zeroed. Caller is expected to
    follow up with an `apply_*` call to populate the ghosts.
    """
    n_fields, N = U.shape
    U_ghost = np.zeros((n_fields, N + 2 * n_ghost), dtype=U.dtype)
    U_ghost[:, n_ghost:n_ghost + N] = U
    return U_ghost


def unpad_ghosts(U_ghost, n_ghost):
    """Return the interior slice `(n_fields, N)` by dropping the ghost
    flanks. Returns a view — copy explicitly if mutation-safety needed.
    """
    n_fields, N_tot = U_ghost.shape
    N = N_tot - 2 * n_ghost
    return U_ghost[:, n_ghost:n_ghost + N]
