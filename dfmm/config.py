"""Centralized simulation configuration.

Replaces magic numbers previously embedded in the scheme kernels with
a single `SimulationConfig` dataclass. Each field has a documented
default that matches the historical hardcoded value; changing one
requires explicit construction with a different value, which makes
sensitivity analyses tractable.

Numba cannot accept dataclass instances directly, so kernels that
care about these values take them as explicit scalar arguments; the
integrator (`dfmm.integrate.run_to`) unpacks the `SimulationConfig`
into the kernel signature.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a single integrator run.

    All fields are immutable — treat the instance as a value-type.
    Create a new instance via `dataclasses.replace(cfg, field=value)`
    to change one option.
    """

    # ---- Numerical floors -------------------------------------------
    # Minimum rho / Pxx in divisions to prevent 1/0. Historically 1e-30
    # (used in the vectorized primitive computation) and 1e-15 for
    # alpha in the Liouville-source division.
    rho_floor: float = 1e-30
    alpha_floor: float = 1e-15
    # Minimum reconstructed pressure (applied after BGK/noise updates
    # to keep the state realizable even under heavy noise draws).
    pressure_floor: float = 1e-8

    # ---- Realizability -----------------------------------------------
    # |beta| is clipped to `realizability_headroom * sqrt(Sigma_vv)` so
    # gamma^2 = Sigma_vv - beta^2 stays strictly positive. 0.999 = 0.1%
    # slack; tighter values (say 0.99) are more conservative, looser
    # values (0.9999) preserve more signal but risk numerical issues.
    realizability_headroom: float = 0.999

    # ---- Noise injection amplitude limiter ---------------------------
    # The noise-KE budget caps the per-cell kinetic-energy injection at
    # `noise_ke_budget_fraction * local_internal_energy` so rare
    # Laplace heavy-tail draws cannot request more KE than available.
    noise_ke_budget_fraction: float = 0.25

    # ---- Time integration --------------------------------------------
    cfl: float = 0.3
    tau: float = 1e-3  # BGK relaxation time

    # ---- Boundary conditions ----------------------------------------
    # 'periodic', 'transmissive', 'reflective', 'dirichlet'. `periodic`
    # MUST be set on both sides simultaneously. Dirichlet BCs also need
    # `bc_state_left` / `bc_state_right` (8-field arrays).
    bc_left: str = 'periodic'
    bc_right: str = 'periodic'
    bc_state_left: Optional[np.ndarray] = None
    bc_state_right: Optional[np.ndarray] = None

    # ---- Reconstruction / scheme options ----------------------------
    # 'first' = first-order HLL on cell averages (shipped behavior).
    # 'muscl' = MUSCL-Hancock with slope-limited reconstruction
    # (Phase 2). Dispatched inside hll_step.
    reconstruction: str = 'first'
    # Slope limiter used when reconstruction='muscl'.
    # One of: 'minmod', 'mc', 'van_leer'.
    limiter: str = 'minmod'

    # ---- Ghost-cell count --------------------------------------------
    # First-order HLL needs n_ghost = 1. MUSCL-Hancock needs n_ghost = 2
    # so the reconstruction stencil can see the neighbour's neighbour.
    # Auto-selected from `reconstruction` if left at the default sentinel.
    n_ghost: int = 0

    def __post_init__(self):
        # Late auto-set of n_ghost based on reconstruction scheme.
        if self.n_ghost == 0:
            object.__setattr__(
                self, 'n_ghost',
                1 if self.reconstruction == 'first' else 2)
        # Light validation. Run at construction time so bad configs
        # fail fast at the integrator entry rather than mid-step.
        if self.cfl <= 0 or self.cfl >= 1.0:
            raise ValueError(f"cfl={self.cfl} must be in (0, 1)")
        if self.tau <= 0:
            raise ValueError(f"tau={self.tau} must be positive")
        if self.reconstruction not in ('first', 'muscl'):
            raise ValueError(
                f"reconstruction={self.reconstruction!r} "
                "must be 'first' or 'muscl'")
        if self.limiter not in ('minmod', 'mc', 'van_leer'):
            raise ValueError(
                f"limiter={self.limiter!r} must be one of "
                "minmod|mc|van_leer")
        # Periodic must be symmetric.
        if (self.bc_left == 'periodic') != (self.bc_right == 'periodic'):
            raise ValueError(
                "periodic BC must be specified on both sides "
                f"(got left={self.bc_left!r}, right={self.bc_right!r})")
