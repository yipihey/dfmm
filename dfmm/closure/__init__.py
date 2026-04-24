"""Kramers-Moyal LES closure: calibrated noise model and ensemble tooling.

Submodules:

    noise_model     Energy-conservative HLL+BGK step with stochastic
                    closure correction, Gaussian noise smoothing, and
                    ensemble driver.
    ensemble        Paired-fixed ensemble helpers for variance reduction
                    in validation spectra.
"""
from . import noise_model, ensemble

__all__ = ["noise_model", "ensemble"]
