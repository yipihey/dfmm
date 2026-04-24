"""
Experiment 06: Maximum-entropy closure kappa(s) curve.

Produces the kurtosis (kappa = <xi^4>) as a function of standardized
skewness s = <xi^3> for the polynomial-exponent maximum-entropy
distribution f(xi) ∝ exp(-(A2+1) xi^2/2 - A3 xi^3 - A4 xi^4). The
constraint equations <xi^2> = 1, <xi^3> = s fix two of the three
parameters; A4 is chosen to ensure integrability (approximately
A4 = max(A3^2/2, small)).

The closure boundary at |s| ~ 1.4 is apparent in the divergence of the
Newton iteration; beyond that the moment family cannot represent the
distribution and the scheme flags it as closure failure.

Produces: paper/figs/maxent_kappa.png
Runtime:  < 10 seconds.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_hermite

from momentlag.schemes.maxent import solve_maxent

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    N_quad = 60
    xi, w = roots_hermite(N_quad)
    # Change of variables: integrand has an exp(xi^2/2) factor built into
    # solve_maxent, so rescale weights appropriately (solve_maxent uses
    # them as unweighted Hermite-Gauss nodes with physicist convention).
    # (The solver includes the Gaussian weighting; we just feed xi, w.)
    w = w*np.exp(xi**2)   # convert to unweighted integration over R

    # JIT warmup
    _ = solve_maxent(0.1, xi, w)

    s_vals = np.linspace(-1.45, 1.45, 121)
    kappa = np.empty_like(s_vals)
    A2 = np.empty_like(s_vals); A3 = np.empty_like(s_vals); A4 = np.empty_like(s_vals)
    for i, s in enumerate(s_vals):
        A2[i], A3[i], A4[i], kappa[i] = solve_maxent(s, xi, w)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(s_vals, kappa, 'C0-', lw=1.4)
    axes[0].axhline(3.0, color='k', ls=':', lw=0.8,
                     label='Gaussian $\\kappa = 3$')
    axes[0].axvline(np.sqrt(2), color='k', ls=':', lw=0.6, alpha=0.5)
    axes[0].axvline(-np.sqrt(2), color='k', ls=':', lw=0.6, alpha=0.5)
    axes[0].set_xlabel('standardized skewness s')
    axes[0].set_ylabel(r'$\kappa = \langle \xi^4 \rangle$')
    axes[0].set_title('Maximum-entropy kurtosis')
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].plot(s_vals, A2, label=r'$A_2$', lw=1.2)
    axes[1].plot(s_vals, A3, label=r'$A_3$', lw=1.2)
    axes[1].plot(s_vals, A4, label=r'$A_4$', lw=1.2)
    axes[1].set_xlabel('s'); axes[1].set_ylabel('coefficients')
    axes[1].set_title('Fitted cubic/quartic exponent coefficients')
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    out = os.path.join(FIG_DIR, "maxent_kappa.png")
    fig.savefig(out, dpi=130)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
