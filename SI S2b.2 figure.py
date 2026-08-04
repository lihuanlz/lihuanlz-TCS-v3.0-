# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 19:39:59 2026

@author: lihua
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:23:39 2026

@author: lihua
"""

# -*- coding: utf-8 -*-
"""
TCS four-layer model: V_R^4L phase diagram in the (u, beta) plane
2026-07-30 (smooth contours v2)

Computes the excess variance ratio V_R^4L - 1 = (N-1)*ICC of the four-layer
partition-occupancy model (S2b) on the (u = M/Omega, beta = Omega/N) plane,
in the digital limit kappa -> 0 (C = min(W, Omega)), and overlays the
analytic boundary estimate g0(u, beta) = beta*u*(1-u)^(beta-2).

Improvements over v1:
  - Omega 600 -> 2520 (24 beta values instead of 12, denser and more even)
  - u grid 40 -> 200 points
  - RectBivariateSpline interpolation to 500x300 dense grid for smooth contours
  - g0 boundary drawn via contour() on dense analytical grid (perfectly smooth)
  - 15 contour levels instead of 9
  - Colorbar added

Outputs:
  VR_phase_diagram.png  - contour map with g0 boundary and operating points
  printed checks        - beta=1 exact law, beta vs N_eff disentangling,
                          envelope property, beta=2 large-N_eff counterexample
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.special import gammaln
from scipy.interpolate import RectBivariateSpline

np.seterr(all='ignore')


def row_digital(M, Omega, N):
    """Digital limit (kappa -> 0): C = min(W, Omega), vectorised over W."""
    beta = Omega // N
    if beta * N != Omega:
        raise ValueError(f"Omega={Omega} must be divisible by N={N}")
    W = np.arange(0, int(M + 14 * np.sqrt(M + 1)) + 2)
    lw = W * np.log(M) - gammaln(W + 1) - M
    pw = np.exp(lw - lw.max())
    pw /= pw.sum()
    Cw = np.minimum(W, Omega)
    loge = (gammaln(Omega - Cw + 1) - gammaln(Omega - Cw - beta + 1)
            - (gammaln(Omega + 1) - gammaln(Omega - beta + 1)))
    e = np.where(Cw <= Omega - beta, np.exp(np.clip(loge, -745, 0)), 0.0)
    loge2 = (gammaln(Omega - Cw + 1) - gammaln(Omega - Cw - 2 * beta + 1)
             - (gammaln(Omega + 1) - gammaln(Omega - 2 * beta + 1)))
    e2 = np.where(Cw <= Omega - 2 * beta, np.exp(np.clip(loge2, -745, 0)), 0.0)
    eps = np.sum(pw * e)
    eps2 = np.sum(pw * e2)
    ICC = (eps2 - eps**2) / (eps * (1.0 - eps))
    return 1.0 - eps, ICC, 1.0 + (N - 1) * ICC


def g0(u, beta):
    """Continuum boundary estimate: V_R - 1 ~ beta*u*(1-u)^(beta-2)."""
    return beta * u * (1.0 - u)**(beta - 2)


# ============================================================
# Printed verification checks
# ============================================================

def run_checks():
    print("CHECK 1: beta=1 exact law  V_R-1 = u/(1-u)  (digital, Omega=500)")
    for u in [0.3, 0.5, 0.8, 0.9]:
        _, _, VR = row_digital(u * 500, 500, 500)
        print(f"   u={u}: V_R-1 = {VR-1:.6f}   exact = {u/(1-u):.6f}")

    print("\nCHECK 2: beta controls V_R at FIXED u=0.5, N_eff=5 (M=5, Omega=10)")
    for N in [2, 5, 10]:
        beta = 10 // N
        _, _, VR = row_digital(5, 10, N)
        print(f"   N={N:2d} beta={beta}: V_R = {VR:.4f}")

    print("\nCHECK 3: N_eff>=50 does NOT save small beta (beta=2, u=0.9)")
    for (M, Om) in [(45, 50), (90, 100), (450, 500), (900, 1000)]:
        _, _, VR = row_digital(M, Om, Om // 2)
        print(f"   M={M:3d} Omega={Om:4d} (N_eff={min(M, Om):3d}): V_R = {VR:.4f}")

    print("\nCHECK 4: envelope property (small N_eff below continuum at same (u,beta))")
    for (Ms, OmS, N) in [(5, 10, 5), (10, 20, 5), (5, 10, 2)]:
        u = Ms / OmS
        beta = OmS // N
        _, _, VRs = row_digital(Ms, OmS, N)
        _, _, VRc = row_digital(500, 1000, 1000 // beta)
        print(f"   u={u:.2f} beta={beta}: small(N_eff={min(Ms, OmS)}) {VRs:.4f}"
              f" <= continuum {VRc:.4f}")


# ============================================================
# Phase diagram (smooth contours)
# ============================================================

def make_figure(Omega=2520, out="VR_phase_diagram.svg"):
    # --- Computation grid: 24 betas (divisors of 2520 in [1,50]) x 200 u ---
    # 2520 = 2^3 * 3^2 * 5 * 7; divisors in [1,50]:
    betas = np.array([d for d in range(1, 51) if Omega % d == 0])
    us = np.linspace(0.30, 0.99, 200)
    G = np.zeros((len(betas), len(us)))
    for i, beta in enumerate(betas):
        N = Omega // beta
        for j, u in enumerate(us):
            _, _, VR = row_digital(u * Omega, Omega, N)
            G[i, j] = min(max(VR - 1, 1e-10), 50.0)

    # --- Spline interpolation to dense grid ---
    # Interpolate in log(beta) space for better spacing
    log_b = np.log(betas)
    spline = RectBivariateSpline(log_b, us, G, kx=3, ky=3, s=0)

    us_d = np.linspace(0.30, 0.99, 500)
    betas_d = np.linspace(1, 50, 300)
    G_d = np.clip(spline(np.log(betas_d), us_d), 1e-10, 100.0)
    U, B = np.meshgrid(us_d, betas_d)

    # --- Contour plot ---
    levels = np.geomspace(1e-3, 5, 15)
    norm = mcolors.LogNorm(vmin=1e-3, vmax=5)

    fig, ax = plt.subplots(figsize=(8, 7))
    cs = ax.contourf(U, B, G_d, levels=levels, cmap='viridis_r',
                     norm=norm, extend='both')
    cb = fig.colorbar(cs, ax=ax, label='$V_R^{4L} - 1$', pad=0.02, shrink=0.95)
    cb.ax.tick_params(labelsize=8)

    cl = ax.contour(U, B, G_d, levels=levels, colors='black',
                    linewidths=0.4, norm=norm)
    ax.clabel(cl, fmt=lambda x: f'{x:g}', fontsize=7, inline_spacing=5)

    # --- Smooth g0 boundary via contour on dense analytical grid ---
    ug = np.linspace(0.3, 0.99, 500)
    bg = np.linspace(1, 50, 300)
    UG, BG = np.meshgrid(ug, bg)
    G0_grid = BG * UG * (1.0 - UG)**(BG - 2.0)

    c01 = ax.contour(UG, BG, G0_grid, levels=[0.01], colors='k',
                     linestyles='--', linewidths=1.5)
    c1 = ax.contour(UG, BG, G0_grid, levels=[0.1], colors='k',
                    linestyles=':', linewidths=1.5)
    ax.clabel(c01, fmt={0.01: '$g_0{=}0.01$'}, fontsize=9, inline=True)
    ax.clabel(c1, fmt={0.1: '$g_0{=}0.1$'}, fontsize=9, inline=True)

    # --- Operating points ---
    ax.annotate('Table Row 1 ($\\beta$=2): $V_R$=1.38',
                xy=(0.5, 2), xytext=(0.56, 14),
                color='b', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='b', lw=0.8))
    ax.annotate('Table Rows 2-3 ($\\beta$=20): $\\approx$1',
                xy=(0.5, 20), xytext=(0.56, 27),
                color='b', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='b', lw=0.8))
    ax.annotate('$\\beta$=2, $u$=0.9, $N_{eff}$=90: $V_R$=2.06',
                xy=(0.9, 2), xytext=(0.75, 17),
                color='b', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='b', lw=0.8))

    ax.set_xlabel('$u = M/\\Omega$', fontsize=11)
    ax.set_ylabel('$\\beta = \\Omega/N$ (sites per partition)', fontsize=11)
    ax.set_title('Fig.S2b.2. $V_R^{4L}-1$ phase diagram (digital limit $\\kappa\\to0$, '
                 f'$\\Omega={Omega}$)\n'
                 'white: exact digital computation; '
                 'black: $g_0$ boundary estimate',
                 fontsize=15,fontweight='bold', y=1)
    ax.set_xlim(0.29, 1.0)
    ax.set_ylim(0, 52)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.show()
    print(f"\nfigure saved: {out}")


if __name__ == "__main__":
    run_checks()
    make_figure()