#!/usr/bin/env python3
# ============================================================================
# Analog readout dilution-series Fisher: condition number vs κ
#
# PURPOSE
#   SI S2d.5 claims "analog precision is best near κ≈0.1 and degrades in
#   both directions (the Fisher analysis of S2c.5)". The existing S2c.5
#   scan only goes down to κ=0.1 — that is the scan BOUNDARY, not a
#   verified interior minimum.
#
#   This code extends the scan to κ=1e-4 to check whether χ truly has an
#   interior minimum near κ≈0.1 or continues to decrease.
#
# METHOD
#   2-parameter Fisher matrix for θ = (ln M₀, ln κ), analog readout.
#
#   Forward model (analog: observe p directly, β does not enter):
#     ξ = M₀ / (f · κ · N)
#     ξ = p/(1-p) + p/κ             (TCS master equation)
#
#   Fisher (analog, Gaussian noise):
#     I_ij = Σ_f (1/σ_f²) · (∂p_f/∂θ_i) · (∂p_f/∂θ_j)
#
#   Noise models:
#     (a) Homoscedastic:  σ_f = σ₀ (constant, all points equally weighted)
#     (b) Heteroscedastic: σ_f² = σ₀² + α·p_f  (readout + Poisson shot)
#     (c) Digital binomial β=1: w_f = n/[p_f(1-p_f)]  (for comparison)
#
#   Condition number χ = λ_max / λ_min.
#
# USAGE
#   python3 analog_fisher_kappa_scan.py
#
# OUTPUT
#   Console tables + analog_fisher_scan_results.npz
# ============================================================================

import numpy as np
from scipy.optimize import brentq
import sys

# ---------------------------------------------------------------------------
# 1. Solve TCS master equation: ξ = p/(1-p) + p/κ
# ---------------------------------------------------------------------------

def solve_p(xi, kappa):
    """Solve ξ = p/(1-p) + p/κ for p ∈ (0,1) via Brentq."""
    if xi <= 0:
        return 0.0
    if kappa <= 0:
        return min(xi, 1.0 - 1e-15)

    def eq(p):
        return p / (1.0 - p) + p / kappa - xi

    lo, hi = 1e-15, 1.0 - 1e-15
    if eq(lo) >= 0:
        return lo
    if eq(hi) <= 0:
        return hi
    return brentq(eq, lo, hi, xtol=1e-14, rtol=1e-14)


# ---------------------------------------------------------------------------
# 2. Jacobian: ∂p/∂(ln M₀), ∂p/∂(ln κ) at a single dilution point
# ---------------------------------------------------------------------------

def jacobian_2param(M0, kappa, N, f):
    """
    Compute analytic derivatives at dilution factor f.

    ξ = M₀ / (f · κ · N)

    ∂ξ/∂(ln M₀) = ξ       (ξ ∝ M₀)
    ∂ξ/∂(ln κ)  = -ξ      (ξ ∝ 1/κ, indirect path)

    dp/dξ = κ(1-p)² / (κ + (1-p)²)   [from implicit differentiation]

    κ also appears DIRECTLY in master equation (p/κ term):
    ∂p/∂(ln κ)|_ξ = p(1-p)² / (κ + (1-p)²)   [direct path]

    Total:
      ∂p/∂(ln M₀) = (dp/dξ) · ξ
      ∂p/∂(ln κ)  = (dp/dξ) · (-ξ)  +  p(1-p)²/(κ+(1-p)²)

    Returns (dM0, dk, p). If p is degenerate (too close to 0 or 1),
    returns (None, None, p).
    """
    xi = M0 / (f * kappa * N)
    p = solve_p(xi, kappa)

    if p < 1e-10 or p > 1.0 - 1e-10:
        return None, None, p

    q = 1.0 - p

    # dp/dξ = κq² / (κ + q²)
    dp_dxi = kappa * q**2 / (kappa + q**2)

    # ∂p/∂(ln M₀) = (dp/dξ) · ξ
    dp_dlnM0 = dp_dxi * xi

    # ∂p/∂(ln κ) = indirect + direct
    dp_dlnkappa_indirect = dp_dxi * (-xi)
    dp_dlnkappa_direct = p * q**2 / (kappa + q**2)
    dp_dlnkappa = dp_dlnkappa_indirect + dp_dlnkappa_direct

    return dp_dlnM0, dp_dlnkappa, p


# ---------------------------------------------------------------------------
# 3. Fisher matrix assembly
# ---------------------------------------------------------------------------

def fisher_2param(M0, kappa, N, f_min, f_max, n_points,
                 noise_model='homo', sigma_0=1.0, alpha_shot=1.0):
    """
    2×2 Fisher matrix for θ = (ln M₀, ln κ).

    noise_model:
      'homo'    — σ_f = sigma_0 (constant)
      'hetero'  — σ_f² = sigma_0² + alpha_shot · p_f   (readout + Poisson)
      'digital'  — binomial weight = alpha_shot / [p_f(1-p_f)]
                   (alpha_shot = n_per_point; for comparison with digital β=1)
    """
    fs = np.logspace(np.log10(f_min), np.log10(f_max), n_points)
    I = np.zeros((2, 2))
    n_used = 0

    for f in fs:
        dM0, dk, p = jacobian_2param(M0, kappa, N, f)
        if dM0 is None:
            continue

        jac = np.array([dM0, dk])

        if noise_model == 'homo':
            w = 1.0 / sigma_0**2
        elif noise_model == 'hetero':
            var = sigma_0**2 + alpha_shot * p
            w = 1.0 / var
        elif noise_model == 'digital':
            var = p * (1.0 - p) / alpha_shot
            if var < 1e-30:
                continue
            w = 1.0 / var
        else:
            raise ValueError(f"Unknown noise model: {noise_model}")

        I += w * np.outer(jac, jac)
        n_used += 1

    if n_used == 0 or I[0, 0] < 1e-300 or I[1, 1] < 1e-300:
        return 0.0, 0.0, np.inf, 0

    evals = np.linalg.eigvalsh(I)
    lmin = evals[0]
    lmax = evals[-1]
    chi = lmax / lmin if lmin > 1e-300 else np.inf
    return lmin, lmax, chi, n_used


# ---------------------------------------------------------------------------
# 4. Scan helper
# ---------------------------------------------------------------------------

def scan_table(M0, N, f_min, f_max, n_points, kappas,
               noise_model, sigma_0=1.0, alpha_shot=1.0, label=""):
    """Print a scan table and return results list."""
    print(f"\n--- {label} ---")
    print(f"  M0={M0:.0e}, N={N:.0e}, {n_points} pts, f in [{f_min:.0e}, {f_max:.0e}]")
    if noise_model == 'homo':
        print(f"  noise: homoscedastic sigma={sigma_0}")
    elif noise_model == 'hetero':
        print(f"  noise: heteroscedastic sigma_0={sigma_0}, alpha_shot={alpha_shot}")
    elif noise_model == 'digital':
        print(f"  noise: digital binomial beta=1, n_per_point={alpha_shot:.0e}")

    print(f"{'kappa':>8s}  {'l_min':>14s}  {'l_max':>14s}  {'chi':>14s}  {'n_used':>7s}")
    print("-" * 60)

    results = []
    for kappa in kappas:
        lmin, lmax, chi, n_used = fisher_2param(
            M0, kappa, N, f_min, f_max, n_points,
            noise_model=noise_model,
            sigma_0=sigma_0, alpha_shot=alpha_shot)
        print(f"{kappa:>8.4f}  {lmin:>14.4e}  {lmax:>14.4e}  {chi:>14.4e}  {n_used:>7d}")
        results.append((kappa, chi, lmin, lmax, n_used))

    return results


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    # ===================================================================
    # 1. Standard design: coarse scan 100 → 1e-4
    # ===================================================================
    M0 = 1e7
    N = 5e5
    n_points = 50
    f_min, f_max = 1.0, 1e5

    # κ values: extend well below 0.1
    kappas_coarse = [100, 50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1,
                     0.05, 0.02, 0.01, 0.005, 0.001, 0.0001]

    print("=" * 90)
    print("Analog readout dilution Fisher: condition number vs kappa")
    print("Extending S2c.5 scan below kappa=0.1 to verify interior minimum")
    print("=" * 90)

    # (a) Analog homoscedastic
    res_homo = scan_table(
        M0, N, f_min, f_max, n_points, kappas_coarse,
        'homo', sigma_0=1.0,
        label="Analog: homoscedastic sigma=1")

    # (b) Analog heteroscedastic (readout + Poisson shot)
    res_hetero = scan_table(
        M0, N, f_min, f_max, n_points, kappas_coarse,
        'hetero', sigma_0=0.01, alpha_shot=1.0,
        label="Analog: heteroscedastic sigma_0=0.01, alpha=1 (Poisson)")

    # (c) Digital binomial beta=1 (comparison with existing S2c.5 data)
    res_digital = scan_table(
        M0, N, f_min, f_max, n_points, kappas_coarse,
        'digital', sigma_0=1.0, alpha_shot=1e5,
        label="Digital binomial beta=1, n=1e5 (S2c.5 comparison)")

    # ===================================================================
    # 2. Fine scan: locate minimum
    # ===================================================================
    print("\n" + "=" * 90)
    print("Fine-grained scan: locate minimum chi")
    print("=" * 90)

    fine_kappas = np.logspace(np.log10(5), np.log10(1e-4), 60)
    fine_kappas = sorted(set(np.round(fine_kappas, 6)), reverse=True)

    print(f"\n{'kappa':>8s}  {'chi_homo':>14s}  {'chi_hetero':>14s}  {'chi_digital':>14s}")
    print("-" * 55)

    fine_results = []
    for kappa in fine_kappas:
        _, _, chi_h, _ = fisher_2param(M0, kappa, N, f_min, f_max, n_points, 'homo')
        _, _, chi_he, _ = fisher_2param(M0, kappa, N, f_min, f_max, n_points, 'hetero',
                                        sigma_0=0.01, alpha_shot=1.0)
        _, _, chi_d, _ = fisher_2param(M0, kappa, N, f_min, f_max, n_points, 'digital',
                                       sigma_0=1.0, alpha_shot=1e5)
        print(f"{kappa:>8.4f}  {chi_h:>14.4e}  {chi_he:>14.4e}  {chi_d:>14.4e}")
        fine_results.append((kappa, chi_h, chi_he, chi_d))

    # ===================================================================
    # 3. Find minimum for each noise model
    # ===================================================================
    print("\n" + "=" * 90)
    print("MINIMUM LOCATION")
    print("=" * 90)

    for label, idx in [("Analog homo", 1), ("Analog hetero", 2), ("Digital beta=1", 3)]:
        chis = [(k, c) for k, *cs in fine_results
                 for i, c in enumerate(cs) if i + 1 == idx and np.isfinite(c)]
        if not chis:
            print(f"  {label}: all chi = inf")
            continue

        min_kappa, min_chi = min(chis, key=lambda x: x[1])
        all_kappas = [k for k, _ in chis]
        all_chis = [c for _, c in chis]
        min_i = all_kappas.index(min_kappa)

        print(f"\n  {label}:")
        print(f"    min chi = {min_chi:.4e} at kappa = {min_kappa}")

        if min_i == 0:
            print(f"    -> Minimum at scan UPPER boundary (kappa={min_kappa})")
            print(f"    -> chi still INCREASING at upper boundary")
        elif min_i == len(all_kappas) - 1:
            print(f"    -> Minimum at scan LOWER boundary (kappa={min_kappa})")
            print(f"    -> chi still DECREASING toward kappa->0")
            print(f"    -> Interior minimum NOT confirmed for this design")
        else:
            print(f"    -> Interior minimum CONFIRMED")
            print(f"       chi(kappa={all_kappas[min_i-1]}) = {all_chis[min_i-1]:.4e}")
            print(f"       chi(kappa={all_kappas[min_i]}) = {all_chis[min_i]:.4e}  <-- min")
            print(f"       chi(kappa={all_kappas[min_i+1]}) = {all_chis[min_i+1]:.4e}")

    # ===================================================================
    # 4. Design dependence
    # ===================================================================
    print("\n" + "=" * 90)
    print("Design dependence: different f ranges (analog homoscedastic)")
    print("=" * 90)

    designs = [
        (1.0, 1e5, 50, "Standard  [1, 1e5], 50 pts"),
        (1.0, 1e7, 50, "Extended  [1, 1e7], 50 pts"),
        (1.0, 1e3, 50, "Narrow    [1, 1e3], 50 pts"),
        (1.0, 1e5, 7,  "Sparse    [1, 1e5],  7 pts"),
    ]

    scan_kappas_design = [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001]

    for f_min_d, f_max_d, n_pts_d, label in designs:
        scan_table(M0, N, f_min_d, f_max_d, n_pts_d, scan_kappas_design,
                   'homo', sigma_0=1.0, label=f"Analog homo: {label}")

    # ===================================================================
    # 5. M0 dependence
    # ===================================================================
    print("\n" + "=" * 90)
    print("M0 dependence (analog homoscedastic, standard design)")
    print("=" * 90)

    for M0_t in [1e4, 1e6, 1e7, 1e8, 1e10]:
        scan_table(M0_t, N, f_min, f_max, n_points,
                   [10, 1, 0.1, 0.01, 0.001, 0.0001],
                   'homo', sigma_0=1.0, label=f"M0 = {M0_t:.0e}")

    # ===================================================================
    # 6. Save results
    # ===================================================================
    np.savez('analog_fisher_scan_results.npz',
             coarse_kappas=np.array(kappas_coarse),
             coarse_homo=np.array([r[1] for r in res_homo]),
             coarse_hetero=np.array([r[1] for r in res_hetero]),
             coarse_digital=np.array([r[1] for r in res_digital]),
             fine_kappas=np.array([r[0] for r in fine_results]),
             fine_homo=np.array([r[1] for r in fine_results]),
             fine_hetero=np.array([r[2] for r in fine_results]),
             fine_digital=np.array([r[3] for r in fine_results]))

    print("\nResults saved: scripts/analog_fisher_scan_results.npz")
    print("Done.")


if __name__ == "__main__":
    main()
