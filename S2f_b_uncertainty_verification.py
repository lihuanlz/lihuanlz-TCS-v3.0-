# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:55:02 2026

@author: lihua
"""

# -*- coding: utf-8 -*-
"""
Verification of S2f.5 claim:
  "With >=5 dilution points, the uncertainty in the fitted b
   becomes negligible relative to binomial sampling noise."

Method:
  1. Simulate dilution series with known b, kappa, xi0
  2. Fit b from the series (also kappa, xi0)
  3. Estimate M at a test concentration using fitted b vs true b
  4. Compare sigma_M(b fitted) / sigma_M(b known)
  5. If ratio ~ 1.0, b uncertainty is negligible

Runs two parameter sets to confirm robustness.
"""

import numpy as np
from scipy.optimize import minimize


# ============================================================
# TCS forward model
# ============================================================

def solve_p_xi(xi, kappa):
    """Solve TCS master equation: xi = p/(1-p) + p/kappa"""
    a = 1.0
    b_coef = -(kappa + 1 + xi * kappa)
    c = xi * kappa
    disc = b_coef ** 2 - 4 * a * c
    if disc < 0:
        return 0.0
    p = (-b_coef - np.sqrt(disc)) / (2 * a)
    return p if 0 < p < 1 else 0.0


def P_pos_from_xi(xi, b, kappa, beta):
    """Forward model: P_pos = b + (1-b)*(1-(1-p)^beta)"""
    p = solve_p_xi(xi, kappa)
    if p <= 0 or p >= 1:
        return b
    return b + (1 - b) * (1 - (1 - p) ** beta)


def neg_log_likelihood(params, z, f, beta, n):
    """NLL for binomial: z_i ~ Binomial(n, P_pos(f_i * xi0; b, kappa, beta))"""
    b, kappa, xi0 = params
    if b < 1e-10 or b > 0.5 or kappa < 1e-4 or kappa > 100 or xi0 < 1e-8:
        return 1e10
    xi_arr = f * xi0
    P = np.array([P_pos_from_xi(xi, b, kappa, beta) for xi in xi_arr])
    P = np.clip(P, 1e-15, 1 - 1e-15)
    return -np.sum(z * np.log(P) + (n - z) * np.log(1 - P))


def fit_params(z, f, beta, n, b_init, kappa_init, xi0_init):
    """Fit b, kappa, xi0 from dilution series data."""
    bounds = [(1e-10, 0.5), (1e-4, 100), (1e-8, 100)]
    result = minimize(neg_log_likelihood, [b_init, kappa_init, xi0_init],
                      args=(z, f, beta, n), method='L-BFGS-B',
                      bounds=bounds,
                      options={'maxiter': 5000, 'ftol': 1e-15})
    if not result.success:
        result = minimize(neg_log_likelihood, [b_init, kappa_init, xi0_init],
                          args=(z, f, beta, n), method='Nelder-Mead',
                          options={'xatol': 1e-10, 'fatol': 1e-10,
                                   'maxiter': 10000})
    return result.x


def invert_P_to_xi(P_obs, b, kappa, beta):
    """Invert P_pos to get xi."""
    if P_obs <= b:
        return 0.0
    val = 1 - (P_obs - b) / (1 - b)
    if val <= 0:
        return 50.0  # near saturation, return large xi
    if val >= 1:
        return 0.0
    p = 1 - val ** (1.0 / beta)
    if p <= 0 or p >= 1:
        return 0.0
    return p / (1 - p) + p / kappa


# ============================================================
# Simulation
# ============================================================

def run_one_scenario(b_true, kappa_true, xi0_true, beta, n,
                     f_dil_range, f_test, n_points_list, n_sims):
    """Run simulation for one parameter set."""

    print(f"\n{'N_pts':>5} | {'sigma_b':>10} {'bias_b':>10} {'sig_b/b':>8} | "
          f"{'sig_M(bfit)':>12} {'sig_M(bknown)':>14} {'ratio':>8}")
    print("-" * 80)

    for n_points in n_points_list:
        f = np.geomspace(f_dil_range[0], f_dil_range[1], n_points)

        M_fits_bfit = []
        M_fits_bknown = []
        b_fits = []

        for sim in range(n_sims):
            # --- Generate dilution series data ---
            xi_arr = f * xi0_true
            P_true = np.array([P_pos_from_xi(xi, b_true, kappa_true, beta)
                               for xi in xi_arr])
            z_dil = np.random.binomial(n, P_true)

            # --- Fit b, kappa, xi0 ---
            b_fit, kappa_fit, xi0_fit = fit_params(
                z_dil, f, beta, n,
                b_init=b_true, kappa_init=kappa_true, xi0_init=xi0_true)
            b_fits.append(b_fit)

            # --- Test measurement at f_test ---
            xi_test = f_test * xi0_true
            P_test_true = P_pos_from_xi(xi_test, b_true, kappa_true, beta)
            z_test = np.random.binomial(n, P_test_true)
            P_test_obs = z_test / n

            # Estimate M with fitted b
            xi_est_bfit = invert_P_to_xi(P_test_obs, b_fit, kappa_true, beta)
            M_fits_bfit.append(xi_est_bfit)

            # Estimate M with known b
            xi_est_bknown = invert_P_to_xi(P_test_obs, b_true, kappa_true, beta)
            M_fits_bknown.append(xi_est_bknown)

        b_fits = np.array(b_fits)
        M_fits_bfit = np.array(M_fits_bfit)
        M_fits_bknown = np.array(M_fits_bknown)

        sigma_b = np.std(b_fits)
        bias_b = np.mean(b_fits) - b_true
        sig_b_over_b = sigma_b / b_true if b_true > 0 else np.nan
        sigma_M_bfit = np.std(M_fits_bfit)
        sigma_M_bknown = np.std(M_fits_bknown)
        ratio = sigma_M_bfit / sigma_M_bknown if sigma_M_bknown > 0 else np.nan

        print(f"{n_points:5d} | {sigma_b:10.2e} {bias_b:10.2e} "
              f"{sig_b_over_b:8.4f} | "
              f"{sigma_M_bfit:12.4e} {sigma_M_bknown:14.4e} {ratio:8.4f}")


def main():
    np.random.seed(42)

    n_points_list = [3, 4, 5, 7, 10, 20]
    n_sims = 100

    # ============================================================
    # Scenario 1: Typical digital ELISA (Simaa-like)
    # ============================================================
    print("=" * 90)
    print("Scenario 1: Digital ELISA (low background, moderate kappa)")
    print(f"  b=0.001, kappa=0.5, xi0=0.05, beta=50, n=10000")
    print(f"  Dilution: 1x to 0.01x, test at 0.5x")
    print(f"  Simulations: {n_sims}")
    print("=" * 90)

    run_one_scenario(
        b_true=0.001, kappa_true=0.5, xi0_true=0.05,
        beta=50, n=10000,
        f_dil_range=(1.0, 0.01), f_test=0.5,
        n_points_list=n_points_list, n_sims=n_sims)

    # ============================================================
    # Scenario 2: Higher background (b=0.01)
    # ============================================================
    print("\n" + "=" * 90)
    print("Scenario 2: Higher background (b=0.01)")
    print(f"  b=0.01, kappa=0.5, xi0=0.05, beta=50, n=10000")
    print(f"  Dilution: 1x to 0.01x, test at 0.5x")
    print(f"  Simulations: {n_sims}")
    print("=" * 90)

    run_one_scenario(
        b_true=0.01, kappa_true=0.5, xi0_true=0.05,
        beta=50, n=10000,
        f_dil_range=(1.0, 0.01), f_test=0.5,
        n_points_list=n_points_list, n_sims=n_sims)

    # ============================================================
    # Scenario 3: Strong binding (kappa=0.01, near calibration-free)
    # ============================================================
    print("\n" + "=" * 90)
    print("Scenario 3: Strong binding (kappa=0.01)")
    print(f"  b=0.001, kappa=0.01, xi0=0.5, beta=50, n=10000")
    print(f"  Dilution: 1x to 0.01x, test at 0.5x")
    print(f"  Simulations: {n_sims}")
    print("=" * 90)

    run_one_scenario(
        b_true=0.001, kappa_true=0.01, xi0_true=0.5,
        beta=50, n=10000,
        f_dil_range=(1.0, 0.01), f_test=0.5,
        n_points_list=n_points_list, n_sims=n_sims)

    # ============================================================
    # Conclusion
    # ============================================================
    print("\n" + "=" * 90)
    print("Conclusion")
    print("=" * 90)
    print()
    print("ratio = sigma_M(b fitted) / sigma_M(b known)")
    print("  ratio ~ 1.0  =>  b uncertainty negligible (claim holds)")
    print("  ratio > 1.05 =>  b uncertainty adds measurable noise (claim fails)")
    print()
    print("Claim: 'With >=5 dilution points, the uncertainty in the fitted b")
    print("  becomes negligible relative to binomial sampling noise.'")
    print()
    print("If ratio < 1.05 at N_pts=5 across all scenarios, the claim is CONFIRMED.")


if __name__ == '__main__':
    main()
