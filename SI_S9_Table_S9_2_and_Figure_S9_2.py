# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 14:17:33 2026

@author: lihua
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, curve_fit
import matplotlib.pyplot as plt
import csv

# ------------------------------------------------------------
# TCS kinetics (n=2, REVERSIBLE)
# ------------------------------------------------------------
def tcs_kinetics(tau, p, xi, kappa):
    """dp/dτ = (1-p)^2 * (ξ - p/κ) - p"""
    return (1 - p)**2 * (xi - p / kappa) - p

def compute_peq(xi, kappa):
    """Solve for equilibrium occupancy p_eq using bisection."""
    def f(p):
        if p >= 1.0:
            return -1e6
        return xi - p/(1-p)**2 - p/kappa
    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(50):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

def generate_tcs_p(kappa, xi, tau_max, n_points=2000):
    """Generate TCS kinetic data p(τ)."""
    tau = np.logspace(-3, np.log10(tau_max), n_points)
    tau = np.sort(np.unique(np.clip(tau, 0, tau_max)))
    sol = solve_ivp(tcs_kinetics, [0, tau_max], [0.0], t_eval=tau,
                    args=(xi, kappa), method='RK45', rtol=1e-9)
    return tau, sol.y[0]

def durbin_watson(res):
    """Compute Durbin-Watson statistic."""
    return np.sum(np.diff(res)**2) / np.sum(res**2)

def run_noisy_demo():
    # Physical parameters
    xi = 10.0
    tau_max = 10.0
    noise_level = 0.001
    kappa_list = [0.001, 0.01, 0.1, 1, 10.0, 100]

    results = []

    for kappa in kappa_list:
        p_eq = compute_peq(xi, kappa)
        tau, p_true = generate_tcs_p(kappa, xi, tau_max)

        np.random.seed(42)
        noise = noise_level * p_eq * np.random.randn(len(tau))
        p_noisy = np.clip(p_true + noise, 0, 1)

        # Simonin normalisation: F = p / p_eq
        F_noisy = p_noisy / p_eq
        F_true = p_true / p_eq
        mask = F_noisy <= 0.85
        tau_cut, F_cut = tau[mask], F_noisy[mask]
        F_true_cut = F_true[mask]

        # ---- Single-parameter model (literature PSO): F = k*τ/(1+k*τ) ----
        try:
            popt1, _ = curve_fit(lambda t, k: k*t/(1+k*t), tau_cut, F_cut, p0=[10.0])
            k_std = popt1[0]
            F_std = k_std * tau_cut / (1 + k_std * tau_cut)
            res1 = F_cut - F_std
            r2_1 = 1 - np.sum(res1**2) / np.sum((F_cut - np.mean(F_cut))**2)
            dw1 = durbin_watson(res1)
        except:
            k_std, r2_1, dw1 = np.nan, np.nan, np.nan

        # ---- TCS degenerate form (κ→∞): F = (ξ/p_eq)·τ / (1 + ξ·τ) ----
        try:
            popt2, _ = curve_fit(lambda t, slope, curve: slope*t/(1+curve*t),
                                 tau_cut, F_cut, p0=[10.0, 10.0])
            slope_fit, curve_fit_val = popt2
            F_mod = slope_fit * tau_cut / (1 + curve_fit_val * tau_cut)
            res2 = F_cut - F_mod
            r2_2 = 1 - np.sum(res2**2) / np.sum((F_cut - np.mean(F_cut))**2)
            dw2 = durbin_watson(res2)
        except:
            slope_fit, curve_fit_val, r2_2, dw2 = np.nan, np.nan, np.nan, np.nan

        # TCS theoretical values
        slope_theory = xi / p_eq   # theoretical ξ/p_eq
        curve_theory = xi          # theoretical ξ

        results.append({
            'kappa': kappa, 'p_eq': p_eq,
            'tau_cut': tau_cut, 'F_cut': F_cut, 'F_true_cut': F_true_cut,
            'k_std': k_std, 'r2_1': r2_1, 'dw1': dw1,
            'slope_fit': slope_fit, 'curve_fit': curve_fit_val,
            'r2_2': r2_2, 'dw2': dw2,
            'slope_theory': slope_theory, 'curve_theory': curve_theory,
            'res1': res1 if 'res1' in locals() else None,
            'res2': res2 if 'res2' in locals() else None,
        })

    # ---- Print summary (academic English) ----
    print("\n" + "="*120)
    print("Summary for different κ (noise = 0.1%)")
    print(f"{'κ':<8} {'p_eq':<8} {'k_PSO':<10} {'R²_PSO':<10} {'DW_PSO':<10} "
          f"{'ξ/p_eq_fit':<10} {'ξ_fit':<10} {'R²_TCS_deg':<10} {'DW_TCS_deg':<10}")
    print(f"{'':8} {'':8} {'':10} {'':10} {'':10} "
          f"{'(theo. ξ/p_eq)':<10} {'(theo. ξ)':<10}")
    print("-"*100)
    for r in results:
        print(f"{r['kappa']:<8.3f} {r['p_eq']:<8.3f} "
              f"{r['k_std']:<10.3f} {r['r2_1']:<10.4f} {r['dw1']:<10.4f} "
              f"{r['slope_fit']:<10.3f} {r['curve_fit']:<10.3f} "
              f"{r['r2_2']:<10.4f} {r['dw2']:<10.4f}")

    # ---- Save CSV (column names in TCS terminology) ----
    csv_filename = "experiment2_results.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['κ', 'p_eq',
                         'k_PSO', 'R²_PSO', 'DW_PSO',
                         'ξ/p_eq_fit', 'ξ_fit', 'R²_TCS_degenerate', 'DW_TCS_degenerate',
                         'ξ/p_eq_theoretical', 'ξ_theoretical'])
        for r in results:
            writer.writerow([r['kappa'], r['p_eq'],
                             r['k_std'], r['r2_1'], r['dw1'],
                             r['slope_fit'], r['curve_fit'], r['r2_2'], r['dw2'],
                             r['slope_theory'], r['curve_theory']])
    print(f"\nResults saved to {csv_filename}")

    # ---- Plotting ----
    fig, axes = plt.subplots(len(kappa_list), 3, figsize=(18, 6*len(kappa_list)))

    for i, r in enumerate(results):
        ax0, ax1, ax2 = axes[i, 0], axes[i, 1], axes[i, 2]
        tau_cut = r['tau_cut']
        F_cut = r['F_cut']
        F_true_cut = r['F_true_cut']

        # Left panel: single-parameter model (PSO)
        ax0.plot(tau_cut, F_cut, 'o', markersize=3, color='gray', alpha=0.7, label='Noisy data')
        ax0.plot(tau_cut, F_true_cut, 'k-', linewidth=1.5, label='True TCS (noise-free)')
        if not np.isnan(r['k_std']):
            ax0.plot(tau_cut, r['k_std'] * tau_cut / (1 + r['k_std'] * tau_cut),
                     'r-', label=f'PSO (k={r["k_std"]:.1f})')
        ax0.plot(tau_cut, r['slope_theory'] * tau_cut / (1 + r['curve_theory'] * tau_cut),
                 'g--', label=f'κ→∞ limit (ξ/p_eq={r["slope_theory"]:.1f}, ξ={r["curve_theory"]:.1f})')
        ax0.set_title(f'κ = {r["kappa"]}   Single-parameter PSO   DW = {r["dw1"]:.2f}')
        ax0.set_xlabel('τ'); ax0.set_ylabel('F')
        ax0.legend(fontsize=7)

        # Middle panel: hyperbolic approximation
        ax1.plot(tau_cut, F_cut, 'o', markersize=3, color='gray', alpha=0.7, label='Noisy data')
        ax1.plot(tau_cut, F_true_cut, 'k-', linewidth=1.5, label='True TCS')
        if not np.isnan(r['slope_fit']):
            ax1.plot(tau_cut, r['slope_fit'] * tau_cut / (1 + r['curve_fit'] * tau_cut),
                     'b-', label=f'ξ/p_eq={r["slope_fit"]:.1f}, ξ={r["curve_fit"]:.1f}')
        ax1.plot(tau_cut, r['slope_theory'] * tau_cut / (1 + r['curve_theory'] * tau_cut),
                 'g--', label=f'κ→∞ limit')
        ax1.set_title(f'κ = {r["kappa"]}   Hyperbolic approximation   DW = {r["dw2"]:.2f}')
        ax1.set_xlabel('τ'); ax1.set_ylabel('F')
        ax1.legend(fontsize=7)

        # Right panel: residuals
        if r['res1'] is not None:
            ax2.plot(tau_cut, r['res1'], 'r.', markersize=3, label='PSO residual')
        if r['res2'] is not None:
            ax2.plot(tau_cut, r['res2'], 'b.', markersize=3, label='Hyperbolic residual')
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.set_title(f'κ = {r["kappa"]}   Residuals')
        ax2.set_xlabel('τ'); ax2.set_ylabel('Residual')
        ax2.legend()

    fig.suptitle('Fig. S9.2. True TCS (black), κ→∞ limit (green dashed), and fits',
                 fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('SI_S9_Table_S9_1_and_Figure_S9_1.svg', dpi=150)
    plt.show()

if __name__ == "__main__":
    run_noisy_demo()