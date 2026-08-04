# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 14:44:37 2026

@author: lihua
"""

#!/usr/bin/env python3
# ============================================================================
# S2c.5.2 Profile Likelihood Analysis (PARALLEL VERSION)
# ============================================================================

import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# ---- Parallel config ----
N_WORKERS = max(1, os.cpu_count() - 2)  # 留2核给系统
print(f"Using {N_WORKERS} workers (total CPUs: {os.cpu_count()})")

# ---------------------------------------------------------------------------
# 1. Forward model
# ---------------------------------------------------------------------------

def solve_p(xi, kappa):
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


def P_pos(f, M0, kappa, beta, N, b=0.0):
    xi = M0 / (f * kappa * N * beta)
    p = solve_p(xi, kappa)
    Pspec = 1.0 - (1.0 - p)**beta
    return b + (1.0 - b) * Pspec


# ---------------------------------------------------------------------------
# 2. Log-likelihood
# ---------------------------------------------------------------------------

def log_likelihood(log_theta, fs, k_obs, n_obs, N, b=0.0):
    M0 = np.exp(log_theta[0])
    kappa = np.exp(log_theta[1])
    beta = np.exp(log_theta[2])
    if beta < 0.01 or beta > 200:
        return -np.inf
    if kappa < 0.001 or kappa > 500:
        return -np.inf
    if M0 < 1 or M0 > 1e12:
        return -np.inf
    ll = 0.0
    for i, f in enumerate(fs):
        P = P_pos(f, M0, kappa, beta, N, b)
        P = np.clip(P, 1e-15, 1.0 - 1e-15)
        k = k_obs[i]
        n = n_obs[i]
        ll += k * np.log(P) + (n - k) * np.log(1.0 - P)
    return ll


# ---------------------------------------------------------------------------
# 3. Worker functions (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _worker_1d(args):
    """Single grid point for 1D profile. Runs in a separate process."""
    ig, gv, param_idx, other_idx, x0_other, fs, k_obs, n_obs, N, b = args

    def neg_ll(x):
        lt = np.zeros(3)
        lt[param_idx] = gv
        lt[other_idx[0]] = x[0]
        lt[other_idx[1]] = x[1]
        return -log_likelihood(lt, fs, k_obs, n_obs, N, b)

    best_val = np.inf
    best_x = None

    # Deterministic starts (avoid np.random in workers — use seed per grid point)
    rng = np.random.RandomState(42 + ig)
    starts = [x0_other.copy()]
    for _ in range(5):
        starts.append(x0_other + rng.randn(2) * 0.5)
    for _ in range(3):
        starts.append(rng.randn(2) * 2)

    for x0 in starts:
        try:
            res = minimize(neg_ll, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except:
            pass

    if best_x is not None:
        return ig, -best_val, best_x
    else:
        return ig, -np.inf, np.array([np.nan, np.nan])


def _worker_2d(args):
    """Single grid point for 2D profile. Runs in a separate process."""
    iM, iK, lM0, lkappa, log_beta_true, fs, k_obs, n_obs, N, b = args

    def neg_ll(x):
        lt = np.array([lM0, lkappa, x[0]])
        return -log_likelihood(lt, fs, k_obs, n_obs, N, b)

    best_val = np.inf
    for x0 in [log_beta_true, log_beta_true + 1, log_beta_true - 1, 0, 1, 3]:
        try:
            res = minimize(neg_ll, [x0], method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
        except:
            pass

    return iM, iK, -best_val


def _worker_2d_Mb(args):
    """Single grid point for 2D profile (M₀, β). Profile out κ."""
    iM, iB, lM0, lBeta, log_kappa_true, fs, k_obs, n_obs, N, b = args

    def neg_ll(x):
        lt = np.array([lM0, x[0], lBeta])
        return -log_likelihood(lt, fs, k_obs, n_obs, N, b)

    best_val = np.inf
    for x0 in [log_kappa_true, log_kappa_true + 1, log_kappa_true - 1, 0, 1, 3]:
        try:
            res = minimize(neg_ll, [x0], method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
        except:
            pass

    return iM, iB, -best_val


# ---------------------------------------------------------------------------
# 4. Parallel profile likelihood
# ---------------------------------------------------------------------------

def profile_likelihood_1d(param_idx, param_grid, log_theta_true, fs, k_obs, n_obs, N, b=0.0):
    n_grid = len(param_grid)
    profile_ll = np.full(n_grid, -np.inf)
    opt_params = np.full((n_grid, 2), np.nan)

    other_idx = [i for i in range(3) if i != param_idx]
    x0_other = log_theta_true[other_idx].copy()

    # Build task list
    tasks = [(ig, gv, param_idx, other_idx, x0_other, fs, k_obs, n_obs, N, b)
             for ig, gv in enumerate(param_grid)]

    t0 = time.time()
    done_count = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_worker_1d, task): task[0] for task in tasks}
        for future in as_completed(futures):
            ig, ll_val, best_x = future.result()
            profile_ll[ig] = ll_val
            opt_params[ig] = best_x
            done_count += 1
            if done_count % 20 == 0:
                print(f"    [{param_idx}] {done_count}/{n_grid} done "
                      f"({time.time()-t0:.1f}s)", flush=True)

    return profile_ll, opt_params


def profile_likelihood_2d(M0_grid, kappa_grid, beta_true, fs, k_obs, n_obs, N, b=0.0):
    nM = len(M0_grid)
    nK = len(kappa_grid)
    profile_ll_2d = np.full((nM, nK), -np.inf)
    log_beta_true = np.log(beta_true)

    # Build task list: all (iM, iK) pairs
    tasks = []
    for iM, lM0 in enumerate(M0_grid):
        for iK, lkappa in enumerate(kappa_grid):
            tasks.append((iM, iK, lM0, lkappa, log_beta_true, fs, k_obs, n_obs, N, b))

    t0 = time.time()
    done_count = 0
    total = len(tasks)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_worker_2d, task): (task[0], task[1]) for task in tasks}
        for future in as_completed(futures):
            iM, iK, ll_val = future.result()
            profile_ll_2d[iM, iK] = ll_val
            done_count += 1
            if done_count % 200 == 0:
                print(f"    2D (M₀,κ): {done_count}/{total} done "
                      f"({time.time()-t0:.1f}s)", flush=True)

    return profile_ll_2d


def profile_likelihood_2d_M0_beta(M0_grid, beta_grid, kappa_true, fs, k_obs, n_obs, N, b=0.0):
    nM = len(M0_grid)
    nB = len(beta_grid)
    profile_ll_2d = np.full((nM, nB), -np.inf)
    log_kappa_true = np.log(kappa_true)

    tasks = []
    for iM, lM0 in enumerate(M0_grid):
        for iB, lBeta in enumerate(beta_grid):
            tasks.append((iM, iB, lM0, lBeta, log_kappa_true, fs, k_obs, n_obs, N, b))

    t0 = time.time()
    done_count = 0
    total = len(tasks)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_worker_2d_Mb, task): (task[0], task[1]) for task in tasks}
        for future in as_completed(futures):
            iM, iB, ll_val = future.result()
            profile_ll_2d[iM, iB] = ll_val
            done_count += 1
            if done_count % 200 == 0:
                print(f"    2D (M₀,β): {done_count}/{total} done "
                      f"({time.time()-t0:.1f}s)", flush=True)

    return profile_ll_2d


# ---------------------------------------------------------------------------
# 5. Verify unbounded (same logic, uses parallel 1D)
# ---------------------------------------------------------------------------

def verify_unbounded(M0_true, kappa_true, beta_true, N_true, n_per_point,
                     n_dilution, f_range=(1.0, 1e5), b=0.0, n_grid_ext=121):
    print(f"\n--- Verify unbounded: β={beta_true} (extended ±5 grid) ---")

    log_theta_true = np.log([M0_true, kappa_true, beta_true])
    fs = np.logspace(np.log10(f_range[0]), np.log10(f_range[1]), n_dilution)

    k_obs = np.zeros(n_dilution)
    for i, f in enumerate(fs):
        P = P_pos(f, M0_true, kappa_true, beta_true, N_true, b)
        P = np.clip(P, 1e-15, 1.0 - 1e-15)
        k_obs[i] = n_per_point * P
    n_obs = np.full(n_dilution, n_per_point)

    M0_grid_ext = np.linspace(np.log(M0_true) - 5, np.log(M0_true) + 5, n_grid_ext)
    pl_M0_ext, _ = profile_likelihood_1d(0, M0_grid_ext, log_theta_true,
                                           fs, k_obs, n_obs, N_true, b)

    ll_true = log_likelihood(log_theta_true, fs, k_obs, n_obs, N_true, b)
    delta_ext = ll_true - pl_M0_ext
    delta_ext = np.where(np.isfinite(delta_ext), delta_ext, 1e15)

    delta_thresh = chi2.ppf(0.95, 1) / 2.0

    right_delta = delta_ext[-1]
    i_mid = len(delta_ext) // 2
    right_portion = delta_ext[i_mid:]
    monotone_right = np.all(np.diff(right_portion) >= -0.1)

    print(f"  Δ log L at right edge (+5, M₀={np.exp(M0_grid_ext[-1]):.1e}): {right_delta:.2f}")
    print(f"  Monotonically increasing in right half: {monotone_right}")
    if right_delta > 10 and monotone_right:
        print(f"  → CI is genuinely unbounded above")
    elif right_delta > delta_thresh:
        print(f"  → CI upper bound exists but very wide")
    else:
        print(f"  → WARNING: CI may be bounded; extend grid further")

    left_delta = delta_ext[0]
    print(f"  Δ log L at left edge (-5, M₀={np.exp(M0_grid_ext[0]):.1e}): {left_delta:.2f}")

    return M0_grid_ext, pl_M0_ext


# ---------------------------------------------------------------------------
# 6. run_analysis (unchanged logic, calls parallel functions)
# ---------------------------------------------------------------------------

def run_analysis(M0_true, kappa_true, beta_true, N_true, n_per_point,
                 n_dilution, f_range=(1.0, 1e5), b=0.0,
                 param_name="default", n_grid_1d=81, n_grid_2d=41):
    print("=" * 80)
    print(f"Profile Likelihood Analysis: {param_name}")
    print(f"  M₀={M0_true:.2e}, κ={kappa_true}, β={beta_true}, N={N_true:.0e}")
    print(f"  n={n_per_point:.0e} per point, {n_dilution} dilution points, b={b}")
    print("=" * 80)

    log_theta_true = np.log([M0_true, kappa_true, beta_true])
    fs = np.logspace(np.log10(f_range[0]), np.log10(f_range[1]), n_dilution)

    k_obs = np.zeros(n_dilution)
    for i, f in enumerate(fs):
        P = P_pos(f, M0_true, kappa_true, beta_true, N_true, b)
        P = np.clip(P, 1e-15, 1.0 - 1e-15)
        k_obs[i] = n_per_point * P
    n_obs = np.full(n_dilution, n_per_point)

    ll_true = log_likelihood(log_theta_true, fs, k_obs, n_obs, N_true, b)
    print(f"\nTrue log-likelihood: {ll_true:.2f}")

    # ---- 1D Profiles (parallel) ----
    print(f"\n--- Profiling ln M₀ ({n_grid_1d} points, {N_WORKERS} workers) ---")
    t0 = time.time()
    M0_grid = np.linspace(np.log(M0_true) - 3, np.log(M0_true) + 3, n_grid_1d)
    pl_M0, opt_M0 = profile_likelihood_1d(0, M0_grid, log_theta_true,
                                            fs, k_obs, n_obs, N_true, b)
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\n--- Profiling ln κ ({n_grid_1d} points) ---")
    t0 = time.time()
    kappa_grid = np.linspace(np.log(kappa_true) - 3, np.log(kappa_true) + 3, n_grid_1d)
    pl_kappa, opt_kappa = profile_likelihood_1d(1, kappa_grid, log_theta_true,
                                                  fs, k_obs, n_obs, N_true, b)
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\n--- Profiling ln β ({n_grid_1d} points) ---")
    t0 = time.time()
    beta_grid = np.linspace(np.log(beta_true) - 3, np.log(beta_true) + 3, n_grid_1d)
    pl_beta, opt_beta = profile_likelihood_1d(2, beta_grid, log_theta_true,
                                                fs, k_obs, n_obs, N_true, b)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ---- 2D Profiles (parallel) ----
    print(f"\n--- 2D Profile (ln M₀, ln κ) ({n_grid_2d}×{n_grid_2d} = {n_grid_2d**2} points) ---")
    t0 = time.time()
    M0_grid_2d = np.linspace(np.log(M0_true) - 3, np.log(M0_true) + 3, n_grid_2d)
    kappa_grid_2d = np.linspace(np.log(kappa_true) - 3, np.log(kappa_true) + 3, n_grid_2d)
    pl_2d = profile_likelihood_2d(M0_grid_2d, kappa_grid_2d, beta_true,
                                   fs, k_obs, n_obs, N_true, b)
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\n--- 2D Profile (ln M₀, ln β) ({n_grid_2d}×{n_grid_2d} points) ---")
    t0 = time.time()
    beta_grid_2d = np.linspace(np.log(beta_true) - 3, np.log(beta_true) + 3, n_grid_2d)
    pl_2d_Mb = profile_likelihood_2d_M0_beta(M0_grid_2d, beta_grid_2d, kappa_true,
                                              fs, k_obs, n_obs, N_true, b)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ---- CI (same as before) ----
    chi2_thresh = chi2.ppf(0.95, 1)
    delta_ll_thresh = chi2_thresh / 2.0

    def get_ci_1d(grid, pl, ll_opt):
        delta = ll_opt - pl
        delta = np.where(np.isfinite(delta), delta, 1e15)
        i_opt = np.argmin(delta)
        ci_low = None
        ci_high = None
        for i in range(i_opt - 1, -1, -1):
            if delta[i] >= delta_ll_thresh:
                frac = (delta[i] - delta_ll_thresh) / (delta[i] - delta[i+1]) if (delta[i] - delta[i+1]) != 0 else 0.5
                ci_low = np.exp(grid[i] + frac * (grid[i+1] - grid[i]))
                break
        for i in range(i_opt, len(grid) - 1):
            if delta[i] >= delta_ll_thresh:
                frac = (delta[i] - delta_ll_thresh) / (delta[i] - delta[i+1]) if (delta[i] - delta[i+1]) != 0 else 0.5
                ci_high = np.exp(grid[i] + frac * (grid[i+1] - grid[i]))
                break
        bounded = ci_low is not None and ci_high is not None
        return ci_low, ci_high, bounded

    ci_M0 = get_ci_1d(M0_grid, pl_M0, ll_true)
    ci_kappa = get_ci_1d(kappa_grid, pl_kappa, ll_true)
    ci_beta = get_ci_1d(beta_grid, pl_beta, ll_true)

    # ---- Print results ----
    print(f"\n{'='*80}")
    print(f"RESULTS: {param_name}")
    print(f"{'='*80}")
    print(f"\n95% CI (χ²(1,0.95) = {chi2_thresh:.3f}, ΔlogL threshold = {delta_ll_thresh:.4f}):")
    for name, ci, true_val in [("M₀", ci_M0, M0_true), ("κ", ci_kappa, kappa_true), ("β", ci_beta, beta_true)]:
        if ci[2]:
            ratio = ci[1] / ci[0]
            print(f"  {name}: true={true_val:.4g}  CI=[{ci[0]:.4g}, {ci[1]:.4g}]  ratio={ratio:.1f}x")
        else:
            print(f"  {name}: true={true_val:.4g}  CI=UNBOUNDED")

    # ---- Ridge slopes ----
    ridge_M0, ridge_kappa = [], []
    for iM in range(n_grid_2d):
        iK_best = np.argmax(pl_2d[iM, :])
        if pl_2d[iM, iK_best] > -np.inf:
            ridge_M0.append(M0_grid_2d[iM])
            ridge_kappa.append(kappa_grid_2d[iK_best])
    ridge_M0 = np.array(ridge_M0)
    ridge_kappa = np.array(ridge_kappa)
    slope = np.polyfit(ridge_M0, ridge_kappa, 1)[0] if len(ridge_M0) > 5 else None

    ridge_M0_b, ridge_beta = [], []
    for iM in range(n_grid_2d):
        iB_best = np.argmax(pl_2d_Mb[iM, :])
        if pl_2d_Mb[iM, iB_best] > -np.inf:
            ridge_M0_b.append(M0_grid_2d[iM])
            ridge_beta.append(beta_grid_2d[iB_best])
    ridge_M0_b = np.array(ridge_M0_b)
    ridge_beta = np.array(ridge_beta)
    slope_Mb = np.polyfit(ridge_M0_b, ridge_beta, 1)[0] if len(ridge_M0_b) > 5 else None

    if slope is not None:
        print(f"\n  Ridge slope (M₀,κ) = {slope:.3f}  (scale degeneracy → 0)")
    if slope_Mb is not None:
        print(f"  Ridge slope (M₀,β) = {slope_Mb:.3f}  (scale degeneracy → 1)")

    # ---- Save ----
    np.savez(f'profile_likelihood_{param_name}.npz',
             M0_grid=M0_grid, pl_M0=pl_M0, opt_M0=opt_M0,
             kappa_grid=kappa_grid, pl_kappa=pl_kappa, opt_kappa=opt_kappa,
             beta_grid=beta_grid, pl_beta=pl_beta, opt_beta=opt_beta,
             M0_grid_2d=M0_grid_2d, kappa_grid_2d=kappa_grid_2d, pl_2d=pl_2d,
             beta_grid_2d=beta_grid_2d, pl_2d_Mb=pl_2d_Mb,
             ll_true=ll_true, log_theta_true=log_theta_true,
             ridge_M0=ridge_M0, ridge_kappa=ridge_kappa,
             ridge_M0_b=ridge_M0_b, ridge_beta=ridge_beta,
             ci_M0=ci_M0, ci_kappa=ci_kappa, ci_beta=ci_beta)

    return {
        'ci_M0': ci_M0, 'ci_kappa': ci_kappa, 'ci_beta': ci_beta,
        'ridge_slope': slope, 'ridge_slope_Mb': slope_Mb,
        'pl_M0': pl_M0, 'pl_kappa': pl_kappa, 'pl_beta': pl_beta,
        'M0_grid': M0_grid, 'kappa_grid': kappa_grid, 'beta_grid': beta_grid,
        'pl_2d': pl_2d, 'M0_grid_2d': M0_grid_2d, 'kappa_grid_2d': kappa_grid_2d,
        'pl_2d_Mb': pl_2d_Mb, 'beta_grid_2d': beta_grid_2d,
        'ridge_M0': ridge_M0, 'ridge_kappa': ridge_kappa,
        'ridge_M0_b': ridge_M0_b, 'ridge_beta': ridge_beta,
        'll_true': ll_true,
    }


# ---------------------------------------------------------------------------
# 7. Plotting (unchanged)
# ---------------------------------------------------------------------------

def plot_results(results, param_name, M0_true, kappa_true, beta_true):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    delta_thresh = chi2.ppf(0.95, 1) / 2.0

    for ax, key, grid_key, true_val, title in [
        (axes[0,0], 'pl_M0', 'M0_grid', M0_true, 'M₀'),
        (axes[0,1], 'pl_kappa', 'kappa_grid', kappa_true, 'κ'),
        (axes[1,0], 'pl_beta', 'beta_grid', beta_true, 'β'),
    ]:
        delta = results['ll_true'] - results[key]
        ax.plot(np.exp(results[grid_key]), delta, 'b-', linewidth=2)
        ax.axhline(delta_thresh, color='r', linestyle='--', label='95% CI')
        ax.axvline(true_val, color='g', linestyle=':', label=f'True={true_val:.4g}')
        ax.set_xlabel(title)
        ax.set_ylabel('Δ log L')
        ax.set_title(f'Profile Likelihood: {title}')
        ax.set_xscale('log')
        ax.legend()
        ax.set_ylim(bottom=-0.5)

    # Panel 4: 2D (M₀, κ)
    ax = axes[1, 1]
    M0_2d = np.exp(results['M0_grid_2d'])
    kappa_2d = np.exp(results['kappa_grid_2d'])
    delta_2d = results['ll_true'] - results['pl_2d']
    delta_2d = np.ma.masked_invalid(delta_2d)
    levels = [l for l in [0.5, 1.0, 1.92, 5.0, 10.0, 50.0] if l < delta_2d.max()]
    if len(levels) > 1:
        CS = ax.contourf(M0_2d, kappa_2d, delta_2d, levels=levels, cmap='YlOrRd')
        plt.colorbar(CS, ax=ax, label='Δ log L')
    ax.plot(M0_true, kappa_true, 'g*', markersize=15, label='True')
    if len(results['ridge_M0']) > 0:
        ax.plot(np.exp(results['ridge_M0']), np.exp(results['ridge_kappa']),
                'k--', linewidth=2, label='Ridge')
    ax.axhline(kappa_true, color='b', linestyle=':', linewidth=1, alpha=0.5,
               label='Scale degen. (κ inv.)')
    ax.set_xlabel(r'$M_0$'); ax.set_ylabel('κ')
    ax.set_title(r'2D Profile: ($M_0$, κ)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'profile_likelihood_{param_name}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_results_M0beta(results, param_name, M0_true, kappa_true, beta_true):
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    M0_2d = np.exp(results['M0_grid_2d'])
    beta_2d = np.exp(results['beta_grid_2d'])
    delta_2d = results['ll_true'] - results['pl_2d_Mb']
    delta_2d = np.ma.masked_invalid(delta_2d)
    levels = [l for l in [0.5, 1.0, 1.92, 5.0, 10.0, 50.0] if l < delta_2d.max()]
    if len(levels) > 1:
        CS = ax.contourf(M0_2d, beta_2d, delta_2d, levels=levels, cmap='YlOrRd')
        plt.colorbar(CS, ax=ax, label='Δ log L')
    ax.plot(M0_true, beta_true, 'g*', markersize=15, label='True')
    if len(results.get('ridge_M0_b', [])) > 0:
        ax.plot(np.exp(results['ridge_M0_b']), np.exp(results['ridge_beta']),
                'k--', linewidth=2, label='Ridge')
    log_M0_range = results['M0_grid_2d']
    log_beta_line = log_M0_range - np.log(M0_true) + np.log(beta_true)
    ax.plot(np.exp(log_M0_range), np.exp(log_beta_line),
            'b:', linewidth=1.5, alpha=0.6, label='Scale degen. (slope=1)')
    ax.set_xlabel(r'$M_0$'); ax.set_ylabel('β')
    ax.set_title(rf'2D Profile: ($M_0$, β)  [β_true={beta_true}]')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f'profile_likelihood_{param_name}_M0beta.png', dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    M0_true = 1e7
    kappa_true = 8.0
    N_true = 5e5
    n_per_point = 1e5
    n_dilution = 50
    f_range = (1.0, 1e5)
    b = 0.0
    n_grid_1d = 81
    n_grid_2d = 41

    all_results = {}

    for beta in [1, 2, 5, 20]:
        param_name = f"beta{beta}"
        print(f"\n{'#'*80}")
        print(f"# β = {beta}")
        print(f"{'#'*80}")

        results = run_analysis(
            M0_true=M0_true, kappa_true=kappa_true, beta_true=beta,
            N_true=N_true, n_per_point=n_per_point, n_dilution=n_dilution,
            f_range=f_range, b=b, param_name=param_name,
            n_grid_1d=n_grid_1d, n_grid_2d=n_grid_2d,
        )
        plot_results(results, param_name, M0_true, kappa_true, beta)
        plot_results_M0beta(results, param_name, M0_true, kappa_true, beta)
        all_results[beta] = results

    # Verify unbounded
    print(f"\n{'#'*80}")
    print(f"# Verify unbounded CI: extended ±5 grid for β≥2")
    print(f"{'#'*80}")
    for beta in [2, 5, 20]:
        verify_unbounded(M0_true, kappa_true, beta, N_true, n_per_point,
                         n_dilution, f_range, b)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'β':>4s} | {'M₀ CI':>12s} | {'κ CI':>12s} | {'β CI':>12s} | {'slope_Mκ':>10s} | {'slope_Mβ':>10s} | {'ident':>8s}")
    print("-" * 85)
    for beta in [1, 2, 5, 20]:
        r = all_results[beta]
        ci_M0, ci_kappa, ci_beta = r['ci_M0'], r['ci_kappa'], r['ci_beta']
        s_Mk = r['ridge_slope']
        s_Mb = r['ridge_slope_Mb']

        m0_str = f"{ci_M0[1]/ci_M0[0]:.1f}x" if ci_M0[2] else "UNBOUNDED"
        k_str = f"{ci_kappa[1]/ci_kappa[0]:.1f}x" if ci_kappa[2] else "UNBOUNDED"
        b_str = f"{ci_beta[1]/ci_beta[0]:.1f}x" if ci_beta[2] else "UNBOUNDED"
        sMk_str = f"{s_Mk:.3f}" if s_Mk is not None else "N/A"
        sMb_str = f"{s_Mb:.3f}" if s_Mb is not None else "N/A"

        if ci_M0[2]:
            m0_ratio = ci_M0[1] / ci_M0[0]
            ident = "YES" if m0_ratio < 10 else ("POOR" if m0_ratio < 100 else "NO")
        else:
            ident = "NO"

        print(f"{beta:>4d} | {m0_str:>12s} | {k_str:>12s} | {b_str:>12s} | {sMk_str:>10s} | {sMb_str:>10s} | {ident:>8s}")

    print("\nDone. Profile Likelihood complete.")

    # ================================================================
    # Fisher Matrix Analysis (fast, sequential)
    # ================================================================
    run_fisher()


def run_fisher():
    """Script 2: Fisher information matrix analysis."""
    M0 = 1e7
    kappa = 8.0
    N = 5e5
    n = 1e5
    n_pts = 50
    f_min, f_max = 1.0, 1e5

    def fisher_matrix(M0, kappa, beta, N, n_per_point=1e5,
                      f_min=1.0, f_max=1e5, n_points=50):
        fs = np.logspace(np.log10(f_min), np.log10(f_max), n_points)
        I = np.zeros((3, 3))
        for f in fs:
            xi = M0 / (f * kappa * N * beta)
            p = solve_p(xi, kappa)
            P = 1.0 - (1.0 - p) ** beta
            if P <= 1e-15 or P >= 1.0 - 1e-15:
                continue
            w = n_per_point / (P * (1.0 - P))
            dp_dxi = kappa * (1.0 - p) ** 2 / (kappa + (1.0 - p) ** 2)
            dxi_dlnM0 = xi
            dxi_dlnkappa = -xi
            dp_dlnkappa_direct = p * (1.0 - p) ** 2 / (kappa + (1.0 - p) ** 2)
            dp_dlnkappa_total = dp_dxi * dxi_dlnkappa + dp_dlnkappa_direct
            dp_dlnM0 = dp_dxi * dxi_dlnM0
            dP_dp = beta * (1.0 - p) ** (beta - 1)
            dP_dlnM0 = dP_dp * dp_dlnM0
            dP_dlnkappa = dP_dp * dp_dlnkappa_total
            dP_dlnbeta_direct = -beta * (1.0 - p) ** beta * np.log(1.0 - p)
            dp_dlnbeta = dp_dxi * (-xi)
            dP_dlnbeta = dP_dp * dp_dlnbeta + dP_dlnbeta_direct
            jac = np.array([dP_dlnM0, dP_dlnkappa, dP_dlnbeta])
            I += w * np.outer(jac, jac)
        evals, evecs = np.linalg.eigh(I)
        lmin = evals[0]
        lmax = evals[-1]
        chi = lmax / lmin if lmin > 0 else np.inf
        ddir = evecs[:, 0]
        if ddir[0] < 0:
            ddir = -ddir
        return lmin, lmax, chi, ddir

    def fisher_matrix_2d(M0, kappa, N, n_per_point=1e5,
                         f_min=1.0, f_max=1e5, n_points=50):
        fs = np.logspace(np.log10(f_min), np.log10(f_max), n_points)
        I = np.zeros((2, 2))
        for f in fs:
            xi = M0 / (f * kappa * N)
            p = solve_p(xi, kappa)
            P = p
            if P <= 1e-15 or P >= 1.0 - 1e-15:
                continue
            w = n_per_point / (P * (1.0 - P))
            dp_dxi = kappa * (1.0 - p) ** 2 / (kappa + (1.0 - p) ** 2)
            dxi_dlnM0 = xi
            dxi_dlnkappa = -xi
            dp_dlnkappa_direct = p * (1.0 - p) ** 2 / (kappa + (1.0 - p) ** 2)
            dp_dlnkappa_total = dp_dxi * dxi_dlnkappa + dp_dlnkappa_direct
            dp_dlnM0 = dp_dxi * dxi_dlnM0
            jac = np.array([dp_dlnM0, dp_dlnkappa_total])
            I += w * np.outer(jac, jac)
        evals = np.linalg.eigvalsh(I)
        lmin = evals[0]
        lmax = evals[-1]
        chi = lmax / lmin if lmin > 0 else np.inf
        PR = (lmin + lmax) ** 2 / (lmin ** 2 + lmax ** 2) if lmin > 0 else 1.0
        return lmin, lmax, chi, PR

    # ---- Table S2c.5.1 ----
    print("\n" + "=" * 90)
    print("Table S2c.5.1: Fisher information matrix eigenvalues and condition number")
    print(f"  M0={M0:.0e}, kappa={kappa}, N={N:.0e}, n={n:.0e} per point, {n_pts} dilution points")
    print("=" * 90)
    print(f"{'beta':>4s}  {'lmin':>12s}  {'lmax':>12s}  {'chi':>12s}  {'dir':>30s}")
    print("-" * 90)
    for beta in [1, 2, 5, 20]:
        lmin, lmax, chi, d = fisher_matrix(M0, kappa, beta, N, n, f_min, f_max, n_pts)
        print(f"{beta:>4d}  {lmin:>12.3f}  {lmax:>12.2e}  {chi:>12.2e}  ({d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f})")

    # ---- Wide scan ----
    print("\n" + "=" * 90)
    print("Wide parameter scan")
    print("=" * 90)
    configs = []
    for M0_t in [1e4, 1e5, 1e6, 1e7, 1e8]:
        for kappa_t in [1, 5, 10, 20, 50]:
            for beta_t in [1, 2, 5, 20, 50]:
                for N_t in [1e4, 1e5, 1e6]:
                    for npts_t in [7, 14, 50]:
                        configs.append((M0_t, kappa_t, beta_t, N_t, npts_t))
    best_chi = np.inf
    best_config = None
    worst_chi = 0
    worst_config = None
    all_chis = []
    for M0_t, kappa_t, beta_t, N_t, npts_t in configs:
        _, _, chi, _ = fisher_matrix(M0_t, kappa_t, beta_t, N_t, 1e5, 1.0, 1e5, npts_t)
        all_chis.append((chi, M0_t, kappa_t, beta_t, N_t, npts_t))
        if chi < best_chi:
            best_chi = chi
            best_config = (M0_t, kappa_t, beta_t, N_t, npts_t)
        if chi > worst_chi:
            worst_chi = chi
            worst_config = (M0_t, kappa_t, beta_t, N_t, npts_t)
    print(f"Scanned {len(configs)} configurations")
    print(f"Best  chi = {best_chi:.2e}  at {best_config}")
    print(f"Worst chi = {worst_chi:.2e}  at {worst_config}")

    # kappa>=5 subset
    k5 = [(c, M, k, b, N, n) for c, M, k, b, N, n in all_chis if k >= 5]
    best_k5 = min(k5, key=lambda x: x[0])
    worst_k5 = max(k5, key=lambda x: x[0])
    print(f"\nk>=5 subset ({len(k5)} configs):")
    print(f"  Best  chi = {best_k5[0]:.2e}  at (M0={best_k5[1]:.0e}, k={best_k5[2]}, b={best_k5[3]}, N={best_k5[4]:.0e}, npts={best_k5[5]})")
    print(f"  Worst chi = {worst_k5[0]:.2e}")

    # ---- Extreme design ----
    print("\n" + "=" * 90)
    print("Extreme design: 98 points, n=1e8, beta=1, kappa=8")
    print("=" * 90)
    _, _, chi, _ = fisher_matrix(1e7, 8.0, 1, 5e5, 1e8, 1.0, 1e7, 98)
    print(f"chi = {chi:.2e}")

    # ---- kappa->0 escape ----
    print("\n" + "=" * 90)
    print("kappa->0 escape (2D Fisher, beta=1 fixed)")
    print(f"  (M0={M0:.0e}, beta=1, N={N:.0e}, n={n:.0e}/point, {n_pts} points)")
    print("  PR->2: both params carry info | PR->1: one param vanishes (rank loss)")
    print("=" * 90)
    print(f"{'kappa':>8s}  {'lmin':>12s}  {'lmax':>12s}  {'chi':>12s}  {'PR':>8s}")
    print("-" * 58)
    for kappa_t in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        lmin, lmax, chi, PR = fisher_matrix_2d(M0, kappa_t, N, n, f_min, f_max, n_pts)
        print(f"{kappa_t:>8.3f}  {lmin:>12.3f}  {lmax:>12.2e}  {chi:>12.2e}  {PR:>8.3f}")

    # ---- Replicate scaling ----
    print("\n" + "=" * 90)
    print("Replicate scaling: chi invariant under n -> lambda*n (kappa=8, beta=5)")
    print("=" * 90)
    print(f"{'n/point':>12s}  {'lmin':>12s}  {'lmax':>12s}  {'chi':>12s}")
    print("-" * 55)
    for n_t in [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]:
        lmin, lmax, chi, _ = fisher_matrix(M0, kappa, 5, N, n_t, f_min, f_max, n_pts)
        print(f"{n_t:>12.0e}  {lmin:>12.3f}  {lmax:>12.2e}  {chi:>12.2e}")

    # ---- Dilution design ----
    print("\n" + "=" * 90)
    print("Dilution design: chi vs number of points (kappa=8, beta=5, n=1e5)")
    print("=" * 90)
    print(f"{'n_points':>10s}  {'lmin':>12s}  {'lmax':>12s}  {'chi':>12s}")
    print("-" * 52)
    for npts_t in [5, 7, 14, 30, 50, 98, 200]:
        lmin, lmax, chi, _ = fisher_matrix(M0, kappa, 5, N, n, f_min, f_max, npts_t)
        print(f"{npts_t:>10d}  {lmin:>12.3f}  {lmax:>12.2e}  {chi:>12.2e}")

    print("\nFisher analysis complete.")


if __name__ == "__main__":
    main()