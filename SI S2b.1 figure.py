# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:15:28 2026

@author: lihua
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 21:24:35 2026

@author: lihua
"""

# -*- coding: utf-8 -*-
"""
Figure S2b.1: CV of the molecular count estimator M.
Exact four-layer Monte Carlo (solid) vs R1 Binomial delta-method (dashed, S2f.9).
- Supports non-zero background b.
- Uses parallel computing (joblib) for MC trials.
- Both panels use M as x-axis.

@author: lihua
Fixed 2026-07-22:
  - Bug fix: P_C_4L now normalizes by partition function Q(W) for each W
  - Conclusion section checks VR_4L ~ 1.0 (the actual SI claim), not ICC_3L vs ICC_4L
  - R1/4L ratio protected against CV_4L -> 0
  - np.std uses ddof=1 (sample standard deviation)
Fixed 2026-07-29 (audit revision):
  - solve_p rationalized: p = 2u/(u+kappa+1+sqrt(disc)); the old form
    (-b - sqrt(disc))/2 suffers catastrophic cancellation as u -> 0
    (panel parameters u >= 1e-6 were unaffected; fixed for consistency
    with SI S2b table s2b2 v5.py)
  - sample_canonical: searchsorted index clamped to max_C (cdf[-1] may fall
    short of 1.0 by ~1e-16 in floating point, returning max_C+1 otherwise)
  - bottom panel: annotation that the R1 curve leaves the axis range at low M
    (the CV < 1.5 plot mask otherwise hides the 1.5-250x divergence reported
    in the caption)
  - legacy Table S2b.2 verification block removed from this file (superseded
    by SI S2b table s2b2 v5.py; its "Fano identity" conclusions were outdated)
"""

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import matplotlib

# ============================================================
# Shared functions
# ============================================================

def solve_p(M, kappa, Omega):
    """Solve TCS master equation for p (rationalized root, stable as u -> 0)."""
    u = M / Omega
    disc = (u + kappa + 1.0)**2 - 4.0 * u
    if disc < 0:
        return None
    p = 2.0 * u / (u + kappa + 1.0 + np.sqrt(disc))
    return p if 0 < p < 1 else None


def cv_r1_delta(M, kappa, Omega, N, n, beta, b=0.0):
    """R1 delta-method CV (S2f.9), with background b."""
    p = solve_p(M, kappa, Omega)
    if p is None:
        return np.nan
    P_spec = 1 - (1 - p)**beta
    P_pos = b + (1 - b) * P_spec
    if P_pos <= 0 or P_pos >= 1:
        return np.nan
    x = 1 - P_spec
    hp = N / (1 - b) * (kappa * x**(-1.0/beta - 1) + x**(1.0/beta - 1))
    var_p = P_pos * (1 - P_pos) / n
    return abs(hp) * np.sqrt(var_p) / M


def r1_invert(P_hat, kappa, Omega, beta, b=0.0):
    """R1 inversion: P_hat -> M_hat, with background correction."""
    P_spec_hat = (P_hat - b) / (1 - b)
    if P_spec_hat <= 0 or P_spec_hat >= 1:
        return np.nan
    x = 1 - P_spec_hat
    pp = 1 - x**(1.0/beta)
    xi_hat = pp / (1 - pp) + pp / kappa
    return xi_hat * kappa * Omega


def sample_canonical(W, kappa, Omega, rng):
    """Sample C ~ P(C|W) from canonical ensemble (S1c.3)."""
    if W == 0:
        return 0
    max_C = min(W, Omega)
    log_binom_W = np.zeros(max_C + 1)
    log_binom_O = np.zeros(max_C + 1)
    log_fact_C = np.zeros(max_C + 1)
    if max_C >= 1:
        num_W = W - np.arange(1, max_C + 1) + 1
        num_O = Omega - np.arange(1, max_C + 1) + 1
        den = np.arange(1, max_C + 1)
        log_binom_W[1:] = np.cumsum(np.log(num_W / den))
        log_binom_O[1:] = np.cumsum(np.log(num_O / den))
        log_fact_C[1:] = np.cumsum(np.log(np.arange(1, max_C + 1, dtype=float)))
    Cs = np.arange(max_C + 1, dtype=float)
    log_probs = log_binom_W + log_binom_O + log_fact_C - Cs * np.log(kappa * Omega)
    log_probs -= np.max(log_probs)
    probs = np.exp(log_probs)
    probs /= probs.sum()
    cdf = np.cumsum(probs)
    return int(min(np.searchsorted(cdf, rng.random()), max_C))


# ============================================================
# MC trial functions (return partition-level indicators)
# ============================================================

def truly_4layer_trial_indicators(seed, M, kappa, Omega, N, n, beta):
    """One trial: canonical + hypergeometric. Returns n booleans (binding-positive)."""
    rng = np.random.default_rng(seed)
    W = rng.poisson(M)
    if W == 0:
        return np.zeros(n, dtype=bool)
    C = sample_canonical(W, kappa, Omega, rng)
    if C > Omega:
        C = Omega
    if C == 0:
        return np.zeros(n, dtype=bool)
    sites = rng.choice(Omega, size=C, replace=False)
    partition_idx = sites // beta
    pos_partitions = np.unique(partition_idx[partition_idx < n])
    indicators = np.zeros(n, dtype=bool)
    indicators[pos_partitions] = True
    return indicators


def dilute_4layer_trial_indicators(seed, M, kappa, Omega, N, n, beta):
    """One trial: Poisson thinning + multinomial. Returns n booleans (binding-positive)."""
    rng = np.random.default_rng(seed)
    gamma = 1.0 / (1.0 + kappa)
    C = rng.poisson(M * gamma)
    if C == 0:
        return np.zeros(n, dtype=bool)
    parts = rng.integers(0, N, size=C)
    obs = parts[parts < n]
    pos = np.unique(obs)
    indicators = np.zeros(n, dtype=bool)
    indicators[pos] = True
    return indicators


# ============================================================
# CV estimators (parallel)
# ============================================================

def cv_truly_4layer(M, kappa, Omega, N, n, beta, b=0.0, n_trials=5000, seed=42):
    """Parallel truly 4-layer MC CV."""
    seeds = [seed + i for i in range(n_trials)]
    indicators = Parallel(n_jobs=-1)(
        delayed(truly_4layer_trial_indicators)(s, M, kappa, Omega, N, n, beta)
        for s in seeds
    )
    rng_bg = np.random.default_rng(seed + n_trials)
    Z_arr = np.array([
        np.sum(ind | (rng_bg.random(n) < b)) for ind in indicators
    ])
    P_hat = Z_arr / n
    valid = (P_hat > b) & (P_hat < 1.0)
    if not np.any(valid):
        return np.nan
    M_hats = np.array([r1_invert(ph, kappa, Omega, beta, b) for ph in P_hat[valid]])
    mean_m = np.mean(M_hats)
    if mean_m <= 0:
        return np.nan
    return np.std(M_hats, ddof=1) / mean_m


def cv_dilute_4layer(M, kappa, Omega, N, n, beta, b=0.0, n_trials=30000, seed=42):
    """Parallel dilute 4-layer MC CV."""
    seeds = [seed + i for i in range(n_trials)]
    indicators = Parallel(n_jobs=-1)(
        delayed(dilute_4layer_trial_indicators)(s, M, kappa, Omega, N, n, beta)
        for s in seeds
    )
    rng_bg = np.random.default_rng(seed + n_trials)
    Z_arr = np.array([
        np.sum(ind | (rng_bg.random(n) < b)) for ind in indicators
    ])
    P_hat = Z_arr / n
    valid = (P_hat > b) & (P_hat < 1.0)
    if not np.any(valid):
        return np.nan
    M_hats = np.array([r1_invert(ph, kappa, Omega, beta, b) for ph in P_hat[valid]])
    mean_m = np.mean(M_hats)
    if mean_m <= 0:
        return np.nan
    return np.std(M_hats, ddof=1) / mean_m


# ============================================================
# Compute and plot
# ============================================================

# --- Parameters ---
N1 = 50; n1 = 50; beta1 = 20; Omega1 = N1 * beta1  # 1000
kappas1 = [0.1, 1.0, 10.0]
colors1 = ['#1f77b4', '#ff7f0e', '#2ca02c']
Ms1 = np.array([1, 2, 5, 10, 20, 50, 100, 200, 350, 500, 800])
b1 = 0.01

N2 = 20000; n2 = 15000; beta2 = 50; Omega2 = N2 * beta2  # 1e6
kappas2 = [0.01, 0.1, 1.0, 10.0]
colors2 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
Ms2 = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
b2 = 0.01

# --- Compute Panel 1 ---
print("=== Panel 1: Non-dilute, truly 4-layer ===")
results1 = {}
for kappa in kappas1:
    print(f"  kappa={kappa}...")
    CV_ex = np.full(len(Ms1), np.nan)
    CV_r1 = np.full(len(Ms1), np.nan)
    for i, M in enumerate(Ms1):
        CV_ex[i] = cv_truly_4layer(M, kappa, Omega1, N1, n1, beta1, b=b1, n_trials=5000)
        CV_r1[i] = cv_r1_delta(M, kappa, Omega1, N1, n1, beta1, b=b1)
        r = CV_r1[i]/CV_ex[i] if (np.isfinite(CV_ex[i]) and CV_ex[i] > 1e-6) else np.nan
        print(f"    M={M:5d}  CV_4L={CV_ex[i]:.4f}  CV_R1={CV_r1[i]:.4f}  R1/4L={r:.4f}")
    results1[kappa] = (CV_ex, CV_r1)

# --- Compute Panel 2 ---
print("\n=== Panel 2: Dilute, large N ===")
results2 = {}
for kappa in kappas2:
    print(f"  kappa={kappa}...")
    CV_ex = np.full(len(Ms2), np.nan)
    CV_r1 = np.full(len(Ms2), np.nan)
    for i, M in enumerate(Ms2):
        CV_ex[i] = cv_dilute_4layer(M, kappa, Omega2, N2, n2, beta2, b=b2, n_trials=30000)
        CV_r1[i] = cv_r1_delta(M, kappa, Omega2, N2, n2, beta2, b=b2)
        r = CV_r1[i]/CV_ex[i] if (np.isfinite(CV_ex[i]) and CV_ex[i] > 1e-6) else np.nan
        print(f"    M={M:7d}  CV_4L={CV_ex[i]:.4f}  CV_R1={CV_r1[i]:.4f}  R1/4L={r:.4f}")
    results2[kappa] = (CV_ex, CV_r1)

# --- Plot ---
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 14))

# Overall title
fig.suptitle('Fig. S2b.1. CV of the molecular count estimator $M$\n'
             'Exact 4-layer Monte Carlo (solid) vs R1 delta-method (dashed)',
             fontsize=15, fontweight='bold', y=0.98)

for kappa, color in zip(kappas1, colors1):
    CV_ex, CV_r1 = results1[kappa]
    valid = np.isfinite(CV_ex) & np.isfinite(CV_r1) & (CV_ex < 1.5) & (CV_r1 < 1.5)
    ax_top.plot(Ms1[valid], CV_ex[valid], 'o-', color=color, linewidth=2, markersize=5,
                label=f'4-layer exact ($\\kappa={kappa}$)')
    ax_top.plot(Ms1[valid], CV_r1[valid], '--', color=color, linewidth=1.2, alpha=0.7,
                label=f'R1 ($\\kappa={kappa}$)')

ax_top.set_xscale('log')
ax_top.set_xlabel('$M$ (molecules)', fontsize=14)
ax_top.set_ylabel('CV', fontsize=14)
ax_top.set_title('a.Non-dilute: truly 4-layer MC (canonical + hypergeometric) vs R1\n'
                 f'$N={N1}$, $n={n1}$, $\\beta={beta1}$ ($\\Omega={Omega1}$), $b={b1}$', fontsize=14)
ax_top.set_ylim(0, 1.4)
ax_top.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
ax_top.legend(fontsize=11, loc='upper right', ncol=2)

# # Panel a label
# ax_top.text(-0.12, 1.02, 'a', transform=ax_top.transAxes,
#             fontsize=18, fontweight='bold', va='bottom', ha='right')


for kappa, color in zip(kappas2, colors2):
    CV_ex, CV_r1 = results2[kappa]
    valid = np.isfinite(CV_ex) & np.isfinite(CV_r1) & (CV_ex < 1.5) & (CV_r1 < 1.5)
    ax_bot.plot(Ms2[valid], CV_ex[valid], 'o-', color=color, linewidth=2, markersize=3,
                label=f'4-layer MC ($\\kappa={kappa}$)')
    ax_bot.plot(Ms2[valid], CV_r1[valid], '--', color=color, linewidth=1.2, alpha=0.7,
                label=f'R1 ($\\kappa={kappa}$)')

ax_bot.set_xscale('log')
ax_bot.set_xlabel('$M$ (molecules)', fontsize=14)
ax_bot.set_ylabel('CV', fontsize=14)
ax_bot.set_title('b.Dilute: 4-layer MC (Poisson thinning + multinomial) vs R1\n'
                 f'$N={N2:,}$, $n={n2:,}$, $\\beta={beta2}$ ($\\Omega={Omega2:,}$), $b={b2}$', fontsize=14)
ax_bot.set_ylim(0, 1.3)
ax_bot.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
# ax_bot.text(0.02, 0.03, 'R1 leaves the axis range at low $M$\n(up to 250$\\times$; see caption)',
#             transform=ax_bot.transAxes, fontsize=10, va='bottom', color='gray')
ax_bot.legend(fontsize=11, loc='upper right', ncol=2)

# # Panel b label
# ax_bot.text(-0.12, 1.02, 'b', transform=ax_bot.transAxes,
#             fontsize=18, fontweight='bold', va='bottom', ha='right')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('CV_exact_vs_R1.svg', dpi=300, bbox_inches='tight')

plt.show()

print("\nDone: CV_exact_vs_R1.svg")