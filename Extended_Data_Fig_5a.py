# -*- coding: utf-8 -*-
"""
Extended Data Fig. 5a: 4PL vs 5PL vs exact TCS comparison.

Shows how 4PL and 5PL approximate the exact TCS root at different κ regimes.
Three curves per κ:
  - TCS exact (solid): quadratic root of p² - (κξ+κ+1)p + κξ = 0
  - 5PL constrained (dashed): B=log2((4κ+1)/(2κ)), BG=1, match at p=0.5
  - 4PL (dotted): B=1, symmetric log-logistic

κ values: ∞, 10, 1, 0.1
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================
# Font settings (consistent with original figure)
# ============================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14


# ============================================
# Model functions
# ============================================
def tcs_exact(xi, kappa):
    """Exact TCS: solve p^2 - (kappa*xi + kappa + 1)*p + kappa*xi = 0"""
    if kappa == np.inf:
        return xi / (1 + xi)
    a = 1.0
    b = -(kappa * xi + kappa + 1)
    c = kappa * xi
    disc = b**2 - 4 * a * c
    p = (-b - np.sqrt(disc)) / (2 * a)
    return p


def five_pl(xi, kappa):
    """5PL with B=log2((4κ+1)/(2κ)), G=1/B, matching TCS at p=0.5 (SI S3.10)"""
    if kappa == np.inf:
        return xi / (1 + xi)
    B = np.log2((4 * kappa + 1) / (2 * kappa))
    xi50 = 1 + 1 / (2 * kappa)
    val = 1 + (2**B - 1) * (xi / xi50)**(-B)
    return val**(-1 / B)


def four_pl(xi, kappa):
    """4PL: B=1, symmetric log-logistic. p = xi/(xi+xi50)"""
    if kappa == np.inf:
        return xi / (1 + xi)
    xi50 = 1 + 1 / (2 * kappa)
    return xi / (xi + xi50)


# ============================================
# Plot
# ============================================
kappas = [np.inf, 10, 1, 0.1]
kappa_labels = [r'$\kappa \to \infty$', r'$\kappa = 10$',
                r'$\kappa = 1$', r'$\kappa = 0.1$']
colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']

xi = np.logspace(-2, 3, 10000)

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.flatten()

for idx, (k, label, col) in enumerate(zip(kappas, kappa_labels, colors)):
    ax = axes[idx]

    p_tcs = tcs_exact(xi, k)
    p_5pl = five_pl(xi, k)
    p_4pl = four_pl(xi, k)

    ax.semilogx(xi, p_tcs, color=col, lw=3, linestyle='-', alpha=0.9,
                label='TCS exact')
    ax.semilogx(xi, p_5pl, color=col, lw=2.5, linestyle='--', alpha=0.8,
                label='5PL (BG=1)')
    ax.semilogx(xi, p_4pl, color=col, lw=2, linestyle=':', alpha=0.7,
                label='4PL (B=1)')

    # Half-saturation reference
    ax.axhline(0.5, color='gray', lw=0.5, linestyle=':', alpha=0.5)

    # Mark xi50
    if k != np.inf:
        xi50 = 1 + 1 / (2 * k)
        ax.axvline(xi50, color='gray', lw=0.5, linestyle=':', alpha=0.3)

    ax.set_xlim(xi.min(), xi.max())
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r'$\xi$', fontsize=14)
    ax.set_ylabel(r'$p$', fontsize=14)

    # Compute B for title
    if k == np.inf:
        B_val = 1.0
        regime = '4PL = 5PL = TCS'
    else:
        B_val = np.log2((4 * k + 1) / (2 * k))
        if k >= 10:
            regime = '4PL adequate'
        elif k >= 1:
            regime = '5PL required'
        else:
            regime = 'Strong depletion'

    ax.set_title(f'{label}  (B={B_val:.3f}, {regime})',
                 fontsize=14, fontweight='bold')

    # Every subplot gets the legend
    ax.legend(fontsize=14, loc='lower right')

plt.suptitle('Extended Data Fig. 5a: TCS Unified Framework — '
             '4PL vs 5PL vs Exact TCS',
             fontsize=16, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('Extended_Data_Fig_5a.svg', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# Verification: print concentration errors
# ============================================
print()
print('=' * 68)
print('Verification: B values and concentration errors at plot κ values')
print('=' * 68)
print(f'{"κ":<10} {"B":<10} {"4PL ΔM%":<12} {"5PL ΔM%":<12} {"Regime"}')
print('-' * 60)

for k in kappas:
    if k == np.inf:
        print(f'{"∞":<10} {"1.000":<10} {"0":<12} {"0":<12} {"All identical"}')
        continue

    B = np.log2((4 * k + 1) / (2 * k))
    xi50 = 1 + 1 / (2 * k)

    max_4pl = 0
    max_5pl = 0
    for x in np.logspace(-2, 3, 10000) * xi50:
        p_exact = tcs_exact(x, k)
        if p_exact is None or p_exact < 0.1 or p_exact > 0.9:
            continue

        if p_exact > 0.001 and p_exact < 0.999:
            xi_4 = p_exact * xi50 / (1 - p_exact)
            try:
                xi_5 = xi50 * ((p_exact**(-B) - 1) / (2**B - 1))**(-1 / B)
                e4 = abs(x - xi_4) / x * 100
                e5 = abs(x - xi_5) / x * 100
                if e4 > max_4pl:
                    max_4pl = e4
                if e5 > max_5pl:
                    max_5pl = e5
            except:
                pass

    if k >= 10:
        regime = '4PL adequate'
    elif k >= 1:
        regime = '5PL required'
    else:
        regime = 'Strong depletion'

    print(f'{k:<10} {B:<10.4f} {max_4pl:<12.2f} {max_5pl:<12.2f} {regime}')
