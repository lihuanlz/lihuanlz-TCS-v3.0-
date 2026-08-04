# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 17:00:39 2026
Updated Jul 20 2026: aligned with S10.11 rewrite — cooperative Scatchard non-linearity.

@author: lihua

Fig. S10.1. Scatchard plots under stoichiometric multivalence and cooperativity.

Panel A: Receptor multivalence and cooperativity (n=1,2,4, f=1)
  Independent sites, arm-level (B_arms/F_arms): linear for all n.
    B/F = 1/κ - B/K_d  (slope=-1/K_d, intercept=1/κ)
    κ = K_d/(n*R_T) changes with n, but slope never depends on n.
  Positive cooperativity (n=2, c=10): arm-level CONCAVE (hump-shaped) non-linearity.
  Negative cooperativity (n=2, c=0.1): arm-level CONVEX non-linearity.
    Non-linearity encodes interaction energetics and direction.

Panel B: Ligand multivalence (f=1,2,4, n=1)
  Arm-level (B_arms/F_arms): linear for all f.
  Molecule-level (B_mol/F_mol): LINEAR for f=1, CONVEX for f>1.
  Convexity arises from combinatorial statistics of multivalent binding.

Master equation (S10.21): f*ξ = x + ν(x)/(n*κ)
  Independent-site case of (S10.21), recovering (S10.4): f*ξ = p/(1-p) + p/κ
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 14

# ============================================================
# Helper: solve independent-site equation
# f*xi = p/(1-p) + p/kappa  =>  p = solve_p(f*xi, kappa)
# ============================================================

def solve_p(xi, kappa, f=1):
    """Solve f*xi = p/(1-p) + p/kappa for p (receptor-site occupancy)."""
    xi_eff = f * xi
    return brentq(lambda p: p/(1-p) + p/kappa - xi_eff, 1e-15, 1-1e-15)

# ============================================================
# Helper: solve cooperative two-site equation
# n=2, K1 and K2 (Adair): ν = (K1*x + 2*K1*K2*x^2) / (1 + K1*x + K1*K2*x^2)
# Mass conservation: L_T = [free] + R_T * ν
# ============================================================

def solve_coop(L_T, R_T, K1, K2):
    """Solve for [free] given L_T, R_T, K1, K2 (two-site cooperative)."""
    def eq(logL):
        L_free = 10**logL
        x = L_free / K1
        c = K1 / K2
        nu = (x + 2*c*x**2) / (1 + x + c*x**2)  # ν = mean sites per receptor
        return L_T - L_free - R_T * nu
    try:
        logL = brentq(eq, -5, 5)
    except:
        return None
    L_free = 10**logL
    x = L_free / K1
    c = K1 / K2
    nu = (x + 2*c*x**2) / (1 + x + c*x**2)
    return L_free, nu

# ============================================================
# Panel A: Receptor multivalence + cooperativity (f=1)
# ============================================================

R_T = 1.0
K_d = 1.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Independent sites: n=1,2,4 (arm-level = molecule-level for f=1) ---
n_values = [1, 2, 4]
n_colors = {1: '#2166ac', 2: '#f46d43', 4: '#1a9850'}

for n in n_values:
    kappa = K_d / (n * R_T)
    B_list, BF_list = [], []
    for xi in np.logspace(-3, 2, 500):
        p = solve_p(xi, kappa, f=1)
        B = n * R_T * p
        F = K_d * xi - B
        if F > 1e-10 and B > 1e-10 and B < n * R_T * 0.999:
            B_list.append(B)
            BF_list.append(B / F)
    ax1.plot(B_list, BF_list, color=n_colors[n], linewidth=2,
             label=f'$n={n}$, $\\kappa={kappa:.2f}$ (indep.)')

# --- Cooperative sites: n=2, positive cooperativity (K2 < K1) ---
# K1 = K_d = 1.0, K2 = K_d/10 = 0.1 (positive coop, c = K1/K2 = 10)
K1_coop = K_d
K2_coop = K_d / 10.0
c_coop = K1_coop / K2_coop  # = 10

B_coop_list, BF_coop_list = [], []
for L_T in np.logspace(-2, 3, 500):
    result = solve_coop(L_T, R_T, K1_coop, K2_coop)
    if result is None:
        continue
    L_free, nu = result
    B = R_T * nu        # bound arms (f=1: arm = molecule)
    F = L_free           # free arms (f=1: arm = molecule)
    if F > 1e-10 and B > 1e-10 and B < R_T * 2 * 0.999:
        B_coop_list.append(B)
        BF_coop_list.append(B / F)

ax1.plot(B_coop_list, BF_coop_list, color='#d62728', linewidth=2.5,
         linestyle='--', label=f'$n=2$, pos. coop. ($c={c_coop:.0f}$)')

# --- Cooperative sites: n=2, negative cooperativity (K2 > K1) ---
# K1 = K_d = 1.0, K2 = 10*K_d = 10.0 (negative coop, c = K1/K2 = 0.1)
K1_neg = K_d
K2_neg = K_d * 10.0
c_neg = K1_neg / K2_neg  # = 0.1

B_neg_list, BF_neg_list = [], []
for L_T in np.logspace(-2, 3, 500):
    result = solve_coop(L_T, R_T, K1_neg, K2_neg)
    if result is None:
        continue
    L_free, nu = result
    B = R_T * nu
    F = L_free
    if F > 1e-10 and B > 1e-10 and B < R_T * 2 * 0.999:
        B_neg_list.append(B)
        BF_neg_list.append(B / F)

ax1.plot(B_neg_list, BF_neg_list, color='#7b3294', linewidth=2.5,
         linestyle=':', label=f'$n=2$, neg. coop. ($c={c_neg:.1f}$)')

ax1.set_xlabel('$B$', fontsize=16)
ax1.set_ylabel('$B/F$', fontsize=16)
ax1.legend(fontsize=10, loc='upper right')
ax1.set_xlim(0, None)
ax1.set_ylim(bottom=0)
ax1.set_title('(A) Receptor multivalence & cooperativity ($f=1$)', fontsize=16)
ax1.annotate('Solid: independent sites (linear)\n'
             
             
             'Dashed: positive coop. (concave, hump-shaped)\n' 'Dotted: negative coop. (convex)',
             xy=(0.62, 0.5), xycoords='axes fraction', fontsize=16, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# ============================================================
# Panel B: Ligand multivalence (n=1)
# f*xi = p/(1-p) + p/kappa
# Arm-level: B_arms = R_T*p, F_arms = f*M - R_T*p
#   B_arms/F_arms = (R_T - B_arms)/K_d (linear)
# Molecule-level: B_mol = M*(1-(1-q)^f), F_mol = M*(1-q)^f
# ============================================================

f_values = [1, 2, 4]
f_colors = {1: '#2166ac', 2: '#f46d43', 4: '#1a9850'}

for f in f_values:
    kappa = K_d / R_T  # Fixed kappa for all f (n=1)
    B_arms_list, BF_arms_list = [], []
    B_mol_list, BF_mol_list = [], []

    for M in np.logspace(-3, 2, 500):
        xi = M / K_d
        p = solve_p(xi, kappa, f=f)

        # Arm-level quantities
        B_arms = R_T * p
        F_arms = f * M - B_arms
        if F_arms > 1e-10 and B_arms > 1e-10 and B_arms < R_T * 0.999:
            B_arms_list.append(B_arms)
            BF_arms_list.append(B_arms / F_arms)

        # Molecule-level quantities
        q = p * R_T / (f * M) if f * M > 0 else 0
        if q < 0 or q >= 1:
            continue
        F_mol = M * (1 - q)**f
        B_mol = M - F_mol
        if F_mol > 1e-10 and B_mol > 1e-10:
            B_mol_list.append(B_mol)
            BF_mol_list.append(B_mol / F_mol)

    # Plot arm-level (solid line)
    ax2.plot(B_arms_list, BF_arms_list, color=f_colors[f], linewidth=2,
             linestyle='-',
             label=f'$f={f}$ (arm)')

    # Plot molecule-level (dashed line) — only if different from arm-level
    if f > 1:
        ax2.plot(B_mol_list, BF_mol_list, color=f_colors[f], linewidth=2,
                 linestyle='--',
                 label=f'$f={f}$ (mol)')

ax2.set_xlabel('$B$', fontsize=16)
ax2.set_ylabel('$B/F$', fontsize=16)
ax2.legend(fontsize=10, loc='upper right')
ax2.set_xlim(0, None)
ax2.set_ylim(bottom=0)
ax2.set_title('(B) Ligand multivalence ($n=1$)', fontsize=16)
ax2.annotate('Solid: $B_{arms}/F_{arms}$ (linear)\n'
             'Dashed: $B_{mol}/F_{mol}$ (convex for $f>1$)',
             xy=(0.62, 0.5), xycoords='axes fraction', fontsize=16, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

fig.suptitle('Fig. S10.1. Scatchard plots: multivalence and cooperativity',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('SI_S10_Fig_S10_1_Scatchard.svg', dpi=300, bbox_inches='tight')
plt.show()
print("SI_S10_Fig_S10_1_Scatchard.svg")
