# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 13:40:29 2026

@author: lihua
"""

"""
Unified comparison: 4PL vs 5PL vs exact TCS root.

Three pairwise concentration errors at κ = 0.1, 1, 10:
  1. 4PL vs TCS:   |xi - xi_4PL| / xi * 100
  2. 5PL vs TCS:   |xi - xi_5PL| / xi * 100
  3. 4PL vs 5PL:   |xi_5PL - xi_4PL| / xi * 100

Exact TCS:  xi = p/(1-p) + p/kappa
            => p^2 - (kappa*xi + kappa + 1)*p + kappa*xi = 0
            => p = [(kappa*xi + kappa + 1) - sqrt((kappa*xi + kappa + 1)^2 - 4*kappa*xi)] / 2

4PL (B=1):  p = xi / (xi + xi50)
            Back-calc: xi = p * xi50 / (1 - p)

5PL (SI S3.10):  p_5PL = [1 + (2^B - 1)*(xi/xi50)^(-B)]^(-1/B)
                 where B = log2((4*kappa+1)/(2*kappa)),  xi50 = 1 + 1/(2*kappa)
                 Back-calc: xi = xi50 * [(p^(-B) - 1) / (2^B - 1)]^(-1/B)

Practical range: p = 0.1 to 0.9 (LoB to near-saturation)
"""

import math


def exact_tcs_p(xi, kappa):
    """Exact TCS occupancy from dimensionless concentration xi."""
    a = 1.0
    b = -(kappa * xi + kappa + 1)
    c = kappa * xi
    disc = b**2 - 4 * a * c
    if disc < 0:
        return None
    p = (-b - math.sqrt(disc)) / (2 * a)
    if p < 0 or p > 1:
        p = (-b + math.sqrt(disc)) / (2 * a)
    if p < 0 or p > 1:
        return None
    return p


def four_pl_p(xi, xi50):
    """4PL occupancy (B=1): p = xi / (xi + xi50)."""
    p = xi / (xi + xi50)
    if p < 0 or p > 1:
        return None
    return p


def five_pl_p(xi, xi50, B):
    """5PL occupancy (SI S3.10)."""
    val = 1 + (2**B - 1) * (xi / xi50) ** (-B)
    if val <= 0:
        return None
    p = val ** (-1 / B)
    if p < 0 or p > 1:
        return None
    return p


def back_calc_4pl(p, xi50):
    """Back-calculate xi from p using 4PL: xi = p * xi50 / (1 - p)."""
    if p <= 0.001 or p >= 0.999:
        return None
    return p * xi50 / (1 - p)


def back_calc_5pl(p, xi50, B):
    """Back-calculate xi from p using 5PL."""
    if p <= 0.001 or p >= 0.999:
        return None
    try:
        return xi50 * ((p ** (-B) - 1) / (2**B - 1)) ** (-1 / B)
    except:
        return None


print("=" * 80)
print("4PL vs 5PL vs exact TCS: pairwise concentration errors")
print("Practical range: p = 0.1 to 0.9")
print("=" * 80)
print()
print(f'{"κ":<8} {"B":<10} {"4PL vs TCS":<14} {"5PL vs TCS":<14} {"4PL vs 5PL":<14}')
print("-" * 62)

for kappa in [0.1, 1, 10]:
    B = math.log2((4 * kappa + 1) / (2 * kappa))
    xi50 = 1 + 1 / (2 * kappa)

    max_4pl_tcs = 0
    max_5pl_tcs = 0
    max_4pl_5pl = 0

    for exp in range(-400, 400):
        xi = xi50 * 10 ** (exp / 100)

        p_exact = exact_tcs_p(xi, kappa)
        if p_exact is None:
            continue
        if p_exact < 0.1 or p_exact > 0.9:
            continue

        # 4PL vs TCS
        xi_4pl = back_calc_4pl(p_exact, xi50)
        if xi_4pl is not None and xi > 0:
            err = abs(xi - xi_4pl) / xi * 100
            if err > max_4pl_tcs:
                max_4pl_tcs = err

        # 5PL vs TCS
        xi_5pl = back_calc_5pl(p_exact, xi50, B)
        if xi_5pl is not None and xi > 0:
            err = abs(xi - xi_5pl) / xi * 100
            if err > max_5pl_tcs:
                max_5pl_tcs = err

        # 4PL vs 5PL (back-calc xi from 4PL p, then compare to 5PL back-calc)
        p_4pl = four_pl_p(xi, xi50)
        if p_4pl is not None:
            xi_from_4pl = back_calc_4pl(p_4pl, xi50)
            xi_from_5pl = back_calc_5pl(p_4pl, xi50, B)
            if xi_from_4pl is not None and xi_from_5pl is not None:
                err = abs(xi_from_4pl - xi_from_5pl) / xi_from_4pl * 100
                if err > max_4pl_5pl:
                    max_4pl_5pl = err

    print(f"{kappa:<8} {B:<10.4f} {max_4pl_tcs:<14.2f} {max_5pl_tcs:<14.2f} {max_4pl_5pl:<14.2f}")

print()
print("Columns:")
print("  4PL vs TCS  = concentration error of 4PL relative to exact TCS root")
print("  5PL vs TCS  = concentration error of 5PL relative to exact TCS root")
print("  4PL vs 5PL  = concentration difference between 4PL and 5PL back-calculation")

# =====================================================================
# Part 2: Full interval scan for Table 1 regime boundaries
# =====================================================================
print()
print("=" * 80)
print("Part 2: Full interval scan — max concentration error in each regime")
print("=" * 80)
print()

regimes = [
    ("κ ≥ 10 (4PL adequate)",      [10, 20, 50, 100, 500, 1000]),
    ("1 ≤ κ ≤ 10 (5PL required)",  [1, 2, 3, 5, 8, 10]),
    ("κ ≤ 1 (strong depletion)",   [0.01, 0.02, 0.05, 0.1, 0.144, 0.169, 0.2, 0.3, 0.5, 0.8, 1]),
]

for regime_name, kappas in regimes:
    min_4pl = 1e9
    max_4pl = 0
    min_5pl = 1e9
    max_5pl = 0

    for kappa in kappas:
        B = math.log2((4 * kappa + 1) / (2 * kappa))
        xi50 = 1 + 1 / (2 * kappa)

        max_4pl_tcs = 0
        max_5pl_tcs = 0

        for exp in range(-400, 400):
            xi = xi50 * 10 ** (exp / 100)
            p_exact = exact_tcs_p(xi, kappa)
            if p_exact is None:
                continue
            if p_exact < 0.1 or p_exact > 0.9:
                continue

            xi_4pl = back_calc_4pl(p_exact, xi50)
            if xi_4pl is not None and xi > 0:
                err = abs(xi - xi_4pl) / xi * 100
                if err > max_4pl_tcs:
                    max_4pl_tcs = err

            xi_5pl = back_calc_5pl(p_exact, xi50, B)
            if xi_5pl is not None and xi > 0:
                err = abs(xi - xi_5pl) / xi * 100
                if err > max_5pl_tcs:
                    max_5pl_tcs = err

        if max_4pl_tcs < min_4pl:
            min_4pl = max_4pl_tcs
        if max_4pl_tcs > max_4pl:
            max_4pl = max_4pl_tcs
        if max_5pl_tcs < min_5pl:
            min_5pl = max_5pl_tcs
        if max_5pl_tcs > max_5pl:
            max_5pl = max_5pl_tcs

    print(f"  {regime_name}")
    print(f"    4PL ΔM: {min_4pl:.2f}% — {max_4pl:.2f}%")
    print(f"    5PL ΔM: {min_5pl:.2f}% — {max_5pl:.2f}%")
    print()

print("These ranges define the ΔM values in Table 1 and SI S3 Remark.")
