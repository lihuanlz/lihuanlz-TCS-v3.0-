#!/usr/bin/env python3
"""
TCS Pharmacology Simulation Toolkit (Final with Times New Roman and LaTeX)
===========================================================================
Complete unified framework with:
- Unit-site, Multi-site, Competitive binding (Cheng-Prusoff)
- Operational model (Black & Leff)
- MWC, KNF allosteric models
- eTCM (extended Ternary Complex Model) with continuation
- Scale degeneracy proof and breaking
- All figures with Times New Roman font and LaTeX-style labels
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
import math
from scipy.optimize import curve_fit, fsolve, brentq
# =============================================================================
# Global plot settings: Times New Roman + LaTeX rendering
# =============================================================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # STIX fonts resemble Times New Roman
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# =============================================================================
# 1. Core TCS Solvers
# =============================================================================

def tcs_unit_site(xi, kappa):
    """Unit-site TCS exact solution via quadratic equation."""
    if kappa <= 0 or xi < 0:
        return np.nan
    a = 1.0
    b = -(kappa + 1.0 + kappa * xi)
    c = kappa * xi
    disc = b**2 - 4*a*c
    if disc < 0:
        return 0.0 if xi == 0 else np.nan
    sqrt_disc = np.sqrt(disc)
    p1 = (-b + sqrt_disc) / (2*a)
    p2 = (-b - sqrt_disc) / (2*a)
    for p in (p1, p2):
        if 0.0 <= p <= 1.0:
            return p
    return np.nan


# def tcs_multi_site(xi, kappa, n):
#     """n identical independent sites TCS. Bisection solver."""
#     if kappa <= 0 or n <= 0 or xi < 0:
#         return 0.0 if xi == 0 else np.nan

#     def f(p):
#         if p <= 0 or p >= 1:
#             return np.inf
#         return (p/(1-p))**(1/n) + p/kappa - xi

#     lo, hi = 0.0, 1.0
#     f_lo = f(1e-12)
#     f_hi = f(1 - 1e-12)
#     if f_lo * f_hi > 0:
#         return 0.0 if xi <= 0 else 1.0

#     for _ in range(60):
#         mid = (lo + hi) / 2
#         f_mid = f(mid)
#         if abs(f_mid) < 1e-15:
#             return mid
#         if f_lo * f_mid < 0:
#             hi = mid
#             f_hi = f_mid
#         else:
#             lo = mid
#             f_lo = f_mid
#     return (lo + hi) / 2

def tcs_multi_site(xi, kappa, n):
    """
    n equivalent independent sites. p = per-site fractional occupancy.

    Master equation: xi = p/(1-p) + p/kappa
    where kappa = K_d/(n*R_T) (n is absorbed into kappa).

    At fixed kappa, the curve p(xi) is identical for all n.
    Reduces to tcs_unit_site when n=1.
    """
    return tcs_unit_site(xi, kappa)


def tcs_multivalent(xi, kappa, f):
    """
    Multivalent ligand (f arms) binding to monovalent receptor (n=1).

    Corrected master equation: f*xi = p/(1-p) + p/kappa
    (S10.22 in final text)

    f enters only as a scalar multiplier on xi:
    p(xi; f) = p(f*xi; f=1) — same curve shape, shifted by 1/f.

    EC50 scales as 1/f; Hill coefficient is independent of f.
    """
    return tcs_unit_site(f * xi, kappa)


# def tcs_competitive(xi_L, xi_I, kappa_L, kappa_I, max_iter=200, tol=1e-12):
#     """Two-ligand competition TCS. Returns (p_L, p_I)."""
#     pL = xi_L / (1 + xi_L + xi_I) if (1+xi_L+xi_I) > 0 else 0.0
#     pI = xi_I / (1 + xi_L + xi_I) if (1+xi_L+xi_I) > 0 else 0.0
#     for _ in range(max_iter):
#         xL = xi_L - pL/kappa_L
#         xI = xi_I - pI/kappa_I
#         if xL < 0: xL = 0.0
#         if xI < 0: xI = 0.0
#         denom = 1 + xL + xI
#         pL_new = xL / denom
#         pI_new = xI / denom
#         if abs(pL_new - pL) < tol and abs(pI_new - pI) < tol:
#             return pL_new, pI_new
#         pL, pI = pL_new, pI_new
#     return pL, pI
def tcs_competitive(xi_L, xi_I, kappa_L, kappa_I):
    """Two-ligand competition TCS. Returns (p_L, p_I)."""
    def xL(q): return xi_L * kappa_L / (kappa_L + q)
    def xI(q): return xi_I * kappa_I / (kappa_I + q)
    def f(q): return q * (1.0 + xL(q) + xI(q)) - 1.0
    qs = np.concatenate(([1e-15], np.logspace(-6, 0, 400)))
    qprev, fprev = qs[0], f(qs[0])
    for q2 in qs[1:]:
        f2 = f(q2)
        if fprev * f2 < 0:
            q = brentq(f, qprev, q2, xtol=1e-14)
            return q * xL(q), q * xI(q)
        qprev, fprev = q2, f2
    q = brentq(f, 1e-15, 1.0, xtol=1e-14)
    return q * xL(q), q * xI(q)

def tcs_operational(xi, kappa_binding, tau, kappa_eff=None):
    """
    Two-layer TCS operational model. Returns signal E/Emax = q.

    Binding layer: p_R solved from xi = p_R/(1-p_R) + p_R/kappa_binding
    Effector layer: tau*p_R = q/(1-q) + q/kappa_eff

    By default kappa_eff = 1/tau (assumes E_T = R_T).
    For independent E_T, pass kappa_eff = K_E/E_T explicitly.

    This solves the exact TCS effector equation (S10.12 in final text).
    The Black-Leff limit is recovered as kappa_eff -> infinity:
        q = tau*p_R / (1 + tau*p_R)
    """
    pR = tcs_unit_site(xi, kappa_binding)
    if kappa_eff is None:
        kappa_eff = 1.0 / tau  # default: E_T = R_T
    xi_eff = tau * pR
    return tcs_unit_site(xi_eff, kappa_eff)


def tcs_operational_blackleff(xi, kappa_binding, tau):
    """
    Black-Leff operational model (zero-depletion limit of effector layer).
    E/Emax = tau*L_T / (K_d + L_T*(1+tau))

    This is the kappa_eff -> infinity limit of tcs_operational.
    """
    pR = tcs_unit_site(xi, kappa_binding)
    # Black-Leff: q = tau*p_R / (1 + tau*p_R)
    return tau * pR / (1.0 + tau * pR)


def mwc_occupancy(x, L, c, n):
    """Classic MWC occupancy function."""
    term_T = L * (1 + c*x)**n
    term_R = (1 + x)**n
    num = L * c * x * (1 + c*x)**(n-1) + x * (1 + x)**(n-1)
    denom = term_T + term_R
    return num / denom


def tcs_mwc(xi, kappa, L, c, n):
    """TCS-MWC: xi = x + p_MWC(x)/kappa."""
    def f(x):
        if x < 0:
            return -np.inf
        p = mwc_occupancy(x, L, c, n)
        return xi - x - p/kappa

    lo, hi = 0.0, xi
    f_lo = f(lo)
    f_hi = f(hi)
    if f_lo * f_hi > 0:
        return mwc_occupancy(xi, L, c, n)

    for _ in range(60):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if abs(f_mid) < 1e-12:
            break
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    x_sol = (lo + hi) / 2
    return mwc_occupancy(x_sol, L, c, n)


def knf_occupancy(x, n, alphas=None):
    """KNF model occupancy (per subunit)."""
    if alphas is None:
        alphas = np.ones(n)
    term = np.zeros(n+1)
    for i in range(n+1):
        comb = math.comb(n, i)
        prod_alpha = np.prod(alphas[:i]) if i > 0 else 1.0
        term[i] = comb * prod_alpha * (x ** i)
    Xi = np.sum(term)
    numerator = np.sum([i * term[i] for i in range(1, n+1)])
    return numerator / (n * Xi)


def tcs_knf(xi, kappa, n, alphas=None):
    """TCS-KNF model."""
    if alphas is None:
        alphas = np.ones(n)

    def f(x):
        if x < 0:
            return -np.inf
        p = knf_occupancy(x, n, alphas)
        return xi - x - p / kappa

    lo, hi = 0.0, xi
    f_lo = f(lo)
    f_hi = f(hi)
    if f_lo * f_hi > 0:
        return knf_occupancy(xi, n, alphas)
    for _ in range(60):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if abs(f_mid) < 1e-12:
            break
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    x_sol = (lo + hi) / 2
    return knf_occupancy(x_sol, n, alphas)


# =============================================================================
# 2. eTCM (extended Ternary Complex Model) with continuation
# =============================================================================

def etcm_signal(xi_L, xi_G, kappa_L, kappa_G, J, alpha, beta, x_init=None, y_init=None):
    """
    TCS-eTCM for GPCR signaling.
    Returns (signal, x_sol, y_sol) for continuation.
    """
    def equations(vars):
        x, y = vars
        if x < 0 or y < 0:
            return [1e10, 1e10]
        Xi = 1 + J + x + J*x/alpha + J*y + J*x*y/(alpha*beta)
        p_RL = x / Xi
        p_RsL = J*x/(alpha*Xi)
        p_RsG = J*y/Xi
        p_RsLG = J*x*y/(alpha*beta*Xi)
        eq1 = xi_L - x - (p_RL + p_RsL + p_RsLG) / kappa_L
        eq2 = xi_G - y - (p_RsG + p_RsLG) / kappa_G
        return [eq1, eq2]

    if x_init is None or y_init is None:
        x0 = max(0.0, xi_L * 0.9)
        y0 = max(0.0, xi_G * 0.9)
    else:
        x0 = max(0.0, x_init)
        y0 = max(0.0, y_init)

    sol = fsolve(equations, [x0, y0], maxfev=2000, xtol=1e-12)
    x_sol, y_sol = sol
    if x_sol < 0: x_sol = 0.0
    if y_sol < 0: y_sol = 0.0

    Xi = 1 + J + x_sol + J*x_sol/alpha + J*y_sol + J*x_sol*y_sol/(alpha*beta)
    S = J * (1 + x_sol/alpha + y_sol + x_sol*y_sol/(alpha*beta)) / Xi
    return S, x_sol, y_sol


# =============================================================================
# 3. Analysis Helpers
# =============================================================================

def hill_equation(x, n_H, EC50):
    return x**n_H / (EC50**n_H + x**n_H)


# def fit_apparent_hill(xi, p_vals):
#     mask = (p_vals > 0.01) & (p_vals < 0.99)
#     if np.sum(mask) < 5:
#         return np.nan, np.nan
#     try:
#         popt, _ = curve_fit(hill_equation, xi[mask], p_vals[mask],
#                             p0=[1.0, np.median(xi[mask])], maxfev=5000)
#         return popt[0], popt[1]
#     except Exception:
#         return np.nan, np.nan

def fit_apparent_hill(xi, p_vals):
    mask = (p_vals > 0.01) & (p_vals < 0.99)
    if np.sum(mask) < 5:
        return np.nan, np.nan, np.nan
    try:
        popt, _ = curve_fit(hill_equation, xi[mask], p_vals[mask],
                            p0=[1.0, np.median(xi[mask])], maxfev=5000)
        n_H, ec50 = popt
        p_pred = hill_equation(xi[mask], n_H, ec50)
        ss_res = np.sum((p_vals[mask] - p_pred) ** 2)
        ss_tot = np.sum((p_vals[mask] - np.mean(p_vals[mask])) ** 2)
        r2 = 1 - ss_res / ss_tot
        return n_H, ec50, r2
    except Exception:
        return np.nan, np.nan, np.nan



# def find_ic50(xi_L, kappa_L, kappa_I):
#     pL0 = tcs_unit_site(xi_L, kappa_L)
#     target = pL0 / 2.0
#     lo, hi = 0.0, 1000.0
#     while True:
#         pL, _ = tcs_competitive(xi_L, hi, kappa_L, kappa_I)
#         if pL <= target:
#             break
#         hi *= 2
#         if hi > 1e6:
#             return np.inf
#     for _ in range(60):
#         mid = (lo + hi) / 2
#         pL, _ = tcs_competitive(xi_L, mid, kappa_L, kappa_I)
#         if pL > target:
#             lo = mid
#         else:
#             hi = mid
#     return (lo + hi) / 2
def find_ic50(xi_L, kappa_L, kappa_I):
    """Exact IC50 for two-ligand competitive TCS (no iteration)."""
    pL_star = tcs_unit_site(xi_L, kappa_L) / 2.0
    xL = xi_L - pL_star / kappa_L
    if xL <= 0:
        return np.inf
    p_free = pL_star / xL
    p_I = 1.0 - pL_star - p_free
    if p_I <= 0:
        return 0.0
    return p_I * (1.0 / p_free + 1.0 / kappa_I)

# =============================================================================
# 4. Data Printing Functions (all using plain text, no Unicode subscript issues)
# =============================================================================

# def print_unit_site_data():
#     xi_vals = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
#     kappas = [100, 10, 1, 0.1, 0.01]
#     data = {'xi': xi_vals}
#     for k in kappas:
#         data[f'kappa={k}'] = [tcs_unit_site(xi, k) for xi in xi_vals]
#     df = pd.DataFrame(data)
#     print("\n=== Unit-site occupancy p ===")
#     print(df.to_string(index=False))


# def print_hill_fit_data():
#     kappa_list = np.logspace(-2, 2, 30)
#     results = []
#     for k in kappa_list:
#         xi_fine = np.logspace(-2, 2, 100)
#         p = np.array([tcs_unit_site(xi, k) for xi in xi_fine])
#         nH, ec50 = fit_apparent_hill(xi_fine, p)
#         results.append([k, nH, ec50])
#     df = pd.DataFrame(results, columns=['kappa', 'n_H_app', 'EC50'])
#     print("\n=== Apparent Hill coefficient ===")
#     print(df.to_string(index=False))


# def print_ic50_data():
#     xi_L = 1.0
#     kappas = [100, 10, 1, 0.5, 0.2, 0.1, 0.05, 0.01]
#     results = []
#     for k in kappas:
#         ic50 = find_ic50(xi_L, k, k)
#         cp_pred = 1 + xi_L
#         results.append([k, ic50, cp_pred, ic50/cp_pred])
#     df = pd.DataFrame(results, columns=['kappa', 'IC50', 'Cheng-Prusoff', 'Ratio'])
#     print("\n=== Competitive binding IC50 ===")
#     print(df.to_string(index=False))


# def print_operational_data():
#     xi_vals = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
#     kappa_binding = 10.0
#     taus = [0.1, 1, 10, 100]
#     data = {'xi': xi_vals}
#     for tau in taus:
#         data[f'tau={tau}'] = [tcs_operational(xi, kappa_binding, tau) for xi in xi_vals]
#     df = pd.DataFrame(data)
#     print("\n=== Operational model signal ===")
#     print(df.to_string(index=False))


# def print_mwc_data():
#     xi_vals = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
#     L, c, n = 1000, 0.01, 4
#     kappas = [100, 10, 1, 0.5, 0.2, 0.1, 0.05]
#     data = {'xi': xi_vals}
#     for k in kappas:
#         data[f'kappa={k}'] = [tcs_mwc(xi, k, L, c, n) for xi in xi_vals]
#     df = pd.DataFrame(data)
#     print(f"\n=== TCS-MWC data (L={L}, c={c}, n={n}) ===")
#     print(df.to_string(index=False))


# def print_multisite_data():
#     xi_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
#     kappa_fixed = 1.0
#     n_values = [1, 2, 4]
#     data = {'xi': xi_vals}
#     for n in n_values:
#         data[f'n={n}'] = [tcs_multi_site(xi, kappa_fixed, n) for xi in xi_vals]
#     df = pd.DataFrame(data)
#     print(f"\n=== Multi-site TCS (kappa={kappa_fixed}) ===")
#     print(df.to_string(index=False))


# def print_knf_data():
#     xi_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
#     n = 4
#     kappa = 10.0
#     alphas = [0.5, 1.0, 2.0, 5.0]
#     data = {'xi': xi_vals}
#     for a in alphas:
#         al = [a] * n
#         data[f'alpha={a}'] = [tcs_knf(xi, kappa, n, al) for xi in xi_vals]
#     df = pd.DataFrame(data)
#     print(f"\n=== KNF data (n={n}, kappa={kappa}) ===")
#     print(df.to_string(index=False))


# def print_etcm_data():
#     xiL_vals = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
#     xi_G = 1.0
#     J, kL, kG = 0.1, 10.0, 10.0
#     data1 = {'xi_L': xiL_vals}
#     for alpha, lab in [(0.1, 'Agonist'), (1.0, 'Neutral'), (10.0, 'Inv.Ag.')]:
#         S_list = []
#         for x in xiL_vals:
#             S, _, _ = etcm_signal(x, xi_G, kL, kG, J, alpha, 1.0)
#             S_list.append(S)
#         data1[f'{lab}'] = S_list
#     df1 = pd.DataFrame(data1)
#     print("\n=== eTCM agonist types (J=0.1, kappa_L=10, kappa_G=10) ===")
#     print(df1.to_string(index=False))

#     # G-protein depletion
#     xiL_vals = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
#     alpha_ag = 0.1
#     kg_list = [100, 1, 0.2, 0.05]
#     data2 = {'xi_L': xiL_vals}
#     for kg in kg_list:
#         S_vals = []
#         x_prev, y_prev = None, None
#         for x in xiL_vals:
#             S, x_sol, y_sol = etcm_signal(x, xi_G, kL, kg, J, alpha_ag, 1.0,
#                                           x_init=x_prev, y_init=y_prev)
#             S_vals.append(S)
#             x_prev, y_prev = x_sol, y_sol
#         data2[f'kappa_G={kg}'] = S_vals
#     df2 = pd.DataFrame(data2)
#     print("\n=== eTCM G-protein depletion (agonist alpha=0.1) ===")
#     print(df2.to_string(index=False))
def print_unit_site_data():
    xi_vals = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    kappas = [100, 10, 1, 0.1, 0.01]
    data = {'xi': xi_vals}
    for k in kappas:
        data[f'kappa={k}'] = [tcs_unit_site(xi, k) for xi in xi_vals]
    df = pd.DataFrame(data)
    print("\n=== Unit-site occupancy p ===")
    print(df.to_string(index=False))
    df.to_csv('unit_site_occupancy.csv', index=False)

# def print_hill_fit_data():
#     kappa_list = np.logspace(-3, 2, 100)
#     results = []
#     for k in kappa_list:
#         xi_fine = np.logspace(-3, 2, 100)
#         p = np.array([tcs_unit_site(xi, k) for xi in xi_fine])
#         nH, ec50 = fit_apparent_hill(xi_fine, p)
#         results.append([k, nH, ec50])
#     df = pd.DataFrame(results, columns=['kappa', 'n_H_app', 'EC50'])
#     print("\n=== Apparent Hill coefficient ===")
#     print(df.to_string(index=False))
#     df.to_csv('apparent_hill.csv', index=False)

def print_hill_fit_data():
    kappa_list = np.logspace(-3, 2, 30)
    results = []
    for k in kappa_list:
        xi_fine = np.logspace(-3, 2, 30)
        p = np.array([tcs_unit_site(xi, k) for xi in xi_fine])
        nH, ec50, r2 = fit_apparent_hill(xi_fine, p)
        results.append([k, nH, ec50, r2])
    df = pd.DataFrame(results, columns=['kappa', 'n_H_app', 'EC50', 'R2'])
    print("\n=== Apparent Hill coefficient ===")
    print(df.to_string(index=False))
    df.to_csv('apparent_hill.csv', index=False)


def print_ic50_data():
    xi_L = 1.0
    kappas = [100, 10, 1, 0.5, 0.2, 0.1, 0.05, 0.01]
    results = []
    for k in kappas:
        ic50 = find_ic50(xi_L, k, k)
        cp_pred = 1 + xi_L
        results.append([k, ic50, cp_pred, ic50/cp_pred])
    df = pd.DataFrame(results, columns=['kappa', 'IC50', 'Cheng-Prusoff', 'Ratio'])
    print("\n=== Competitive binding IC50 ===")
    print(df.to_string(index=False))
    df.to_csv('competitive_ic50.csv', index=False)

def print_operational_data():
    """
    Operational model: compare exact TCS vs Black-Leff.
    At L_T = K_d (xi=1), Black-Leff gives tau/(tau+2).
    At finite kappa_eff, exact TCS gives lower signal.
    """
    xi_vals = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    kappa_binding = 10.0  # Clark binding limit (kappa_b >> 1)
    taus = [0.1, 1, 10, 100]

    print("\n=== Operational model: exact TCS (kappa_eff=1/tau) ===")
    data = {'xi': xi_vals}
    for tau in taus:
        data[f'tau={tau}'] = [tcs_operational(xi, kappa_binding, tau) for xi in xi_vals]
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    df.to_csv('operational_model_exact.csv', index=False)

    print("\n=== Operational model: Black-Leff limit (kappa_eff->inf) ===")
    data_bl = {'xi': xi_vals}
    for tau in taus:
        data_bl[f'tau={tau}'] = [tcs_operational_blackleff(xi, kappa_binding, tau) for xi in xi_vals]
    df_bl = pd.DataFrame(data_bl)
    print(df_bl.to_string(index=False))
    df_bl.to_csv('operational_model_blackleff.csv', index=False)

    # Verification: at L_T=K_d (xi=1), Black-Leff = tau/(tau+2)
    print("\n=== Verification: E/Emax at L_T=K_d (xi=1) ===")
    print(f"{'tau':>6} {'exact(kappa_eff=1/tau)':>22} {'Black-Leff':>12} {'tau/(tau+2)':>12}")
    for tau in taus:
        q_exact = tcs_operational(1.0, kappa_binding, tau)
        q_bl = tcs_operational_blackleff(1.0, kappa_binding, tau)
        q_formula = tau / (tau + 2)
        print(f"{tau:>6.1f} {q_exact:>22.6f} {q_bl:>12.6f} {q_formula:>12.6f}")

def print_mwc_data():
    xi_vals = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    L, c, n = 1000, 0.01, 4
    kappas = [100, 10, 1, 0.5, 0.2, 0.1, 0.05]
    data = {'xi': xi_vals}
    for k in kappas:
        data[f'kappa={k}'] = [tcs_mwc(xi, k, L, c, n) for xi in xi_vals]
    df = pd.DataFrame(data)
    print(f"\n=== TCS-MWC data (L={L}, c={c}, n={n}) ===")
    print(df.to_string(index=False))
    df.to_csv('mwc_data.csv', index=False)

def print_multisite_data():
    """Multi-site: at fixed kappa, curve is identical for all n (n absorbed into kappa)."""
    xi_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    kappa_fixed = 1.0
    n_values = [1, 2, 4]
    data = {'xi': xi_vals}
    for n in n_values:
        data[f'n={n}'] = [tcs_multi_site(xi, kappa_fixed, n) for xi in xi_vals]
    df = pd.DataFrame(data)
    print(f"\n=== Multi-site TCS (kappa={kappa_fixed}) — curves identical for all n ===")
    print(df.to_string(index=False))
    df.to_csv('multisite.csv', index=False)


def print_multivalent_data():
    """Multivalent ligand: f shifts EC50 as 1/f, n_H unchanged."""
    xi_vals = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    kappa = 0.1
    f_values = [1, 2, 4]
    data = {'xi': xi_vals}
    for f in f_values:
        data[f'f={f}'] = [tcs_multivalent(xi, kappa, f) for xi in xi_vals]
    df = pd.DataFrame(data)
    print(f"\n=== Multivalent ligand TCS (kappa={kappa}) — EC50 scales as 1/f ===")
    print(df.to_string(index=False))
    df.to_csv('multivalent.csv', index=False)

    # Verify Hill coefficient independence of f
    print("\n=== Hill fit: n_H independent of f ===")
    print(f"{'f':>4} {'n_H_app':>10} {'EC50':>10} {'R2':>10}")
    xi_fine = np.logspace(-3, 3, 500)
    for f in [1, 2, 4]:
        p = np.array([tcs_multivalent(xi, 0.016, f) for xi in xi_fine])
        nH, ec50, r2 = fit_apparent_hill(xi_fine, p)
        print(f"{f:>4d} {nH:>10.4f} {ec50:>10.4f} {r2:>10.6f}")

def print_knf_data():
    xi_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    n = 4
    kappa = 10.0
    alphas = [0.5, 1.0, 2.0, 5.0]
    data = {'xi': xi_vals}
    for a in alphas:
        al = [a] * n
        data[f'alpha={a}'] = [tcs_knf(xi, kappa, n, al) for xi in xi_vals]
    df = pd.DataFrame(data)
    print(f"\n=== KNF data (n={n}, kappa={kappa}) ===")
    print(df.to_string(index=False))
    df.to_csv('knf_data.csv', index=False)

def print_etcm_data():
    xiL_vals = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
    xi_G = 1.0
    J, kL, kG = 0.1, 10.0, 10.0
    data1 = {'xi_L': xiL_vals}
    for alpha, lab in [(0.1, 'Agonist'), (1.0, 'Neutral'), (10.0, 'Inv.Ag.')]:
        S_list = []
        for x in xiL_vals:
            S, _, _ = etcm_signal(x, xi_G, kL, kG, J, alpha, 1.0)
            S_list.append(S)
        data1[f'{lab}'] = S_list
    df1 = pd.DataFrame(data1)
    print("\n=== eTCM agonist types (J=0.1, kappa_L=10, kappa_G=10) ===")
    print(df1.to_string(index=False))
    df1.to_csv('etcm_agonists.csv', index=False)

    # G-protein depletion
    xiL_vals = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
    alpha_ag = 0.1
    kg_list = [100, 1, 0.2, 0.05]
    data2 = {'xi_L': xiL_vals}
    for kg in kg_list:
        S_vals = []
        x_prev, y_prev = None, None
        for x in xiL_vals:
            S, x_sol, y_sol = etcm_signal(x, xi_G, kL, kg, J, alpha_ag, 1.0,
                                          x_init=x_prev, y_init=y_prev)
            S_vals.append(S)
            x_prev, y_prev = x_sol, y_sol
        data2[f'kappa_G={kg}'] = S_vals
    df2 = pd.DataFrame(data2)
    print("\n=== eTCM G-protein depletion (agonist alpha=0.1) ===")
    print(df2.to_string(index=False))
    df2.to_csv('etcm_g_depletion.csv', index=False)

# =============================================================================
# 5. Combined Figure with All Panels (A–L) + LaTeX labels
# =============================================================================

def figure_panels_A_F():
    """Fig. S10.2 (upper): panels A–F"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    fig.suptitle(
        'Fig. S10.2. TCS Unification of Pharmacological Receptor Theories (A–F)',
        fontsize=16, fontweight='bold', y=0.98)

    # A: Unit-site
    ax = axes[0, 0]
    xi = np.logspace(-2, 2, 200)
    for k in [100, 10, 1, 0.1, 0.01]:
        ax.plot(xi, [tcs_unit_site(v, k) for v in xi], label=f'$\kappa={k}$')
    ax.set_xscale('log')
    ax.set_xlabel('$\\xi = L_T / K_d$',fontsize=14)
    ax.set_ylabel('$p$',fontsize=14)
    ax.set_title('A. Unit-site TCS',fontsize=14)
    ax.legend(fontsize=14)
    # ax.grid(alpha=0.3)

    # B: Apparent Hill
    ax = axes[0, 1]
    klist = np.logspace(-3, 2, 100)
    nH = []
    for k in klist:
        xi_f = np.logspace(-3, 2, 100)
        p = np.array([tcs_unit_site(v, k) for v in xi_f])
        # n, _ = fit_apparent_hill(xi_f, p)
        n, _, _ = fit_apparent_hill(xi_f, p)
        nH.append(n)
    ax.plot(klist, nH, 'o-', markersize=4, color='steelblue')
    ax.set_xscale('log')
    ax.set_xlabel('$\\kappa$',fontsize=14)
    ax.set_ylabel('$n_H$',fontsize=14)
    ax.set_title('B. Apparent cooperativity ($n=1$)',fontsize=14)
    ax.axhline(1, color='gray', linestyle='--')
    # ax.grid(alpha=0.3)

    # C: Competitive binding
    ax = axes[1, 0]
    xi_L = 1.0
    ks = [100, 10, 1, 0.5, 0.2, 0.1, 0.05, 0.01]
    ic50s = [find_ic50(xi_L, k, k) for k in ks]
    ax.plot(ks, ic50s, 'o-', color='darkred', label='TCS $IC_{50}$')
    ax.axhline(2, color='gray', linestyle='--', label='Cheng-Prusoff')
    ax.set_xscale('log')
    ax.set_xlabel('$\\kappa$',fontsize=14)
    ax.set_ylabel('$IC_{50}$',fontsize=14)
    ax.set_title('C. Competitive binding',fontsize=14)
    ax.legend(fontsize=14)
    # ax.grid(alpha=0.3)

    # D: Operational model
    ax = axes[1, 1]
    xi = np.logspace(-2, 2, 200)
    for tau in [0.1, 1, 10, 100]:
        ax.plot(xi, [tcs_operational(v, 10.0, tau) for v in xi], label=f'$\\tau={tau}$')
    ax.set_xscale('log')
    ax.set_xlabel('$\\xi$',fontsize=14)
    ax.set_ylabel('$E/E_{\\rm max}$',fontsize=14)
    ax.set_title('D. Operational model',fontsize=14)
    ax.legend(fontsize=14)
    # ax.grid(alpha=0.3)

    # E: MWC
    ax = axes[2, 0]
    xi = np.logspace(-1, 2, 300)
    for k in [100, 1, 0.5, 0.2]:
        ax.plot(xi, [tcs_mwc(v, k, 1000, 0.01, 4) for v in xi], label=f'$\\kappa={k}$')
    ax.set_xscale('log')
    ax.set_xlabel('$\\xi = L_T / K_R$',fontsize=14)
    ax.set_ylabel('$p$',fontsize=14)
    ax.set_title('E. MWC allosteric ($L_0=1000, c=0.01, n=4$)',fontsize=14)
    ax.legend(fontsize=14)
    # ax.grid(alpha=0.3)

    # F: Multi-site
    ax = axes[2, 1]
    xi = np.logspace(-2, 2, 200)
    # for n in [1, 2, 4]:
    #     ax.plot(xi, [tcs_multi_site(v, 1.0, n) for v in xi], label=f'$n={n}$')
        
        # Figure F 面板中：
    for n in [1, 2, 4]:
        ax.plot(xi, [tcs_multi_site(v, 1.0, n) for v in xi], label=f'$n={n}$')
    # kappa=1.0 改为 kappa_R=1.0（数值不变，但语义正确）

        
        
        
    ax.set_xscale('log')
    ax.set_xlabel('$\\xi$',fontsize=14)
    ax.set_ylabel('$p$',fontsize=14)
    ax.set_title('F. Multi-site ($\\kappa=1$)',fontsize=14)
    ax.legend(fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('SI S10.2b.svg', dpi=300, bbox_inches='tight')
    plt.show()


def figure_panels_G_L():
    """Fig. S10.2 (lower): panels G–L"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    fig.suptitle(
        'Fig. S10.2 (continued). TCS Unification of Pharmacological Receptor Theories (G–L)',
        fontsize=16, fontweight='bold', y=0.98)

    # G: KNF alpha
    ax = axes[0, 0]
    xi = np.logspace(-2, 2, 200)
    for a in [0.1, 0.5, 1.0, 2.0, 10.0]:
        al = [a] * 4
        ax.plot(xi, [tcs_knf(v, 100.0, 4, al) for v in xi], label=f'$\\alpha={a}$')
    ax.set_xscale('log')
    ax.set_xlabel('$\\xi$',fontsize=14)
    ax.set_ylabel('$p$',fontsize=14)
    ax.set_title('G. KNF cooperativity ($n=4, \\kappa=100$)',fontsize=14)
    ax.legend(fontsize=14)
    # ax.grid(alpha=0.3)

    # H: KNF depletion
    ax = axes[0, 1]
    al = [5.0] * 4
    for k in [100, 1, 0.2, 0.05]:
        ax.plot(xi, [tcs_knf(v, k, 4, al) for v in xi], label=f'$\\kappa={k}$')
    ax.set_xscale('log')
    ax.set_xlabel('$\\xi$',fontsize=14)
    ax.set_ylabel('$p$',fontsize=14)
    ax.set_title('H. KNF depletion ($n=4, \\alpha=5$)',fontsize=14)
    ax.legend(fontsize=14)
    # ax.grid(alpha=0.3)

    # I: eTCM agonists
    ax = axes[1, 0]
    xiL = np.logspace(-3, 2, 200)
    J, kL, kG = 0.1, 10.0, 10.0
    xi_G = 1.0
    for alpha, lab in [(0.1, 'Agonist ($\\alpha=0.1$)'),
                       (1.0, 'Neutral ($\\alpha=1$)'),
                       (10.0, 'Inverse agonist ($\\alpha=10$)')]:
        S_list = []
        for x in xiL:
            S, _, _ = etcm_signal(x, xi_G, kL, kG, J, alpha, 1.0)
            S_list.append(S)
        ax.plot(xiL, S_list, label=lab)
    basal_S, _, _ = etcm_signal(0, xi_G, kL, kG, J, 1.0, 1.0)
    ax.axhline(basal_S, color='gray', ls='--', label='Basal')
    ax.set_xscale('log')
    ax.set_xlabel('$\\xi_L = L_T / K_L$')
    ax.set_ylabel('Signal $S$')
    ax.set_title('I. eTCM: agonist types ($J=0.1$)')
    ax.legend(fontsize=14)
    # ax.grid(alpha=0.3)

    # J: eTCM G-protein depletion
    ax = axes[1, 1]
    J, kL = 0.1, 10.0
    xi_G = 1.0
    alpha_ag = 0.1
    kg_list = [100, 1, 0.2, 0.05]
    for kg in kg_list:
        S_vals = []
        x_prev, y_prev = None, None
        for x in xiL:
            S, x_sol, y_sol = etcm_signal(x, xi_G, kL, kg, J, alpha_ag, 1.0,
                                          x_init=x_prev, y_init=y_prev)
            S_vals.append(S)
            x_prev, y_prev = x_sol, y_sol
        ax.plot(xiL, S_vals, label=f'$\\kappa_G={kg}$')
    basal_S, _, _ = etcm_signal(0, xi_G, kL, 100, J, 1.0, 1.0)
    ax.axhline(basal_S, color='gray', ls='--', label='Basal')
    ax.set_xscale('log')
    ax.set_xlabel('$\\xi_L$',fontsize=14)
    ax.set_ylabel('Signal $S$',fontsize=14)
    ax.set_title('J. eTCM: G-protein depletion ($\\alpha=0.1$)',fontsize=14)
    ax.legend(fontsize=14)
    # ax.grid(alpha=0.3)

    # # K: Proof of Scale Degeneracy
    # ax = axes[2, 0]
    # L_T_scan = np.logspace(-1, 2, 200)
    # param_sets = [
    #     (1.0, 10.0, 1.0, '$a=1$: $L_T=1, R_T=10, K_d=1$'),
    #     (2.0, 20.0, 2.0, '$a=2$: $L_T=2, R_T=20, K_d=2$'),
    #     (5.0, 50.0, 5.0, '$a=5$: $L_T=5, R_T=50, K_d=5$'),
    # ]
    # for L_T_base, R_T_base, K_d_base, label in param_sets:
    #     p_vals = []
    #     for L_T in L_T_scan:
    #         xi = L_T / K_d_base
    #         kappa = K_d_base / R_T_base
    #         p = tcs_unit_site(xi, kappa)
    #         p_vals.append(p)
    #     ax.plot(L_T_scan, p_vals, lw=2, label=label)
    # ax.set_xscale('log')
    # ax.set_xlabel('$L_T$',fontsize=14)
    # ax.set_ylabel('$p$',fontsize=14)
    # ax.set_title('K. Proof of Scale Degeneracy',fontsize=14)
    # ax.legend(fontsize=14)
    # # ax.grid(alpha=0.3)
    # ax.set_ylim(0, 1.02)
    # ax.annotate('$\\xi = L_T/K_d = 1.0$\n$\\kappa = K_d/R_T = 0.1$\n$\\rightarrow p$ identical',
    #             xy=(0.05, 0.95), xycoords='axes fraction',
    #             fontsize=9, ha='left', va='top',
    #             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    
    
        # K: Proof of Scale Degeneracy
    ax = axes[2, 0]
    xi_scan = np.logspace(-1, 2, 200)                # 直接扫描 ξ
    # 三个参数组具有相同的 κ = 0.1，因此 p(ξ) 完全一样
    # param_sets = [
    #     (1.0, 10.0, 1.0, r'$\xi = L_T/K_d,\ \kappa = 0.1$'),
    #     (2.0, 20.0, 2.0, r'$\xi = L_T/K_d,\ \kappa = 0.1$'),
    #     (5.0, 50.0, 5.0, r'$\xi = L_T/K_d,\ \kappa = 0.1$'),
    # ]
    
    
    
    
#     param_sets = [
#     (1.0, 10.0, 1.0, r'$\xi = 1,\ \kappa = 0.1$'),
#     (2.0, 20.0, 2.0, r'$\xi = 2,\ \kappa = 0.05$'),
#     (5.0, 50.0, 5.0, r'$\xi = 5,\ \kappa = 0.02$'),
# ]
    param_sets = [
        (1.0, 10.0, 1.0, r'$\lambda=1$: $(L_T, R_T, K_d)=(1, 10, 1)$'),
        (2.0, 20.0, 2.0, r'$\lambda=2$: $(L_T, R_T, K_d)=(2, 20, 2)$'),
        (5.0, 50.0, 5.0, r'$\lambda=5$: $(L_T, R_T, K_d)=(5, 50, 5)$'),
    ]

    for L_T_base, R_T_base, K_d_base, label in param_sets:
        kappa = K_d_base / R_T_base   # 0.1 for all
        p_vals = [tcs_unit_site(xi, kappa) for xi in xi_scan]
        ax.plot(xi_scan, p_vals, lw=2, label=label)
    
    ax.set_xscale('log')
    ax.set_xlabel(r'$\xi = L_T / K_d$', fontsize=14)
    ax.set_ylabel('$p$', fontsize=14)
    ax.set_title('K. Proof of Scale Degeneracy', fontsize=14)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.annotate(
        'All three curves coincide exactly:\n'
        '$p$ depends only on $\\xi$ and $\\kappa$,\n'
        'not on individual $L_T, R_T, K_d$.',
        xy=(0.05, 0.95), xycoords='axes fraction',
        fontsize=9, ha='left', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    
    
    
    
    
    
    
    
        # K: Proof of Scale Degeneracy
    ax = axes[2, 0]
    u_scan = np.logspace(-1, 2, 200)   # dimensionless titration range xi = L_T/K_d
    # three systems related by (L_T, R_T, K_d) -> lambda * (L_T, R_T, K_d)
    param_sets = [
        (1.0, 10.0, 1.0, r'$\lambda=1$: $(L_T, R_T, K_d)=(1, 10, 1)$'),
        (2.0, 20.0, 2.0, r'$\lambda=2$: $(L_T, R_T, K_d)=(2, 20, 2)$'),
        (5.0, 50.0, 5.0, r'$\lambda=5$: $(L_T, R_T, K_d)=(5, 50, 5)$'),
    ]
    for lam, R_T, K_d, label in param_sets:
        L_T = lam * u_scan           # this system's absolute ligand titration
        xi = L_T / K_d               # -> identical xi for all three systems
        kappa = K_d / R_T            # -> 0.1 for all three systems
        p_vals = [tcs_unit_site(x, kappa) for x in xi]
        ax.plot(xi, p_vals, lw=2, label=label)
    
    ax.set_xscale('log')
    ax.set_xlabel(r'$\xi = L_T / K_d$', fontsize=14)
    ax.set_ylabel('$p$', fontsize=14)
    ax.set_title('K. Proof of Scale Degeneracy', fontsize=14)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.annotate(
        'All three curves coincide exactly:\n'
        '$p$ depends only on $\\xi$ and $\\kappa$,\n'
        'not on individual $L_T, R_T, K_d$.',
        xy=(0.05, 0.95), xycoords='axes fraction',
        fontsize=9, ha='left', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    # L: Breaking Scale Degeneracy
    ax = axes[2, 1]
    L_T_scan = np.logspace(-1, 2, 200)    # ← 补上这一行
    K_d_true = 1.0
    R_T_low = 20.0
    R_T_high = 0.05
    for R_T, kappa_val, label, color in [
        (R_T_low, K_d_true/R_T_low,
         f'$\\kappa = {K_d_true/R_T_low:.2f}$ ($R_T = {R_T_low}$)', 'darkred'),
        (R_T_high, K_d_true/R_T_high,
         f'$\\kappa = {K_d_true/R_T_high:.0f}$ ($R_T = {R_T_high}$)', 'steelblue'),
    ]:
        p_vals = []
        for L_T in L_T_scan:
            xi = L_T / K_d_true
            p = tcs_unit_site(xi, kappa_val)
            p_vals.append(p)
        ax.plot(L_T_scan, p_vals, lw=2.5, label=label, color=color)

    ax.axvline(1.0, color='steelblue', ls='--', alpha=0.5, lw=1)
    ax.annotate('$EC_{50} \\approx K_d$', xy=(1.0, 0.5), xytext=(0.3, 0.6),
                arrowprops=dict(arrowstyle='->', color='steelblue'),
                color='steelblue', fontsize=9)
    ax.axvline(10.0, color='darkred', ls='--', alpha=0.5, lw=1)
    ax.annotate('$EC_{50} \\approx R_T/2$', xy=(10.0, 0.5), xytext=(30, 0.4),
                arrowprops=dict(arrowstyle='->', color='darkred'),
                color='darkred', fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel('$L_T$',fontsize=14)
    ax.set_ylabel('$p$',fontsize=14)
    ax.set_title('L. Breaking Scale Degeneracy',fontsize=14)
    ax.legend(fontsize=14)
    # ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.02)
    ax.annotate('From two curves:\n$K_d = 1.0$ (unique)\n$R_T$ determined',
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=9, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('SI S10.2a.svg', dpi=300, bbox_inches='tight')
    plt.show()


# =============================================================================
# 6. Main: run all
# =============================================================================

if __name__ == "__main__":
    print("=== TCS Pharmacology Simulation Suite (Final with LaTeX) ===")

    # Print data tables
    print_unit_site_data()
    print_hill_fit_data()
    print_ic50_data()
    print_operational_data()
    print_mwc_data()
    print_multisite_data()
    print_multivalent_data()
    print_knf_data()
    print_etcm_data()

    # Generate figures (split into two for readability)
    figure_panels_A_F()
    figure_panels_G_L()

    print("\nAll figures saved and data printed.")