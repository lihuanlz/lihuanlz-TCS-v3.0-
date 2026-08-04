# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 20:29:53 2026

@author: lihua
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 14:33:48 2026
v5 fixes (2026-07-29):
  1. Restored the Omega % N == 0 check (v4 had dropped it; section 10 used
     Omega = 6/56/112, leaving orphan sites uncovered by any partition and
     breaking partition exchangeability). Section 10 re-parametrised to
     (M, Omega) = (9,10), (45,50), (90,100): M/Omega = 0.9 exactly.
  2. Column "C/Omega" renamed to "W_L/Om": v4 computed (M - Omega*p)/Omega,
     the FREE-ligand fraction, not the occupancy C/Omega = p. H2 evidence
     reworded for both readings.
  3. SUMMARY rewritten: upper bounds now match the printed data (v4 claimed
     "N_eff=100: VR4L < 1.0001" while its own kappa=5 row gives 1.000197);
     "for ALL kappa tested" removed; new paragraph on the exponentially
     saturated corner (1 - P1 -> 0), where ICC_4L -> +1 and VR4L -> N
     (both variances vanish there).
  4. P_C_3L clamp (lp > 700) replaced by exact log-shift normalisation.
  5. solve_p rationalised against catastrophic cancellation:
     p = 2u / (u + kappa + 1 + sqrt(disc)).
  6. Rows hitting the 1e-8 guard are recomputed in 50-digit precision
     (mpmath, optional; only for M <= MP_MAX) and marked '*'. Larger
     guarded rows are annotated with the confirmed asymptotics
     ICC_4L -> +1, VR4L -> N.

@author: lihua
"""

#!/usr/bin/env python3
"""
Verification of Table S2b.2: ICC and VR4L across N_eff, kappa, and p

Tests three hypotheses:
  H1: VR4L is controlled by p (SI's stated condition "p<<1")
  H2: VR4L is controlled by C/Omega (occupancy) or free-ligand fraction
  H3: VR4L is controlled by N_eff = min(M, Omega), within the
      non-saturated working region (1 - P1 resolvable)

Key disambiguation tests:
  - Same p, different N_eff  -> if VR4L changes, H1 is wrong
  - Same N_eff, different p  -> if VR4L~1 for high p, H1 is wrong
"""

import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson
import sys

# Optional high-precision path for guard-triggered (saturated) rows
try:
    from mpmath import mp, mpf, loggamma as mp_loggamma, exp as mp_exp
    HAS_MP = True
except ImportError:
    HAS_MP = False

MP_MAX_M = 250   # high-precision recomputation only for M <= this (runtime)


def log_binom(n, k):
    """Log of binomial coefficient C(n, k)."""
    if k < 0 or k > n or n < 0:
        return -np.inf
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def solve_p(M, kappa, Omega):
    """Solve master equation xi = p/(1-p) + p/kappa for p.
    Rationalised form: avoids cancellation at small u = M/Omega."""
    u = M / Omega
    disc = (u + kappa + 1.0)**2 - 4.0 * u
    if disc < 0:
        return np.nan
    p = 2.0 * u / (u + kappa + 1.0 + np.sqrt(disc))
    return p if 0 < p < 1 else np.nan


def compute_icc_from_PC(P_C_func, Omega, N, beta, C_max):
    """Compute ICC and VR from P(C) distribution."""
    C_arr = np.arange(C_max + 1)
    P_C = np.array([P_C_func(int(C)) for C in C_arr])
    total = P_C.sum()
    if total <= 0 or not np.isfinite(total):
        return np.nan, np.nan, np.nan
    P_C /= total

    P_empty1 = 0.0
    P_empty2 = 0.0
    for i, C in enumerate(C_arr):
        if P_C[i] < 1e-20:
            continue
        if C <= Omega - beta:
            P_empty1 += P_C[i] * np.exp(log_binom(Omega - beta, C) - log_binom(Omega, C))
        if C <= Omega - 2 * beta:
            P_empty2 += P_C[i] * np.exp(log_binom(Omega - 2*beta, C) - log_binom(Omega, C))

    P1 = 1.0 - P_empty1
    if P1 <= 0 or P1 >= 1:
        return np.nan, np.nan, P1
    if (1.0 - P1) < 1e-8:
        return np.nan, np.nan, P1

    P11 = 1.0 - 2.0 * P_empty1 + P_empty2
    ICC = (P11 - P1**2) / (P1 * (1.0 - P1))
    return ICC, 1.0 + (N - 1) * ICC, P1


def compute_log_Q(W, Omega, KV):
    """Log partition function Q(W)."""
    max_C = min(W, Omega)
    if max_C < 0:
        return -np.inf
    Cs = np.arange(max_C + 1, dtype=float)
    log_w = np.array([
        log_binom(W, int(c)) + log_binom(Omega, int(c)) + gammaln(c + 1) - c * np.log(KV)
        for c in Cs
    ])
    return np.logaddexp.reduce(log_w)


def compute_row(M, Omega, N, kappa):
    """Compute ICC_3L, VR_3L, ICC_4L, VR_4L for given parameters."""
    beta = Omega // N
    if beta * N != Omega:
        raise ValueError(f"Omega={Omega} must be divisible by N={N}")
    if M > Omega:
        raise ValueError("M must not exceed Omega")
    KV = kappa * Omega

    # --- Three-layer (conditional on W=M): exact log-shift normalisation ---
    C_max_3L = min(M, Omega)
    Cs_3L = np.arange(C_max_3L + 1)
    log_w_3L = np.array([
        log_binom(M, int(c)) + log_binom(Omega, int(c))
        + gammaln(c + 1) - c * np.log(KV) for c in Cs_3L
    ])
    log_w_3L -= log_w_3L.max()
    w_3L = np.exp(log_w_3L)
    w_3L_map = dict(zip(Cs_3L.tolist(), w_3L.tolist()))

    def P_C_3L(C):
        return w_3L_map.get(C, 0.0)

    ICC_3L, VR_3L, P1_3L = compute_icc_from_PC(P_C_3L, Omega, N, beta, C_max_3L)

    # --- Four-layer (marginalize over W ~ Poisson(M)) ---
    W_max = min(int(M + 5 * np.sqrt(max(M, 1)) + 10), 2 * M + 50)
    C_max_4L = min(W_max, Omega)
    poisson_wts = poisson.pmf(np.arange(W_max + 1), M)
    log_Q = np.array([compute_log_Q(W, Omega, KV) for W in range(W_max + 1)])

    def P_C_4L(C):
        if C < 0 or C > C_max_4L:
            return 0.0
        total = 0.0
        for W in range(C, W_max + 1):
            if poisson_wts[W] < 1e-20:
                continue
            if C > W or C > Omega:
                continue
            lp = (log_binom(W, C) + log_binom(Omega, C) +
                  gammaln(C + 1) - C * np.log(KV))
            total += poisson_wts[W] * np.exp(lp - log_Q[W])
        return total

    ICC_4L, VR_4L, P1_4L = compute_icc_from_PC(P_C_4L, Omega, N, beta, C_max_4L)

    return {
        'ICC_3L': ICC_3L, 'VR_3L': VR_3L,
        'ICC_4L': ICC_4L, 'VR_4L': VR_4L,
        'P1_3L': P1_3L, 'P1_4L': P1_4L,
    }


def mp_compute_row(M, Omega, N, kappa, dps=50):
    """High-precision recomputation of a guard-triggered row (mpmath)."""
    mp.dps = dps
    beta = Omega // N
    KV = mpf(str(kappa)) * Omega

    def moments(W):
        Cmax = min(W, Omega)
        ws = []
        for C in range(Cmax + 1):
            lw = (mp_loggamma(W+1) - mp_loggamma(C+1) - mp_loggamma(W-C+1)
                  + mp_loggamma(Omega+1) - mp_loggamma(C+1) - mp_loggamma(Omega-C+1)
                  + mp_loggamma(C+1) - C * mp.log(KV))
            ws.append(mp_exp(lw))
        Q = sum(ws)
        Pe1 = Pe2 = mpf(0)
        for C, w_ in enumerate(ws):
            p_ = w_ / Q
            if C <= Omega - beta:
                Pe1 += p_ * mp_exp(mp_loggamma(Omega-beta+1) - mp_loggamma(C+1)
                                   - mp_loggamma(Omega-beta-C+1)
                                   - (mp_loggamma(Omega+1) - mp_loggamma(C+1)
                                      - mp_loggamma(Omega-C+1)))
            if C <= Omega - 2 * beta:
                Pe2 += p_ * mp_exp(mp_loggamma(Omega-2*beta+1) - mp_loggamma(C+1)
                                   - mp_loggamma(Omega-2*beta-C+1)
                                   - (mp_loggamma(Omega+1) - mp_loggamma(C+1)
                                      - mp_loggamma(Omega-C+1)))
        P1 = 1 - Pe1
        P11 = 1 - 2 * Pe1 + Pe2
        return P1, P11

    P1, P11 = moments(M)
    ICC3 = (P11 - P1**2) / (P1 * (1 - P1))
    VR3 = 1 + (N - 1) * ICC3

    P1m = P11m = mpf(0)
    for W in range(0, int(M + 8 * M**0.5) + 5):
        pw = mp_exp(-M + W * mp.log(M) - mp_loggamma(W + 1))
        a, b = moments(W)
        P1m += pw * a
        P11m += pw * b
    ICC4 = (P11m - P1m**2) / (P1m * (1 - P1m))
    VR4 = 1 + (N - 1) * ICC4

    return {
        'ICC_3L': float(ICC3), 'VR_3L': float(VR3),
        'ICC_4L': float(ICC4), 'VR_4L': float(VR4),
        'P1_3L': float(P1), 'P1_4L': float(P1m),
    }


def print_header(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)
    hdr = (f"{'M':>6} {'Omega':>6} {'N':>3} {'kappa':>7} {'p':>7} "
           f"{'N_eff':>6} {'W_L/Om':>8} | "
           f"{'ICC_3L':>12} {'VR_3L':>10} {'VR_4L':>10}")
    print(hdr)
    print("-" * 90)


def print_row(M, Omega, N, kappa, res):
    """res: dict from compute_row. Guard-triggered rows are recomputed in
    high precision when possible (marked '*'); otherwise annotated."""
    p = solve_p(M, kappa, Omega)
    N_eff = min(M, Omega)
    if np.isfinite(p):
        C_free = M - Omega * p          # free ligand W_L = M - E[C|W=M]
        C_str = f"{C_free / Omega:8.4f}"
        p_str = f"{p:7.4f}"
    else:
        p_str, C_str = "sat", "N/A"

    if np.isfinite(res['VR_3L']) and np.isfinite(res['VR_4L']):
        icc_str = f"{res['ICC_3L']:12.4e}"
        vr3_str = f"{res['VR_3L']:10.6f}"
        vr4_str = f"{res['VR_4L']:10.6f}"
        tag = " "
    elif HAS_MP and M <= MP_MAX_M:
        res_mp = mp_compute_row(M, Omega, N, kappa)
        icc_str = f"{res_mp['ICC_3L']:12.4e}"
        vr3_str = f"{res_mp['VR_3L']:10.6f}"
        vr4_str = f"{res_mp['VR_4L']:10.6f}"
        tag = "*"   # * = 50-digit recomputation of a guard-triggered row
    else:
        icc_str = "    saturated"
        vr3_str = "  ~1 (ICC3)"
        vr4_str = "unresolved"
        tag = " "

    print(f"{M:6d} {Omega:6d} {N:3d} {kappa:7.3f} {p_str:>7} "
          f"{N_eff:6d} {C_str:>8} | {icc_str} {vr3_str} {vr4_str}{tag}")


def main():
    print("VR4L Verification: N_eff vs p vs C/Omega  (v5)")
    print(f"Python {sys.version.split()[0]}, mpmath: {'yes' if HAS_MP else 'NO'}")
    print("(* = 50-digit recomputation of guard-triggered row; 'saturated' rows:")
    print(" 1-P1 below double resolution, VR4L unresolved (approaches N in the")
    print(" deep low-kappa corner, confirmed at N_eff=200: VR4L=5.000)")
    print()

    # 1. Original Table S2b.2
    print_header("1. Original Table S2b.2")
    for M, Omega, N, kappa in [(5, 10, 5, 0.001), (50, 100, 5, 0.5), (100, 200, 5, 0.5)]:
        print_row(M, Omega, N, kappa, compute_row(M, Omega, N, kappa))

    # 2. N_eff=5, vary kappa
    print_header("2. N_eff=5, vary kappa — Does p control VR4L at fixed N_eff?")
    for kappa in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
        print_row(5, 10, 5, kappa, compute_row(5, 10, 5, kappa))

    # 3. N_eff=10, vary kappa
    print_header("3. N_eff=10, vary kappa")
    for kappa in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]:
        print_row(10, 20, 5, kappa, compute_row(10, 20, 5, kappa))

    # 4. N_eff=50, vary kappa (KEY)
    print_header("4. N_eff=50, vary kappa — KEY TEST: high p with large N_eff")
    for kappa in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]:
        print_row(50, 100, 5, kappa, compute_row(50, 100, 5, kappa))

    # 5. N_eff=100, vary kappa
    print_header("5. N_eff=100, vary kappa")
    for kappa in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]:
        print_row(100, 200, 5, kappa, compute_row(100, 200, 5, kappa))

    # 6. N_eff=500, vary kappa
    print_header("6. N_eff=500, vary kappa")
    for kappa in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
        print_row(500, 1000, 5, kappa, compute_row(500, 1000, 5, kappa))

    # 7. Same p~0.5, vary N_eff (KEY)
    print_header("7. p~0.5 (kappa=0.001, M/Omega=0.5), vary N_eff — KEY: same p, different N_eff")
    for M, Omega in [(5, 10), (10, 20), (20, 40), (50, 100), (100, 200), (200, 400), (500, 1000)]:
        print_row(M, Omega, 5, 0.001, compute_row(M, Omega, 5, 0.001))

    # 8. Same p~0.29, vary N_eff
    print_header("8. p~0.29 (kappa=0.5, M/Omega=0.5), vary N_eff")
    for M, Omega in [(5, 10), (10, 20), (20, 40), (50, 100), (100, 200), (200, 400)]:
        print_row(M, Omega, 5, 0.5, compute_row(M, Omega, 5, 0.5))

    # 9. M/Omega=0.1 (dilute)
    print_header("9. M/Omega=0.1 (dilute), vary N_eff and kappa")
    for M, Omega in [(5, 50), (50, 500), (100, 1000)]:
        for kappa in [0.001, 0.1, 1.0, 10.0]:
            print_row(M, Omega, 5, kappa, compute_row(M, Omega, 5, kappa))

    # 10. M/Omega=0.9 (near saturation); Omega divisible by N=5
    print_header("10. M/Omega=0.9 (near saturation), vary N_eff and kappa")
    for M, Omega in [(9, 10), (45, 50), (90, 100)]:
        for kappa in [0.001, 0.1, 1.0, 10.0]:
            print_row(M, Omega, 5, kappa, compute_row(M, Omega, 5, kappa))

    # Summary
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print("""
H1: VR4L controlled by p (SI's "p<<1" condition)
  REFUTED: p=0.499 + N_eff=50 -> VR4L=1.0001 (perfect despite p not <<1)
           p=0.219 + N_eff=50 -> VR4L=1.0013 (worse than p=0.499!)

H2: VR4L controlled by occupancy C/Omega (= p), or by free fraction (M-Omega p)/Omega
  REFUTED for both readings: at fixed occupancy p=0.499, VR4L runs
           1.379 (N_eff=5) -> 1.0001 (N_eff=50): occupancy unchanged, VR4L changes.
           Free fraction ~0.001 (Row 1) gives 1.38; ~0.46 (kappa=10, N_eff=5) gives 1.000.

H3: VR4L controlled by N_eff = min(M, Omega) — within the non-saturated region
  CONFIRMED with saturation caveat:
    N_eff=5:   VR4L ranges 1.00-1.38 (kappa as 2nd-order effect)
    N_eff=50:  VR4L <= 1.0014 for every resolvable kappa (0.001 to 100)
    N_eff=100: VR4L <= 1.0002 for resolvable kappa; guarded points recomputed
               in 50 digits give 1.0048 (kappa=0.001), 1.0032 (kappa=0.01)
  SATURATED CORNER (1-P1 below ~1e-8): ICC_4L -> +1 and VR4L -> N
               (e.g. N_eff=200, kappa=0.001: VR4L=5.000 in 50-digit arithmetic).
               Points on the way into the corner already show VR4L rising again
               with N_eff (kappa=0.5, N_eff=200: VR4L=1.029), so larger N_eff
               does NOT help there. Both variances vanish in the corner, which
               is therefore outside the operating range of the Binomial
               approximation (cf. the delta-method failure at saturation,
               Fig. S2b.1).

Conclusion: The correct condition for Binomial collapse (VR4L~1) is
  N_eff = min(M, Omega) >= 50 with partitions not exponentially saturated,
  NOT p << 1.

The SI statements attributing Row 1 failure to "p << 1" or "C << Omega"
are incorrect and should be revised (see S2b modification list, items P1-x).
""")
    print("Done.")


if __name__ == "__main__":
    main()