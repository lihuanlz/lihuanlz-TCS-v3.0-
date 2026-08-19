# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 19:17:09 2026

@author: lihua
"""

                      # -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:18:45 2026

@author: lihua
"""

# -*- coding: utf-8 -*-
"""
Verification of SI subsection S2b.7/S2b.8: estimator optimality (R1 vs the
Cramer-Rao bound) and detection-limit tail theory (v1, 2026-07-30)

WHAT THIS SCRIPT COMPUTES (all exact, no Monte Carlo anywhere)
--------------------------------------------------------------
The exact distribution P(Z=z; M) of the positive-partition count Z in the
four-layer model:
    W ~ Poisson(M)  ->  C|W canonical ensemble (KV = kappa*Omega)
    -> uniform subset of C occupied sites among Omega = N*beta sites
    -> X_i = 1{partition i nonempty}; observe n of the N partitions
    -> background: X_i^obs = X_i OR Bernoulli(b);  Z = sum of observed.
Allocation layer P(S=s|C) is computed by a generating-function DP:
    A(C,s) = C(N,s) * [t^C] ((1+t)^beta - 1)^s,
with the kernel normalised by 2^(-beta) to avoid overflow, and EXACT ZEROS
MASKED (never floored): a numerical floor of 1e-300 multiplied back by
2^(beta*s) resurrects as phantom mass at small C (bug found and fixed
2026-07-30; moment check below now passes at all M including M=1).
Fisher information uses the Poisson score identity
    d p(W;M)/dM = p(W;M) * (W/M - 1),
so dP(Z=z;M)/dM is exact (no finite differences).

OUTPUTS
  CHECK 1  moments of the exact distribution vs deficit-form moment formulas
  CHECK 2  dilute limit: ICC = 0 (exact Binomial) sanity
  TABLE A  CV of R1 (exact estimator distribution) vs CV_CRB, panel-1 geometry
  TABLE B  dilute (panel-2) CV_CRB(M) and the information-theoretic LoQ
  TABLE C  bias-corrected R1 (second-order delta), exact validation
  TABLE D  LoD power curves: exact vs normal vs Edgeworth
  TABLE E  beta=2 over-dispersion stress test (phase-diagram red zone)
  TABLE F  limit-law check: exact vs two-stage Poisson mixture (TV distance)

Verified numbers behind the SI text (subsection S2b.7/S2b.8, Table S2b.3).
Runtime: a few minutes (TABLE A dominates). Requires only numpy/scipy.
"""

import numpy as np
from scipy.special import gammaln
from scipy.stats import binom, norm
from scipy.optimize import brentq

np.seterr(all='ignore')

# ============================================================
# Exact engine
# ============================================================

def canonical_probs(W, Omega, KV):
    """P(C|W), canonical ensemble (S1c.3), log-shift normalised."""
    maxC = min(W, Omega); C = np.arange(maxC + 1)
    lw = (gammaln(W + 1) - gammaln(C + 1) - gammaln(W - C + 1)
          + gammaln(Omega + 1) - gammaln(C + 1) - gammaln(Omega - C + 1)
          + gammaln(C + 1) - C * np.log(KV))
    lw -= lw.max(); w = np.exp(lw)
    return C, w / w.sum()


def poisson_grid(M):
    """W grid and Poisson(M) weights, tail-safe (14 sigma)."""
    Wmax = int(M + 14 * np.sqrt(M + 1)) + 2
    W = np.arange(0, Wmax + 1)
    lw = W * np.log(M) - gammaln(W + 1) - M
    pw = np.exp(lw - lw.max())
    return W, pw / pw.sum()


def alloc_DP(Omega, N, beta, Cmax):
    """B[s, C] = [t^C]((1+t)^beta - 1)^s * 2^(-beta*s), s=0..N, C=0..Cmax.
    Normalised kernel keeps all entries in [0, 1]."""
    k = np.arange(0, min(beta, Cmax) + 1)
    kern = np.exp(gammaln(beta + 1) - gammaln(k + 1) - gammaln(beta - k + 1)) * 2.0 ** (-beta)
    kern[0] = 0.0  # a nonempty partition holds >= 1 occupied site
    B = np.zeros((N + 1, Cmax + 1)); B[0, 0] = 1.0
    for s in range(1, N + 1):
        B[s] = np.convolve(B[s - 1], kern)[:Cmax + 1]
    return B


def PZ_given_W(W, Omega, N, beta, KV, B, n_obs):
    """P(Z_sig = z | W), z = 0..n_obs, marginalised over C|W.
    Exact-zero entries of B are MASKED (never replaced by a floor)."""
    C, w = canonical_probs(W, Omega, KV)
    C = C.astype(int)
    logC_Om = gammaln(Omega + 1) - gammaln(C + 1) - gammaln(Omega - C + 1)
    PzC = np.zeros((len(C), n_obs + 1))
    logC_N = gammaln(N + 1) - gammaln(np.arange(N + 1) + 1) - gammaln(N - np.arange(N + 1) + 1)
    for s in range(0, N + 1):
        BsC = B[s, C]
        pos = BsC > 0
        if not np.any(pos):
            continue
        logQ = np.full(len(C), -np.inf)
        logQ[pos] = np.log(BsC[pos]) + beta * s * np.log(2) + logC_N[s] - logC_Om[pos]
        Qs = np.exp(np.clip(logQ, -745, 50))
        # observing n_obs of N exchangeable partitions: hypergeometric subsample
        zmax = min(s, n_obs)
        zs = np.arange(0, zmax + 1)
        hw = np.exp(gammaln(s + 1) - gammaln(zs + 1) - gammaln(s - zs + 1)
                    + gammaln(N - s + 1) - gammaln(n_obs - zs + 1) - gammaln(N - s - n_obs + zs + 1)
                    - (gammaln(N + 1) - gammaln(n_obs + 1) - gammaln(N - n_obs + 1)))
        PzC[:, :zmax + 1] += Qs[:, None] * hw[None, :]
    return np.sum(w[:, None] * PzC, axis=0)


def PZ_and_score(M, Omega, N, beta, kappa, n_obs, b=0.0, B=None):
    """Exact P(Z=z) and dP(Z=z)/dM (Poisson score), z = 0..n_obs."""
    if B is None:
        B = alloc_DP(Omega, N, beta, min(Omega, int(M + 14 * np.sqrt(M + 1)) + 2))
    W, pw = poisson_grid(M); KV = kappa * Omega
    Pz = np.zeros(n_obs + 1); Sz = np.zeros(n_obs + 1)
    for i, wv in enumerate(W):
        if pw[i] < 1e-15:
            continue
        pzw = PZ_given_W(int(wv), Omega, N, beta, KV, B, n_obs)
        Pz += pw[i] * pzw
        Sz += pw[i] * (wv / M - 1.0) * pzw
    if b > 0:  # Z = Z_sig + Binomial(n_obs - Z_sig, b)
        PzB = np.zeros(n_obs + 1); SzB = np.zeros(n_obs + 1)
        for s in range(n_obs + 1):
            if Pz[s] < 1e-18 and abs(Sz[s]) < 1e-18:
                continue
            ks = np.arange(0, n_obs - s + 1)
            bm = binom.pmf(ks, n_obs - s, b)
            PzB[s:s + len(ks)] += Pz[s] * bm
            SzB[s:s + len(ks)] += Sz[s] * bm
        Pz, Sz = PzB, SzB
    return Pz / Pz.sum(), Sz / Pz.sum()


def fisher_exact(M, Omega, N, beta, kappa, n_obs, b=0.0, B=None):
    """I(Z;M) = sum_z (dP/dM)^2 / P; returns (I, Pz)."""
    Pz, Sz = PZ_and_score(M, Omega, N, beta, kappa, n_obs, b, B)
    return np.sum(Sz ** 2 / np.maximum(Pz, 1e-300)), Pz


def eps_k_marginal(M, Omega, N, beta, kappa, k):
    """eps_k = P(k specified partitions all empty), deficit form (positive sums)."""
    KV = kappa * Omega
    W, pw = poisson_grid(M)
    tot = 0.0
    for i, wv in enumerate(W):
        if pw[i] < 1e-15:
            continue
        C, w = canonical_probs(int(wv), Omega, KV)
        r = np.exp(gammaln(Omega - k * beta + 1) - gammaln(Omega - k * beta - C + 1)
                   - gammaln(Omega + 1) + gammaln(Omega - C + 1))
        tot += pw[i] * np.sum(w * r)
    return tot


def ppos_exact_and_deriv(M, Omega, N, beta, kappa, b=0.0):
    """P_pos(M) = b + (1-b)(1-eps_1) and dP_pos/dM (Poisson score)."""
    KV = kappa * Omega
    W, pw = poisson_grid(M)
    e1 = 0.0; de1 = 0.0
    for i, wv in enumerate(W):
        if pw[i] < 1e-15:
            continue
        C, w = canonical_probs(int(wv), Omega, KV)
        r = np.exp(gammaln(Omega - beta + 1) - gammaln(Omega - beta - C + 1)
                   - gammaln(Omega + 1) + gammaln(Omega - C + 1))
        ew = np.sum(w * r)
        e1 += pw[i] * ew
        de1 += pw[i] * (wv / M - 1.0) * ew
    return b + (1 - b) * (1 - e1), -(1 - b) * de1


def r1_invert(Ph, kappa, Omega, beta, b):
    """R1 inversion P_hat -> M_hat (master equation), with background correction."""
    Ps = (Ph - b) / (1 - b)
    if Ps <= 0 or Ps >= 1:
        return np.nan
    x = 1 - Ps
    pp = 1 - x ** (1.0 / beta)
    return (pp / (1 - pp) + pp / kappa) * kappa * Omega


def moments_z(Pz):
    zs = np.arange(len(Pz))
    mu = np.sum(zs * Pz)
    var = np.sum(zs ** 2 * Pz) - mu ** 2
    k3 = np.sum((zs - mu) ** 3 * Pz)
    return mu, var, k3 / var ** 1.5 if var > 0 else np.nan


# ============================================================
# Geometry (matches Fig. S2b.1)
# ============================================================
Om1, N1, beta1, n1, b1 = 1000, 50, 20, 50, 0.01      # panel 1: non-dilute
N2, n2, beta2, b2 = 20000, 15000, 50, 0.01           # panel 2: dilute (Omega2 = N2*beta2 = 1e6)
Om3, N3, beta3, n3 = 100, 50, 2, 50                  # phase-diagram red zone

print("Building allocation DP (panel-1 geometry, this is the slow precomputation)...")
B1 = alloc_DP(Om1, N1, beta1, Om1)
B3 = alloc_DP(Om3, N3, beta3, Om3)
print("done.\n")

# ============================================================
# CHECK 1: exact-distribution moments vs deficit-form formulas
# ============================================================
print("CHECK 1: moments of P(Z;M) vs deficit-form moment formulas (b=0)")
ok = True
for M in [1, 5, 50, 200, 350]:
    Pz, _ = PZ_and_score(M, Om1, N1, beta1, 1.0, n1, b=0.0, B=B1)
    mu, var, _ = moments_z(Pz)
    e1 = eps_k_marginal(M, Om1, N1, beta1, 1.0, 1)
    e2 = eps_k_marginal(M, Om1, N1, beta1, 1.0, 2)
    p1 = 1 - e1
    ICC = (e2 - e1 ** 2) / (e1 * (1 - e1))
    mu_f = n1 * p1
    var_f = n1 * p1 * (1 - p1) * (1 + (n1 - 1) * ICC)
    match = abs(mu - mu_f) < 1e-6 and abs(var - var_f) < 1e-6
    ok &= match
    print(f"  M={M:4d}: E[Z] dist={mu:9.4f} formula={mu_f:9.4f} | "
          f"Var dist={var:9.4f} formula={var_f:9.4f} | {'OK' if match else 'FAIL'}")
print(f"CHECK 1 {'PASSED' if ok else 'FAILED'}\n")

# ============================================================
# CHECK 2: dilute limit has ICC = 0 (Z exactly Binomial)
# ============================================================
kap_d, N_d = 1.0, 20000
gam = 1.0 / (1 + kap_d)
Md = 100
lam = Md * gam / N_d
e1d = np.exp(-lam)               # eps_k = exp(-k*M*gamma/N) in the Poisson-thinning limit
e2d = np.exp(-2 * lam)
ICC_d = (e2d - e1d ** 2) / (e1d * (1 - e1d))
print(f"CHECK 2: dilute ICC = {ICC_d:.2e} (exactly 0 by construction); "
      f"Z ~ Binomial(n, b+(1-b)(1-exp(-M/((1+kappa)N))))\n")

# ============================================================
# TABLE A: CV_R1 (exact estimator distribution) vs CV_CRB, panel 1
# ============================================================
print("TABLE A: R1 vs Cramer-Rao bound, panel-1 geometry (Omega=1000, N=n=50, beta=20, b=0.01)")
print("  [2026-07-31] added two columns for Table S2b.3 (audit item C-3):")
print("    diff% = bias% - pred%  (oracle residual: predicted term evaluated at the true P_pos)")
print("    corr% = exact residual bias of the implementable bias-corrected estimator")
print("            M_corr = M_hat - 0.5*f''(P_hat)*Var(P_hat)  (Eq. (S2b.8), per-trial f'')")
print("    NOTE: diff% and corr% diverge at larger M (f'' is convex, E[f''(P_hat)] > f''(E[P_hat]));")
print("    corr% verifies the SI text 'subtracting the term gives a bias-corrected estimator'")
print("    and matches TABLE C below. Use corr% as the new last column of Table S2b.3.")
print(f"{'M':>5s} {'kap':>5s} {'CV_R1':>8s} {'CV_CRB':>8s} {'ratio':>6s} {'bias%':>7s} {'pred%':>7s} {'diff%':>8s} {'corr%':>8s}")
tableA = {}
for kap in [0.1, 1.0]:
    for M in [10, 50, 100, 200, 500, 800]:
        I, Pz = fisher_exact(M, Om1, N1, beta1, kap, n1, b=b1, B=B1)
        CV_CRB = np.sqrt(1.0 / I) / M
        zs = np.arange(n1 + 1)
        Mh = np.array([r1_invert(z / n1, kap, Om1, beta1, b1) for z in zs])
        okm = np.isfinite(Mh) & (Pz > 1e-15)
        P_ok = Pz[okm] / Pz[okm].sum()
        EM = np.sum(P_ok * Mh[okm]); VM = np.sum(P_ok * Mh[okm] ** 2) - EM ** 2
        CV_R1 = np.sqrt(VM) / EM; bias = (EM - M) / M * 100
        Ppos, _ = ppos_exact_and_deriv(M, Om1, N1, beta1, kap, b=b1)
        h = 1e-4
        f2 = (r1_invert(Ppos + h, kap, Om1, beta1, b1)
              - 2 * r1_invert(Ppos, kap, Om1, beta1, b1)
              + r1_invert(Ppos - h, kap, Om1, beta1, b1)) / h ** 2
        VPh = np.sum(P_ok * (zs[okm] / n1) ** 2) - np.sum(P_ok * (zs[okm] / n1)) ** 2
        bias_pred = 0.5 * f2 * VPh / M * 100
        # --- added 2026-07-31: bias after correction (per-trial exact, Eq. (S2b.8)) ---
        Ph = zs[okm] / n1
        f2h = np.array([(r1_invert(p + h, kap, Om1, beta1, b1)
                         - 2 * r1_invert(p, kap, Om1, beta1, b1)
                         + r1_invert(p - h, kap, Om1, beta1, b1)) / h ** 2 for p in Ph])
        Mc = Mh[okm] - 0.5 * f2h * VPh
        bias_corr = (np.sum(P_ok * Mc) - M) / M * 100
        # --- end of addition ---
        tableA[(M, kap)] = (CV_R1, CV_CRB, bias, bias_pred)
        print(f"{M:5d} {kap:5.1f} {CV_R1:8.4f} {CV_CRB:8.4f} {CV_R1 / CV_CRB:6.3f} "
              f"{bias:7.2f} {bias_pred:7.2f} {bias - bias_pred:8.2f} {bias_corr:8.2f}")
print("(ratio < 1 at saturation reflects masking of P_hat=1 trials + bias; CRB applies to unbiased estimators)")
print("(diff%/corr% at M=500/800 are shown for completeness only; Table S2b.3 reports n/a there, as masking dominates)\n")

# ============================================================
# TABLE B: dilute (panel-2) information floor and LoQ
# ============================================================
def dilute_CV_CRB(M, kap):
    lam = M / (N2 * (1 + kap)); q = 1 - np.exp(-lam)
    P = b2 + (1 - b2) * q
    dP = (1 - b2) * np.exp(-lam) / (N2 * (1 + kap))
    return np.sqrt(P * (1 - P) / (n2 * dP ** 2)) / M

print("TABLE B: dilute (panel-2) CV_CRB(M); LoQ = M where CV_CRB = 25%")
for kap in [0.01, 0.1, 1.0, 10.0]:
    loq = brentq(lambda M: dilute_CV_CRB(M, kap) - 0.25, 1.0, 50000.0)
    print(f"  kap={kap:5.2f}: M_LoQ = {loq:7.0f}  (lambda = {loq / (N2 * (1 + kap)):.4f})")

print("\n  panel-1 LoQ (exact Fisher):")
for kap, br in [(1.0, (200.0, 400.0)), (0.1, (200.0, 300.0))]:
    f = lambda M: np.sqrt(1.0 / fisher_exact(M, Om1, N1, beta1, kap, n1, b=b1, B=B1)[0]) / M - 0.25
    loq = brentq(f, br[0], br[1], xtol=0.5)
    print(f"  kap={kap:4.1f}: M_LoQ = {loq:7.0f}")

# ============================================================
# TABLE C: bias-corrected R1, exact validation (panel 1, kap=1)
# ============================================================
print("\nTABLE C: bias-corrected R1, M_corr = M - 0.5 f''(P_hat) Var(P_hat)  (panel 1, kap=1)")
print(f"{'M':>5s} {'bias_R1%':>9s} {'bias_corr%':>10s} {'CV_R1':>7s} {'CV_corr':>8s}")
for M in [50, 100, 200]:
    kap = 1.0
    Pz, _ = PZ_and_score(M, Om1, N1, beta1, kap, n1, b=b1, B=B1)
    zs = np.arange(n1 + 1)
    Mh = np.array([r1_invert(z / n1, kap, Om1, beta1, b1) for z in zs])
    okm = np.isfinite(Mh) & (Pz > 1e-15)
    P_ok = Pz[okm] / Pz[okm].sum(); Ph = zs[okm] / n1
    h = 1e-4
    f2h = np.array([(r1_invert(p + h, kap, Om1, beta1, b1)
                     - 2 * r1_invert(p, kap, Om1, beta1, b1)
                     + r1_invert(p - h, kap, Om1, beta1, b1)) / h ** 2 for p in Ph])
    VPh = np.sum(P_ok * Ph ** 2) - np.sum(P_ok * Ph) ** 2
    Mc = Mh[okm] - 0.5 * f2h * VPh
    wm = lambda x: np.sum(P_ok * x)
    ws = lambda x: np.sqrt(np.sum(P_ok * x ** 2) - wm(x) ** 2)
    print(f"{M:5d} {(wm(Mh[okm]) - M) / M * 100:9.2f} {(wm(Mc) - M) / M * 100:10.2f} "
          f"{ws(Mh[okm]) / wm(Mh[okm]):7.4f} {ws(Mc) / wm(Mc):8.4f}")

# ============================================================
# TABLE D: LoD power curves: exact vs normal vs Edgeworth
# ============================================================
print("\nTABLE D: power P(Z>=z*; M) near the LoD (panel 1, b=0.01; z* from exact Binomial at M=0)")
thr = binom.ppf(0.95, n1, b1) + 1
alpha = 1 - binom.cdf(thr - 1, n1, b1)
print(f"  z* = {thr:.0f}, exact false-positive rate at M=0: alpha = {alpha:.4f}")
for kap in [1.0, 0.1]:
    lod = {}
    print(f"  --- kap={kap} ---")
    print(f"  {'M':>4s} {'exact':>8s} {'normal':>8s} {'Edgew':>8s} {'gamma1':>7s}")
    for M in range(1, 25):
        Pz, _ = PZ_and_score(M, Om1, N1, beta1, kap, n1, b=b1, B=B1)
        mu, var, g1 = moments_z(Pz)
        zs = np.arange(n1 + 1)
        tail_ex = np.sum(Pz[zs >= thr])
        x = (thr - 0.5 - mu) / np.sqrt(var)
        tail_no = 1 - norm.cdf(x)
        tail_ed = 1 - norm.cdf(x) + g1 / 6 * (x ** 2 - 1) * norm.pdf(x)
        if tail_ex >= 0.95 and 'ex' not in lod: lod['ex'] = M
        if tail_no >= 0.95 and 'no' not in lod: lod['no'] = M
        if tail_ed >= 0.95 and 'ed' not in lod: lod['ed'] = M
        if M <= 8 or M % 4 == 0:
            print(f"  {M:4d} {tail_ex:8.4f} {tail_no:8.4f} {tail_ed:8.4f} {g1:7.3f}")
    print(f"  LoD (95% power): exact={lod.get('ex')}, normal={lod.get('no')}, Edgeworth={lod.get('ed')}")

# ============================================================
# TABLE E: beta=2 stress test (phase-diagram red zone)
# ============================================================
print("\nTABLE E: beta=2, Omega=100, N=n=50, kap=0.01 (VR up to ~1.6); tails still tracked")
thr3 = binom.ppf(0.95, n3, b1) + 1
print(f"{'M':>4s} {'u':>5s} {'VR':>6s} {'gamma1':>7s} {'exact':>8s} {'normal':>8s} {'Edgew':>8s}")
for M in [3, 5, 8, 12, 20, 30, 50]:
    e1 = eps_k_marginal(M, Om3, N3, beta3, 0.01, 1)
    e2 = eps_k_marginal(M, Om3, N3, beta3, 0.01, 2)
    ICC = (e2 - e1 ** 2) / (e1 * (1 - e1))
    VR = 1 + (N3 - 1) * ICC
    Pz, _ = PZ_and_score(M, Om3, N3, beta3, 0.01, n3, b=b1, B=B3)
    mu, var, g1 = moments_z(Pz)
    zs = np.arange(n3 + 1)
    tail_ex = np.sum(Pz[zs >= thr3])
    x = (thr3 - 0.5 - mu) / np.sqrt(var)
    tail_no = 1 - norm.cdf(x)
    tail_ed = 1 - norm.cdf(x) + g1 / 6 * (x ** 2 - 1) * norm.pdf(x)
    print(f"{M:4d} {M / Om3:5.2f} {VR:6.2f} {g1:7.2f} {tail_ex:8.4f} {tail_no:8.4f} {tail_ed:8.4f}")

# ============================================================
# TABLE F: limit law — exact vs two-stage Poisson mixture
# ============================================================
print("\nTABLE F: limit-law check Z/n -> q(C(W)); two-stage mixture Z ~ Binomial(n, q(C))")
M = 30
Pz_ex, _ = PZ_and_score(M, Om3, N3, beta3, 0.01, n3, b=0.0, B=B3)
W, pw = poisson_grid(M)
Pz_mix = np.zeros(n3 + 1)
for i, wv in enumerate(W):
    if pw[i] < 1e-15:
        continue
    C, w = canonical_probs(int(wv), Om3, 0.01 * Om3)
    q = 1 - np.exp(gammaln(Om3 - beta3 + 1) - gammaln(Om3 - beta3 - C + 1)
                   - gammaln(Om3 + 1) + gammaln(Om3 - C + 1))
    for j in range(len(C)):
        Pz_mix += pw[i] * w[j] * binom.pmf(np.arange(n3 + 1), n3, q[j])
tv = 0.5 * np.sum(np.abs(Pz_ex - Pz_mix))
mu_e, var_e, _ = moments_z(Pz_ex)
mu_m, var_m, _ = moments_z(Pz_mix)
print(f"  beta=2, M=30, N=50: TV distance = {tv:.4f}")
print(f"  exact: E={mu_e:.3f} sd={np.sqrt(var_e):.3f} | mixture: E={mu_m:.3f} sd={np.sqrt(var_m):.3f}")
print("  (mixture OVERESTIMATES the spread: it drops the negative hypergeometric")
print("   covariance Cov(X_i,X_j|C) < 0; the correction vanishes as C/N -> 0)")

# ============================================================
# SUMMARY
# ============================================================
print("\n================ SUMMARY (numbers quoted in S2b.7/S2b.8) ================")
# working range = rows with negligible masking (|bias| < 5%)
r = [v[0] / v[1] for v in tableA.values() if abs(v[2]) < 5.0]
print(f"1. CV_R1 / CV_CRB in the working range (|bias|<5%): {min(r):.3f} - {max(r):.3f}")
print("2. Dilute LoQ (25% CV): M = 78 / 85 / 155 / 851 for kappa = 0.01/0.1/1/10 (TABLE B)")
print("3. Panel-1 LoQ (25% CV): M = 381 (kap=1), 210 (kap=0.1) (TABLE B)")
print("4. Bias-corrected R1: +1.4/+1.8% -> -0.0/-0.2% at M=50/100; +3.7% -> -2.4% at M=200 (TABLE C)")
print("5. Saturation: CV_CRB diverges (0.35 at M=500, 0.94 at M=800, kap=1) — information,")
print("   not estimator, is lost; no statistical remedy (TABLE A)")
print("6. LoD: exact = 12 (kap=1) / 7 (kap=0.1); normal = 13 / 7 — conservative, <=1 molecule (TABLE D)")
print("7. beta=2 red zone: |tail error| <= ~3% even at VR=1.6; LoD lives at low u where VR~1 (TABLE E)")
print("8. Limit law: Z/n converges to the Poisson mixture q(C(W)), not a constant;")
print("   mixture sd overestimates (sign-known); TV = 0.12 at the beta=2 corner (TABLE F)")
print("=========================================================================")