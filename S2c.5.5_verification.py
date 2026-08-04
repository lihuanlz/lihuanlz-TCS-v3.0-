# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:33:18 2026

@author: lihua
"""

# -*- coding: utf-8 -*-
"""
S2c.5.5 verification suite
==========================
Verifies, independently of the original pipeline:

  1. solve_p: TCS master-equation root (round-trip recovery)
  2. Fisher determinant identity (Theorem S2c.5.5.1, Eq. S2c.5.5.4)
  3. Asymptotic scaling constants (Theorem S2c.5.5.2): C_inf = 5.547, W0^2 V0 = 20.959
  4. Full Brioschi formula (Eq. S2c.5.6.5, with determinant correction term)
     on test metrics: unit sphere (R = 2) and a general non-orthogonal metric
  5. L = 2 flatness theorem (Remark S2c.5.5.2): R = 0 across the grid
  6. L >= 3 extrinsic-curvature sign scan: K < 0, no R = 0 boundary

Coordinate convention (critical): the Fisher metric is derived for
theta = (kappa, ln M0), so all differentiation is done in the NATIVE
coordinates (kappa, ln M0). Differentiating the same components with
respect to (kappa, ln xi1) without the Jacobian J = [[1, 0], [1/kappa, 1]]
computes the curvature of a different, non-tensorial matrix field and
produces a spurious phase structure.
"""

import numpy as np

# ----------------------------------------------------------------------
# 0. Model
# ----------------------------------------------------------------------

def solve_p(kappa, xi):
    """Solve the TCS master equation xi = p/(1-p) + p/kappa for p."""
    if xi <= 0:
        return 1e-12
    disc = (kappa * xi + kappa - 1.0) ** 2 + 4.0 * kappa
    p = ((kappa + 1.0 + kappa * xi) - np.sqrt(disc)) / 2.0
    return np.clip(p, 1e-14, 1.0 - 1e-14)


def fisher_g(kappa, lnM0, r=10.0, L=2, n=100.0):
    """Fisher metric for theta = (kappa, ln M0), dilution series xi_l = xi_1 / r**l,
    xi_1 = M0 / (kappa * Omega) with Omega = 1. Components as in Eq. S2c.5.5.3."""
    M0 = np.exp(lnM0)
    p = np.array([solve_p(kappa, M0 / (kappa * r**l)) for l in range(L)])
    q = 1.0 - p
    D = kappa + q**2
    B = kappa + q
    w = n * p * q / D**2
    W = np.sum(w)
    Bbar = np.sum(w * B) / W
    V = np.sum(w * (B - Bbar) ** 2) / W
    return np.array([[W, -W * Bbar], [-W * Bbar, W * (Bbar**2 + V)]])


# ----------------------------------------------------------------------
# 1. solve_p round trip
# ----------------------------------------------------------------------

def check_solve_p():
    ok = True
    for kappa in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        for p in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]:
            xi = p / (1 - p) + p / kappa
            if abs(solve_p(kappa, xi) - p) > 1e-10:
                ok = False
    return ok


# ----------------------------------------------------------------------
# 2. Determinant identity  det(g) = W^2 V = sum_{l<m} w_l w_m (B_l - B_m)^2
# ----------------------------------------------------------------------

def check_det_identity():
    err = 0.0
    for kappa in [0.1, 1.0, 5.0]:
        for M0 in [0.1, 1.0, 10.0]:
            g = fisher_g(kappa, np.log(M0), L=3)
            det_direct = np.linalg.det(g)
            # right-hand side via weights
            p = np.array([solve_p(kappa, M0 / (kappa * 10.0**l)) for l in range(3)])
            q = 1 - p
            D = kappa + q**2
            B = kappa + q
            w = 100.0 * p * q / D**2
            rhs = 0.0
            for l in range(3):
                for m in range(l + 1, 3):
                    rhs += w[l] * w[m] * (B[l] - B[m]) ** 2
            err = max(err, abs(det_direct - rhs) / max(abs(rhs), 1e-30))
    return err


# ----------------------------------------------------------------------
# 3. Asymptotic constants (Theorem S2c.5.5.2)
# ----------------------------------------------------------------------

def check_asymptotics(r=10.0, n=100.0):
    # Langmuir constant, operating point p1 = 0.30
    p1, q1 = 0.30, 0.70
    xi1 = p1 / q1
    xi2 = xi1 / r
    p2 = xi2 / (1 + xi2)          # Langmuir-limit occupancy
    q2 = 1 - p2
    C_inf = (n * p1 * q1) * (n * p2 * q2) * (q1 - q2) ** 2
    # numerical det(g) * kappa^4 at large kappa
    kap = 1e3
    g = fisher_g(kap, np.log(kap * xi1), L=2)
    C_num = np.linalg.det(g) * kap**4
    # strong-depletion constant
    s1, s2 = 0.30, 0.03
    w1 = n * s1 / (1 - s1) ** 3
    w2 = n * s2 / (1 - s2) ** 3
    WV0 = w1 * w2 * (s1 - s2) ** 2
    kap = 1e-4
    g = fisher_g(kap, np.log(s1), L=2)  # strong-depletion occupancy s1 = 0.30 held fixed
    WV_num = np.linalg.det(g)
    return C_inf, C_num, WV0, WV_num


# ----------------------------------------------------------------------
# 4. Full Brioschi formula (Eq. S2c.5.6.5, corrected)
# ----------------------------------------------------------------------

def R_brioschi(g_func, u0, v0, h=1e-3):
    """Full Brioschi: derivative terms PLUS the 3x3 first-derivative
    determinant term. g_func(u, v) returns the 2x2 metric in the SAME
    coordinates (u, v) used for differentiation."""
    us = u0 + np.array([-2, -1, 0, 1, 2]) * h
    vs = v0 + np.array([-2, -1, 0, 1, 2]) * h
    g = np.zeros((5, 5, 2, 2))
    for i in range(5):
        for j in range(5):
            g[i, j] = g_func(us[i], vs[j])
    detg = g[:, :, 0, 0] * g[:, :, 1, 1] - g[:, :, 0, 1] ** 2
    if np.any(detg <= 0):
        return np.nan
    sg = np.sqrt(detg)
    C = np.zeros((5, 5))
    Dq = np.zeros((5, 5))
    for i in range(1, 4):
        for j in range(1, 4):
            g22u = (g[i + 1, j, 1, 1] - g[i - 1, j, 1, 1]) / (2 * h)
            g12v = (g[i, j + 1, 0, 1] - g[i, j - 1, 0, 1]) / (2 * h)
            g12u = (g[i + 1, j, 0, 1] - g[i - 1, j, 0, 1]) / (2 * h)
            g11v = (g[i, j + 1, 0, 0] - g[i, j - 1, 0, 0]) / (2 * h)
            C[i, j] = (g22u - g12v) / sg[i, j]
            Dq[i, j] = (g12u - g11v) / sg[i, j]
    dC = (C[3, 2] - C[1, 2]) / (2 * h)
    dD = (Dq[2, 3] - Dq[2, 1]) / (2 * h)
    gc = g[2, 2]
    gu = (g[3, 2] - g[1, 2]) / (2 * h)
    gv = (g[2, 3] - g[2, 1]) / (2 * h)
    M3 = np.linalg.det(np.array([
        [gc[0, 0], gc[0, 1], gc[1, 1]],
        [gu[0, 0], gu[0, 1], gu[1, 1]],
        [gv[0, 0], gv[0, 1], gv[1, 1]],
    ]))
    K = -1.0 / (2 * sg[2, 2]) * (dC - dD) - M3 / (4 * detg[2, 2] ** 2)
    return 2.0 * K


def check_brioschi_tests():
    R_sph = R_brioschi(lambda u, v: np.array([[1.0, 0.0], [0.0, np.sin(u) ** 2]]), 1.0, 0.5)
    R_non = R_brioschi(lambda u, v: np.array([[np.exp(u), u * v / 2], [u * v / 2, np.exp(v)]]), 1.0, 0.7)
    return R_sph, R_non  # expect 2.0 and 0.062121 (strict Christoffel value)


# ----------------------------------------------------------------------
# 5. L = 2 flatness (Remark S2c.5.5.2)
# ----------------------------------------------------------------------

def check_L2_flat():
    worst = 0.0
    for kappa in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for p1 in np.linspace(0.1, 0.9, 20):
            xi1 = p1 / (1 - p1) + p1 / kappa
            R = R_brioschi(lambda u, v: fisher_g(u, v, L=2), kappa, np.log(kappa * xi1), h=3e-4)
            if np.isfinite(R):
                worst = max(worst, abs(R))
    return worst  # expect ~ h-noise, true value 0


# ----------------------------------------------------------------------
# 6. L >= 3 extrinsic curvature sign scan
# ----------------------------------------------------------------------

def K_extrinsic(kappa, lnM0, r=10.0, L=3, n=100.0, h=1e-4):
    """Gaussian curvature of the 2D Fisher surface embedded in flat
    (phi_1..phi_L) space, phi_l = 2 sqrt(n) asin(sqrt(p_l)). Second
    fundamental form; valid for any L >= 3."""
    M0 = np.exp(lnM0)
    sq = np.sqrt(n)

    def derivs(k_, M_):
        X = np.zeros((L, 5))
        for l in range(L):
            xi_l = M_ / (k_ * r**l)
            xi_u, xi_v = -xi_l / k_, xi_l
            xi_uu, xi_vv, xi_uv = 2 * xi_l / k_**2, xi_l, -xi_l / k_
            p_ = solve_p(k_, xi_l)
            q_ = 1 - p_
            D = k_ + q_**2
            px = k_ * q_**2 / D
            pk = p_ * q_**2 / (k_ * D)
            # second derivatives via implicit differentiation (closed forms)
            # closed-form second derivatives (implicit differentiation of the
            # master equation, verified symbolically against SymPy):
            p_xx = -2.0 * k_**3 * q_**3 / D**3
            p_uu = -2.0 * p_ * q_**2 * (k_ + q_) / (k_ * D**3)
            p_ux = q_**3 * (2 * p_ * q_**2 + D * (q_ - 2 * p_)) / D**3
            p_u = pk + px * xi_u
            p_v = px * xi_v
            p_uu2 = p_uu + 2 * p_ux * xi_u + p_xx * xi_u**2 + px * xi_uu
            p_vv2 = p_xx * xi_v**2 + px * xi_vv
            p_uv2 = p_ux * xi_v + p_xx * xi_u * xi_v + px * xi_uv
            dphi = sq / np.sqrt(p_ * q_)
            d2phi = sq * (2 * p_ - 1) / (2 * (p_ * q_) ** 1.5)
            X[l] = [dphi * p_u, dphi * p_v,
                    d2phi * p_u**2 + dphi * p_uu2,
                    d2phi * p_u * p_v + dphi * p_uv2,
                    d2phi * p_v**2 + dphi * p_vv2]
        return X

    X = derivs(kappa, M0)
    Xu, Xv, Xuu, Xuv, Xvv = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    E, F, G = Xu @ Xu, Xu @ Xv, Xv @ Xv
    det1 = E * G - F * F
    Ginv = np.linalg.inv(np.array([[E, F], [F, G]]))

    def normal_part(Z):
        a = Ginv @ np.array([Xu @ Z, Xv @ Z])
        return Z - a[0] * Xu - a[1] * Xv

    IIuu, IIuv, IIvv = normal_part(Xuu), normal_part(Xuv), normal_part(Xvv)
    return (IIuu @ IIvv - IIuv @ IIuv) / det1


def check_L3_sign():
    Kmin, Kmax, n_pos = 0.0, 0.0, 0
    for kappa in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        for p1 in np.linspace(0.05, 0.95, 25):
            xi1 = p1 / (1 - p1) + p1 / kappa
            K = K_extrinsic(kappa, np.log(kappa * xi1), L=3)
            if np.isfinite(K):
                Kmin, Kmax = min(Kmin, K), max(Kmax, K)
                n_pos += K > 0
    return Kmin, Kmax, n_pos


# ----------------------------------------------------------------------
# Run all
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 66)
    print("S2c.5.5 verification suite")
    print("=" * 66)

    print("\n[1] solve_p round-trip recovery:", "PASS" if check_solve_p() else "FAIL")

    err = check_det_identity()
    print(f"[2] determinant identity (S2c.5.5.4): max rel err = {err:.2e}",
          "PASS" if err < 1e-8 else "FAIL")

    C_inf, C_num, WV0, WV_num = check_asymptotics()
    print(f"[3] Langmuir: C_inf = {C_inf:.3f} (expect 5.547), "
          f"det*kappa^4 -> {C_num:.3f}")
    print(f"    strong depletion: W0^2 V0 = {WV0:.3f} (expect 20.959), "
          f"det(kappa=1e-4) = {WV_num:.3f}, ratio = {WV_num / WV0:.4f}")

    R_sph, R_non = check_brioschi_tests()
    print(f"[4] Brioschi tests: sphere R = {R_sph:.6f} (expect 2.0), "
          f"non-orthogonal R = {R_non:.6f} (expect 0.062121)")

    worst = check_L2_flat()
    print(f"[5] L = 2 flatness: max |R| over grid = {worst:.2e} "
          f"(h-noise; true value 0)")

    Kmin, Kmax, n_pos = check_L3_sign()
    print(f"[6] L = 3 extrinsic curvature: K in [{Kmin:.3e}, {Kmax:.3e}], "
          f"positive points = {n_pos} (0 => no R = 0 boundary)")