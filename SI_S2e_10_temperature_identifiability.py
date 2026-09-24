#!/usr/bin/env python3
"""
SI S2e.10 — Temperature protocols and scale degeneracy: structural identifiability + Fisher cost
=================================================================================================
Deterministic verification script. No randomness in the reported numbers.

Model
-----
Digital partition readout in the dilute limit (S2c.3.2):
    P_pos(T) = 1 - exp(-Y(T)/N),   Y(T) = M / (1 + kappa(T)),
    kappa(T) = kappa0 * exp(-dH/R * (1/T - 1/T0))          (van 't Hoff, constant dH)
Unknowns: (M, kappa0, dH). Observable per temperature: Y_j (via -N ln(1 - P_pos)).

Results produced
----------------
1. Symbolic determinant of the log-parameter Jacobian for n_T = 3
   (identifiability condition: dH != 0, distinct temperatures, 0 < kappa0 < inf).
2. Fisher CV(M) design table (delta-method counting noise, geometry of Fig. S2b.1:
   N = 20000 partitions, n = 15000 observed, M = 100 molecules).
"""
import numpy as np
import sympy as sp

R = 8.314  # J mol^-1 K^-1

# ---------- 1. Structural identifiability (symbolic) ----------
s0, s1, s2, u1, u2 = sp.symbols('s0 s1 s2 u1 u2')
J = sp.Matrix([[1, -s0, 0], [1, -s1, s1*u1], [1, -s2, s2*u2]])
det = sp.expand(J.det())
print('det J (n_T=3) =', det)
print('det = 0 iff dH = 0 (s0 = s1 = s2) or any u_j = 0 (T_j = T_0); nonzero otherwise.')
print('kappa0 -> 0: s_j -> 0, first column stays (1,1,1): M remains identifiable, (kappa0, dH) do not.')
print()

# ---------- 2. Fisher design table (numeric, deterministic) ----------
def fisher_cv(Ts, kap0, dH, M=100.0, N=20000, n=15000, reps=1):
    T0 = Ts[0]; span = 1/Ts[0] - 1/Ts[-1]; psi = dH/R*span
    rows, sig = [], []
    for T in Ts:
        tau = (1/T0 - 1/T)/span
        kj = kap0*np.exp(psi*tau); sj = kj/(1+kj)
        rows.append([1.0, -sj, -sj*tau])
        Yj = M/(1+kj); P = 1 - np.exp(-Yj/N)
        sig.append(np.sqrt(N*N*P/(n*(1-P)))/Yj)   # delta-method counting noise
    Jm = np.array(rows); W = np.diag(reps/np.array(sig)**2)
    F = Jm.T @ W @ Jm
    return float(np.sqrt(np.linalg.solve(F, np.array([1.0,0,0]))[0]))

def floor_cv(Ts, kap0, dH, M=100.0, N=20000, n=15000):
    w = 0.0
    for T in Ts:
        kj = kap0*np.exp(dH/R*(1/Ts[0]-1/T)); Yj = M/(1+kj); P = 1-np.exp(-Yj/N)
        w += 1.0/(N*N*P/(n*(1-P))/Yj**2)
    return float(np.sqrt(1/w))

T3 = [298.15, 308.15, 318.15]                 # 3 temperatures, 20 K span
T6 = list(np.linspace(298.15, 323.15, 6))     # 6 temperatures, 25 K span
print('CV(M) at M = 100, dH = 40 kJ/mol (Fisher, idealized counting noise)')
print(f'{"kappa0":>7} {"floor(kappa known)":>19} {"3T x 1":>8} {"3T x 9":>8} {"6T x 1":>8} {"6T x 4":>8}')
for k0 in [0.01, 0.1, 0.3, 1.0]:
    print(f'{k0:>7} {floor_cv(T3,k0,40000):>19.3f} {fisher_cv(T3,k0,40000):>8.3f} '
          f'{fisher_cv(T3,k0,40000,reps=9):>8.3f} {fisher_cv(T6,k0,40000):>8.3f} '
          f'{fisher_cv(T6,k0,40000,reps=4):>8.3f}')
print()
print('dH sensitivity (6T x 1, kappa0 = 0.1):',
      {dH: round(fisher_cv(T6,0.1,dH),3) for dH in [20000,40000,80000]})
print('dH -> 0 failure (6T x 1, kappa0 = 0.1, dH = 2 kJ/mol): CV =',
      round(fisher_cv(T6,0.1,2000),1))
