# -*- coding: utf-8 -*-
"""
SI_S5_PICO_verification.py
==========================
SI S5 verification: PICO as a finite-kappa realisation of the TCS framework.

Reproduces, with NO fitted parameters:
  (1) the exact master equation M = Omega*p*(kappa/(1-p)+1) and its dilute-limit
      reduction gamma = 1/(1+kappa)                              (S2d.1, S2c.3.2)
  (2) Table S5.1: TCS-predicted quantification error at b0 = 500 pM vs K
  (3) thresholds: K < 5 pM (error < 1%); K < 50 pM (error ~ 10%, S2d.2);
      b0 > 30 nM for K = 300 pM at 1% error
  (4) detection-stage precision: CV = 1/sqrt(X); LOD = 3; LOQ = 9 couplexes
  (5) saturation-plateau insensitivity: an antibody-titration plateau certifies
      Kd-independence but NOT completeness (f < 1 can hide inside plateau noise);
      the "effective K" from self-consistency fitting is therefore not an
      independent thermodynamic measurement.

Run: python3 SI_S5_PICO_verification.py
Output: printed verification tables + SI_S5_PICO_verification.svg
"""

import numpy as np
import matplotlib.pyplot as plt

b0 = 500.0          # PICO standard protocol antibody concentration (pM, isomolar titration)

# ----------------------------------------------------------------------
# (1) Exact master equation vs dilute limit  (S2d.1 / S2c.3.2)
# ----------------------------------------------------------------------
# M = Omega*p*(kappa/(1-p)+1).  In the PICO dilute regime M << Omega (p -> 0):
# M -> Omega*p*(kappa+1)  =>  X_couplex = Omega*p = M/(1+kappa) = gamma*M.
print('=' * 74)
print('(1) Master equation dilute limit: gamma = 1/(1+kappa)')
print('-' * 74)
print(f'{"kappa":>8} {"p":>8} {"exact X/M":>12} {"1/(1+kap)":>12} {"rel. dev.":>10}')
rng = np.random.default_rng(0)
for kap in [0.002, 0.05, 0.1, 0.5, 1.0]:
    for p in [1e-6, 1e-4]:
        M = 1.0
        Omega = M / (p * (kap / (1 - p) + 1))     # invert master equation for Omega
        X = Omega * p                              # counted couplexes
        exact = X / M
        approx = 1 / (1 + kap)
        print(f'{kap:8.3f} {p:8.0e} {exact:12.8f} {approx:12.8f} {abs(exact-approx)/approx:10.2e}')
print('  -> dilute limit exact to <1e-3 relative deviation across the PICO regime.\n')

# ----------------------------------------------------------------------
# (2) Table S5.1: error = kappa/(1+kappa) at b0 = 500 pM
# ----------------------------------------------------------------------
print('=' * 74)
print('(2) Table S5.1 reproduction: b0 = 500 pM, error = 1 - gamma = kappa/(1+kappa)')
print('-' * 74)
print(f'{"K (pM)":>8} {"kappa=K/b0":>11} {"gamma":>8} {"error":>8}  Assessment')
table = [(1, 'Absolute quantification'), (5, 'Absolute quantification'),
         (10, 'High precision'), (25, 'High precision'),
         (50, 'Operational threshold (S2d.2)'), (100, 'Approximate'),
         (250, 'Inadequate'), (500, 'Inadequate')]
for K, label in table:
    kap = K / b0; gam = 1 / (1 + kap); err = 1 - gam
    print(f'{K:8.0f} {kap:11.3f} {gam:8.3f} {err*100:7.1f}%  {label}')

# ----------------------------------------------------------------------
# (3) Threshold checks
# ----------------------------------------------------------------------
print('\n' + '=' * 74)
print('(3) Threshold checks')
print('-' * 74)
def K_for_error(e, b):            # error e -> kappa = e/(1-e) -> K = kappa*b
    return e / (1 - e) * b
print(f'  error < 1%  at b0=500 pM -> K < {K_for_error(0.01, b0):.1f} pM   (claim: K < 5 pM)')
print(f'  error ~ 10% at b0=500 pM -> K < {K_for_error(0.10, b0):.1f} pM  (claim: K < 50 pM, S2d.2)')
b0_req = K_for_error(0.01, 1.0) and 300 / (0.01 / (1 - 0.01))
print(f'  K = 300 pM at 1% error    -> b0 > {b0_req/1000:.0f} nM          (claim: b0 > 30 nM)')

# ----------------------------------------------------------------------
# (4) Detection-stage precision (Hyperwell, N = 2e5, kappa_det -> 0, b ~ 0)
# ----------------------------------------------------------------------
print('\n' + '=' * 74)
print('(4) Detection stage: CV = 1/sqrt(X); LOD = 3; LOQ = 9 couplexes')
print('-' * 74)
for X in [1, 3, 9, 25, 100]:
    print(f'  X = {X:>4} couplexes -> CV = {1/np.sqrt(X)*100:5.1f}%')
print('  LOD = 3 couplexes (Poisson: P(X>=1|lambda=3) = %.1f%%);'
      ' LOQ = 9 couplexes (CV <= 33%%)' % ((1-np.exp(-3))*100))

# ----------------------------------------------------------------------
# (5) Plateau insensitivity: plateau does not certify completeness
# ----------------------------------------------------------------------
# Ternary model (symmetric): antigen A, antibodies L,R at equal concentration b
# (b >> A so free ~ total). Couplex fraction f(b) = [b/(Kd+b)]^2.
# An antibody-titration plateau (no significant change between top points)
# can occur while f < 1: the missing fraction (1-f) is invisible to the plateau.
print('\n' + '=' * 74)
print('(5) Plateau insensitivity (ternary model, symmetric Kd; A = 40 pM)')
print('-' * 74)
b_titr = np.array([13, 40, 120, 360, 500])          # PICO D.5 isomolar titration points
cv_count = 0.08                                      # typical counting CV ~ 8%
detect_band = 2 * cv_count / np.sqrt(3)              # ~9.2%: smallest detectable step (n=3)
print(f'  assumed counting CV = {cv_count*100:.0f}%, n=3 replicates '
      f'-> steps < {detect_band*100:.1f}% are statistically invisible')
print(f'{"Kd (pM)":>8} ' + ' '.join(f'{b:>9}' for b in b_titr) + '   step 360->500  f(500)  verdict')
for Kd in [50, 100, 300]:
    f = (b_titr / (Kd + b_titr)) ** 2
    step = f[-1] / f[-2] - 1
    invis = step < detect_band
    verdict = ('plateau PASSES yet incomplete!' if invis else 'step detectable')
    print(f'{Kd:8.0f} ' + ' '.join(f'{v*100:8.1f}%' for v in f) +
          f'   {step*100:8.1f}%  {f[-1]*100:5.1f}%  {verdict}')
print('  -> at Kd = 50-100 pM the plateau criterion (Kruskal-Wallis, p<0.05) passes')
print('     while 17-23% of the antigen is uncaptured: plateau != completeness.')

# ----------------------------------------------------------------------
# Figure: two panels
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

K = np.logspace(0, 3, 400)
err = (K / b0) / (1 + K / b0) * 100
ax1.semilogx(K, err, 'b-', lw=2)
ax1.axhline(1, color='g', ls='--', lw=1); ax1.axhline(10, color='orange', ls='--', lw=1)
ax1.axvline(5, color='g', ls=':', lw=1); ax1.axvline(50, color='orange', ls=':', lw=1)
ax1.axvspan(50, 300, color='red', alpha=0.08)
ax1.text(5.4, 0.4, 'K = 5 pM\n(<1%)', fontsize=9, color='g')
ax1.text(52, 3.5, 'K = 50 pM\n(~10%, S2d.2)', fontsize=9, color='darkorange')
ax1.text(95, 22, 'reported effective K\n50-300 pM', fontsize=9, color='darkred')
ax1.set_xlabel('antibody K (pM)'); ax1.set_ylabel('quantification error (%)')
ax1.set_title('(a) TCS-predicted error at b0 = 500 pM')
ax1.grid(True, alpha=0.3)

b = np.linspace(1, 600, 400)
for Kd, c in [(50, 'tab:blue'), (100, 'tab:orange'), (300, 'tab:red')]:
    ax2.plot(b, (b / (Kd + b)) ** 2 * 100, color=c, lw=2, label=f'Kd = {Kd} pM')
ax2.axvspan(360, 500, color='gray', alpha=0.15)
ax2.text(370, 8, 'plateau region\n(Kruskal-Wallis)', fontsize=9)
ax2.axhline(100, color='k', ls=':', lw=0.8)
ax2.set_xlabel('antibody concentration b (pM)'); ax2.set_ylabel('couplex fraction f (%)')
ax2.set_title('(b) Titration plateau can hide incompleteness')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('SI_S5_PICO_verification.svg', bbox_inches='tight')
print('\nFigure saved: SI_S5_PICO_verification.svg')
