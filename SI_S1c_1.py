# -*- coding: utf-8 -*-
"""
S1c Verification: exact eq (S1c.8) vs master eq (S1c.12)
Plot relative error in xi as function of N_eff and kappa.
Optimized for Spyder: no forced backend, progress bar, save then display.
"""

import numpy as np
from scipy.special import gammaln
import matplotlib.pyplot as plt
from tqdm import tqdm          # 进度条，如果没有请先 pip install tqdm
import matplotlib

def log_binom(n, k):
    """Log binomial coefficient."""
    if k < 0 or k > n:
        return -np.inf
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def compute_rel_err(M, Omega, kappa):
    """Relative error (percent) of master eq."""
    KV = kappa * Omega
    C_max = min(M, Omega)
    C_arr = np.arange(C_max + 1, dtype=float)

    # compute log terms efficiently
    log_terms = np.array([
        log_binom(M, int(c)) + log_binom(Omega, int(c)) + gammaln(c + 1) - c * np.log(KV)
        for c in C_arr
    ])
    log_Q = np.logaddexp.reduce(log_terms)
    P_C = np.exp(log_terms - log_Q)

    mean_C = np.sum(C_arr * P_C)
    var_C = np.sum(C_arr**2 * P_C) - mean_C**2
    p = mean_C / Omega

    if p <= 0 or p >= 1:
        return np.nan

    xi_exact = M / KV
    Delta = var_C / (kappa * Omega**2 * (1 - p))
    return Delta / xi_exact * 100


# ----- 参数网格 -----
Neff_vals = np.arange(3, 201)            # 198 个点
kappa_vals = np.logspace(-2, 2, 50)      # 50 个点

# 初始化结果矩阵
ERR = np.full((len(kappa_vals), len(Neff_vals)), np.nan)

# 双重循环 + 进度条
for i, kappa in enumerate(tqdm(kappa_vals, desc="kappa loop")):
    for j, neff in enumerate(Neff_vals):
        M, Omega = neff, neff
        ERR[i, j] = compute_rel_err(M, Omega, kappa)

# ----- 绘图 -----
fig, ax = plt.subplots(figsize=(7, 6))

levels = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
cs = ax.contourf(Neff_vals, kappa_vals, ERR, levels=levels, cmap='YlOrRd',
                 norm=matplotlib.colors.LogNorm())
plt.colorbar(cs, ax=ax, label='Relative error (%)', shrink=0.8)

cs2 = ax.contour(Neff_vals, kappa_vals, ERR, levels=[0.1, 0.5, 1.0, 5.0, 10.0],
                 colors='black', linewidths=0.8)
ax.clabel(cs2, fmt={0.1:'0.1%',0.5:'0.5%',1.0:'1%',5.0:'5%',10.0:'10%'}, fontsize=8)

ax.axvline(x=100, color='blue', linestyle='--', linewidth=1.5, label=r'$N_{\mathrm{eff}}=100$')
ax.axvline(x=50, color='blue', linestyle=':', linewidth=1, alpha=0.5, label=r'$N_{\mathrm{eff}}=50$')

ax.set_yscale('log')
ax.set_xlabel(r'$N_{\mathrm{eff}} \equiv \min(M, \Omega)$')
ax.set_ylabel(r'$\kappa$')
ax.set_title('Fig. S1c.1. S1c.8 (exact) vs S1c.12 (master eq): relative error in ξ\n'
             r'$M=\Omega=N_{\mathrm{eff}}$, $\Delta = \mathrm{Var}(C)/[\kappa\Omega^2(1-p)]$',fontsize=15,fontweight='bold', y=1)
ax.legend(fontsize=15, loc='upper right')
ax.set_xlim(3, 200)

plt.tight_layout()
# 先保存，再显示（Spyder 绘图窗口会自动弹出）
plt.savefig('S1c.1.svg', dpi=300, bbox_inches='tight')
plt.show()          # 在 Spyder 里通常直接就能看到，不加也行

# ----- 输出表格 -----
print("\n" + "="*80)
print("Relative error (%) at key (N_eff, kappa) points")
header = f"{'N_eff':>6} | {'k=0.01':>8} {'k=0.1':>8} {'k=0.5':>8} {'k=1':>8} {'k=5':>8} {'k=100':>8}"
print(header)
print("-" * len(header))
for neff in [5,10,20,50,100,200]:
    row = f"{neff:6d} |"
    for kap in [0.01,0.1,0.5,1.0,5.0,100.0]:
        err = compute_rel_err(neff, neff, kap)
        row += f" {err:7.3f}%"
    print(row)