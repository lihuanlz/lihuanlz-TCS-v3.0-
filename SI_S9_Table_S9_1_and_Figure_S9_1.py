# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 14:17:11 2026

@author: lihua
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import csv

# =========================================================
# 1. TCS 不可逆动力学方程 (n=2)
# =========================================================
def tcs_kinetics(tau, p, xi, kappa):
    """dp/dτ = (1-p)^2 * (ξ - p/κ)"""
    dpdtau = (1 - p)**2 * (xi - p / kappa)
    return dpdtau

# =========================================================
# 2. 生成 TCS 动力学数据
# =========================================================
def generate_TCS_curve(kappa, xi, C_site, tau_max, n_points=100):
    tau_eval = np.linspace(0, tau_max, n_points)
    sol = solve_ivp(tcs_kinetics, [0, tau_max], [0.0], t_eval=tau_eval,
                    args=(xi, kappa), method='RK45', rtol=1e-9)
    p = sol.y[0]
    q_tau = p * C_site
    return tau_eval, q_tau

# =========================================================
# 3. PSO 非线性模型
# =========================================================
def pso_nonlinear(t, qe, k):
    """qt = qe^2 * k * t / (1 + qe * k * t)"""
    return (qe**2 * k * t) / (1 + qe * k * t)

# =========================================================
# 4. PSO 线性拟合
# =========================================================
def fit_PSO_linear(tau, q_tau):
    mask = q_tau > 0
    t_clean = tau[mask]
    q_clean = q_tau[mask]
    y = t_clean / q_clean
    coeffs = np.polyfit(t_clean, y, 1)
    B = coeffs[0]
    A = coeffs[1]
    qe = 1.0 / B
    k = 1.0 / (A * qe**2)
    y_pred = A + B * t_clean
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot
    return qe, k, r2

# =========================================================
# 5. PSO 非线性拟合
# =========================================================
def fit_PSO_nonlinear(tau, q_tau):
    qe_guess = max(q_tau)
    k_guess = 0.1
    try:
        popt, pcov = curve_fit(pso_nonlinear, tau, q_tau,
                               p0=[qe_guess, k_guess], maxfev=10000)
        qe_fit, k_fit = popt
        q_pred = pso_nonlinear(tau, qe_fit, k_fit)
        ss_res = np.sum((q_tau - q_pred)**2)
        ss_tot = np.sum((q_tau - np.mean(q_tau))**2)
        r2 = 1 - ss_res / ss_tot
        return qe_fit, k_fit, r2
    except:
        return np.nan, np.nan, np.nan

# =========================================================
# 6. 主实验与绘图
# =========================================================
def run_experiment1():
    C_site_true = 100.0   # 真实 q_max
    xi = 10.0
    tau_max = 5.0
    kappa_values = [0.01, 0.1, 1.0, 10, 100]

    print("Experiment 1: Parameter recovery under known truth\n")
    print("True q_max = {:.1f}".format(C_site_true))
    header = "{:<10} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
        "κ", "Method", "Fitted q_e", "Bias %", "Fitted k", "R²")
    print(header)
    print("-" * 85)

    # 用于存储 CSV 数据
    csv_rows = [["κ", "Method", "Fitted q_e", "Bias %", "Fitted k", "R²"]]

    results = {}
    for kappa in kappa_values:
        tau, qt = generate_TCS_curve(kappa, xi, C_site_true, tau_max)
        qe_lin, k_lin, r2_lin = fit_PSO_linear(tau, qt)
        bias_lin = (qe_lin - C_site_true) / C_site_true * 100

        qe_nl, k_nl, r2_nl = fit_PSO_nonlinear(tau, qt)
        if not np.isnan(qe_nl):
            bias_nl = (qe_nl - C_site_true) / C_site_true * 100
        else:
            bias_nl = float('nan')

        print("{:<10} {:<15} {:<15.2f} {:<15.2f} {:<15.4f} {:<15.4f}".format(
            kappa, "Linear", qe_lin, bias_lin, k_lin, r2_lin))
        print("{:<10} {:<15} {:<15.2f} {:<15.2f} {:<15.4f} {:<15.4f}".format(
            "", "Nonlinear", qe_nl, bias_nl, k_nl, r2_nl))
        print()

        # 追加 CSV 行
        csv_rows.append([kappa, "Linear", round(qe_lin, 2), round(bias_lin, 2),
                         round(k_lin, 4), round(r2_lin, 4)])
        csv_rows.append([kappa, "Nonlinear", round(qe_nl, 2) if not np.isnan(qe_nl) else "NaN",
                         round(bias_nl, 2) if not np.isnan(bias_nl) else "NaN",
                         round(k_nl, 4) if not np.isnan(k_nl) else "NaN",
                         round(r2_nl, 4) if not np.isnan(r2_nl) else "NaN"])

        results[kappa] = {
            'tau': tau, 'qt': qt,
            'qe_lin': qe_lin, 'k_lin': k_lin,
            'qe_nl': qe_nl, 'k_nl': k_nl
        }

    # 保存 CSV 文件
    csv_filename = "experiment1_results.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"Results saved to {csv_filename}\n")

    # -------- 绘制四个 κ 的对比图 --------
    plot_kappas = [0.01, 0.1, 1.0, 10]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, kappa in enumerate(plot_kappas):
        ax = axes[i]
        data = results[kappa]
        tau = data['tau']
        qt = data['qt']
        qe_lin = data['qe_lin']
        k_lin = data['k_lin']
        qe_nl = data['qe_nl']
        k_nl = data['k_nl']

        ax.plot(tau, qt, 'ko', markersize=3, label='TCS data')

        q_lin_pred = pso_nonlinear(tau, qe_lin, k_lin)
        ss_res_lin = np.sum((qt - q_lin_pred)**2)
        ss_tot_lin = np.sum((qt - np.mean(qt))**2)
        r2_lin = 1 - ss_res_lin / ss_tot_lin

        q_nl_pred = pso_nonlinear(tau, qe_nl, k_nl)
        ss_res_nl = np.sum((qt - q_nl_pred)**2)
        r2_nl = 1 - ss_res_nl / ss_tot_lin

        tau_fine = np.linspace(0, tau_max, 200)
        ax.plot(tau_fine, pso_nonlinear(tau_fine, qe_lin, k_lin), 'r--',
                label=f'Linear: q_e={qe_lin:.1f}, R²={r2_lin:.4f}')
        ax.plot(tau_fine, pso_nonlinear(tau_fine, qe_nl, k_nl), 'b-',
                label=f'Nonlinear: q_e={qe_nl:.1f}, R²={r2_nl:.4f}')

        ax.set_xlabel('τ (dimensionless time)')
        ax.set_ylabel('q_τ')
        ax.set_title(f'κ = {kappa} (True q_max = 100)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Fig. S9.1. Parameter recovery under known truth: '
                 'linearised vs. nonlinear PSO fits to TCS data at '
                 'κ = 0.01, 0.1, 1.0, 10',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('ESI_S9_Table_S9_1_and_Figure_S9_1.svg', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_experiment1()