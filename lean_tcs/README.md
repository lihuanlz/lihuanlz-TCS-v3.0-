# Formal Verification of TCS (Lean 4 + Mathlib)

Machine-checked proofs of the mathematical core of the manuscript's
Supplementary Information ("Supplementary and additional Material"),
organized into 11 files, one per SI section or topic.

## 1. How to run

Case A: you already have a Lean 4 project with mathlib on your machine.
Copy the 11 `.lean` files inside `TCS/` into it (do NOT copy the lakefile
or the toolchain file, to avoid clashing with your local versions), open
them in VS Code, and check that no red squiggles appear.

Case B: build from scratch.
```bash
cd lean_tcs
lake exe cache get     # downloads the precompiled mathlib v4.15.0 cache (needs internet)
lake build
```
The toolchain is pinned to `leanprover/lean4:v4.15.0` and mathlib to
`v4.15.0` (see `lean-toolchain` and `lake-manifest.json`); elan installs
the matching Lean automatically.

## 2. Verification status

Full build from a clean state: **all 12 modules compile with 0 errors and
0 warnings** (`Build completed successfully.`), no `sorry` anywhere.
Every theorem below has been machine-checked, line by line, by the Lean
kernel. The suite contains 100+ checked declarations: the original core
chain plus 76 additional frozen statements covering every numbered
theorem of the SI and the platform reductions.

## 3. SI section ↔ file ↔ theorem map

| SI source | File | Theorem | Statement |
|---|---|---|---|
| Theorem S2e.1 | S1c_MasterEquation.lean | theorem_S2e1 | Eliminating C_free from the Langmuir relation and mass conservation gives M = Ωp + KW·p/(1−p) |
| Eq. (S2e.4)/(S1c.12) | S1c_MasterEquation.lean | master_dimensionless, kappa_form | Dimensionless master equation ξ = p/(1−p) + p/κ |
| Finite-Ω correction | S1c_MasterEquation.lean | finite_omega_correction | ξ = p/(1−p) + (p/κ)(1−1/Ω) |
| Quadratic form | S1c_MasterEquation.lean | master_quadratic | Master equation ⟺ p² − (κξ+κ+1)p + κξ = 0 |
| Monotonicity | S1c_MasterEquation.lean | master_strictMono | p/(1−p) + p/κ is strictly increasing on (0,1) |
| Theorem S2e.5 | S1c_MasterEquation.lean | theorem_S2e5 | For κ>0, ξ>0 a unique p∈(0,1) exists, the smaller quadratic root |
| Definitions S2e.2/S2e.3 | S1d_ScaleGroup.lean | kappa, xi | κ ≡ KW/Ω, ξ ≡ M/(KW) |
| Theorem S2e.6 | S1d_ScaleGroup.lean | kappa_scale, xi_scale | ξ and κ are invariant under common rescaling |
| Table S1d.1 | S1d_ScaleGroup.lean | coating_kappa, coating_xi, dilution_xi, dilution_kappa | Coating moves only ξ; dilution moves only κ (orthogonality) |
| Table S1d.2 | S1d_ScaleGroup.lean | dilution_coupled | Counterexample: under dilution the (M/Ω, κ) pair scales both coordinates at once |
| Theorem S1d.3 (algebraic core) | S1d_ScaleGroup.lean | separation | Same (ξ,κ) ⟹ the two systems differ only by a common scaling s>0 |
| Theorem S2e.2 | S2e_ScaleDegeneracy.lean | theorem_S2e2 | Scale invariance of the master equation |
| Theorem S2e.3 | S2e_ScaleDegeneracy.lean | theorem_S2e3 | A continuum of rescaled systems shares one readout |
| M not identifiable | S2e_ScaleDegeneracy.lean | M_not_identifiable | For s≠1, sM≠M, yet the readout is the same |
| Theorem S2e.4 (algebraic core) | S2e_ScaleDegeneracy.lean | theorem_S2e4_core | Same p but different κ ⟹ different ξ: raw readouts are not cross-platform comparable |
| Only knowable combination | S2e_ScaleDegeneracy.lean | only_combination | ξ − p/κ = p/(1−p) |
| S2b main equation | S2b_DigitalStatistics.lean | P_pos, poisson_limit, correction_factor | P_pos formula; κ=0 reduces to dPCR; γ=1/(1+κ) |
| S2b estimator | S2b_DigitalStatistics.lean | estimator_inversion, estimator_dPCR_limit | Legitimacy of the inversion μ̂ = −(1+κ)ln(1−(P−b)/(1−b)) |
| S2b Lemma 1, Lemma 2 | S2b_Lemmas.lean | lemma1_binomial_subset, lemma2_choose_identity | m-subset binomial weight identity; C(N,r)C(r,n) = C(N,n)C(N−n,r−n) |
| S2d asymptotics | S2d_Asymptotics.lean | tendsto_langmuir, tendsto_zero_depletion, occupancy_lower_bound | κ→∞ recovers Langmuir; κ→0⁺ gives p≤κξ→0; ξ→∞ drives p→1 (lower bound) |
| S2a binomial statistics | S2a_AnalogStatistics.lean | binomial_total/mean/factorial_moment/variance, binomial_cv_sq, fano_identity | E=np, Var=np(1−p), CV²=(1−p)/(np), CV²·np/(1−p)=1 |
| S2c.5.5 Fisher chain | S2c_Fisher.lean | lagrange_identity, fisher_B_diff, fisher_det_identity, fisher_det_pos_iff, sensitivity_ratio | det J = W²V via the Lagrange double-sum identity; positivity iff two B values differ; sensitivity ratio equals B |
| S2c.5.5.6/.7 scaling limits | S2c_Fisher.lean | fisher_det_langmuir_scaling, fisher_det_zero_limit, fisher_det_zero_pos | κ⁴·det J → C∞ > 0 as κ→∞; det J → positive limit as κ→0⁺ |
| S2c.4 digital readout | S2c_Fisher.lean | digitalReadout, s2c41_factorization, s2c41_digital_degeneracy, s2c41_kappa_zero | Theorem S2c.4.1: factorisation, finite-κ degeneracy, κ=0 readability |
| S10 master equation | S10_MasterEquation.lean | theorem_S10_2_master, S10_3_clark, S10_4_ratio_identity, S10_4_ratio | ξ = x + p/κ; Clark limit p = ξ/(1+ξ); exact ratio identity and its κ→0⁺ limit |
| S10.5 Hill slope | S10_MasterEquation.lean | hillSlope0, S10_5_hill_tendsto_two/one, S10_5_hill_bounds | Apparent Hill slope at half-saturation lies in (1,2): →2 as κ→0⁺, →1 as κ→∞ |
| S10.6/S10.7 pharmacology | S10_MasterEquation.lean | S10_6_chengprusoff, S10_effector_quadratic, S10_effector_root, S10_blackleff_q, S10_blackleff_emax | Cheng-Prusoff x_I = 1 + x_L; effector quadratic and its physical root; Black-Leff q and Emax |
| S10.8 receptor reserve | S10_MasterEquation.lean | S10_8_reserve_identity, S10_8_reserve_bound | q/(κτp_R) = (1−q)/(κ+1−q) and the resulting bound |
| S10.10 skeletons | S10_MasterEquation.lean | TcsSkeleton, xi, S10_10_mwc_skeleton, S10_10_knf_skeleton | MWC/KNF skeletons reduce to the master form (rfl) |
| S10.11-S10.13 eTCM | S10_MasterEquation.lean | signalETCM, S10_11_basal, S10_11_alpha, S10_12_depleted, S10_13_degeneracy | Basal signal, α-monotonicity, depleted limit, degeneracy |
| S10.14 scale limits | S10_MasterEquation.lean | S10_14_scale, S10_14_independent, S10_14_zero_depletion, S10_14_strong_depletion | Scale law; independent-site binomial limit n·x/(1+x); both depletion limits |
| S3 platform algebra | S3_S7_Platforms.lean | s3_xi50, s3_langmuir_bridge, s3_4pl_reduction, s3_5pl_inverse | ξ₅₀ = 1 + 1/(2κ); Langmuir bridge; 4PL reduction; 5PL inverse |
| S3 constraints | S3_S7_Platforms.lean | s3_constraint1_linearity, s3_constraint2_halfsat, s3_constraint3_slope_deriv/solve/limit | BG=1 linearity; half-saturation constraint C' = ξ₅₀(2^B−1)^{1/B}; slope matching and its κ→∞ limit |
| S4 | S3_S7_Platforms.lean | s4_master_a, s4_sensitivity, s4_aai_limit | Master reduction, sensitivity derivative, AAI limit |
| S5 | S3_S7_Platforms.lean | master_solved_M, s5_gamma_identity, s5_gamma_limit, s5_error | Solved-M form; γ identity and limit; error term |
| S6 qPCR | S3_S7_Platforms.lean | s6_ct | Ct logb law |
| S7 DIGISA | S3_S7_Platforms.lean | s7_digitisa_conservation, s7_digitisa_share, s7_strong | Conservation, share identity, strong-depletion form |
| S8 | S8_S11_Platforms.lean | s8_jensen, s8_dirac, s8_poisson_zero | Jensen bound for heterogeneous K_i; Dirac (homogeneous) case; Poisson zero term |
| S9 PFO/PSO kinetics | S8_S11_Platforms.lean | s9_equilibrium, s9_n1, s9_pfo_solution, s9_pso_solution, s9_pso_linearity | Equilibrium, n=1 case, verified PFO/PSO solutions (HasDerivAt + initial value), PSO linearity |
| S11 enzyme kinetics | S8_S11_Platforms.lean | s11_mm_limit, s11_substrate_inhibition, s11_cleland, s11_conservation | Michaelis-Menten limit; substrate inhibition joint limits; Cleland form; conservation |
| S1a | S8_S11_Platforms.lean | s1a_free_pool, s1a_normalisation, s1a_harmonic_limit, s1a_arithmetic_limit | Free-pool form; normalisation; harmonic-mean limit; arithmetic-mean limit |

## 4. What is NOT covered (and why)

1. The meta-theorem on necessary-and-sufficient conditions in S2c
   (universal quantification over identifiability model classes and
   "arbitrary measurements") is a meta-level statement; this suite
   formalizes its algebraic carrier (the full S2e chain plus the S2b
   estimator).
2. The four-layer stochastic model behind the Fano identity F₀+F₁=1
   (Poisson→Poisson→Poisson→Binomial mixed-moment computation) remains a
   large workload; the innermost binomial fluctuation, its identity, and
   the m-subset generalisation at PMF-weight level (S2b Lemma 1) are
   formalized here.
3. The partition-function derivation in S1c is a statistical-physics
   modelling assumption, not a purely mathematical theorem, so it is
   not formalized.

## 5. How to convince yourself (without reading any proof)

See `VERIFICATION.md`: every theorem statement is reproduced in plain
mathematical notation with its manuscript anchor, plus a 5-minute
"break one symbol and watch the compiler scream" test you can run
yourself.

## 6. Suggested use

Put this suite in the GitHub repository (e.g. under `formal/`), and add
one sentence to the cover letter or the SI: "The core theorems of the
Supplementary Information are machine-verified in Lean 4 (repo: ...)".
