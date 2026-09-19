# How to Verify That These Proofs Are Really Our Math

(A self-check guide for co-authors who do not read Lean code)

## 1. The trust model: you do not have to trust anyone

A Lean development has two parts:

- **The statement** (`theorem ... : <equation> :=`) — what is being
  claimed. This is ordinary mathematics written with keyboard symbols.
  Checking that it matches the manuscript is a *reading* task, not a
  programming task.
- **The proof** (`by ...`) — why it is true. This part you can skip
  entirely: the Lean kernel re-checks every proof step mechanically.
  If `lake build` prints `Build completed successfully`, the proofs are
  correct with 100% certainty — that is the whole point of formal
  verification. No reviewer, including us, needs to read them.

So the only question you need to answer is: **"Are the statements our
equations?"** Section 4 below lists every statement in plain
mathematical notation next to its manuscript anchor. If each line
matches what is written in the SI, then the compiled artifact *is* our
theory, machine-verified.

## 2. How to read a statement: a 30-second anatomy

Take `theorem_S2e5` in `TCS/S1c_MasterEquation.lean`:

```lean
theorem theorem_S2e5 {κv ξv : ℝ} (hκ : 0 < κv) (hξ : 0 < ξv) :
    ∃! p : ℝ, p ∈ Set.Ioo (0:ℝ) 1 ∧ ξv = p / (1 - p) + p / κv := by
```

Reading guide:

| Lean fragment | Mathematical meaning |
|---|---|
| `{κv ξv : ℝ}` | κ and ξ are real numbers |
| `(hκ : 0 < κv) (hξ : 0 < ξv)` | assume κ > 0 and ξ > 0 |
| `∃! p : ℝ` | there exists a **unique** real p |
| `p ∈ Set.Ioo (0:ℝ) 1` | p lies in the open interval (0, 1) |
| `ξv = p / (1 - p) + p / κv` | ξ = p/(1−p) + p/κ — the master equation |

So this line says: *for κ > 0 and ξ > 0 there is a unique occupancy
p ∈ (0,1) satisfying the master equation (S2e.4).* Note the anchor
subtlety: in the SI, Theorem S2e.5 is the *derivation* of the master
equation itself (covered by `master_dimensionless`, see the table
below); `theorem_S2e5` additionally proves the equation has exactly
one physical solution, a fact the SI uses throughout without a
separate number. Everything after `:= by` is the proof; ignore it.

## 3. Symbol table (manuscript ↔ Lean)

| Manuscript | Lean name | Notes |
|---|---|---|
| p (occupancy / bound fraction) | `p` | |
| M (total ligand molecules) | `M` | |
| Ω (receptor/capture-site molecules) | `Ω` | |
| K (dissociation constant, molar) | `K` | |
| W = V·N_A (molarity→count conversion) | `W` | kept symbolic |
| κ ≡ K·W/Ω | `kappa`, or `κv` when quantified | Definition S2e.2 |
| ξ ≡ M/(K·W) | `xi`, or `ξv` when quantified | Definition S2e.3 |
| C_free (free ligand concentration) | `C` | |
| s (common scaling factor) | `s` | |
| b (background false-positive rate) | `b` | S2b |
| μ (mean molecular loading per partition) | `μ` | S2b |
| P_pos (positive-partition probability) | `P_pos b κ μ` | S2b |
| n, k (partition count, occupied count) | `n : ℕ`, `k : ℕ` | S2a |
| C(n,k) (binomial coefficient) | `Nat.choose n k` | |
| E[·], Var[·] | explicit sums `∑ k ∈ range (n+1), ...` | S2a |

## 4. Every statement, in plain math, with its anchor

### S1c_MasterEquation.lean — the master equation chain

| Theorem | Plain-math statement | Manuscript anchor |
|---|---|---|
| `theorem_S2e1` | If p/(1−p) = C/K (Langmuir) and M = C·W + Ω·p (mass conservation), then M = Ω·p + K·W·p/(1−p). | Theorem S2e.1 (Axioms S2e.1 + S2e.2) |
| `master_dimensionless` | From the master equation: M/(K·W) = p/(1−p) + p·Ω/(K·W). | **Theorem S2e.5**, Eq. (S2e.4) / (S1c.12) |
| `kappa_form` | p/(K·W/Ω) = p·Ω/(K·W). | Definition S2e.2 bridge |
| `finite_omega_correction` | If M = (Ω−1)·p + K·W·p/(1−p), then ξ = p/(1−p) + (p/κ)·(1 − 1/Ω). | Finite-Ω remark in S1c/S2e |
| `master_quadratic` | ξ = p/(1−p) + p/κ ⟺ p² − (κξ+κ+1)p + κξ = 0 (provided p≠1, κ≠0). | Quadratic form in S2e |
| `master_strictMono` | For κ>0, f(p) = p/(1−p) + p/κ is strictly increasing on (0,1). | Monotonicity lemma behind the uniqueness claim below |
| `theorem_S2e5` | For κ>0, ξ>0: a unique p∈(0,1) satisfies the master equation. | Existence and uniqueness of the solution of Eq. (S2e.4); used implicitly throughout the SI, not separately numbered there (SI Theorem S2e.5 itself is `master_dimensionless` above) |

### S1d_ScaleGroup.lean — the scale group and (ξ, κ) orthogonality

| Theorem | Plain-math statement | Manuscript anchor |
|---|---|---|
| `kappa`, `xi` | κ ≡ K·W/Ω and ξ ≡ M/(K·W) (definitions). | Definitions S2e.2, S2e.3 |
| `kappa_scale` | κ(sK, W, sΩ) = κ(K, W, Ω). | Theorem S2e.6 (i) |
| `xi_scale` | ξ(sM, sK, W) = ξ(M, K, W). | Theorem S2e.6 (ii) |
| `coating_kappa` | Coating (M fixed, K,Ω ×s): κ unchanged. | Table S1d.1, coating row |
| `coating_xi` | Coating: ξ → ξ/s. | Table S1d.1, coating row |
| `dilution_xi` | Dilution (Ω fixed, M,K ×s): ξ unchanged. | Table S1d.1, dilution row |
| `dilution_kappa` | Dilution: κ → s·κ. | Table S1d.1, dilution row |
| `dilution_coupled` | Under dilution, BOTH M/Ω and κ are multiplied by s. | Table S1d.2 (counterexample) |
| `separation` | If two systems (M₁,Ω₁,K₁), (M₂,Ω₂,K₂) with M₁,M₂>0 give the same ξ and the same κ, then ∃ s>0 with M₂=sM₁, Ω₂=sΩ₁, K₂=sK₁. | Theorem S1d.3 (algebraic core) |

### S2e_ScaleDegeneracy.lean — degeneracy and incomparability

| Theorem | Plain-math statement | Manuscript anchor |
|---|---|---|
| `theorem_S2e2` | If (M,Ω,K) satisfies the master equation at occupancy p, so does (sM,sΩ,sK) at the same p, for every s. | Theorem S2e.2 |
| `theorem_S2e3` | The whole continuum {sM,sΩ,sK}_{s>0} shares the same readout p. | Theorem S2e.3 |
| `M_not_identifiable` | If M≠0 and s≠1 then s·M ≠ M (the rescaled system is genuinely different, yet indistinguishable by S2e.2). | "M is not identifiable" remark |
| `theorem_S2e4_core` | If 0<p<1 and κ₁≠κ₂ (both ≠0), then the ξ₁, ξ₂ satisfying the master equation with the same p obey ξ₁≠ξ₂. | Theorem S2e.4 (algebraic core) |
| `only_combination` | From ξ = p/(1−p) + p/κ it follows that ξ − p/κ = p/(1−p): only this combination is observable. | "Only knowable combination" remark |

### S2b_DigitalStatistics.lean — the digital platform

| Theorem | Plain-math statement | Manuscript anchor |
|---|---|---|
| `P_pos` (definition) | P_pos(b,κ,μ) = b + (1−b)·(1 − e^{−μ/(1+κ)}). | S2b main equation |
| `poisson_limit` | P_pos(b,0,μ) = b + (1−b)·(1 − e^{−μ}). | Poisson (dPCR) limit |
| `correction_factor` | P_pos(b,κ,μ) = b + (1−b)·(1 − e^{−γ·μ}) with γ = 1/(1+κ). | Correction-factor form |
| `estimator_inversion` | If P = P_pos(b,κ,μ) with b<1, κ>−1, b≤P<1, then μ = −(1+κ)·ln(1 − (P−b)/(1−b)). | S2b estimator inversion |
| `estimator_dPCR_limit` | For b=0, κ=0: every 0≤P<1 equals P_pos(0,0,μ) for μ = −ln(1−P). | Classical dPCR as κ=0 case |

### S2a_AnalogStatistics.lean — sampling statistics

| Theorem | Plain-math statement | Manuscript anchor |
|---|---|---|
| `binomial_total` | Σₖ C(n,k) pᵏ(1−p)ⁿ⁻ᵏ = 1. | S2a (normalization) |
| `binomial_mean` | E[k] = n·p. | S2a |
| `binomial_factorial_moment` | E[k(k−1)] = n(n−1)p². | S2a |
| `binomial_variance` | Var[k] = E[(k−np)²] = n·p·(1−p). | S2a |
| `binomial_cv_sq` | (n p (1−p)) / (n p)² = (1−p)/(n p), i.e. CV² = (1−p)/(np). | S2a |
| `fano_identity` | ((1−p)/(np)) · (np/(1−p)) = 1. | Fano-type identity, S2a |

### S2d_Asymptotics.lean — limiting regimes

| Theorem | Plain-math statement | Manuscript anchor |
|---|---|---|
| `tendsto_langmuir` | Fix ξ>0. If p(κ)∈(0,1) satisfies the master equation for every κ>0, then p(κ) → ξ/(1+ξ) as κ → ∞. | κ→∞ Langmuir limit, S2d |
| `tendsto_zero_depletion` | Fix ξ>0. Then p(κ) → 0 as κ → 0⁺ (with the pointwise bound 0 < p(κ) ≤ κξ). | Strong-depletion limit, S2d |
| `occupancy_lower_bound` | Fix κ>0. If ξ > 1/κ then p ≥ (ξ − 1/κ)/(1 + ξ − 1/κ) (which → 1 as ξ → ∞). | Full-occupancy bound, S2d |

The five sections below were added in the full-coverage extension: every
numbered theorem of the SI and the platform reductions, 76 statements in
all. The description column quotes each theorem's own docstring first
line, which carries its SI anchor verbatim.

### S2b_Lemmas.lean

| Theorem | Statement (from its docstring) |
|---|---|
| `lemma2_choose_identity` | Lemma 2 (SI S2b), choose identity. |
| `lemma1_binomial_subset` | Lemma 1 (SI S2b), binomial subset sum, weight-level form. |

### S2c_Fisher.lean

| Theorem | Statement (from its docstring) |
|---|---|
| `lagrange_identity` | Lagrange identity (named in the proof of Theorem S2c.5.5.1): |
| `fisher_B_diff` | B-difference identity (step in the proof of Theorem S2c.5.5.1): |
| `fisher_det_identity` | Theorem S2c.5.5.1 (SI S2c.5.5), Fisher determinant identity. |
| `fisher_det_pos_iff` | Corollary (SI S2c.5.5), degeneracy condition. For nonnegative weights, |
| `sensitivity_ratio` | (S2c.5.5.5), sensitivity ratio. With the master-equation sensitivities |
| `fisher_det_langmuir_scaling` | Theorem S2c.5.5.2, Langmuir scaling (S2c.5.5.6). With occupancies held |
| `fisher_det_zero_limit` | Theorem S2c.5.5.2, strong-depletion limit (S2c.5.5.7), convergence. |
| `fisher_det_zero_pos` | (S2c.5.5.7), positivity of the strong-depletion limit. If 0 < s_l < 1, |
| `digitalReadout` | Digital positive-rate readout on a Type 1 platform (SI S2c.4): |
| `s2c41_factorization` | Theorem S2c.4.1, factorisation. In the strong-depletion linear regime |
| `s2c41_digital_degeneracy` | Theorem S2c.4.1, digital degeneracy at finite κ. The readout is |
| `s2c41_kappa_zero` | Theorem S2c.4.1, κ = 0 readability. At κ = 0 the readout is |

### S10_MasterEquation.lean

| Theorem | Statement (from its docstring) |
|---|---|
| `theorem_S10_2_master` | Theorem S10.2 (SI S10.2), ligand mass conservation, and Eq. (S10.3). |
| `S10_3_clark` | Theorem S10.3 (SI S10.3), Clark limit. At κ → ∞ the depletion term |
| `S10_4_ratio_identity` | Theorem S10.4 (SI S10.3), strong-depletion ratio, exact identity. |
| `S10_4_ratio` | Theorem S10.4 (SI S10.3), strong-depletion limit. As κ → 0⁺ at fixed |
| `hillSlope0` | Local Hill slope at half-saturation, Eq. (S10.8). |
| `S10_5_hill_tendsto_two` | Eq. (S10.8), lower limit. n_H⁽⁰⁾ → 2 as κ → 0⁺. |
| `S10_5_hill_tendsto_one` | Eq. (S10.8), upper limit. n_H⁽⁰⁾ → 1 as κ → ∞. |
| `S10_5_hill_bounds` | Eq. (S10.8), bounds. For κ > 0 the analytical bound satisfies |
| `S10_6_chengprusoff` | Theorem S10.6 (SI S10.5), Cheng-Prusoff. For competitive binding (S10.9) |
| `S10_effector_quadratic` | Eq. (S10.12) ⟺ Eq. (S10.13). The effector equation |
| `S10_effector_root` | Eq. (S10.14), physical root. The smaller root of the effector quadratic |
| `S10_blackleff_q` | Black-Leff limit (SI S10.6), receptor layer. q/(1 − q) = τ·p_R |
| `S10_blackleff_emax` | Eq. (S10.15), Black-Leff operational model. Substituting the Clark |
| `S10_8_reserve_identity` | Theorem S10.8 (SI S10.6), receptor reserve, exact ratio identity. |
| `S10_8_reserve_bound` | Theorem S10.8 (SI S10.6), receptor reserve, quantitative bound. |
| `TcsSkeleton` | Theorem S10.10 (SI S10.7), unification skeleton. TCS-MWC and TCS-KNF |
| `TcsSkeleton` | The skeleton map ξ(x) = x + p(x)/κ. |
| `S10_10_mwc_skeleton` | Theorem S10.10, MWC instance. Any MWC occupancy function yields the |
| `S10_10_knf_skeleton` | Theorem S10.10, KNF instance. Same skeleton for KNF (sequential |
| `signalETCM` | The eTCM fractional signal, Eq. (S10.18). |
| `S10_11_basal` | Theorem S10.11 (SI S10.8), constitutive activity. With no ligand and no |
| `S10_11_alpha` | Theorem S10.11 (SI S10.8), inverse agonism. For x > 0, y = 0 the signal |
| `S10_12_depleted` | Theorem S10.12 (SI S10.8), G-protein depletion reduces signal. |
| `S10_13_degeneracy` | Theorem S10.13 (SI S10.9), scale degeneracy. The scaling (S10.19) |
| `S10_14_scale` | Theorem S10.14 (SI S10.11), scale invariance. Under |
| `S10_14_independent` | Eq. (S10.20) with binomial coefficients, limit (i). For n independent |
| `S10_14_zero_depletion` | Theorem S10.14, limit (ii), zero depletion. If ν stays bounded in [0, n] |
| `S10_14_strong_depletion` | Theorem S10.14, limit (iii), strong depletion. At fixed total input |

### S3_S7_Platforms.lean

| Theorem | Statement (from its docstring) |
|---|---|
| `s3_xi50` | (S3.4), half-saturation loading. Substituting p = 1/2 into the master |
| `s3_langmuir_bridge` | (S3.5), Langmuir bridge. In the zero-depletion limit the master equation |
| `s3_4pl_reduction` | (S3.6), 4PL occupancy factor. The 4PL occupancy factor |
| `s3_5pl_inverse` | (S3.7)/(S3.8), 5PL inverse. The Richards function |
| `s3_constraint1_linearity` | Constraint 1 (S3.9a), low-concentration linearity. As ξ → 0⁺ the 5PL |
| `s3_constraint2_halfsat` | Constraint 2 (S3.9b), half-saturation matching. Substituting p = 1/2, |
| `s3_constraint3_slope_deriv` | Constraint 3 (S3.9c), 5PL slope at half-saturation. With G = 1/B the |
| `s3_constraint3_limit` | (S3.9c), limit. (4κ+1)/(2κ) = 2 + 1/(2κ) → 2 as κ → ∞; by continuity |
| `s4_master_a` | (S4.1)/(S4.2), AAI correspondence. With a = 1/κ the TCS master equation |
| `s4_sensitivity` | (S4.4), AAI sensitivity. Differentiating (S4.2) implicitly: |
| `s4_aai_limit` | (S4.3), AAI limit. In the AAI limit (a → 0, equivalently κ → ∞) the |
| `master_solved_M` | Mass form of the master equation (used across S5/S6/S7): |
| `s5_gamma_identity` | S5, PICO ratio identity (γ form, cf. S2c.3.2). From the mass form, |
| `s5_gamma_limit` | S5, dilute-regime limit (Table S5.1 header, γ = 1/(1+κ)). In the PICO |
| `s5_error` | S5, quantification error (Table S5.1). The finite-κ relative error is |
| `s6_ct` | S6, threshold-cycle identity. Ideal doubling A_t = M·2^t with threshold |
| `s7_digitisa_conservation` | S7.2, DigitISA mass conservation. Full conservation reads |
| `s7_digitisa_share` | S7.2, DigitISA omitted-term share. The omitted term is the fraction |

### S8_S11_Platforms.lean

| Theorem | Statement (from its docstring) |
|---|---|
| `s8_jensen` | S8.4, Jensen bias of heterogeneity. For 0 < p < 1 the map |
| `s8_dirac` | (S8.5), homogeneous (Dirac) limit. When the heterogeneity distribution |
| `s8_poisson_zero` | (S8.2), Poisson-mixed readout (PoissonPlus integral, series form). |
| `s9_equilibrium` | (S9.3) equilibrium is (S9.4). Setting dp/dτ = 0 in the kinetic |
| `s9_n1` | S9.1, n = 1 reduction. At n = 1, (S9.4) is the monovalent master |
| `s9_pfo_solution` | S9.3, PFO solution. p(τ) = 1 − e^(−ξτ) solves the (n = 1, |
| `s9_pso_solution` | S9.4, PSO solution. p(τ) = ξτ/(1 + ξτ) solves the (n = 2, |
| `s9_pso_linearity` | S9.4, PSO t/q linearity. For the PSO solution, τ/p = τ + 1/ξ exactly: |
| `s11_mm_limit` | (S11.4), Michaelis-Menten limit. In the zero-depletion limit |
| `s11_substrate_inhibition` | S11.4, sequential substrate inhibition, elimination and normalisation. |
| `s11_cleland` | (S11.7), Cleland equation. In the zero-depletion limit |
| `s11_conservation` | (S11.6), substrate conservation in dimensionless form. Dividing |
| `s1a_free_pool` | S1a.1, free-pool identity. With [AgAb₂]_free = [Ag]_free·[Ab₂]₀/K_D2 |
| `s1a_normalisation` | S1a.1, normalisation. Dividing the sandwich probability by p_Ab₂ |
| `s1a_harmonic_limit` | S1a.2(i), harmonic-mean limit. For polyclonal capture with clone |
| `s1a_arithmetic_limit` | S1a.2(i), arithmetic-mean limit. With ⟨p⟩(C) = Σ w_i·C/(C + K_i) and |

## 5. The 5-minute "break it yourself" test

Do not take our word that the compiler checks anything. Break the code
and watch it fail:

1. Open `TCS/S2e_ScaleDegeneracy.lean`.
2. In `theorem_S2e2`, change the conclusion
   `s * M = (s * Ω) * p + (s * K) * W * (p / (1 - p))`
   to `s * M = (s * Ω) * p + (s * K) * W * (p / (1 + p))`
   (flip one minus to a plus — i.e., claim a wrong equation).
3. Run `lake build` (or just wait for VS Code to re-check).

Expected result: the build FAILS with an error at `theorem_S2e2`
(Lean reports that `ring` cannot close the goal — the "proof" no longer
proves the statement, because the statement is now false). Undo the
change and the build is green again.

Conclusion: the compiler is genuinely checking the mathematics, not
rubber-stamping it. A green build means the statements in Section 4 —
and only those statements — are theorems.

## 6. Check there is no cheating (`sorry`)

Lean has an escape hatch keyword, `sorry`, which means "trust me, skip
this proof". A development that used it would still compile (with a
warning). To rule this out, run in the project root:

```bash
grep -rn "sorry" TCS/
```

Expected result: **no matches**. The current build also reports 0
warnings, which independently confirms no `sorry` was used anywhere.
