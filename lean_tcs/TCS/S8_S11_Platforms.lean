import Mathlib

/-!
# Batches 6+7 (worker4), File 2: platform reductions S8, S9, S11, S1a

Manuscript anchors: `si_refs/s89.txt` (S8-S9), `si_refs/s11.txt` (S11),
`si_refs/s1a.txt` (S1a).

Note (S11.2): the exact quadratic root (S11.2) is already formalised as
`master_quadratic` in `TCS/S1c_MasterEquation.lean`; it is not restated here.

Frozen statements from `si_refs/statements.md` (Batches 6+7), verbatim;
only the proofs are filled in. Auxiliary lemmas are private.
-/

noncomputable section

/-- **S8.4, Jensen bias of heterogeneity.** For 0 < p < 1 the map
x ↦ (1 − p)^x is convex on ℝ, so by Jensen's inequality over any weight
distribution, 𝔼[(1−p)^β] ≥ (1−p)^(𝔼β): neglecting bead-size heterogeneity
biases P_neg upward and hence underestimates M. -/
theorem s8_jensen {p : ℝ} (hp0 : 0 < p) (hp1 : p < 1) {ι : Type*} [Fintype ι]
    (w : ι → ℝ) (hw : ∀ i, 0 ≤ w i) (hws : ∑ i, w i = 1) (β : ι → ℝ) :
    (1 - p) ^ (∑ i, w i * β i) ≤ ∑ i, w i * (1 - p) ^ (β i) := by
  have := hp0
  have hc : (0 : ℝ) < 1 - p := by linarith
  have hconv : ConvexOn ℝ Set.univ (fun x : ℝ => (1 - p) ^ x) := by
    refine ⟨convex_univ, fun x _ y _ a b ha hb hab => ?_⟩
    simp only [smul_eq_mul]
    simp only [Real.rpow_def_of_pos hc]
    have hlin : Real.log (1 - p) * (a * x + b * y)
        = a * (Real.log (1 - p) * x) + b * (Real.log (1 - p) * y) := by ring
    rw [hlin]
    exact convexOn_exp.2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab
  have h := hconv.map_sum_le (t := Finset.univ) (w := w) (p := β)
    (fun i _ => hw i) hws (fun i _ => Set.mem_univ _)
  simpa [smul_eq_mul] using h

/-- **(S8.5), homogeneous (Dirac) limit.** When the heterogeneity distribution
collapses to a single unit type (Unique ι, weights summing to 1), (S8.1)
reduces to P_neg = (1 − p)^β₀ = 1 − P_specific, the homogeneous TCS
observation model. -/
theorem s8_dirac {ι : Type*} [Fintype ι] [Unique ι] (w : ι → ℝ)
    (hw : ∀ i, 0 ≤ w i) (hws : ∑ i, w i = 1) (β : ι → ℝ) (p : ℝ) :
    ∑ i, w i * (1 - p) ^ (β i) = (1 - p) ^ (β default) := by
  have := hw
  rw [Finset.univ_unique, Finset.sum_singleton] at hws ⊢
  rw [hws, one_mul]

/-- **(S8.2), Poisson-mixed readout (PoissonPlus integral, series form).**
Marginalising the binomial observation over Poisson(λ) unit sizes:
P_neg = e^{−λ}·Σ_k (λ(1−p))^k/k! = e^{−λ}·e^{λ(1−p)} = e^{−λp}. This is the
PoissonPlus volume-correction integral evaluated exactly. -/
theorem s8_poisson_zero {l p : ℝ} (hl : 0 ≤ l) :
    ∑' k : ℕ, (l ^ k / (k : ℕ).factorial) * Real.exp (-l) * (1 - p) ^ k
      = Real.exp (-l * p) := by
  have := hl
  have hexp : Real.exp (l * (1 - p))
      = ∑' k : ℕ, (l * (1 - p)) ^ k / (Nat.factorial k : ℝ) := by
    rw [Real.exp_eq_exp_ℝ, NormedSpace.exp_eq_tsum_div]
  have hs : ∀ k : ℕ, (l ^ k / (k : ℕ).factorial) * Real.exp (-l) * (1 - p) ^ k
      = Real.exp (-l) * ((l * (1 - p)) ^ k / (Nat.factorial k : ℝ)) := by
    intro k
    rw [mul_pow]
    ring
  rw [tsum_congr hs, tsum_mul_left, ← hexp, ← Real.exp_add]
  congr 1
  ring

/-- **(S9.3) equilibrium is (S9.4).** Setting dp/dτ = 0 in the kinetic
equation (S9.3) gives the multivalent TCS master equation
ξ = p/(1 − p)^n + p/κ. -/
theorem s9_equilibrium {n : ℕ} {p ξ κ : ℝ} (hp : p ≠ 1) (hκ : κ ≠ 0)
    (h : (1 - p) ^ n * (ξ - p / κ) - p = 0) :
    ξ = p / (1 - p) ^ n + p / κ := by
  have := hκ
  have hp' : (1 : ℝ) - p ≠ 0 := sub_ne_zero.mpr (Ne.symm hp)
  have h1 : (1 - p) ^ n ≠ 0 := pow_ne_zero n hp'
  have h2 : (1 - p) ^ n * (ξ * κ - p) = p * κ := by
    field_simp at h
    linear_combination h
  field_simp
  linear_combination h2

/-- **S9.1, n = 1 reduction.** At n = 1, (S9.4) is the monovalent master
equation ξ = p/(1 − p) + p/κ. -/
theorem s9_n1 {p κ : ℝ} :
    p / (1 - p) ^ 1 + p / κ = p / (1 - p) + p / κ := by
  rw [pow_one]

/-- **S9.3, PFO solution.** p(τ) = 1 − e^(−ξτ) solves the (n = 1,
irreversible, κ → ∞) corner (S9.6): dp/dτ = ξ(1 − p), p(0) = 0. Hence the PFO
rate constant k₁ = k_on·T_tot is a composite quantity. -/
theorem s9_pfo_solution {ξ : ℝ} (τ : ℝ) :
    HasDerivAt (fun τ : ℝ => 1 - Real.exp (-ξ * τ))
      (ξ * (1 - (1 - Real.exp (-ξ * τ)))) τ ∧
    (1 - Real.exp (-ξ * (0 : ℝ))) = 0 := by
  constructor
  · have h1 : HasDerivAt (fun τ : ℝ => -ξ * τ) (-ξ) τ := by
      simpa using (hasDerivAt_id τ).const_mul (-ξ)
    have h2 := h1.exp
    have h3 := h2.const_sub 1
    exact h3.congr_deriv (by ring)
  · simp

/-- **S9.4, PSO solution.** p(τ) = ξτ/(1 + ξτ) solves the (n = 2,
irreversible, κ → ∞) corner (S9.6): dp/dτ = ξ(1 − p)², p(0) = 0. Hence the PSO
rate law is the same corner, not evidence of a bimolecular surface reaction. -/
theorem s9_pso_solution {ξ : ℝ} (hξ : 0 < ξ) (τ : ℝ) (hτ : 0 ≤ τ) :
    HasDerivAt (fun τ : ℝ => ξ * τ / (1 + ξ * τ))
      (ξ * (1 - ξ * τ / (1 + ξ * τ)) ^ 2) τ ∧
    (ξ * (0 : ℝ) / (1 + ξ * (0 : ℝ))) = 0 := by
  have hden : (0 : ℝ) < 1 + ξ * τ := by
    have h0 : (0 : ℝ) ≤ ξ * τ := mul_nonneg hξ.le hτ
    linarith
  have hden' : (1 : ℝ) + ξ * τ ≠ 0 := ne_of_gt hden
  constructor
  · have hnum : HasDerivAt (fun τ : ℝ => ξ * τ) ξ τ := by
      simpa using (hasDerivAt_id τ).const_mul ξ
    have hd : HasDerivAt (fun τ : ℝ => 1 + ξ * τ) ξ τ := by
      simpa using hnum.const_add 1
    have hq := hnum.div hd hden'
    refine hq.congr_deriv ?_
    field_simp
    ring
  · simp

/-- **S9.4, PSO t/q linearity.** For the PSO solution, τ/p = τ + 1/ξ exactly:
the linearity of t/q(t) versus t is an exact geometric consequence of the
(n = 2, irreversible, κ → ∞) corner. -/
theorem s9_pso_linearity {ξ : ℝ} (hξ : 0 < ξ) {τ : ℝ} (hτ : 0 < τ) :
    τ / (ξ * τ / (1 + ξ * τ)) = τ + 1 / ξ := by
  have h1 : (1 : ℝ) + ξ * τ ≠ 0 := ne_of_gt (by positivity)
  have h2 : ξ * τ ≠ 0 := mul_ne_zero (ne_of_gt hξ) (ne_of_gt hτ)
  have h3 : ξ ≠ 0 := ne_of_gt hξ
  field_simp
  ring

/-- **(S11.4), Michaelis-Menten limit.** In the zero-depletion limit
p = ξ/(1 + ξ) with ξ = S₀/K_m, so v₀ = V_max·S₀/(K_m + S₀). -/
theorem s11_mm_limit {S₀ K_m : ℝ} (hK : 0 < K_m) (hS : 0 ≤ S₀) :
    (S₀ / K_m) / (1 + S₀ / K_m) = S₀ / (K_m + S₀) := by
  have hK' : K_m ≠ 0 := ne_of_gt hK
  have h1 : (1 : ℝ) + S₀ / K_m ≠ 0 := ne_of_gt (by positivity)
  have h2 : K_m + S₀ ≠ 0 := ne_of_gt (by positivity)
  field_simp

/- **(S11.2) note.** The exact quadratic root (S11.2) is already formalised as
`master_quadratic` in `TCS/S1c_MasterEquation.lean`; it is NOT restated here.
(No frozen statement in that block.) -/

/-- **S11.4, sequential substrate inhibition, elimination and normalisation.**
With mass action p₁/p₀ = x (x = [S]_free/K_d1) and p₂/p₁ = x·c
(c = K_d1/K_d2), the normalisation p₀ + p₁ + p₂ = 1 holds for
p₀ = 1/(1 + x + x²c), p₁ = x/(1 + x + x²c), p₂ = x²c/(1 + x + x²c), and the
mass-action ratios are satisfied. -/
theorem s11_substrate_inhibition {x c : ℝ} (hx : 0 < x) (hc : 0 < c) :
    let p₀ := 1 / (1 + x + x ^ 2 * c)
    let p₁ := x / (1 + x + x ^ 2 * c)
    let p₂ := x ^ 2 * c / (1 + x + x ^ 2 * c)
    p₀ + p₁ + p₂ = 1 ∧ p₁ / p₀ = x ∧ p₂ / p₁ = x * c := by
  have hden : (0 : ℝ) < 1 + x + x ^ 2 * c := by positivity
  have hne : (1 : ℝ) + x + x ^ 2 * c ≠ 0 := ne_of_gt hden
  have hx' : x ≠ 0 := ne_of_gt hx
  refine ⟨?_, ?_, ?_⟩
  · field_simp
  · field_simp
  · field_simp
    ring

/-- **(S11.7), Cleland equation.** In the zero-depletion limit
([S]_free = S₀) the sequential-inhibition occupancy gives the classical
Cleland substrate-inhibition equation v₀ = V_max·S₀/(K_m + S₀ + S₀²/K_i)
(with K_m = K_d1, K_i = K_d2). -/
theorem s11_cleland {S₀ K_m K_i : ℝ} (hK : 0 < K_m) (hKi : 0 < K_i) (hS : 0 < S₀) :
    (S₀ / K_m) / (1 + S₀ / K_m + (S₀ / K_m) ^ 2 * (K_m / K_i))
      = S₀ / (K_m + S₀ + S₀ ^ 2 / K_i) := by
  have hK' : K_m ≠ 0 := ne_of_gt hK
  have hKi' : K_i ≠ 0 := ne_of_gt hKi
  have h1 : (1 : ℝ) + S₀ / K_m + (S₀ / K_m) ^ 2 * (K_m / K_i) ≠ 0 :=
    ne_of_gt (by positivity)
  have h2 : K_m + S₀ + S₀ ^ 2 / K_i ≠ 0 := ne_of_gt (by positivity)
  field_simp
  ring

/-- **(S11.6), substrate conservation in dimensionless form.** Dividing
S₀ = [S]_free + E_T·p₁ + 2·E_T·p₂ by K_d1 gives
ξ₁ = p₁/p₀ + (p₁ + 2p₂)/κ_R,1 (using [S]_free/K_d1 = p₁/p₀ and
κ_R,1 = K_d1/E_T); the factor 2 reflects the sequential ES₂ stoichiometry. -/
theorem s11_conservation {S₀ S_free E_T K_d1 p₁ p₂ : ℝ}
    (hE : 0 < E_T) (hK : K_d1 ≠ 0)
    (h : S₀ = S_free + E_T * p₁ + 2 * E_T * p₂) :
    S₀ / K_d1 = S_free / K_d1 + (p₁ + 2 * p₂) / (K_d1 / E_T) := by
  have hE' : E_T ≠ 0 := ne_of_gt hE
  rw [h]
  field_simp
  ring

/-- **S1a.1, free-pool identity.** With [AgAb₂]_free = [Ag]_free·[Ab₂]₀/K_D2
(mass action at [Ab₂]₀ ≫ [Ag]₀), the solution-phase pool is
C_free = [Ag]_free·([Ab₂]₀ + K_D2)/K_D2. -/
theorem s1a_free_pool {ag ab20 K_d2 : ℝ} (hK : K_d2 ≠ 0) :
    ag + ag * ab20 / K_d2 = ag * (ab20 + K_d2) / K_d2 := by
  field_simp
  ring

/-- **S1a.1, normalisation.** Dividing the sandwich probability by p_Ab₂
yields the Langmuir form in C_free alone: all detection-antibody parameters
([Ab₂]₀, K_D2, steric hindrance) cancel in the normalisation. -/
theorem s1a_normalisation {C_free K_d1 p_ab2 : ℝ} (h2 : p_ab2 ≠ 0) :
    (C_free / (C_free + K_d1) * p_ab2) / p_ab2 = C_free / (C_free + K_d1) := by
  exact mul_div_cancel_right₀ _ h2

/-- **S1a.2(i), harmonic-mean limit.** For polyclonal capture with clone
weights w and constants K_i: ⟨p⟩/C = Σ w_i/(C + K_i) → Σ w_i/K_i as C → 0⁺.
At low concentration the effective affinity is the harmonic mean 1/⟨1/K_D1⟩,
the LoD-relevant regime. -/
theorem s1a_harmonic_limit {ι : Type*} [Fintype ι] (w K : ι → ℝ)
    (hK : ∀ i, 0 < K i) :
    Filter.Tendsto (fun C : ℝ => ∑ i, w i / (C + K i))
      (nhdsWithin 0 (Set.Ioi 0)) (nhds (∑ i, w i / K i)) := by
  apply tendsto_finset_sum
  intro i _
  have h1 : Filter.Tendsto (fun C : ℝ => C + K i) (nhdsWithin 0 (Set.Ioi 0))
      (nhds (0 + K i)) :=
    Filter.Tendsto.mono_left (Filter.tendsto_id.add_const (K i)) nhdsWithin_le_nhds
  rw [zero_add] at h1
  exact tendsto_const_nhds.div h1 (ne_of_gt (hK i))

/-- **S1a.2(i), arithmetic-mean limit.** With ⟨p⟩(C) = Σ w_i·C/(C + K_i) and
K_eff defined by ⟨p⟩ = C/(C + K_eff), equivalently K_eff(C) = C(1 − ⟨p⟩)/⟨p⟩:
at high concentration C(1 − ⟨p⟩) → Σ w_iK_i and ⟨p⟩ → 1 = Σ w_i, so
K_eff(C) → ⟨K_D1⟩: the effective affinity is the arithmetic mean. -/
theorem s1a_arithmetic_limit {ι : Type*} [Fintype ι] (w K : ι → ℝ)
    (hK : ∀ i, 0 < K i) (hws : ∑ i, w i = 1) :
    Filter.Tendsto
      (fun C : ℝ => C * (1 - ∑ i, w i * (C / (C + K i))) /
        (∑ i, w i * (C / (C + K i))))
      Filter.atTop (nhds (∑ i, w i * K i)) := by
  have hterm : ∀ i, Filter.Tendsto (fun C : ℝ => C / (C + K i)) Filter.atTop (nhds 1) := by
    intro i
    have hKi : (0 : ℝ) < K i := hK i
    have hK0 : Filter.Tendsto (fun C : ℝ => K i / C) Filter.atTop (nhds 0) :=
      tendsto_const_nhds.div_atTop Filter.tendsto_id
    have h1 : Filter.Tendsto (fun C : ℝ => 1 + K i / C) Filter.atTop (nhds (1 + 0)) :=
      tendsto_const_nhds.add hK0
    rw [add_zero] at h1
    have h2 : Filter.Tendsto (fun C : ℝ => 1 / (1 + K i / C)) Filter.atTop (nhds (1 / 1)) :=
      tendsto_const_nhds.div h1 one_ne_zero
    rw [div_one] at h2
    refine Filter.Tendsto.congr' ?_ h2
    filter_upwards [Filter.eventually_gt_atTop 0] with C hC
    have hC' : C ≠ 0 := ne_of_gt hC
    have h3 : C + K i ≠ 0 := ne_of_gt (by positivity)
    have h4 : (1 : ℝ) + K i / C ≠ 0 := ne_of_gt (by positivity)
    field_simp
  have hp_lim : Filter.Tendsto (fun C : ℝ => ∑ i, w i * (C / (C + K i)))
      Filter.atTop (nhds 1) := by
    have h := tendsto_finset_sum Finset.univ (fun i _ => (hterm i).const_mul (w i))
    simpa [hws] using h
  have hid : (fun C : ℝ => ∑ i, w i * K i * (C / (C + K i)))
      =ᶠ[Filter.atTop] (fun C : ℝ => C * (1 - ∑ i, w i * (C / (C + K i)))) := by
    filter_upwards [Filter.eventually_gt_atTop 0] with C hC
    have hCi : ∀ i, C + K i ≠ 0 := fun i => ne_of_gt (add_pos hC (hK i))
    have step1 : 1 - ∑ i, w i * (C / (C + K i)) = ∑ i, w i * (K i / (C + K i)) := by
      have h1 : ∀ i, 1 - C / (C + K i) = K i / (C + K i) := fun i => by
        have h2 := hCi i
        field_simp
      calc 1 - ∑ i, w i * (C / (C + K i))
          = ∑ i, w i - ∑ i, w i * (C / (C + K i)) := by rw [hws]
        _ = ∑ i, (w i - w i * (C / (C + K i))) := by rw [Finset.sum_sub_distrib]
        _ = ∑ i, w i * (K i / (C + K i)) := by
          apply Finset.sum_congr rfl
          intro i _
          rw [← h1 i]
          ring
    rw [step1, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    ring
  have hq_lim : Filter.Tendsto (fun C : ℝ => C * (1 - ∑ i, w i * (C / (C + K i))))
      Filter.atTop (nhds (∑ i, w i * K i)) := by
    have h := tendsto_finset_sum Finset.univ (fun i _ => (hterm i).const_mul (w i * K i))
    simp only [mul_one] at h
    exact Filter.Tendsto.congr' hid h
  have hdiv := hq_lim.div hp_lim one_ne_zero
  simpa using hdiv
