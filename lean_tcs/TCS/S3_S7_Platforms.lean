import Mathlib

/-!
# Batches 6+7 (worker4), File 1: platform reductions S3, S4, S5, S6, S7

Manuscript anchors: `si_refs/s34567.txt` (S3-S7).

Frozen statements from `si_refs/statements.md` (Batches 6+7), verbatim;
only the proofs are filled in. Auxiliary lemmas are private.
-/

noncomputable section

/-- **(S3.4), half-saturation loading.** Substituting p = 1/2 into the master
equation ξ = p/(1 − p) + p/κ gives ξ₅₀ = 1 + 1/(2κ). -/
theorem s3_xi50 {κ : ℝ} (hκ : κ ≠ 0) :
    (1 / 2 : ℝ) / (1 - 1 / 2) + (1 / 2) / κ = 1 + 1 / (2 * κ) := by
  field_simp
  ring

/-- **(S3.5), Langmuir bridge.** In the zero-depletion limit the master equation
reduces to ξ = p/(1 − p), equivalently p = 1/(1 + ξ⁻¹), the Clark/Langmuir
form (and the 4PL occupancy factor). -/
theorem s3_langmuir_bridge {p ξ : ℝ} (hξ : 0 < ξ) (h : ξ = p / (1 - p)) :
    p = 1 / (1 + ξ⁻¹) := by
  have h1 : (1 : ℝ) - p ≠ 0 := by
    intro hc
    rw [hc, div_zero] at h
    exact (ne_of_gt hξ) h
  have h2 : (1 : ℝ) + ξ⁻¹ ≠ 0 := by
    have h3 : (0 : ℝ) < ξ⁻¹ := inv_pos.mpr hξ
    positivity
  have key : p * (ξ + 1) = ξ := by
    field_simp at h
    linear_combination -h
  rw [inv_eq_one_div]
  field_simp
  linear_combination key

/-- **(S3.6), 4PL occupancy factor.** The 4PL occupancy factor
1/(1 + (x/K)⁻¹) at B = 1 is exactly x/(K + x): the 4PL with C = K is the
Langmuir isotherm, exact in the zero-depletion limit. -/
theorem s3_4pl_reduction {x K : ℝ} (hx : 0 < x) (hK : 0 < K) :
    1 / (1 + (x / K)⁻¹) = x / (K + x) := by
  have := hK
  have hx' : x ≠ 0 := ne_of_gt hx
  have e1 : (1 : ℝ) + (x / K)⁻¹ = (x + K) / x := by
    rw [inv_div, add_div, div_self hx']
  rw [e1, div_div_eq_mul_div, one_mul, add_comm]

/-- **(S3.7)/(S3.8), 5PL inverse.** The Richards function
p = [1 + (ξ/C')^(−B)]^(−G) inverts to ξ = C'(p^(−1/G) − 1)^(−1/B). -/
theorem s3_5pl_inverse {B G C' ξ p : ℝ} (hB : 0 < B) (hG : 0 < G) (hC : 0 < C')
    (hξ : 0 < ξ) (hp : 0 < p) (hp1 : p < 1)
    (h : p = (1 + (ξ / C') ^ (-B : ℝ)) ^ (-G : ℝ)) :
    ξ = C' * (p ^ (-1 / G : ℝ) - 1) ^ (-1 / B : ℝ) := by
  have hB' : B ≠ 0 := ne_of_gt hB
  have hG' : G ≠ 0 := ne_of_gt hG
  have hC' : C' ≠ 0 := ne_of_gt hC
  have hξC : (0 : ℝ) < ξ / C' := by positivity
  have hu : (0 : ℝ) < (ξ / C') ^ (-B : ℝ) := Real.rpow_pos_of_pos hξC _
  have hv : (0 : ℝ) < 1 + (ξ / C') ^ (-B : ℝ) := by linarith
  have hp_eq : p ^ (-1 / G : ℝ) = 1 + (ξ / C') ^ (-B : ℝ) := by
    have hGG : (-G) * (-1 / G) = (1 : ℝ) := by field_simp
    rw [h, ← Real.rpow_mul hv.le, hGG, Real.rpow_one]
  have hpos : (0 : ℝ) < p ^ (-1 / G : ℝ) - 1 := by rw [hp_eq]; linarith
  have hu_eq : (p ^ (-1 / G : ℝ) - 1) ^ (-1 / B : ℝ) = ξ / C' := by
    have hbase : p ^ (-1 / G : ℝ) - 1 = (ξ / C') ^ (-B : ℝ) := by rw [hp_eq]; ring
    have hBB : (-B) * (-1 / B) = (1 : ℝ) := by field_simp
    rw [hbase, ← Real.rpow_mul hξC.le, hBB, Real.rpow_one]
  rw [hu_eq]
  field_simp

/-- **Constraint 1 (S3.9a), low-concentration linearity.** As ξ → 0⁺ the 5PL
response satisfies p/(ξ/C')^(B·G) → 1; hence p ∝ ξ at low concentration iff
B·G = 1 (then p ≈ (ξ/C') with unit prefactor; for BG ≠ 1 the response is
power-law, not linear). -/
theorem s3_constraint1_linearity {B G C' : ℝ} (hB : 0 < B) (hG : 0 < G) (hC : 0 < C') :
    Filter.Tendsto
      (fun ξ : ℝ => (1 + (ξ / C') ^ (-B : ℝ)) ^ (-G : ℝ) / (ξ / C') ^ (B * G : ℝ))
      (nhdsWithin 0 (Set.Ioi 0)) (nhds 1) := by
  have := hG
  -- u(ξ) = (ξ/C')^(-B) tends to +∞ as ξ → 0⁺
  have hbase : Filter.Tendsto (fun ξ : ℝ => (ξ / C') ^ (-B : ℝ))
      (nhdsWithin 0 (Set.Ioi 0)) Filter.atTop := by
    have h1 : Filter.Tendsto (fun ξ : ℝ => ξ / C')
        (nhdsWithin 0 (Set.Ioi 0)) (nhdsWithin 0 (Set.Ioi 0)) := by
      rw [tendsto_nhdsWithin_iff]
      constructor
      · have hlim : Filter.Tendsto (fun ξ : ℝ => ξ / C') (nhds 0) (nhds (0 / C')) :=
          Filter.tendsto_id.div_const C'
        rw [zero_div] at hlim
        exact hlim.mono_left nhdsWithin_le_nhds
      · filter_upwards [self_mem_nhdsWithin] with ξ hξ
        exact div_pos hξ hC
    have h2 : Filter.Tendsto (fun ξ : ℝ => (ξ / C')⁻¹)
        (nhdsWithin 0 (Set.Ioi 0)) Filter.atTop :=
      tendsto_inv_nhdsGT_zero.comp h1
    have h3 : Filter.Tendsto (fun ξ : ℝ => ((ξ / C')⁻¹) ^ B)
        (nhdsWithin 0 (Set.Ioi 0)) Filter.atTop :=
      (tendsto_rpow_atTop hB).comp h2
    refine Filter.Tendsto.congr' ?_ h3
    filter_upwards [self_mem_nhdsWithin] with ξ hξ
    have hb : (0 : ℝ) < ξ / C' := div_pos hξ hC
    rw [Real.rpow_neg hb.le, Real.inv_rpow hb.le]
  -- the ratio eventually equals ((1 + u)/u)^(-G)
  have hrw : (fun ξ : ℝ => (1 + (ξ / C') ^ (-B : ℝ)) ^ (-G : ℝ) / (ξ / C') ^ (B * G : ℝ))
      =ᶠ[nhdsWithin 0 (Set.Ioi 0)]
        (fun ξ : ℝ => ((1 + (ξ / C') ^ (-B : ℝ)) / (ξ / C') ^ (-B : ℝ)) ^ (-G : ℝ)) := by
    filter_upwards [self_mem_nhdsWithin] with ξ hξ
    have hb : (0 : ℝ) < ξ / C' := div_pos hξ hC
    have hu : (0 : ℝ) < (ξ / C') ^ (-B : ℝ) := Real.rpow_pos_of_pos hb _
    have h1 : (0 : ℝ) ≤ 1 + (ξ / C') ^ (-B : ℝ) := by linarith [hu.le]
    have h2 : (ξ / C') ^ (B * G : ℝ) = ((ξ / C') ^ (-B : ℝ)) ^ (-G : ℝ) := by
      rw [← Real.rpow_mul hb.le]
      congr 1
      ring
    rw [h2, ← Real.div_rpow h1 hu.le]
  -- ((1 + u)/u)^(-G) → 1 as u → ∞
  have hlim : Filter.Tendsto (fun u : ℝ => ((1 + u) / u) ^ (-G : ℝ))
      Filter.atTop (nhds 1) := by
    have h3 : Filter.Tendsto (fun u : ℝ => (1 + u) / u) Filter.atTop (nhds 1) := by
      have hrw2 : (fun u : ℝ => 1 + u⁻¹) =ᶠ[Filter.atTop] (fun u : ℝ => (1 + u) / u) := by
        filter_upwards [Filter.eventually_gt_atTop 0] with u hu
        have hu' : u ≠ 0 := ne_of_gt hu
        field_simp
        ring
      refine Filter.Tendsto.congr' hrw2 ?_
      have h4 : Filter.Tendsto (fun u : ℝ => 1 + u⁻¹) Filter.atTop (nhds (1 + 0)) :=
        tendsto_const_nhds.add tendsto_inv_atTop_zero
      simpa using h4
    have h5 : Filter.Tendsto (fun u : ℝ => ((1 + u) / u) ^ (-G : ℝ))
        Filter.atTop (nhds ((1 : ℝ) ^ (-G : ℝ))) :=
      (Real.continuousAt_rpow_const 1 (-G) (Or.inl one_ne_zero)).tendsto.comp h3
    simpa [Real.one_rpow] using h5
  exact Filter.Tendsto.congr' hrw.symm (hlim.comp hbase)

/-- **Constraint 2 (S3.9b), half-saturation matching.** Substituting p = 1/2,
ξ = ξ₅₀, G = 1/B into (S3.8) forces C' = ξ₅₀·(2^B − 1)^(1/B). -/
theorem s3_constraint2_halfsat {B C' ξ50 : ℝ} (hB : 0 < B) (hC : 0 < C') (hξ : 0 < ξ50)
    (h : ξ50 = C' * ((1 / 2 : ℝ) ^ (-B : ℝ) - 1) ^ (-1 / B : ℝ)) :
    C' = ξ50 * ((2 : ℝ) ^ B - 1) ^ (1 / B : ℝ) := by
  have hr : (1 : ℝ) < (2 : ℝ) ^ B := Real.one_lt_rpow one_lt_two hB
  have hpos : (0 : ℝ) < (2 : ℝ) ^ B - 1 := by linarith
  have h1 : (1 / 2 : ℝ) ^ (-B : ℝ) = (2 : ℝ) ^ B := by
    rw [show (1 / 2 : ℝ) = (2 : ℝ)⁻¹ by norm_num, Real.inv_rpow zero_le_two,
      Real.rpow_neg zero_le_two, inv_inv]
  rw [h1] at h
  have h2 : ((2 : ℝ) ^ B - 1) ^ (1 / B : ℝ) = (((2 : ℝ) ^ B - 1) ^ (-1 / B : ℝ))⁻¹ := by
    rw [← Real.rpow_neg hpos.le]
    congr 1
    ring
  rw [h2, h]
  have hX : ((2 : ℝ) ^ B - 1) ^ (-1 / B : ℝ) ≠ 0 := ne_of_gt (Real.rpow_pos_of_pos hpos _)
  have hC' : C' ≠ 0 := ne_of_gt hC
  have hξ' : ξ50 ≠ 0 := ne_of_gt hξ
  field_simp

/-- **Constraint 3 (S3.9c), 5PL slope at half-saturation.** With G = 1/B the
inverse 5PL is ξ(p) = C'(p^(−B) − 1)^(−1/B); substituting Constraint 2
(C' = ξ₅₀(2^B − 1)^(1/B)) and differentiating at p = 1/2 gives
dξ/dp = ξ₅₀·2^(B+1)/(2^B − 1). (C' is inlined via Constraint 2 in the
statement; the derivative computation is the content.) -/
theorem s3_constraint3_slope_deriv {B ξ50 : ℝ} (hB : 0 < B) (hξ : 0 < ξ50) :
    deriv (fun p : ℝ => ξ50 * ((2 : ℝ) ^ B - 1) ^ (1 / B : ℝ) *
        (p ^ (-B : ℝ) - 1) ^ (-1 / B : ℝ)) (1 / 2)
      = ξ50 * 2 ^ (B + 1) / ((2 : ℝ) ^ B - 1) := by
  have hξ' : ξ50 ≠ 0 := ne_of_gt hξ
  have hB' : B ≠ 0 := ne_of_gt hB
  have hr : (1 : ℝ) < (2 : ℝ) ^ B := Real.one_lt_rpow one_lt_two hB
  have hrpos : (0 : ℝ) < (2 : ℝ) ^ B - 1 := by linarith
  have hhalf : (1 / 2 : ℝ) ^ (-B : ℝ) = (2 : ℝ) ^ B := by
    rw [show (1 / 2 : ℝ) = (2 : ℝ)⁻¹ by norm_num, Real.inv_rpow zero_le_two,
      Real.rpow_neg zero_le_two, inv_inv]
  have hhalf2 : (1 / 2 : ℝ) ^ (-B - 1 : ℝ) = (2 : ℝ) ^ (B + 1) := by
    rw [show (1 / 2 : ℝ) = (2 : ℝ)⁻¹ by norm_num, Real.inv_rpow zero_le_two,
      show -B - 1 = -(B + 1 : ℝ) by ring, Real.rpow_neg zero_le_two, inv_inv]
  have h1 : HasDerivAt (fun p : ℝ => p ^ (-B : ℝ) - 1)
      (-B * (1 / 2 : ℝ) ^ (-B - 1 : ℝ)) (1 / 2) := by
    have hrp := Real.hasDerivAt_rpow_const (x := (1 : ℝ) / 2) (p := -B)
      (Or.inl (by norm_num))
    exact hrp.sub_const 1
  have h2 : HasDerivAt (fun u : ℝ => u ^ (-1 / B : ℝ))
      ((-1 / B) * ((1 / 2 : ℝ) ^ (-B : ℝ) - 1) ^ (-1 / B - 1 : ℝ))
      ((1 / 2 : ℝ) ^ (-B : ℝ) - 1) := by
    apply Real.hasDerivAt_rpow_const (Or.inl ?_)
    rw [hhalf]
    exact ne_of_gt hrpos
  have h3 := h2.comp (1 / 2) h1
  have h4 := h3.const_mul (ξ50 * ((2 : ℝ) ^ B - 1) ^ (1 / B : ℝ))
  have key : ((2 : ℝ) ^ B - 1) ^ (1 / B : ℝ) * ((2 : ℝ) ^ B - 1) ^ (-1 / B - 1 : ℝ)
      = ((2 : ℝ) ^ B - 1)⁻¹ := by
    rw [← Real.rpow_neg_one, ← Real.rpow_add hrpos]
    congr 1
    ring
  have h5 : HasDerivAt (fun p : ℝ => ξ50 * ((2 : ℝ) ^ B - 1) ^ (1 / B : ℝ) *
        (p ^ (-B : ℝ) - 1) ^ (-1 / B : ℝ))
      (ξ50 * 2 ^ (B + 1) / ((2 : ℝ) ^ B - 1)) (1 / 2 : ℝ) := by
    apply h4.congr_deriv
    have hRHS : ξ50 * 2 ^ (B + 1) / ((2 : ℝ) ^ B - 1)
        = ξ50 * 2 ^ (B + 1) * ((2 : ℝ) ^ B - 1)⁻¹ := by
      rw [div_eq_mul_inv]
    rw [hhalf, hhalf2, hRHS, ← key]
    field_simp
    ring
  exact h5.deriv

/-- **(S3.9c), solving for B.** Matching the 5PL half-saturation slope
ξ₅₀·2^(B+1)/(2^B − 1) (with ξ₅₀ = 1 + 1/(2κ) from S3.4) to the TCS slope
4 + 1/κ forces 2^B = (4κ + 1)/(2κ), i.e. B = log₂((4κ+1)/(2κ)). -/
theorem s3_constraint3_solve {κ B : ℝ} (hκ : 0 < κ)
    (h : (1 + 1 / (2 * κ)) * (2 : ℝ) ^ (B + 1) / ((2 : ℝ) ^ B - 1) = 4 + 1 / κ) :
    (2 : ℝ) ^ B = (4 * κ + 1) / (2 * κ) := by
  have hκ' : (κ : ℝ) ≠ 0 := ne_of_gt hκ
  set u := (2 : ℝ) ^ B with hu
  have hu1 : u ≠ 1 := by
    intro h1
    rw [h1, sub_self, div_zero] at h
    have hpos : (0 : ℝ) < 4 + 1 / κ := by positivity
    rw [← h] at hpos
    exact lt_irrefl _ hpos
  have hu1' : u - 1 ≠ 0 := sub_ne_zero.mpr hu1
  have h2 : (2 : ℝ) ^ (B + 1) = 2 * u := by
    rw [hu, Real.rpow_add (by norm_num : (0 : ℝ) < 2), Real.rpow_one]
    ring
  rw [h2] at h
  have h2k : (2 : ℝ) * κ ≠ 0 := by positivity
  have key2 : (2 * κ + 1) * u = (4 * κ + 1) * (u - 1) := by
    apply mul_right_cancel₀ h2k
    field_simp at h
    linear_combination h
  field_simp
  linear_combination -key2

/-- **(S3.9c), limit.** (4κ+1)/(2κ) = 2 + 1/(2κ) → 2 as κ → ∞; by continuity
of log₂, B(κ) → 1 and G(κ) = 1/B(κ) → 1 (the 5PL collapses to the 4PL). -/
theorem s3_constraint3_limit :
    Filter.Tendsto (fun κ : ℝ => (4 * κ + 1) / (2 * κ)) Filter.atTop (nhds 2) := by
  have h1 : Filter.Tendsto (fun κ : ℝ => (1 / 2) / κ) Filter.atTop (nhds 0) :=
    tendsto_const_nhds.div_atTop Filter.tendsto_id
  have h2 : Filter.Tendsto (fun κ : ℝ => 2 + (1 / 2) / κ) Filter.atTop (nhds (2 + 0)) :=
    tendsto_const_nhds.add h1
  rw [add_zero] at h2
  refine Filter.Tendsto.congr' ?_ h2
  filter_upwards [Filter.eventually_gt_atTop 0] with κ hκ
  have hκ' : κ ≠ 0 := ne_of_gt hκ
  field_simp
  ring

/-- **(S4.1)/(S4.2), AAI correspondence.** With a = 1/κ the TCS master equation
is the AAI isotherm ξ = p/(1 − p) + p·a. -/
theorem s4_master_a {p κ a : ℝ} (ha : a = 1 / κ) :
    p / (1 - p) + p * a = p / (1 - p) + p / κ := by
  rw [ha]
  ring

/-- **(S4.4), AAI sensitivity.** Differentiating (S4.2) implicitly:
dξ/dp = (1 − p)^(−2) + a, so the sensitivity is S = dp/dξ = 1/((1−p)^(−2) + a).
We compute the derivative of the inverse map directly. -/
theorem s4_sensitivity {a p : ℝ} (hp : p ≠ 1) :
    deriv (fun x : ℝ => x / (1 - x) + x * a) p = 1 / (1 - p) ^ 2 + a := by
  have h1 : (1 : ℝ) - p ≠ 0 := sub_ne_zero.mpr (Ne.symm hp)
  have hd1 : HasDerivAt (fun x : ℝ => x / (1 - x)) (1 / (1 - p) ^ 2) p := by
    have hden : HasDerivAt (fun x : ℝ => 1 - x) (-1) p := (hasDerivAt_id p).const_sub 1
    have hq := (hasDerivAt_id p).div hden h1
    refine hq.congr_deriv ?_
    field_simp
  have hd2 : HasDerivAt (fun x : ℝ => x * a) a p := by
    simpa using (hasDerivAt_id p).mul_const a
  exact (hd1.add hd2).deriv

/-- **(S4.3), AAI limit.** In the AAI limit (a → 0, equivalently κ → ∞) the
isotherm reduces to p = ξ/(1 + ξ) = C/(K + C): occupancy independent of the
antibody dose, the defining ambient-analyte property. -/
theorem s4_aai_limit {C K : ℝ} (hK : 0 < K) (hC : 0 ≤ C) :
    (C / K) / (1 + C / K) = C / (K + C) := by
  have hK' : K ≠ 0 := ne_of_gt hK
  have h1 : (1 : ℝ) + C / K ≠ 0 := ne_of_gt (by positivity)
  have h2 : K + C ≠ 0 := ne_of_gt (by positivity)
  field_simp

/-- **Mass form of the master equation** (used across S5/S6/S7):
ξ = M/(κΩ) with ξ = p/(1 − p) + p/κ rearranges to
M = Ω·p·(κ + 1 − p)/(1 − p) = Ωp(κ/(1−p) + 1). -/
theorem master_solved_M {κ Ω M p : ℝ} (hκ : 0 < κ) (hΩ : 0 < Ω) (hp : p ≠ 1) :
    M / (κ * Ω) = p / (1 - p) + p / κ ↔ M = Ω * p * (κ + 1 - p) / (1 - p) := by
  have hκ' : κ ≠ 0 := ne_of_gt hκ
  have hΩ' : Ω ≠ 0 := ne_of_gt hΩ
  have hp' : (1 : ℝ) - p ≠ 0 := sub_ne_zero.mpr (Ne.symm hp)
  constructor
  · intro h
    field_simp at h ⊢
    apply mul_right_cancel₀ hκ'
    linear_combination h
  · intro h
    field_simp at h ⊢
    linear_combination κ * h

/-- **S5, PICO ratio identity (γ form, cf. S2c.3.2).** From the mass form,
Ωp/M = (1 − p)/(κ + 1 − p) exactly. -/
theorem s5_gamma_identity {κ Ω M p : ℝ} (hκ : 0 < κ) (hΩ : 0 < Ω)
    (hp0 : 0 < p) (hp1 : p < 1)
    (h : M = Ω * p * (κ + 1 - p) / (1 - p)) :
    Ω * p / M = (1 - p) / (κ + 1 - p) := by
  have h1 : Ω * p ≠ 0 := mul_ne_zero (ne_of_gt hΩ) (ne_of_gt hp0)
  have h2 : κ + 1 - p ≠ 0 := ne_of_gt (by linarith)
  have h3 : (1 : ℝ) - p ≠ 0 := ne_of_gt (by linarith)
  rw [h]
  field_simp
  ring

/-- **S5, dilute-regime limit (Table S5.1 header, γ = 1/(1+κ)).** In the PICO
dilute regime p → 0: Ωp/M = (1 − p)/(κ + 1 − p) → 1/(1 + κ) = γ. -/
theorem s5_gamma_limit {κ : ℝ} (hκ : 0 < κ) :
    Filter.Tendsto (fun p : ℝ => (1 - p) / (κ + 1 - p)) (nhds 0) (nhds (1 / (1 + κ))) := by
  have hnum : Filter.Tendsto (fun p : ℝ => 1 - p) (nhds 0) (nhds (1 - 0)) :=
    tendsto_const_nhds.sub Filter.tendsto_id
  have hden : Filter.Tendsto (fun p : ℝ => κ + 1 - p) (nhds 0) (nhds (κ + 1 - 0)) :=
    tendsto_const_nhds.sub Filter.tendsto_id
  have h0 : κ + 1 - 0 ≠ 0 := by rw [sub_zero]; exact ne_of_gt (by linarith)
  have h := hnum.div hden h0
  have heq : (1 - 0) / (κ + 1 - 0) = 1 / (1 + κ) := by
    rw [sub_zero, sub_zero, add_comm]
  rw [heq] at h
  exact h

/-- **S5, quantification error (Table S5.1).** The finite-κ relative error is
1 − γ = κ/(1 + κ). -/
theorem s5_error {κ : ℝ} (hκ : 0 < κ) :
    1 - 1 / (1 + κ) = κ / (1 + κ) := by
  have h1 : (1 : ℝ) + κ ≠ 0 := ne_of_gt (by linarith)
  field_simp

/-- **S6, threshold-cycle identity.** Ideal doubling A_t = M·2^t with threshold
crossing F_th = α·M·2^Ct gives Ct = log₂(F_th/α) − log₂ M: Ct is exactly
linear in log₂ M with slope −1, the theoretical basis of the qPCR standard
curve. The prefactor F_th/α is the scale-degeneracy parameter of S2e. -/
theorem s6_ct {α M F_th : ℝ} (hα : 0 < α) (hM : 0 < M) (hF : 0 < F_th) (Ct : ℝ)
    (h : α * M * 2 ^ Ct = F_th) :
    Ct = Real.logb 2 (F_th / α) - Real.logb 2 M := by
  have hprod : (0 : ℝ) < α * M := by positivity
  have hprod' : α * M ≠ 0 := ne_of_gt hprod
  have hpos : (0 : ℝ) < F_th / (α * M) := by positivity
  have h2 : (2 : ℝ) ^ Ct = F_th / (α * M) := by
    field_simp
    linear_combination h
  have h3 : Real.logb 2 (F_th / α) - Real.logb 2 M = Real.logb 2 (F_th / (α * M)) := by
    rw [Real.logb_div hF.ne' hα.ne', Real.logb_div hF.ne' hprod.ne',
      Real.logb_mul hα.ne' hM.ne']
    ring
  rw [h3]
  symm
  rw [Real.logb_eq_iff_rpow_eq two_pos (by norm_num) hpos]
  exact h2

/-- **S7.2, DigitISA mass conservation.** Full conservation reads
c_target = c_complex + c_complex·K/c_free = c_complex·(1 + K/c_free); the
DigitISA working formula c_target = c_complex·K/c_free omits the first term. -/
theorem s7_digitisa_conservation {c_comp K c_free : ℝ} (hcf : c_free ≠ 0) :
    c_comp + c_comp * K / c_free = c_comp * (1 + K / c_free) := by
  field_simp
  ring

/-- **S7.2, DigitISA omitted-term share.** The omitted term is the fraction
c_complex/c_target = c_free/(c_free + K) of the true total, small iff
K ≪ c_free (strong depletion). -/
theorem s7_digitisa_share {c_comp K c_free : ℝ} (hc : 0 < c_comp) (hK : 0 < K)
    (hcf : 0 < c_free) :
    c_comp / (c_comp + c_comp * K / c_free) = c_free / (c_free + K) := by
  have h1 : c_free + K ≠ 0 := ne_of_gt (by linarith)
  have h2 : c_comp + c_comp * K / c_free ≠ 0 := by
    have heq : c_comp + c_comp * K / c_free = c_comp * (c_free + K) / c_free := by
      field_simp
      ring
    rw [heq]
    exact div_ne_zero (mul_ne_zero (ne_of_gt hc) h1) (ne_of_gt hcf)
  field_simp
  ring

/-- **S7.2, strong-depletion limit.** As K/c_free → 0⁺ (the κ → 0 corner,
biotin-streptavidin regime), the complex share tends to 1:
c_target ≈ c_complex exactly in the limit. -/
theorem s7_strong :
    Filter.Tendsto (fun r : ℝ => 1 / (1 + r)) (nhdsWithin 0 (Set.Ioi 0)) (nhds 1) := by
  have hden : Filter.Tendsto (fun r : ℝ => 1 + r) (nhdsWithin 0 (Set.Ioi 0))
      (nhds (1 + 0)) :=
    Filter.Tendsto.mono_left (tendsto_const_nhds.add Filter.tendsto_id) nhdsWithin_le_nhds
  rw [add_zero] at hden
  have h : Filter.Tendsto (fun r : ℝ => 1 / (1 + r)) (nhdsWithin 0 (Set.Ioi 0))
      (nhds (1 / 1)) :=
    Filter.Tendsto.div tendsto_const_nhds hden one_ne_zero
  simpa using h
