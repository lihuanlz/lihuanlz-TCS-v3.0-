/-
=====================================================================
S2c: Fisher determinant algebra (S2c.5.5) and the structural core of
the digital readout (S2c.4.1)
=====================================================================
Corresponds to Section S2c of the SI:
  - Theorem S2c.5.5.1: det(g) = W^2 V as a half double sum of weighted
    squared B-differences (via the Lagrange identity)
  - Degeneracy condition: det(g) > 0 iff two occupancies with nonzero
    weight differ
  - Theorem S2c.5.5.2: Langmuir scaling det(g) ~ C / κ^4 at large κ and
    the strong-depletion limit at κ -> 0 with rank-2 positivity
  - Theorem S2c.4.1: digital readout factorisation, finite-κ scale
    degeneracy, and unique readability at κ = 0

Lean 4 + Mathlib, toolchain leanprover/lean4:v4.15.0
-/

import Mathlib

namespace TCS

/-- **Lagrange identity** (named in the proof of Theorem S2c.5.5.1):
(Σa²)(Σb²) − (Σab)² = ½·Σ_lΣ_m (a_l b_m − a_m b_l)². -/
theorem lagrange_identity {ι : Type*} [Fintype ι] (a b : ι → ℝ) :
    (∑ l, (a l) ^ 2) * (∑ l, (b l) ^ 2) - (∑ l, a l * b l) ^ 2
      = (1 / 2) * ∑ l, ∑ m, (a l * b m - a m * b l) ^ 2 := by
  have hexp : ∀ l m : ι, (a l * b m - a m * b l) ^ 2 =
      (a l) ^ 2 * (b m) ^ 2 + (b l) ^ 2 * (a m) ^ 2
        - 2 * ((a l * b l) * (a m * b m)) := fun l m => by ring
  simp only [hexp, Finset.sum_add_distrib, Finset.sum_sub_distrib,
    ← Finset.mul_sum, ← Finset.sum_mul]
  ring

/-- B-difference identity (step in the proof of Theorem S2c.5.5.1):
with B_l = κ + (1 − p_l), the κ cancels: B_l − B_m = p_m − p_l. -/
theorem fisher_B_diff {κ p_l p_m : ℝ} :
    (κ + (1 - p_l)) - (κ + (1 - p_m)) = p_m - p_l := by
  ring

/-- **Theorem S2c.5.5.1 (SI S2c.5.5), Fisher determinant identity.**
det(g) = W²V = (Σw)(ΣwB²) − (ΣwB)² = ½·Σ_lΣ_m w_l w_m (B_l − B_m)².
The SI proof takes a = √w, b = √w·B in the Lagrange identity; here it is a
pure polynomial identity, valid for all real weights. -/
theorem fisher_det_identity {ι : Type*} [Fintype ι] (w B : ι → ℝ) :
    (∑ l, w l) * (∑ l, w l * (B l) ^ 2) - (∑ l, w l * B l) ^ 2
      = (1 / 2) * ∑ l, ∑ m, w l * w m * (B l - B m) ^ 2 := by
  have hexp : ∀ l m : ι, w l * w m * (B l - B m) ^ 2 =
      (w l * (B l) ^ 2) * w m + w l * (w m * (B m) ^ 2)
        - 2 * ((w l * B l) * (w m * B m)) := fun l m => by ring
  simp only [hexp, Finset.sum_add_distrib, Finset.sum_sub_distrib,
    ← Finset.mul_sum, ← Finset.sum_mul]
  ring

/-- **Corollary (SI S2c.5.5), degeneracy condition.** For nonnegative weights,
det(g) > 0 iff at least two occupancies with nonzero weight differ:
∃ l m, w l ≠ 0 ∧ w m ≠ 0 ∧ B l ≠ B m. Combined with `fisher_B_diff` this is
exactly "at least two occupancies p_l ≠ p_m". -/
theorem fisher_det_pos_iff {ι : Type*} [Fintype ι] (w B : ι → ℝ)
    (hw : ∀ l, 0 ≤ w l) :
    0 < (1 / 2) * ∑ l, ∑ m, w l * w m * (B l - B m) ^ 2 ↔
      ∃ l m, w l ≠ 0 ∧ w m ≠ 0 ∧ B l ≠ B m := by
  constructor
  · intro h
    by_contra hcon
    push_neg at hcon
    have hz : ∀ l m : ι, w l * w m * (B l - B m) ^ 2 = 0 := by
      intro l m
      by_cases hl : w l = 0
      · rw [hl]; ring
      by_cases hm : w m = 0
      · rw [hm]; ring
      have hB : B l = B m := hcon l m hl hm
      rw [hB]; ring
    have hsum0 : ∑ l, ∑ m, w l * w m * (B l - B m) ^ 2 = 0 :=
      Finset.sum_eq_zero (fun l _ => Finset.sum_eq_zero (fun m _ => hz l m))
    rw [hsum0] at h
    norm_num at h
  · rintro ⟨l, m, hl, hm, hB⟩
    have hwl : 0 < w l := lt_of_le_of_ne' (hw l) hl
    have hwm : 0 < w m := lt_of_le_of_ne' (hw m) hm
    have hterm : 0 < w l * w m * (B l - B m) ^ 2 :=
      mul_pos (mul_pos hwl hwm) (sq_pos_of_ne_zero (sub_ne_zero.mpr hB))
    have hnn : ∀ l' m' : ι, 0 ≤ w l' * w m' * (B l' - B m') ^ 2 :=
      fun l' m' => mul_nonneg (mul_nonneg (hw l') (hw m')) (sq_nonneg _)
    have hinner : 0 < ∑ m', w l * w m' * (B l - B m') ^ 2 :=
      Finset.sum_pos' (fun m' _ => hnn l m') ⟨m, Finset.mem_univ m, hterm⟩
    have houter : 0 < ∑ l', ∑ m', w l' * w m' * (B l' - B m') ^ 2 :=
      Finset.sum_pos' (fun l' _ => Finset.sum_nonneg (fun m' _ => hnn l' m'))
        ⟨l, Finset.mem_univ l, hinner⟩
    linarith

/-- **(S2c.5.5.5), sensitivity ratio.** With the master-equation sensitivities
(S2c.5.5.2), ∂p/∂κ = −pq/D and ∂p/∂(ln M₀) = pqB/D (obtained by implicit
differentiation of ξ = p(κ + q)/(κq); we take them as given and prove the
ratio), the per-point sensitivity ratio equals B_l = κ + (1 − p_l). This is the
mechanism enabling parameter separation when occupancies differ. -/
theorem sensitivity_ratio {p q B D : ℝ} (hp : 0 < p) (hq : 0 < q) (hD : 0 < D) :
    (p * q * B / D) / |(-(p * q) / D)| = B := by
  have h1 : (0:ℝ) < p * q / D := div_pos (mul_pos hp hq) hD
  have habs : |(-(p * q) / D)| = p * q / D := by
    rw [show (-(p * q) / D) = -(p * q / D) by ring, abs_neg, abs_of_pos h1]
  have hD' : D ≠ 0 := ne_of_gt hD
  have hpq : p * q / D ≠ 0 := ne_of_gt h1
  rw [habs]
  field_simp

/-- Helper for Theorem S2c.5.5.2: κ/(κ + a) → 1 as κ → ∞. -/
private lemma tendsto_self_div_add_atTop (a : ℝ) :
    Filter.Tendsto (fun κ : ℝ => κ / (κ + a)) Filter.atTop (nhds 1) := by
  have h1 : Filter.Tendsto (fun κ : ℝ => (κ:ℝ)⁻¹) Filter.atTop (nhds 0) :=
    tendsto_inv_atTop_zero
  have h2 : Filter.Tendsto (fun κ : ℝ => 1 + a * κ⁻¹) Filter.atTop
      (nhds (1 + a * 0)) := tendsto_const_nhds.add (h1.const_mul a)
  have h3 : Filter.Tendsto (fun κ : ℝ => (1 + a * κ⁻¹)⁻¹) Filter.atTop
      (nhds 1) := by
    have h := h2.inv₀ (by norm_num : (1 : ℝ) + a * 0 ≠ 0)
    simpa only [Pi.inv_apply, add_zero, mul_zero, inv_one] using h
  have hev : (fun κ : ℝ => (1 + a * κ⁻¹)⁻¹) =ᶠ[Filter.atTop]
      (fun κ => κ / (κ + a)) := by
    filter_upwards [Filter.eventually_gt_atTop (0:ℝ),
      Filter.eventually_gt_atTop (-a + 1)] with κ hκ hκa
    have hκ0 : κ ≠ 0 := ne_of_gt hκ
    have hκa0 : κ + a ≠ 0 := by
      intro hcontra
      linarith
    field_simp
  have h4 := h3.congr' hev
  simpa using h4

/-- **Theorem S2c.5.5.2, Langmuir scaling (S2c.5.5.6).** With occupancies held
fixed (κ-independent), w_l(κ) = c_l/(κ + q_l²)² where c_l = n_l p_l q_l, and
B-differences are κ-independent; then κ⁴·det(g)(κ) → C∞ as κ → ∞, where
C∞ = ½·ΣΣ c_l c_m (ΔB_lm)². Hence det(g) ~ C∞/κ⁴ → 0: asymptotic rank collapse. -/
theorem fisher_det_langmuir_scaling {ι : Type*} [Fintype ι]
    (c q ΔB : ι → ℝ) :
    Filter.Tendsto
      (fun κ : ℝ => κ ^ 4 * ((1 / 2) * ∑ l, ∑ m,
        (c l / (κ + (q l) ^ 2) ^ 2) * (c m / (κ + (q m) ^ 2) ^ 2) *
          (ΔB l - ΔB m) ^ 2))
      Filter.atTop
      (nhds ((1 / 2) * ∑ l, ∑ m, c l * c m * (ΔB l - ΔB m) ^ 2)) := by
  have hpair : ∀ l m : ι, Filter.Tendsto
      (fun κ : ℝ => κ ^ 4 * ((c l / (κ + (q l) ^ 2) ^ 2) *
        (c m / (κ + (q m) ^ 2) ^ 2) * (ΔB l - ΔB m) ^ 2))
      Filter.atTop (nhds (c l * c m * (ΔB l - ΔB m) ^ 2)) := by
    intro l m
    have hA : Filter.Tendsto (fun κ : ℝ => (κ / (κ + (q l) ^ 2)) ^ 2)
        Filter.atTop (nhds 1) := by
      simpa using (tendsto_self_div_add_atTop ((q l) ^ 2)).pow 2
    have hB : Filter.Tendsto (fun κ : ℝ => (κ / (κ + (q m) ^ 2)) ^ 2)
        Filter.atTop (nhds 1) := by
      simpa using (tendsto_self_div_add_atTop ((q m) ^ 2)).pow 2
    have hprod : Filter.Tendsto
        (fun κ : ℝ => (κ / (κ + (q l) ^ 2)) ^ 2 * (κ / (κ + (q m) ^ 2)) ^ 2 *
          (c l * c m * (ΔB l - ΔB m) ^ 2))
        Filter.atTop (nhds (1 * 1 * (c l * c m * (ΔB l - ΔB m) ^ 2))) :=
      (hA.mul hB).mul tendsto_const_nhds
    have hev :
        (fun κ : ℝ => (κ / (κ + (q l) ^ 2)) ^ 2 * (κ / (κ + (q m) ^ 2)) ^ 2 *
            (c l * c m * (ΔB l - ΔB m) ^ 2))
          =ᶠ[Filter.atTop]
        (fun κ : ℝ => κ ^ 4 * ((c l / (κ + (q l) ^ 2) ^ 2) *
          (c m / (κ + (q m) ^ 2) ^ 2) * (ΔB l - ΔB m) ^ 2)) := by
      filter_upwards with κ
      rw [div_pow, div_pow, div_mul_div_comm, div_mul_div_comm]
      ring
    have hlim := hprod.congr' hev
    simpa using hlim
  have hsum : Filter.Tendsto
      (fun κ : ℝ => ∑ l, ∑ m, κ ^ 4 * ((c l / (κ + (q l) ^ 2) ^ 2) *
        (c m / (κ + (q m) ^ 2) ^ 2) * (ΔB l - ΔB m) ^ 2))
      Filter.atTop (nhds (∑ l, ∑ m, c l * c m * (ΔB l - ΔB m) ^ 2)) :=
    tendsto_finset_sum Finset.univ
      (fun l _ => tendsto_finset_sum Finset.univ (fun m _ => hpair l m))
  have hfun : ∀ κ : ℝ, κ ^ 4 * ((1 / 2) * ∑ l, ∑ m,
        (c l / (κ + (q l) ^ 2) ^ 2) * (c m / (κ + (q m) ^ 2) ^ 2) *
          (ΔB l - ΔB m) ^ 2)
      = (1 / 2) * ∑ l, ∑ m, κ ^ 4 * ((c l / (κ + (q l) ^ 2) ^ 2) *
          (c m / (κ + (q m) ^ 2) ^ 2) * (ΔB l - ΔB m) ^ 2) := by
    intro κ
    conv_lhs => rw [mul_left_comm (κ ^ 4) ((1:ℝ) / 2) _]
    congr 1
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro l _
    rw [Finset.mul_sum]
  rw [show (fun κ : ℝ => κ ^ 4 * ((1 / 2) * ∑ l, ∑ m,
        (c l / (κ + (q l) ^ 2) ^ 2) * (c m / (κ + (q m) ^ 2) ^ 2) *
          (ΔB l - ΔB m) ^ 2))
      = fun κ : ℝ => (1 / 2) * ∑ l, ∑ m, κ ^ 4 *
        ((c l / (κ + (q l) ^ 2) ^ 2) * (c m / (κ + (q m) ^ 2) ^ 2) *
          (ΔB l - ΔB m) ^ 2) from funext hfun]
  exact hsum.const_mul (1 / 2)

/-- **Theorem S2c.5.5.2, strong-depletion limit (S2c.5.5.7), convergence.**
If the occupancies converge p_l(κ) → s_l with s_l < 1 as κ → 0⁺, then
det(g)(κ) → W₀²V₀ with W₀ = Σ n_l s_l/(1 − s_l)³, since each weight
w_l(κ) = n_l p_l q_l/(κ + q_l²)² → n_l s_l/(1 − s_l)³ and
B_l − B_m → s_m − s_l. -/
theorem fisher_det_zero_limit {ι : Type*} [Fintype ι] (n s : ι → ℝ)
    (p : ι → ℝ → ℝ) (hs : ∀ l, s l < 1)
    (hp : ∀ l, Filter.Tendsto (p l) (nhdsWithin 0 (Set.Ioi 0)) (nhds (s l))) :
    Filter.Tendsto
      (fun κ => (1 / 2) * ∑ l, ∑ m,
        (n l * p l κ * (1 - p l κ) / (κ + (1 - p l κ) ^ 2) ^ 2) *
        (n m * p m κ * (1 - p m κ) / (κ + (1 - p m κ) ^ 2) ^ 2) *
        ((κ + (1 - p l κ)) - (κ + (1 - p m κ))) ^ 2)
      (nhdsWithin 0 (Set.Ioi 0))
      (nhds ((1 / 2) * ∑ l, ∑ m,
        (n l * s l / (1 - s l) ^ 3) * (n m * s m / (1 - s m) ^ 3) *
          (s m - s l) ^ 2)) := by
  have hκid : Filter.Tendsto (fun κ : ℝ => κ) (nhdsWithin 0 (Set.Ioi 0))
      (nhds (0:ℝ)) := tendsto_nhdsWithin_of_tendsto_nhds continuousAt_id.tendsto
  have h1p : ∀ l, Filter.Tendsto (fun κ => 1 - p l κ) (nhdsWithin 0 (Set.Ioi 0))
      (nhds (1 - s l)) := fun l => tendsto_const_nhds.sub (hp l)
  have hW : ∀ l, Filter.Tendsto
      (fun κ => n l * p l κ * (1 - p l κ) / (κ + (1 - p l κ) ^ 2) ^ 2)
      (nhdsWithin 0 (Set.Ioi 0)) (nhds (n l * s l / (1 - s l) ^ 3)) := by
    intro l
    have h1s : (0:ℝ) < 1 - s l := sub_pos.mpr (hs l)
    have h1s0 : (1:ℝ) - s l ≠ 0 := ne_of_gt h1s
    have hnum : Filter.Tendsto (fun κ => n l * p l κ * (1 - p l κ))
        (nhdsWithin 0 (Set.Ioi 0)) (nhds (n l * s l * (1 - s l))) :=
      (tendsto_const_nhds.mul (hp l)).mul (h1p l)
    have hden : Filter.Tendsto (fun κ => (κ + (1 - p l κ) ^ 2) ^ 2)
        (nhdsWithin 0 (Set.Ioi 0)) (nhds ((0 + (1 - s l) ^ 2) ^ 2)) :=
      (hκid.add ((h1p l).pow 2)).pow 2
    have hden0 : (0 + (1 - s l) ^ 2) ^ 2 ≠ 0 := by
      have hsq : (0:ℝ) < (1 - s l) ^ 2 := sq_pos_of_pos h1s
      have hpos : (0:ℝ) < 0 + (1 - s l) ^ 2 := by linarith
      exact ne_of_gt (pow_pos hpos 2)
    have hdiv := hnum.div hden hden0
    have hlim : n l * s l * (1 - s l) / ((0 + (1 - s l) ^ 2) ^ 2)
        = n l * s l / (1 - s l) ^ 3 := by
      field_simp
      ring
    rwa [hlim] at hdiv
  have hpair : ∀ l m : ι, Filter.Tendsto
      (fun κ => (n l * p l κ * (1 - p l κ) / (κ + (1 - p l κ) ^ 2) ^ 2) *
        (n m * p m κ * (1 - p m κ) / (κ + (1 - p m κ) ^ 2) ^ 2) *
        ((κ + (1 - p l κ)) - (κ + (1 - p m κ))) ^ 2)
      (nhdsWithin 0 (Set.Ioi 0))
      (nhds ((n l * s l / (1 - s l) ^ 3) * (n m * s m / (1 - s m) ^ 3) *
        (s m - s l) ^ 2)) := by
    intro l m
    have hΔ : Filter.Tendsto (fun κ => (κ + (1 - p l κ)) - (κ + (1 - p m κ)))
        (nhdsWithin 0 (Set.Ioi 0)) (nhds (s m - s l)) := by
      have h := (hκid.add (h1p l)).sub (hκid.add (h1p m))
      have hval : (0 + (1 - s l)) - (0 + (1 - s m)) = s m - s l := by ring
      rwa [hval] at h
    have hΔ2 : Filter.Tendsto
        (fun κ => ((κ + (1 - p l κ)) - (κ + (1 - p m κ))) ^ 2)
        (nhdsWithin 0 (Set.Ioi 0)) (nhds ((s m - s l) ^ 2)) :=
      hΔ.pow 2
    exact ((hW l).mul (hW m)).mul hΔ2
  have hsum : Filter.Tendsto
      (fun κ => ∑ l, ∑ m,
        (n l * p l κ * (1 - p l κ) / (κ + (1 - p l κ) ^ 2) ^ 2) *
        (n m * p m κ * (1 - p m κ) / (κ + (1 - p m κ) ^ 2) ^ 2) *
        ((κ + (1 - p l κ)) - (κ + (1 - p m κ))) ^ 2)
      (nhdsWithin 0 (Set.Ioi 0))
      (nhds (∑ l, ∑ m,
        (n l * s l / (1 - s l) ^ 3) * (n m * s m / (1 - s m) ^ 3) *
          (s m - s l) ^ 2)) :=
    tendsto_finset_sum Finset.univ
      (fun l _ => tendsto_finset_sum Finset.univ (fun m _ => hpair l m))
  exact hsum.const_mul (1 / 2)

/-- **(S2c.5.5.7), positivity of the strong-depletion limit.** If 0 < s_l < 1,
n_l > 0 and s is injective, then W₀²V₀ > 0: at κ = 0 the Fisher matrix retains
rank 2, no information barrier opposes the calibration-free reading. -/
theorem fisher_det_zero_pos {ι : Type*} [Fintype ι] [Nontrivial ι] (n s : ι → ℝ)
    (hn : ∀ l, 0 < n l) (hs0 : ∀ l, 0 < s l) (hs1 : ∀ l, s l < 1)
    (hsd : Function.Injective s) :
    0 < (1 / 2) * ∑ l, ∑ m,
      (n l * s l / (1 - s l) ^ 3) * (n m * s m / (1 - s m) ^ 3) *
        (s m - s l) ^ 2 := by
  obtain ⟨l, m, hlm⟩ := exists_pair_ne ι
  have hsm : s m ≠ s l := fun h => hlm (hsd h.symm)
  have hWpos : ∀ l', 0 < n l' * s l' / (1 - s l') ^ 3 := fun l' =>
    div_pos (mul_pos (hn l') (hs0 l')) (pow_pos (sub_pos.mpr (hs1 l')) 3)
  have hterm : 0 < (n l * s l / (1 - s l) ^ 3) * (n m * s m / (1 - s m) ^ 3) *
      (s m - s l) ^ 2 :=
    mul_pos (mul_pos (hWpos l) (hWpos m)) (sq_pos_of_ne_zero (sub_ne_zero.mpr hsm))
  have hnn : ∀ l' m' : ι, 0 ≤ (n l' * s l' / (1 - s l') ^ 3) *
      (n m' * s m' / (1 - s m') ^ 3) * (s m' - s l') ^ 2 := fun l' m' =>
    mul_nonneg (mul_nonneg (hWpos l').le (hWpos m').le) (sq_nonneg _)
  have hinner : 0 < ∑ m', (n l * s l / (1 - s l) ^ 3) *
      (n m' * s m' / (1 - s m') ^ 3) * (s m' - s l) ^ 2 :=
    Finset.sum_pos' (fun m' _ => hnn l m') ⟨m, Finset.mem_univ m, hterm⟩
  have houter : 0 < ∑ l', ∑ m', (n l' * s l' / (1 - s l') ^ 3) *
      (n m' * s m' / (1 - s m') ^ 3) * (s m' - s l') ^ 2 :=
    Finset.sum_pos' (fun l' _ => Finset.sum_nonneg (fun m' _ => hnn l' m'))
      ⟨l, Finset.mem_univ l, hinner⟩
  linarith

/-- Digital positive-rate readout on a Type 1 platform (SI S2c.4):
P_specific = 1 − (1 − p)^β. -/
def digitalReadout (β : ℕ) (p : ℝ) : ℝ := 1 - (1 - p) ^ β

/-- **Theorem S2c.4.1, factorisation.** In the strong-depletion linear regime
p = μ/(1 + κ) with μ = M/Ω (the finite-Ω correction `finite_omega_correction` proved in
`TCS.S1c_MasterEquation`), the readout depends on (μ, κ) only through the single
group μ/(1 + κ). The content is the dependence structure: the readout factors
through μ/(1 + κ) by definition. -/
theorem s2c41_factorization (β : ℕ) (μ κ : ℝ) :
    digitalReadout β (μ / (1 + κ)) = 1 - (1 - μ / (1 + κ)) ^ β := rfl

/-- **Theorem S2c.4.1, digital degeneracy at finite κ.** The readout is
invariant along the rescaling μ' = μ·(1 + κ')/(1 + κ), while μ' ≠ μ whenever
κ' ≠ κ and μ ≠ 0: at finite κ the digital readout alone cannot resolve μ
(hence M) without independent knowledge of κ. This is why digital readout
removes the gain-factor coupling but not the scale degeneracy. -/
theorem s2c41_digital_degeneracy {β : ℕ} {μ κ κ' : ℝ}
    (hκ : 1 + κ ≠ 0) (hκ' : 1 + κ' ≠ 0) (hne : κ' ≠ κ) (hμ : μ ≠ 0) :
    1 - (1 - (μ * (1 + κ') / (1 + κ)) / (1 + κ')) ^ β
      = 1 - (1 - μ / (1 + κ)) ^ β ∧
    μ * (1 + κ') / (1 + κ) ≠ μ := by
  constructor
  · have hX : (μ * (1 + κ') / (1 + κ)) / (1 + κ') = μ / (1 + κ) := by
      field_simp
      ring
    rw [hX]
  · intro h
    have hmul : μ * (1 + κ') / (1 + κ) * (1 + κ) = μ * (1 + κ) := by rw [h]
    rw [div_mul_cancel₀ _ hκ] at hmul
    have h2 : μ * κ' = μ * κ := by linear_combination hmul
    exact hne (mul_left_cancel₀ hμ h2)

/-- **Theorem S2c.4.1, κ = 0 readability.** At κ = 0 the readout is
P = 1 − (1 − μ)^β, injective in μ on [0, 1] for β ≥ 1: equal readouts force
equal μ, so μ (and with known Ω, the mass M) is uniquely recoverable.
Together with `poisson_limit` and `estimator_dPCR_limit` (TCS.S2b_DigitalStatistics) this
packages the calibration-free equivalence: unique readability holds exactly
at κ → 0, and `s2c41_digital_degeneracy` shows it fails at every finite κ. -/
theorem s2c41_kappa_zero {β : ℕ} (hβ : 1 ≤ β) {μ₁ μ₂ : ℝ}
    (h1 : μ₁ ∈ Set.Icc 0 1) (h2 : μ₂ ∈ Set.Icc 0 1)
    (heq : 1 - (1 - μ₁) ^ β = 1 - (1 - μ₂) ^ β) :
    μ₁ = μ₂ := by
  have hβ0 : β ≠ 0 := by omega
  have h11 : μ₁ ≤ 1 := h1.2
  have h21 : μ₂ ≤ 1 := h2.2
  have h1m : (0:ℝ) ≤ 1 - μ₁ := by linarith
  have h2m : (0:ℝ) ≤ 1 - μ₂ := by linarith
  replace heq : (1 - μ₁) ^ β = (1 - μ₂) ^ β := by linarith
  have hEq : 1 - μ₁ = 1 - μ₂ := (pow_left_inj₀ h1m h2m hβ0).mp heq
  linarith

end TCS
