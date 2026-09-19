/-
=====================================================================
S10: The generalised master equation and pharmacology reductions
=====================================================================
Corresponds to Section S10 of the SI ("Supplementary and additional
Material"):
  - (S10.2)/(S10.3): conservation and the dimensionless master form
    ξ = x + p/κ
  - (S10.3): Clark limit; the exact strong-depletion ratio identity and
    its limit p/(κξ) → 1 as κ → 0⁺
  - (S10.8): the apparent Hill slope at half-saturation lies in (1, 2)
    and interpolates between 2 (κ → 0⁺) and 1 (κ → ∞)
  - (S10.10): Cheng-Prusoff relation IC₅₀ = K_I·(1 + L_T/K_L)
  - (S10.12)-(S10.15): the effector (operational) quadratic, its
    physical root, the Black-Leff limit and the receptor-reserve bounds
  - (S10.17): the TCS-MWC / TCS-KNF unification skeleton
  - (S10.18): the eTCM signal: basal activity, inverse agonism,
    G-protein depletion
  - (S10.19): scale degeneracy
  - (S10.20)/(S10.21): the generalised master equation, the
    independent-sites limit ν = n·x/(1 + x), and the zero/strong
    depletion limits

Lean 4 + Mathlib, toolchain leanprover/lean4:v4.15.0
-/

import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Positivity.Basic
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.FunProp
import Mathlib.Tactic.PushNeg
import Mathlib.Tactic.Coe
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Nat.Choose.Sum
import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Topology.Order.Basic
import Mathlib.Topology.Algebra.GroupWithZero
import Mathlib.Topology.Algebra.Order.Field
import Mathlib.Order.Filter.AtTopBot.Group
import Mathlib.Order.Filter.AtTopBot.Archimedean
import Mathlib.Order.Filter.AtTopBot.Field

open Filter Topology Set

namespace TCS

/-- **Theorem S10.2 (SI S10.2), ligand mass conservation, and Eq. (S10.3).**
L_T = L_free + n·R_T·p divided by K_d gives the dimensionless master form
ξ = x + p/κ with ξ := L_T/K_d, x := L_free/K_d, κ := K_d/(n·R_T). -/
theorem theorem_S10_2_master {L_T L_free R_T K_d : ℝ} {n : ℕ} {p : ℝ}
    (hn : (0 : ℝ) < n) (hRT : 0 < R_T) (hKd : K_d ≠ 0)
    (hcons : L_T = L_free + n * R_T * p) :
    L_T / K_d = L_free / K_d + p / (K_d / ((n : ℝ) * R_T)) := by
  have hnR : (n : ℝ) * R_T ≠ 0 := mul_ne_zero (ne_of_gt hn) (ne_of_gt hRT)
  rw [hcons]
  field_simp [hKd, hnR]
  ring

/-- **Theorem S10.3 (SI S10.3), Clark limit.** At κ → ∞ the depletion term
vanishes and the master equation reduces to Clark's equation: the algebraic
content is that ξ = p/(1 − p) rearranges to p = ξ/(1 + ξ). Hypothesis hp
excludes the degenerate point p = 1 (outside the physical domain (0,1)). -/
theorem S10_3_clark {p ξ : ℝ} (hp : p ≠ 1) (hξ : ξ ≠ -1) (h : ξ = p / (1 - p)) :
    p = ξ / (1 + ξ) := by
  have h1p : (1 : ℝ) - p ≠ 0 := sub_ne_zero.mpr (fun h1 => hp h1.symm)
  have h1ξ : (1 : ℝ) + ξ ≠ 0 := by
    intro h1
    exact hξ (by linarith [h1])
  have h2 : ξ * (1 - p) = p := by
    rw [h]
    exact div_mul_cancel₀ p h1p
  rw [eq_div_iff h1ξ]
  linear_combination -h2

/-- **Theorem S10.4 (SI S10.3), strong-depletion ratio, exact identity.**
For the master equation ξ = p/(1 − p) + p/κ the depletion share satisfies
p/(κ·ξ) = (1 − p)/(1 − p + κ) exactly. -/
theorem S10_4_ratio_identity {p κ ξ : ℝ} (hp0 : 0 < p) (hp1 : p < 1) (hκ : 0 < κ)
    (h : ξ = p / (1 - p) + p / κ) :
    p / (κ * ξ) = (1 - p) / (1 - p + κ) := by
  have h1p : (0 : ℝ) < 1 - p := by linarith
  have hκ1p : (0 : ℝ) < κ + 1 - p := by linarith
  have hκξ : κ * ξ = p * (κ + 1 - p) / (1 - p) := by
    rw [h]
    field_simp [hκ.ne', h1p.ne']
    ring
  have hnum0 : p * (κ + 1 - p) / (1 - p) ≠ 0 :=
    ne_of_gt (div_pos (mul_pos hp0 hκ1p) h1p)
  have h1 : (0 : ℝ) < 1 - p + κ := by linarith
  rw [hκξ, div_eq_div_iff hnum0 (ne_of_gt h1)]
  field_simp [h1p.ne']
  left
  ring

/-- **Theorem S10.4 (SI S10.3), strong-depletion limit.** As κ → 0⁺ at fixed
0 < p < 1, the depletion share tends to 1: p/(κξ) = (1 − p)/(1 − p + κ) → 1. -/
theorem S10_4_ratio (p : ℝ) (hp0 : 0 < p) (hp1 : p < 1) :
    Filter.Tendsto (fun κ : ℝ => (1 - p) / (1 - p + κ))
      (nhdsWithin 0 (Set.Ioi 0)) (nhds 1) := by
  have _ := hp0
  have h1p : (0 : ℝ) < 1 - p := by linarith
  have hden : Filter.Tendsto (fun κ : ℝ => 1 - p + κ) (𝓝 0) (𝓝 (1 - p + 0)) :=
    tendsto_const_nhds.add tendsto_id
  have hnum : Filter.Tendsto (fun _ : ℝ => 1 - p) (𝓝 0) (𝓝 ((1 : ℝ) - p)) :=
    tendsto_const_nhds
  have h2 := hnum.div hden (by rw [add_zero]; exact ne_of_gt h1p)
  rw [add_zero, div_self (ne_of_gt h1p)] at h2
  exact h2.mono_left nhdsWithin_le_nhds

/-- Local Hill slope at half-saturation, Eq. (S10.8). -/
noncomputable def hillSlope0 (κ : ℝ) : ℝ := 2 * (2 * κ + 1) / (4 * κ + 1)

/-- **Eq. (S10.8), lower limit.** n_H⁽⁰⁾ → 2 as κ → 0⁺. -/
theorem S10_5_hill_tendsto_two :
    Filter.Tendsto hillSlope0 (nhdsWithin 0 (Set.Ioi 0)) (nhds 2) := by
  have hcont : ContinuousAt hillSlope0 0 := by
    show ContinuousAt (fun κ : ℝ => 2 * (2 * κ + 1) / (4 * κ + 1)) 0
    apply ContinuousAt.div
    · fun_prop
    · fun_prop
    · norm_num
  have hval : hillSlope0 0 = 2 := by norm_num [hillSlope0]
  have h2 := hcont.tendsto
  rw [hval] at h2
  exact h2.mono_left nhdsWithin_le_nhds

/-- **Eq. (S10.8), upper limit.** n_H⁽⁰⁾ → 1 as κ → ∞. -/
theorem S10_5_hill_tendsto_one :
    Filter.Tendsto hillSlope0 Filter.atTop (nhds 1) := by
  have heq : hillSlope0 =ᶠ[Filter.atTop] fun κ : ℝ => 1 + 1 / (4 * κ + 1) := by
    filter_upwards [eventually_gt_atTop (0 : ℝ)] with κ hκ
    have hd : (4 : ℝ) * κ + 1 ≠ 0 := ne_of_gt (by positivity)
    simp only [hillSlope0]
    field_simp [hd]
    ring
  have hg : Filter.Tendsto (fun κ : ℝ => 4 * κ + 1) Filter.atTop Filter.atTop := by
    have h1 : Filter.Tendsto (fun κ : ℝ => 4 * κ) Filter.atTop Filter.atTop :=
      Tendsto.const_mul_atTop (by norm_num : (0 : ℝ) < 4) tendsto_id
    exact tendsto_atTop_add_const_right Filter.atTop (1 : ℝ) h1
  have h0 : Filter.Tendsto (fun κ : ℝ => 1 / (4 * κ + 1)) Filter.atTop (𝓝 0) := by
    have hi := tendsto_inv_atTop_zero.comp hg
    simpa only [one_div, Function.comp] using hi
  have hfin : Filter.Tendsto (fun κ : ℝ => 1 + 1 / (4 * κ + 1)) Filter.atTop
      (𝓝 (1 + 0)) := tendsto_const_nhds.add h0
  rw [add_zero] at hfin
  exact Tendsto.congr' heq.symm hfin

/-- **Eq. (S10.8), bounds.** For κ > 0 the analytical bound satisfies
1 < n_H⁽⁰⁾(κ) < 2, so the apparent Hill coefficient is strictly between 1 and 2. -/
theorem S10_5_hill_bounds {κ : ℝ} (hκ : 0 < κ) :
    1 < hillSlope0 κ ∧ hillSlope0 κ < 2 := by
  have hd : (0 : ℝ) < 4 * κ + 1 := by positivity
  simp only [hillSlope0]
  constructor
  · rw [lt_div_iff₀ hd]
    linarith
  · rw [div_lt_iff₀ hd]
    linarith

/-- **Theorem S10.6 (SI S10.5), Cheng-Prusoff.** For competitive binding (S10.9)
with no depletion, half-inhibition (p_L equal to half of its inhibitor-free value
x_L/(1 + x_L)) forces x_I = 1 + x_L, i.e. IC₅₀ = K_I · (1 + L_T/K_L). -/
theorem S10_6_chengprusoff {x_L x_I p_L p_I : ℝ}
    (hxL : 0 < x_L) (hθ : 1 - p_L - p_I ≠ 0)
    (hL : p_L = x_L * (1 - p_L - p_I)) (hI : p_I = x_I * (1 - p_L - p_I))
    (hhalf : p_L = x_L / (2 * (1 + x_L))) :
    x_I = 1 + x_L := by
  have _ := hθ
  have hxL0 : x_L ≠ 0 := ne_of_gt hxL
  have hden : (2 : ℝ) * (1 + x_L) ≠ 0 := by positivity
  have hθ' : 1 - p_L - p_I = 1 / (2 * (1 + x_L)) := by
    apply mul_left_cancel₀ hxL0
    calc x_L * (1 - p_L - p_I) = p_L := hL.symm
      _ = x_L / (2 * (1 + x_L)) := hhalf
      _ = x_L * (1 / (2 * (1 + x_L))) := by rw [mul_one_div]
  have hkey : (1 + x_L + x_I) * (1 - p_L - p_I) = 1 := by
    linear_combination -hL - hI
  have hfin : (1 + x_L + x_I) * (1 / (2 * (1 + x_L))) = 1 := by
    rw [← hθ']; exact hkey
  field_simp [hden] at hfin
  linarith [hfin]

/-- **Eq. (S10.12) ⟺ Eq. (S10.13).** The effector equation
τ·p_R = q/(1 − q) + q/κ_eff is equivalent to the quadratic
q² − (κ_eff τ p_R + κ_eff + 1) q + κ_eff τ p_R = 0 on 0 < q < 1, κ_eff > 0. -/
theorem S10_effector_quadratic {κ τ pR q : ℝ} (hκ : 0 < κ) (hq0 : 0 < q) (hq1 : q < 1) :
    τ * pR = q / (1 - q) + q / κ ↔
      q ^ 2 - (κ * τ * pR + κ + 1) * q + κ * τ * pR = 0 := by
  have _ := hq0
  have h1q : (0 : ℝ) < 1 - q := by linarith
  have hX : ((1 : ℝ) - q) * κ ≠ 0 := mul_ne_zero (ne_of_gt h1q) (ne_of_gt hκ)
  have key : (q / (1 - q) + q / κ) * ((1 - q) * κ) = q * κ + q * (1 - q) := by
    field_simp [h1q.ne', hκ.ne']
  constructor
  · intro h
    have h2 : (τ * pR) * ((1 - q) * κ) = q * κ + q * (1 - q) := by rw [h]; exact key
    linear_combination h2
  · intro h
    have h2 : (τ * pR) * ((1 - q) * κ) = q * κ + q * (1 - q) := by linear_combination h
    have h3 : (τ * pR) * ((1 - q) * κ) = (q / (1 - q) + q / κ) * ((1 - q) * κ) := by
      rw [h2, key]
    exact mul_right_cancel₀ hX h3

/-- **Eq. (S10.14), physical root.** The smaller root of the effector quadratic
is the physical one: it lies in (0, 1) and satisfies the quadratic. -/
theorem S10_effector_root {κ τ pR : ℝ} (hκ : 0 < κ) (ha : 0 < τ * pR) :
    let s := κ * τ * pR + κ + 1
    let q := (s - Real.sqrt (s ^ 2 - 4 * κ * τ * pR)) / 2
    q ^ 2 - s * q + κ * τ * pR = 0 ∧ 0 < q ∧ q < 1 := by
  intro s q
  have hsd : s = κ * τ * pR + κ + 1 := rfl
  have hqd : q = (s - Real.sqrt (s ^ 2 - 4 * κ * τ * pR)) / 2 := rfl
  have hkt : (0 : ℝ) < κ * τ * pR := by
    have h := mul_pos hκ ha
    rw [← mul_assoc] at h
    exact h
  have hspos : (0 : ℝ) < s := by rw [hsd]; linarith [hkt, hκ]
  have hdpos : (0 : ℝ) < s ^ 2 - 4 * κ * τ * pR := by
    have hd' : s ^ 2 - 4 * κ * τ * pR = (κ * τ * pR + κ - 1) ^ 2 + 4 * κ := by
      rw [hsd]; ring
    rw [hd']
    nlinarith [sq_nonneg (κ * τ * pR + κ - 1), hκ]
  have hdlt : s ^ 2 - 4 * κ * τ * pR < s ^ 2 := by linarith [hkt]
  have hsqrt_pos : (0 : ℝ) < Real.sqrt (s ^ 2 - 4 * κ * τ * pR) :=
    Real.sqrt_pos.mpr hdpos
  have hsqrt_lt : Real.sqrt (s ^ 2 - 4 * κ * τ * pR) < s :=
    (Real.sqrt_lt' hspos).mpr hdlt
  have hsq : Real.sqrt (s ^ 2 - 4 * κ * τ * pR) ^ 2 = s ^ 2 - 4 * κ * τ * pR :=
    Real.sq_sqrt hdpos.le
  have hroot : q ^ 2 - s * q + κ * τ * pR = 0 := by
    rw [hqd]
    linear_combination hsq / 4
  have hqpos : (0 : ℝ) < q := by rw [hqd]; linarith [hsqrt_lt]
  have hq1 : q < 1 := by
    rw [hqd]
    by_cases hs2 : s ≤ 2
    · linarith [hsqrt_pos, hs2]
    · push_neg at hs2
      have hgt : s - 2 < Real.sqrt (s ^ 2 - 4 * κ * τ * pR) := by
        have h2 : (s - 2) ^ 2 < s ^ 2 - 4 * κ * τ * pR := by
          rw [hsd]; nlinarith [mul_pos hκ ha, hκ, ha]
        exact (Real.lt_sqrt (by linarith)).mpr h2
      linarith [hgt]
  exact ⟨hroot, hqpos, hq1⟩

/-- **Black-Leff limit (SI S10.6), receptor layer.** q/(1 − q) = τ·p_R
rearranges to q = τ·p_R/(1 + τ·p_R). -/
theorem S10_blackleff_q {τ pR q : ℝ} (hτ : 0 ≤ τ * pR) (hq : q ≠ 1)
    (h : q / (1 - q) = τ * pR) :
    q = τ * pR / (1 + τ * pR) := by
  have h1q : (1 : ℝ) - q ≠ 0 := sub_ne_zero.mpr (fun hqq => hq hqq.symm)
  have hd : (1 : ℝ) + τ * pR ≠ 0 := ne_of_gt (by linarith)
  have h2 : q = τ * pR * (1 - q) := by
    rw [← h]
    exact (div_mul_cancel₀ q h1q).symm
  rw [eq_div_iff hd]
  linear_combination h2

/-- **Eq. (S10.15), Black-Leff operational model.** Substituting the Clark
occupancy p_R = L_T/(K_d + L_T) into q = τ·p_R/(1 + τ·p_R) gives
E/E_max = τ·L_T/(K_d + L_T(1 + τ)). -/
theorem S10_blackleff_emax {τ K_d L_T : ℝ} (hK : 0 < K_d) (hL : 0 ≤ L_T) :
    let pR := L_T / (K_d + L_T)
    τ * pR / (1 + τ * pR) = τ * L_T / (K_d + L_T * (1 + τ)) := by
  intro pR
  have hpd : pR = L_T / (K_d + L_T) := rfl
  have hKL : (0 : ℝ) < K_d + L_T := by linarith
  have hKL0 : K_d + L_T ≠ 0 := ne_of_gt hKL
  have hfac : (K_d + L_T) * (1 + τ * (L_T / (K_d + L_T)))
      = K_d + L_T * (1 + τ) := by
    field_simp
    ring
  have hfac2 : 1 + τ * (L_T / (K_d + L_T)) = (K_d + L_T * (1 + τ)) / (K_d + L_T) := by
    rw [eq_div_iff hKL0, mul_comm]
    exact hfac
  by_cases hD : K_d + L_T * (1 + τ) = 0
  · have h2 : 1 + τ * (L_T / (K_d + L_T)) = 0 := by
      by_contra hc
      have hprod : (K_d + L_T) * (1 + τ * (L_T / (K_d + L_T))) = 0 := hfac.trans hD
      rcases mul_eq_zero.mp hprod with h0 | h0
      · exact hKL0 h0
      · exact hc h0
    rw [hpd, h2, div_zero, hD, div_zero]
  · rw [hpd, hfac2, ← mul_div_assoc]
    rw [div_eq_div_iff (div_ne_zero hD hKL0) hD, div_mul_eq_mul_div, ← mul_div_assoc]

/-- **Theorem S10.8 (SI S10.6), receptor reserve, exact ratio identity.**
From (S10.12): q/(κ_eff·τ·p_R) = (1 − q)/(κ_eff + 1 − q). -/
theorem S10_8_reserve_identity {κ τ pR q : ℝ} (hκ : 0 < κ) (hq0 : 0 < q) (hq1 : q < 1)
    (h : τ * pR = q / (1 - q) + q / κ) :
    q / (κ * τ * pR) = (1 - q) / (κ + 1 - q) := by
  have h1q : (0 : ℝ) < 1 - q := by linarith
  have hκ1q : (0 : ℝ) < κ + 1 - q := by linarith
  have hκτ : κ * (τ * pR) = q * (κ + 1 - q) / (1 - q) := by
    rw [h]
    field_simp [hκ.ne', h1q.ne']
    ring
  have hassoc : κ * τ * pR = κ * (τ * pR) := by ring
  have hden : κ * τ * pR ≠ 0 := by
    have hpos : (0 : ℝ) < κ * (τ * pR) := by
      rw [hκτ]
      exact div_pos (mul_pos hq0 hκ1q) h1q
    rw [← mul_assoc] at hpos
    exact ne_of_gt hpos
  rw [div_eq_div_iff hden (ne_of_gt hκ1q), hassoc, hκτ]
  field_simp [h1q.ne']

/-- **Theorem S10.8 (SI S10.6), receptor reserve, quantitative bound.**
If q ≤ 1/2 then the reserve ratio deviates from 1 by at most 2κ_eff:
0 ≤ 1 − q/(κ_eff τ p_R) ≤ 2κ_eff. Hence q → κ_eff·τ·p_R as κ_eff → 0. -/
theorem S10_8_reserve_bound {κ τ pR q : ℝ} (hκ : 0 < κ) (hq0 : 0 < q) (hq2 : q ≤ 1 / 2)
    (h : τ * pR = q / (1 - q) + q / κ) :
    0 ≤ 1 - q / (κ * τ * pR) ∧ 1 - q / (κ * τ * pR) ≤ 2 * κ := by
  have hq1 : q < 1 := by linarith
  have hκ1q : (0 : ℝ) < κ + 1 - q := by linarith
  have hid := S10_8_reserve_identity hκ hq0 hq1 h
  have h2 : 1 - q / (κ * τ * pR) = κ / (κ + 1 - q) := by
    rw [hid]
    field_simp [ne_of_gt hκ1q]
  rw [h2]
  constructor
  · exact div_nonneg hκ.le hκ1q.le
  · rw [div_le_iff₀ hκ1q]
    nlinarith [hκ, hq2, mul_pos hκ hκ1q,
      mul_nonneg hκ.le (show (0 : ℝ) ≤ 1 / 2 - q by linarith)]

/-- **Theorem S10.10 (SI S10.7), unification skeleton.** TCS-MWC and TCS-KNF
share the skeleton ξ = x + p(x)/κ (Eq. S10.17); they differ only in p(x).
We record the skeleton as a structure so both models are literally the same map. -/
structure TcsSkeleton where
  p : ℝ → ℝ

/-- The skeleton map ξ(x) = x + p(x)/κ. -/
noncomputable def TcsSkeleton.xi (s : TcsSkeleton) (κ x : ℝ) : ℝ := x + s.p x / κ

/-- **Theorem S10.10, MWC instance.** Any MWC occupancy function yields the
skeleton (S10.16); the statement is definitionally true, which is the point:
the model enters only through p. -/
theorem S10_10_mwc_skeleton (pMWC : ℝ → ℝ) (κ x : ℝ) :
    (TcsSkeleton.mk pMWC).xi κ x = x + pMWC x / κ := rfl

/-- **Theorem S10.10, KNF instance.** Same skeleton for KNF (sequential
cooperativity factors affect only p, not the skeleton). -/
theorem S10_10_knf_skeleton (pKNF : ℝ → ℝ) (κ x : ℝ) :
    (TcsSkeleton.mk pKNF).xi κ x = x + pKNF x / κ := rfl

/-- The eTCM fractional signal, Eq. (S10.18). -/
noncomputable def signalETCM (x y α β J : ℝ) : ℝ :=
  J * (1 + x / α + y + x * y / (α * β)) /
    (1 + J + x + J * x / α + J * y + J * x * y / (α * β))

/-- **Theorem S10.11 (SI S10.8), constitutive activity.** With no ligand and no
G-protein (x = y = 0) the basal signal is J/(1 + J) > 0 whenever J > 0. -/
theorem S10_11_basal {α β J : ℝ} (hJ : 0 < J) :
    signalETCM 0 0 α β J = J / (1 + J) ∧ 0 < signalETCM 0 0 α β J := by
  have h0 : signalETCM 0 0 α β J = J / (1 + J) := by
    simp only [signalETCM, zero_div, mul_zero, zero_mul, add_zero, mul_one]
  refine ⟨h0, ?_⟩
  rw [h0]
  positivity

/-- Monotonicity helper for the eTCM signal: for `0 < J < C` the function
u ↦ J·(1 + u)/(C + J·u) is strictly increasing on `u > 0`. Cross multiplying,
the difference of the two sides is `J·(C − J)·(u₁ − u₂)`. -/
private lemma eTCM_mono_u {J C u₁ u₂ : ℝ} (hJ : 0 < J) (hC : J < C)
    (hu1 : 0 < u₁) (hu2 : 0 < u₂) (hu : u₂ < u₁) :
    J * (1 + u₂) / (C + J * u₂) < J * (1 + u₁) / (C + J * u₁) := by
  have hCpos : (0 : ℝ) < C := by linarith [hJ, hC]
  have hd1 : (0 : ℝ) < C + J * u₁ := by linarith [hCpos, mul_pos hJ hu1]
  have hd2 : (0 : ℝ) < C + J * u₂ := by linarith [hCpos, mul_pos hJ hu2]
  rw [div_lt_div_iff₀ hd2 hd1]
  have hdiff : J * (1 + u₁) * (C + J * u₂) - J * (1 + u₂) * (C + J * u₁)
      = J * (C - J) * (u₁ - u₂) := by ring
  have hpos : (0 : ℝ) < J * (C - J) * (u₁ - u₂) :=
    mul_pos (mul_pos hJ (sub_pos.mpr hC)) (sub_pos.mpr hu)
  rw [← sub_pos, hdiff]
  exact hpos

/-- **Theorem S10.11 (SI S10.8), inverse agonism.** For x > 0, y = 0 the signal
is strictly decreasing in α: larger α (inverse agonist direction) lowers S. -/
theorem S10_11_alpha {J x α₁ α₂ : ℝ} (hJ : 0 < J) (hx : 0 < x)
    (hα₁ : 0 < α₁) (hα₁₂ : α₁ < α₂) :
    signalETCM x 0 α₂ 1 J < signalETCM x 0 α₁ 1 J := by
  have hα₂ : (0 : ℝ) < α₂ := by linarith
  have hu1 : 0 < x / α₁ := div_pos hx hα₁
  have hu2 : 0 < x / α₂ := div_pos hx hα₂
  have hu : x / α₂ < x / α₁ := div_lt_div_of_pos_left hx hα₁ hα₁₂
  have hC : J < 1 + J + x := by linarith
  have hsimpl : ∀ α : ℝ, signalETCM x 0 α 1 J
      = J * (1 + x / α) / (1 + J + x + J * (x / α)) := by
    intro α
    simp only [signalETCM, mul_zero, zero_mul, zero_div, add_zero, mul_one,
      mul_div_assoc]
  rw [hsimpl α₂, hsimpl α₁]
  exact eTCM_mono_u hJ hC hu1 hu2 hu

/-- Monotonicity helper for the eTCM signal in the free G-protein:
for `B > 0` and `A < C` the function y ↦ (A + B·y)/(C + B·y) is strictly
increasing on the region where the denominator is positive. Cross multiplying,
the difference of the two sides is `B·(C − A)·(y₂ − y₁)`. -/
private lemma strictMono_ratio_linear {A B C y₁ y₂ : ℝ} (hB : 0 < B) (hAC : A < C)
    (hC1 : 0 < C + B * y₁) (hC2 : 0 < C + B * y₂) (hy : y₁ < y₂) :
    (A + B * y₁) / (C + B * y₁) < (A + B * y₂) / (C + B * y₂) := by
  rw [div_lt_div_iff₀ hC1 hC2]
  have hdiff : (A + B * y₂) * (C + B * y₁) - (A + B * y₁) * (C + B * y₂)
      = B * (C - A) * (y₂ - y₁) := by ring
  have hpos : (0 : ℝ) < B * (C - A) * (y₂ - y₁) :=
    mul_pos (mul_pos hB (sub_pos.mpr hAC)) (sub_pos.mpr hy)
  rw [← sub_pos, hdiff]
  exact hpos

/-- **Theorem S10.12 (SI S10.8), G-protein depletion reduces signal.**
The signal is strictly increasing in the free G-protein y (at x ≥ 0): any
mechanism that depletes free G (low κ_G) therefore lowers the maximal signal. -/
theorem S10_12_depleted {x y₁ y₂ α β J : ℝ} (hJ : 0 < J) (hx : 0 ≤ x)
    (hα : 0 < α) (hβ : 0 < β) (hy₁ : 0 ≤ y₁) (hy₁₂ : y₁ < y₂) :
    signalETCM x y₁ α β J < signalETCM x y₂ α β J := by
  have hα0 : α ≠ 0 := ne_of_gt hα
  have hβ0 : β ≠ 0 := ne_of_gt hβ
  set A := J * (1 + x / α) with hAd
  set B := J * (1 + x / (α * β)) with hBd
  set C := 1 + J + x + J * x / α with hCd
  have hB : (0 : ℝ) < B := by rw [hBd]; positivity
  have hCpos : (0 : ℝ) < C := by rw [hCd]; positivity
  have hAC : A < C := by
    rw [hAd, hCd]
    have hnn : (0 : ℝ) ≤ J * x / α := div_nonneg (mul_nonneg hJ.le hx) hα.le
    have hJx : J * (1 + x / α) = J + J * x / α := by field_simp; ring
    rw [hJx]
    linarith [hx, hnn]
  have hy2 : (0 : ℝ) ≤ y₂ := le_trans hy₁ hy₁₂.le
  have hden1 : (0 : ℝ) < C + B * y₁ := by
    linarith [hCpos, hB, mul_nonneg hB.le hy₁]
  have hden2 : (0 : ℝ) < C + B * y₂ := by
    linarith [hCpos, hB, mul_nonneg hB.le hy2]
  have hform : ∀ y : ℝ, signalETCM x y α β J = (A + B * y) / (C + B * y) := by
    intro y
    have hn : J * (1 + x / α + y + x * y / (α * β)) = A + B * y := by
      rw [hAd, hBd]
      field_simp
      ring
    have hd : 1 + J + x + J * x / α + J * y + J * x * y / (α * β)
        = C + B * y := by
      rw [hCd, hBd]
      field_simp
      ring
    unfold signalETCM
    rw [hn, hd]
  rw [hform y₁, hform y₂]
  exact strictMono_ratio_linear hB hAC hden1 hden2 hy₁₂

/-- **Theorem S10.13 (SI S10.9), scale degeneracy.** The scaling (S10.19)
L_T → aL_T, R_T → aR_T, K_d → aK_d leaves ξ and κ invariant, and for a ≠ 1,
K_d ≠ 0 the scaled parameters genuinely differ: (L_T, R_T, K_d) is not
identifiable from a single curve, only the groups (ξ, κ) are. -/
theorem S10_13_degeneracy {L_T R_T K_d a : ℝ} {n : ℕ}
    (hK : K_d ≠ 0) (hR : R_T ≠ 0) (ha : a ≠ 0) (ha1 : a ≠ 1) :
    a * L_T / (a * K_d) = L_T / K_d ∧
      a * K_d / ((n : ℝ) * (a * R_T)) = K_d / ((n : ℝ) * R_T) ∧
      a * K_d ≠ K_d := by
  have _ := hR
  refine ⟨mul_div_mul_left _ _ ha, ?_, ?_⟩
  · have hcong : (n : ℝ) * (a * R_T) = a * ((n : ℝ) * R_T) := by ring
    rw [hcong]
    exact mul_div_mul_left _ _ ha
  · intro h
    apply ha1
    have h' : a * K_d = 1 * K_d := by rw [h, one_mul]
    exact mul_right_cancel₀ hK h'

/-- **Theorem S10.14 (SI S10.11), scale invariance.** Under
L_T, R_T, K → s·L_T, s·R_T, s·K with s > 0, the invariants ξ = L_T/K,
κ = K/(n·R_T), x = c/K are all unchanged. -/
theorem S10_14_scale {L_T R_T K c s : ℝ} {n : ℕ}
    (hK : K ≠ 0) (hR : R_T ≠ 0) (hs : s ≠ 0) :
    s * L_T / (s * K) = L_T / K ∧
      s * K / ((n : ℝ) * (s * R_T)) = K / ((n : ℝ) * R_T) ∧
      s * c / (s * K) = c / K := by
  have _ := hK
  have _ := hR
  refine ⟨mul_div_mul_left _ _ hs, ?_, mul_div_mul_left _ _ hs⟩
  have hcong : (n : ℝ) * (s * R_T) = s * ((n : ℝ) * R_T) := by ring
  rw [hcong]
  exact mul_div_mul_left _ _ hs

/-- The binomial absorption identity i·C(n, i) = n·C(n − 1, i − 1), real form.
Proved from the factorial characterisation of the binomial coefficients. -/
private lemma choose_absorb_cast (n i : ℕ) (hi : 1 ≤ i) (hin : i ≤ n) :
    (i : ℝ) * (Nat.choose n i : ℝ) = (n : ℝ) * (Nat.choose (n - 1) (i - 1) : ℝ) := by
  have h1 : Nat.choose n i * Nat.factorial i * Nat.factorial (n - i) = Nat.factorial n :=
    Nat.choose_mul_factorial_mul_factorial hin
  have h2 : Nat.choose (n - 1) (i - 1) * Nat.factorial (i - 1) * Nat.factorial (n - i)
      = Nat.factorial (n - 1) := by
    have hsub : n - 1 - (i - 1) = n - i := by omega
    have h2' := Nat.choose_mul_factorial_mul_factorial (show i - 1 ≤ n - 1 by omega)
    rwa [hsub] at h2'
  have hfi : Nat.factorial i = i * Nat.factorial (i - 1) :=
    (Nat.mul_factorial_pred (by omega : 0 < i)).symm
  have hfn : Nat.factorial n = n * Nat.factorial (n - 1) :=
    (Nat.mul_factorial_pred (by omega : 0 < n)).symm
  have hnat : i * Nat.choose n i = n * Nat.choose (n - 1) (i - 1) := by
    have hK : (0 : ℕ) < Nat.factorial (i - 1) * Nat.factorial (n - i) :=
      Nat.mul_pos (Nat.factorial_pos _) (Nat.factorial_pos _)
    apply Nat.mul_right_cancel hK
    have e1 : i * Nat.choose n i * (Nat.factorial (i - 1) * Nat.factorial (n - i))
        = Nat.choose n i * Nat.factorial i * Nat.factorial (n - i) := by
      rw [hfi]; ring
    have e2 : n * Nat.choose (n - 1) (i - 1) * (Nat.factorial (i - 1) * Nat.factorial (n - i))
        = n * (Nat.choose (n - 1) (i - 1) * Nat.factorial (i - 1) * Nat.factorial (n - i)) := by
      ring
    rw [e1, h1, e2, h2, ← hfn]
  exact_mod_cast hnat

/-- The binomial-theorem form of the binding polynomial denominator:
∑ C(m, i)·x^i = (1 + x)^m. -/
private lemma sum_choose_pow (m : ℕ) (x : ℝ) :
    (∑ i ∈ Finset.range (m + 1), (Nat.choose m i : ℝ) * x ^ i) = (1 + x) ^ m := by
  have hp := Commute.add_pow (Commute.one_right x) m
  simp only [one_pow, mul_one] at hp
  rw [add_comm (1 : ℝ) x, hp]
  exact Finset.sum_congr rfl fun i _ => mul_comm _ _

/-- Numerator of the binding-polynomial ratio for n identical independent
sites: ∑ i·C(n, i)·x^i = n·x·(1 + x)^(n − 1). -/
private lemma sum_choose_weighted {n : ℕ} (hn : 0 < n) (x : ℝ) :
    (∑ i ∈ Finset.range (n + 1), (i : ℝ) * (Nat.choose n i : ℝ) * x ^ i)
      = (n : ℝ) * x * (1 + x) ^ (n - 1) := by
  have hn1 : 1 ≤ n := hn
  rw [Finset.sum_range_succ']
  simp only [Nat.cast_zero, zero_mul, zero_add, add_zero]
  have hterm : ∀ i ∈ Finset.range n,
      ((i + 1 : ℕ) : ℝ) * (Nat.choose n (i + 1) : ℝ) * x ^ (i + 1)
        = (n : ℝ) * x * ((Nat.choose (n - 1) i : ℝ) * x ^ i) := by
    intro i hi
    have hi1 : i + 1 ≤ n := by
      rw [Finset.mem_range] at hi; omega
    have hab := choose_absorb_cast n (i + 1) (by omega) hi1
    rw [Nat.add_sub_cancel] at hab
    rw [pow_succ, hab]
    ring
  rw [Finset.sum_congr rfl hterm, ← Finset.mul_sum]
  have hrange : Finset.range ((n - 1) + 1) = Finset.range n := by
    rw [Nat.sub_add_cancel hn1]
  rw [← hrange, sum_choose_pow (n - 1) x]

/-- **Eq. (S10.20) with binomial coefficients, limit (i).** For n independent
sites with identical K, p_i = C(n, i), the binding polynomial gives
ν = n·x/(1 + x). Numerator uses the absorption identity
i·C(n, i) = n·C(n − 1, i − 1); denominator is the binomial theorem. -/
theorem S10_14_independent {n : ℕ} (hn : 0 < n) {x : ℝ} (hx : 0 ≤ x) :
    (∑ i ∈ Finset.range (n + 1), (i : ℝ) * (Nat.choose n i : ℝ) * x ^ i)
      / (∑ i ∈ Finset.range (n + 1), (Nat.choose n i : ℝ) * x ^ i)
    = (n : ℝ) * x / (1 + x) := by
  have h1x : (0 : ℝ) < 1 + x := by linarith
  rw [sum_choose_weighted hn x, sum_choose_pow n x]
  have hpow : (1 + x : ℝ) ^ n = (1 + x) ^ (n - 1) * (1 + x) := by
    nth_rewrite 1 [← Nat.sub_add_cancel hn]
    rw [pow_succ]
  rw [hpow]
  have hnep : (1 + x : ℝ) ^ (n - 1) ≠ 0 := ne_of_gt (pow_pos h1x _)
  have hne1 : (1 : ℝ) + x ≠ 0 := ne_of_gt h1x
  field_simp [hnep, hne1]
  ring

/-- **Theorem S10.14, limit (ii), zero depletion.** If ν stays bounded in [0, n]
and f·ξ = x(κ) + ν(κ)/(n·κ) with x(κ) ≥ 0, then x(κ) → f·ξ as κ → ∞:
the Middendorf form is recovered. -/
theorem S10_14_zero_depletion {n : ℕ} (hn : 0 < n) {f ξ : ℝ} (hf : 0 < f)
    (x ν : ℝ → ℝ) (hν : ∀ κ, 0 ≤ ν κ ∧ ν κ ≤ (n : ℝ))
    (hx : ∀ κ, 0 < κ → 0 ≤ x κ ∧ f * ξ = x κ + ν κ / ((n : ℝ) * κ)) :
    Filter.Tendsto x Filter.atTop (nhds (f * ξ)) := by
  have _ := hf
  have hnf : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hub : ∀ᶠ κ in Filter.atTop, x κ ≤ f * ξ := by
    filter_upwards [eventually_gt_atTop (0 : ℝ)] with κ hκ
    obtain ⟨_, heq⟩ := hx κ hκ
    obtain ⟨hν0, _⟩ := hν κ
    have hnn : (0 : ℝ) < (n : ℝ) * κ := mul_pos hnf hκ
    have h1 : (0 : ℝ) ≤ ν κ / ((n : ℝ) * κ) := div_nonneg hν0 hnn.le
    linarith
  have hlb : ∀ᶠ κ in Filter.atTop, f * ξ - 1 / κ ≤ x κ := by
    filter_upwards [eventually_gt_atTop (0 : ℝ)] with κ hκ
    obtain ⟨_, heq⟩ := hx κ hκ
    obtain ⟨hν0, hνn⟩ := hν κ
    have hnn : (0 : ℝ) < (n : ℝ) * κ := mul_pos hnf hκ
    have h2 : ν κ / ((n : ℝ) * κ) ≤ 1 / κ := by
      rw [div_le_div_iff₀ hnn hκ]
      have h3 : (0 : ℝ) ≤ (n : ℝ) - ν κ := by linarith
      nlinarith [hνn, hκ, hnf, mul_nonneg h3 hκ.le]
    linarith
  have h1 : Filter.Tendsto (fun κ : ℝ => f * ξ - 1 / κ) Filter.atTop (𝓝 (f * ξ)) := by
    have hsub : Filter.Tendsto (fun κ : ℝ => f * ξ - κ⁻¹) Filter.atTop
        (𝓝 (f * ξ - 0)) := tendsto_const_nhds.sub tendsto_inv_atTop_zero
    rw [sub_zero] at hsub
    simpa [one_div] using hsub
  exact tendsto_of_tendsto_of_tendsto_of_le_of_le' h1 tendsto_const_nhds hlb hub

/-- **Theorem S10.14, limit (iii), strong depletion.** At fixed total input
f·ξ = c > 0, if x(κ) → 0 as κ → 0⁺ and the master equation holds, then
ν(κ)/(n·f·κ·ξ) → 1: the bound fraction is the stoichiometric ratio. -/
theorem S10_14_strong_depletion {n : ℕ} (hn : 0 < n) {f c : ℝ} (hf : 0 < f) (hc : 0 < c)
    (x ν : ℝ → ℝ)
    (h : ∀ κ, 0 < κ → f * c = x κ + ν κ / ((n : ℝ) * κ))
    (hx0 : Filter.Tendsto x (nhdsWithin 0 (Set.Ioi 0)) (nhds 0)) :
    Filter.Tendsto (fun κ => ν κ / ((n : ℝ) * f * κ * c))
      (nhdsWithin 0 (Set.Ioi 0)) (nhds 1) := by
  have hnf : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hfc : (0 : ℝ) < f * c := mul_pos hf hc
  have heq : (fun κ => ν κ / ((n : ℝ) * f * κ * c)) =ᶠ[nhdsWithin 0 (Set.Ioi 0)]
      (fun κ => 1 - x κ / (f * c)) := by
    filter_upwards [self_mem_nhdsWithin] with κ hκ
    have hκ' : (0 : ℝ) < κ := hκ
    have he := h κ hκ'
    have hnk : (0 : ℝ) < (n : ℝ) * κ := mul_pos hnf hκ'
    have hνeq : ν κ = (f * c - x κ) * ((n : ℝ) * κ) := by
      have he2 : ν κ / ((n : ℝ) * κ) = f * c - x κ := by linarith [he]
      rw [div_eq_iff (ne_of_gt hnk)] at he2
      exact he2
    have hnfkc : (0 : ℝ) < (n : ℝ) * f * κ * c := by positivity
    rw [hνeq, div_eq_iff (ne_of_gt hnfkc)]
    have h3 : x κ / (f * c) * ((n : ℝ) * f * κ * c) = x κ * ((n : ℝ) * κ) := by
      rw [div_mul_eq_mul_div, div_eq_iff (ne_of_gt hfc)]
      ring
    rw [sub_mul, sub_mul, one_mul, h3]
    ring
  have hlim : Filter.Tendsto (fun κ => 1 - x κ / (f * c)) (nhdsWithin 0 (Set.Ioi 0))
      (𝓝 (1 - 0 / (f * c))) :=
    tendsto_const_nhds.sub (hx0.div_const (f * c))
  rw [zero_div, sub_zero] at hlim
  exact Tendsto.congr' heq.symm hlim

end TCS
