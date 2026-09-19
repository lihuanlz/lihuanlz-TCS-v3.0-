/-
=====================================================================
S2d: Asymptotic behaviour of the master equation
=====================================================================
Corresponds to Section S2d of the SI ("Supplementary and additional
Material"):
  - κ → ∞ (fixed ξ): p → ξ/(1+ξ), recovering the Langmuir
    zero-depletion limit
    (squeezing: g(p) = p/(1−p) = ξ − p/κ ∈ (ξ − 1/κ, ξ),
     then compose with the continuity of u ↦ u/(1+u))
  - κ → 0⁺ (fixed ξ): p → 0, with p ≤ κξ (the strong-depletion
    regime)
  - ξ → ∞ (fixed κ): lower bound p ≥ (ξ − 1/κ)/(1 + ξ − 1/κ) → 1
    (full occupancy)

Here p(κ) is uniquely determined by the master equation (Theorem S2e.5
guarantees well-definedness); this file states the limits with p as a
function constrained by the master equation.

Lean 4 + Mathlib, toolchain leanprover/lean4:v4.15.0
-/

import Mathlib

open Filter Topology Set

namespace TCS

/-- Langmuir limit as κ → ∞: for fixed ξ > 0, if for every κ > 0 the
value p(κ) ∈ (0,1) satisfies the master equation ξ = p/(1−p) + p/κ,
then p(κ) → ξ/(1+ξ). -/
theorem tendsto_langmuir {ξv : ℝ} (hξ : (0:ℝ) < ξv) (p : ℝ → ℝ)
    (hp : ∀ κv, 0 < κv → 0 < p κv ∧ p κv < 1 ∧
      ξv = p κv / (1 - p κv) + p κv / κv) :
    Tendsto p atTop (𝓝 (ξv / (1 + ξv))) := by
  -- Step 1: squeezing bounds ξ − 1/κ < g < ξ for g(κ) = p/(1−p)
  have hgb : ∀ κv, 0 < κv →
      ξv - 1 / κv < p κv / (1 - p κv) ∧ p κv / (1 - p κv) < ξv := by
    intro κv hκ
    obtain ⟨hp0, hp1, hpeq⟩ := hp κv hκ
    have h1m : (0:ℝ) < 1 - p κv := by linarith
    have hlt1 : p κv / κv < 1 / κv := by rwa [div_lt_div_iff_of_pos_right hκ]
    have hpκ : (0:ℝ) < p κv / κv := div_pos hp0 hκ
    constructor
    · rw [hpeq]
      linarith [hlt1]
    · rw [hpeq]
      linarith [hpκ]
  -- Step 2: squeezing gives g → ξ
  have hglim : Tendsto (fun κv => p κv / (1 - p κv)) atTop (𝓝 ξv) := by
    have h1 : Tendsto (fun κv : ℝ => ξv - 1 / κv) atTop (𝓝 (ξv - 0)) := by
      have hi : Tendsto (fun κv : ℝ => κv⁻¹) atTop (𝓝 (0:ℝ)) :=
        tendsto_inv_atTop_zero
      simpa [one_div] using tendsto_const_nhds.sub hi
    rw [sub_zero] at h1
    exact tendsto_of_tendsto_of_tendsto_of_le_of_le' h1 tendsto_const_nhds
      (by filter_upwards [eventually_gt_atTop (0:ℝ)] with κv hκ
          using (hgb κv hκ).1.le)
      (by filter_upwards [eventually_gt_atTop (0:ℝ)] with κv hκ
          using (hgb κv hκ).2.le)
  -- Step 3: the algebraic identity p = g/(1+g)
  have hpg : ∀ᶠ κv in atTop,
      p κv = (p κv / (1 - p κv)) / (1 + p κv / (1 - p κv)) := by
    filter_upwards [eventually_gt_atTop (0:ℝ)] with κv hκ
    obtain ⟨hp0, hp1, _⟩ := hp κv hκ
    have h1m : (1:ℝ) - p κv ≠ 0 := ne_of_gt (by linarith)
    have h2 : 1 + p κv / (1 - p κv) = 1 / (1 - p κv) := by
      field_simp
    rw [h2, div_eq_mul_inv, inv_div, div_one]
    exact (div_mul_cancel₀ _ h1m).symm
  -- Step 4: u ↦ u/(1+u) is continuous at u = ξ; compose
  have hcont : Tendsto (fun u : ℝ => u / (1 + u)) (𝓝 ξv) (𝓝 (ξv / (1 + ξv))) :=
    tendsto_id.div (tendsto_const_nhds.add tendsto_id)
      (ne_of_gt (by linarith : (0:ℝ) < 1 + ξv))
  have hcomp : Tendsto
      (fun κv => (p κv / (1 - p κv)) / (1 + p κv / (1 - p κv)))
      atTop (𝓝 (ξv / (1 + ξv))) := hcont.comp hglim
  have hpg' : ∀ᶠ κv in atTop,
      (p κv / (1 - p κv)) / (1 + p κv / (1 - p κv)) = p κv :=
    hpg.mono fun _ h => h.symm
  exact Tendsto.congr' hpg' hcomp

/-- Strong-depletion limit as κ → 0⁺: for fixed ξ > 0, p(κ) → 0,
and the squeezing bound 0 < p(κ) ≤ κξ holds pointwise. -/
theorem tendsto_zero_depletion {ξv : ℝ} (_hξ : (0:ℝ) < ξv) (p : ℝ → ℝ)
    (hp : ∀ κv, 0 < κv → 0 < p κv ∧ p κv < 1 ∧
      ξv = p κv / (1 - p κv) + p κv / κv) :
    Tendsto p (𝓝[>] 0) (𝓝 0) := by
  have hub : ∀ᶠ κv in 𝓝[>] (0:ℝ), p κv ≤ κv * ξv := by
    filter_upwards [self_mem_nhdsWithin] with κv hκ
    obtain ⟨hp0, hp1, hpeq⟩ := hp κv hκ
    have hgpos : (0:ℝ) < p κv / (1 - p κv) := div_pos hp0 (by linarith)
    have hlt : p κv / κv < ξv := by linarith [hpeq, hgpos]
    rw [div_lt_iff₀ hκ] at hlt
    linarith [hlt]
  have hlb : ∀ᶠ κv in 𝓝[>] (0:ℝ), (0:ℝ) ≤ p κv := by
    filter_upwards [self_mem_nhdsWithin] with κv hκ
    exact (hp κv hκ).1.le
  have h0 : Tendsto (fun κv : ℝ => κv * ξv) (𝓝[>] 0) (𝓝 0) := by
    have h : Tendsto (fun κv : ℝ => κv * ξv) (𝓝[>] 0) (𝓝 ((0:ℝ) * ξv)) := by
      have hid : Tendsto (id : ℝ → ℝ) (𝓝[>] (0:ℝ)) (𝓝 (0:ℝ)) :=
        tendsto_id.mono_left nhdsWithin_le_nhds
      have hc : Tendsto (fun _ : ℝ => ξv) (𝓝[>] (0:ℝ)) (𝓝 ξv) := tendsto_const_nhds
      exact hid.mul hc
    rwa [zero_mul] at h
  exact tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds h0 hlb hub

/-- Full-occupancy lower bound as ξ → ∞: for fixed κ > 0, whenever
ξ > 1/κ we have p ≥ (ξ − 1/κ)/(1 + ξ − 1/κ), and the right-hand side
tends to 1 as ξ → ∞, hence p → 1. -/
theorem occupancy_lower_bound {κv ξv p : ℝ} (hκ : 0 < κv) (hξ : 1 / κv < ξv)
    (hp0 : 0 < p) (hp1 : p < 1)
    (h : ξv = p / (1 - p) + p / κv) :
    (ξv - 1 / κv) / (1 + (ξv - 1 / κv)) ≤ p := by
  have h1m : (0:ℝ) < 1 - p := by linarith
  have hpκ : p / κv ≤ 1 / κv := by
    rw [div_le_div_iff_of_pos_right hκ]
    exact hp1.le
  have hg : ξv - 1 / κv ≤ p / (1 - p) := by linarith [h, hpκ]
  have hApos : (0:ℝ) < ξv - 1 / κv := by linarith [hξ]
  have hA : (0:ℝ) < 1 + (ξv - 1 / κv) := by linarith
  have hgpos : (0:ℝ) < p / (1 - p) := div_pos hp0 h1m
  have hB : (0:ℝ) < 1 + p / (1 - p) := by linarith
  -- u ↦ u/(1+u) is increasing on u > −1: cross-multiplication gives
  -- exactly hg
  have hmono : (ξv - 1 / κv) / (1 + (ξv - 1 / κv)) ≤
      (p / (1 - p)) / (1 + p / (1 - p)) := by
    rw [div_le_div_iff₀ hA hB]
    nlinarith [hg, hApos, hgpos, mul_pos hApos hgpos]
  have hfin : (p / (1 - p)) / (1 + p / (1 - p)) = p := by
    have h1m' : (1:ℝ) - p ≠ 0 := ne_of_gt h1m
    have h2 : 1 + p / (1 - p) = 1 / (1 - p) := by
      field_simp
    rw [h2, div_eq_mul_inv, inv_div, div_one]
    exact div_mul_cancel₀ _ h1m'
  rw [hfin] at hmono
  exact hmono

end TCS
