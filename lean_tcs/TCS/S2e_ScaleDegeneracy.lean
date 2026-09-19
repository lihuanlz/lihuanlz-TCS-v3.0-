/-
=====================================================================
S2e: Scale degeneracy and cross-platform incomparability
=====================================================================
Corresponds to Section S2e of the SI ("Supplementary and additional
Material"):
  - Theorem S2e.2 (scale invariance): if (M,Ω,K) satisfies the master
    equation with occupancy p, then for every s the rescaled system
    {sM, sΩ, sK} satisfies the master equation with the SAME p
  - Theorem S2e.3 (scale degeneracy): the one-parameter continuum
    {sM, sΩ, sK}_{s>0} all give the same readout p, so a single
    equilibrium measurement cannot tell them apart
  - M is not identifiable: for s ≠ 1 and M ≠ 0 we have sM ≠ M, yet the
    readout is identical
  - Algebraic core of Theorem S2e.4 (cross-platform incomparability):
    the same readout p on two platforms with different κ corresponds
    to different molecular loadings ξ; without knowing κ, raw readouts
    cannot be compared across platforms
  - "The only knowable combination": the master equation yields only
    ξ − p/κ = p/(1−p); ξ (the molecule number) and κ are not
    separately knowable

Lean 4 + Mathlib, toolchain leanprover/lean4:v4.15.0
-/

import Mathlib

namespace TCS

/-- Theorem S2e.2 (scale invariance). -/
theorem theorem_S2e2 {M Ω K W p s : ℝ}
    (h : M = Ω * p + K * W * (p / (1 - p))) :
    s * M = (s * Ω) * p + (s * K) * W * (p / (1 - p)) := by
  rw [h]
  ring

/-- Theorem S2e.3 (scale degeneracy): the entire continuum s > 0 shares
the same readout p. -/
theorem theorem_S2e3 {M Ω K W p : ℝ}
    (h : M = Ω * p + K * W * (p / (1 - p))) :
    ∀ s : ℝ, 0 < s → s * M = (s * Ω) * p + (s * K) * W * (p / (1 - p)) := by
  intro s _
  exact theorem_S2e2 h

/-- M is not identifiable: for s ≠ 1 and M ≠ 0 we have s·M ≠ M, yet by
Theorem S2e.2 the two systems give exactly the same readout. -/
theorem M_not_identifiable {M s : ℝ} (hM : M ≠ 0) (hs : s ≠ 1) : s * M ≠ M := by
  intro h
  have h0 : (s - 1) * M = 0 := by linear_combination h
  rcases mul_eq_zero.mp h0 with h1 | h1
  · exact hs (eq_of_sub_eq_zero h1)
  · exact hM h1

/-- Algebraic core of Theorem S2e.4 (cross-platform incomparability):
two platforms read out the same occupancy p (0 < p < 1), but when
κ₁ ≠ κ₂ their molecular loadings differ, ξ₁ ≠ ξ₂. Hence raw readouts
cannot be compared directly across platforms. -/
theorem theorem_S2e4_core {p κ₁ κ₂ ξ₁ ξ₂ : ℝ} (hp0 : 0 < p) (_hp1 : p < 1)
    (hκ : κ₁ ≠ κ₂) (_hκ₁ : κ₁ ≠ 0) (_hκ₂ : κ₂ ≠ 0)
    (h1 : ξ₁ = p / (1 - p) + p / κ₁) (h2 : ξ₂ = p / (1 - p) + p / κ₂) :
    ξ₁ ≠ ξ₂ := by
  intro hξ
  have hpk : p / κ₁ = p / κ₂ := by linarith [h1, h2, hξ]
  have hinv : κ₁⁻¹ = κ₂⁻¹ := by
    rw [div_eq_mul_inv, div_eq_mul_inv] at hpk
    exact mul_left_cancel₀ (ne_of_gt hp0) hpk
  have : κ₁ = κ₂ := by
    rw [← inv_inv κ₁, ← inv_inv κ₂, hinv]
  exact hκ this

/-- "The only knowable combination": from the master equation only the
single combination ξ − p/κ = p/(1−p) can be read off; ξ and κ are not
individually knowable (a direct corollary of scale degeneracy). -/
theorem only_combination {ξv κv p : ℝ}
    (h : ξv = p / (1 - p) + p / κv) :
    ξv - p / κv = p / (1 - p) := by
  rw [h]
  ring

end TCS
