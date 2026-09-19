/-
=====================================================================
S1d: The scale-group action, orthogonality, and uniqueness of
separation by (ξ, κ)
=====================================================================
Corresponds to Section S1d of the SI ("Supplementary and additional
Material"):
  - Definition S2e.2: κ ≡ K·W/Ω  (W = V·N_A)
  - Definition S2e.3: ξ ≡ M/(K·W)
  - Theorem S2e.6: ξ and κ are invariant under the common rescaling
    {M,Ω,K} → {sM,sΩ,sK}
  - Table S1d.1 (orthogonality): coating (M fixed, Ω and K both
    multiplied by s) moves ξ only, leaving κ unchanged; dilution
    (Ω fixed, M and K both multiplied by s) moves κ only, leaving
    ξ unchanged
  - Table S1d.2 (counterexample): the coordinate pair (M/Ω, κ) is
    multiplied by s in BOTH coordinates under dilution, so it cannot
    separate sample from platform
  - Uniqueness of separation (the algebraic core of Theorem S1d.3):
    if two triples (M,Ω,K) give the same (ξ,κ), they differ only by
    a common scaling s > 0

Lean 4 + Mathlib, toolchain leanprover/lean4:v4.15.0
-/

import Mathlib

namespace TCS

/-- Definition S2e.2: κ ≡ K·W/Ω, with W = V·N_A. -/
noncomputable def kappa (K W Ω : ℝ) : ℝ := K * W / Ω

/-- Definition S2e.3: ξ ≡ M/(K·W). -/
noncomputable def xi (M K W : ℝ) : ℝ := M / (K * W)

/-- Theorem S2e.6 (i): κ is invariant under {K,Ω} → {sK,sΩ}. -/
theorem kappa_scale {K W Ω s : ℝ} (hΩ : Ω ≠ 0) (_hs : s ≠ 0)
    (hsΩ : s * Ω ≠ 0) :
    kappa (s * K) W (s * Ω) = kappa K W Ω := by
  simp only [kappa]
  field_simp
  ring

/-- Theorem S2e.6 (ii): ξ is invariant under {M,K} → {sM,sK}. -/
theorem xi_scale {M K W s : ℝ} (hKW : K * W ≠ 0) (_hs : s ≠ 0)
    (hsKW : s * K * W ≠ 0) :
    xi (s * M) (s * K) W = xi M K W := by
  simp only [xi]
  field_simp
  ring

/-- Table S1d.1 (coating): with M fixed and Ω, K both multiplied by s,
κ does not change. -/
theorem coating_kappa {K W Ω s : ℝ} (hΩ : Ω ≠ 0) (hs : s ≠ 0)
    (hsΩ : s * Ω ≠ 0) :
    kappa (s * K) W (s * Ω) = kappa K W Ω :=
  kappa_scale hΩ hs hsΩ

/-- Table S1d.1 (coating): ξ becomes 1/s of its original value (this
coordinate is moved alone). -/
theorem coating_xi {M K W s : ℝ} (_hKW : K * W ≠ 0) (_hs : s ≠ 0)
    (_hsKW : s * K * W ≠ 0) :
    xi M (s * K) W = xi M K W / s := by
  simp only [xi]
  rw [div_div, mul_assoc, mul_comm s (K * W)]

/-- Table S1d.1 (dilution): with Ω fixed and M, K both multiplied by s,
ξ does not change. -/
theorem dilution_xi {M K W s : ℝ} (hKW : K * W ≠ 0) (hs : s ≠ 0)
    (hsKW : s * K * W ≠ 0) :
    xi (s * M) (s * K) W = xi M K W :=
  xi_scale hKW hs hsKW

/-- Table S1d.1 (dilution): κ becomes s times its original value (this
coordinate is moved alone). -/
theorem dilution_kappa {K W Ω s : ℝ} (hΩ : Ω ≠ 0) :
    kappa (s * K) W Ω = s * kappa K W Ω := by
  simp only [kappa]
  field_simp
  ring

/-- Table S1d.2 (counterexample): under dilution, the coordinate pair
(M/Ω, κ) is multiplied by s in both coordinates at once, so (M/Ω, κ)
cannot separate sample from platform the way (ξ, κ) does. -/
theorem dilution_coupled {M K W Ω s : ℝ} (hΩ : Ω ≠ 0) :
    (s * M) / Ω = s * (M / Ω) ∧ kappa (s * K) W Ω = s * kappa K W Ω := by
  constructor
  · field_simp
  · exact dilution_kappa hΩ

/-- Uniqueness of separation (the algebraic core of Theorem S1d.3):
if two parameter triples (M₁,Ω₁,K₁) and (M₂,Ω₂,K₂) give the same (ξ,κ),
then there exists s > 0 such that M₂ = s·M₁, Ω₂ = s·Ω₁, K₂ = s·K₁;
in other words, (ξ,κ) quotients the parameter space exactly onto the
common-scaling orbits. -/
theorem separation {M₁ M₂ Ω₁ Ω₂ K₁ K₂ W : ℝ}
    (hM₁ : (0:ℝ) < M₁) (hM₂ : (0:ℝ) < M₂)
    (hK₁ : K₁ ≠ 0) (hΩ₁ : Ω₁ ≠ 0) (hΩ₂ : Ω₂ ≠ 0) (hW : W ≠ 0)
    (hxi : M₁ / (K₁ * W) = M₂ / (K₂ * W))
    (hκ : K₁ * W / Ω₁ = K₂ * W / Ω₂) :
    ∃ s : ℝ, 0 < s ∧ M₂ = s * M₁ ∧ Ω₂ = s * Ω₁ ∧ K₂ = s * K₁ := by
  have hM₁0 : M₁ ≠ 0 := ne_of_gt hM₁
  have hK₁W : K₁ * W ≠ 0 := mul_ne_zero hK₁ hW
  have hK₂ : K₂ ≠ 0 := by
    by_contra h
    rw [h] at hxi
    simp only [zero_mul, div_zero] at hxi
    rw [div_eq_zero_iff] at hxi
    rcases hxi with h1 | h1
    · exact absurd h1 hM₁0
    · exact absurd h1 hK₁W
  have hK₂W : K₂ * W ≠ 0 := mul_ne_zero hK₂ hW
  have hxi' : M₁ * (K₂ * W) = M₂ * (K₁ * W) := by
    field_simp at hxi
    linarith [hxi]
  have hκ' : K₁ * W * Ω₂ = K₂ * W * Ω₁ := by
    field_simp at hκ
    linarith [hκ]
  refine ⟨M₂ / M₁, div_pos hM₂ hM₁, by rw [div_mul_cancel₀ _ hM₁0], ?_, ?_⟩
  · -- Ω₂ = (M₂/M₁)·Ω₁: eliminate K₂ using hκ' and hxi'
    have key : Ω₂ * M₁ * (K₁ * W) = M₂ * Ω₁ * (K₁ * W) := by
      linear_combination M₁ * hκ' + Ω₁ * hxi'
    have hΩ := mul_right_cancel₀ hK₁W key
    field_simp
    linarith [hΩ]
  · -- K₂ = (M₂/M₁)·K₁: eliminate W using hxi'
    have key : (K₂ * M₁ - M₂ * K₁) * W = 0 := by
      linear_combination hxi'
    have h0 : K₂ * M₁ = M₂ * K₁ := by
      rcases mul_eq_zero.mp key with h1 | h1
      · linarith [h1]
      · exact absurd h1 hW
    field_simp
    linarith [h0]

end TCS
