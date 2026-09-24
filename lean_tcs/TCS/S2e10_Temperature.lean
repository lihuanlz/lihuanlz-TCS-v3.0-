/-
=====================================================================
S2e.10: Temperature protocols — scope and limits of symmetry breaking
=====================================================================
Corresponds to Remark S2e.10 of the SI (upgraded three-subsection
version, S2e.10.1–S2e.10.3):

  - S2e.10.1 (occupancy readout): at ANY temperature factor g, the
    pair (ξ, κ) is invariant under (M, Ω, K₀) → (sM, sΩ, sK₀); the
    van 't Hoff shift K_j = K₀·g_j is multiplicative and rides along
    with the same rescaling at every temperature
  - S2e.10.2 (digital readout): Y = M/(1 + κ₀g) is NOT scale
    invariant. Algebraic cores: ΔH = 0 ⇒ g = 1 at every temperature
    (degeneracy persists); two-temperature inversion recovers M in
    closed form; κ₀ = 0 recovers the dPCR limit of Theorem S2c.4.1
  - The analytic content of the determinant condition (Eq. S2e.10b) —
    strict monotonicity of exp with distinct temperatures implies
    det J ≠ 0 — is deliberately NOT formalized here (analysis-level
    statement; everything below is fully checked by the kernel).

Lean 4 + Mathlib, toolchain leanprover/lean4:v4.15.0
-/

import TCS.S1d_ScaleGroup   -- reuses `kappa` (K·W/Ω) and `xi` (M/(K·W)), Definition S2e.2 / S2e.3

namespace TCS

/-- van 't Hoff affinity factor (Eq. S2e.10a):
g(ΔH, T, T₀) = exp(−(ΔH/R)·(1/T − 1/T₀)). -/
noncomputable def vantHoff (ΔH R T T₀ : ℝ) : ℝ :=
  Real.exp (-(ΔH / R) * (1 / T - 1 / T₀))

/-- Digital observable at one temperature: Y = M / (1 + κ₀·g). -/
noncomputable def Ydig (M κ₀ g : ℝ) : ℝ := M / (1 + κ₀ * g)

/-- S2e.10.1 (analog temperature invariance): at any temperature
factor g ≠ 0, both ξ and κ computed at K = K₀·g are unchanged under
(M, Ω, K₀) → (sM, sΩ, sK₀). The degeneracy survives every
multi-temperature occupancy protocol. -/
theorem analog_temperature_invariance {M Ω K₀ W s g : ℝ}
    (hW : W ≠ 0) (hΩ : Ω ≠ 0) (hK : K₀ ≠ 0) (hs : s ≠ 0) (hg : g ≠ 0) :
    xi (s * M) (s * K₀ * g) W = xi M (K₀ * g) W ∧
    kappa (s * K₀ * g) W (s * Ω) = kappa (K₀ * g) W Ω := by
  have hsKW : s * K₀ * g * W ≠ 0 := mul_ne_zero (mul_ne_zero (mul_ne_zero hs hK) hg) hW
  have hKW : K₀ * g * W ≠ 0 := mul_ne_zero (mul_ne_zero hK hg) hW
  have hsΩ : s * Ω ≠ 0 := mul_ne_zero hs hΩ
  constructor
  · simp only [xi]; field_simp; ring
  · simp only [kappa]; field_simp; ring

/-- S2e.10.2 (failure mode, ΔH = 0): if the van 't Hoff enthalpy
vanishes then g = 1 at every temperature; all temperatures give the
same observable and the degeneracy persists. -/
theorem degeneracy_dH_zero (R T T₀ : ℝ) :
    vantHoff 0 R T T₀ = 1 := by
  simp [vantHoff]

/-- S2e.10.2 (two-temperature inversion): with Y₀ measured at the
reference temperature (g = 1) and Y₁ at a second temperature with
g₁ ≠ 1, the molecule number is recovered in closed form,
M = Y₀·Y₁·(g₁ − 1) / (g₁·Y₁ − Y₀). -/
theorem two_temp_inversion {M κ₀ g₁ : ℝ}
    (h1 : 1 + κ₀ ≠ 0) (h2 : 1 + κ₀ * g₁ ≠ 0)
    (hden : g₁ * Ydig M κ₀ g₁ - Ydig M κ₀ 1 ≠ 0) :
    M = Ydig M κ₀ 1 * Ydig M κ₀ g₁ * (g₁ - 1) /
        (g₁ * Ydig M κ₀ g₁ - Ydig M κ₀ 1) := by
  have e0 : Ydig M κ₀ 1 * (1 + κ₀) = M := by
    simp only [Ydig, mul_one]
    exact div_mul_cancel₀ M h1
  have e1 : Ydig M κ₀ g₁ * (1 + κ₀ * g₁) = M := div_mul_cancel₀ M h2
  rw [eq_div_iff hden]
  linear_combination -g₁ * Ydig M κ₀ g₁ * e0 + Ydig M κ₀ 1 * e1

/-- S2e.10.2 (κ₀ recovery): once M is known, the reference-temperature
depletion factor follows from the reference-temperature readout,
κ₀ = M/Y₀ − 1. -/
theorem kappa0_recovery {M κ₀ : ℝ} (hM : M ≠ 0) (h : 1 + κ₀ ≠ 0) :
    κ₀ = M / Ydig M κ₀ 1 - 1 := by
  have hY : Ydig M κ₀ 1 ≠ 0 := div_ne_zero hM (by simpa using h)
  simp only [Ydig, mul_one] at hY ⊢
  field_simp

/-- S2e.10.2 (dPCR limit, consistency with Theorem S2c.4.1): at
κ₀ = 0 the digital observable is M itself, at every temperature. -/
theorem kappa_zero_limit (M g : ℝ) : Ydig M 0 g = M := by
  simp [Ydig]

/-- Optional helper: positive enthalpy and heating give g > 1.
Not load-bearing for the SI text. -/
theorem vantHoff_gt_one {ΔH R T T₀ : ℝ} (hΔH : 0 < ΔH) (hR : 0 < R)
    (hT₀ : 0 < T₀) (hTT : T₀ < T) : 1 < vantHoff ΔH R T T₀ := by
  simp only [vantHoff, Real.one_lt_exp_iff]
  -- −(ΔH/R)(1/T − 1/T₀) > 0  ⟺  1/T < 1/T₀  ⟺  T > T₀
  have key : -(ΔH / R) * (1 / T - 1 / T₀) = (ΔH / R) * (1 / T₀ - 1 / T) := by ring
  rw [key]
  exact mul_pos (div_pos hΔH hR) (sub_pos.mpr (one_div_lt_one_div_of_lt hT₀ hTT))

end TCS
