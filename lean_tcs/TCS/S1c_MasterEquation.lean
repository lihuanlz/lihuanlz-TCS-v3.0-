/-
=====================================================================
S1c / S2e: Core theorems of the master equation
=====================================================================
Corresponds to the SI ("Supplementary and additional Material"):
  - Theorem S2e.1: from Axiom S2e.1 (the Langmuir relation
    p/(1-p) = C_free/K) and Axiom S2e.2 (mass conservation
    M = C_free·W + Ω·p, W = V·N_A), eliminating C_free gives
    M = Ω·p + K·W·p/(1-p)
  - Eq. (S2e.4) / (S1c.12): the dimensionless master equation
    ξ = p/(1-p) + p/κ
  - Quadratic form: p² − (κξ+κ+1)·p + κξ = 0
  - The right-hand side is strictly monotone on (0,1)
  - Theorem S2e.5: for κ>0 and ξ>0 there exists a unique p∈(0,1)
    satisfying the master equation, and it is given by the smaller
    quadratic root

Lean 4 + Mathlib, toolchain leanprover/lean4:v4.15.0
-/

import Mathlib

open Set

namespace TCS

/-- Theorem S2e.1 (elimination theorem).
From the Langmuir relation (Axiom S2e.1) and mass conservation
(Axiom S2e.2, W = V·N_A), eliminate the free concentration C to obtain
the master equation M = Ω·p + K·W·p/(1−p). -/
theorem theorem_S2e1 {p C K W Ω M : ℝ} (hp : (1:ℝ) - p ≠ 0) (hK : K ≠ 0)
    (hlangmuir : p / (1 - p) = C / K) (hmass : M = C * W + Ω * p) :
    M = Ω * p + K * W * (p / (1 - p)) := by
  have hCK : C * (1 - p) = K * p := by
    field_simp at hlangmuir
    linear_combination -hlangmuir
  rw [hmass]
  field_simp
  linear_combination W * hCK

/-- Derivation of Eq. (S2e.4): divide both sides of the master equation by
K·W to get M/(K·W) = p/(1−p) + p·Ω/(K·W), i.e. ξ = p/(1−p) + p/κ. -/
theorem master_dimensionless {M Ω K W p : ℝ} (hKW : K * W ≠ 0) (_hΩ : Ω ≠ 0)
    (hp : (1:ℝ) - p ≠ 0)
    (h : M = Ω * p + K * W * (p / (1 - p))) :
    M / (K * W) = p / (1 - p) + (p * Ω) / (K * W) := by
  rw [h]
  field_simp
  ring

/-- p/(K·W/Ω) = p·Ω/(K·W): bridge to the κ = K·W/Ω form (Definition S2e.2). -/
theorem kappa_form {p K W Ω κv : ℝ} (hκ : κv = K * W / Ω) (_hΩ : Ω ≠ 0)
    (hKW : K * W ≠ 0) :
    p / κv = (p * Ω) / (K * W) := by
  rw [hκ]
  field_simp

/-- Finite-Ω correction: when mass conservation reads
M = (Ω−1)·p + K·W·p/(1−p), the dimensionless form becomes
ξ = p/(1−p) + (p/κ)·(1 − 1/Ω). -/
theorem finite_omega_correction {M Ω K W p : ℝ} (hKW : K * W ≠ 0) (_hΩ : Ω ≠ 0)
    (hp : (1:ℝ) - p ≠ 0)
    (h : M = (Ω - 1) * p + K * W * (p / (1 - p))) :
    M / (K * W) = p / (1 - p) + (p * Ω) / (K * W) * (1 - 1 / Ω) := by
  rw [h]
  field_simp
  ring

/-- Master equation ⟺ quadratic form: ξ = p/(1−p) + p/κ holds if and only if
p² − (κξ+κ+1)·p + κξ = 0 (multiply both sides by (1−p)·κ ≠ 0). -/
theorem master_quadratic {p κv ξv : ℝ} (hp : (1:ℝ) - p ≠ 0) (hκ : κv ≠ 0) :
    ξv = p / (1 - p) + p / κv ↔
      p ^ 2 - (κv * ξv + κv + 1) * p + κv * ξv = 0 := by
  have key : (p / (1 - p) + p / κv) * ((1 - p) * κv) = p * κv + p * (1 - p) := by
    field_simp
  constructor
  · intro h
    have h2 : ξv * ((1 - p) * κv) = p * κv + p * (1 - p) := by
      rw [h]; exact key
    linear_combination h2
  · intro h
    have hX : ((1:ℝ) - p) * κv ≠ 0 := mul_ne_zero hp hκ
    have h2 : ξv * ((1 - p) * κv) = p * κv + p * (1 - p) := by
      linear_combination h
    have h3 : ξv * ((1 - p) * κv) = (p / (1 - p) + p / κv) * ((1 - p) * κv) := by
      rw [h2, key]
    exact mul_right_cancel₀ hX h3

/-- Strict monotonicity: for κ > 0, f(p) = p/(1−p) + p/κ is strictly
increasing on (0,1). (After cross-multiplication the difference equals
(b−a)·(κ + (1−a)(1−b)) / [(1−a)(1−b)κ] > 0.) -/
theorem master_strictMono {κv : ℝ} (hκ : 0 < κv) :
    StrictMonoOn (fun p : ℝ => p / (1 - p) + p / κv) (Set.Ioo (0:ℝ) 1) := by
  intro a ha b hb hab
  have ha1 : (0:ℝ) < 1 - a := by linarith [ha.2]
  have hb1 : (0:ℝ) < 1 - b := by linarith [hb.2]
  have ha1n : (1:ℝ) - a ≠ 0 := ne_of_gt ha1
  have hb1n : (1:ℝ) - b ≠ 0 := ne_of_gt hb1
  have hκn : κv ≠ 0 := ne_of_gt hκ
  have hba : (0:ℝ) < b - a := sub_pos.mpr hab
  have hpos2 : (0:ℝ) < κv + (1 - a) * (1 - b) := by positivity
  have key : (b / (1 - b) + b / κv) - (a / (1 - a) + a / κv)
      = (b - a) * (κv + (1 - a) * (1 - b)) / ((1 - a) * (1 - b) * κv) := by
    field_simp
    ring
  apply lt_of_sub_pos
  show (0:ℝ) < (b / (1 - b) + b / κv) - (a / (1 - a) + a / κv)
  rw [key]
  exact div_pos (mul_pos hba hpos2) (by positivity)

/-- Theorem S2e.5 (existence and uniqueness of the occupancy).
For κ > 0 and ξ > 0, there exists a unique p ∈ (0,1) satisfying the master
equation ξ = p/(1−p) + p/κ; this p is the smaller root (B − √d)/2 of the
quadratic p² − (κξ+κ+1)p + κξ = 0, where B = κξ+κ+1 and
d = B² − 4κξ = (κξ−1)² + κ² + 2κ²ξ + 2κ > 0. -/
theorem theorem_S2e5 {κv ξv : ℝ} (hκ : 0 < κv) (hξ : 0 < ξv) :
    ∃! p : ℝ, p ∈ Set.Ioo (0:ℝ) 1 ∧ ξv = p / (1 - p) + p / κv := by
  set B := κv * ξv + κv + 1 with hB
  have hBpos : 0 < B := by nlinarith [mul_pos hκ hξ, hκ]
  set d := B ^ 2 - 4 * κv * ξv with hd
  have hd' : d = (κv * ξv - 1) ^ 2 + κv ^ 2 + 2 * (κv * (κv * ξv)) + 2 * κv := by
    rw [hd, hB]; ring
  have hdpos : 0 < d := by
    rw [hd']
    positivity
  have hdlt : d < B ^ 2 := by
    rw [hd]
    linarith [mul_pos hκ hξ]
  set s := Real.sqrt d with hs
  have hs2 : s ^ 2 = d := Real.sq_sqrt (le_of_lt hdpos)
  have hspos : 0 < s := Real.sqrt_pos_of_pos hdpos
  have hsB : s < B := (Real.sqrt_lt' hBpos).mpr hdlt
  have hpgt : 0 < (B - s) / 2 := by linarith [hsB]
  have hlt1 : (B - s) / 2 < 1 := by
    by_cases hB2 : B ≤ 2
    · linarith [hspos, hB2]
    · push_neg at hB2
      have hsB2 : B - 2 < s := by
        have h2 : (B - 2) ^ 2 < d := by
          rw [hd]
          nlinarith [hκ, hB]
        exact (Real.lt_sqrt (by linarith : (0:ℝ) ≤ B - 2)).mpr h2
      linarith [hsB2]
  have hpmem : (B - s) / 2 ∈ Set.Ioo (0:ℝ) 1 := ⟨hpgt, hlt1⟩
  have hp1 : (1:ℝ) - (B - s) / 2 ≠ 0 := ne_of_gt (by linarith [hlt1])
  have hκ0 : κv ≠ 0 := ne_of_gt hκ
  have hroot : ((B - s) / 2) ^ 2 - (κv * ξv + κv + 1) * ((B - s) / 2) + κv * ξv = 0 := by
    rw [← hB]
    have key : ((B - s) / 2) ^ 2 - B * ((B - s) / 2) + κv * ξv
        = (s ^ 2 - (B ^ 2 - 4 * κv * ξv)) / 4 := by ring
    rw [key, hs2, hd]
    ring
  have hpeq : ξv = (B - s) / 2 / (1 - (B - s) / 2) + (B - s) / 2 / κv :=
    (master_quadratic hp1 hκ0).mpr hroot
  refine ⟨(B - s) / 2, ⟨hpmem, hpeq⟩, ?_⟩
  intro q hq
  rcases hq with ⟨hqmem, hqeq⟩
  rcases lt_trichotomy q ((B - s) / 2) with h | h | h
  · exfalso
    have hlt := master_strictMono hκ hqmem hpmem h
    exact absurd (hqeq.symm.trans hpeq) (ne_of_lt hlt)
  · exact h
  · exfalso
    have hlt := master_strictMono hκ hpmem hqmem h
    exact absurd (hpeq.symm.trans hqeq) (ne_of_lt hlt)

end TCS
