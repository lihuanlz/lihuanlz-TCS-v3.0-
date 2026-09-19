/-
=====================================================================
S2b: Statistics of the digital (partitioned) platform and the
calibration-free estimator
=====================================================================
Corresponds to Section S2b of the SI ("Supplementary and additional
Material"):
  - Positive-partition probability:
    P_pos = b + (1−b)·(1 − exp(−μ/(1+κ))),
    where b is the background false-positive rate, μ is the mean
    molecular loading, and κ is the ligand-depletion parameter
  - Correction factor: γ = 1/(1+κ)
  - At κ = 0 this reduces to the classical Poisson formula of dPCR
    (the Poisson limit)
  - Estimator inversion: μ̂ = −(1+κ)·ln(1 − (P−b)/(1−b)),
    i.e. the molecular loading is uniquely recovered from the
    positive fraction

Lean 4 + Mathlib, toolchain leanprover/lean4:v4.15.0
-/

import Mathlib

namespace TCS

/-- Positive-partition probability (the main equation of S2b). -/
noncomputable def P_pos (b κ μ : ℝ) : ℝ :=
  b + (1 - b) * (1 - Real.exp (-μ / (1 + κ)))

/-- Poisson limit: at κ = 0 the formula reduces to the classical dPCR
formula P_pos = b + (1−b)·(1 − e^{−μ}). -/
theorem poisson_limit {b μ : ℝ} :
    P_pos b 0 μ = b + (1 - b) * (1 - Real.exp (-μ)) := by
  simp [P_pos]

/-- Correction-factor form: P_pos = b + (1−b)·(1 − exp(−γ·μ)) with
γ = 1/(1+κ). -/
theorem correction_factor {b κ μ : ℝ} (hκ : (1:ℝ) + κ ≠ 0) :
    P_pos b κ μ = b + (1 - b) * (1 - Real.exp (-(1 / (1 + κ)) * μ)) := by
  have h : -μ / (1 + κ) = -(1 / (1 + κ)) * μ := by
    field_simp
  simp [P_pos, h]

/-- Estimator inversion (legitimacy of the S2b estimator):
if P = P_pos(b,κ,μ) with b ≤ P < 1, b < 1 and 1+κ > 0, then
μ = −(1+κ)·ln(1 − (P−b)/(1−b)), and the inversion is unique. -/
theorem estimator_inversion {b κ μ P : ℝ} (hb : b < 1) (hκ : (0:ℝ) < 1 + κ)
    (_hP0 : b ≤ P) (hP1 : P < 1) (h : P = P_pos b κ μ) :
    μ = -(1 + κ) * Real.log (1 - (P - b) / (1 - b)) := by
  have hb1 : (0:ℝ) < 1 - b := by linarith
  have hb1' : (1:ℝ) - b ≠ 0 := ne_of_gt hb1
  have hκ' : (1:ℝ) + κ ≠ 0 := ne_of_gt hκ
  have hratio : (P - b) / (1 - b) = 1 - Real.exp (-μ / (1 + κ)) := by
    rw [h, P_pos]
    field_simp
  have hexp : Real.exp (-μ / (1 + κ)) = 1 - (P - b) / (1 - b) := by
    linarith [hratio]
  have hargpos : (0:ℝ) < 1 - (P - b) / (1 - b) := by
    have h1 : (P - b) / (1 - b) < 1 := by
      rw [div_lt_one hb1]
      linarith [hP1]
    linarith [h1]
  have harg : (1:ℝ) - (P - b) / (1 - b) = (1 - P) / (1 - b) := by
    field_simp
  have hlog : Real.log ((1 - P) / (1 - b)) = -μ / (1 + κ) := by
    rw [← harg, ← hexp, Real.log_exp]
  rw [harg, hlog]
  field_simp
  ring

/-- Classical dPCR special case: for b = 0 and κ = 0, every 0 ≤ P < 1 is
realized by μ = −ln(1−P); the classical calibration-free estimator is
the κ = 0 special case. -/
theorem estimator_dPCR_limit {P : ℝ} (_hP0 : (0:ℝ) ≤ P) (hP1 : P < 1) :
    ∃ μ : ℝ, P = P_pos 0 0 μ ∧ μ = -Real.log (1 - P) := by
  refine ⟨-Real.log (1 - P), ?_, rfl⟩
  have hpos : (0:ℝ) < 1 - P := by linarith
  simp [P_pos]
  rw [Real.exp_log hpos]
  ring

end TCS
