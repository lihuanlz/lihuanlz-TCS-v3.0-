/-
=====================================================================
S2a: Binomial statistics of sampling fluctuations and the
CV = 1/√M scaling
=====================================================================
Corresponds to Section S2a of the SI ("Supplementary and additional
Material"):
  - Mean of the binomial distribution B(n, p): E[k] = n·p
  - Factorial moment of the binomial distribution:
    E[k(k−1)] = n(n−1)·p²
  - Variance: Var[k] = n·p·(1−p)
  - Relative fluctuation: CV² = (1−p)/(n·p), together with the
    Fano-type identity CV² · (n·p/(1−p)) = 1; in the low-occupancy
    limit p → 0⁺, CV²·(n·p) → 1, i.e. CV ≈ 1/√M
    (M = n·p is the expected count)

Proof route: k·C(n,k) = n·C(n−1,k−1) and
k(k−1)·C(n,k) = n(n−1)·C(n−2,k−2) (corollaries of
Nat.succ_mul_choose_eq), followed by reindexing and the binomial
theorem (add_pow).

Lean 4 + Mathlib, toolchain leanprover/lean4:v4.15.0
-/

import Mathlib

open Finset

namespace TCS

/-- Total probability is normalized: Σₖ C(n,k) pᵏ (1−p)ⁿ⁻ᵏ = 1. -/
theorem binomial_total (n : ℕ) (p : ℝ) :
    ∑ k ∈ Finset.range (n + 1),
      (Nat.choose n k : ℝ) * p ^ k * (1 - p) ^ (n - k) = 1 := by
  have h : ∑ k ∈ Finset.range (n + 1),
      (Nat.choose n k : ℝ) * p ^ k * (1 - p) ^ (n - k) = (p + (1 - p)) ^ n := by
    rw [add_pow]
    apply Finset.sum_congr rfl
    intro k _
    ring
  rw [h, show p + (1 - p) = (1:ℝ) by ring, one_pow]

/-- Binomial mean: E[k] = n·p. -/
theorem binomial_mean (n : ℕ) (p : ℝ) :
    ∑ k ∈ Finset.range (n + 1),
      (k : ℝ) * (Nat.choose n k : ℝ) * p ^ k * (1 - p) ^ (n - k)
      = (n : ℝ) * p := by
  rcases n with _ | n
  · simp
  rw [Finset.sum_range_succ']
  simp only [Nat.cast_zero, zero_mul, add_zero]
  have hpows : (p + (1 - p)) ^ n = ∑ k ∈ Finset.range (n + 1),
      (Nat.choose n k : ℝ) * p ^ k * (1 - p) ^ (n - k) := by
    rw [add_pow]
    apply Finset.sum_congr rfl
    intro k _
    ring
  have hsum : ∑ k ∈ Finset.range (n + 1),
        ((k + 1 : ℕ) : ℝ) * (Nat.choose (n + 1) (k + 1) : ℝ) * p ^ (k + 1) *
          (1 - p) ^ (n + 1 - (k + 1))
        = ((n + 1 : ℕ) : ℝ) * p * (p + (1 - p)) ^ n := by
    rw [hpows, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro k _
    have hchoose : ((k + 1 : ℕ) : ℝ) * (Nat.choose (n + 1) (k + 1) : ℝ) =
        ((n + 1 : ℕ) : ℝ) * (Nat.choose n k : ℝ) := by
      have h := Nat.succ_mul_choose_eq n k
      have h' : ((n + 1 : ℕ) : ℝ) * (Nat.choose n k : ℝ) =
          (Nat.choose (n + 1) (k + 1) : ℝ) * ((k + 1 : ℕ) : ℝ) := by
        exact_mod_cast h
      rw [mul_comm]
      exact h'.symm
    have hexp : n + 1 - (k + 1) = n - k := by omega
    rw [hchoose, hexp, pow_succ]
    ring
  rw [hsum, show p + (1 - p) = (1:ℝ) by ring, one_pow, mul_one]

/-- Binomial factorial moment: E[k(k−1)] = n(n−1)·p². -/
theorem binomial_factorial_moment (n : ℕ) (p : ℝ) :
    ∑ k ∈ Finset.range (n + 1),
      (k : ℝ) * ((k : ℝ) - 1) * (Nat.choose n k : ℝ) * p ^ k * (1 - p) ^ (n - k)
      = (n : ℝ) * ((n : ℝ) - 1) * p ^ 2 := by
  rcases n with _ | n
  · simp
  rcases n with _ | n
  · norm_num [Finset.sum_range_succ]
  -- The n + 2 case: reindex k ↦ k+2 and use
  -- k(k−1)C(n,k) = n(n−1)C(n−2,k−2)
  rw [Finset.sum_range_succ']
  simp only [Nat.cast_zero, zero_mul, add_zero]
  rw [Finset.sum_range_succ']
  simp only [zero_add, Nat.cast_one, sub_self, mul_zero, zero_mul, add_zero]
  have hpows : (p + (1 - p)) ^ n = ∑ k ∈ Finset.range (n + 1),
      (Nat.choose n k : ℝ) * p ^ k * (1 - p) ^ (n - k) := by
    rw [add_pow]
    apply Finset.sum_congr rfl
    intro k _
    ring
  have hchoose2 : ∀ k : ℕ,
      ((k + 2 : ℕ) : ℝ) * ((k + 1 : ℕ) : ℝ) * (Nat.choose (n + 2) (k + 2) : ℝ) =
      ((n + 2 : ℕ) : ℝ) * ((n + 1 : ℕ) : ℝ) * (Nat.choose n k : ℝ) := by
    intro k
    have h1 := Nat.succ_mul_choose_eq (n + 1) (k + 1)
    have h2 := Nat.succ_mul_choose_eq n k
    have e1 : ((n + 2 : ℕ) : ℝ) * (Nat.choose (n + 1) (k + 1) : ℝ) =
        (Nat.choose (n + 2) (k + 2) : ℝ) * ((k + 2 : ℕ) : ℝ) := by
      exact_mod_cast h1
    have e2 : ((n + 1 : ℕ) : ℝ) * (Nat.choose n k : ℝ) =
        (Nat.choose (n + 1) (k + 1) : ℝ) * ((k + 1 : ℕ) : ℝ) := by
      exact_mod_cast h2
    have e1' := congrArg (· * ((k + 1 : ℕ) : ℝ)) e1
    have e2' := congrArg (· * ((n + 2 : ℕ) : ℝ)) e2
    linear_combination -e1' - e2'
  have hmain : ∑ k ∈ Finset.range (n + 1),
        ((k + 2 : ℕ) : ℝ) * (((k + 2 : ℕ) : ℝ) - 1) *
          (Nat.choose (n + 2) (k + 2) : ℝ) * p ^ (k + 2) * (1 - p) ^ (n + 2 - (k + 2))
        = ((n + 2 : ℕ) : ℝ) * (((n + 2 : ℕ) : ℝ) - 1) * p ^ 2 * (p + (1 - p)) ^ n := by
    rw [hpows, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro k _
    have hk1 : ((k + 2 : ℕ) : ℝ) - 1 = ((k + 1 : ℕ) : ℝ) := by push_cast; ring
    have hn1 : ((n + 2 : ℕ) : ℝ) - 1 = ((n + 1 : ℕ) : ℝ) := by push_cast; ring
    have hexp : n + 2 - (k + 2) = n - k := by omega
    rw [hk1, hchoose2 k, hexp, hn1, show p ^ (k + 2) = p ^ 2 * p ^ k by ring]
    ring
  rw [hmain, show p + (1 - p) = (1:ℝ) by ring, one_pow, mul_one]

/-- Binomial variance: Var[k] = E[(k − n·p)²] = n·p·(1−p).
This follows by expanding
E[(k−np)²] = E[k(k−1)] + (1−2np)·E[k] + (np)²·1. -/
theorem binomial_variance (n : ℕ) (p : ℝ) :
    ∑ k ∈ Finset.range (n + 1),
      ((k : ℝ) - (n : ℝ) * p) ^ 2 * (Nat.choose n k : ℝ) * p ^ k * (1 - p) ^ (n - k)
      = (n : ℝ) * p * (1 - p) := by
  have hexpand : ∀ k : ℕ,
      ((k : ℝ) - (n : ℝ) * p) ^ 2 * (Nat.choose n k : ℝ) * p ^ k * (1 - p) ^ (n - k) =
      ((k : ℝ) * ((k : ℝ) - 1) * (Nat.choose n k : ℝ) * p ^ k * (1 - p) ^ (n - k)) +
        (1 - 2 * ((n : ℝ) * p)) *
          ((k : ℝ) * (Nat.choose n k : ℝ) * p ^ k * (1 - p) ^ (n - k)) +
        ((n : ℝ) * p) ^ 2 * ((Nat.choose n k : ℝ) * p ^ k * (1 - p) ^ (n - k)) := by
    intro k
    ring
  rw [Finset.sum_congr rfl (fun k _ => hexpand k),
    Finset.sum_add_distrib, Finset.sum_add_distrib,
    ← Finset.mul_sum, ← Finset.mul_sum,
    binomial_factorial_moment, binomial_mean, binomial_total]
  ring

/-- Relative fluctuation (squared coefficient of variation):
CV² = Var/E² = (1−p)/(n·p). -/
theorem binomial_cv_sq (n : ℕ) (p : ℝ) :
    ((n : ℝ) * p * (1 - p)) / (((n : ℝ) * p) ^ 2) = (1 - p) / ((n : ℝ) * p) := by
  by_cases h : ((n : ℝ) * p) = 0
  · simp [h]
  · field_simp
    ring

/-- Fano-type identity: CV² · (n·p/(1−p)) = 1.
In the low-occupancy limit p → 0⁺ we have n·p/(1−p) ≈ M (the expected
count), i.e. CV ≈ 1/√M. -/
theorem fano_identity (n : ℕ) {p : ℝ} (hp : p ≠ 1) (h : ((n : ℝ) * p) ≠ 0) :
    ((1 - p) / ((n : ℝ) * p)) * (((n : ℝ) * p) / (1 - p)) = 1 := by
  have h1p : (1:ℝ) - p ≠ 0 := sub_ne_zero.mpr hp.symm
  field_simp

end TCS
