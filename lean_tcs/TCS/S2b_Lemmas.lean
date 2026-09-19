/-
=====================================================================
S2b: Lemmas 1 and 2 (binomial subset sum, choose identity)
=====================================================================
Corresponds to Section S2b of the SI:
  - Lemma 1: binomial subset sum at PMF-weight level
  - Lemma 2: the choose identity C(N,r)·C(r,n) = C(N,n)·C(N−n,r−n)

Lean 4 + Mathlib, toolchain leanprover/lean4:v4.15.0
-/

import Mathlib

/-- **Lemma 2 (SI S2b), choose identity.**
C(N, r) · C(r, n) = C(N, n) · C(N − n, r − n): counting length-n sub-subsets of
r-subsets of an N-set two ways. Hypothesis `hn : n ≤ r` is necessary
(without it the right side can be nonzero while the left vanishes). -/
theorem lemma2_choose_identity (N r n : ℕ) (hn : n ≤ r) :
    Nat.choose N r * Nat.choose r n = Nat.choose N n * Nat.choose (N - n) (r - n) := by
  rcases le_or_lt r N with hle | hgt
  · -- r ≤ N: both sides equal N! / (n! (r−n)! (N−r)!) via factorial identities.
    have hNr : Nat.choose N r * Nat.factorial r * Nat.factorial (N - r)
        = Nat.factorial N :=
      Nat.choose_mul_factorial_mul_factorial hle
    have hrn : Nat.choose r n * Nat.factorial n * Nat.factorial (r - n)
        = Nat.factorial r :=
      Nat.choose_mul_factorial_mul_factorial hn
    have hNn : Nat.choose N n * Nat.factorial n * Nat.factorial (N - n)
        = Nat.factorial N :=
      Nat.choose_mul_factorial_mul_factorial (le_trans hn hle)
    have hsub : r - n ≤ N - n := Nat.sub_le_sub_right hle n
    have hmem : N - n - (r - n) = N - r := by omega
    have hNn' : Nat.choose (N - n) (r - n) * Nat.factorial (r - n) * Nat.factorial (N - r)
        = Nat.factorial (N - n) := by
      rw [← hmem]
      exact Nat.choose_mul_factorial_mul_factorial hsub
    have hF : 0 < Nat.factorial n * Nat.factorial (r - n) * Nat.factorial (N - r) :=
      Nat.mul_pos (Nat.mul_pos (Nat.factorial_pos _) (Nat.factorial_pos _))
        (Nat.factorial_pos _)
    apply Nat.mul_right_cancel hF
    calc (Nat.choose N r * Nat.choose r n)
          * (Nat.factorial n * Nat.factorial (r - n) * Nat.factorial (N - r))
        = Nat.choose N r * (Nat.choose r n * Nat.factorial n * Nat.factorial (r - n))
            * Nat.factorial (N - r) := by ring
      _ = Nat.choose N r * Nat.factorial r * Nat.factorial (N - r) := by rw [hrn]
      _ = Nat.factorial N := hNr
      _ = Nat.choose N n * Nat.factorial n * Nat.factorial (N - n) := hNn.symm
      _ = Nat.choose N n * Nat.factorial n *
            (Nat.choose (N - n) (r - n) * Nat.factorial (r - n)
              * Nat.factorial (N - r)) := by rw [hNn']
      _ = (Nat.choose N n * Nat.choose (N - n) (r - n)) *
            (Nat.factorial n * Nat.factorial (r - n) * Nat.factorial (N - r)) := by ring
  · -- N < r: each side has a vanishing choose factor.
    have hNr0 : Nat.choose N r = 0 := Nat.choose_eq_zero_iff.2 hgt
    rcases le_or_lt n N with hnN | hnN
    · -- n ≤ N: then N - n < r - n, so the second factor on the right vanishes.
      have h2 : Nat.choose (N - n) (r - n) = 0 := by
        apply Nat.choose_eq_zero_iff.2
        omega
      rw [hNr0, h2, zero_mul, mul_zero]
    · -- N < n: the first factor on the right vanishes.
      have h1 : Nat.choose N n = 0 := Nat.choose_eq_zero_iff.2 hnN
      rw [hNr0, h1, zero_mul, zero_mul]

/-- **Lemma 1 (SI S2b), binomial subset sum, weight-level form.**
For any fixed m index set of i.i.d. Bernoulli(p) indicators, the number of
sub-configurations of size k carries total weight C(m, k) · p^k · (1 − p)^(m − k):
each size-k subset has weight p^|s| (1 − p)^(m − |s|) and there are C(m, k) of them.
This is the exact content of "the sum over any m-subset is Binomial(m, p)" at the
level of PMF weights; the n-case is already formalised in `TCS.S2a_AnalogStatistics`. -/
theorem lemma1_binomial_subset (m k : ℕ) (hkm : k ≤ m) (p : ℝ) :
    ∑ s ∈ (Finset.univ : Finset (Fin m)).powersetCard k,
        p ^ s.card * (1 - p) ^ (m - s.card)
      = (Nat.choose m k : ℝ) * p ^ k * (1 - p) ^ (m - k) := by
  have := hkm
  have hconst : ∀ s ∈ (Finset.univ : Finset (Fin m)).powersetCard k,
      p ^ s.card * (1 - p) ^ (m - s.card) = p ^ k * (1 - p) ^ (m - k) := by
    intro s hs
    rw [(Finset.mem_powersetCard.1 hs).2]
  rw [Finset.sum_congr rfl hconst, Finset.sum_const, Finset.card_powersetCard,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_assoc]
