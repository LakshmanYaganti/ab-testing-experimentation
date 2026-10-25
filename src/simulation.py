"""Simulation studies: does the test itself behave as advertised?

Because the data-generating process is known, we can audit the statistics
instead of trusting the asymptotics:

1. Type-I error study -- simulate ~1,000 experiments where treatment does
   NOTHING (both arms convert at the baseline rate), run the two-proportion
   z-test on each at alpha = 0.05, and count how often it (wrongly) rejects.
   If the test is well calibrated the empirical false-positive rate lands
   near 0.05 (Monte Carlo SE at 1,000 sims is ~0.7pp).

2. Power study -- simulate experiments at the true design (+1.5pp) with the
   actual sample size and check the rejection rate against the ~80% the
   experiment was powered for.

Both studies are vectorized (binomial counts -> z statistics) so 1,000
simulations run in well under a second.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

N_SIMS = 1_000
SEED = 99


def _ztest_pvalues(
    conv_a: np.ndarray, n_a: int, conv_b: np.ndarray, n_b: int
) -> np.ndarray:
    """Vectorized pooled two-proportion z-test p-values."""
    rate_a = conv_a / n_a
    rate_b = conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (rate_b - rate_a) / se
    return 2 * stats.norm.sf(np.abs(z))


@dataclass
class TypeIResult:
    n_sims: int
    alpha: float
    false_positives: int
    empirical_fpr: float
    mc_se: float


@dataclass
class PowerStudyResult:
    n_sims: int
    alpha: float
    true_lift: float
    rejections: int
    empirical_power: float
    mc_se: float


def type_i_error_study(
    baseline_rate: float = 0.12,
    n_per_arm: int = 50_000,
    n_sims: int = N_SIMS,
    alpha: float = 0.05,
    seed: int = SEED,
) -> TypeIResult:
    rng = np.random.default_rng(seed)
    conv_a = rng.binomial(n_per_arm, baseline_rate, size=n_sims)
    conv_b = rng.binomial(n_per_arm, baseline_rate, size=n_sims)  # null: same rate
    pvals = _ztest_pvalues(conv_a, n_per_arm, conv_b, n_per_arm)
    fpr = float(np.mean(pvals < alpha))
    return TypeIResult(
        n_sims=n_sims,
        alpha=alpha,
        false_positives=int(np.sum(pvals < alpha)),
        empirical_fpr=fpr,
        mc_se=float(np.sqrt(fpr * (1 - fpr) / n_sims)),
    )


def power_study(
    baseline_rate: float = 0.12,
    true_lift: float = 0.015,
    n_per_arm: int = 50_000,
    n_sims: int = N_SIMS,
    alpha: float = 0.05,
    seed: int = SEED + 1,
) -> PowerStudyResult:
    rng = np.random.default_rng(seed)
    conv_a = rng.binomial(n_per_arm, baseline_rate, size=n_sims)
    conv_b = rng.binomial(n_per_arm, baseline_rate + true_lift, size=n_sims)
    pvals = _ztest_pvalues(conv_a, n_per_arm, conv_b, n_per_arm)
    pw = float(np.mean(pvals < alpha))
    return PowerStudyResult(
        n_sims=n_sims,
        alpha=alpha,
        true_lift=true_lift,
        rejections=int(np.sum(pvals < alpha)),
        empirical_power=pw,
        mc_se=float(np.sqrt(pw * (1 - pw) / n_sims)),
    )


if __name__ == "__main__":
    t1 = type_i_error_study()
    print(
        f"Type-I study : {t1.false_positives}/{t1.n_sims} false positives "
        f"-> empirical FPR {t1.empirical_fpr:.3f} (nominal {t1.alpha}, MC SE {t1.mc_se:.3f})"
    )
    pw = power_study()
    print(
        f"Power study  : {pw.rejections}/{pw.n_sims} rejections at true lift "
        f"+{pw.true_lift:.3f} -> empirical power {pw.empirical_power:.3f} (MC SE {pw.mc_se:.3f})"
    )
