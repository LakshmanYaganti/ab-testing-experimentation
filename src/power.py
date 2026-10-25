"""Power analysis: how many users per arm do we need, and what is the
smallest effect (MDE) a given sample size can detect?

Uses Cohen's h (the arcsine effect size for proportions) with the normal
approximation, via statsmodels' NormalIndPower.
"""

from __future__ import annotations

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

_power = NormalIndPower()


def sample_size_per_arm(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Users per arm needed to detect an absolute lift of `mde`.

    `mde` is in absolute percentage-point units (e.g. 0.015 = +1.5pp on a
    0.12 baseline). Larger MDE -> smaller required sample.
    """
    if mde <= 0:
        raise ValueError("mde must be positive")
    h = proportion_effectsize(baseline_rate + mde, baseline_rate)
    n = _power.solve_power(
        effect_size=h, alpha=alpha, power=power, ratio=1.0, alternative="two-sided"
    )
    return int(np_ceil(n))


def mde_for_sample_size(
    baseline_rate: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Smallest absolute lift detectable with `n_per_arm` users per arm.

    Solves numerically: find the Cohen's h that the given n can detect,
    then map it back to a difference in proportions.
    """
    h = _power.solve_power(
        effect_size=None,
        nobs1=n_per_arm,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided",
    )
    # Invert Cohen's h for the treatment rate given the baseline rate.
    from scipy.optimize import brentq

    def gap(p_b: float) -> float:
        return proportion_effectsize(p_b, baseline_rate) - h

    return float(brentq(gap, baseline_rate + 1e-9, 0.999) - baseline_rate)


def np_ceil(x: float) -> int:
    import math

    return math.ceil(x)


if __name__ == "__main__":
    base = 0.12
    for mde in (0.005, 0.010, 0.015, 0.020, 0.030):
        n = sample_size_per_arm(base, mde)
        print(f"Baseline {base:.0%}, MDE {mde:+.1%} -> {n:,} users/arm")
    print(
        f"With 50,000 users/arm at a {base:.0%} baseline, "
        f"MDE = {mde_for_sample_size(base, 50_000):+.4f}"
    )
