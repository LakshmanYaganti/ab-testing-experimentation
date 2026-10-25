"""Frequentist primary analysis for a two-arm conversion experiment.

- Two-proportion z-test (pooled SE for the p-value, unpooled SE for the
  confidence interval on the difference -- the standard, defensible combo).
- Chi-square test of independence as a cross-check (for a 2x2 table the
  chi-square statistic equals z^2, so the p-values must agree).
- Sample Ratio Mismatch (SRM) check: chi-square goodness-of-fit of the
  observed arm sizes against the intended split. A significant SRM means
  the randomization or logging is broken and the readout should stop here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ZTestResult:
    n_a: int
    n_b: int
    conv_a: int
    conv_b: int
    rate_a: float
    rate_b: float
    diff: float  # rate_b - rate_a
    ci_low: float
    ci_high: float
    z_stat: float
    p_value: float


def two_proportion_ztest(
    conv_a: int, n_a: int, conv_b: int, n_b: int, alpha: float = 0.05
) -> ZTestResult:
    """Two-proportion z-test for arm B vs arm A."""
    rate_a = conv_a / n_a
    rate_b = conv_b / n_b
    diff = rate_b - rate_a

    # Pooled standard error -> test statistic / p-value.
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z_stat = diff / se_pool
    p_value = 2 * stats.norm.sf(abs(z_stat))

    # Unpooled standard error -> Wald CI on the difference.
    se_unpooled = np.sqrt(
        rate_a * (1 - rate_a) / n_a + rate_b * (1 - rate_b) / n_b
    )
    half = stats.norm.ppf(1 - alpha / 2) * se_unpooled

    return ZTestResult(
        n_a=n_a,
        n_b=n_b,
        conv_a=conv_a,
        conv_b=conv_b,
        rate_a=rate_a,
        rate_b=rate_b,
        diff=diff,
        ci_low=diff - half,
        ci_high=diff + half,
        z_stat=float(z_stat),
        p_value=float(p_value),
    )


def chi_square_test(conv_a: int, n_a: int, conv_b: int, n_b: int) -> float:
    """Chi-square test of independence on the 2x2 table; returns the p-value."""
    table = np.array(
        [[conv_a, n_a - conv_a], [conv_b, n_b - conv_b]], dtype=float
    )
    _, p_value, _, _ = stats.chi2_contingency(table, correction=False)
    return float(p_value)


@dataclass
class SRMResult:
    n_a: int
    n_b: int
    expected_ratio_a: float
    chi2_stat: float
    p_value: float
    srm_detected: bool


def srm_check(
    n_a: int, n_b: int, expected_ratio_a: float = 0.5, alpha: float = 0.001
) -> SRMResult:
    """Chi-square goodness-of-fit of arm sizes against the intended split.

    Uses a strict alpha (0.001) as is conventional for SRM: with experiment-
    scale sample sizes, trivial imbalances are not the concern -- a broken
    assignment pipeline is.
    """
    total = n_a + n_b
    expected = np.array(
        [total * expected_ratio_a, total * (1 - expected_ratio_a)]
    )
    observed = np.array([n_a, n_b], dtype=float)
    chi2_stat = float(np.sum((observed - expected) ** 2 / expected))
    p_value = float(stats.chi2.sf(chi2_stat, df=1))
    return SRMResult(
        n_a=n_a,
        n_b=n_b,
        expected_ratio_a=expected_ratio_a,
        chi2_stat=chi2_stat,
        p_value=p_value,
        srm_detected=bool(p_value < alpha),
    )


def summarize(df) -> dict:
    """Run the full frequentist readout on an experiment DataFrame."""
    a = df[df["arm"] == "control"]
    b = df[df["arm"] == "treatment"]
    res = two_proportion_ztest(
        int(a["converted"].sum()), len(a), int(b["converted"].sum()), len(b)
    )
    chi2_p = chi_square_test(res.conv_a, res.n_a, res.conv_b, res.n_b)
    srm = srm_check(res.n_a, res.n_b)
    return {"ztest": res, "chi2_p_value": chi2_p, "srm": srm}


if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv(
        __file__.rsplit("/", 2)[0] + "/data/experiment.csv"
    )
    out = summarize(data)
    z, s = out["ztest"], out["srm"]
    print(f"Control   : {z.conv_a:,}/{z.n_a:,} = {z.rate_a:.4f}")
    print(f"Treatment : {z.conv_b:,}/{z.n_b:,} = {z.rate_b:.4f}")
    print(f"Diff      : {z.diff:+.4f}  95% CI [{z.ci_low:+.4f}, {z.ci_high:+.4f}]")
    print(f"z = {z.z_stat:.3f}, p = {z.p_value:.3e}")
    print(f"Chi-square cross-check p = {out['chi2_p_value']:.3e}")
    print(f"SRM: chi2 = {s.chi2_stat:.3f}, p = {s.p_value:.3f}, detected = {s.srm_detected}")
