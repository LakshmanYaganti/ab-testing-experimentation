"""CUPED (Controlled-experiment Using Pre-Experiment Data).

CUPED removes the part of the outcome's variance that is explained by a
pre-experiment covariate X (here: sessions in the 30 days before the
experiment, which cannot be affected by treatment):

    theta     = Cov(X, Y) / Var(X)              (estimated on pooled data)
    Y_adj     = Y - theta * (X - mean(X))
    effect    = mean(Y_adj | B) - mean(Y_adj | A)

Because X is pre-treatment, the adjustment is unbiased; because X correlates
with Y, Var(Y_adj) = Var(Y) * (1 - rho^2) < Var(Y). Same estimand, tighter
confidence interval -- equivalently, the same power with fewer users.

Reference: Deng, Knoblich, Lu (2013), "Applying the Delta Method in
Metric Analytics: A Practical Guide with Novel Ideas", KDD '13.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CUPEDResult:
    theta: float
    correlation: float
    raw_effect: float
    raw_se: float
    adjusted_effect: float
    adjusted_se: float
    variance_reduction_pct: float  # % reduction in outcome variance
    se_reduction_pct: float  # % reduction in the effect's standard error


def cuped_adjust(y: np.ndarray, x: np.ndarray, theta: float) -> np.ndarray:
    """Return CUPED-adjusted outcomes."""
    return y - theta * (x - np.mean(x))


def run_cuped(df) -> CUPEDResult:
    """Run the CUPED analysis on an experiment DataFrame."""
    y = df["converted"].to_numpy(dtype=float)
    x = df["prior_sessions"].to_numpy(dtype=float)
    is_b = (df["arm"] == "treatment").to_numpy()

    theta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    corr = float(np.corrcoef(x, y)[0, 1])

    y_adj = cuped_adjust(y, x, theta)

    def effect_and_se(values: np.ndarray) -> tuple:
        a, b = values[~is_b], values[is_b]
        effect = float(np.mean(b) - np.mean(a))
        se = float(
            np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
        )
        return effect, se

    raw_effect, raw_se = effect_and_se(y)
    adj_effect, adj_se = effect_and_se(y_adj)

    var_red = 100 * (1 - np.var(y_adj, ddof=1) / np.var(y, ddof=1))
    se_red = 100 * (1 - adj_se / raw_se)

    return CUPEDResult(
        theta=theta,
        correlation=corr,
        raw_effect=raw_effect,
        raw_se=raw_se,
        adjusted_effect=adj_effect,
        adjusted_se=adj_se,
        variance_reduction_pct=float(var_red),
        se_reduction_pct=float(se_red),
    )


if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv(__file__.rsplit("/", 2)[0] + "/data/experiment.csv")
    r = run_cuped(data)
    print(f"theta (Cov/Var)        = {r.theta:.5f}")
    print(f"corr(prior_sessions, converted) = {r.correlation:.3f}")
    print(f"Raw effect      = {r.raw_effect:+.4f} (SE {r.raw_se:.4f})")
    print(f"CUPED effect    = {r.adjusted_effect:+.4f} (SE {r.adjusted_se:.4f})")
    print(f"Variance reduced by {r.variance_reduction_pct:.1f}%  |  "
          f"SE reduced by {r.se_reduction_pct:.1f}%")
