"""Bayesian readout with a Beta-Binomial model.

Each arm's conversion rate gets a Beta posterior (uniform Beta(1, 1) prior):

    p_arm | data ~ Beta(1 + conversions, 1 + non-conversions)

From posterior draws we report the two quantities decision-makers actually
ask about:

- P(B > A): the probability the treatment is better at all.
- Expected loss of shipping B: E[max(p_A - p_B, 0)] -- how much conversion
  we expect to give up, on the occasions B is actually worse. This is the
  risk side of the decision, which a p-value never gives you.

Monte Carlo with a fixed seed; with 200k draws the Monte Carlo error on
P(B > A) is well below 0.001 for the probabilities seen here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_DRAWS = 200_000
SEED = 7


@dataclass
class BayesianResult:
    prob_b_beats_a: float
    expected_loss_ship_b: float  # E[max(p_A - p_B, 0)]
    expected_loss_ship_a: float  # E[max(p_B - p_A, 0)]
    mean_diff: float
    ci_low: float  # 95% credible interval on (p_B - p_A)
    ci_high: float
    post_a: tuple  # (alpha, beta) of arm A's posterior
    post_b: tuple


def beta_posterior(conv: int, n: int) -> tuple:
    """Beta posterior parameters under a uniform prior."""
    return (1 + conv, 1 + n - conv)


def bayesian_analysis(
    conv_a: int,
    n_a: int,
    conv_b: int,
    n_b: int,
    n_draws: int = N_DRAWS,
    seed: int = SEED,
) -> BayesianResult:
    rng = np.random.default_rng(seed)
    post_a = beta_posterior(conv_a, n_a)
    post_b = beta_posterior(conv_b, n_b)

    draws_a = rng.beta(*post_a, size=n_draws)
    draws_b = rng.beta(*post_b, size=n_draws)
    diff = draws_b - draws_a

    return BayesianResult(
        prob_b_beats_a=float(np.mean(diff > 0)),
        expected_loss_ship_b=float(np.mean(np.maximum(-diff, 0.0))),
        expected_loss_ship_a=float(np.mean(np.maximum(diff, 0.0))),
        mean_diff=float(np.mean(diff)),
        ci_low=float(np.quantile(diff, 0.025)),
        ci_high=float(np.quantile(diff, 0.975)),
        post_a=post_a,
        post_b=post_b,
    )


if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv(__file__.rsplit("/", 2)[0] + "/data/experiment.csv")
    a = data[data["arm"] == "control"]
    b = data[data["arm"] == "treatment"]
    r = bayesian_analysis(
        int(a["converted"].sum()), len(a), int(b["converted"].sum()), len(b)
    )
    print(f"P(B > A)               = {r.prob_b_beats_a:.4f}")
    print(f"Posterior mean diff    = {r.mean_diff:+.4f}")
    print(f"95% credible interval  = [{r.ci_low:+.4f}, {r.ci_high:+.4f}]")
    print(f"Expected loss, ship B  = {r.expected_loss_ship_b:.6f}")
    print(f"Expected loss, ship A  = {r.expected_loss_ship_a:.6f}")
