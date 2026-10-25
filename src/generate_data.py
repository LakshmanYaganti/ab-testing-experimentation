"""Seeded generator for a simulated checkout-flow A/B experiment.

The data is SIMULATED with known ground truth on purpose: because we know the
true effect, we can verify that the analysis machinery (z-test, CUPED,
Bayesian posterior, simulation studies) actually behaves the way the theory
says it should. Nothing here comes from a real company or real users.

Ground truth
------------
- Control conversion rate : 12.0% (marginal, after averaging over users)
- Treatment true lift     : +1.5 percentage points (additive, every user)
- Sample size             : ~50,000 users per arm
- Covariate               : `prior_sessions` = sessions in the 30 days BEFORE
                            the experiment. It is generated before assignment
                            and is positively correlated with conversion,
                            which is exactly what CUPED needs.
- Segments                : `user_type` (new / returning) and
                            `device` (mobile / desktop). Returning users and
                            desktop users have somewhat higher base rates.

Run directly to (re)generate the CSV:

    python -m src.generate_data            # writes data/experiment.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261024

N_CONTROL = 50_000
N_TREATMENT = 50_000

# Per-user conversion probability model (before the treatment effect):
#   p = BASE_INTERCEPT + SESSION_SLOPE * prior_sessions
#       + RETURNING_BONUS (if returning) + DESKTOP_BONUS (if desktop)
# prior_sessions has mean ~6.8, so the marginal control rate lands at ~12.0%
# and corr(prior_sessions, converted) at ~0.24 -- engagement genuinely
# stratifies conversion risk, which is what gives CUPED something to remove.
BASE_INTERCEPT = 0.001
SESSION_SLOPE = 0.016
RETURNING_BONUS = 0.012
DESKTOP_BONUS = 0.006
TRUE_LIFT = 0.015  # +1.5 percentage points for treatment users


def generate_experiment(
    n_control: int = N_CONTROL,
    n_treatment: int = N_TREATMENT,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return one experiment dataset as a DataFrame (deterministic by seed)."""
    rng = np.random.default_rng(seed)
    n = n_control + n_treatment

    arm = np.array(["control"] * n_control + ["treatment"] * n_treatment)

    # Segments are drawn independently of assignment (as in a clean RCT).
    user_type = rng.choice(["new", "returning"], size=n, p=[0.40, 0.60])
    device = rng.choice(["mobile", "desktop"], size=n, p=[0.62, 0.38])

    # Pre-experiment covariate: prior 30-day sessions. Returning users are
    # more active. Negative-binomial-ish counts via Poisson mixtures.
    lam = np.where(user_type == "returning", 10.0, 2.0)
    prior_sessions = rng.poisson(lam).astype(float)

    # Conversion probability per user, then the treatment effect on top.
    p = (
        BASE_INTERCEPT
        + SESSION_SLOPE * prior_sessions
        + RETURNING_BONUS * (user_type == "returning")
        + DESKTOP_BONUS * (device == "desktop")
    )
    p = np.clip(p, 0.005, 0.85)
    p = p + np.where(arm == "treatment", TRUE_LIFT, 0.0)

    converted = (rng.random(n) < p).astype(int)

    df = pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "arm": arm,
            "user_type": user_type,
            "device": device,
            "prior_sessions": prior_sessions.astype(int),
            "converted": converted,
        }
    )
    # Shuffle so the CSV is not visibly two blocks; keep user_id stable.
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "experiment.csv"
    df = generate_experiment()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    rates = df.groupby("arm")["converted"].mean()
    counts = df.groupby("arm").size()
    print(f"Wrote {out} ({len(df):,} rows)")
    print(f"Users per arm : control={counts['control']:,}, treatment={counts['treatment']:,}")
    print(f"Observed conv : control={rates['control']:.4f}, treatment={rates['treatment']:.4f}")
    print(f"Observed diff : {rates['treatment'] - rates['control']:+.4f} (true lift = +{TRUE_LIFT})")


if __name__ == "__main__":
    main()
