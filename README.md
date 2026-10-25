# A/B Testing & Experimentation Analysis

**A complete, rigorous experiment readout — SRM checks, frequentist + Bayesian inference, CUPED, power analysis, and a simulation audit of the test itself — for a simulated checkout-flow experiment with known ground truth.**

Most "A/B testing projects" stop at *"p < 0.05, ship it."* Real product data science doesn't. A readout you'd defend in front of a experimentation review board has to answer harder questions: Was the randomization even valid (SRM)? How big is the effect, not just "is it nonzero" (CI, not just p)? What's the *risk* of shipping (expected loss, not just P(B > A))? Can we get the same answer with less traffic (CUPED, power)? And does the test itself keep its statistical promises (simulation audit)? This repo is built around those questions.

## ⚠️ The data is simulated — deliberately, and that's the point

The experiment data is **simulated with a seeded, committed generator** (`src/generate_data.py`) and **known ground truth**:

| Design parameter | Value |
|---|---|
| Scenario | Checkout-flow redesign, control vs. treatment, 50/50 |
| Users | 50,000 per arm (100,000 total) |
| Control conversion (true) | 12.0% |
| Treatment effect (true) | **+1.5 percentage points** |
| Pre-experiment covariate | Sessions in prior 30 days (`prior_sessions`), ρ ≈ 0.24 with conversion |
| Segments | `user_type` (new / returning), `device` (mobile / desktop) |

Nothing here is from a real company, and no number is presented as if it were. Simulated data with known truth is a *strength* for an experimentation portfolio: it lets you verify the statistics instead of trusting them. You cannot check whether a z-test really has a 5% false-positive rate on real data — here, Section 6 of the notebook does exactly that, by brute force.

## Results (from the actual analysis run)

| Readout | Result |
|---|---|
| Control conversion | 11.90% (5,948 / 50,000) |
| Treatment conversion | 13.43% (6,713 / 50,000) |
| **Observed effect** | **+1.53pp** (95% CI **[+1.12, +1.94]pp**), +12.9% relative |
| z-test | z = 7.27, **p = 3.5e-13** (chi-square cross-check: identical p) |
| SRM check | χ² = 0.000, p = 1.000 → **no SRM** |
| Bayesian P(B > A) | **1.0000** (no losing draws in 200,000 posterior samples) |
| Expected loss of shipping B | **0.000000** |
| CUPED | effect +1.63pp, outcome variance **−5.8%**, SE −3.0% (θ = 0.0170, ρ = 0.241) |
| Segments | All 4 segments positive; all survive Bonferroni (α = 0.0125) |
| **Type-I audit** | **54 / 1,000** false positives under the null → empirical FPR **0.054** vs. nominal 0.05 |
| Power audit | 1,000 / 1,000 rejections at the true +1.5pp (design MDE at 50k/arm is +0.58pp — the test was overpowered; see memo) |
| **Decision** | **SHIP** — full reasoning in [`decision_memo.md`](decision_memo.md) |

## What's inside

```
src/
  generate_data.py   Seeded simulator with known ground truth (+1.5pp)
  analysis.py        Two-proportion z-test (CI on the diff), chi-square
                     cross-check, SRM check (chi-square on arm sizes)
  power.py           Sample-size / MDE calculator (Cohen's h, statsmodels)
  bayesian.py        Beta-Binomial posterior: P(B>A), expected loss of
                     shipping either arm, credible interval
  cuped.py           CUPED adjustment on the pre-period covariate:
                     unbiased effect, variance-reduction %
  simulation.py      Type-I error study (1,000 null experiments) and
                     power study (1,000 experiments at the true lift)
notebooks/
  01_experiment_analysis.ipynb   The full executed readout, ending in a
                                   ship/no-ship decision
data/                Generator output lands here (CSV gitignored)
tests/               14 pytest tests pinning the math to known values
decision_memo.md     One-page experiment readout memo
```

## How to run

```bash
pip install -r requirements.txt

python -m src.generate_data     # writes data/experiment.csv (seeded)
python -m src.analysis          # primary readout: z-test, chi-square, SRM
python -m src.bayesian          # P(B>A) + expected loss
python -m src.cuped             # CUPED effect + variance reduction
python -m src.power             # sample-size table + MDE at 50k/arm
python -m src.simulation        # Type-I / power simulation studies

python -m pytest tests/ -q      # 14 tests
jupyter notebook notebooks/01_experiment_analysis.ipynb
```

## Method notes (the rigor is the portfolio piece)

- **SRM before results.** A sample-ratio mismatch means broken randomization or logging; the readout stops there. Checked at the conventional strict α = 0.001, alongside covariate/segment balance.
- **Interval, not just p-value.** The decision uses the whole CI: the lower bound (+1.12pp) is itself material, which matters more than p ≈ 0.
- **Bayesian layer for the decision, not the headline.** P(B > A) saturates at 1.0000 here, so the informative quantity is *expected loss* — how much conversion you'd give up if B were secretly worse (≈ 0).
- **CUPED done honestly.** θ from pooled data, covariate strictly pre-experiment, and the small point-estimate move (+1.53 → +1.63pp) reported rather than hidden. 5.8% variance reduction is modest — binary metrics with a session-count covariate are like that — and it's free power on every future run.
- **Segments with a correction.** Four segment tests at α = 0.05 imply a ~19% family-wise false-alarm rate; segments are judged against Bonferroni α = 0.0125 and used to detect harm/heterogeneity, not to cherry-pick. (Watch the trap: new users show +43% *relative* lift vs. +9.6% for returning — from nearly identical *absolute* lifts on very different baselines.)
- **The audit.** Because ground truth is known, the notebook replays the experiment 1,000× under the null and 1,000× at the design effect. Empirical FPR 0.054 ≈ nominal 0.05; empirical power 1.000, consistent with the design being powered for +0.58pp, not +1.5pp — an overpowering lesson the memo calls out.

## Tech stack

Python · NumPy · pandas · SciPy · statsmodels · matplotlib · Jupyter · pytest

## License

MIT — see [LICENSE](LICENSE).

## Author

**Lakshman Sai Yaganti** — Data Scientist @ Square
