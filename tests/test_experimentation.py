"""Tests for the experimentation toolkit.

Each test pins down a property the statistics *should* have -- mostly on
small, hand-computable inputs -- so a regression in the math fails loudly.
"""

import numpy as np
import pytest
from scipy import stats

from src.analysis import chi_square_test, srm_check, two_proportion_ztest
from src.bayesian import bayesian_analysis, beta_posterior
from src.cuped import run_cuped
from src.generate_data import generate_experiment
from src.power import mde_for_sample_size, sample_size_per_arm


# ---------------------------------------------------------------- z-test ---

def test_ztest_known_values():
    # Hand-checkable: 10/100 vs 20/100 -> diff 0.10, pooled p = 0.15.
    r = two_proportion_ztest(conv_a=10, n_a=100, conv_b=20, n_b=100)
    assert r.rate_a == pytest.approx(0.10)
    assert r.rate_b == pytest.approx(0.20)
    assert r.diff == pytest.approx(0.10)
    se_pool = np.sqrt(0.15 * 0.85 * (1 / 100 + 1 / 100))
    assert r.z_stat == pytest.approx(0.10 / se_pool)
    assert r.p_value == pytest.approx(2 * stats.norm.sf(abs(r.z_stat)))
    assert r.ci_low < r.diff < r.ci_high


def test_ztest_no_effect_is_not_significant():
    r = two_proportion_ztest(conv_a=500, n_a=5000, conv_b=505, n_b=5000)
    assert r.p_value > 0.05
    assert r.ci_low < 0 < r.ci_high


def test_ztest_matches_chi_square():
    # For a 2x2 table, chi-square = z^2, so p-values must agree closely.
    r = two_proportion_ztest(conv_a=5948, n_a=50_000, conv_b=6713, n_b=50_000)
    p_chi2 = chi_square_test(5948, 50_000, 6713, 50_000)
    assert p_chi2 == pytest.approx(r.p_value, rel=1e-6)


# -------------------------------------------------------------------- SRM ---

def test_srm_clean_split_passes():
    r = srm_check(50_000, 50_000)
    assert r.chi2_stat == pytest.approx(0.0)
    assert not r.srm_detected


def test_srm_detects_skewed_split():
    # 55/45 at experiment scale is a broken pipeline, not bad luck.
    r = srm_check(55_000, 45_000)
    assert r.srm_detected
    assert r.p_value < 0.001


def test_srm_borderline_jitter_not_flagged_at_strict_alpha():
    # +/-0.3% jitter on 100k users should not trip the alpha=0.001 SRM check.
    r = srm_check(50_150, 49_850)
    assert not r.srm_detected


# ------------------------------------------------------------------ power ---

def test_power_larger_mde_needs_smaller_n():
    n_small_mde = sample_size_per_arm(0.12, 0.005)
    n_large_mde = sample_size_per_arm(0.12, 0.020)
    assert n_small_mde > n_large_mde


def test_power_higher_power_needs_larger_n():
    n80 = sample_size_per_arm(0.12, 0.015, power=0.80)
    n90 = sample_size_per_arm(0.12, 0.015, power=0.90)
    assert n90 > n80


def test_power_mde_round_trip():
    # MDE(n) should approximately invert sample_size(MDE).
    mde = mde_for_sample_size(0.12, 50_000)
    n_back = sample_size_per_arm(0.12, mde)
    assert n_back == pytest.approx(50_000, rel=0.05)
    assert 0 < mde < 0.02


# --------------------------------------------------------------- bayesian ---

def test_beta_posterior_updates():
    assert beta_posterior(conv=30, n=100) == (31, 71)


def test_bayesian_large_lift_high_probability():
    r = bayesian_analysis(conv_a=1_200, n_a=10_000, conv_b=1_600, n_b=10_000)
    assert r.prob_b_beats_a > 0.999
    assert r.expected_loss_ship_b < r.expected_loss_ship_a
    assert r.expected_loss_ship_b < 0.0005


def test_bayesian_no_effect_is_a_coin_flip():
    r = bayesian_analysis(conv_a=1_200, n_a=10_000, conv_b=1_200, n_b=10_000)
    assert 0.4 < r.prob_b_beats_a < 0.6
    assert r.ci_low < 0 < r.ci_high


# ------------------------------------------------------------------ CUPED ---

def test_cuped_reduces_variance_and_keeps_effect():
    # Full-size data: SE on the effect is ~0.002, so the tolerances below
    # are ~3 standard errors -- tight enough to catch bias, safe from noise.
    df = generate_experiment(seed=777)
    r = run_cuped(df)
    assert r.correlation > 0.1  # covariate genuinely predicts conversion
    assert r.variance_reduction_pct > 0
    assert r.adjusted_se < r.raw_se
    # Adjustment must not move the estimate far: same estimand, less noise.
    assert r.adjusted_effect == pytest.approx(r.raw_effect, abs=0.004)
    # And both should sit near the known ground-truth lift (+0.015).
    assert r.adjusted_effect == pytest.approx(0.015, abs=0.006)


# -------------------------------------------------------------- generator ---

def test_generator_ground_truth_and_determinism():
    df1 = generate_experiment(seed=5)
    df2 = generate_experiment(seed=5)
    assert df1.equals(df2)
    rates = df1.groupby("arm")["converted"].mean()
    assert rates["control"] == pytest.approx(0.12, abs=0.01)
    assert rates["treatment"] - rates["control"] == pytest.approx(0.015, abs=0.008)
    assert set(df1.columns) == {
        "user_id", "arm", "user_type", "device", "prior_sessions", "converted",
    }
