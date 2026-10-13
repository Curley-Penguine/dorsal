import numpy as np

from dorsal.analytics.backtest import backtest
from dorsal.analytics.forecast import Forecaster, Overlay, _kaplan_meier, _km_quantile


def test_forecast_is_reproducible(hero):
    a = Forecaster(hero[0], n_paths=300, seed=7).run().balances
    b = Forecaster(hero[0], n_paths=300, seed=7).run().balances
    assert np.array_equal(a, b)


def test_empty_overlay_matches_baseline(fc):
    assert np.array_equal(fc.run().balances, fc.run(Overlay()).balances)


def test_hero_has_a_predictable_gap(fc):
    s = fc.run().summary()
    assert s["p_shortfall"] > 0.8
    assert s["median_low_date"] == "2026-11-20"


def test_hold_and_split_raise_the_low_point(fc, hero):
    biz = hero[0]
    base = fc.run()
    pg = next(b for b in biz.open_bills() if b.reference == "PG-2291")
    half = pg.amount / 2
    from datetime import timedelta
    ov = Overlay(holds={"B-904"}, bill_schedule={pg.id: [(pg.due_date, half), (pg.due_date + timedelta(days=28), half)]})
    better = fc.run(ov)
    assert better.expected_deficit < base.expected_deficit
    assert better.p_shortfall < base.p_shortfall


def test_kaplan_meier_accounts_for_censoring():
    # 10 paid at ~5 days late; 10 still open and already 30 days late
    paid = [(5, 1)] * 10
    censored = [(30, 0)] * 10
    naive_median = float(_km_quantile(_kaplan_meier(paid), np.array([0.75]))[0])
    km = _kaplan_meier(paid + censored)
    km_q = float(_km_quantile(km, np.array([0.75]))[0])
    assert km_q > naive_median  # unpaid invoices push the tail out


def test_conditional_sampling_respects_lateness():
    km = _kaplan_meier([(d, 1) for d in range(0, 40)])
    out = _km_quantile(km, np.linspace(0.01, 0.99, 50), at_least=25)
    assert (out >= 25).all()


def test_backtest_beats_naive_baseline(hero):
    r = backtest(hero[0], weeks_back=(13, 17, 21), n_paths=300)
    assert r["skill_vs_naive"] > 0.3
    assert 0.5 <= r["coverage_p10_p90"] <= 1.0
