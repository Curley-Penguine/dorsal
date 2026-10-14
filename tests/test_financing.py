from datetime import date, timedelta

from dorsal.planning import financing as fm


def test_irr_apr_simple_interest():
    d0 = date(2026, 1, 1)
    flows = [(d0, 10000.0), (d0 + timedelta(days=30), -10000.0 * (1 + 0.10 * 30 / 365))]
    assert abs(fm.irr_apr(flows) - 0.10) < 0.002


def test_merchant_cash_advance_is_flagged_high_cost(hero, fin):
    biz = hero[0]
    need = fm.Need(6000, biz.today + timedelta(days=10), biz.today + timedelta(days=60))
    quotes = {q["product_id"]: q for q in fm.match(biz, fin, need, daily_card_sales=1400)}
    assert quotes["mca"]["high_cost"] is True
    assert quotes["mca"]["recommended"] is False


def test_slow_product_is_not_in_time(hero, fin):
    biz = hero[0]
    need = fm.Need(30000, biz.today + timedelta(days=5), biz.today + timedelta(days=60))
    q = next(q for q in fm.match(biz, fin, need) if q["product_id"] == "sba-term")
    assert q["arrives_in_time"] is False


def test_best_option_is_cheapest_usable(hero, fin):
    biz = hero[0]
    need = fm.Need(6000, biz.today + timedelta(days=20), biz.today + timedelta(days=80))
    quotes = fm.match(biz, fin, need, daily_card_sales=1400)
    best = quotes[0]
    assert best["recommended"]
    usable = [q for q in quotes if q["eligible"] and q["arrives_in_time"] and not q["high_cost"]]
    assert best["total_cost"] == min(q["total_cost"] for q in usable)


def test_ineligible_products_explain_why(hero, fin):
    biz = hero[0]
    biz.credit_score_band = "D"
    need = fm.Need(6000, biz.today + timedelta(days=20), biz.today + timedelta(days=80))
    q = next(q for q in fm.match(biz, fin, need) if q["product_id"] == "flex-wc")
    assert not q["eligible"] and any("band" in r for r in q["reasons"])
