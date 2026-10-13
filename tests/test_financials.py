def test_health_score_and_reasons(fin):
    h = fin["health"]
    assert 0 <= h["score"] <= 100
    assert any("days of spending" in r for r in h["reasons"])      # demo has thin cash


def test_margins_are_plausible(fin):
    t = fin["trailing_12m"]
    assert t["revenue"] > 0
    assert 0 < t["gross_margin"] < 1


def test_customers_ranked_by_share(fin):
    shares = [c["share_of_credit_sales"] for c in fin["customers"]]
    assert shares == sorted(shares, reverse=True)
    assert abs(sum(shares) - 1) < 0.01
