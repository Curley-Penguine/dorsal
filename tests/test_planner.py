from dorsal.planning.planner import Targets, optimize


def test_plan_meets_risk_targets(hero, fc, fin):
    plan = optimize(hero[0], fc, fin, Targets())
    assert plan["feasible"]
    assert plan["after"]["p_shortfall"] <= 0.02
    assert plan["before"]["p_shortfall"] > 0.8


def test_plan_never_touches_protected_obligations(hero, fc, fin):
    biz = hero[0]
    plan = optimize(biz, fc, fin, Targets())
    for a in plan["actions"]:
        bid = a.params.get("bill_id")
        if bid:
            assert not biz.vendor(biz.bill(bid).vendor_id).protected


def test_plan_catches_the_duplicate_bill(hero, fc, fin):
    plan = optimize(hero[0], fc, fin, Targets())
    assert any(a.type == "hold_bill" and a.params["bill_id"] == "B-904" for a in plan["actions"])


def test_financing_respects_cap_and_cost_limit(hero, fc, fin):
    plan = optimize(hero[0], fc, fin, Targets(max_borrow=25000))
    for a in plan["actions"]:
        if a.type == "draw_financing":
            assert a.params["amount"] <= 25000
            assert a.params["apr"] <= 0.36


def test_every_step_has_attribution(hero, fc, fin):
    plan = optimize(hero[0], fc, fin, Targets())
    assert all("impact" in s for s in plan["steps"])
