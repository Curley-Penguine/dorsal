import pytest

from dorsal.actions import Action


def _first(rt, type_):
    return next(a for a in rt.plan()["actions"] if a.type == type_)


def test_tiers_route_correctly(rt):
    assert rt.policy.evaluate(_first(rt, "send_reminder"), rt.biz).outcome == "auto"
    assert rt.policy.evaluate(_first(rt, "hold_bill"), rt.biz).outcome == "auto"
    assert rt.policy.evaluate(_first(rt, "request_split"), rt.biz).outcome == "approve"
    d = rt.policy.evaluate(_first(rt, "draw_financing"), rt.biz)
    assert d.outcome == "approve" and d.step_up


def test_auto_limit_unlocks_only_discounts(rt):
    rt.update_settings({"auto_cost_limit": 1000})
    assert rt.policy.evaluate(_first(rt, "offer_discount"), rt.biz).outcome == "auto"
    assert rt.policy.evaluate(_first(rt, "request_split"), rt.biz).outcome == "approve"


def test_owner_settings_are_validated(rt):
    with pytest.raises(ValueError):
        rt.update_settings({"auto_cost_limit": 50000})
    with pytest.raises(ValueError):
        rt.update_settings({"tiers": {}})


def test_approval_executes_and_logs(rt):
    a = _first(rt, "request_split")
    rt.propose(a)
    rec = rt.approve(a.id)
    assert rec["status"] == "executed"
    assert a.params["bill_id"] in rt.world.schedule_overrides
    assert any(e["event"] == "approval.granted" for e in rt.audit.entries)


def test_step_up_requires_pin(rt):
    from dorsal.security.approvals import ApprovalError
    a = _first(rt, "draw_financing")
    rt.propose(a)
    with pytest.raises(ApprovalError):
        rt.approve(a.id, pin="1111")
    assert rt.approve(a.id, pin="2468")["status"] == "executed"


def test_execution_rechecks_policy(rt):
    a = _first(rt, "request_split")
    rt.propose(a)
    rt.update_settings({"paused": True})
    rec = rt.approve(a.id)
    assert rec["status"] == "denied"
    assert a.params["bill_id"] not in rt.world.schedule_overrides
