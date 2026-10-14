from dorsal.analytics.alerts import build_alerts, find_duplicate_bills, payment_promises


def test_demo_alerts(hero, fc, fin):
    alerts = build_alerts(hero[0], fc.run(), fin)
    codes = [a["code"] for a in alerts]
    for code in ("CASH_SHORTFALL", "AP_DUPLICATE", "SECURITY_SUSPICIOUS_MESSAGE", "AR_OVERDUE", "AR_PROMISE"):
        assert code in codes
    severities = [a["severity"] for a in alerts]
    order = {"critical": 0, "warning": 1, "info": 2}
    assert severities == sorted(severities, key=order.get)       # most urgent first


def test_duplicate_bill_found(hero):
    pairs = find_duplicate_bills(hero[0])
    assert ("B-903", "B-904") in pairs


def test_promise_only_from_known_customer(hero):
    promises = payment_promises(hero[0])
    assert promises and all(p["customer"] == "Northshore Grocers" for p in promises)
