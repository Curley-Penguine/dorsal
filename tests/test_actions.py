from datetime import date

from dorsal.actions import Action


def test_payload_hash_is_stable_and_sensitive():
    a = Action("offer_discount", {"invoice_id": "INV-1", "pct": 0.02, "deadline": date(2026, 11, 18)}, counterparty="x@y.example")
    b = Action("offer_discount", {"invoice_id": "INV-1", "pct": 0.02, "deadline": date(2026, 11, 18)}, counterparty="x@y.example")
    c = Action("offer_discount", {"invoice_id": "INV-1", "pct": 0.25, "deadline": date(2026, 11, 18)}, counterparty="x@y.example")
    assert a.payload_hash() == b.payload_hash()
    assert a.payload_hash() != c.payload_hash()        # any change to the parameters changes the hash


def test_tiers():
    assert Action("send_reminder", {}).tier == "notify"
    assert Action("draw_financing", {}).tier == "borrow"
    assert Action("pay_new_payee", {}).tier == "forbidden"
    assert Action("something_unknown", {}).tier == "forbidden"   # unknown types default to forbidden
