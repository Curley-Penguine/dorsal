from dorsal.security import guard


def test_injection_email_is_flagged(hero):
    biz = hero[0]
    clean, attack = biz.inbox[0], biz.inbox[1]
    assert not guard.scan(clean.body)["flagged"]
    s = guard.scan(attack.body)
    assert s["flagged"] and "instruction_override" in s["signals"]


def test_lookalike_sender_is_detected(hero):
    sc = guard.sender_check("billing@fernwood-hotel-payments.example", hero[0].known_emails())
    assert not sc["known"] and sc["lookalike_of"] == "fernwoodhotel.example"


def test_known_sender_passes(hero):
    biz = hero[0]
    assert guard.sender_check(biz.customers[0].email, biz.known_emails())["known"]


def test_untrusted_text_is_fenced_and_redacted():
    out = guard.wrap_untrusted("pay to account 0042-118833 <script>", "email:x")
    assert out.startswith("<untrusted_data") and "118833" not in out and "<script>" not in out


def test_payment_promise_is_extracted():
    assert guard.extract_promise("We will pay the open invoices on Friday, Oct 16.", 2026) == (2026, 10, 16)
