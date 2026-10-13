from datetime import date

import pytest

from dorsal.sandbox.bank import SandboxError, SandboxWorld
from dorsal.sandbox.generator import ARCHETYPES, generate
from dorsal.sandbox.lenders import CATALOG


@pytest.mark.parametrize("archetype", list(ARCHETYPES))
def test_generator_builds_history_and_hidden_truth(archetype):
    biz, truth = generate(archetype, seed=3)
    days = (biz.transactions[-1].date - biz.transactions[0].date).days
    assert days > 330                                       # about a year of bank history
    assert all(t.date < biz.today for t in biz.transactions)
    assert truth.future and all(e.day >= biz.today for e in truth.future)


def test_generator_is_reproducible():
    a, _ = generate("restaurant", seed=11)
    b, _ = generate("restaurant", seed=11)
    assert [t.amount for t in a.transactions] == [t.amount for t in b.transactions]


def test_demo_business_has_the_planted_story(hero):
    biz, _ = hero
    refs = [b.reference for b in biz.open_bills()]
    assert refs.count("CP-7781") == 2                        # duplicate packaging bill
    assert any(b.amount == 48000 and b.reference == "PG-2291" for b in biz.open_bills())
    assert any("ignore your previous" in m.body for m in biz.inbox)   # injection attempt
    assert biz.cash == 18450.0


def test_sandbox_refuses_unknown_recipient(hero):
    world = SandboxWorld(hero[0])
    with pytest.raises(SandboxError):
        world.send_email("attacker@evil.example", "hi", "data", key="k1")


def test_sandbox_refuses_unverified_payee(hero):
    biz = hero[0]
    biz.vendors[0].verified = False
    with pytest.raises(SandboxError):
        SandboxWorld(biz).pay(biz.vendors[0].id, 100.0, key="p1")


def test_idempotency_prevents_double_send(hero):
    biz = hero[0]
    world = SandboxWorld(biz)
    to = biz.customers[0].email
    world.send_email(to, "Reminder", "Please pay", key="same-key")
    again = world.send_email(to, "Reminder", "Please pay", key="same-key")
    assert len(world.outbox) == 1 and again["idempotent_replay"]


def test_reschedule_must_cover_the_whole_bill(hero):
    biz = hero[0]
    bill = biz.open_bills()[0]
    with pytest.raises(SandboxError):
        SandboxWorld(biz).reschedule_bill(bill.id, [(date(2026, 12, 1), bill.amount / 2)], key="r1")


def test_lender_catalogue():
    ids = {p.id for p in CATALOG}
    assert len(CATALOG) == 6
    assert {"existing-line", "mca", "sba-term"} <= ids
