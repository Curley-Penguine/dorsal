"""Synthetic SME world generator.

Produces a `Business` (what Dorsal can observe: 12 months of bank history,
open invoices and bills, an inbox) plus a hidden `WorldTruth` (the true
behavioural parameters and the realised next 91 days). The truth is used
only by the evaluation harness to score plans against what "actually"
happens; the forecasting and planning code never sees it.

All businesses, people and lenders are fictional.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from dorsal.models import (Bill, Business, CreditLine, Customer, Invoice, Message,
                         Recurring, Transaction, Vendor)

HORIZON = 91


@dataclass
class Event:
    day: date
    amount: float          # + in, - out
    kind: str              # sales | collection | bill | recurring | new_collection | new_bill
    ref: str = ""


@dataclass
class WorldTruth:
    delay_mean: dict[str, float]
    delay_shape: float
    discount_accept_p: dict[str, float]
    reminder_factor: float
    future: list[Event] = field(default_factory=list)


# --------------------------------------------------------------------------
# Archetypes: rough, illustrative shapes of common small businesses.
# weekday factors are Mon..Sun.
# --------------------------------------------------------------------------
ARCHETYPES: dict[str, dict] = {
    "coffee_roaster": dict(
        industry="Coffee roasting and cafe", retail=1420, weekday=[0.92, 0.88, 0.9, 0.95, 1.0, 1.4, 1.15],
        season={1: .9, 2: .9, 3: .95, 4: 1, 5: 1, 6: .97, 7: .95, 8: .95, 9: 1, 10: 1.03, 11: 1.08, 12: 1.2},
        b2b_weekly=5600, n_customers=(5, 7), payroll=12800, rent=5900, loan=2350, utilities=1450,
        supplies_weekly=1300, cogs_ratio=0.22, packaging_ratio=0.05, tax_ratio=0.05, band="B"),
    "restaurant": dict(
        industry="Restaurant", retail=3100, weekday=[0.7, 0.75, 0.85, 0.95, 1.25, 1.45, 1.05],
        season={1: .85, 2: .88, 3: .95, 4: 1, 5: 1.05, 6: 1.08, 7: 1.1, 8: 1.08, 9: 1, 10: .98, 11: .98, 12: 1.15},
        b2b_weekly=1200, n_customers=(2, 3), payroll=19000, rent=9800, loan=1800, utilities=2600,
        supplies_weekly=1600, cogs_ratio=0.24, packaging_ratio=0.02, tax_ratio=0.05, band="C"),
    "distributor": dict(
        industry="Wholesale distribution", retail=0, weekday=[1] * 7, season={m: 1 for m in range(1, 13)},
        b2b_weekly=26000, n_customers=(8, 12), payroll=8000, rent=7200, loan=4100, utilities=1300,
        supplies_weekly=600, cogs_ratio=0.57, packaging_ratio=0.02, tax_ratio=0.03, band="B"),
    "agency": dict(
        industry="Design and marketing agency", retail=0, weekday=[1] * 7, season={m: 1 for m in range(1, 13)},
        b2b_weekly=15500, n_customers=(4, 8), payroll=19000, rent=4800, loan=0, utilities=900,
        supplies_weekly=300, cogs_ratio=0.08, packaging_ratio=0.0, tax_ratio=0.05, band="A"),
    "ecommerce": dict(
        industry="Online retail", retail=2300, weekday=[1.1, 1.05, 1, 1, 0.95, 0.9, 1.0],
        season={1: .8, 2: .85, 3: .9, 4: .95, 5: 1, 6: 1, 7: .95, 8: 1, 9: 1, 10: 1.05, 11: 1.35, 12: 1.45},
        b2b_weekly=0, n_customers=(0, 0), payroll=9800, rent=2400, loan=1200, utilities=700,
        supplies_weekly=900, cogs_ratio=0.45, packaging_ratio=0.06, tax_ratio=0.05, band="C"),
}

CUSTOMER_NAMES = ["Alder Street Market", "Birchwood Deli", "Cobalt Hotels", "Driftwood Grocers", "Elm & Oak Cafe",
                  "Fairview Office Co", "Granite Fitness", "Hillside Bakery", "Ironbark Kitchens", "Juniper Hall",
                  "Kestrel Foods", "Lumen Studios", "Maple Row Inns", "Northgate Retail", "Orchard Lane Co-op"]
DELAY_MEANS = [0, 3, 6, 10, 15, 22, 30]


def _gamma_days(rng: np.random.Generator, mean_late: float, shape: float, size=None):
    """Days late, allowing a little early payment (-2 days)."""
    scale = max(mean_late + 2.0, 0.5) / shape
    return np.round(rng.gamma(shape, scale, size=size)) - 2


def generate(archetype: str = "coffee_roaster", seed: int = 0, today: date = date(2026, 10, 12),
             scale: float | None = None, cash_weeks: float | None = None,
             customers_spec: list[dict] | None = None, name: str | None = None,
             shock: bool = False) -> tuple[Business, WorldTruth]:
    """Generate one business and its hidden truth."""
    rng = np.random.default_rng(seed)
    a = ARCHETYPES[archetype]
    if scale is None:
        scale = float(np.exp(rng.normal(0, 0.35)))
    start = today - timedelta(days=365)
    end = today + timedelta(days=HORIZON - 1)
    events: list[Event] = []            # every cash event over the whole window
    invoices: list[Invoice] = []
    bills: list[Bill] = []

    # ---- customers --------------------------------------------------------
    customers: list[Customer] = []
    delay_mean: dict[str, float] = {}
    accept_p: dict[str, float] = {}
    order: dict[str, tuple[float, float]] = {}  # weekly prob, mean amount
    if customers_spec is None:
        lo, hi = a["n_customers"]
        n = int(rng.integers(lo, hi + 1)) if hi else 0
        names = rng.choice(CUSTOMER_NAMES, size=n, replace=False) if n else []
        weights = rng.dirichlet(np.ones(n) * 1.3) if n else []
        customers_spec = []
        for k, nm in enumerate(names):
            customers_spec.append(dict(
                name=str(nm), terms=int(rng.choice([15, 30, 30, 45])), late=float(rng.choice(DELAY_MEANS)),
                p=float(rng.uniform(0.35, 0.9)), share=float(weights[k]), accept=float(rng.uniform(0.3, 0.8))))
    total_b2b = a["b2b_weekly"] * scale
    for k, spec in enumerate(customers_spec):
        cid = f"C{k + 1:02d}"
        slug = spec["name"].lower().replace("&", "and").replace("'", "").replace(" ", "")
        customers.append(Customer(cid, spec["name"], spec.get("email", f"ap@{slug}.example"), spec["terms"]))
        delay_mean[cid] = spec["late"]
        accept_p[cid] = spec.get("accept", 0.55)
        weekly_value = spec.get("weekly", total_b2b * spec.get("share", 1 / max(1, len(customers_spec))))
        p = spec.get("p", 0.6)
        order[cid] = (p, weekly_value / p)
    shape = 2.5

    # ---- invoices (issued Mondays/Thursdays) --------------------------------
    inv_no = 1000
    d = start
    while d <= end:
        if d.weekday() in (0, 3):
            for c in customers:
                p, mean_amt = order[c.id]
                if rng.random() < p / 2:  # two issue days per week
                    amt = round(float(mean_amt * rng.lognormal(0, 0.25)), -1)
                    inv_no += 1
                    due = d + timedelta(days=c.terms_days)
                    late = float(_gamma_days(rng, delay_mean[c.id], shape))
                    paid = due + timedelta(days=int(late)) if rng.random() > 0.01 else None
                    inv = Invoice(f"INV-{inv_no}", c.id, amt, d, due)
                    inv._true_paid = paid  # type: ignore[attr-defined]
                    invoices.append(inv)
        d += timedelta(days=1)

    # ---- daily card / cafe sales -------------------------------------------
    retail = a["retail"] * scale
    revenue_by_month: dict[tuple[int, int], float] = {}
    if retail:
        level_noise = 1.0
        d = start
        while d <= end:
            level_noise = 0.9 * level_noise + 0.1 * float(rng.lognormal(0, 0.08))
            amt = retail * a["weekday"][d.weekday()] * a["season"][d.month] * level_noise * float(rng.lognormal(0, 0.12))
            events.append(Event(d, round(amt, 2), "sales"))
            key = (d.year, d.month)
            revenue_by_month[key] = revenue_by_month.get(key, 0) + amt
            d += timedelta(days=1)
    for inv in invoices:
        key = (inv.issue_date.year, inv.issue_date.month)
        revenue_by_month[key] = revenue_by_month.get(key, 0) + inv.amount
        tp = inv._true_paid  # type: ignore[attr-defined]
        if tp is not None and tp <= end:
            events.append(Event(tp, inv.amount, "collection", inv.id))

    # ---- vendors and their bills --------------------------------------------
    vendors = [
        Vendor("V01", "Primary supplier", "orders@primary-supplier.example", "cogs", split_allowed=bool(rng.random() < 0.5)),
        Vendor("V02", "Packaging supplier", "billing@packaging.example", "packaging"),
        Vendor("V03", "Business insurer", "accounts@insurer.example", "insurance"),
        Vendor("V09", "Tax authority", "remit@tax-authority.example", "tax", protected=True),
    ]
    bill_no = 300

    def add_bill(vendor_id: str, amount: float, issue: date, terms: int, discretionary=False, ref=None, memo=""):
        nonlocal bill_no
        bill_no += 1
        b = Bill(f"B-{bill_no}", vendor_id, round(amount, 2), issue, issue + timedelta(days=terms),
                 reference=ref or f"{vendor_id}-{bill_no}", discretionary=discretionary, memo=memo)
        bills.append(b)
        return b

    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        prev = (y, m - 1) if m > 1 else (y - 1, 12)
        base_rev = revenue_by_month.get(prev) or revenue_by_month.get((y, m), 0)
        if a["cogs_ratio"] and base_rev:
            add_bill("V01", a["cogs_ratio"] * base_rev * float(rng.lognormal(0, 0.1)), date(y, m, 8), 30)
        if a["packaging_ratio"] and base_rev:
            add_bill("V02", a["packaging_ratio"] * base_rev * float(rng.lognormal(0, 0.1)), date(y, m, 12), 21)
        if m in (1, 4, 7, 10):
            add_bill("V03", 950 * scale, date(y, m, 3), 30)
        if m in (1, 4, 7, 10):  # quarterly sales-tax remittance on the previous quarter
            q_months = [((y, mm) if mm > 0 else (y - 1, mm + 12)) for mm in (m - 3, m - 2, m - 1)]
            q_rev = sum(revenue_by_month.get(k, 0) for k in q_months)
            if q_rev:
                issue = date(y, m, 1)
                add_bill("V09", a["tax_ratio"] * q_rev, issue, 59)
        m += 1
        if m > 12:
            m, y = 1, y + 1

    # ---- recurring obligations ----------------------------------------------
    recurring = [
        Recurring("R-PAY", "Payroll", "payroll", round(a["payroll"] * scale, -1), "biweekly", date(2024, 1, 5), protected=True),
        Recurring("R-RENT", "Rent", "rent", round(a["rent"] * scale, -1), "monthly", date(2024, 1, 1), protected=True),
        Recurring("R-UTIL", "Utilities and software", "utilities", round(a["utilities"] * scale, -1), "monthly", date(2024, 1, 20)),
        Recurring("R-SUP", "Milk and supplies", "supplies", round(a["supplies_weekly"] * scale, -1), "weekly", date(2024, 1, 3)),
    ]
    if a["loan"]:
        recurring.append(Recurring("R-LOAN", "Equipment loan", "loan", round(a["loan"] * scale, -1), "monthly", date(2024, 1, 15), protected=True))

    # ---- optional stress shock (used by the evaluation harness) -------------
    if shock:
        kind = rng.choice(["inventory", "equipment", "big_order"])
        when = today + timedelta(days=int(rng.integers(21, 60)))
        monthly_cost = (a["payroll"] * 2.17 + a["rent"]) * scale
        if kind == "equipment":
            vendors.append(Vendor("V05", "Equipment dealer", "sales@equipment.example", "equipment"))
            add_bill("V05", monthly_cost * rng.uniform(0.4, 0.8), when - timedelta(days=30), 30, discretionary=True)
        else:
            amt = monthly_cost * rng.uniform(0.6, 1.2)
            add_bill("V01", amt, today - timedelta(days=int(rng.integers(3, 15))), int((when - today).days) + 10)

    # ---- realise recurring + bills as cash events ------------------------------
    for r in recurring:
        for dd in r.occurrences(start, end):
            amt = r.amount * (float(rng.lognormal(0, 0.05)) if r.category == "supplies" else 1.0)
            events.append(Event(dd, -round(amt, 2), "recurring", r.id))
    for b in bills:
        if b.due_date <= end:
            events.append(Event(b.due_date, -b.amount, "bill", b.id))

    # ---- split history / future at `today` ------------------------------------
    business_invoices, business_bills, txns = [], [], []
    future: list[Event] = []
    visible_invoice_ids, visible_bill_ids = set(), set()
    for inv in invoices:
        if inv.issue_date < today:
            tp = inv._true_paid  # type: ignore[attr-defined]
            inv.paid_date = tp if (tp is not None and tp < today) else None
            business_invoices.append(inv)
            visible_invoice_ids.add(inv.id)
    for b in bills:
        if b.issue_date < today:
            b.paid_date = b.due_date if b.due_date < today else None
            business_bills.append(b)
            visible_bill_ids.add(b.id)
    cust_name = {c.id: c.name for c in customers}
    vend_name = {v.id: v.name for v in vendors}
    rec_by_id = {r.id: r for r in recurring}
    inv_by_id = {i.id: i for i in invoices}
    bill_by_id = {b.id: b for b in bills}
    for ev in events:
        if ev.day < today:
            if ev.kind == "sales":
                txns.append(Transaction(ev.day, ev.amount, "sales", "Card and cafe sales"))
            elif ev.kind == "collection":
                txns.append(Transaction(ev.day, ev.amount, "collection", cust_name[inv_by_id[ev.ref].customer_id], ev.ref))
            elif ev.kind == "bill":
                vb = bill_by_id[ev.ref]
                txns.append(Transaction(ev.day, ev.amount, vendor_category(vendors, vb.vendor_id), vend_name[vb.vendor_id], ev.ref))
            else:
                r = rec_by_id[ev.ref]
                txns.append(Transaction(ev.day, ev.amount, r.category, r.name, r.id))
        else:
            kind = ev.kind
            if kind == "collection" and ev.ref not in visible_invoice_ids:
                kind = "new_collection"
            if kind == "bill" and ev.ref not in visible_bill_ids:
                kind = "new_bill"
            future.append(Event(ev.day, ev.amount, kind, ev.ref))
    txns.sort(key=lambda t: t.date)
    future.sort(key=lambda e: e.day)

    # opening balance chosen so that cash today = `cash_weeks` of fixed costs
    weekly_fixed = (sum(r.amount * {"weekly": 1, "biweekly": .5, "monthly": 12 / 52}[r.kind] for r in recurring))
    if cash_weeks is None:
        cash_weeks = float(rng.uniform(1.5, 5))
    cash = round(weekly_fixed * cash_weeks, 2)
    biz = Business(
        id=f"biz-{archetype}-{seed}", name=name or f"{a['industry']} #{seed}", industry=a["industry"],
        today=today, cash=cash, months_in_business=int(rng.integers(14, 120)), credit_score_band=a["band"],
        customers=customers, vendors=vendors, invoices=business_invoices, bills=business_bills,
        recurring=recurring, transactions=txns,
        credit_line=CreditLine("Community credit union", round(weekly_fixed * 3, -3), 0.115),
        has_card_sales=bool(retail))
    truth = WorldTruth(delay_mean, shape, accept_p, reminder_factor=float(rng.uniform(0.45, 0.75)), future=future)
    return biz, truth


def vendor_category(vendors: list[Vendor], vid: str) -> str:
    cat = next(v.category for v in vendors if v.id == vid)
    return {"cogs": "cogs", "packaging": "cogs", "tax": "tax"}.get(cat, cat)
