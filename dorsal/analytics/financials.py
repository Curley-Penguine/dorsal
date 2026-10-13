"""Business financial analysis from bank history and the ledger.

Produces the KPIs a lender or accountant would look at, plus a 0-100
health score with plain-language reason codes. Benchmarks are illustrative
industry ranges for the demo, not authoritative statistics.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from statistics import median

from dorsal.models import Business

COGS_VENDOR_CATS = {"cogs", "packaging"}
COGS_TXN_CATS = {"supplies"}
OPEX_TXN_CATS = {"payroll", "rent", "utilities", "insurance", "services", "equipment"}

# Illustrative ranges used for context in the demo (low, high).
BENCHMARKS = {
    "Coffee roasting and cafe": {"gross_margin": (0.55, 0.70), "ebitda_margin": (0.06, 0.15), "dso": (20, 40)},
    "Restaurant": {"gross_margin": (0.60, 0.70), "ebitda_margin": (0.05, 0.12), "dso": (0, 15)},
    "Wholesale distribution": {"gross_margin": (0.20, 0.35), "ebitda_margin": (0.03, 0.08), "dso": (30, 55)},
    "Design and marketing agency": {"gross_margin": (0.70, 0.90), "ebitda_margin": (0.08, 0.20), "dso": (35, 60)},
    "Online retail": {"gross_margin": (0.35, 0.55), "ebitda_margin": (0.04, 0.12), "dso": (0, 10)},
}


def _window(biz: Business, days: int, end: date | None = None) -> tuple[date, date]:
    end = end or biz.today
    return end - timedelta(days=days), end


def _pnl(biz: Business, start: date, end: date) -> dict:
    card = sum(t.amount for t in biz.transactions if t.category == "sales" and start <= t.date < end)
    credit = sum(i.amount for i in biz.invoices if start <= i.issue_date < end)
    revenue = card + credit
    cogs_bills = sum(b.amount for b in biz.bills
                     if start <= b.issue_date < end and biz.vendor(b.vendor_id).category in COGS_VENDOR_CATS)
    cogs_cash = -sum(t.amount for t in biz.transactions if t.category in COGS_TXN_CATS and start <= t.date < end)
    cogs = cogs_bills + cogs_cash
    opex = -sum(t.amount for t in biz.transactions if t.category in OPEX_TXN_CATS and start <= t.date < end)
    taxes = -sum(t.amount for t in biz.transactions if t.category == "tax" and start <= t.date < end)
    debt = -sum(t.amount for t in biz.transactions if t.category == "loan" and start <= t.date < end)
    ebitda = revenue - cogs - opex - taxes
    return {"revenue": revenue, "card_sales": card, "credit_sales": credit, "cogs": cogs, "opex": opex,
            "taxes": taxes, "debt_service": debt, "ebitda": ebitda,
            "gross_margin": (revenue - cogs) / revenue if revenue else None,
            "ebitda_margin": ebitda / revenue if revenue else None}


def analyze(biz: Business) -> dict:
    today = biz.today
    y = _pnl(biz, *_window(biz, 365))
    q = _pnl(biz, *_window(biz, 91))
    q_prev = _pnl(biz, today - timedelta(days=182), today - timedelta(days=91))

    # working capital ---------------------------------------------------------
    open_ar = sum(i.amount for i in biz.open_invoices())
    open_ap = sum(b.amount for b in biz.open_bills() if biz.vendor(b.vendor_id).category != "tax")
    credit_90 = q["credit_sales"]
    cogs_bills_90 = sum(b.amount for b in biz.bills if today - timedelta(days=91) <= b.issue_date < today
                        and biz.vendor(b.vendor_id).category in COGS_VENDOR_CATS)
    dso = open_ar / credit_90 * 91 if credit_90 else 0.0
    dpo = open_ap / cogs_bills_90 * 91 if cogs_bills_90 else 0.0
    outflow_90 = -sum(t.amount for t in biz.transactions if t.amount < 0 and t.date >= today - timedelta(days=91))
    daily_out = outflow_90 / 91 if outflow_90 else 1.0
    cash_days = biz.cash / daily_out
    due_30 = sum(b.amount for b in biz.open_bills() if b.due_date <= today + timedelta(days=30))
    due_30 += sum(r.amount * len(r.occurrences(today, today + timedelta(days=30))) for r in biz.recurring)
    quick_ratio = (biz.cash + open_ar) / due_30 if due_30 else None
    dscr = y["ebitda"] / y["debt_service"] if y["debt_service"] else None

    # collections -------------------------------------------------------------
    per_customer = []
    credit_by_c = defaultdict(float)
    for inv in biz.invoices:
        if inv.issue_date >= today - timedelta(days=365):
            credit_by_c[inv.customer_id] += inv.amount
    total_credit = sum(credit_by_c.values()) or 1.0
    for c in biz.customers:
        lates = [i.days_late() for i in biz.invoices if i.customer_id == c.id and i.paid_date]
        opens = [i for i in biz.open_invoices() if i.customer_id == c.id]
        overdue = [i for i in opens if i.due_date < today]
        per_customer.append({
            "customer_id": c.id, "name": c.name, "terms_days": c.terms_days,
            "share_of_credit_sales": round(credit_by_c[c.id] / total_credit, 3),
            "paid_invoices": len(lates),
            "median_days_late": median(lates) if lates else None,
            "on_time_rate": round(sum(1 for x in lates if x <= 0) / len(lates), 2) if lates else None,
            "open_amount": round(sum(i.amount for i in opens), 2),
            "overdue_amount": round(sum(i.amount for i in overdue), 2),
        })
    per_customer.sort(key=lambda r: -r["share_of_credit_sales"])
    shares = [r["share_of_credit_sales"] * (y["credit_sales"] / y["revenue"] if y["revenue"] else 0) for r in per_customer]
    hhi = sum(s * s for s in shares)
    top_share = shares[0] if shares else 0.0

    # monthly trend --------------------------------------------------------------
    monthly = []
    first = date(today.year, today.month, 1)
    starts, yy, mm = [], first.year, first.month
    for _ in range(12):
        mm -= 1
        if mm == 0:
            mm, yy = 12, yy - 1
        starts.append(date(yy, mm, 1))
    starts.reverse()
    for i, m_start in enumerate(starts):
        m_end = starts[i + 1] if i + 1 < len(starts) else first
        p = _pnl(biz, m_start, m_end)
        monthly.append({"month": m_start.strftime("%Y-%m"), "revenue": round(p["revenue"], 0),
                        "gross_margin": round(p["gross_margin"], 3) if p["gross_margin"] is not None else None,
                        "ebitda": round(p["ebitda"], 0)})

    # expense mix (last 90 days) ------------------------------------------------
    mix = defaultdict(float)
    for t in biz.transactions:
        if t.amount < 0 and t.date >= today - timedelta(days=91):
            mix[t.category] += -t.amount
    expense_mix = sorted(({"category": k, "amount": round(v, 0), "share": round(v / outflow_90, 3)}
                          for k, v in mix.items()), key=lambda r: -r["amount"])

    # health score ------------------------------------------------------------------
    reasons = []

    def clamp(x):
        return max(0.0, min(100.0, x))
    liquidity = clamp(cash_days / 30 * 100)
    if cash_days < 14:
        reasons.append(f"Cash covers only {cash_days:.0f} days of spending (aim for 30+).")
    profit = clamp(((y["ebitda_margin"] or 0) + 0.02) / 0.14 * 100)
    if (y["ebitda_margin"] or 0) < 0.05:
        reasons.append(f"Operating margin is thin at {100 * (y['ebitda_margin'] or 0):.1f}%.")
    coverage = clamp(((dscr or 3) - 1.0) / 1.0 * 100)
    if dscr is not None and dscr < 1.25:
        reasons.append(f"Debt service coverage is {dscr:.2f}x; lenders usually look for 1.25x or more.")
    avg_terms = (sum(c.terms_days * credit_by_c[c.id] for c in biz.customers) / total_credit) if biz.customers else 30
    collections = clamp(100 - max(0.0, dso - avg_terms) * 4) if credit_90 else 100.0
    if credit_90 and dso > avg_terms + 7:
        reasons.append(f"Customers take {dso:.0f} days to pay on average versus {avg_terms:.0f}-day terms.")
    concentration = clamp(100 - max(0.0, top_share - 0.15) * 300)
    if top_share > 0.25:
        reasons.append(f"Your largest customer is {100 * top_share:.0f}% of revenue.")
    score = round(0.3 * liquidity + 0.25 * profit + 0.15 * coverage + 0.2 * collections + 0.1 * concentration)
    growth = (q["revenue"] / q_prev["revenue"] - 1) if q_prev["revenue"] else None
    gm_change = ((q["gross_margin"] or 0) - (q_prev["gross_margin"] or 0)) if q_prev["revenue"] else None

    r = lambda x, n=3: None if x is None else round(x, n)  # noqa: E731
    return {
        "as_of": today.isoformat(),
        "trailing_12m": {k: r(v, 3 if "margin" in k else 0) for k, v in y.items()},
        "last_quarter": {k: r(v, 3 if "margin" in k else 0) for k, v in q.items()},
        "revenue_growth_qoq": r(growth),
        "gross_margin_change_qoq": r(gm_change),
        "working_capital": {"cash": round(biz.cash, 0), "open_receivables": round(open_ar, 0),
                            "open_payables": round(open_ap, 0), "dso_days": round(dso, 1), "dpo_days": round(dpo, 1),
                            "cash_conversion_gap_days": round(dso - dpo, 1), "cash_days_on_hand": round(cash_days, 1),
                            "quick_ratio_30d": r(quick_ratio, 2)},
        "dscr": r(dscr, 2),
        "customers": per_customer,
        "concentration": {"top_customer_share_of_revenue": round(top_share, 3), "hhi": round(hhi, 4)},
        "monthly": monthly,
        "expense_mix_90d": expense_mix,
        "health": {"score": score, "components": {"liquidity": round(liquidity), "profitability": round(profit),
                                                  "debt_coverage": round(coverage), "collections": round(collections),
                                                  "concentration": round(concentration)},
                   "reasons": reasons},
        "benchmarks": BENCHMARKS.get(biz.industry, {}),
    }
