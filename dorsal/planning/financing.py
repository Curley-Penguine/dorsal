"""Financing-option matching.

For a specific cash need (amount, needed-by date, repay-by date) each
product in the marketplace is:

1. checked for eligibility, with a plain-language reason for every "no";
2. turned into a dated cash-flow schedule (fees, interest, advances,
   holdbacks, amortisation);
3. priced by the *true* annualised cost: the APR implied by the internal
   rate of return of that schedule, so a 1.32 factor rate, a 1.6%-per-30-days
   fee and a 9.8% interest rate can be compared on one scale;
4. checked for timing (does the money arrive before the gap?) and
   affordability (debt service coverage after borrowing);
5. ranked: in time and eligible first, then by total dollar cost.

Products above a 36% APR are flagged as high-cost and never recommended
automatically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from dorsal.models import Business
from dorsal.sandbox.lenders import CATALOG, FinancingProduct

HIGH_COST_APR = 0.36


def irr_apr(flows: list[tuple[date, float]]) -> Optional[float]:
    """Nominal APR (daily IRR x 365) of a borrower's cash-flow schedule."""
    if not flows:
        return None
    t0 = min(d for d, _ in flows)
    pts = [((d - t0).days, a) for d, a in flows]

    if max(t for t, _ in pts) > 4000:
        return None

    def npv(r):
        return sum(a * math.exp(-t * math.log1p(r)) for t, a in pts)
    lo, hi = -0.0009, 0.05   # daily rates
    try:
        f_lo, f_hi = npv(lo), npv(hi)
    except OverflowError:
        return None
    if f_lo * f_hi > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2 * 365


@dataclass
class Need:
    amount: float
    needed_by: date
    repay_by: date


def _eligibility(p: FinancingProduct, biz: Business, fin: dict) -> list[str]:
    reasons = []
    rev = fin["trailing_12m"]["revenue"] or 0
    dscr = fin.get("dscr")
    if p.id == "existing-line":
        if biz.credit_line is None or biz.credit_line.available < p.min_amount:
            reasons.append("No unused credit line available.")
        return reasons
    if biz.months_in_business < p.min_months:
        reasons.append(f"Needs {p.min_months}+ months in business (you have {biz.months_in_business}).")
    if rev < p.min_annual_revenue:
        reasons.append(f"Needs ${p.min_annual_revenue:,.0f}+ yearly revenue.")
    if p.min_dscr and dscr is not None and dscr < p.min_dscr:   # None = no existing debt
        reasons.append(f"Needs debt service coverage of {p.min_dscr}x (you have {dscr}).")
    if (fin["trailing_12m"]["ebitda"] or 0) <= 0 and p.kind in ("revolving", "term_loan"):
        reasons.append("Needs positive operating profit over the last 12 months.")
    if biz.credit_score_band not in p.bands:
        reasons.append(f"Credit band {biz.credit_score_band} is outside this lender's range.")
    if p.requires_card_sales and not biz.has_card_sales:
        reasons.append("Requires daily card sales.")
    if p.requires_invoices and not biz.open_invoices():
        reasons.append("Requires unpaid invoices to borrow against.")
    return reasons


def quote(p: FinancingProduct, biz: Business, fin: dict, need: Need,
          expected_pay_dates: dict[str, date] | None = None, daily_card_sales: float = 0.0) -> dict:
    today = biz.today
    reasons = _eligibility(p, biz, fin)
    max_amt = biz.credit_line.available if p.id == "existing-line" and biz.credit_line else p.max_amount
    amount = max(need.amount, p.min_amount)
    partial = False
    if amount > max_amt:
        partial = True
        amount = max_amt
    fund_date = max(today, need.needed_by - timedelta(days=p.days_to_fund)) + timedelta(days=p.days_to_fund)
    fund_date = max(fund_date, today + timedelta(days=p.days_to_fund))
    arrives_in_time = fund_date <= need.needed_by
    flows: list[tuple[date, float]] = []
    apr_flows: list[tuple[date, float]] = []
    detail = ""

    if p.kind in ("credit_line", "revolving"):
        days = max(1, (need.repay_by - fund_date).days)
        fee = amount * p.origination_fee_pct
        interest = amount * p.apr * days / 365
        flows = [(fund_date, amount - fee), (need.repay_by, -(amount + interest))]
        apr_flows = flows
        detail = f"Draw ${amount:,.0f} on {fund_date}, repay ${amount + interest:,.0f} on {need.repay_by} ({days} days)."
    elif p.kind == "invoice_finance":
        pays = expected_pay_dates or {}
        cands = sorted([i for i in biz.open_invoices() if (today - i.due_date).days <= 30 and i.id in pays],
                       key=lambda i: -i.amount)
        advanced, used = 0.0, []
        for inv in cands:
            if advanced >= amount:
                break
            adv = inv.amount * p.advance_rate
            used.append((inv, adv))
            advanced += adv
        if not used:
            reasons.append("No eligible invoices (must be under 30 days overdue).")
        else:
            if advanced < amount:
                partial = True
            for inv, adv in used:
                pay = max(pays[inv.id], fund_date + timedelta(days=1))
                periods = math.ceil((pay - fund_date).days / 30)
                fee = inv.amount * p.fee_per_30d * periods
                flows += [(fund_date, adv), (pay, -(adv + fee))]
            amount = advanced
            apr_flows = flows
            detail = f"Advance on {len(used)} invoices; repaid automatically as customers pay."
    elif p.kind == "term_loan":
        fee = amount * p.origination_fee_pct
        r = p.apr / 12
        n = max(1, p.term_days // 30)
        pmt = amount * r / (1 - (1 + r) ** -n)
        sched = [(fund_date + timedelta(days=30 * (k + 1)), -pmt) for k in range(n)]
        apr_flows = [(fund_date, amount - fee)] + sched
        # inside the horizon we show the draw and the monthly payments only
        flows = [(fund_date, amount - fee)] + [(d, a) for d, a in sched if d <= need.repay_by]
        detail = f"{n} monthly payments of ${pmt:,.0f}; ${fee:,.0f} upfront fee."
    elif p.kind == "mca":
        repay_total = amount * p.factor_rate
        per_day = max(daily_card_sales * p.holdback_pct, 1.0)
        days = math.ceil(repay_total / per_day)
        if days > 540:
            reasons.append("Card sales are too low to repay this advance.")
            days = 540
        flows = [(fund_date, amount)] + [(fund_date + timedelta(days=k + 1), -per_day) for k in range(days)]
        flows[-1] = (flows[-1][0], -(repay_total - per_day * (days - 1)))
        apr_flows = flows
        detail = f"Repay ${repay_total:,.0f} as {100 * p.holdback_pct:.0f}% of daily card sales (about {days} days)."

    cost = -sum(a for _, a in apr_flows) if apr_flows else None
    apr = irr_apr(apr_flows) if apr_flows else None
    high_cost = apr is not None and apr > HIGH_COST_APR
    # affordability: new annualised debt service against trailing EBITDA
    ebitda = fin["trailing_12m"]["ebitda"] or 0
    debt = fin["trailing_12m"]["debt_service"] or 0
    if p.kind == "term_loan" and apr_flows:
        new_debt = -sum(a for _, a in apr_flows[1:13])
    else:
        new_debt = 0.0
    dscr_after = ebitda / (debt + new_debt) if (debt + new_debt) > 0 else None
    eligible = not reasons
    return {
        "product_id": p.id, "lender": p.lender, "name": p.name, "kind": p.kind,
        "eligible": eligible, "reasons": reasons, "amount": round(amount, 2),
        "funds_on": fund_date.isoformat(), "arrives_in_time": arrives_in_time,
        "total_cost": None if cost is None else round(cost, 2),
        "apr": None if apr is None else round(apr, 4),
        "high_cost": high_cost, "dscr_after": None if dscr_after is None else round(dscr_after, 2),
        "early_repay_ok": p.early_repay_ok, "detail": detail, "notes": p.notes,
        "covers_full_need": not partial and amount >= need.amount - 0.01,
        "flows": [(d.isoformat(), round(a, 2)) for d, a in flows],
    }


def match(biz: Business, fin: dict, need: Need, expected_pay_dates: dict[str, date] | None = None,
          daily_card_sales: float = 0.0, catalog: list[FinancingProduct] | None = None) -> list[dict]:
    quotes = [quote(p, biz, fin, need, expected_pay_dates, daily_card_sales) for p in (catalog or CATALOG)]

    def rank(q):
        usable = q["eligible"] and q["arrives_in_time"] and not q["high_cost"]
        return (0 if usable else 1, 0 if q["eligible"] else 1, q["total_cost"] if q["total_cost"] is not None else 1e12)
    quotes.sort(key=rank)
    for k, q in enumerate(quotes):
        q["rank"] = k + 1
        q["recommended"] = k == 0 and q["eligible"] and q["arrives_in_time"] and not q["high_cost"]
        why = []
        if not q["eligible"]:
            why.append("Not eligible: " + " ".join(q["reasons"]))
        elif not q["arrives_in_time"]:
            why.append(f"Money arrives {q['funds_on']}, after it is needed ({need.needed_by}).")
        if q["high_cost"]:
            why.append(f"High cost: true APR about {100 * q['apr']:.0f}%. Dorsal never recommends this automatically.")
        if q["eligible"] and q["arrives_in_time"] and not q["high_cost"]:
            why.append(f"Costs ${q['total_cost']:,.0f} in total (true APR {100 * (q['apr'] or 0):.1f}%).")
        if q["eligible"] and not q["covers_full_need"]:
            why.append(f"Covers only ${q['amount']:,.0f} of the ${need.amount:,.0f} needed.")
        q["verdict"] = " ".join(why)
    return quotes


def product(pid: str) -> FinancingProduct:
    return next(p for p in CATALOG if p.id == pid)
