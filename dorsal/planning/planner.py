"""Turning a forecast into a plan.

1. `candidates()` generates every lever the business actually has:
   reminders on overdue invoices, holds on suspected duplicate bills,
   early-payment discounts for invoices that would land after the low point,
   supplier split-payment options written into supply agreements, deferral
   of discretionary purchases, and financing.
2. `optimize()` solves a chance-constrained problem:

       minimise   expected $ cost + goodwill penalty
       subject to P(balance < buffer at any point in 13 weeks) <= target_p_buffer
                  P(balance < 0     at any point in 13 weeks) <= target_p_short

   It runs a greedy pass over non-financing levers ranked by risk reduction
   per dollar, sizes financing (if still needed) with a binary search per
   eligible product, then prunes any step the plan no longer needs. Every
   evaluation re-runs the Monte Carlo forecast with common random numbers.
3. Each step in the final plan gets a leave-one-out attribution: how much
   worse the plan is without it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np

from dorsal.actions import Action
from dorsal.analytics.alerts import find_duplicate_bills, payment_promises
from dorsal.analytics.forecast import Forecaster, ForecastResult, Overlay
from dorsal.models import Business
from dorsal.planning import financing as fin_mod

GOODWILL_DOLLARS = 1500.0   # how many dollars one unit of goodwill cost is worth


@dataclass
class Targets:
    buffer: float = 12000.0
    max_p_below_buffer: float = 0.10
    max_p_shortfall: float = 0.02
    max_borrow: float = 25000.0


def _money(x: float) -> str:
    return ("-$" if x < 0 else "$") + f"{abs(x):,.0f}"


def known_info_overlay(biz: Business) -> Overlay:
    """Information Dorsal treats as fact before planning: payment promises
    from verified customer addresses."""
    ov = Overlay()
    for p in payment_promises(biz):
        ov.promises[p["invoice_id"]] = date.fromisoformat(p["promised_date"])
    return ov


def _low_info(res: ForecastResult, buffer: float) -> tuple[int, int]:
    p50 = res.quantiles()[50]
    low = res.risk_from + int(np.argmin(p50[res.risk_from:]))
    p_day = (res.balances < buffer).mean(axis=0)
    first = next((i for i, p in enumerate(p_day) if p >= 0.1), low)
    return low, first


def candidates(biz: Business, fc: Forecaster, base: ForecastResult, targets: Targets) -> list[Action]:
    today = biz.today
    low_i, first_i = _low_info(base, targets.buffer)
    low_day = today + timedelta(days=low_i)
    first_risk = today + timedelta(days=first_i)
    acts: list[Action] = []

    # 1. reminders, one email per customer with overdue invoices (3-60 days)
    by_c: dict[str, list] = {}
    for inv in biz.open_invoices():
        od = (today - inv.due_date).days
        if 3 <= od <= 60:
            by_c.setdefault(inv.customer_id, []).append(inv)
    for cid, invs in by_c.items():
        c = biz.customer(cid)
        lines = "\n".join(f"- {i.id}: ${i.amount:,.2f}, due {i.due_date:%b %d}" for i in invs)
        total = sum(i.amount for i in invs)
        acts.append(Action(
            "send_reminder", {"customer_id": cid, "invoice_ids": [i.id for i in invs]},
            title=f"Remind {c.name} about {len(invs)} overdue invoice{'s' if len(invs) > 1 else ''} ({_money(total)})",
            rationale=f"{c.name} is past due. A polite reminder with a pay link usually shortens the wait.",
            counterparty=c.email, relationship_cost=0.02,
            overlay=Overlay(reminders={i.id for i in invs}),
            draft=(f"Subject: Friendly reminder from {biz.name}\n\nHi,\n\nA quick reminder that these invoices are now "
                   f"past due:\n{lines}\n\nYou can pay online with the link below, or reply if anything needs fixing."
                   f"\n\nThank you,\n{biz.name}")))

    # 2. holds on suspected duplicate bills
    for orig, dup in find_duplicate_bills(biz):
        b, v = biz.bill(dup), biz.vendor(biz.bill(dup).vendor_id)
        acts.append(Action(
            "hold_bill", {"bill_id": dup, "duplicate_of": orig},
            title=f"Hold {v.name} bill {dup} ({_money(b.amount)}) as a likely duplicate",
            rationale=f"Same vendor reference ({b.reference}) as {orig}. Holding is reversible; "
                      "Dorsal asks the vendor to confirm.",
            counterparty=v.email, overlay=Overlay(holds={dup}),
            draft=(f"Subject: Possible duplicate invoice {b.reference}\n\nHi,\n\nWe received invoice {b.reference} "
                   f"for ${b.amount:,.2f} twice. We are paying the first copy on its due date and holding the second. "
                   f"Please let us know if the second is a separate charge.\n\nThanks,\n{biz.name}")))

    # 3. early-payment discounts for big invoices expected to land after the low point
    pays = fc.expected_pay_dates()
    for inv in biz.open_invoices():
        exp = pays.get(inv.id)
        if inv.amount < 2000 or exp is None or exp <= low_day - timedelta(days=1):
            continue
        deadline = min(inv.due_date, low_day - timedelta(days=2))
        if deadline < today + timedelta(days=7):
            continue
        pct = 0.02 if deadline >= inv.due_date - timedelta(days=3) else 0.025
        c = biz.customer(inv.customer_id)
        exp_cost = fc.a.discount_accept_p * pct * inv.amount
        acts.append(Action(
            "offer_discount", {"invoice_id": inv.id, "pct": pct, "deadline": deadline},
            title=f"Offer {c.name} {100 * pct:g}% off {inv.id} if paid by {deadline:%b %d}",
            rationale=f"{c.name} usually pays about {fc.delay_stats.get(c.id, {}).get('median_days_late') or 0:.0f} days "
                      f"late, so {_money(inv.amount)} would likely arrive around {exp:%b %d}, after the low point on "
                      f"{low_day:%b %d}. The discount costs {_money(pct * inv.amount)} if accepted.",
            cost=exp_cost, counterparty=c.email,
            overlay=Overlay(offers={inv.id: {"pct": pct, "deadline": deadline}}),
            draft=(f"Subject: {100 * pct:g}% off {inv.id} if paid by {deadline:%b %d}\n\nHi,\n\nInvoice {inv.id} "
                   f"(${inv.amount:,.2f}) is due {inv.due_date:%b %d}. If it is paid by {deadline:%b %d} we will take "
                   f"{100 * pct:g}% off, so you would pay ${inv.amount * (1 - pct):,.2f}. The pay link below shows the "
                   f"discounted amount until then.\n\nThank you,\n{biz.name}")))

    # 4. supplier split payments allowed by the supply agreement
    for b in biz.open_bills():
        v = biz.vendor(b.vendor_id)
        if v.protected or not v.split_allowed or b.amount < 10000 or b.due_date > low_day + timedelta(days=7):
            continue
        half = round(b.amount / 2, 2)
        second = b.due_date + timedelta(days=v.split_terms_days)
        payments = [(b.due_date, half), (second, round(b.amount - half, 2))]
        acts.append(Action(
            "request_split", {"bill_id": b.id, "payments": payments},
            title=f"Use the split-payment option on {v.name} {b.reference} ({_money(b.amount)})",
            rationale=f"Your agreement with {v.name} allows paying large orders in two halves. Paying "
                      f"{_money(half)} on {b.due_date:%b %d} and the rest on {second:%b %d} moves half the bill past "
                      f"the low point at no cost.",
            counterparty=v.email, relationship_cost=0.15,
            overlay=Overlay(bill_schedule={b.id: payments}),
            draft=(f"Subject: Split payment for {b.reference}\n\nHi,\n\nWe would like to use the split-payment option in "
                   f"our supply agreement for {b.reference} (${b.amount:,.2f}): ${half:,.2f} on {b.due_date:%b %d} and "
                   f"${b.amount - half:,.2f} on {second:%b %d}. Please confirm.\n\nThanks,\n{biz.name}")))

    # 5. defer discretionary purchases
    for b in biz.open_bills():
        if not b.discretionary or b.due_date > low_day + timedelta(days=3):
            continue
        v = biz.vendor(b.vendor_id)
        # pay on the date that leaves the most headroom afterwards (bad-case curve)
        p10 = base.quantiles()[10]
        start_i = max(low_i + 14, (b.due_date - today).days + 21)
        options = range(start_i, base.horizon - 1)
        best_i = max(options, key=lambda i: (float(p10[i:].min()), -i)) if len(options) else base.horizon - 2
        new_date = today + timedelta(days=best_i)
        acts.append(Action(
            "reschedule_bill", {"bill_id": b.id, "new_date": new_date},
            title=f"Move the {v.name} purchase ({_money(b.amount)}) to {new_date:%b %d}",
            rationale=f"This is a planned purchase, not an obligation. Paying on {new_date:%b %d} keeps "
                      f"{_money(b.amount)} in the bank through the low point.",
            counterparty=v.email, relationship_cost=0.05,
            overlay=Overlay(bill_schedule={b.id: [(new_date, b.amount)]}),
            draft=(f"Subject: New delivery date for {b.reference}\n\nHi,\n\nPlease move our order {b.reference} "
                   f"to a {new_date:%b %d} delivery and invoice date at the quoted price.\n\nThanks,\n{biz.name}")))
    return acts


def _evaluate(fc: Forecaster, overlay: Overlay, targets: Targets) -> ForecastResult:
    res = fc.run(overlay)
    res.buffer = targets.buffer
    return res


def _feasible(res: ForecastResult, t: Targets) -> bool:
    return res.p_below(t.buffer) <= t.max_p_below_buffer and res.p_shortfall <= t.max_p_shortfall


def _combined(base: Overlay, acts: list[Action]) -> Overlay:
    ov = base
    for a in acts:
        ov = ov.merged(a.overlay)
    return ov


def _total_cost(acts: list[Action]) -> float:
    return sum(a.cost + GOODWILL_DOLLARS * a.relationship_cost for a in acts)


def financing_action(biz: Business, q: dict) -> Action:
    flows = [(date.fromisoformat(d), a) for d, a in q["flows"]]
    return Action(
        "draw_financing", {"product_id": q["product_id"], "amount": q["amount"], "apr": q["apr"], "flows": flows},
        title=f"Borrow {_money(q['amount'])} from {q['lender']} ({q['name']})",
        rationale=f"{q['detail']} Total cost {_money(q['total_cost'] or 0)}, true APR {100 * (q['apr'] or 0):.1f}%.",
        cost=q["total_cost"] or 0.0, counterparty=q["lender"],
        overlay=Overlay(extra_flows=flows),
        draft=f"Financing request\nLender: {q['lender']}\nProduct: {q['name']}\nAmount: ${q['amount']:,.2f}\n"
              f"Funds on: {q['funds_on']}\n{q['detail']}")


def _size_financing(biz: Business, fc: Forecaster, fin: dict, base: Overlay, chosen: list[Action],
                    targets: Targets) -> tuple[Optional[Action], list[dict]]:
    """Size and time a financing draw against the remaining deficit profile,
    then pick the cheapest eligible, in-time, non-high-cost product that makes
    the plan meet the risk targets."""
    cur = _evaluate(fc, _combined(base, chosen), targets)
    qlevel = max(1, int(round(50 * targets.max_p_below_buffer)))  # pointwise quantile stricter than the path target
    qcurve = np.percentile(cur.balances, qlevel, axis=0)
    deficit = targets.buffer - qcurve
    deficit[: cur.risk_from] = 0
    pays = fc.expected_pay_dates()
    daily_sales = float(fc.sales_det.mean()) if fc.has_sales else 0.0
    if (deficit <= 0).all():
        return None, []
    idx = np.nonzero(deficit > 0)[0]
    first, last = int(idx[0]), int(idx[-1])
    needed_by = biz.today + timedelta(days=max(1, first - 1))
    amount0 = float(np.ceil(deficit.max() / 1000) * 1000)
    temporary = last < fc.H - 14   # gap closes well before the horizon ends

    def repay_day(amount: float) -> date:
        """Earliest day after the gap when repaying keeps the bad-case curve above the buffer."""
        for r in range(last + 1, fc.H):
            if float(qcurve[r:].min()) - amount >= targets.buffer:
                return biz.today + timedelta(days=r)
        # no safe repayment inside the 13 weeks: repay from receipts after the horizon
        # (Dorsal re-forecasts weekly and proposes the repayment when it is safe)
        return biz.today + timedelta(days=fc.H + 30)
    repay_by = repay_day(amount0)

    best: Optional[tuple[float, Action, dict]] = None
    fallback: Optional[tuple[float, Action, dict]] = None
    for q0 in fin_mod.match(biz, fin, fin_mod.Need(amount0, needed_by, repay_by), pays, daily_sales):
        if not (q0["eligible"] and q0["arrives_in_time"]) or q0["high_cost"]:
            continue
        prod = fin_mod.product(q0["product_id"])
        if temporary and prod.kind == "term_loan":
            continue   # a temporary gap is matched with short-term money
        cap = min(targets.max_borrow,
                  biz.credit_line.available if prod.id == "existing-line" and biz.credit_line else prod.max_amount)
        amounts = sorted({min(cap, max(prod.min_amount, amount0 + step)) for step in (-2000, 0, 2000, 4000, 7000, 10000)})
        for amt in amounts:
            qq = fin_mod.quote(prod, biz, fin, fin_mod.Need(amt, needed_by, repay_day(amt)), pays, daily_sales)
            if not qq["eligible"]:
                continue
            act = financing_action(biz, qq)
            res = _evaluate(fc, _combined(base, chosen + [act]), targets)
            if _feasible(res, targets):
                if best is None or act.cost < best[0]:
                    best = (act.cost, act, qq)
                break
            score = res.p_below(targets.buffer)
            if fallback is None or score < fallback[0]:
                fallback = (score, act, qq)
    pick = best or fallback
    final_amt = pick[2]["amount"] if pick else amount0
    quotes = fin_mod.match(biz, fin, fin_mod.Need(final_amt, needed_by, repay_by), pays, daily_sales)
    return (pick[1] if pick else None), quotes


def optimize(biz: Business, fc: Forecaster, fin: dict, targets: Targets,
             committed: Optional[Overlay] = None) -> dict:
    base = known_info_overlay(biz).merged(committed or Overlay())
    before = _evaluate(fc, base, targets)
    cands = candidates(biz, fc, before, targets)

    chosen: list[Action] = []
    # protective, free steps are always included
    for a in cands:
        if a.type in ("hold_bill",):
            chosen.append(a)
    pool = [a for a in cands if a not in chosen]
    cur = _evaluate(fc, _combined(base, chosen), targets)
    # greedy over remaining levers
    while not _feasible(cur, targets) and pool:
        scored = []
        for a in pool:
            res = _evaluate(fc, _combined(base, chosen + [a]), targets)
            gain = cur.expected_deficit - res.expected_deficit
            scored.append((gain / (a.cost + GOODWILL_DOLLARS * a.relationship_cost + 25.0), gain, a, res))
        scored.sort(key=lambda x: -x[0])
        ratio, gain, a, res = scored[0]
        if gain <= 1.0:
            break
        chosen.append(a)
        pool.remove(a)
        cur = res
    # reminders are near-free and improve collections even when they don't move the low point
    for a in list(pool):
        if a.type == "send_reminder":
            chosen.append(a)
            pool.remove(a)
    cur = _evaluate(fc, _combined(base, chosen), targets)

    fin_action, quotes = None, []
    if not _feasible(cur, targets):
        fin_action, quotes = _size_financing(biz, fc, fin, base, chosen, targets)
        if fin_action:
            chosen.append(fin_action)
            cur = _evaluate(fc, _combined(base, chosen), targets)
    else:
        low_i, first_i = _low_info(cur, targets.buffer)
        quotes = fin_mod.match(biz, fin, fin_mod.Need(10000, biz.today + timedelta(days=max(1, first_i)),
                                                      biz.today + timedelta(days=min(fc.H - 1, low_i + 30))),
                               fc.expected_pay_dates(), float(fc.sales_det.mean()) if fc.has_sales else 0.0)

    # prune steps the plan no longer needs (most expensive first)
    if _feasible(cur, targets):
        for a in sorted([x for x in chosen if x.type not in ("hold_bill", "send_reminder")],
                        key=lambda x: -(x.cost + GOODWILL_DOLLARS * x.relationship_cost)):
            trial = [x for x in chosen if x is not a]
            res = _evaluate(fc, _combined(base, trial), targets)
            if _feasible(res, targets):
                chosen = trial
                cur = res

    # leave-one-out attribution
    steps = []
    for a in chosen:
        without = _evaluate(fc, _combined(base, [x for x in chosen if x is not a]), targets)
        d = a.to_dict()
        d["impact"] = {
            "p_shortfall_without": round(without.p_shortfall, 3),
            "p_below_buffer_without": round(without.p_below(targets.buffer), 3),
            "median_low_gain": round(float(cur.quantiles()[50].min() - without.quantiles()[50].min()), 0),
            "expected_deficit_gain": round(without.expected_deficit - cur.expected_deficit, 0)}
        steps.append(d)
    return {
        "targets": targets.__dict__,
        "before": before.summary(),
        "after": cur.summary() | {"p_below_buffer": round(cur.p_below(targets.buffer), 3)},
        "feasible": _feasible(cur, targets),
        "steps": steps,
        "actions": chosen,
        "considered": [a.to_dict() for a in cands],
        "financing_quotes": quotes,
        "total_expected_cost": round(sum(a.cost for a in chosen), 2),
        "result": cur,
        "baseline": before,
    }
