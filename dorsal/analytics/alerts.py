"""Alerts: things the owner should know about today, ranked by severity.

Each alert has a stable code, a severity, a plain-language message, the
evidence behind it, and (where relevant) the lever Dorsal would use.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from statistics import median

from dorsal.analytics.forecast import ForecastResult
from dorsal.models import Business
from dorsal.security import guard

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def _money(x: float) -> str:
    return ("-$" if x < 0 else "$") + f"{abs(x):,.0f}"


def find_duplicate_bills(biz: Business) -> list[tuple[str, str]]:
    """Pairs (original, suspected duplicate) of open bills: same vendor and
    same vendor reference, or same vendor and amount within 10 days."""
    pairs = []
    open_bills = sorted(biz.open_bills(), key=lambda b: b.issue_date)
    for i, a in enumerate(open_bills):
        for b in open_bills[i + 1:]:
            if a.vendor_id != b.vendor_id:
                continue
            same_ref = a.reference and a.reference == b.reference
            same_amt = abs(a.amount - b.amount) < 0.01 and abs((b.issue_date - a.issue_date).days) <= 10
            if same_ref or same_amt:
                pairs.append((a.id, b.id))
    return pairs


def payment_promises(biz: Business) -> list[dict]:
    """Payment promises found in emails from known customer addresses."""
    out = []
    for m in biz.inbox:
        sender = m.from_email.lower()
        cust = next((c for c in biz.customers if c.email.lower() == sender), None)
        if not cust:
            continue
        found = guard.extract_promise(m.body, biz.today.year)
        if not found:
            continue
        d = date(*found)
        opens = [i for i in biz.open_invoices() if i.customer_id == cust.id and i.due_date <= d]
        for inv in opens:
            out.append({"invoice_id": inv.id, "customer": cust.name, "promised_date": d.isoformat(), "message_id": m.id})
    return out


def inbox_security(biz: Business) -> list[dict]:
    known = biz.known_emails()
    out = []
    for m in biz.inbox:
        s = guard.scan(m.subject + "\n" + m.body)
        sc = guard.sender_check(m.from_email, known)
        if s["flagged"] or (not sc["known"] and sc.get("lookalike_of")):
            out.append({"message_id": m.id, "from": m.from_email, "subject": m.subject, "risk": s,
                        "sender": sc})
    return out


def build_alerts(biz: Business, fc: ForecastResult, fin: dict | None = None) -> list[dict]:
    alerts: list[dict] = []
    today = biz.today
    s = fc.summary()

    def add(code, severity, title, detail, evidence=None, lever=None):
        alerts.append({"code": code, "severity": severity, "title": title, "detail": detail,
                       "evidence": evidence or {}, "lever": lever})

    # cash -----------------------------------------------------------------------
    if s["p_shortfall"] >= 0.2:
        add("CASH_SHORTFALL", "critical",
            f"{round(100 * s['p_shortfall'])}% chance of running out of cash in the next 13 weeks",
            f"In the typical case cash bottoms out at {_money(s['median_low'])} on {s['median_low_date']}. "
            f"In a bad case (1 in 10) it reaches {_money(s['p10_low'])}.",
            {k: s[k] for k in ("p_shortfall", "median_low", "median_low_date", "p10_low")}, "plan")
    elif s["p_below_buffer"] >= 0.3:
        add("BUFFER_BREACH", "warning",
            f"{round(100 * s['p_below_buffer'])}% chance of dipping below your {_money(fc.buffer)} buffer",
            f"Typical low point {_money(s['median_low'])} on {s['median_low_date']}.",
            {k: s[k] for k in ("p_below_buffer", "median_low", "median_low_date")}, "plan")

    # receivables -------------------------------------------------------------------
    overdue = [i for i in biz.open_invoices() if i.due_date < today]
    aged = [i for i in overdue if (today - i.due_date).days > 60]
    recent_overdue = [i for i in overdue if i not in aged and (today - i.due_date).days >= 3]
    if recent_overdue:
        tot = sum(i.amount for i in recent_overdue)
        names = sorted({biz.customer(i.customer_id).name for i in recent_overdue})
        add("AR_OVERDUE", "warning", f"{len(recent_overdue)} invoices overdue ({_money(tot)})",
            "Overdue at: " + ", ".join(names) + ".",
            {"invoice_ids": [i.id for i in recent_overdue], "total": round(tot, 2)}, "remind")
    for i in aged:
        add("AR_AGED", "warning", f"{i.id} from {biz.customer(i.customer_id).name} is {(today - i.due_date).days} days overdue",
            f"{_money(i.amount)} may not be collected, so the forecast gives it a high chance of non-payment. "
            "Consider a phone call or a payment plan.",
            {"invoice_id": i.id, "amount": i.amount}, None)

    # payables ------------------------------------------------------------------------
    for a_id, b_id in find_duplicate_bills(biz):
        a, b = biz.bill(a_id), biz.bill(b_id)
        add("AP_DUPLICATE", "critical", f"Possible duplicate bill from {biz.vendor(b.vendor_id).name}",
            f"{b.id} ({_money(b.amount)}, ref {b.reference}) matches {a.id} issued {a.issue_date}. "
            "Paying both would cost you an extra " + _money(b.amount) + ".",
            {"original": a.id, "duplicate": b.id, "amount": b.amount}, "hold_duplicate")
    by_vendor = defaultdict(list)
    for bl in biz.bills:
        by_vendor[bl.vendor_id].append(bl.amount)
    for bl in biz.open_bills():
        hist = [x for x in by_vendor[bl.vendor_id] if x != bl.amount]
        if len(hist) >= 4 and bl.amount > 2.0 * median(hist):
            add("AP_UNUSUAL", "info", f"{biz.vendor(bl.vendor_id).name} bill is {bl.amount / median(hist):.1f}x the usual size",
                f"{bl.id} for {_money(bl.amount)} due {bl.due_date}. Usual bill is about {_money(median(hist))}.",
                {"bill_id": bl.id, "usual": round(median(hist), 2)}, None)

    # inbox security ----------------------------------------------------------------------
    for m in inbox_security(biz):
        look = m["sender"].get("lookalike_of")
        add("SECURITY_SUSPICIOUS_MESSAGE", "critical", f"Suspicious email: \"{m['subject']}\"",
            ("Sender domain imitates " + look + ". " if look else "") +
            "It asks for a payment to new bank details and contains instructions aimed at an AI assistant. "
            "Dorsal will not act on it. Verify by phone using a number you already have.",
            {"message_id": m["message_id"], "from": m["from"], "signals": m["risk"]["signals"],
             "score": m["risk"]["score"]}, None)
    for p in payment_promises(biz):
        add("AR_PROMISE", "info", f"{p['customer']} promised to pay {p['invoice_id']} on {p['promised_date']}",
            "Dorsal has added this promise to the forecast and will check whether it is kept.", p, None)

    # analysis-driven ------------------------------------------------------------------------
    if fin:
        top = fin["concentration"]["top_customer_share_of_revenue"]
        if top > 0.25:
            add("CONCENTRATION", "warning", f"One customer is {100 * top:.0f}% of revenue",
                "Losing them would hit cash hard. Consider widening the customer base.", {"share": top}, None)
        gmc = fin.get("gross_margin_change_qoq")
        if gmc is not None and gmc < -0.03:
            add("MARGIN_DECLINE", "warning", f"Gross margin fell {abs(100 * gmc):.1f} points last quarter",
                "Supplier prices or discounts may be rising faster than your prices.", {"change": gmc}, None)
        cd = fin["working_capital"]["cash_days_on_hand"]
        if cd < 10:
            add("LOW_CASH_DAYS", "info", f"Cash on hand covers {cd:.0f} days of spending",
                "A buffer of 2 to 4 weeks gives room for late payers.", {"days": cd}, None)

    alerts.sort(key=lambda a: SEVERITY_ORDER[a["severity"]])
    return alerts
