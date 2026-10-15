"""Serialises the runtime into the JSON the web console renders."""
from __future__ import annotations

from datetime import timedelta

import numpy as np

from dorsal.analytics.alerts import inbox_security
from dorsal.security import guard


def _r(a) -> list:
    return [int(round(float(x))) for x in a]


def _curves(res) -> dict:
    q = res.quantiles()
    return {"p10": _r(q[10]), "p50": _r(q[50]), "p90": _r(q[90]),
            "p_below_zero": [round(float(x), 3) for x in (res.balances < 0).mean(axis=0)]}


def step_status(rt, action_id: str) -> str:
    if action_id in rt.pending:
        return "pending"
    for rec in reversed(rt.records):
        if rec["action"].id == action_id:
            st = rec["status"]
            if st == "executed":
                return "auto" if rec.get("executed_by") == "dorsal" else "approved"
            return st
    return "not_proposed"


def snapshot(rt, briefing: str | None = None, provider: str = "offline") -> dict:
    biz = rt.biz
    plan = rt.plan()
    base = rt.baseline()
    cur = rt.forecast()
    fin = rt.analysis()
    fc = rt.forecaster()
    pays = fc.expected_pay_dates()
    steps = []
    for s in plan["steps"]:
        st = step_status(rt, s["id"])
        pend = rt.pending.get(s["id"])
        step_up = bool(pend and pend["decision"].step_up) or s["tier"] in ("borrow", "move_money")
        steps.append({**s, "status": st, "step_up": step_up})
    # executed steps that are no longer in the (re-optimised) plan stay visible
    plan_ids = {s["id"] for s in steps}
    for rec in rt.records:
        a = rec["action"]
        if a.id not in plan_ids and rec["status"] in ("executed",) and a.proposed_by != "external":
            d = a.to_dict()
            d["status"] = "auto" if rec.get("executed_by") == "dorsal" else "approved"
            d["impact"] = None
            d["step_up"] = False
            steps.append(d)
            plan_ids.add(a.id)
    blocked = [{"title": r["action"].title or r["action"].type, "type": r["action"].type, "reasons": r["decision"].reasons,
                "rules": r["decision"].rules, "proposed_by": r["action"].proposed_by}
               for r in rt.records if r["status"] == "denied"]
    known = biz.known_emails()
    flagged = {m["message_id"] for m in inbox_security(biz)}
    return {
        "provider": provider,
        "business": {"name": biz.name, "industry": biz.industry, "today": biz.today.isoformat(), "cash": round(biz.cash, 2),
                     "months_in_business": biz.months_in_business},
        "settings": rt.policy.settings,
        "policy": {"tiers": rt.policy.cfg["tiers"], "protected": rt.policy.cfg["protected_categories"]},
        "dates": [(biz.today + timedelta(days=i)).isoformat() for i in range(cur.horizon)],
        "baseline": {"summary": base.summary(), "curves": _curves(base)},
        "current": {"summary": cur.summary(), "curves": _curves(cur), "weekly": cur.weekly(), "events": cur.events[:40]},
        "plan": {"steps": steps, "before": plan["before"], "after": plan["after"], "feasible": plan["feasible"],
                 "total_expected_cost": plan["total_expected_cost"], "targets": plan["targets"]},
        "financing": plan["financing_quotes"],
        "alerts": rt.alerts(),
        "blocked": blocked,
        "analysis": {k: fin[k] for k in ("trailing_12m", "last_quarter", "revenue_growth_qoq", "gross_margin_change_qoq",
                                          "working_capital", "dscr", "concentration", "monthly", "expense_mix_90d",
                                          "health", "benchmarks", "customers")},
        "delay_model": fc.delay_stats,
        "invoices": [{"id": i.id, "customer": biz.customer(i.customer_id).name, "amount": i.amount,
                      "issued": i.issue_date.isoformat(), "due": i.due_date.isoformat(),
                      "expected": pays[i.id].isoformat() if i.id in pays else None,
                      "days_overdue": max(0, (biz.today - i.due_date).days), "reminded": i.id in rt.world.reminders,
                      "offer": i.id in rt.world.offers} for i in biz.open_invoices()],
        "bills": [{"id": b.id, "vendor": biz.vendor(b.vendor_id).name, "amount": b.amount, "due": b.due_date.isoformat(),
                   "reference": b.reference, "discretionary": b.discretionary, "protected": biz.vendor(b.vendor_id).protected,
                   "held": b.id in rt.world.holds,
                   "schedule": [(d.isoformat(), a) for d, a in rt.world.schedule_overrides.get(b.id, [])]}
                  for b in biz.open_bills()],
        "recurring": [{"name": r.name, "amount": r.amount, "kind": r.kind, "protected": r.protected} for r in biz.recurring],
        "inbox": [{"id": m.id, "from": m.from_email, "subject": m.subject, "body": guard.redact(m.body),
                   "received": m.received.isoformat(), "flagged": m.id in flagged,
                   "sender": guard.sender_check(m.from_email, known), "risk": guard.scan(m.subject + " " + m.body)}
                  for m in biz.inbox],
        "outbox": rt.world.outbox[-30:],
        "audit": {"entries": rt.audit.entries[-80:], "verify": rt.audit.verify(), "count": len(rt.audit.entries)},
        "briefing": briefing,
    }
