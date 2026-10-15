"""Evaluation harness: does Dorsal actually prevent cash shortfalls, and at what cost?

For N synthetic businesses (five archetypes, random size, cash cushion,
customer behaviour, and a 50% chance of a cash shock such as a large
inventory or equipment bill, plus planted duplicate bills) we:

1. give each strategy only what the owner could see today (history + open
   items), and let it act;
2. play the *true* next 13 weeks from the hidden world model, including
   whether customers really accept discounts and respond to reminders;
3. score the realised bank balance.

Strategies
----------
none        no proactive management; shortfalls are covered by overdraft
            (22% APR plus a $35 fee per day overdrawn).
naive_rule  what simple cash tools do: a single-line forecast assuming
            customers pay on the due date; if it dips below the buffer, draw
            the credit line for the gap.
dorsal        probabilistic forecast + chance-constrained plan, every step
            approved by the owner.

Outputs a JSON and Markdown report in docs/evidence/evaluation/.
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from dorsal.analytics.alerts import find_duplicate_bills
from dorsal.analytics.financials import analyze
from dorsal.analytics.forecast import Forecaster
from dorsal.models import Bill
from dorsal.planning.planner import Targets, optimize
from dorsal.sandbox.generator import ARCHETYPES, HORIZON, Event, generate

OD_APR, OD_FEE = 0.22, 35.0


def _plant_duplicate(biz, truth, rng) -> None:
    cands = [b for b in biz.open_bills() if not biz.vendor(b.vendor_id).protected and b.due_date > biz.today]
    if not cands:
        return
    src = cands[int(rng.integers(0, len(cands)))]
    dup = Bill(src.id + "D", src.vendor_id, src.amount, src.issue_date + timedelta(days=3),
               src.due_date + timedelta(days=3), reference=src.reference, memo="Re-sent invoice")
    biz.bills.append(dup)
    truth.future.append(Event(dup.due_date, -dup.amount, "bill", dup.id))
    truth.future.sort(key=lambda e: e.day)
    truth.duplicates = getattr(truth, "duplicates", set()) | {dup.id}


def realise(biz, truth, actions, seed: int) -> dict:
    """Play the true future with a set of executed actions."""
    rng = np.random.default_rng(seed)
    today = biz.today
    reminders, offers, sched, holds, extra = set(), {}, {}, set(), []
    cost = 0.0
    for a in actions:
        p = a.params
        if a.type == "send_reminder":
            reminders |= set(p["invoice_ids"])
        elif a.type == "offer_discount":
            offers[p["invoice_id"]] = p
        elif a.type in ("request_split",):
            sched[p["bill_id"]] = p["payments"]
        elif a.type == "reschedule_bill":
            sched[p["bill_id"]] = [(p["new_date"], biz.bill(p["bill_id"]).amount)]
        elif a.type == "hold_bill":
            holds.add(p["bill_id"])
        elif a.type == "draw_financing":
            extra += [(d, amt) for d, amt in p["flows"]]
            cost += a.cost
    flows = np.zeros(HORIZON)

    def add(d: date, amt: float):
        i = (d - today).days
        if 0 <= i < HORIZON:
            flows[i] += amt
    inv_cust = {i.id: i.customer_id for i in biz.invoices}
    dups = getattr(truth, "duplicates", set())
    for e in truth.future:
        day, amt = e.day, e.amount
        if e.kind == "collection" and e.ref in inv_cust:
            inv = biz.invoice(e.ref)
            if e.ref in reminders and inv.due_date < today:
                day = today + timedelta(days=math.ceil(max(1, (day - today).days) * truth.reminder_factor))
            if e.ref in offers:
                o = offers[e.ref]
                if rng.random() < truth.discount_accept_p.get(inv_cust[e.ref], 0.5):
                    day = min(day, o["deadline"] - timedelta(days=int(rng.integers(0, 3))))
                    disc = amt * o["pct"]
                    amt -= disc
                    cost += disc
        if e.kind == "bill":
            if e.ref in holds:
                if e.ref in dups:
                    continue                      # duplicate correctly never paid
                day = day + timedelta(days=7)     # false positive: paid a week later after vendor confirms
            if e.ref in sched:
                for d, a2 in sched[e.ref]:
                    add(d, -a2)
                continue
        add(day, amt)
    for d, a in extra:
        add(d, a)
    bal = biz.cash + np.cumsum(flows)
    neg = np.minimum(bal, 0)
    od_cost = float(-neg.sum() * OD_APR / 365 + OD_FEE * (bal < 0).sum())
    return {"min": float(bal[3:].min()), "shortfall": bool((bal[3:] < 0).any()), "days_negative": int((bal < 0).sum()),
            "days_below_buffer": None, "balance": bal, "action_cost": cost, "overdraft_cost": od_cost,
            "total_cost": cost + od_cost}


def naive_actions(biz, buffer: float):
    """Single-line forecast assuming on-time payment; draw the line for the gap."""
    from dorsal.actions import Action
    today = biz.today
    flows = np.zeros(HORIZON)
    recent = sum(t.amount for t in biz.transactions if t.category == "sales" and t.date >= today - timedelta(days=28)) / 28
    flows += recent
    for inv in biz.open_invoices():
        i = max(1, (inv.due_date - today).days)
        if i < HORIZON:
            flows[i] += inv.amount
    for b in biz.open_bills():
        i = max(0, (b.due_date - today).days)
        if i < HORIZON:
            flows[i] -= b.amount
    for r in biz.recurring:
        for d in r.occurrences(today, today + timedelta(days=HORIZON - 1)):
            flows[(d - today).days] -= r.amount
    bal = biz.cash + np.cumsum(flows)
    low = int(np.argmin(bal[3:])) + 3
    if bal[low] >= buffer or biz.credit_line is None:
        return []
    amt = min(biz.credit_line.available, math.ceil((buffer - bal[low]) / 1000) * 1000)
    draw = today + timedelta(days=max(1, low - 2))
    repay = today + timedelta(days=HORIZON + 14)
    interest = amt * biz.credit_line.apr * (repay - draw).days / 365
    return [Action("draw_financing", {"product_id": "existing-line", "amount": amt,
                                      "flows": [(draw, amt), (repay, -(amt + interest))]},
                   title="Naive credit-line draw", cost=interest)]


def evaluate(n: int = 200, out: str = "docs/evidence/evaluation", n_paths: int = 400, seed0: int = 1000) -> dict:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    arch = list(ARCHETYPES)
    rows = []
    t0 = time.time()
    for k in range(n):
        seed = seed0 + k
        rng = np.random.default_rng(seed)
        a = arch[k % len(arch)]
        biz, truth = generate(a, seed=seed, shock=bool(rng.random() < 0.5), cash_weeks=float(rng.uniform(1.0, 4.0)))
        if rng.random() < 0.25:
            _plant_duplicate(biz, truth, rng)
        buffer = round(sum(r.amount * {"weekly": 1, "biweekly": .5, "monthly": 12 / 52}[r.kind] for r in biz.recurring), -2)
        targets = Targets(buffer=buffer, max_borrow=max(25000.0, buffer * 3))
        fin = analyze(biz)
        fc = Forecaster(biz, n_paths=n_paths, seed=seed, buffer=buffer)
        t_plan = time.time()
        plan = optimize(biz, fc, fin, targets)
        plan_secs = time.time() - t_plan
        strategies = {"none": [], "naive_rule": naive_actions(biz, buffer), "dorsal": plan["actions"]}
        row = {"id": biz.id, "archetype": a, "buffer": buffer, "cash": biz.cash,
               "dorsal_pred_p_short": plan["before"]["p_shortfall"], "plan_secs": round(plan_secs, 2),
               "dorsal_steps": len(plan["actions"]),
               "dorsal_approvals": sum(1 for x in plan["actions"] if x.tier in ("commit", "borrow", "move_money")),
               "duplicate_planted": bool(getattr(truth, "duplicates", set())),
               "duplicate_caught": bool(getattr(truth, "duplicates", set()) & {d for _, d in find_duplicate_bills(biz)})}
        for name, acts in strategies.items():
            r = realise(biz, truth, acts, seed)
            below_buf = int((r["balance"][3:] < buffer).sum())
            row[name] = {"shortfall": r["shortfall"], "min": round(r["min"], 0), "days_negative": r["days_negative"],
                         "days_below_buffer": below_buf, "action_cost": round(r["action_cost"], 2),
                         "overdraft_cost": round(r["overdraft_cost"], 2), "total_cost": round(r["total_cost"], 2)}
        rows.append(row)
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{n} businesses  ({time.time() - t0:.0f}s)")

    def agg(name):
        xs = [r[name] for r in rows]
        return {"shortfall_rate": round(float(np.mean([x["shortfall"] for x in xs])), 3),
                "mean_days_negative": round(float(np.mean([x["days_negative"] for x in xs])), 2),
                "mean_days_below_buffer": round(float(np.mean([x["days_below_buffer"] for x in xs])), 2),
                "mean_total_cost": round(float(np.mean([x["total_cost"] for x in xs])), 2),
                "mean_overdraft_cost": round(float(np.mean([x["overdraft_cost"] for x in xs])), 2),
                "mean_action_cost": round(float(np.mean([x["action_cost"] for x in xs])), 2)}
    summary = {name: agg(name) for name in ("none", "naive_rule", "dorsal")}
    at_risk = [r for r in rows if r["none"]["shortfall"]]
    summary["n_businesses"] = n
    summary["n_would_run_short_without_help"] = len(at_risk)
    summary["dorsal_prevented_share"] = round(float(np.mean([not r["dorsal"]["shortfall"] for r in at_risk])), 3) if at_risk else None
    summary["naive_prevented_share"] = round(float(np.mean([not r["naive_rule"]["shortfall"] for r in at_risk])), 3) if at_risk else None
    safe = [r for r in rows if r["none"]["min"] >= r["buffer"]]
    summary["dorsal_cost_on_safe_businesses"] = round(float(np.mean([r["dorsal"]["action_cost"] for r in safe])), 2) if safe else None
    summary["dorsal_false_alarm_rate"] = round(float(np.mean([r["dorsal"]["action_cost"] > 1 for r in safe])), 3) if safe else None
    dup = [r for r in rows if r["duplicate_planted"]]
    summary["duplicates_planted"] = len(dup)
    summary["duplicates_caught"] = sum(r["duplicate_caught"] for r in dup)
    summary["median_plan_seconds"] = float(np.median([r["plan_secs"] for r in rows]))
    summary["mean_approvals_per_business"] = round(float(np.mean([r["dorsal_approvals"] for r in rows])), 2)
    by_arch = defaultdict(dict)
    for a in arch:
        sub = [r for r in rows if r["archetype"] == a]
        for name in ("none", "naive_rule", "dorsal"):
            by_arch[a][name] = round(float(np.mean([r[name]["shortfall"] for r in sub])), 3)
    summary["shortfall_rate_by_archetype"] = by_arch
    (out_dir / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=str))
    md = _markdown(summary)
    (out_dir / "REPORT.md").write_text(md)
    print(md)
    return summary


def _markdown(s: dict) -> str:
    L = ["# Dorsal evaluation on synthetic businesses", "",
         f"Businesses: **{s['n_businesses']}** (5 archetypes; 50% with a cash shock; 25% with a planted duplicate bill).",
         f"Without help, **{s['n_would_run_short_without_help']}** of them run out of cash within 13 weeks.", "",
         "| Strategy | Ran out of cash | Days overdrawn (mean) | Days below buffer (mean) | Total cost (mean) |",
         "|---|---|---|---|---|"]
    names = {"none": "No agent", "naive_rule": "Naive rule (on-time assumption + credit line)", "dorsal": "Dorsal"}
    for k in ("none", "naive_rule", "dorsal"):
        x = s[k]
        L.append(f"| {names[k]} | {100 * x['shortfall_rate']:.1f}% | {x['mean_days_negative']:.2f} | "
                 f"{x['mean_days_below_buffer']:.1f} | ${x['mean_total_cost']:,.0f} |")
    L += ["", f"- Of the businesses that would have run short, Dorsal prevented **{100 * (s['dorsal_prevented_share'] or 0):.0f}%** "
              f"(naive rule: {100 * (s['naive_prevented_share'] or 0):.0f}%).",
          f"- On businesses that never needed help, Dorsal spent ${s['dorsal_cost_on_safe_businesses']:.2f} on average "
          f"(cost-incurring steps on {100 * (s['dorsal_false_alarm_rate'] or 0):.0f}% of them).",
          f"- Duplicate bills caught: {s['duplicates_caught']} of {s['duplicates_planted']}.",
          f"- Owner approvals needed per business: {s['mean_approvals_per_business']} on average.",
          f"- Median planning time: {s['median_plan_seconds']:.1f} s per business.", "",
          "Total cost = discounts given + financing interest and fees + overdraft interest and fees.",
          "All businesses are synthetic; behavioural parameters (discount uptake, reminder effect) are drawn from ranges, "
          "not taken from real customer data."]
    return "\n".join(L) + "\n"
