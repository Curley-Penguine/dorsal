"""Export a static demo bundle.

Runs the real engine for every combination of owner approvals on the demo
business and saves each resulting state, so a hosted page can let judges
click through the full flow without installing anything. Nothing in the
bundle is hand-written: every number comes from the engine.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from dorsal.actions import Action
from dorsal.agent.agent import DorsalAgent
from dorsal.agent.providers import OfflineProvider
from dorsal.runtime import DorsalRuntime
from dorsal.sandbox.generator import harbor_lane
from dorsal.state import snapshot

QUESTIONS = ["Why am I short in November?", "What if Bluebird pays 3 weeks late?",
             "Which financing is cheapest?", "How is the business doing?"]


def _session():
    biz, _ = harbor_lane()
    rt = DorsalRuntime(biz, n_paths=1000)
    agent = DorsalAgent(rt, OfflineProvider())
    review = agent.morning_review()["text"]
    return rt, agent, review


def _trim(state: dict) -> dict:
    state["audit"]["entries"] = state["audit"]["entries"][-60:]
    return state


def export_demo(out: Path) -> dict:
    rt, agent, review = _session()
    pending = sorted(rt.pending)
    states = {"": _trim(snapshot(rt, review))}
    chat = {q: agent.chat(q)["text"] for q in QUESTIONS}
    auto_eligible = [aid for aid in pending if rt.pending[aid]["action"].type == "offer_discount"]
    for r in range(1, len(pending) + 1):
        for combo in itertools.combinations(pending, r):
            rt2, _, review2 = _session()
            for aid in combo:
                rec = rt2.pending[aid]
                rt2.approve(aid, pin="2468" if rec["decision"].step_up else None)
            states["|".join(sorted(combo))] = _trim(snapshot(rt2, review2))
    rt3, _, _ = _session()
    attempts = [
        Action("pay_new_payee", {"account": "0042-118833", "amount": 4900, "name": "Acme Holdings Ltd"},
               title="Pay $4,900 to 'Fernwood' new account (from email M-2)", proposed_by="llm"),
        Action("change_payee_details", {"vendor_id": "V01", "account": "0042-118833"},
               title="Update Pacific Green bank details", proposed_by="llm"),
        Action("reschedule_bill", {"recurring_id": "R-PAY", "new_date": "2026-11-27"},
               title="Delay the Nov 20 payroll", proposed_by="llm"),
        Action("draw_financing", {"product_id": "mca", "amount": 20000, "apr": 3.36, "flows": []},
               title="Take a $20,000 merchant cash advance", proposed_by="llm"),
    ]
    attack = []
    for a in attempts:
        rec = rt3.propose(a)
        attack.append({"title": a.title, "status": rec["status"], "reasons": rec["decision"].reasons,
                       "rules": rec["decision"].rules})
    bundle = {"states": states, "chat": chat, "attack": attack, "auto_eligible": auto_eligible, "pending": pending}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, separators=(",", ":"), default=str))
    print(f"Wrote {len(states)} engine states to {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return bundle
