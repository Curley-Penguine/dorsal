"""Deterministic policy engine: the permission tiers.

Every action proposal, whoever made it (planner, LLM, owner, or text that
arrived in an email), passes through `PolicyEngine.evaluate()` and gets one
of three outcomes:

* ``auto``     - Dorsal may execute now, without asking.
* ``approve``  - the owner must approve this exact action (and re-enter a
                 PIN for money movement and borrowing).
* ``deny``     - never executed, whatever the approval.

The engine is plain Python reading `config/policy.yaml`. The language model
cannot call it with different rules, cannot edit the file, and never sees an
execution tool: it can only *propose*.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from dorsal.actions import Action
from dorsal.models import Business
from dorsal.security import guard

DEFAULT_POLICY = Path(__file__).resolve().parents[2] / "config" / "policy.yaml"


@dataclass
class Decision:
    action_id: str
    tier: str
    outcome: str                  # auto | approve | deny
    step_up: bool = False
    reasons: list = field(default_factory=list)
    rules: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class PolicyEngine:
    def __init__(self, path: Path | str | None = None, overrides: Optional[dict] = None):
        with open(path or DEFAULT_POLICY) as f:
            self.cfg = yaml.safe_load(f)
        if overrides:
            self.cfg["owner_settings"].update(overrides)
        self.counters = {"notify_today": 0, "moved_today": 0.0}

    # ---- owner settings (only callable from the owner UI / API) -----------
    @property
    def settings(self) -> dict:
        return copy.deepcopy(self.cfg["owner_settings"])

    def update_owner_settings(self, changes: dict) -> dict:
        allowed = {"buffer", "auto_cost_limit", "risk_target_p_below_buffer", "risk_target_p_shortfall", "paused"}
        bad = set(changes) - allowed
        if bad:
            raise ValueError(f"Not an owner setting: {sorted(bad)}")
        if "auto_cost_limit" in changes and not (0 <= float(changes["auto_cost_limit"]) <= 1000):
            raise ValueError("auto_cost_limit must be between $0 and $1,000")
        if "buffer" in changes and float(changes["buffer"]) < 0:
            raise ValueError("buffer must be positive")
        self.cfg["owner_settings"].update(changes)
        return self.settings

    # ---- evaluation ---------------------------------------------------------
    def evaluate(self, a: Action, biz: Business) -> Decision:
        tiers = self.cfg["tiers"]
        s = self.cfg["owner_settings"]
        tier = a.tier
        d = Decision(a.id, tier, "approve")

        def deny(rule: str, reason: str) -> Decision:
            d.outcome = "deny"
            d.rules.append(rule)
            d.reasons.append(reason)
            return d

        if s.get("paused"):
            return deny("P0-kill-switch", "Dorsal is paused by the owner.")
        if tier == "forbidden" or tiers.get(tier, {}).get("mode") == "deny":
            return deny("P1-forbidden-type", f"'{a.type}' is never allowed for the agent.")
        if a.proposed_by == "external" and tier in ("commit", "borrow", "move_money"):
            return deny("P2-untrusted-origin", "Requested by third-party content (email or memo), not by the owner.")

        # counterparty and data-leak checks for anything that sends a message
        known = biz.known_emails()
        if a.counterparty and "@" in a.counterparty and a.counterparty.lower() not in known:
            return deny("P3-unknown-recipient", f"{a.counterparty} is not an existing contact.")
        if a.draft:
            leak = self._leak_check(a, biz)
            if leak:
                return deny("P4-data-leak", leak)
            sc = guard.scan(a.draft)
            if sc["flagged"]:
                return deny("P5-unsafe-content", "Outgoing message contains instruction-like or bank-detail content.")

        # protected obligations
        bill_ids = [a.params.get("bill_id")] if a.params.get("bill_id") else []
        for bid in bill_ids:
            try:
                b = biz.bill(bid)
            except StopIteration:
                return deny("P6-unknown-object", f"No bill {bid}.")
            v = biz.vendor(b.vendor_id)
            cat = "tax" if v.category == "tax" else v.category
            if v.protected or cat in self.cfg["protected_categories"]:
                if a.type != "pay_bill":
                    return deny("P7-protected", f"{v.name} is a protected obligation; Dorsal never delays or alters it.")
        if a.params.get("recurring_id"):
            r = next((r for r in biz.recurring if r.id == a.params["recurring_id"]), None)
            if r is None or r.protected or r.category in self.cfg["protected_categories"]:
                return deny("P7-protected", "Payroll, tax, rent and loan payments are protected.")

        # tier-specific rules
        if tier in ("read",):
            d.outcome = "auto"
        elif tier == "notify":
            if self.counters["notify_today"] >= tiers["notify"].get("max_per_day", 20):
                d.outcome = "approve"
                d.reasons.append("Daily limit for automatic messages reached.")
            else:
                d.outcome = "auto"
        elif tier == "protect":
            if a.type == "hold_bill" and not a.params.get("duplicate_of"):
                d.outcome = "approve"
                d.reasons.append("Holds are automatic only for suspected duplicates.")
            else:
                d.outcome = "auto"
        elif tier == "commit":
            if a.type in s.get("auto_commit_types", []) and a.cost <= float(s.get("auto_cost_limit", 0)):
                d.outcome = "auto"
                d.reasons.append(f"Expected cost ${a.cost:,.0f} is within the owner's ${s['auto_cost_limit']} limit.")
            else:
                d.outcome = "approve"
                d.reasons.append("Changes a commitment to a customer or supplier.")
        elif tier == "borrow":
            t = tiers["borrow"]
            amt = float(a.params.get("amount", 0))
            if amt > t["max_amount"]:
                return deny("P8-borrow-cap", f"Borrowing ${amt:,.0f} exceeds the ${t['max_amount']:,.0f} cap.")
            apr = a.params.get("apr")
            if apr is not None and float(apr) > t["max_apr"]:
                return deny("P9-high-cost", f"APR {100 * float(apr):.0f}% is above the {100 * t['max_apr']:.0f}% limit.")
            d.outcome, d.step_up = "approve", bool(t.get("step_up"))
            d.reasons.append("Borrowing always needs the owner's approval and PIN.")
        elif tier == "move_money":
            t = tiers["move_money"]
            try:
                b = biz.bill(a.params["bill_id"])
                v = biz.vendor(b.vendor_id)
            except (KeyError, StopIteration):
                return deny("P6-unknown-object", "Payments must settle an existing bill.")
            if not v.verified or a.params.get("payee_id") != v.id:
                return deny("P10-unverified-payee", "Payee must be the bill's verified vendor.")
            amt = float(a.params.get("amount", 0))
            if amt > b.amount + 0.01:
                return deny("P11-overpayment", "Payment exceeds the bill amount.")
            if self.counters["moved_today"] + amt > t["daily_limit"]:
                return deny("P12-daily-limit", f"Would exceed the ${t['daily_limit']:,.0f} daily payment limit.")
            d.outcome, d.step_up = "approve", bool(t.get("step_up"))
        else:
            return deny("P1-forbidden-type", f"Unknown tier for '{a.type}'.")
        return d

    def _leak_check(self, a: Action, biz: Business) -> Optional[str]:
        """An outgoing message may only mention the recipient's own invoices."""
        mentioned = set(re.findall(r"INV-\d+", a.draft))
        if not mentioned or not a.counterparty:
            return None
        cust = next((c for c in biz.customers if c.email.lower() == a.counterparty.lower()), None)
        if cust is None:
            return None
        own = {i.id for i in biz.invoices if i.customer_id == cust.id}
        foreign = mentioned - own
        if foreign:
            return f"Message to {cust.name} mentions another customer's invoices ({', '.join(sorted(foreign))})."
        return None

    def record_execution(self, a: Action) -> None:
        if a.tier == "notify":
            self.counters["notify_today"] += 1
        if a.tier == "move_money":
            self.counters["moved_today"] += float(a.params.get("amount", 0))
