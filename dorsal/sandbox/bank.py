"""Sandbox bank, accounting and email systems.

These stand in for the real APIs a production deployment would call
(bank payments API, accounting ledger, email). Every operation is written
to an operation log, which forms the "execution evidence" for the
competition. The sandbox enforces its own basic controls (verified payees,
idempotency) as a second line of defence behind Dorsal's policy engine.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from dorsal.models import Business


class SandboxError(Exception):
    pass


@dataclass
class SandboxWorld:
    business: Business
    oplog_path: Optional[Path] = None
    outbox: list[dict] = field(default_factory=list)
    schedule_overrides: dict[str, list[tuple[date, float]]] = field(default_factory=dict)  # bill_id -> payments
    holds: set[str] = field(default_factory=set)
    offers: dict[str, dict] = field(default_factory=dict)          # invoice_id -> discount offer
    financing: list[dict] = field(default_factory=list)            # drawn facilities with cash-flow schedules
    reminders: set[str] = field(default_factory=set)
    _idempotency: dict[str, dict] = field(default_factory=dict)
    oplog: list[dict] = field(default_factory=list)

    # ---- logging ---------------------------------------------------------
    def _log(self, op: str, **data) -> dict:
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "system": "sandbox", "op": op, **data}
        self.oplog.append(rec)
        if self.oplog_path:
            self.oplog_path.parent.mkdir(parents=True, exist_ok=True)
            with self.oplog_path.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def _once(self, key: str, fn):
        if key in self._idempotency:
            return {**self._idempotency[key], "idempotent_replay": True}
        res = fn()
        self._idempotency[key] = res
        return res

    # ---- email ---------------------------------------------------------------
    def send_email(self, to: str, subject: str, body: str, key: str) -> dict:
        if to.lower() not in self.business.known_emails():
            raise SandboxError(f"Recipient {to} is not a known contact")

        def go():
            msg = {"to": to, "subject": subject, "body": body}
            self.outbox.append(msg)
            return self._log("email.sent", to=to, subject=subject)
        return self._once(key, go)

    # ---- accounts payable -------------------------------------------------
    def reschedule_bill(self, bill_id: str, payments: list[tuple[date, float]], key: str) -> dict:
        bill = self.business.bill(bill_id)
        if abs(sum(a for _, a in payments) - bill.amount) > 0.01:
            raise SandboxError("Rescheduled payments must add up to the bill amount")

        def go():
            self.schedule_overrides[bill_id] = payments
            return self._log("ap.rescheduled", bill_id=bill_id, payments=[(d.isoformat(), a) for d, a in payments])
        return self._once(key, go)

    def hold_bill(self, bill_id: str, key: str) -> dict:
        self.business.bill(bill_id)

        def go():
            self.holds.add(bill_id)
            return self._log("ap.hold_placed", bill_id=bill_id)
        return self._once(key, go)

    def release_hold(self, bill_id: str) -> dict:
        self.holds.discard(bill_id)
        return self._log("ap.hold_released", bill_id=bill_id)

    def pay(self, payee_id: str, amount: float, key: str) -> dict:
        v = self.business.vendor(payee_id)
        if not v.verified:
            raise SandboxError("Payee bank details are not verified")

        def go():
            self.business.cash -= amount
            return self._log("bank.payment_sent", payee=v.name, amount=amount)
        return self._once(key, go)

    # ---- accounts receivable ----------------------------------------------
    def mark_reminded(self, invoice_id: str, key: str) -> dict:
        def go():
            self.reminders.add(invoice_id)
            return self._log("ar.reminder_logged", invoice_id=invoice_id)
        return self._once(key, go)

    def create_discount_offer(self, invoice_id: str, pct: float, deadline: date, key: str) -> dict:
        inv = self.business.invoice(invoice_id)

        def go():
            self.offers[invoice_id] = {"pct": pct, "deadline": deadline, "pay_amount": round(inv.amount * (1 - pct), 2)}
            return self._log("ar.discount_offer_created", invoice_id=invoice_id, pct=pct, deadline=deadline.isoformat(),
                             pay_amount=self.offers[invoice_id]["pay_amount"])
        return self._once(key, go)

    # ---- financing -----------------------------------------------------------
    def draw_financing(self, product_id: str, amount: float, flows: list[tuple[date, float]], key: str) -> dict:
        if product_id == "existing-line":
            cl = self.business.credit_line
            if cl is None or amount > cl.available + 0.01:
                raise SandboxError("Draw exceeds available credit")

        def go():
            if product_id == "existing-line":
                self.business.credit_line.drawn += amount
            self.financing.append({"product_id": product_id, "amount": amount, "flows": flows})
            return self._log("lending.drawn", product_id=product_id, amount=amount,
                             flows=[(d.isoformat(), round(a, 2)) for d, a in flows])
        return self._once(key, go)

    def overlay(self) -> dict:
        """Current commitments, in the form the forecaster consumes."""
        return {"bill_schedule": dict(self.schedule_overrides), "holds": set(self.holds),
                "offers": dict(self.offers), "reminders": set(self.reminders),
                "extra_flows": [f for fin in self.financing for f in fin["flows"]]}
