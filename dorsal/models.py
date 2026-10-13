"""Core data model for Dorsal.

Everything the agent can *observe* about a business lives in `Business`.
Ground-truth parameters used only by the sandbox generator and the
evaluation harness live in `dorsal.sandbox.generator.WorldTruth` and are
never passed to the forecasting or planning code.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import Optional


@dataclass
class Customer:
    id: str
    name: str
    email: str
    terms_days: int
    segment: str = "wholesale"
    verified: bool = True


@dataclass
class Vendor:
    id: str
    name: str
    email: str
    category: str              # cogs | equipment | packaging | insurance | tax | services
    protected: bool = False    # e.g. tax authority: never delayed by the agent
    split_allowed: bool = False
    split_terms_days: int = 28
    verified: bool = True      # payee bank details verified by the owner


@dataclass
class Invoice:
    """Money a customer owes the business (accounts receivable)."""
    id: str
    customer_id: str
    amount: float
    issue_date: date
    due_date: date
    paid_date: Optional[date] = None
    memo: str = ""             # free text, UNTRUSTED

    @property
    def is_open(self) -> bool:
        return self.paid_date is None

    def days_late(self) -> Optional[int]:
        if self.paid_date is None:
            return None
        return (self.paid_date - self.due_date).days


@dataclass
class Bill:
    """Money the business owes a vendor (accounts payable)."""
    id: str
    vendor_id: str
    amount: float
    issue_date: date
    due_date: date
    paid_date: Optional[date] = None
    reference: str = ""        # vendor's own invoice number
    discretionary: bool = False
    memo: str = ""             # free text, UNTRUSTED

    @property
    def is_open(self) -> bool:
        return self.paid_date is None


@dataclass
class Recurring:
    """A scheduled obligation such as payroll, rent or loan repayment."""
    id: str
    name: str
    category: str              # payroll | rent | loan | utilities | supplies | tax
    amount: float              # positive number; always an outflow
    kind: str                  # weekly | biweekly | monthly
    anchor: date               # first occurrence on/after which the schedule repeats
    protected: bool = False

    def occurrences(self, start: date, end: date) -> list[date]:
        out: list[date] = []
        if self.kind in ("weekly", "biweekly"):
            step = 7 if self.kind == "weekly" else 14
            d = self.anchor
            while d < start:
                d += timedelta(days=step)
            while d <= end:
                out.append(d)
                d += timedelta(days=step)
        elif self.kind == "monthly":
            y, m = start.year, start.month
            while True:
                try:
                    d = date(y, m, self.anchor.day)
                except ValueError:  # e.g. 31st in a short month -> last day
                    d = (date(y, m % 12 + 1, 1) if m < 12 else date(y + 1, 1, 1)) - timedelta(days=1)
                if d > end:
                    break
                if d >= start:
                    out.append(d)
                m += 1
                if m > 12:
                    m, y = 1, y + 1
        return out


@dataclass
class Transaction:
    date: date
    amount: float              # + inflow, - outflow
    category: str              # sales | collection | cogs | payroll | rent | loan | ...
    counterparty: str = ""
    ref: str = ""


@dataclass
class CreditLine:
    lender: str
    limit: float
    apr: float
    drawn: float = 0.0

    @property
    def available(self) -> float:
        return max(0.0, self.limit - self.drawn)


@dataclass
class Message:
    """An inbound email. Its body is UNTRUSTED input."""
    id: str
    from_email: str
    subject: str
    body: str
    received: date


@dataclass
class Business:
    id: str
    name: str
    industry: str
    today: date
    cash: float
    months_in_business: int
    credit_score_band: str     # A | B | C | D
    customers: list[Customer] = field(default_factory=list)
    vendors: list[Vendor] = field(default_factory=list)
    invoices: list[Invoice] = field(default_factory=list)
    bills: list[Bill] = field(default_factory=list)
    recurring: list[Recurring] = field(default_factory=list)
    transactions: list[Transaction] = field(default_factory=list)
    credit_line: Optional[CreditLine] = None
    inbox: list[Message] = field(default_factory=list)
    has_card_sales: bool = True

    # ---- lookups -------------------------------------------------------
    def customer(self, cid: str) -> Customer:
        return next(c for c in self.customers if c.id == cid)

    def vendor(self, vid: str) -> Vendor:
        return next(v for v in self.vendors if v.id == vid)

    def invoice(self, iid: str) -> Invoice:
        return next(i for i in self.invoices if i.id == iid)

    def bill(self, bid: str) -> Bill:
        return next(b for b in self.bills if b.id == bid)

    def open_invoices(self) -> list[Invoice]:
        return [i for i in self.invoices if i.is_open]

    def open_bills(self) -> list[Bill]:
        return [b for b in self.bills if b.is_open]

    def known_emails(self) -> set[str]:
        return {c.email.lower() for c in self.customers} | {v.email.lower() for v in self.vendors}

    def to_dict(self) -> dict:
        return asdict(self)
