"""Sandbox financing marketplace.

Fictional lenders and products with realistic pricing structures. The
matcher in `dorsal.planning.financing` turns each into a dated cash-flow
schedule and computes the true APR, so products with very different fee
structures (interest, origination fees, factor rates, per-30-day fees)
can be compared fairly.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FinancingProduct:
    id: str
    lender: str
    name: str
    kind: str                       # credit_line | invoice_finance | term_loan | mca | revolving
    min_amount: float
    max_amount: float
    days_to_fund: int
    apr: float = 0.0                # nominal interest rate
    origination_fee_pct: float = 0.0
    factor_rate: float = 0.0        # merchant cash advance: repay amount * factor
    holdback_pct: float = 0.0       # MCA: share of daily card sales taken
    advance_rate: float = 0.0       # invoice finance: share of invoice advanced
    fee_per_30d: float = 0.0        # invoice finance fee per 30 days outstanding
    term_days: int = 0              # term loans: amortisation length
    min_months: int = 0
    min_annual_revenue: float = 0.0
    min_dscr: float = 0.0
    bands: tuple = ("A", "B", "C", "D")
    requires_card_sales: bool = False
    requires_invoices: bool = False
    early_repay_ok: bool = True
    notes: str = ""
    tags: list = field(default_factory=list)


CATALOG: list[FinancingProduct] = [
    FinancingProduct(
        "existing-line", "Your bank", "Existing business credit line", "credit_line",
        min_amount=1000, max_amount=0, days_to_fund=0, apr=0.115,
        notes="Already approved. Interest only on what you draw; repay any time."),
    FinancingProduct(
        "flex-wc", "Harbor Digital Bank", "Flex working-capital loan", "revolving",
        min_amount=5000, max_amount=150000, days_to_fund=1, apr=0.098, origination_fee_pct=0.0,
        min_months=12, min_annual_revenue=150000, min_dscr=1.15, bands=("A", "B"),
        notes="Unsecured, fully online. Daily interest, repay any time with no penalty."),
    FinancingProduct(
        "seasonal", "Harbor Digital Bank", "Seasonal credit extension", "revolving",
        min_amount=5000, max_amount=60000, days_to_fund=2, apr=0.089, origination_fee_pct=0.005,
        min_months=24, min_annual_revenue=200000, min_dscr=1.2, bands=("A", "B"),
        notes="Sized to a predictable seasonal gap and repaid from the season's receipts."),
    FinancingProduct(
        "inv-fin", "Ledgerly Capital", "Invoice financing", "invoice_finance",
        min_amount=2000, max_amount=250000, days_to_fund=2, advance_rate=0.85, fee_per_30d=0.016,
        min_months=6, requires_invoices=True,
        notes="Advance against unpaid invoices; repaid when your customer pays."),
    FinancingProduct(
        "sba-term", "Provincial SME Guarantee Program", "Government-backed term loan", "term_loan",
        min_amount=25000, max_amount=350000, days_to_fund=21, apr=0.072, origination_fee_pct=0.02,
        term_days=3 * 365, min_months=12, min_dscr=1.25, bands=("A", "B", "C"),
        notes="Cheap long-term money, but approval and funding take about three weeks."),
    FinancingProduct(
        "mca", "QuickCash Advance", "Merchant cash advance", "mca",
        min_amount=5000, max_amount=100000, days_to_fund=0, factor_rate=1.32, holdback_pct=0.14,
        min_months=4, requires_card_sales=True, early_repay_ok=False,
        notes="Fast and easy to qualify, repaid from a fixed share of daily card sales."),
]
