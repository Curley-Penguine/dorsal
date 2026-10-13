"""Probabilistic 13-week cash forecast (Monte Carlo).

Model components, all fitted from the business's own history:

* Card / cafe sales: weekday profile x recent level x shrunk monthly
  seasonality, with week-block bootstrapped multiplicative residuals and a
  per-path level shock.
* Open invoices: each customer's days-late distribution is learned from
  their paid invoices, shrunk towards the portfolio-wide distribution
  (empirical-Bayes style mixture, weight n/(n+k)). Invoices that are
  already overdue are sampled *conditionally* on being at least that late.
  Aged invoices carry a non-payment probability.
* Future invoices: per-customer ordering frequency and order size, then the
  same payment-timing model.
* Bills and obligations: open bills, recurring obligations (payroll, rent,
  loan...) and projected regular supplier bills.

All random numbers are drawn once in `fit()`, so evaluating different plans
with `run(overlay)` uses common random numbers: differences between plans
reflect the plans, not simulation noise.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from statistics import median
from typing import Optional

import numpy as np

from dorsal.models import Business

QUANTILES = (5, 10, 25, 50, 75, 90, 95)


@dataclass
class Assumptions:
    """Behavioural parameters that cannot be learned from bank data alone.
    They are surfaced to the owner and listed in the security report."""
    shrinkage_k: float = 5.0           # pseudo-count for delay-distribution shrinkage
    reminder_factor: float = 0.6       # remaining wait after a reminder, as a share of the original
    discount_accept_p: float = 0.5     # chance a customer takes an early-payment discount
    promise_kept_p: float = 0.8        # chance a written payment promise is kept
    default_p: float = 0.01            # chance a current invoice is not paid within the horizon
    aged_default_p: float = 0.35       # same, for invoices more than 60 days overdue
    seasonality_shrink: float = 0.6    # how much of last year's monthly pattern to trust
    level_sd: float = 0.05             # per-path uncertainty in the sales level


@dataclass
class Overlay:
    """Changes to apply on top of the business's current schedule."""
    bill_schedule: dict = field(default_factory=dict)      # bill_id -> [(date, amount)]
    holds: set = field(default_factory=set)                # bill ids not paid in horizon
    offers: dict = field(default_factory=dict)             # invoice_id -> {pct, deadline}
    reminders: set = field(default_factory=set)            # invoice ids reminded
    promises: dict = field(default_factory=dict)           # invoice_id -> promised date
    extra_flows: list = field(default_factory=list)        # [(date, amount)]
    extra_delays: dict = field(default_factory=dict)       # invoice_id -> extra days (what-if)
    sales_factor: float = 1.0                              # what-if: sales shock

    def merged(self, other: "Overlay") -> "Overlay":
        return Overlay(
            bill_schedule={**self.bill_schedule, **other.bill_schedule},
            holds=self.holds | other.holds,
            offers={**self.offers, **other.offers},
            reminders=self.reminders | other.reminders,
            promises={**self.promises, **other.promises},
            extra_flows=list(self.extra_flows) + list(other.extra_flows),
            extra_delays={**self.extra_delays, **other.extra_delays},
            sales_factor=self.sales_factor * other.sales_factor)

    @staticmethod
    def from_world(o: dict) -> "Overlay":
        return Overlay(bill_schedule=o.get("bill_schedule", {}), holds=o.get("holds", set()),
                       offers=o.get("offers", {}), reminders=o.get("reminders", set()),
                       extra_flows=o.get("extra_flows", []))


@dataclass
class ForecastResult:
    start: date
    horizon: int
    buffer: float
    balances: np.ndarray               # (paths, horizon) closing balance per day
    flows_by_cat: dict                 # category -> (horizon,) mean daily flow
    events: list                       # notable dated events (median view)
    risk_from: int = 3                 # risk is measured from this day on (actions need time to land)

    @property
    def dates(self) -> list[date]:
        return [self.start + timedelta(days=i) for i in range(self.horizon)]

    @property
    def mins(self) -> np.ndarray:
        return self.balances[:, self.risk_from:].min(axis=1)

    def quantiles(self) -> dict[int, np.ndarray]:
        q = np.percentile(self.balances, QUANTILES, axis=0)
        return {k: q[i] for i, k in enumerate(QUANTILES)}

    def p_below(self, level: float) -> float:
        return float((self.mins < level).mean())

    @property
    def p_shortfall(self) -> float:
        return self.p_below(0.0)

    @property
    def p_below_buffer(self) -> float:
        return self.p_below(self.buffer)

    @property
    def expected_deficit(self) -> float:
        """Average depth below the buffer at the worst point, over all paths."""
        return float(np.maximum(0.0, self.buffer - self.mins).mean())

    def summary(self) -> dict:
        q = self.quantiles()
        p50 = q[50]
        low_i = self.risk_from + int(np.argmin(p50[self.risk_from:]))
        p_day = (self.balances < 0).mean(axis=0)
        first_risk = next((i for i, p in enumerate(p_day) if p >= 0.2), None)
        return {
            "start": self.start.isoformat(),
            "horizon_days": self.horizon,
            "buffer": self.buffer,
            "p_shortfall": round(self.p_shortfall, 3),
            "p_below_buffer": round(self.p_below_buffer, 3),
            "expected_deficit": round(self.expected_deficit, 2),
            "median_low": round(float(p50[low_i]), 2),
            "median_low_date": (self.start + timedelta(days=low_i)).isoformat(),
            "p10_low": round(float(q[10][self.risk_from:].min()), 2),
            "p90_low": round(float(q[90][self.risk_from:].min()), 2),
            "first_day_shortfall_risk_20pct": None if first_risk is None else (self.start + timedelta(days=first_risk)).isoformat(),
            "ending_p50": round(float(p50[-1]), 2),
        }

    def weekly(self) -> list[dict]:
        q = self.quantiles()
        out = []
        for w in range(self.horizon // 7):
            i = w * 7 + 6
            row = {"week_ending": (self.start + timedelta(days=i)).isoformat(),
                   "p10": round(float(q[10][i]), 0), "p50": round(float(q[50][i]), 0), "p90": round(float(q[90][i]), 0),
                   "p_below_zero": round(float((self.balances[:, i] < 0).mean()), 3)}
            for cat, arr in self.flows_by_cat.items():
                row[cat] = round(float(arr[w * 7:(w + 1) * 7].sum()), 0)
            out.append(row)
        return out


class Forecaster:
    def __init__(self, biz: Business, horizon: int = 91, n_paths: int = 1000, seed: int = 42,
                 buffer: float = 0.0, assumptions: Optional[Assumptions] = None):
        self.biz = biz
        self.H = horizon
        self.N = n_paths
        self.seed = seed
        self.buffer = buffer
        self.a = assumptions or Assumptions()
        self.today = biz.today
        self._fit()

    # ------------------------------------------------------------------ fit
    def _fit(self) -> None:
        biz, H, N = self.biz, self.H, self.N
        rng = np.random.default_rng(self.seed)
        self.rng = rng
        today = self.today

        # --- sales model -----------------------------------------------------
        sales = defaultdict(float)
        for t in biz.transactions:
            if t.category == "sales":
                sales[t.date] += t.amount
        self.has_sales = bool(sales)
        self.sales_det = np.zeros(H)
        self.sales_resid = np.ones((N, H))
        self.sales_level_shock = np.ones((N, 1))
        if sales:
            hist_days = [today - timedelta(days=i) for i in range(1, 183)]
            hist = np.array([sales.get(d, 0.0) for d in hist_days])[::-1]          # oldest -> newest
            hdates = hist_days[::-1]
            wd = np.array([d.weekday() for d in hdates])
            overall = hist.mean() or 1.0
            wfac = np.array([hist[wd == k].mean() / overall if (wd == k).any() else 1.0 for k in range(7)])
            # monthly seasonality from the last 12 months, shrunk towards 1
            month_tot, month_n = defaultdict(float), defaultdict(int)
            for d, v in sales.items():
                if d >= today - timedelta(days=365):
                    month_tot[d.month] += v
                    month_n[d.month] += 1
            year_mean = (sum(month_tot.values()) / max(1, sum(month_n.values())))
            sfac = {m: 1 + self.a.seasonality_shrink * ((month_tot[m] / month_n[m]) / year_mean - 1)
                    for m in month_tot if month_n[m] >= 20}
            recent = hist[-28:] / wfac[wd[-28:]]
            cur_s = np.mean([sfac.get(d.month, 1.0) for d in hdates[-28:]])
            level = recent.mean() / cur_s
            self.sales_model = {"weekday_factors": wfac.round(3).tolist(), "level": round(level, 2),
                                "season": {m: round(v, 3) for m, v in sorted(sfac.items())}}
            fut = [today + timedelta(days=i) for i in range(H)]
            self.sales_det = np.array([level * wfac[d.weekday()] * sfac.get(d.month, 1.0) for d in fut])
            # residuals relative to a centred 28-day level
            fitted = np.array([wfac[wd[i]] for i in range(len(hist))])
            ratio = hist / np.maximum(fitted, 1e-9)
            lvl = np.convolve(ratio, np.ones(28) / 28, mode="same")
            resid = ratio / np.maximum(lvl, 1e-9)
            resid = resid[14:-14] if len(resid) > 40 else resid
            n_weeks = len(resid) // 7
            blocks = resid[: n_weeks * 7].reshape(n_weeks, 7)
            pick = rng.integers(0, n_weeks, size=(N, math.ceil(H / 7)))
            self.sales_resid = blocks[pick].reshape(N, -1)[:, :H]
            self.sales_level_shock = np.exp(rng.normal(0, self.a.level_sd, size=(N, 1)))

        # --- payment timing model (Kaplan-Meier, handles right-censoring) -----------
        # Paid invoices are observed events. Open invoices are right-censored: we
        # only know they are *at least* (today - due) days late. Ignoring them
        # would bias lateness downwards for slow payers.
        obs: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for inv in biz.invoices:
            if inv.paid_date is not None:
                obs[inv.customer_id].append((inv.days_late(), 1))
            else:
                obs[inv.customer_id].append(((today - inv.due_date).days, 0))
        all_obs = [o for v in obs.values() for o in v] or [(5, 1)]
        self.km_pool = _kaplan_meier(all_obs)
        self.km_own = {c: _kaplan_meier(v) for c, v in obs.items() if any(e for _, e in v)}
        self.n_events = {c: sum(e for _, e in v) for c, v in obs.items()}
        self.delay_stats = {}
        for c in biz.customers:
            km = self.km_own.get(c.id)
            self.delay_stats[c.id] = {
                "n_paid": self.n_events.get(c.id, 0),
                "median_days_late": None if km is None else float(_km_quantile(km, np.array([0.5]))[0]),
                "p90_days_late": None if km is None else float(_km_quantile(km, np.array([0.9]))[0])}

        # --- per-invoice random draws (open invoices) ---------------------------
        self.open_inv = biz.open_invoices()
        k = len(self.open_inv)
        self.u_choose = rng.random((k, N))
        self.u_q = rng.random((k, N))
        self.u_accept = rng.random((k, N))
        self.u_default = rng.random((k, N))
        self.u_promise = rng.random((k, N))
        self.u_jitter = rng.integers(0, 4, size=(k, N))

        # --- future invoicing ---------------------------------------------------------
        horizon_issue = [today + timedelta(days=i) for i in range(H) if (today + timedelta(days=i)).weekday() in (0, 3)]
        self.future_issue_days = horizon_issue
        lookback = today - timedelta(days=182)
        issue_slots = sum(1 for i in range(182) if (lookback + timedelta(days=i)).weekday() in (0, 3)) or 1
        self.order_model = {}
        for c in biz.customers:
            amts = [inv.amount for inv in biz.invoices if inv.customer_id == c.id and inv.issue_date >= lookback]
            if not amts:
                continue
            logs = np.log(amts)
            self.order_model[c.id] = {"p": min(1.0, len(amts) / issue_slots), "mu": float(logs.mean()),
                                      "sd": float(logs.std() or 0.2)}
        m = len(self.order_model) * len(horizon_issue)
        self.f_u_order = rng.random((m, N))
        self.f_z_amt = rng.normal(size=(m, N))
        self.f_u_choose = rng.random((m, N))
        self.f_u_q = rng.random((m, N))

        # --- projected regular supplier bills ---------------------------------------
        self.projected_bills = self._project_bills()
        self.pb_z = rng.normal(size=(max(1, len(self.projected_bills)), N))

    def _revenue_by_month(self) -> dict:
        rev = defaultdict(float)
        for t in self.biz.transactions:
            if t.category == "sales":
                rev[(t.date.year, t.date.month)] += t.amount
        for inv in self.biz.invoices:
            rev[(inv.issue_date.year, inv.issue_date.month)] += inv.amount
        return rev

    def _project_bills(self) -> list[tuple[date, float, str, str, float]]:
        """Project regular supplier bills not yet received.

        Monthly suppliers are modelled as a learned share of the previous
        month's revenue (purchases follow sales), with the spread of that
        share giving per-path amount uncertainty. Quarterly payees (insurance,
        tax) repeat their last amount."""
        biz, today = self.biz, self.today
        end = today + timedelta(days=self.H - 1)
        out = []
        rev = self._revenue_by_month()
        # expected revenue for the current, partially observed month
        last28 = sum(t.amount for t in biz.transactions if t.category == "sales" and t.date >= today - timedelta(days=28))
        last28 += sum(i.amount for i in biz.invoices if i.issue_date >= today - timedelta(days=28))
        run_rate_month = last28 / 28 * 30.4

        def prev_month(y, m):
            return (y - 1, 12) if m == 1 else (y, m - 1)

        def month_rev(key):
            if key >= (today.year, today.month):
                return run_rate_month
            return rev.get(key, 0.0)

        by_vendor = defaultdict(list)
        for b in biz.bills:
            if b.issue_date >= today - timedelta(days=365) and not b.discretionary:
                by_vendor[b.vendor_id].append(b)
        for vid, bl in by_vendor.items():
            bl.sort(key=lambda b: b.issue_date)
            recent6 = [b for b in bl if b.issue_date >= today - timedelta(days=183)]
            terms = int(median([(b.due_date - b.issue_date).days for b in bl]))
            last_issue = bl[-1].issue_date
            vname = biz.vendor(vid).name
            if len(recent6) >= 4:   # monthly supplier
                ratios = []
                for b in bl:
                    base = rev.get(prev_month(b.issue_date.year, b.issue_date.month), 0.0)
                    first_hist = min((t.date for t in biz.transactions), default=today)
                    pm = prev_month(b.issue_date.year, b.issue_date.month)
                    if base > 0 and date(*pm, 1) >= first_hist:
                        ratios.append(b.amount / base)
                use_ratio = len(ratios) >= 3
                if use_ratio:
                    lr = np.log(ratios)
                    r_med, sd = float(np.exp(np.median(lr))), float(np.std(lr))
                else:
                    amts = [b.amount for b in recent6]
                    r_med, sd = 0.0, float(np.std(np.log(amts)))
                dom = int(median([b.issue_date.day for b in recent6]))
                y, mth = today.year, today.month
                for _ in range(5):
                    try:
                        issue = date(y, mth, dom)
                    except ValueError:
                        issue = date(y, mth, 28)
                    # a regular bill is assumed already covered if a real bill from this
                    # vendor was issued in the 25 days before the projected issue date
                    covered = any(0 <= (issue - b.issue_date).days <= 25 for b in bl)
                    if issue > last_issue and issue >= today and not covered and issue + timedelta(days=terms) <= end:
                        amt = r_med * month_rev(prev_month(y, mth)) if use_ratio else float(median([b.amount for b in recent6][-4:]))
                        out.append((issue + timedelta(days=terms), amt, vid, f"Projected {vname} bill", min(sd, 0.6)))
                    mth += 1
                    if mth > 12:
                        mth, y = 1, y + 1
            elif len(bl) >= 2:      # quarterly-ish
                gap = (bl[-1].issue_date - bl[-2].issue_date).days
                if 70 <= gap <= 110:
                    nxt = last_issue + timedelta(days=gap)
                    if nxt >= today and nxt + timedelta(days=terms) <= end:
                        out.append((nxt + timedelta(days=terms), bl[-1].amount, vid, f"Projected {vname} bill", 0.1))
        return out

    # ------------------------------------------------------------- helpers
    def _idx(self, d: date) -> int:
        return (d - self.today).days

    def _sample_delay(self, cid: str, u_choose: np.ndarray, u_q: np.ndarray, at_least: Optional[int] = None) -> np.ndarray:
        """Days late, from the customer's own KM curve shrunk towards the
        portfolio curve; optionally conditional on being >= `at_least`."""
        n = self.n_events.get(cid, 0)
        w = n / (n + self.a.shrinkage_k)
        pool = _km_quantile(self.km_pool, u_q, at_least)
        own_km = self.km_own.get(cid)
        if own_km is None:
            return pool
        own = _km_quantile(own_km, u_q, at_least)
        return np.where(u_choose < w, own, pool)

    def expected_pay_dates(self, q: float = 0.5) -> dict[str, date]:
        """Quantile of the payment date for each open invoice (default median)."""
        out = {}
        for j, inv in enumerate(self.open_inv):
            overdue = (self.today - inv.due_date).days
            late = self._sample_delay(inv.customer_id, self.u_choose[j], self.u_q[j], overdue if overdue > 0 else None)
            idx = np.maximum((inv.due_date - self.today).days + late, 1)
            out[inv.id] = self.today + timedelta(days=int(np.quantile(idx, q)))
        return out

    # --------------------------------------------------------------------- run
    def run(self, overlay: Optional[Overlay] = None, include_open_ar: bool = True) -> ForecastResult:
        ov = overlay or Overlay()
        biz, H, N, today, a = self.biz, self.H, self.N, self.today, self.a
        bal_flows = np.zeros((N, H))
        cats: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(H))
        events = []

        def add_det(d: date, amt: float, cat: str, label: str = ""):
            i = self._idx(d)
            if i < 0:
                i = 0
            if i >= H:
                return
            bal_flows[:, i] += amt
            cats[cat][i] += amt
            if label and abs(amt) >= 2500:
                events.append({"date": d.isoformat(), "amount": round(amt, 2), "label": label, "category": cat})

        # sales
        if self.has_sales:
            s = self.sales_det[None, :] * self.sales_resid * self.sales_level_shock * ov.sales_factor
            bal_flows += s
            cats["sales"] += s.mean(axis=0)

        # open invoices
        if include_open_ar:
            for j, inv in enumerate(self.open_inv):
                overdue = (today - inv.due_date).days
                at_least = overdue if overdue > 0 else None
                late = self._sample_delay(inv.customer_id, self.u_choose[j], self.u_q[j], at_least)
                paid_idx = (inv.due_date - today).days + late
                paid_idx = np.maximum(paid_idx, 1 + self.u_jitter[j] % 3)
                amount = np.full(N, inv.amount)
                if inv.id in ov.reminders and overdue > 0:
                    paid_idx = np.ceil(paid_idx * a.reminder_factor)
                if inv.id in ov.promises:
                    pidx = self._idx(ov.promises[inv.id])
                    kept = self.u_promise[j] < a.promise_kept_p
                    paid_idx = np.where(kept, max(pidx, 0), np.maximum(paid_idx, pidx + 3))
                if inv.id in ov.offers:
                    off = ov.offers[inv.id]
                    didx = self._idx(off["deadline"])
                    accept = self.u_accept[j] < a.discount_accept_p
                    early = np.minimum(paid_idx, didx - (self.u_jitter[j] % 4))
                    paid_idx = np.where(accept, early, paid_idx)
                    amount = np.where(accept, inv.amount * (1 - off["pct"]), amount)
                if inv.id in ov.extra_delays:
                    paid_idx = paid_idx + ov.extra_delays[inv.id]
                p_def = a.aged_default_p if overdue > 60 else a.default_p
                unpaid = self.u_default[j] < p_def
                paid_idx = np.where(unpaid, H + 999, paid_idx).astype(int)
                mask = paid_idx < H
                rows = np.nonzero(mask)[0]
                np.add.at(bal_flows, (rows, paid_idx[mask]), amount[mask])
                np.add.at(cats["collections"], paid_idx[mask], amount[mask] / N)
                med = int(np.median(np.where(mask, paid_idx, H)))
                if med < H and inv.amount >= 2500:
                    events.append({"date": (today + timedelta(days=med)).isoformat(), "amount": inv.amount,
                                   "label": f"{biz.customer(inv.customer_id).name} pays {inv.id} (median)",
                                   "category": "collections"})

        # future invoices
        r = 0
        for cid, om in self.order_model.items():
            terms = biz.customer(cid).terms_days
            for d in self.future_issue_days:
                order = self.f_u_order[r] < om["p"]  # p is per issue slot
                amt = np.exp(om["mu"] + om["sd"] * self.f_z_amt[r])
                late = self._sample_delay(cid, self.f_u_choose[r], self.f_u_q[r])
                idx = (d - today).days + terms + late
                idx = idx.astype(int)
                mask = order & (idx < H) & (idx >= 0)
                rows = np.nonzero(mask)[0]
                np.add.at(bal_flows, (rows, idx[mask]), amt[mask])
                np.add.at(cats["new_invoices"], idx[mask], amt[mask] / N)
                r += 1

        # open bills
        for b in biz.open_bills():
            v = biz.vendor(b.vendor_id)
            if b.id in ov.holds:
                continue
            sched = ov.bill_schedule.get(b.id) or [(b.due_date, b.amount)]
            for d, amt in sched:
                add_det(d, -amt, "tax" if v.category == "tax" else "bills", f"{v.name} {b.reference or b.id}")
        for k, (d, amt, vid, label, sd) in enumerate(self.projected_bills):
            i = self._idx(d)
            if 0 <= i < H:
                path_amt = amt * np.exp(sd * self.pb_z[k] - sd * sd / 2)
                bal_flows[:, i] -= path_amt
                cats["bills"][i] -= path_amt.mean()
                if amt >= 2500:
                    events.append({"date": d.isoformat(), "amount": -round(amt, 2), "label": label, "category": "bills"})

        # recurring obligations
        end = today + timedelta(days=H - 1)
        for rcur in biz.recurring:
            for d in rcur.occurrences(today, end):
                add_det(d, -rcur.amount, rcur.category, rcur.name if rcur.amount >= 2500 else "")

        # financing and other overlay flows
        for d, amt in ov.extra_flows:
            add_det(d, amt, "financing", "Financing" if amt > 0 else "Financing repayment")

        balances = biz.cash + np.cumsum(bal_flows, axis=1)
        events.sort(key=lambda e: e["date"])
        return ForecastResult(today, H, self.buffer, balances, dict(cats), events)


def _kaplan_meier(obs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (event times, CDF at those times, mean of the observed tail)."""
    arr = np.array(obs, dtype=float)
    times = np.unique(arr[arr[:, 1] == 1, 0])
    surv, s = [], 1.0
    for t in times:
        at_risk = np.sum(arr[:, 0] >= t)
        d = np.sum((arr[:, 0] == t) & (arr[:, 1] == 1))
        if at_risk > 0:
            s *= 1 - d / at_risk
        surv.append(s)
    cdf = 1 - np.array(surv)
    ev = arr[arr[:, 1] == 1, 0]
    tail = float(np.mean(ev[ev >= np.percentile(ev, 75)]) - np.percentile(ev, 75)) if len(ev) >= 4 else 7.0
    return times, cdf, max(tail, 3.0)


def _km_quantile(km, u: np.ndarray, at_least: Optional[int] = None) -> np.ndarray:
    times, cdf, tail = km
    if len(times) == 0:
        return np.full(np.shape(u), float(at_least or 5))
    lo = 0.0
    if at_least is not None:
        k = np.searchsorted(times, at_least, side="left")
        lo = cdf[k - 1] if k > 0 else 0.0
    uu = lo + np.asarray(u) * (1 - lo)
    idx = np.searchsorted(cdf, uu, side="left")
    top = cdf[-1]
    inside = idx < len(times)
    out = np.where(inside, times[np.minimum(idx, len(times) - 1)], 0.0)
    # beyond the last observed event (censored mass): extend with an exponential tail
    excess = np.clip((uu - top) / max(1 - top, 1e-9), 0, 0.999)
    out = np.where(inside, out, times[-1] + 1 - tail * np.log(1 - excess))
    if at_least is not None:
        out = np.maximum(out, at_least)
    return out


def deterministic_forecast(biz: Business, horizon: int = 91) -> np.ndarray:
    """Naive single-line forecast used as a baseline: every invoice is paid at
    due date + the customer's average lateness, sales at the recent average."""
    f = Forecaster(biz, horizon=horizon, n_paths=200, seed=1)
    res = f.run()
    return res.balances.mean(axis=0)
