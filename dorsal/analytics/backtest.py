"""Rolling-origin backtest of the cash forecast.

For several past dates ("origins") we rebuild exactly what Dorsal would have
known on that day, forecast forward, and compare with what really happened
in the bank account. We report:

* coverage of the P10-P90 band (well calibrated = about 80%),
* mean absolute error of the median forecast, and
* skill versus a naive baseline that extrapolates the last 4 weeks' average
  daily net cash flow (skill > 0 means Dorsal beats the baseline).
"""
from __future__ import annotations

import copy
from datetime import date, timedelta

import numpy as np

from dorsal.analytics.forecast import Forecaster
from dorsal.models import Business


def view_as_of(biz: Business, origin: date) -> Business:
    v = copy.deepcopy(biz)
    later = sum(t.amount for t in biz.transactions if origin <= t.date < biz.today)
    v.today = origin
    v.cash = biz.cash - later
    v.transactions = [t for t in v.transactions if t.date < origin]
    v.invoices = [i for i in v.invoices if i.issue_date < origin]
    for i in v.invoices:
        if i.paid_date is not None and i.paid_date >= origin:
            i.paid_date = None
    v.bills = [b for b in v.bills if b.issue_date < origin]
    for b in v.bills:
        if b.paid_date is not None and b.paid_date >= origin:
            b.paid_date = None
    v.inbox = []
    return v


def actual_path(biz: Business, origin: date, horizon: int) -> np.ndarray:
    start_cash = biz.cash - sum(t.amount for t in biz.transactions if origin <= t.date < biz.today)
    flows = np.zeros(horizon)
    for t in biz.transactions:
        i = (t.date - origin).days
        if 0 <= i < horizon:
            flows[i] += t.amount
    return start_cash + np.cumsum(flows)


def backtest(biz: Business, weeks_back: tuple[int, ...] = (13, 15, 17, 19, 21, 23, 25), horizon: int = 91,
             n_paths: int = 500) -> dict:
    rows = []
    for wb in weeks_back:
        origin = biz.today - timedelta(days=7 * wb)
        h = min(horizon, (biz.today - origin).days)
        view = view_as_of(biz, origin)
        if not any(t.date < origin - timedelta(days=150) for t in view.transactions):
            continue
        fc = Forecaster(view, horizon=h, n_paths=n_paths, seed=wb).run()
        q = fc.quantiles()
        act = actual_path(biz, origin, h)
        checkpoints = list(range(6, h, 7))
        hist_net = [t.amount for t in view.transactions if t.date >= origin - timedelta(days=28)]
        naive_daily = sum(hist_net) / 28
        naive = view.cash + naive_daily * (np.arange(h) + 1)
        scale = np.mean(np.abs([t.amount for t in view.transactions if t.date >= origin - timedelta(days=91)])) * 7
        for i in checkpoints:
            rows.append({
                "origin": origin.isoformat(), "day": i + 1, "actual": float(act[i]),
                "p10": float(q[10][i]), "p50": float(q[50][i]), "p90": float(q[90][i]),
                "in_band": bool(q[10][i] <= act[i] <= q[90][i]),
                "abs_err": float(abs(q[50][i] - act[i])), "naive_abs_err": float(abs(naive[i] - act[i])),
                "scale": float(scale)})
    if not rows:
        return {"n": 0}
    cov = np.mean([r["in_band"] for r in rows])
    mae = np.mean([r["abs_err"] for r in rows])
    nmae = np.mean([r["naive_abs_err"] for r in rows])
    return {"n_checkpoints": len(rows), "origins": len(weeks_back), "coverage_p10_p90": round(float(cov), 3),
            "mae_median": round(float(mae), 0), "mae_naive": round(float(nmae), 0),
            "skill_vs_naive": round(float(1 - mae / nmae), 3) if nmae else None,
            "mae_in_weeks_of_flow": round(float(mae / np.mean([r["scale"] for r in rows])), 3), "rows": rows}
