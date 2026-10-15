# Dorsal: technical documentation

Dorsal is an AI agent that manages a small business's cash. It forecasts, analyses, plans and acts, and it can only act
through permission tiers enforced outside the language model. This document covers the system architecture, the core
algorithms and the security design. Measured results are in `docs/evidence/`.

## 1. System architecture

| Layer | Module | Responsibility |
|---|---|---|
| Data | `dorsal/models.py`, `dorsal/sandbox/` | Business, customers, vendors, invoices, bills, recurring obligations, transactions, inbox. Sandbox bank, ledger, email and lender marketplace stand in for production APIs. |
| Analytics | `dorsal/analytics/` | Probabilistic forecast, backtest, financial analysis, alerts |
| Planning | `dorsal/planning/` | Lever generation, chance-constrained optimiser, financing matcher |
| Agent | `dorsal/agent/` | Tool registry, LLM providers, agent loop (morning review, chat) |
| Control | `dorsal/security/`, `dorsal/runtime.py` | Policy engine, approvals, audit log, untrusted-content guard, execution pipeline |
| Interface | `dorsal/api.py`, `dorsal/web/index.html` | HTTP API and web console |

**Execution pipeline** (`dorsal/runtime.py`). Every action, whoever proposed it, follows one path:

1. `propose(action)`: the policy engine classifies it as `auto`, `approve` or `deny`. The proposal and the decision are
   written to the audit log.
2. `auto` actions execute immediately. `approve` actions wait in a queue. `deny` actions are recorded and never run.
3. `approve(action_id, pin)`: for step-up tiers the owner's PIN is checked; then an HMAC-signed approval token is issued
   for that action's exact payload hash.
4. `_execute`: the policy is evaluated **again** (state may have changed, for example the owner pressed the kill
   switch), the token is verified, and only then is the sandbox called. The result is audit-logged.

**Agent.** The agent has 10 read tools and 2 propose tools (`propose_plan_step`, `propose_action`). There is no tool to
execute, approve, pay, or edit policy. It runs with Claude (Anthropic API), any OpenAI-compatible model (DeepSeek, Qwen,
OpenAI) or offline, where a deterministic playbook calls the same tools. Offline mode makes the whole system
reproducible for reviewers without an API key.

## 2. Core algorithms

### 2.1 Probabilistic 13-week forecast (`analytics/forecast.py`)

Cash on day *t* in simulated future *k* is today's cash plus all simulated flows up to *t*. 1,000 futures are simulated.

- **Card and counter sales.** Weekday profile × recent level (last 28 days, de-seasonalised) × monthly seasonality from
  the last year, shrunk 40% towards 1 to avoid over-fitting one year. Uncertainty comes from week-block bootstrapped
  multiplicative residuals (preserving within-week correlation) and a per-future level shock.
- **Customer payment timing.** For each customer, days late is modelled with a **Kaplan–Meier** estimator. Paid invoices
  are events; open invoices are right-censored at their current lateness. Ignoring open invoices (the usual approach)
  biases slow payers towards looking punctual. Customer curves are blended with the portfolio curve with weight
  *n / (n + 5)*, an empirical-Bayes-style shrinkage for customers with little history. Overdue invoices are sampled
  **conditionally** on being at least as late as they already are. Invoices more than 60 days overdue carry a 35%
  non-payment probability.
- **Future invoices.** Each customer's ordering frequency and log-normal order size are learned from 26 weeks of
  history, then paid through the same timing model.
- **Bills.** Open bills, recurring obligations, and projected supplier bills. Monthly suppliers are modelled as a learned
  share of the previous month's revenue (purchases follow sales), with amount uncertainty from the spread of that share.
- **Common random numbers.** All random draws are made once at fit time. Evaluating a plan re-runs the simulation with
  the same draws, so the difference between two plans reflects the plans, not simulation noise. One forecast takes
  about 0.1 s.

Outputs: daily P10/P50/P90, probability of going below zero and below the buffer, expected depth below the buffer,
weekly flows by category, and dated driver events.

**Validation** (`analytics/backtest.py`, `python -m dorsal backtest`). Rolling-origin backtest on 16 businesses × 7 origins
× 13 weekly checkpoints. For each origin, Dorsal sees only what was known that day; the forecast is compared with what
actually happened in the account.

| Metric | Result | Target |
|---|---|---|
| Share of actual balances inside the P10–P90 band | **80.5%** | 80% (well calibrated) |
| Error reduction versus naive extrapolation | **62.6%** | > 0 |

### 2.2 Chance-constrained plan optimiser (`planning/planner.py`)

```
minimise    expected $ cost of steps + goodwill penalty
subject to  P(min balance over 13 weeks < buffer) ≤ 10%
            P(min balance over 13 weeks < 0)      ≤ 2%
```

Probabilities are estimated by the Monte Carlo forecast (risk measured from day 3, since actions need time to land).

1. **Levers** generated from the business's own data: reminders on overdue invoices; holds on suspected duplicate bills;
   early-payment discounts on large invoices expected to land after the low point (deadline set before the low point);
   split payments where a supply agreement allows them; deferral of discretionary purchases to the date that leaves
   the most headroom afterwards; financing.
2. **Greedy pass**: repeatedly add the lever with the largest reduction in expected shortfall depth per dollar of cost
   (cost includes a goodwill penalty for steps that strain a relationship), until the targets are met.
3. **Financing**: if targets are still unmet, compute the remaining deficit profile at a strict quantile, size the
   amount from it, time the draw before the first deficit and the repayment at the earliest day it is safe, then test
   each eligible product and keep the cheapest that meets the targets. Temporary gaps are matched with short-term money.
4. **Prune**: drop any costly step the plan no longer needs.
5. **Attribution**: leave-one-out impact for each step (risk without it, shortfall depth it removes).

On the demo business the plan takes the chance of running out of cash from >99% to under 1% for an expected cost of
$314, and runs in about 1.3 s.

### 2.3 Financing matcher (`planning/financing.py`)

Each product (existing credit line, digital working-capital loan, seasonal credit extension, invoice financing,
government-backed term loan, merchant cash advance) is:

1. checked for eligibility with a reason for every "no" (months trading, revenue, debt-service coverage, credit band,
   card sales, invoices);
2. converted into a dated cash-flow schedule (fees, interest, advances, holdbacks, amortisation);
3. priced by **true APR = daily IRR × 365** of that schedule, so a 1.32 factor rate, a 1.6%-per-30-days fee and a
   9.8% interest rate are compared on one scale;
4. checked for timing (funds arrive before they are needed) and affordability (debt-service coverage after borrowing);
5. ranked: usable first, then by total dollar cost. Anything above 36% APR is flagged high-cost and never recommended.

### 2.4 Financial analysis and alerts (`analytics/financials.py`, `analytics/alerts.py`)

Trailing-12-month and quarterly P&L from bank data and the ledger; gross and operating margin; DSO, DPO and the cash
conversion gap; cash days on hand; debt-service coverage; customer concentration (top share, HHI); expense mix; monthly
trend; a 0–100 health score with plain-language reasons. Alerts cover shortfall risk, overdue and aged invoices,
duplicate bills (same vendor and reference, or same amount within 10 days), unusually large bills, payment promises
extracted from known customers' emails, and suspicious emails.

## 3. Security design

Summary here; full tiers and known-risk list in `SECURITY_SELF_ASSESSMENT.md`.

- **Separation of proposing and doing.** The model proposes; deterministic code decides. The policy lives in
  `config/policy.yaml` and only the owner can change the four owner settings, each change audit-logged.
- **Seven permission tiers**: read, notify, protect (automatic); commit (owner approval); borrow and move money
  (approval + PIN, caps); forbidden (never).
- **Approvals bound to payloads.** HMAC-SHA256 tokens over action id, payload hash, expiry (10 min) and a single-use
  nonce. Changing any parameter after approval invalidates it.
- **Defence in depth.** Policy re-checked at execution; sandbox APIs independently refuse unknown recipients,
  unverified payees and over-limit draws; idempotency keys prevent double execution.
- **Untrusted content.** Emails and memos are scanned for injection and fraud signals, sender domains are checked for
  look-alikes, and text is fenced and redacted before reaching the model. Outgoing messages are checked for leaks of
  other customers' data and for unsafe content.
- **Tamper-evident audit.** Every tool call, proposal, decision, approval and execution is hash-chained.
- **Kill switch.** The owner can pause Dorsal; everything except reading is then refused.

## 4. Evaluation

`python -m dorsal evaluate --n 200`: 200 synthetic businesses across five archetypes, half with a cash shock, a quarter
with a planted duplicate bill. Each strategy sees only today's information; the true next 13 weeks are then played out,
including whether customers really accept discounts and respond to reminders.

| Strategy | Ran out of cash | Days overdrawn | Days below buffer | Mean cost |
|---|---|---|---|---|
| No agent | 29.0% | 4.83 | 8.5 | $242 |
| Naive rule (on-time assumption + credit line) | 26.0% | 3.33 | 6.5 | $279 |
| **Dorsal** | **4.0%** | **0.45** | **1.2** | $347 |

Dorsal prevented 86% of the shortfalls that would otherwise happen (naive rule: 10%), caught 50 of 50 planted duplicate
bills, and needed 1.41 owner approvals per business on average. It spends about $105 more per business per quarter than
doing nothing: in effect an insurance premium. The overdraft cost model ($35 per day plus 22% APR) does not count the
harder costs of a missed payroll or a lost supplier.

## 5. Limitations

- All data is synthetic. Behavioural parameters (discount uptake, reminder effect) are assumptions, listed in
  `analytics/forecast.py::Assumptions`; in production they would be learned from outcomes.
- The plan is computed once per review; production would re-plan daily as payments arrive.
- Financing quotes come from a sandbox catalogue; production would call lenders' quote APIs.
- The offline chat answers a fixed set of question types; free-form conversation needs a language model.
