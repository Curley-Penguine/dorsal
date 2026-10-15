# Engineering notes: what broke and how it was fixed

Real problems found while building and testing Dorsal, in roughly the order they were found. Each one is something a
test, a backtest or a review caught; none of them reached the final submission.

## 1. Demo dates landed in the wrong year

**Symptom.** In the first web prototype, a September invoice appeared as due in September of the *following* year, so
it looked 11 months away instead of overdue, and the demo's numbers were quietly wrong.

**Cause.** The helper that turns "month, day" into a date assumed any month before October belonged to next year
(`month >= 10 ? 2026 : 2027`).

**Fix.** Months from June onward now map to 2026 and earlier months to 2027, which matches the demo's 13-week window.
Caught by checking that the overdue invoices in the demo really showed as overdue.

## 2. The first forecast was overconfident

**Symptom.** The backtest showed that the forecast's "80% range" contained only about 60–65% of real outcomes for some
business types (distributors and agencies), and the median forecast was biased upwards by tens of thousands of dollars.

**Cause, found by comparing forecast and actual cash flows by category.** Two things:
- Future supplier bills were projected as a flat repeat of recent bills. For businesses whose purchases follow sales,
  this badly underestimated busy months.
- Customer lateness was learned only from invoices that had already been paid. Invoices still unpaid (the slowest
  ones) were ignored, so slow payers looked punctual.

**Fix.** Supplier bills are now modelled as a learned share of the previous month's revenue, with their own
uncertainty. Payment timing now uses a Kaplan–Meier estimator that counts still-unpaid invoices as censored
observations. Backtest coverage rose to **80.5%** (target 80%) across 16 businesses, with **63% lower error** than a
naive baseline.

## 3. The financing maths crashed on some businesses

**Symptom.** The 200-business evaluation stopped with `OverflowError: Numerical result out of range`.

**Cause.** For a business with very small card sales, a merchant cash advance's repayment schedule (a fixed share of
daily sales) stretched over thousands of days, and the true-APR calculation raised numbers to powers that overflowed.

**Fix.** The APR solver now works in log space and returns "unknown" for schedules longer than about 11 years, and the
matcher marks such advances as unrepayable ("card sales are too low to repay this advance") instead of pricing them.

## 4. A business with no debt was refused every loan

**Symptom.** In the evaluation, an agency with no existing loans could not borrow at all and ran out of cash.

**Cause.** Several products require a minimum debt service coverage ratio. With no debt, that ratio is undefined
(`None`), and the eligibility check treated "undefined" as "fails".

**Fix.** "No existing debt" now passes the coverage check; a separate rule requires positive operating profit for
term and revolving loans. Partial coverage ("covers only $X of the $Y needed") is reported instead of being treated
as ineligibility.

## 5. The optimizer chose a three-year loan for a three-week gap

**Symptom.** For a short, temporary shortfall, the plan picked a government-backed term loan costing about $3,400 in
interest, when a revolving line would have cost about $100.

**Cause.** Financing was sized and timed against the median forecast, and products were compared on cost over very
different repayment periods.

**Fix.** Financing is now sized from the remaining deficit at a strict quantile, drawn just before the first deficit
and repaid at the earliest safe date. Temporary gaps are matched only with short-term money.

## 6. The plan looked impossible because of today's balance

**Symptom.** Some plans were reported as infeasible even though every future shortfall was covered.

**Cause.** Risk was measured from day 0, when no action can have taken effect yet.

**Fix.** Risk is measured from day 3 onward (`risk_from`), which reflects the time a reminder or payment change needs
to land.

## 7. Approving one step reshuffled the whole plan

**Symptom.** In the web console, approving one step caused the remaining steps to change, because the plan was
re-optimised after every approval.

**Fix.** The plan is computed once per review and stays stable while the owner works through approvals. The forecast
still updates to show the effect of each approval.

## 8. A test assumed every business has card sales

**Symptom.** The new sandbox test failed for the design agency: it expected more than 300 bank transactions a year.

**Cause.** The test was wrong, not the code. Agencies invoice clients and don't take daily card payments, so they have
far fewer bank lines.

**Fix.** The test now checks for a full year of history instead of a transaction count.

## 9. The automatic checks failed on the first pull request

**Symptom.** Pull request #1 showed a red ✗ on GitHub even though every test passed locally.

**Cause.** Tests were being run locally with `python -m pytest`, which adds the project folder to Python's search path;
GitHub Actions runs plain `pytest`, which does not, so `import dorsal` failed and the run stopped with exit code 4.

**Fix.** `pythonpath = ["."]` in the pytest configuration makes both behave the same. The fix was pushed to the same
branch and the pull request was merged only after the checks turned green.

## 10. The web console scrolled sideways on phones

**Symptom.** At phone width the page was 640px wide instead of 390px.

**Cause.** Grid panels let wide content (the KPI tiles and tabs) set their minimum width.

**Fix.** Panels use `grid-template-columns: minmax(0, 1fr)` and their children `min-width: 0`, so content wraps or
scrolls inside its own container.
