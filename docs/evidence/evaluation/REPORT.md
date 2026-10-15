# Dorsal evaluation on synthetic businesses

Businesses: **200** (5 archetypes; 50% with a cash shock; 25% with a planted duplicate bill).
Without help, **58** of them run out of cash within 13 weeks.

| Strategy | Ran out of cash | Days overdrawn (mean) | Days below buffer (mean) | Total cost (mean) |
|---|---|---|---|---|
| No agent | 29.0% | 4.83 | 8.5 | $242 |
| Naive rule (on-time assumption + credit line) | 26.0% | 3.33 | 6.5 | $279 |
| Dorsal | 4.0% | 0.45 | 1.2 | $347 |

- Of the businesses that would have run short, Dorsal prevented **86%** (naive rule: 10%).
- On businesses that never needed help, Dorsal spent $118.58 on average (cost-incurring steps on 30% of them).
- Duplicate bills caught: 50 of 50.
- Owner approvals needed per business: 1.41 on average.
- Median planning time: 0.2 s per business.

Total cost = discounts given + financing interest and fees + overdraft interest and fees.
All businesses are synthetic; behavioural parameters (discount uptake, reminder effect) are drawn from ranges, not taken from real customer data.
