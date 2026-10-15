# Dorsal: an AI cash-flow copilot for small businesses

**Team Dorsal** · 2026 Shenzhen International FinTech Competition (FinTechathon), International Track · Topic E: SME Finance Copilot
Theme: *AI as a Financial Participant.*

> **Why "Dorsal"?** A shark's dorsal fin is what keeps an apex predator stable at full speed: it stops the shark
> rolling over when it turns hard. Dorsal does the same for a small business's cash, so the owner can move fast
> without capsizing.

Small businesses rarely fail because they are unprofitable. They fail because cash runs out between paying suppliers and
getting paid by customers. Dorsal watches a business's bank account, invoices and bills, forecasts the next 13 weeks as a
range of possible futures, and then **acts**: it chases late payers, catches duplicate bills, negotiates payment timing
and arranges the cheapest suitable financing. It acts only inside permission tiers the owner controls, and those tiers
are enforced by deterministic code the AI cannot change.

| | |
|---|---|
| **Forecast** | Monte Carlo, 1,000 futures; per-customer payment timing learned with a Kaplan–Meier survival model (handles unpaid invoices); backtested to **80.5% coverage of the 80% band** and **63% lower error than a naive forecast** |
| **Plan** | Chance-constrained optimiser: cheapest set of steps that keeps P(balance < buffer) ≤ 10% |
| **Financing matching** | 6 product types priced by true APR from each product's cash flows (IRR); eligibility, timing and affordability checks; products above 36% APR are never recommended |
| **Financial analysis** | Margins, DSO/DPO, debt-service coverage, concentration, health score with reason codes |
| **Alerts** | Shortfall risk, overdue and aged invoices, duplicate and unusual bills, payment promises, spoofed and prompt-injected emails |
| **Security** | 7 permission tiers, signed single-use approvals bound to the exact action, PIN step-up, hash-chained audit log, injection fencing, kill switch; **20/20 red-team attacks stopped** |
| **Evaluation** | On 200 synthetic businesses, Dorsal cut the share that ran out of cash from **29% to 4%** (prevented 86% of shortfalls; a naive rule prevented 10%) |

## Development

Dorsal was integrated module by module through pull requests with automatic test checks. Problems found
along the way, and how they were fixed, are recorded in [`docs/ENGINEERING_NOTES.md`](docs/ENGINEERING_NOTES.md).

All businesses, people and lenders in this repository are fictional. Bank, email and lender connections are sandboxed.

---

## Quick start (5 minutes, no API key needed)

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Curley-Penguine/dorsal.git
cd dorsal
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m dorsal demo --approve-all     # run the agent end to end and write evidence logs
python -m dorsal serve                  # open http://127.0.0.1:8000 for the web console
```

The sandbox owner PIN is `2468`.

### Or with Docker

```bash
cp .env.example .env
docker compose up --build             # then open http://localhost:8000
```

### Using a real language model (optional)

Dorsal runs fully offline by default: a deterministic playbook drives the same tools, so every result is reproducible.
To let a language model drive the agent instead:

```bash
pip install anthropic openai
export DORSAL_LLM=anthropic  ANTHROPIC_API_KEY=...                      # Claude
# or any OpenAI-compatible endpoint, e.g. DeepSeek or Qwen:
export DORSAL_LLM=openai OPENAI_API_KEY=... DORSAL_OPENAI_BASE_URL=https://api.deepseek.com DORSAL_OPENAI_MODEL=deepseek-chat
python -m dorsal serve
```

The model only ever *proposes*. The policy engine decides what happens, whichever model is used.

---

## Commands

| Command | What it does |
|---|---|
| `python -m dorsal demo [--approve-all]` | Morning review on the demo business; writes `audit.jsonl`, `sandbox_operations.jsonl`, `morning_review.json` |
| `python -m dorsal serve` | Web console and HTTP API |
| `python -m dorsal redteam` | 20 attack scenarios against the policy engine (exit code 1 if any succeeds) |
| `python -m dorsal backtest` | Rolling-origin accuracy and calibration of the forecast on 16 businesses |
| `python -m dorsal evaluate --n 200` | Dorsal versus "no agent" and a naive rule on 200 synthetic businesses |
| `python -m dorsal verify-audit PATH` | Re-computes an audit log's hash chain |
| `python -m dorsal export-demo` | Builds the static bundle behind the hosted demo page |
| `pytest` | 56 automated tests |

---

## How it works

```
 bank history, invoices, bills, inbox            owner (web console)
             │                                        │ approve / decline / PIN / settings
             ▼                                        ▼
 ┌──────────────────────┐   tools (read)   ┌────────────────────┐
 │  Analytics engine    │◄─────────────────│  Agent (LLM or     │
 │  forecast · analysis │                  │  offline playbook) │
 │  alerts · financing  │─────────────────►│                    │
 │  plan optimiser      │   data, plans    └─────────┬──────────┘
 └──────────────────────┘                            │ propose only
                                                     ▼
                                       ┌──────────────────────────┐
                                       │  Policy engine (YAML,    │  deny ─► audit
                                       │  deterministic code)     │
                                       └───────┬──────────┬───────┘
                                          auto │          │ approve ─► owner (+PIN) ─► signed token
                                               ▼          ▼
                                       ┌──────────────────────────┐
                                       │  Executor: re-checks     │──► sandbox bank, ledger,
                                       │  policy and token        │    email, lenders
                                       └──────────┬───────────────┘
                                                  ▼
                                       hash-chained audit log
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the algorithms,
[`docs/ENGINEERING_NOTES.md`](docs/ENGINEERING_NOTES.md) for what broke during development and how it was fixed, and
[`docs/SECURITY_SELF_ASSESSMENT.md`](docs/SECURITY_SELF_ASSESSMENT.md) for the permission tiers and known risks.

## Repository layout

```
dorsal/
  sandbox/      synthetic SME generator, sandbox bank/email, lender catalogue
  analytics/    forecast (Monte Carlo + Kaplan-Meier), backtest, financial analysis, alerts
  planning/     plan optimiser, financing matcher (IRR-based APR)
  security/     policy engine, approvals, audit chain, injection guard
  agent/        tools, LLM providers (offline / Anthropic / OpenAI-compatible), agent loop
  runtime.py    propose → policy → approve → execute pipeline
  api.py        FastAPI server; web/index.html is the console
  redteam.py    20 attack scenarios
  evaluate.py   200-business evaluation harness
config/policy.yaml   the permission policy
tests/               pytest suite
docs/evidence/       execution evidence (logs, red-team, backtest, evaluation)
```

## Execution evidence

`docs/evidence/` contains logs produced by the commands above:

- `demo_run/audit.jsonl`: every proposal, policy decision, approval and execution, hash-chained
- `demo_run/sandbox_operations.jsonl`: every operation against the sandbox bank, email and lenders
- `redteam_results.json`: 20 attacks, expected versus observed outcome
- `backtest.json`: forecast calibration and accuracy
- `evaluation/REPORT.md` and `results.json`: 200-business comparison

## Licence and third-party components

Original code by Team Dorsal. Uses NumPy (BSD), PyYAML (MIT), FastAPI (MIT), Uvicorn (BSD), Pydantic (MIT) and,
optionally, the Anthropic and OpenAI Python SDKs (MIT/Apache-2.0). Fonts in the web console are from Google Fonts
(SIL Open Font License).
