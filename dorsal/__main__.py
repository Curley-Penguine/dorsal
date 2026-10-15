"""Command line interface.

    python -m dorsal demo        run the morning review on the demo business and save evidence
    python -m dorsal serve       start the web console on http://127.0.0.1:8000
    python -m dorsal redteam     run the 20 attack scenarios against the policy engine
    python -m dorsal backtest    rolling-origin accuracy and calibration of the forecast
    python -m dorsal evaluate    compare Dorsal with baselines on many synthetic businesses
    python -m dorsal verify-audit PATH   check an audit log's hash chain
    python -m dorsal export-demo write the static demo bundle used by the hosted page
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_demo(args):
    from dorsal.agent.agent import DorsalAgent
    from dorsal.agent.providers import get_provider
    from dorsal.runtime import DorsalRuntime
    from dorsal.sandbox.generator import harbor_lane
    out = Path(args.out)
    biz, _ = harbor_lane()
    rt = DorsalRuntime(biz, evidence_dir=out)
    agent = DorsalAgent(rt, get_provider(args.llm))
    review = agent.morning_review()
    print("=" * 72 + "\nMORNING REVIEW (" + review["provider"] + ")\n" + "=" * 72)
    print(review["text"])
    if args.approve_all:
        print("\nOwner approves every pending step (PIN entered for borrowing)...")
        for aid, rec in list(rt.pending.items()):
            rt.approve(aid, pin=args.pin if rec["decision"].step_up else None)
            print("  approved:", rec["action"].title)
    s = rt.forecast().summary()
    print(f"\nNow: P(shortfall)={s['p_shortfall']:.1%}  P(below buffer)={rt.forecast().p_below(rt.targets.buffer):.1%}  "
          f"typical low ${s['median_low']:,.0f} on {s['median_low_date']}")
    print("Audit chain:", rt.audit.verify())
    (out / "morning_review.json").write_text(json.dumps(review, indent=2, default=str))
    print(f"\nEvidence written to {out}/ (audit.jsonl, sandbox_operations.jsonl, morning_review.json)")


def cmd_serve(args):
    import uvicorn
    uvicorn.run("dorsal.api:app", host=args.host, port=args.port, reload=False)


def cmd_redteam(args):
    from dorsal.redteam import run_redteam
    res = run_redteam()
    for r in res:
        print(("PASS " if r["passed"] else "FAIL ") + r["id"] + "  " + r["attack"])
        print("      expected: " + r["expected"])
        print("      observed: " + r["observed"][:160])
    n = sum(r["passed"] for r in res)
    print(f"\n{n}/{len(res)} attacks stopped")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
    sys.exit(0 if n == len(res) else 1)


def cmd_backtest(args):
    from dorsal.analytics.backtest import backtest
    from dorsal.sandbox.generator import ARCHETYPES, generate, harbor_lane
    rows = []
    biz, _ = harbor_lane()
    r = backtest(biz)
    rows.append({"business": "Harbor Lane (demo)", **{k: v for k, v in r.items() if k != "rows"}})
    for a in ARCHETYPES:
        for s in range(args.seeds):
            biz, _ = generate(a, seed=100 + s)
            r = backtest(biz, n_paths=300)
            rows.append({"business": f"{a} #{s}", **{k: v for k, v in r.items() if k != "rows"}})
    for row in rows:
        print(f"{row['business']:<24} coverage(P10-P90)={row['coverage_p10_p90']:.0%}  skill vs naive={row['skill_vs_naive']:+.0%}")
    import numpy as np
    cov = np.mean([r["coverage_p10_p90"] for r in rows])
    skill = np.mean([r["skill_vs_naive"] for r in rows])
    print(f"\nMean coverage {cov:.1%} (target 80%), mean skill vs naive {skill:+.1%}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"rows": rows, "mean_coverage": cov, "mean_skill": skill}, indent=2, default=float))


def cmd_evaluate(args):
    from dorsal.evaluate import evaluate
    evaluate(n=args.n, out=args.out, n_paths=args.paths)


def cmd_verify(args):
    from dorsal.security.audit import AuditLog
    v = AuditLog.verify_file(args.path)
    print(json.dumps(v, indent=2))
    sys.exit(0 if v["ok"] else 1)


def cmd_export(args):
    from dorsal.export import export_demo
    export_demo(Path(args.out))


def main(argv=None):
    p = argparse.ArgumentParser(prog="dorsal", description="Dorsal: an AI cash-flow copilot for small businesses")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--out", default="docs/evidence/demo_run")
    d.add_argument("--llm", default=None, help="offline | anthropic | openai (default: $DORSAL_LLM or offline)")
    d.add_argument("--approve-all", action="store_true")
    d.add_argument("--pin", default="2468")
    d.set_defaults(fn=cmd_demo)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)
    r = sub.add_parser("redteam")
    r.add_argument("--out", default=None)
    r.set_defaults(fn=cmd_redteam)
    b = sub.add_parser("backtest")
    b.add_argument("--seeds", type=int, default=3)
    b.add_argument("--out", default=None)
    b.set_defaults(fn=cmd_backtest)
    e = sub.add_parser("evaluate")
    e.add_argument("--n", type=int, default=200)
    e.add_argument("--paths", type=int, default=400)
    e.add_argument("--out", default="docs/evidence/evaluation")
    e.set_defaults(fn=cmd_evaluate)
    v = sub.add_parser("verify-audit")
    v.add_argument("path")
    v.set_defaults(fn=cmd_verify)
    x = sub.add_parser("export-demo")
    x.add_argument("--out", default="docs/demo_bundle.json")
    x.set_defaults(fn=cmd_export)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
