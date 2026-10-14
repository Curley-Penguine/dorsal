"""Dorsal runtime: ties the engine, the policy and the sandbox together.

Flow for every action, no matter who proposed it:

    propose -> policy.evaluate -> auto?    -> execute (re-checked) -> audit
                                -> approve? -> queue -> owner approves (+PIN)
                                               -> signed token -> execute -> audit
                                -> deny?    -> audit, never executed
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

from dorsal.actions import Action
from dorsal.analytics.alerts import build_alerts
from dorsal.analytics.financials import analyze
from dorsal.analytics.forecast import Forecaster, Overlay
from dorsal.models import Business
from dorsal.planning.planner import Targets, known_info_overlay, optimize
from dorsal.sandbox.bank import SandboxError, SandboxWorld
from dorsal.security.approvals import ApprovalError, ApprovalService
from dorsal.security.audit import AuditLog
from dorsal.security.policy import PolicyEngine


class DorsalRuntime:
    def __init__(self, biz: Business, evidence_dir: Optional[Path | str] = None, policy_path=None,
                 policy_overrides: Optional[dict] = None, n_paths: int = 1000, seed: int = 42,
                 approvals: Optional[ApprovalService] = None):
        self.biz = biz
        self.evidence_dir = Path(evidence_dir) if evidence_dir else None
        if self.evidence_dir:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
            for f in ("audit.jsonl", "sandbox_operations.jsonl"):
                (self.evidence_dir / f).unlink(missing_ok=True)
        self.world = SandboxWorld(biz, oplog_path=self.evidence_dir / "sandbox_operations.jsonl" if self.evidence_dir else None)
        self.policy = PolicyEngine(policy_path, policy_overrides)
        self.approvals = approvals or ApprovalService(ttl_seconds=self.policy.cfg["approval"]["token_ttl_seconds"],
                                                      max_pin_attempts=self.policy.cfg["approval"]["max_pin_attempts"])
        self.audit = AuditLog(self.evidence_dir / "audit.jsonl" if self.evidence_dir else None)
        self.n_paths, self.seed = n_paths, seed
        self.pending: dict[str, dict] = {}
        self.records: list[dict] = []
        self._fc: Optional[Forecaster] = None
        self._plan: Optional[dict] = None
        self.audit.append("system", "session.started", {"business": biz.name, "today": biz.today,
                                                        "policy_version": self.policy.cfg["version"]})

    # ---- engine access ------------------------------------------------------------
    @property
    def targets(self) -> Targets:
        s = self.policy.settings
        return Targets(buffer=float(s["buffer"]), max_p_below_buffer=float(s["risk_target_p_below_buffer"]),
                       max_p_shortfall=float(s["risk_target_p_shortfall"]),
                       max_borrow=float(self.policy.cfg["tiers"]["borrow"]["max_amount"]))

    def forecaster(self) -> Forecaster:
        if self._fc is None:
            self._fc = Forecaster(self.biz, n_paths=self.n_paths, seed=self.seed, buffer=self.targets.buffer)
        self._fc.buffer = self.targets.buffer
        return self._fc

    def committed_overlay(self) -> Overlay:
        return known_info_overlay(self.biz).merged(Overlay.from_world(self.world.overlay()))

    def forecast(self, what_if: Optional[Overlay] = None):
        ov = self.committed_overlay()
        if what_if:
            ov = ov.merged(what_if)
        res = self.forecaster().run(ov)
        res.buffer = self.targets.buffer
        return res

    def baseline(self):
        res = self.forecaster().run(known_info_overlay(self.biz))
        res.buffer = self.targets.buffer
        return res

    def analysis(self) -> dict:
        return analyze(self.biz)

    def alerts(self) -> list[dict]:
        return build_alerts(self.biz, self.forecast(), self.analysis())

    def plan(self, refresh: bool = False) -> dict:
        if self._plan is None or refresh:
            self._plan = optimize(self.biz, self.forecaster(), self.analysis(), self.targets,
                                  committed=Overlay.from_world(self.world.overlay()))
        return self._plan

    # ---- action lifecycle -------------------------------------------------------------
    def propose(self, action: Action, actor: str = "agent") -> dict:
        decision = self.policy.evaluate(action, self.biz)
        self.audit.append(actor, "action.proposed", {"action": action.to_dict()})
        self.audit.append("policy", "policy.decision", decision.to_dict())
        rec = {"action": action, "decision": decision, "status": decision.outcome}
        if decision.outcome == "deny":
            rec["status"] = "denied"
            self.records.append(rec)
        elif decision.outcome == "auto":
            self._execute(action, token=None, rec=rec, actor="dorsal")
        else:
            rec["status"] = "pending"
            self.pending[action.id] = rec
        return rec

    def approve(self, action_id: str, pin: Optional[str] = None, approver: str = "owner") -> dict:
        rec = self.pending.get(action_id)
        if rec is None:
            raise ApprovalError("Nothing pending with that id")
        if rec["decision"].step_up and not self.approvals.verify_pin(pin):
            self.audit.append(approver, "approval.pin_failed", {"action_id": action_id})
            raise ApprovalError("PIN is incorrect")
        token = self.approvals.issue(rec["action"], approver)
        self.audit.append(approver, "approval.granted", {"action_id": action_id, "payload_hash": rec["action"].payload_hash(),
                                                         "step_up": rec["decision"].step_up})
        del self.pending[action_id]
        return self._execute(rec["action"], token=token, rec=rec, actor=approver)

    def decline(self, action_id: str, approver: str = "owner") -> dict:
        rec = self.pending.pop(action_id, None)
        if rec is None:
            raise ApprovalError("Nothing pending with that id")
        rec["status"] = "declined"
        self.records.append(rec)
        self.audit.append(approver, "approval.declined", {"action_id": action_id})
        return rec

    def _execute(self, action: Action, token: Optional[str], rec: dict, actor: str) -> dict:
        # Re-check policy at execution time (state may have changed since proposal).
        d = self.policy.evaluate(action, self.biz)
        if d.outcome == "deny":
            rec["status"] = "denied"
            self.audit.append("policy", "execution.blocked", d.to_dict())
            self.records.append(rec)
            return rec
        if d.outcome == "approve":
            if token is None:
                rec["status"] = "blocked"
                self.audit.append("policy", "execution.blocked", {"action_id": action.id, "reason": "approval required"})
                self.records.append(rec)
                return rec
            self.approvals.verify(token, action)
        try:
            result = self._dispatch(action)
        except (SandboxError, KeyError, StopIteration) as e:
            rec["status"] = "failed"
            rec["error"] = str(e)
            self.audit.append("sandbox", "execution.failed", {"action_id": action.id, "error": str(e)})
            self.records.append(rec)
            return rec
        self.policy.record_execution(action)
        rec["status"] = "executed"
        rec["result"] = result
        rec["executed_by"] = actor
        self.records.append(rec)
        self.audit.append(actor, "action.executed", {"action_id": action.id, "type": action.type,
                                                      "payload_hash": action.payload_hash(), "result": result})
        return rec

    def _email(self, action: Action) -> Optional[dict]:
        if not action.draft or not action.counterparty or "@" not in action.counterparty:
            return None
        lines = action.draft.split("\n")
        subject = lines[0].replace("Subject:", "").strip() if lines[0].startswith("Subject:") else action.title
        body = "\n".join(lines[1:]).strip()
        return self.world.send_email(action.counterparty, subject, body, key=action.id + ":email")

    def _dispatch(self, a: Action) -> dict:
        w, p = self.world, a.params
        out = {}
        if a.type == "send_reminder":
            for iid in p["invoice_ids"]:
                w.mark_reminded(iid, key=f"{a.id}:{iid}")
            out["email"] = self._email(a)
        elif a.type == "hold_bill":
            out["hold"] = w.hold_bill(p["bill_id"], key=a.id)
            out["email"] = self._email(a)
        elif a.type == "offer_discount":
            dl = p["deadline"] if isinstance(p["deadline"], date) else date.fromisoformat(p["deadline"])
            out["offer"] = w.create_discount_offer(p["invoice_id"], p["pct"], dl, key=a.id)
            out["email"] = self._email(a)
        elif a.type == "request_split":
            pays = [(d if isinstance(d, date) else date.fromisoformat(d), amt) for d, amt in p["payments"]]
            out["schedule"] = w.reschedule_bill(p["bill_id"], pays, key=a.id)
            out["email"] = self._email(a)
        elif a.type == "reschedule_bill":
            nd = p["new_date"] if isinstance(p["new_date"], date) else date.fromisoformat(p["new_date"])
            out["schedule"] = w.reschedule_bill(p["bill_id"], [(nd, self.biz.bill(p["bill_id"]).amount)], key=a.id)
            out["email"] = self._email(a)
        elif a.type == "draw_financing":
            flows = [(d if isinstance(d, date) else date.fromisoformat(d), amt) for d, amt in p["flows"]]
            out["draw"] = w.draw_financing(p["product_id"], p["amount"], flows, key=a.id)
        elif a.type == "pay_bill":
            out["payment"] = w.pay(p["payee_id"], p["amount"], key=a.id)
            self.biz.bill(p["bill_id"]).paid_date = self.biz.today
        elif a.type == "notify_owner":
            out["notified"] = True
        else:
            raise SandboxError(f"No executor for {a.type}")
        return out

    # ---- owner settings -------------------------------------------------------------
    def update_settings(self, changes: dict, actor: str = "owner") -> dict:
        before = self.policy.settings
        after = self.policy.update_owner_settings(changes)
        self.audit.append(actor, "policy.settings_changed", {"before": {k: before[k] for k in changes},
                                                              "after": {k: after[k] for k in changes}})
        # newly auto-eligible pending actions run now, under the new settings
        for aid, rec in list(self.pending.items()):
            d = self.policy.evaluate(rec["action"], self.biz)
            if d.outcome == "auto":
                del self.pending[aid]
                rec["decision"] = d
                self._execute(rec["action"], None, rec, actor="dorsal")
        return after
