import copy

from dorsal.redteam import run_redteam
from dorsal.security.audit import AuditLog


def test_all_red_team_attacks_are_stopped():
    results = run_redteam()
    failed = [r for r in results if not r["passed"]]
    assert not failed, failed
    assert len(results) >= 20


def test_audit_chain_file_roundtrip(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    for i in range(5):
        log.append("t", "e", {"i": i, "note": "acct 0042-118833"})
    assert AuditLog.verify_file(tmp_path / "a.jsonl")["ok"]
    assert "118833" not in (tmp_path / "a.jsonl").read_text()
    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    lines.pop(2)  # delete an entry
    (tmp_path / "a.jsonl").write_text("\n".join(lines) + "\n")
    assert not AuditLog.verify_file(tmp_path / "a.jsonl")["ok"]


def test_approval_tokens_are_bound_to_payload(rt):
    from dorsal.security.approvals import ApprovalError
    a = next(x for x in rt.plan()["actions"] if x.type == "offer_discount")
    tok = rt.approvals.issue(a)
    b = copy.deepcopy(a)
    b.params["deadline"] = b.params["deadline"].replace(day=1)
    try:
        rt.approvals.verify(tok, b)
        assert False, "tampered action accepted"
    except ApprovalError:
        pass
