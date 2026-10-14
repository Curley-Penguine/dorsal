"""Untrusted-content handling.

Invoice memos, bill memos and emails are written by third parties, so Dorsal
treats them as data, never as instructions:

1. `scan()` scores text for prompt-injection and payment-fraud signals.
2. `sender_check()` compares a sender's domain with the known contact list
   to catch look-alike (spoofed) domains.
3. `wrap_untrusted()` fences text before it reaches the LLM, with a
   standing instruction that fenced content carries no authority.
4. `redact()` masks account numbers and other identifiers before text is
   sent to an external model or written to logs.

These are the *first* line of defence. The decisive control is the policy
engine: even if a model were fully hijacked, it can only propose actions,
and proposals to pay new payees or change bank details are always denied.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

INJECTION_PATTERNS = [
    (r"ignore (all |your |any )?(previous|prior|above) (instructions|rules|messages)", "instruction_override", 0.6),
    (r"(disregard|bypass|override|skip) (your |the )?(rules|policy|policies|approval|guardrails|limits)", "policy_bypass", 0.6),
    (r"\b(ai|assistant|agent|bot|model)\b.{0,40}\b(processing|reading|handling)\b", "addresses_the_ai", 0.35),
    (r"do not (notify|tell|inform|alert) (the )?(owner|user|anyone)", "secrecy_request", 0.5),
    (r"(new|updated|changed) (bank|account|remittance|payment) (details|account|information|instructions)", "bank_detail_change", 0.45),
    (r"changed banks", "bank_detail_change", 0.45),
    (r"\b(urgent|immediately|right away|asap)\b", "urgency", 0.15),
    (r"\b\d{3,6}[- ]?\d{4,10}\b", "account_number", 0.2),
    (r"(system prompt|developer message|you are now|new instructions)", "role_hijack", 0.5),
]


def scan(text: str) -> dict:
    """Return a risk score in [0, 1] and the matched signals."""
    t = (text or "").lower()
    hits, score = [], 0.0
    for pat, label, w in INJECTION_PATTERNS:
        if re.search(pat, t):
            if label not in hits:
                hits.append(label)
                score = 1 - (1 - score) * (1 - w)
    return {"score": round(score, 3), "signals": hits, "flagged": score >= 0.5}


def _domain(email: str) -> str:
    return email.split("@")[-1].lower().strip()


def sender_check(sender: str, known_emails: set[str]) -> dict:
    """Detect look-alike sender domains (e.g. fernwood-hotel-payments vs fernwoodhotel)."""
    s_dom = _domain(sender)
    known_domains = {_domain(e) for e in known_emails}
    if sender.lower() in known_emails or s_dom in known_domains:
        return {"known": True, "lookalike_of": None}
    best, best_r = None, 0.0
    s_core = re.sub(r"[^a-z]", "", s_dom.split(".")[0])
    for d in known_domains:
        core = re.sub(r"[^a-z]", "", d.split(".")[0])
        r = SequenceMatcher(None, s_core, core).ratio()
        if core and (core in s_core or s_core in core):
            r = max(r, 0.9)
        if r > best_r:
            best, best_r = d, r
    return {"known": False, "lookalike_of": best if best_r >= 0.6 else None, "similarity": round(best_r, 2)}


ACCOUNT_RE = re.compile(r"\b(\d{3,6})([- ]?)(\d{4,10})\b")
EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+)\b")


def redact(text: str) -> str:
    text = ACCOUNT_RE.sub(lambda m: "[ACCOUNT ****" + m.group(3)[-2:] + "]", text or "")
    return text


def wrap_untrusted(text: str, source: str) -> str:
    """Fence third-party text for the LLM. The fence and the system prompt
    tell the model that nothing inside it is an instruction."""
    clean = redact(text).replace("<", "&lt;").replace(">", "&gt;")
    return f'<untrusted_data source="{source}">\n{clean}\n</untrusted_data>'


PROMISE_RE = re.compile(
    r"(?:will|we'll|going to)\s+pay[^.]*?\b(?:on\s+)?(?:(mon|tues|wednes|thurs|fri|satur|sun)day,?\s+)?"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})", re.I)
MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def extract_promise(text: str, year: int) -> tuple[int, int, int] | None:
    """Find a payment promise like 'we will pay on Friday, Oct 16'."""
    m = PROMISE_RE.search(text or "")
    if not m:
        return None
    return (year, MONTHS[m.group(2).lower()[:3]], int(m.group(3)))
