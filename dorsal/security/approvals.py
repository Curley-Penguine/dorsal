"""Signed, single-use approval tokens and owner step-up verification.

An approval token is an HMAC-SHA256 signature over (action id, payload hash,
expiry, nonce). The executor re-computes the action's payload hash at
execution time, so an approval for "offer 2% on INV-1941" cannot be replayed
for "offer 20% on INV-1941", reused twice, or used after it expires.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time

from dorsal.actions import Action


class ApprovalError(Exception):
    pass


class ApprovalService:
    def __init__(self, secret: bytes | None = None, ttl_seconds: int = 600, owner_pin: str | None = None,
                 max_pin_attempts: int = 5):
        self.secret = secret or os.environ.get("DORSAL_APPROVAL_SECRET", "").encode() or secrets.token_bytes(32)
        self.ttl = ttl_seconds
        pin = owner_pin or os.environ.get("DORSAL_OWNER_PIN", "2468")
        self._pin_hash = hashlib.sha256(pin.encode()).digest()
        self.max_attempts = max_pin_attempts
        self.failed_attempts = 0
        self.used_nonces: set[str] = set()

    # ---- step-up -----------------------------------------------------------------
    def verify_pin(self, pin: str | None) -> bool:
        if self.failed_attempts >= self.max_attempts:
            raise ApprovalError("Too many wrong PIN attempts. Approvals are locked; restart the session.")
        ok = pin is not None and hmac.compare_digest(hashlib.sha256(pin.encode()).digest(), self._pin_hash)
        if not ok:
            self.failed_attempts += 1
        else:
            self.failed_attempts = 0
        return ok

    # ---- tokens ------------------------------------------------------------------------
    def _sign(self, body: bytes) -> str:
        return hmac.new(self.secret, body, hashlib.sha256).hexdigest()

    def issue(self, action: Action, approver: str = "owner", now: float | None = None) -> str:
        payload = {"aid": action.id, "h": action.payload_hash(), "exp": int((now or time.time()) + self.ttl),
                   "n": secrets.token_hex(8), "by": approver}
        body = base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True).encode())
        return body.decode() + "." + self._sign(body)

    def verify(self, token: str, action: Action, now: float | None = None) -> dict:
        try:
            body_s, sig = token.rsplit(".", 1)
        except ValueError:
            raise ApprovalError("Malformed approval token")
        body = body_s.encode()
        if not hmac.compare_digest(self._sign(body), sig):
            raise ApprovalError("Approval signature is invalid")
        payload = json.loads(base64.urlsafe_b64decode(body))
        if payload["aid"] != action.id:
            raise ApprovalError("Approval is for a different action")
        if payload["h"] != action.payload_hash():
            raise ApprovalError("Action was changed after it was approved")
        if (now or time.time()) > payload["exp"]:
            raise ApprovalError("Approval has expired")
        if payload["n"] in self.used_nonces:
            raise ApprovalError("Approval has already been used")
        self.used_nonces.add(payload["n"])
        return payload
