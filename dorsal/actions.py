"""The single action format shared by the planner, the agent, the policy
engine and the executor.

An Action is a *proposal*. Nothing happens until the policy engine has
classified it and, where required, the owner has approved it. The payload
hash binds an approval to the exact parameters that were approved.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from dorsal.analytics.forecast import Overlay

# action type -> permission tier name (see dorsal.security.policy)
ACTION_TIERS = {
    "send_reminder": "notify",
    "hold_bill": "protect",
    "notify_owner": "notify",
    "offer_discount": "commit",
    "request_split": "commit",
    "reschedule_bill": "commit",
    "draw_financing": "borrow",
    "pay_bill": "move_money",
    "pay_new_payee": "forbidden",
    "change_payee_details": "forbidden",
    "change_policy": "forbidden",
}


def _canon(o):
    if isinstance(o, date):
        return o.isoformat()
    if isinstance(o, (set, tuple)):
        return sorted(o) if isinstance(o, set) else list(o)
    raise TypeError(type(o))


@dataclass
class Action:
    type: str
    params: dict
    title: str = ""
    rationale: str = ""
    cost: float = 0.0                     # expected cost in dollars
    counterparty: Optional[str] = None    # email address or payee id
    relationship_cost: float = 0.0        # 0..1 soft cost to customer/supplier goodwill
    overlay: Overlay = field(default_factory=Overlay)
    draft: str = ""                       # message Dorsal would send, if any
    proposed_by: str = "planner"          # planner | llm | user | external
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = f"{self.type}:{self.payload_hash()[:10]}"

    @property
    def tier(self) -> str:
        return ACTION_TIERS.get(self.type, "forbidden")

    def payload(self) -> dict:
        return {"type": self.type, "params": self.params, "counterparty": self.counterparty, "draft": self.draft}

    def payload_hash(self) -> str:
        blob = json.dumps(self.payload(), sort_keys=True, default=_canon, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "tier": self.tier, "title": self.title,
                "rationale": self.rationale, "cost": round(self.cost, 2), "counterparty": self.counterparty,
                "params": json.loads(json.dumps(self.params, default=_canon)), "draft": self.draft,
                "proposed_by": self.proposed_by, "payload_hash": self.payload_hash()}
