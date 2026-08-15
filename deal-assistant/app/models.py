"""
Data models and loaders for the Deal Assistant.

Everything is read from the small JSON seed files in data/ — no external
integrations. Keeping these as plain Pydantic models (rather than passing
raw dicts around) means every tool has a typed, validated contract, which
matters later for the "never invent a price/discount/deal" guardrail: if a
field isn't in the schema, the agent can't have gotten it from anywhere
except the model making it up.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SourceType = Literal["offer", "coupon", "cashback", "card_reward"]


class Deal(BaseModel):
    id: str
    brand: str
    product: str
    category: str
    source_type: SourceType
    price: float
    currency: str = "INR"
    discount_type: Optional[str] = None  # percent | flat | cashback_percent | cashback_flat
    discount_value: float = 0.0
    min_purchase: float = 0.0
    max_discount: float = 0.0
    card: Optional[str] = None  # only set when source_type == "card_reward"
    description: str = ""
    valid_until: Optional[str] = None

    def effective_price(self, amount: Optional[float] = None) -> float:
        """
        Price after applying this deal's own discount/cashback, ignoring
        card rewards (those are computed separately via reward_rules,
        since they depend on which card the user pays with).
        """
        base = amount if amount is not None else self.price
        if self.source_type == "card_reward" or self.discount_type is None:
            return base

        if self.discount_type == "percent":
            raw = base * (self.discount_value / 100)
        elif self.discount_type == "flat":
            raw = self.discount_value
        elif self.discount_type == "cashback_percent":
            raw = base * (self.discount_value / 100)
        elif self.discount_type == "cashback_flat":
            raw = self.discount_value
        else:
            raw = 0.0

        if self.max_discount:
            raw = min(raw, self.max_discount)

        if base < self.min_purchase:
            return base  # doesn't qualify
        return round(base - raw, 2)


class RewardRule(BaseModel):
    card: str
    base_rate: float
    category_multipliers: dict[str, float]
    monthly_cap: dict[str, float]
    notes: str = ""

    def reward_for(self, amount: float, category: str) -> dict:
        """
        Compute reward earned paying `amount` in `category` on this card,
        honouring the category multiplier and the monthly cap. This is the
        core of "correct reward math" — every number returned is traceable
        to base_rate / multiplier / cap fields in reward_rules.json.
        """
        multiplier = self.category_multipliers.get(category, self.category_multipliers.get("other", 1))
        cap = self.monthly_cap.get(category, self.monthly_cap.get("other", 0))
        raw_reward = amount * self.base_rate * multiplier
        capped = cap > 0 and raw_reward > cap
        reward = min(raw_reward, cap) if cap > 0 else raw_reward
        effective_price = round(amount - reward, 2)
        return {
            "card": self.card,
            "category": category,
            "amount": amount,
            "multiplier": multiplier,
            "raw_reward": round(raw_reward, 2),
            "cap_applied": capped,
            "monthly_cap": cap,
            "reward": round(reward, 2),
            "effective_price": effective_price,
        }


@lru_cache
def load_deals() -> list[Deal]:
    with open(DATA_DIR / "deals.json", encoding="utf-8") as f:
        raw = json.load(f)
    return [Deal(**d) for d in raw]


@lru_cache
def load_reward_rules() -> dict[str, RewardRule]:
    with open(DATA_DIR / "reward_rules.json", encoding="utf-8") as f:
        raw = json.load(f)
    return {card: RewardRule(card=card, **rule) for card, rule in raw.items()}
