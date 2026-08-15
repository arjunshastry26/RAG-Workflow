"""
Tools the planning agent can call. Each returns a small structured dict
(not free text) so the model's answer can be grounded in — and the eval
harness / guardrails can check citations against — exactly what was
retrieved or computed. Nothing here calls an LLM.
"""
from __future__ import annotations

from langchain_core.tools import tool

from app.guardrails import sanitize_text
from app.models import load_reward_rules
from app.retriever import get_retriever


def _deal_to_dict(deal, score: float | None = None) -> dict:
    d = {
        "id": deal.id,
        "brand": deal.brand,
        "product": deal.product,
        "category": deal.category,
        "source_type": deal.source_type,
        "price": deal.price,
        "currency": deal.currency,
        "card": deal.card,
        "description": sanitize_text(deal.description),
        "effective_price": deal.effective_price(),
    }
    if score is not None:
        d["relevance_score"] = round(float(score), 3)
    return d


@tool
def search_deals(brand: str) -> dict:
    """Look up every deal in the dataset from a given brand (e.g. 'Amazon',
    'BigBasket', 'IndiGo', 'Netflix'). Returns matching deal records with id,
    product, category, source type, price, and effective price so the answer
    can cite real records. Returns status 'no_reliable_deal_found' if nothing
    matches with enough confidence -- treat that as a hard stop, not a cue to
    guess."""
    retriever = get_retriever()
    results, weak = retriever.search(brand, top_k=10, brand=brand)
    if not results:
        results, weak = retriever.search(brand, top_k=10)
    if weak:
        return {"status": "no_reliable_deal_found", "query": brand}
    return {"status": "ok", "results": [_deal_to_dict(d, s) for d, s in results]}


@tool
def compare_prices(product: str) -> dict:
    """Compare all available deals for a given product (e.g. 'Netflix Premium
    subscription', 'IndiGo flight Bangalore to Delhi') across offers, coupons,
    and cashback sources. Returns deals sorted by effective price after each
    deal's own discount (not counting card rewards -- use best_card for
    that), cheapest first. Returns status 'no_reliable_deal_found' if
    retrieval is weak."""
    retriever = get_retriever()
    results, weak = retriever.search(product, top_k=10)
    if weak:
        return {"status": "no_reliable_deal_found", "query": product}
    ranked = sorted(results, key=lambda x: x[0].effective_price())
    return {"status": "ok", "results": [_deal_to_dict(d, s) for d, s in ranked]}


@tool
def best_card(amount: float, category: str = "other") -> dict:
    """Given a purchase amount in INR and its category (one of 'groceries',
    'subscription', 'flights', 'electronics', or 'other' if unknown), return
    every card's reward for that purchase -- honouring each card's category
    multiplier and monthly cashback cap -- ranked best first. Always pass the
    real category when you know it; omitting it falls back to the 'other'
    rate, which understates category-boosted cards."""
    rules = load_reward_rules()
    ranked = sorted(
        (rule.reward_for(amount, category) for rule in rules.values()),
        key=lambda r: -r["reward"],
    )
    return {"status": "ok", "category_used": category, "ranked_cards": ranked}


@tool
def get_reward_rules(card: str) -> dict:
    """Look up the reward rules (base rate, category multipliers, monthly
    caps) for a named credit card, e.g. 'HDFC Millennia Card'. Returns
    status 'no_reliable_deal_found' if no card matches."""
    rules = load_reward_rules()
    match = next((r for name, r in rules.items() if card.lower() in name.lower()), None)
    if match is None:
        return {"status": "no_reliable_deal_found", "query": card}
    return {
        "status": "ok",
        "card": match.card,
        "base_rate": match.base_rate,
        "category_multipliers": match.category_multipliers,
        "monthly_cap": match.monthly_cap,
        "notes": match.notes,
    }


@tool
def price_drop_watch(product: str, target_price: float) -> dict:
    """Bonus tool: check whether any current deal for a product has an
    effective price at or below a target price the user is watching for
    (e.g. 'tell me if the flight drops below 5000'). Returns the cheapest
    matching deal and whether it has hit the target."""
    retriever = get_retriever()
    results, weak = retriever.search(product, top_k=10)
    if weak:
        return {"status": "no_reliable_deal_found", "query": product}
    cheapest_deal, score = min(results, key=lambda x: x[0].effective_price())
    hit = cheapest_deal.effective_price() <= target_price
    return {
        "status": "ok",
        "target_price": target_price,
        "cheapest_deal": _deal_to_dict(cheapest_deal, score),
        "target_hit": hit,
        "difference": round(cheapest_deal.effective_price() - target_price, 2),
    }


ALL_TOOLS = [search_deals, compare_prices, best_card, get_reward_rules, price_drop_watch]
