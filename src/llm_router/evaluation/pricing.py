"""Model-specific cost estimation from registry pricing metadata.

Units:
- Token counts are non-negative integers.
- Prices are `Decimal` amounts in the pricing currency per million tokens.
- `TokenUsage.input_tokens` counts *uncached* input only (the Claude API `usage.input_tokens`
  semantics). Cache reads and cache writes are counted separately, so every token is priced
  exactly once.

Estimates (`CostEstimate`) are never mixed with provider-reported charges (`BilledCost`).
A missing price yields `UnknownCost` rather than a partial or zero total.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from llm_router.models.registry import ModelCandidate, PriceSet, Pricing, PricingTier

PER_MILLION = Decimal(1_000_000)


class CostComponent(StrEnum):
    # Values match the price field names on `PriceSet`.
    INPUT = "input"
    OUTPUT = "output"
    CACHE_READ = "cache_read"
    CACHE_WRITE_5M = "cache_write_5m"
    CACHE_WRITE_1H = "cache_write_1h"


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int  # uncached input tokens
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{f.name} must be a non-negative integer, got {value!r}")

    @property
    def prompt_tokens(self) -> int:
        """All input tokens, as used for long-context tier selection."""
        return (
            self.input_tokens
            + self.cache_read_tokens
            + self.cache_write_5m_tokens
            + self.cache_write_1h_tokens
        )

    def tokens(self, component: CostComponent) -> int:
        return {
            CostComponent.INPUT: self.input_tokens,
            CostComponent.OUTPUT: self.output_tokens,
            CostComponent.CACHE_READ: self.cache_read_tokens,
            CostComponent.CACHE_WRITE_5M: self.cache_write_5m_tokens,
            CostComponent.CACHE_WRITE_1H: self.cache_write_1h_tokens,
        }[component]


@dataclass(frozen=True, slots=True)
class CostEstimate:
    model_id: str
    currency: str
    tier: str | None  # None means base prices
    components: dict[CostComponent, Decimal]  # only components with tokens > 0
    total: Decimal
    kind: Literal["estimate"] = "estimate"


@dataclass(frozen=True, slots=True)
class UnknownCost:
    model_id: str
    tier: str | None
    missing: tuple[CostComponent, ...]
    reason: str
    kind: Literal["unknown"] = "unknown"


CostResult = CostEstimate | UnknownCost


@dataclass(frozen=True, slots=True)
class BilledCost:
    """A cost reported by the provider (usage/billing data), kept separate from estimates."""

    model_id: str
    amount: Decimal
    currency: str
    source: str  # e.g. "anthropic-usage-report", "aws-cost-explorer"
    reference: str | None = None  # request ID, invoice line, or report ID
    kind: Literal["billed"] = field(default="billed", init=False)

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError("billed amount cannot be negative")


def select_tier(pricing: Pricing, prompt_tokens: int) -> PricingTier | None:
    """Return the tier with the highest threshold strictly below `prompt_tokens`, else None.

    Thresholds are exclusive: a prompt of exactly the threshold size uses the lower prices.
    """
    selected = None
    for tier in pricing.tiers:  # validated to be in increasing threshold order
        if prompt_tokens > tier.applies_when_prompt_tokens_over:
            selected = tier
    return selected


def estimate_cost(model_id: str, pricing: Pricing, usage: TokenUsage) -> CostResult:
    """Estimate the cost of one request. Prices are required only for components that are used."""
    tier = select_tier(pricing, usage.prompt_tokens)
    prices: PriceSet = tier if tier is not None else pricing
    tier_name = tier.name if tier is not None else None

    components: dict[CostComponent, Decimal] = {}
    missing: list[CostComponent] = []
    for component in CostComponent:
        tokens = usage.tokens(component)
        if tokens == 0:
            continue
        price: Decimal | None = getattr(prices, component.value)
        if price is None:
            missing.append(component)
        else:
            components[component] = Decimal(tokens) / PER_MILLION * price

    if missing:
        names = ", ".join(c.value for c in missing)
        return UnknownCost(
            model_id=model_id,
            tier=tier_name,
            missing=tuple(missing),
            reason=f"no price for {names} ({tier_name or 'base'} prices)",
        )
    return CostEstimate(
        model_id=model_id,
        currency=pricing.currency,
        tier=tier_name,
        components=components,
        total=sum(components.values(), Decimal(0)),
    )


def estimate_candidate_cost(candidate: ModelCandidate, usage: TokenUsage) -> CostResult:
    return estimate_cost(candidate.model_id, candidate.pricing, usage)
