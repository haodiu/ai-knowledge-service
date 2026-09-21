"""Deterministic tier resolution (invariant #1). The LLM never sees or decides this."""
from dataclasses import dataclass

from app.retrieval.schemas import Tier


def allowed_tiers(tier: Tier) -> list[Tier]:
    """`internal` users see general documents too; `general` users never see internal ones.

    ASSUMPTION (Plan §7 defines the JWT `tier` claim but not this mapping): the internal tier is a
    superset. Week 6 feeds this from the verified JWT claim; until then the only caller is the CLI.
    """
    if tier is Tier.INTERNAL:
        return [Tier.GENERAL, Tier.INTERNAL]
    return [Tier.GENERAL]


@dataclass(frozen=True)
class AuthorizationContext:
    """Who is asking and what they may see. Built by trusted code (the verified JWT in Week 6;
    the CLI's --tier flag until then). It lives in the request-scoped runtime context and is
    never placed in graph state, never sent to a model, and never derived from model output."""

    user_id: str
    tier: Tier

    @property
    def allowed_tiers(self) -> list[Tier]:
        return allowed_tiers(self.tier)
