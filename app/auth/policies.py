"""Deterministic tier resolution (invariant #1). The LLM never sees or decides this."""
from app.retrieval.schemas import Tier


def allowed_tiers(tier: Tier) -> list[Tier]:
    """`internal` users see general documents too; `general` users never see internal ones.

    ASSUMPTION (Plan §7 defines the JWT `tier` claim but not this mapping): the internal tier is a
    superset. Week 6 feeds this from the verified JWT claim; until then the only caller is the CLI.
    """
    if tier is Tier.INTERNAL:
        return [Tier.GENERAL, Tier.INTERNAL]
    return [Tier.GENERAL]
