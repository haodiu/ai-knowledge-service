from collections.abc import Sequence


def to_vector_literal(vec: Sequence[float]) -> str:
    """pgvector text form, bound as a parameter and cast with CAST(:p AS vector)."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"
