import math

from app.db.models import EMBEDDING_DIM
from app.ingestion.fake_embedder import fake_embed


def test_dimension_matches_the_vector_column() -> None:
    (vec,) = fake_embed(["hello"])
    assert len(vec) == EMBEDDING_DIM == 1536


def test_deterministic_and_text_sensitive() -> None:
    a1, a2, b = fake_embed(["alpha", "alpha", "beta"])
    assert a1 == a2
    assert a1 != b


def test_unit_norm_and_batch_order() -> None:
    out = fake_embed(["x", "y", "z"])
    assert len(out) == 3
    assert out[0] == fake_embed(["x"])[0]
    for vec in out:
        assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, rel_tol=1e-9)
