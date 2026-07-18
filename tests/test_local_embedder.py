"""Tests for sage/embed/local_embedder.py.

Most tests fake `_get_model()`, mirroring sage/retrieval/reranker.py's own
test convention (tests/test_reranker.py) exactly -- consistent with this
codebase's existing style, and avoids a real ~130MB model load/download on
every test run. One test (marked below) deliberately exercises the REAL
`BAAI/bge-small-en-v1.5` model end-to-end: unlike the Gemini embeddings path
this replaced, there's no API key or per-call quota risk to a real call here,
only a one-time model download (cached locally after the first run) --
worth the one real integration test for genuine confidence the actual
embedding path (not just the interface) works, without slowing down the
rest of the suite.
"""

import numpy as np

from sage.embed import local_embedder


class FakeSentenceTransformer:
    """Deterministic per input text, no real model load."""

    def __init__(self, dimensions: int = 8):
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        self.calls.append(list(texts))
        return np.array([self._vector(t) for t in texts])

    def _vector(self, text: str) -> list[float]:
        h = abs(hash(text))
        return [((h >> i) % 997) / 997.0 for i in range(self.dimensions)]


def test_embed_text_returns_a_single_vector(monkeypatch):
    fake_model = FakeSentenceTransformer(dimensions=8)
    monkeypatch.setattr(local_embedder, "_get_model", lambda: fake_model)

    vector = local_embedder.embed_text("Apple margins declined.")

    assert len(vector) == 8
    assert isinstance(vector, list)


def test_embed_texts_returns_one_vector_per_text_in_order(monkeypatch):
    fake_model = FakeSentenceTransformer(dimensions=8)
    monkeypatch.setattr(local_embedder, "_get_model", lambda: fake_model)

    vectors = local_embedder.embed_texts(["a", "b", "c"])

    assert len(vectors) == 3
    # Deterministic: embedding "a" alone matches its slot in the batch.
    assert vectors[0] == local_embedder.embed_text("a")


def test_embed_texts_is_a_single_encode_call_not_one_per_text(monkeypatch):
    fake_model = FakeSentenceTransformer(dimensions=8)
    monkeypatch.setattr(local_embedder, "_get_model", lambda: fake_model)

    local_embedder.embed_texts(["a", "b", "c", "d", "e"])

    # sentence-transformers batches internally -- no need to chunk calls
    # ourselves the way the removed Gemini path had to for its rate limit.
    assert len(fake_model.calls) == 1
    assert fake_model.calls[0] == ["a", "b", "c", "d", "e"]


def test_embed_texts_empty_list_short_circuits_without_loading_model(monkeypatch):
    def _fail():
        raise AssertionError("model should not load for an empty input list")

    monkeypatch.setattr(local_embedder, "_get_model", _fail)

    assert local_embedder.embed_texts([]) == []


def test_embed_query_prepends_the_bge_instruction_prefix(monkeypatch):
    fake_model = FakeSentenceTransformer(dimensions=8)
    monkeypatch.setattr(local_embedder, "_get_model", lambda: fake_model)

    local_embedder.embed_query("Apple margins")

    assert fake_model.calls[-1] == [
        "Represent this sentence for searching relevant passages: Apple margins"
    ]


def test_embed_query_differs_from_embed_text_for_the_same_input(monkeypatch):
    fake_model = FakeSentenceTransformer(dimensions=8)
    monkeypatch.setattr(local_embedder, "_get_model", lambda: fake_model)

    assert local_embedder.embed_query("Apple margins") != local_embedder.embed_text("Apple margins")


def test_get_model_is_a_lazy_singleton(monkeypatch):
    load_count = {"n": 0}

    def fake_constructor(model_name):
        load_count["n"] += 1
        return FakeSentenceTransformer(dimensions=4)

    monkeypatch.setattr(local_embedder, "_model", None)
    monkeypatch.setattr(local_embedder, "SentenceTransformer", fake_constructor)

    local_embedder._get_model()
    local_embedder._get_model()

    assert load_count["n"] == 1


# --- Real model integration test (see module docstring) ---


def test_embed_texts_with_real_model_produces_valid_deterministic_vectors():
    # No monkeypatching here -- this test deliberately loads and runs the
    # real BAAI/bge-small-en-v1.5 model (pytest's monkeypatch fixture used by
    # every other test in this file is function-scoped, so it can't leak in).
    from config import settings

    vectors = local_embedder.embed_texts(["Apple reported strong Services growth."])

    assert len(vectors) == 1
    assert len(vectors[0]) == 384  # BAAI/bge-small-en-v1.5's native dimension
    assert any(v != 0.0 for v in vectors[0])

    # Determinism: embedding the same text twice yields identical vectors.
    again = local_embedder.embed_text("Apple reported strong Services growth.")
    assert vectors[0] == again

    assert settings.LOCAL_EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"
