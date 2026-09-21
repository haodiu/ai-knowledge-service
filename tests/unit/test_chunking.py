from app.ingestion.chunking import chunk_text


def test_deterministic() -> None:
    text = "One.\n\nTwo two.\n\n" + "word " * 500
    assert chunk_text(text) == chunk_text(text)


def test_short_paragraphs_are_packed_together() -> None:
    assert chunk_text("A\n\nB\n\nC") == ["A\n\nB\n\nC"]


def test_paragraphs_are_never_split_when_they_fit() -> None:
    chunks = chunk_text("aaaa\n\nbbbb\n\ncccc", max_chars=10)
    assert chunks == ["aaaa\n\nbbbb", "cccc"]


def test_no_chunk_exceeds_the_limit_and_no_words_are_lost() -> None:
    words = [f"w{i}" for i in range(400)]
    chunks = chunk_text(" ".join(words), max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks).split() == words


def test_a_single_oversized_word_is_hard_sliced() -> None:
    chunks = chunk_text("x" * 250, max_chars=100)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_blank_and_whitespace_only_input_yields_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text(" \n\n \t\n") == []


def test_crlf_is_normalised() -> None:
    assert chunk_text("A\r\n\r\nB") == ["A\n\nB"]
