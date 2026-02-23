"""Unit tests for eovs.search -- SemanticSearcher."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from eovs.search import SemanticSearcher


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture()
def mock_data_dir(tmp_path: Path) -> Path:
    """Create a fake data directory with metadata + embeddings."""
    metadata = [
        {"id": 0, "filename": "1.jpg", "class_name": "Forest"},
        {"id": 1, "filename": "2.jpg", "class_name": "River"},
        {"id": 2, "filename": "3.jpg", "class_name": "Highway"},
    ]
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8",
    )

    rng = np.random.default_rng(42)
    embeddings = rng.random((3, 512), dtype=np.float32)
    np.savez(tmp_path / "embeddings.npz", embeddings=embeddings)

    return tmp_path


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


@patch("eovs.search.SentenceTransformer")
def test_semantic_searcher_init(
    mock_st_cls: MagicMock,
    mock_data_dir: Path,
) -> None:
    """SemanticSearcher loads metadata and embeddings correctly."""
    mock_st_cls.return_value = MagicMock()

    searcher = SemanticSearcher(data_dir=mock_data_dir)

    assert len(searcher.metadata) == 3
    assert searcher.image_embeddings.shape == (3, 512)
    mock_st_cls.assert_called_once()


@patch("eovs.search.SentenceTransformer")
def test_semantic_searcher_query(
    mock_st_cls: MagicMock,
    mock_data_dir: Path,
) -> None:
    """search() returns top_k results with the expected keys."""
    rng = np.random.default_rng(99)
    fake_query_emb = rng.random((1, 512), dtype=np.float32)

    mock_model = MagicMock()
    mock_model.encode.return_value = fake_query_emb
    mock_st_cls.return_value = mock_model

    searcher = SemanticSearcher(data_dir=mock_data_dir)
    results = searcher.search("fake query", top_k=2)

    assert len(results) == 2
    for result in results:
        assert hasattr(result, "filename")
        assert hasattr(result, "class_name")
        assert hasattr(result, "score")
        assert isinstance(result.score, float)


@patch("eovs.search.SentenceTransformer")
def test_semantic_searcher_empty_query(
    mock_st_cls: MagicMock,
    mock_data_dir: Path,
) -> None:
    """search() returns an empty list when the query is empty."""
    mock_st_cls.return_value = MagicMock()

    searcher = SemanticSearcher(data_dir=mock_data_dir)
    results = searcher.search("", top_k=5)

    assert results == []
