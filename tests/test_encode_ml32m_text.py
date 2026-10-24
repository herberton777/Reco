"""Offline tests for the full MovieLens 32M semantic-text encoder.

These use a small fake encoder, no network, no Torch and no MovieLens download.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.encode_ml32m_text import (
    _parse_cutoff,
    clean_tag,
    encode_catalog,
    movie_tag_vector_at,
    pool_movie_tags,
)


class FakeEncoder:
    def __init__(self, dimension=8):
        self.dimension = dimension
        self.texts_seen = []

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, batch_size, convert_to_numpy, normalize_embeddings,
               show_progress_bar):
        assert batch_size >= 1 and convert_to_numpy
        assert normalize_embeddings and not show_progress_bar
        self.texts_seen += list(texts)
        vectors = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
            rng = np.random.default_rng(seed)
            v = rng.normal(size=self.dimension).astype("float32")
            vectors.append(v / np.linalg.norm(v))
        return np.asarray(vectors)


def _fixture(tmp_path):
    movies = tmp_path / "movies.csv"
    tags = tmp_path / "tags.csv"
    pd.DataFrame([
        {"movieId": 20, "title": "Film B (2002)", "genres": "Action|Drama"},
        {"movieId": 10, "title": "Film A (2001)", "genres": "Comedy"},
        {"movieId": 30, "title": "Film C (2003)", "genres": "(no genres listed)"},
    ]).to_csv(movies, index=False)
    pd.DataFrame([
        {"userId": 1, "movieId": 10, "tag": "  FUNNY  ", "timestamp": 100},
        {"userId": 2, "movieId": 10, "tag": "funny", "timestamp": 110},
        {"userId": 3, "movieId": 20, "tag": "  Historic Drama ", "timestamp": 150},
        {"userId": 4, "movieId": 10, "tag": "sad", "timestamp": 200},
        {"userId": 5, "movieId": 20, "tag": "future secret", "timestamp": 300},
        {"userId": 6, "movieId": 10, "tag": "", "timestamp": 100},
    ]).to_csv(tags, index=False)
    return movies, tags


def test_full_title_and_deduplicated_tag_encoding(tmp_path):
    movies, tags = _fixture(tmp_path)
    fake = FakeEncoder()
    output = tmp_path / "semantic"
    metadata = encode_catalog(
        movies, tags, output, encoder=fake, model_id="fake",
        batch_size=2, text_chunk_size=2, tag_chunk_size=2,
    )
    assert metadata["movies"] == 3
    assert metadata["distinct_tag_texts"] == 4
    assert metadata["indexed_tag_events"] == 5
    assert metadata["skipped_empty_tag_events"] == 1
    assert len(fake.texts_seen) == 3 + 4
    assert fake.texts_seen[:2] == [
        "Movie: Film A (2001). Genres: Comedy",
        "Movie: Film B (2002). Genres: Action, Drama",
    ]
    ids = np.load(output / "movie_ids.npy", allow_pickle=False)
    assert list(ids) == [10, 20, 30]
    title = np.load(output / "title_embeddings.npy", mmap_mode="r")
    tag = np.load(output / "tag_embeddings.npy", mmap_mode="r")
    assert title.shape == (3, 8)
    assert tag.shape == (4, 8)
    assert np.allclose(np.linalg.norm(title, axis=1), 1)
    with sqlite3.connect(output / "tag_index.sqlite") as db:
        assert db.execute("SELECT count(*) FROM tag_texts").fetchone()[0] == 4
        assert db.execute("SELECT count(*) FROM tag_events").fetchone()[0] == 5
        funny_id = db.execute(
            "SELECT id FROM tag_texts WHERE tag='funny'").fetchone()[0]
        assert db.execute(
            "SELECT count(*) FROM tag_events WHERE tag_id=?",
            (funny_id,)).fetchone()[0] == 2
    assert json.loads((output / "manifest.json").read_text())["dimension"] == 8
    with pytest.raises(FileExistsError):
        encode_catalog(movies, tags, output, encoder=fake)


def test_tag_profiles_are_strictly_point_in_time(tmp_path):
    movies, tags = _fixture(tmp_path)
    directory = tmp_path / "semantic"
    encode_catalog(movies, tags, directory, encoder=FakeEncoder(),
                   tag_chunk_size=2, text_chunk_size=2)
    db = sqlite3.connect(directory / "tag_index.sqlite")
    embeddings = np.load(directory / "tag_embeddings.npy", mmap_mode="r")
    before = movie_tag_vector_at(db, embeddings, movie_id=10, cutoff=100)
    at_110 = movie_tag_vector_at(db, embeddings, movie_id=10, cutoff=110)
    at_150 = movie_tag_vector_at(db, embeddings, movie_id=10, cutoff=150)
    at_250 = movie_tag_vector_at(db, embeddings, movie_id=10, cutoff=250)
    assert np.allclose(before, 0)
    assert not np.allclose(at_110, 0)
    # timestamp 110 is excluded at cutoff 110, and inclusion at 150 does
    # not change normalized single-tag pooling.
    assert np.allclose(at_110, at_150)
    assert not np.allclose(at_150, at_250)
    db.close()

    result = tmp_path / "pooled_250.npy"
    meta = pool_movie_tags(directory, 250, result)
    assert meta["movies_with_prior_tags"] == 2
    pooled = np.load(result, mmap_mode="r")
    assert pooled.shape == (3, 8)
    assert np.allclose(pooled[2], 0)  # Film C has no tags
    assert (tmp_path / "pooled_250.json").is_file()
    with pytest.raises(FileExistsError):
        pool_movie_tags(directory, 250, result)

    earlier = tmp_path / "pooled_100.npy"
    meta2 = pool_movie_tags(directory, 100, earlier)
    assert meta2["movies_with_prior_tags"] == 0


def test_parsing_and_tag_cleanup():
    assert clean_tag("  Sci-Fi  ") == "sci-fi"
    assert clean_tag(None) == ""
    assert clean_tag("\u3000DraMA \t") == "drama"
    assert _parse_cutoff("2020-01-01") == 1577836800
    assert _parse_cutoff("2020-01-01T00:00:00Z") == 1577836800
    assert _parse_cutoff("1577836800") == 1577836800


def test_bad_movie_id_rejected(tmp_path):
    movies, tags = _fixture(tmp_path)
    tagged = pd.read_csv(tags)
    tagged.loc[0, "movieId"] = 999
    tagged.to_csv(tags, index=False)
    with pytest.raises(ValueError, match="unknown movieId"):
        encode_catalog(movies, tags, tmp_path / "out",
                       encoder=FakeEncoder(), tag_chunk_size=2)
