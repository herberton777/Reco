"""Encode *all* MovieLens 32M movie titles and distinct tag texts.

Movie tags are timestamped human contributions: never pool future tag events
into historical validation/test features. This module keeps every event with
its timestamp, and requires a point-in-time cutoff to pool per-movie tags.

Train-time dependencies: sentence-transformers, numpy, pandas.
Tests can inject a deterministic fake encoder without downloading a model.
Usage: python -m src.encode_ml32m_text --help
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import unicodedata

import numpy as np
import pandas as pd


DDL = """
CREATE TABLE tag_texts (
    id INTEGER PRIMARY KEY,
    tag TEXT NOT NULL UNIQUE
);
CREATE TABLE tag_events (
    movie_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL REFERENCES tag_texts(id),
    timestamp INTEGER NOT NULL
);
CREATE INDEX ix_tag_events_movie_time ON tag_events(movie_id,timestamp);
"""


def clean_tag(value):
    if value is None or pd.isna(value):
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value))
                    .casefold().split())


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for data in iter(lambda: source.read(1 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def _movie_data(movies_file):
    movies = pd.read_csv(movies_file, usecols=["movieId", "title", "genres"])
    if movies.empty or movies["movieId"].isna().any():
        raise ValueError("movies.csv has no valid movie IDs")
    movies["movieId"] = movies["movieId"].astype("int64")
    if movies["movieId"].duplicated().any():
        raise ValueError("Duplicate movie IDs")
    return movies.sort_values("movieId", kind="stable").reset_index(drop=True)


def _insert_tag_events(db, tags_file, valid_movie_ids, chunksize):
    """Deduplicate tag *texts*, but preserve all movie/tag/timestamp events.

    This avoids loading two million+ raw tag applications into RAM and
    retains time cutoffs for historical evaluation.
    """
    events = skipped = 0
    query_batch = 500  # stays below older SQLite host-parameter defaults
    reader = pd.read_csv(
        tags_file, usecols=["movieId", "tag", "timestamp"],
        chunksize=chunksize, dtype={"tag": "string"},
    )
    for chunk in reader:
        rows = []
        for movie, tag, stamp in chunk.itertuples(index=False, name=None):
            value = clean_tag(tag)
            if not value:
                skipped += 1
                continue
            if pd.isna(stamp) or not np.isfinite(float(stamp)) or (
                    float(stamp) != int(stamp)):
                raise ValueError("Invalid tag timestamp")
            movie_id = int(movie)
            if movie_id not in valid_movie_ids:
                raise ValueError(f"Tag references unknown movieId={movie_id}")
            rows.append((movie_id, value, int(stamp)))
        if not rows:
            continue
        texts = sorted({tag for _, tag, _ in rows})
        with db:
            db.executemany(
                "INSERT OR IGNORE INTO tag_texts(tag) VALUES (?)",
                ((tag,) for tag in texts),
            )
            mapping = {}
            for at in range(0, len(texts), query_batch):
                group = texts[at:at + query_batch]
                placeholders = ",".join("?" for _ in group)
                for row in db.execute(
                    f"SELECT id,tag FROM tag_texts WHERE tag IN ({placeholders})",
                    group,
                ):
                    mapping[row[1]] = row[0]
            if len(mapping) != len(texts):
                raise RuntimeError("Internal failure mapping deduplicated tags")
            db.executemany(
                "INSERT INTO tag_events(movie_id, tag_id, timestamp) VALUES (?,?,?)",
                ((movie, mapping[tag], stamp) for movie, tag, stamp in rows),
            )
        events += len(rows)
        print(f"Indexed {events:,} tag events; {skipped:,} blank tags skipped",
              flush=True)
    return events, skipped


def _encode_chunks(encoder, texts, indices, destination, dimension, batch_size):
    """Write one bounded text chunk at a time into an NPY memory-mapped matrix."""
    if not texts:
        return
    values = encoder.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(texts), dimension):
        raise ValueError(f"Sentence model shape mismatch: got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Sentence model generated nonfinite embeddings")
    destination[indices] = values


def encode_catalog(movies_file, tags_file, output, *,
                   encoder, batch_size=64, text_chunk_size=2048,
                   tag_chunk_size=100000, model_id="test-model",
                   title_only=False):
    """Index every movie and distinct tag, and encode them using bounded memory.

    The directory must not exist (never accidentally overwrite artifacts).
    """
    movies_file, tags_file, output = map(Path, (movies_file, tags_file, output))
    if not movies_file.is_file() or not tags_file.is_file():
        raise FileNotFoundError("Provide existing MovieLens movies.csv and tags.csv")
    if output.exists():
        raise FileExistsError("Output directory exists; use a new --output: " + str(output))
    if min(batch_size, text_chunk_size, tag_chunk_size) < 1:
        raise ValueError("Chunk sizes and batch size must be positive")

    movies = _movie_data(movies_file)
    dims = encoder.get_sentence_embedding_dimension()
    if not isinstance(dims, int) or dims < 1:
        raise ValueError("Sentence encoder must have a positive output dimension")

    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "movie_ids.npy", movies["movieId"].to_numpy(dtype="int64"),
            allow_pickle=False)
    db = sqlite3.connect(output / "tag_index.sqlite")
    try:
        db.executescript(DDL)
        db.execute("PRAGMA foreign_keys = ON")
        events, skipped = _insert_tag_events(
            db, tags_file, set(movies["movieId"].tolist()), tag_chunk_size,
        )
        tag_count = db.execute("SELECT COUNT(*) FROM tag_texts").fetchone()[0]
        max_tag_id = db.execute("SELECT COALESCE(MAX(id),0) FROM tag_texts").fetchone()[0]
        if tag_count != max_tag_id:
            raise RuntimeError("Tag ID gap: cannot use tag_id - 1 as embedding row")

        titles = np.lib.format.open_memmap(
            output / "title_embeddings.npy", mode="w+",
            dtype="float32", shape=(len(movies), dims),
        )
        for at in range(0, len(movies), text_chunk_size):
            frame = movies.iloc[at:at + text_chunk_size]
            texts = [
                str(title) if title_only else
                f"Movie: {title}. Genres: {str(genres).replace('|', ', ')}"
                for title, genres in zip(frame["title"], frame["genres"])
            ]
            _encode_chunks(
                encoder, texts, slice(at, at + len(frame)), titles, dims, batch_size,
            )
            print(f"Encoded {at + len(frame):,}/{len(movies):,} movie titles", flush=True)
        titles.flush()

        # npy stores row = tag_id - 1. Support files with zero usable tags by
        # storing zero rows via np.save (open_memmap cannot map empty payload).
        if tag_count:
            embeddings = np.lib.format.open_memmap(
                output / "tag_embeddings.npy", mode="w+",
                dtype="float32", shape=(tag_count, dims),
            )
            cursor = db.execute("SELECT id,tag FROM tag_texts ORDER BY id")
            completed = 0
            while True:
                batch = cursor.fetchmany(text_chunk_size)
                if not batch:
                    break
                rows = np.asarray([tag_id - 1 for tag_id, _ in batch], dtype=np.int64)
                texts = [tag for _, tag in batch]
                _encode_chunks(encoder, texts, rows, embeddings, dims, batch_size)
                completed += len(batch)
                print(f"Encoded {completed:,}/{tag_count:,} unique tags", flush=True)
            embeddings.flush()
        else:
            np.save(output / "tag_embeddings.npy",
                    np.zeros((0, dims), dtype="float32"), allow_pickle=False)

    finally:
        db.close()

    manifest = {
        "model_id": str(model_id),
        "dimension": int(dims),
        "movies": len(movies),
        "distinct_tag_texts": int(tag_count),
        "indexed_tag_events": int(events),
        "skipped_empty_tag_events": int(skipped),
        "movie_embedding_text": "title" if title_only else "title + genres",
        "tag_embeddings_are_text_only": True,
        "movie_tag_profiles": "NOT aggregated; generate at a strictly earlier cutoff",
        "row_mapping": {
            "title_embeddings.npy": "movie_ids.npy, same row",
            "tag_embeddings.npy": "tag_index.sqlite tag_texts.id - 1",
        },
        "input_sha256": {
            "movies.csv": _sha256_file(movies_file),
            "tags.csv": _sha256_file(tags_file),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return manifest


def _parse_cutoff(value):
    if value.isdigit():
        return int(value)
    iso = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def movie_tag_vector_at(db, embeddings, movie_id, cutoff):
    """Compute one movie's *past-only* tag vector for a query timestamp.

    Repeated tag applications per movie are tempered with log(1+count).
    No tag event at or after the cutoff may enter the result.
    """
    pooled = np.zeros(embeddings.shape[1], dtype=np.float64)
    total_weight = 0.0
    for tag_id, count in db.execute(
        """SELECT tag_id, COUNT(*) FROM tag_events
           WHERE movie_id = ? AND timestamp < ?
           GROUP BY tag_id""",
        (int(movie_id), int(cutoff)),
    ):
        weight = math.log1p(int(count))
        pooled += weight * embeddings[int(tag_id) - 1]
        total_weight += weight
    if total_weight > 0:
        pooled /= total_weight
        norm = float(np.linalg.norm(pooled))
        if norm > 0:
            pooled /= norm
    return np.asarray(pooled, dtype="float32")


def pool_movie_tags(cache_dir, cutoff, output):
    """Produce per-movie tag profiles for a single historical UTC cutoff."""
    cache_dir, output = Path(cache_dir), Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite pooled features: " + str(output))
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    ids = np.load(cache_dir / "movie_ids.npy", mmap_mode="r", allow_pickle=False)
    vectors = np.load(cache_dir / "tag_embeddings.npy", mmap_mode="r", allow_pickle=False)
    if vectors.shape[1] != manifest["dimension"]:
        raise ValueError("Tag embedding dimension doesn't match manifest")
    if len(ids) != manifest["movies"]:
        raise ValueError("Movie count doesn't match manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    pooled = np.lib.format.open_memmap(
        output, mode="w+", dtype="float32", shape=(len(ids), vectors.shape[1]),
    )
    db = sqlite3.connect(
        "file:" + str((cache_dir / "tag_index.sqlite").resolve()) + "?mode=ro", uri=True,
    )
    try:
        for row, movie_id in enumerate(ids):
            pooled[row] = movie_tag_vector_at(db, vectors, int(movie_id), cutoff)
            if (row + 1) % 10000 == 0:
                print(f"Pooled {row + 1:,}/{len(ids):,} movies", flush=True)
    finally:
        db.close()
    pooled.flush()
    coverage = int(np.count_nonzero(np.any(pooled != 0, axis=1)))
    pool_meta = {
        "source_embedding_directory": str(cache_dir.resolve()),
        "strictly_before_timestamp_utc": int(cutoff),
        "movie_count": len(ids),
        "movies_with_prior_tags": coverage,
        "weights": "log1p(number of prior tag applications per movie/tag text)",
        "normalization": "l2 after weighted average, zero for missing tags",
        "row_mapping": "movie_ids.npy, same row",
    }
    (output.with_suffix(".json")).write_text(
        json.dumps(pool_meta, indent=2), encoding="utf-8",
    )
    return pool_meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    make = command.add_parser("encode", help="Encode ALL titles and distinct tags")
    make.add_argument("--data-dir", type=Path, default=Path("data/ml-32m"))
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    make.add_argument("--device", default="cpu")
    make.add_argument("--batch-size", type=int, default=64)
    make.add_argument("--text-chunk-size", type=int, default=2048)
    make.add_argument("--tag-chunk-size", type=int, default=100000)
    make.add_argument("--title-only", action="store_true")
    pool = command.add_parser("pool-tags", help="Pool tags only BEFORE a UTC cutoff")
    pool.add_argument("--cache-dir", type=Path, required=True)
    pool.add_argument("--cutoff", required=True,
                      help="Unix UTC seconds or ISO 8601 timestamp, e.g. 2020-01-01")
    pool.add_argument("--output", type=Path, required=True,
                      help="New .npy file for per-movie time-filtered tag profiles")
    args = parser.parse_args(argv)
    if args.command == "encode":
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(args.model, device=args.device)
        outcome = encode_catalog(
            args.data_dir / "movies.csv", args.data_dir / "tags.csv",
            args.output, encoder=encoder, model_id=args.model,
            batch_size=args.batch_size, text_chunk_size=args.text_chunk_size,
            tag_chunk_size=args.tag_chunk_size, title_only=args.title_only,
        )
    else:
        if args.output.suffix.lower() != ".npy":
            parser.error("--output must end in .npy")
        outcome = pool_movie_tags(
            args.cache_dir, _parse_cutoff(args.cutoff), args.output,
        )
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
