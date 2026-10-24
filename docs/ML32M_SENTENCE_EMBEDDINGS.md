# Encode every MovieLens 32M film and free-text tag with Sentence Transformers

The official MovieLens 32M release has **87,585 movie titles** and
**2,000,072 timestamped tag applications** in `movies.csv` and `tags.csv`.
Tags are repeated, sometimes differing only by capitalization or whitespace.
This encoder produces an embedding for **every movie** and **every distinct
normalized tag phrase** (no 20,000-movie cap). It retains all nonblank tag
events, with their movie IDs and UTC timestamps, in a searchable SQLite index.

The implementation does **not** modify HOFM3/HOFM4 checkpoints, select
new positives, or claim an NDCG improvement. It produces text features for a
later model/retraining step.

## 1. Install Sentence Transformers

From the prepared GitHub Codespace, after safely synchronizing `main`:

```bash
cd /workspaces/Reco
python -m pip install sentence-transformers
```

The MovieLens 32M dataset must already be extracted to:

```text
data/ml-32m/movies.csv
data/ml-32m/tags.csv
```

Check paths before starting:

```bash
ls -lh data/ml-32m/{movies,tags}.csv
```

The official release contains these files; this script does **not** download
the dataset or alter the original CSVs.

## 2. Encode the full catalog (CPU-friendly and streaming)

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
python -u -m src.encode_ml32m_text encode \
  --data-dir data/ml-32m \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --device cpu \
  --batch-size 64 \
  --text-chunk-size 2048 \
  --tag-chunk-size 100000 \
  --output artifacts/semantic/ml32m_minilm_full_v1 \
  2>&1 | tee semantic_ml32m_v1.log
```

`all-MiniLM-L6-v2` produces **384-dimensional vectors** and is a
reasonable fast baseline. The code calls `SentenceTransformer.encode`
on batches and requests normalized float32 output. On the first invocation,
the chosen model's weights are fetched from Hugging Face.

To test a larger model (new output folder, **do not overwrite**):

```bash
python -u -m src.encode_ml32m_text encode \
  --data-dir data/ml-32m \
  --model sentence-transformers/all-mpnet-base-v2 \
  --device cpu --batch-size 32 \
  --output artifacts/semantic/ml32m_mpnet_full_v1
```

`all-mpnet-base-v2` has 768 dimensions. It is slower and uses more
storage, and it does **not** guarantee higher MovieLens NDCG.

By default, each film is encoded with `Movie: TITLE. Genres: GENRES`,
rather than title alone. Use `--title-only` to encode just the title,
as a controlled ablation. **No user tags are inserted into film-title
embeddings**, so this branch remains independent of tag timestamps.

## 3. Output and mapping

Output directory `artifacts/semantic/ml32m_minilm_full_v1/`:

| Artifact | Content |
| --- | --- |
| `movie_ids.npy` | Sorted movie IDs, one per title embedding row |
| `title_embeddings.npy` | Shape approximately `(87585, 384)`; normalized film/title+genre vectors |
| `tag_embeddings.npy` | Shape `(N_distinct_tag_phrases, 384)`; L2-normalized unique tag text vectors |
| `tag_index.sqlite` | `tag_texts(id,tag)` and `tag_events(movie_id,tag_id,timestamp)`; original event times preserved |
| `manifest.json` | Model ID, dimensions, counts, original CSV SHA256 fingerprints, vector row mapping |

The actual tag vocabulary size is counted from your `tags.csv` and
is **not assumed** in the script.

The pipeline streams tag input chunks through SQLite and writes vectors
into NPY files using memory mapping, rather than loading millions of
tag-event rows into memory. It refuses to replace an existing output
directory. If a run is interrupted, its partial output is intentionally
not treated as a complete cache: choose a new destination or explicitly
inspect and archive the partial folder before restarting.

For **384-dimensional float32 vectors**, title embeddings occupy
approximately **128 MiB**. Every further 100,000 distinct tag phrases
requires approximately **146 MiB**; SQLite tag events and model memory
are additional.

## 4. Aggregate tag vectors strictly BEFORE a cutoff

**Critical for honest NDCG@10:** the MovieLens `tags.csv` includes tags
written at different times. It is unsafe to pool *every tag ever applied*
to a film and reuse that profile for a historical train/test experiment.

The encoder intentionally **does not** create an all-time per-movie
tag embedding. Instead, pool only tag events with `timestamp < cutoff`:

```bash
python -m src.encode_ml32m_text pool-tags \
  --cache-dir artifacts/semantic/ml32m_minilm_full_v1 \
  --cutoff 2020-01-01T00:00:00Z \
  --output artifacts/semantic/movie_tags_before_2020.npy
```

Replace the example cutoff with the **actual cutoff in your evaluation
protocol**. It produces vectors aligned with `movie_ids.npy` and a
sidecar JSON containing the strict UTC cutoff and number of films with
historical tags. Films with no earlier tags get zero vectors.

For film i at cutoff t, the tag representation is:

\[
v_{i,t} = \operatorname{L2norm}\left(
\frac{\sum_{s \in T_i(t)}
 \log(1+n_{i,s,t})\;e_s}
{\sum_{s \in T_i(t)}\log(1+n_{i,s,t})}
\right)
\]

where `e_s` is the normalized Sentence Transformer vector for unique
tag text s, and `n_{i,s,t}` counts applications **strictly before t**.
This tames repeated tags instead of giving unbounded extra weight to
popular phrases. For empty tag history, the result is zero.

For per-user chronological evaluation where *each recommendation decision*
has a distinct timestamp, a single global cutoff is not sufficient. Use
the `movie_tag_vector_at(db, embeddings, movie_id, query_timestamp)`
helper at each decision (with caching only for exactly identical
query timestamps). Do not use a cutoff later than the prediction time.

## 5. Load vectors in HOFM3/HOFM4 or LambdaRank

```python
import numpy as np

directory = "artifacts/semantic/ml32m_minilm_full_v1"
ids = np.load(f"{directory}/movie_ids.npy", mmap_mode="r")
title_vec = np.load(f"{directory}/title_embeddings.npy", mmap_mode="r")
tag_vec = np.load("artifacts/semantic/movie_tags_before_2020.npy", mmap_mode="r")

movie_id_to_row = {int(m): ix for ix, m in enumerate(ids)}

movie_id = 1
row = movie_id_to_row[movie_id]
title_feature = title_vec[row]  # shape: (384,)
tag_feature = tag_vec[row]      # shape: (384,)
```

HOFM3/HOFM4 currently expects a particular title/content embedding
dimension. **Do not silently replace its embedding matrix** with a
different dimensionality or mix incompatible row orders. Adapt the
features through a trainable projection layer or retrain the appropriate
content module, and verify the code's original movie ID ↔ row mapping.

For LambdaRank, use compact **scalar** signals derived without leakage,
such as cosine similarity between a user's *past-only* preference vector
and a candidate's title/tag vector. Avoid feeding all 768 floating-point
dimensions directly into a small reranker.

## 6. Check against your 0.13 reference

Keep the same movie availability, candidate pool, users, timestamp cutoffs,
NDCG@10 implementation and held-out test set. Compare:

- Existing title representation (baseline)
- Full-catalog title+genre MiniLM vectors
- MiniLM title+genre plus cutoff-filtered tags
- MPNet as a separate embedding-model ablation

Do not expect the new vectors to improve NDCG automatically. The scripts
generate the embeddings, but **do not connect them to the patch-based HOFM
training model yet**, and do not run training/evaluation on the user's
Codespace.

Offline regression tests:

```bash
python -m pytest -q tests/test_encode_ml32m_text.py
```
