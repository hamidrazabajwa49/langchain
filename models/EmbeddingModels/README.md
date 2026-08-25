# EmbeddingModels

Text embeddings across two providers, plus a mini-project that proves the point of embeddings: they capture *meaning*, not just matching words.

## Concept: what is an embedding?

An embedding model converts a piece of text into a fixed-length vector of numbers — a point in high-dimensional space. Texts with similar meaning end up as vectors that are geometrically close together, even if they don't share any words. This is the foundation underneath semantic search, RAG retrieval, and recommendation systems.

Two operations show up across every embedding provider:
- `embed_query(text)` — embeds a single piece of text (usually a search query)
- `embed_documents([texts])` — embeds a batch of texts (usually your corpus/knowledge base)

## Files

### `huggingface_local_query.py` — HuggingFace embeddings
```python
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
```
- Runs locally via `sentence-transformers`, no API key needed
- MiniLM-L6-v2's native output is a 384-dimension vector
- **Known bug**: the script calls `embedding.embed_documents(document)` but the object is named `embeddings` (plural) — this throws a `NameError` as written. Left in intentionally as an example of the kind of small naming bug that's easy to make when switching between provider SDKs that don't share a naming convention.

### `openai_query.py` — OpenAI embeddings
```python
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)
```
- `text-embedding-3-large` natively outputs 3072 dimensions, but OpenAI's newer embedding models support truncating to a smaller size via the `dimensions` parameter — here forced down to just 32
- Smaller dimensions = less storage/compute downstream, at some cost to retrieval precision
- Auth: `OPENAI_API_KEY` from `.env`

### `semantic_similarity_mini_project.py` — the proof-of-concept
```python
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=500)
```
A small semantic search pipeline built from scratch (no vector DB — just `sklearn`):

1. **Corpus**: 10 cricket-related sentences (batting, bowling, catches, rain delays, etc.)
2. **Query**: `"Who scored a hundred in the finals?"`
3. **Embed everything**: `embed_documents()` on the corpus, `embed_query()` on the query
4. **Rank by similarity**: `sklearn.metrics.pairwise.cosine_similarity([query_embedding], doc_embeddings)`
5. **Pick the best match**: sort by score, take the highest

**Result**: the top-ranked sentence is about a player scoring a century in a World Cup final — despite sharing *zero* exact keywords with the query ("hundred" vs "century," no shared "scored," different phrasing entirely). The match is purely semantic.

### Cosine similarity, intuitively
Cosine similarity measures the angle between two vectors, not their distance. A score of `1.0` means the vectors point in exactly the same direction (same meaning); `0.0` means unrelated; negative means opposite. It's preferred over raw distance for embeddings because it ignores vector magnitude — two texts can be "about the same thing" at different levels of intensity/length and still score close to `1.0`.

## Key takeaway

Embedding dimensionality is a knob, not a fixed property — `dimensions=32` vs `dimensions=500` vs MiniLM's native 384 will all produce structurally different vectors. Anything comparing vectors downstream (cosine similarity, a vector database index) needs the query and corpus embedded with the **same model and same dimension setting** — mixing them silently breaks the comparison.

## Run

```bash
python openai_query.py                      # needs OPENAI_API_KEY
python huggingface_local_query.py            # no key needed (has a known bug, see above)
python semantic_similarity_mini_project.py   # needs OPENAI_API_KEY; prints query, best match, and similarity score
```
