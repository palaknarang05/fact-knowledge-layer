# Fact Knowledge Layer

A generic pipeline that ingests PDFs, extracts grounded facts, and reconciles
them across documents — flagging corroboration, contradiction, and
context-resolved apparent conflicts. Built for the Superjoin VIT 2026
Engineering Intern assignment.

## Setup and Run Instructions

**Requirements:** Python 3.11+, an OpenAI API key (or any OpenAI-compatible
endpoint — see "Approach" below for swapping in LiteLLM).

```bash
git clone <this-repo>
cd fact-layer
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
```

Run the backend:

```bash
uvicorn backend.main:app --reload --port 8000
```

Run the UI (separate terminal):

```bash
streamlit run frontend/app.py
```

Open the Streamlit URL it prints (typically `http://localhost:8501`), go to
**Upload**, drop one or more PDFs, click **Process**, then explore **Facts
Explorer**, **Fact Relationships**, and **Extraction Failures**.

The API is also directly usable, e.g.:

```bash
curl -F "files=@some-report.pdf" http://localhost:8000/documents/upload
curl http://localhost:8000/relationships?relationship_type=CONTRADICTION
```

Interactive API docs: `http://localhost:8000/docs`.

Everything persists to `./data/` (SQLite DB + a persistent Chroma index), so
stopping and restarting the backend does not lose ingested facts.

## Video Demo

`<link to a ≤3 minute demo video — add before submitting>`

Demo shows: uploading a PDF, the extracted facts view with source quotes,
and one example each of CORROBORATED, CONTRADICTION, RESOLVED_BY_CONTEXT,
and an extraction failure, using the provided `delhivery/` and
`india-macroeconomy/` starter datasets.

## Approach

**Pipeline shape:** PDF → per-page text (`pdfplumber`) → page-aware chunks
(~6k chars, small overlap) → LLM structured-output extraction per chunk
(`backend/extractor.py`) → facts stored in SQLite → facts embedded and
upserted into a persistent Chroma collection → nearest-neighbour candidate
clustering across the *entire* fact store, not just the new document →
LLM structured-output judgment per cluster (`backend/reconciler.py`) →
relationships stored, deduplicated by a stable key so re-running never
double-creates a relationship.

**Why this shape:**

- *Structured outputs over free text.* Every LLM call returns a Pydantic
  model (`FactLLM`/`ExtractionResult`, `RelationshipLLM`/`ReconciliationResult`
  in `backend/models.py`), so the rest of the system never parses prose.
  This is also what makes the "dynamic schema" requirement cheap: a fact
  carries an open `extra: Dict[str, str]` for whatever attributes the
  document's own content suggests (segment name, standalone-vs-consolidated,
  director status, currency basis...) without touching the database schema.

- *Grounding is enforced in code, not just prompted for.* The extractor
  checks that every `exact_quote` actually appears (after de-hyphenation/
  whitespace normalization) in the exact text chunk sent to the model. If it
  doesn't, the fact is kept but flagged `grounded=False` and logged as an
  extraction failure — required case #4 is a structural guarantee, not
  something I had to go hunting for.

- *Reconciliation is embeddings-for-recall + LLM-for-precision.* Chroma
  narrows a new fact down to a handful of semantically close candidates
  (across all documents, not just the one being uploaded); an LLM then
  makes the actual CORROBORATED / CONTRADICTION / RESOLVED_BY_CONTEXT /
  EXTRACTION_FAILURE / UNRELATED call, because "same number, different
  meaning" (e.g. standalone vs consolidated revenue) is a judgment call
  embeddings alone get wrong. `UNRELATED` verdicts are discarded rather
  than stored, so the relationship table only holds genuine findings.

- *Connected components, not just pairs.* Candidate facts are grouped via
  union-find before the LLM call, so three or more mutually-corroborating
  facts (e.g. the same revenue figure stated in the prospectus, annual
  report, and earnings deck) get judged together in one call instead of
  three separate pairwise calls that could disagree with each other.

- *Nothing here is Delhivery- or India-macro-specific.* No filenames,
  section headers, or currency/units are hard-coded anywhere in the
  pipeline; the starter datasets were used only to sanity-check output, not
  to shape the extraction prompt.

**AI tools used:** Claude (Anthropic) for architecture and code generation
of this scaffold; OpenAI `gpt-4o-mini` for extraction and reconciliation at
runtime; OpenAI `text-embedding-3-small` for the similarity index.

**Trade-offs made deliberately for a prototype timeline:**
- SQLite + a single-process Chroma client instead of a real vector DB/queue
  — trivial to swap, not worth it at this scale.
- Reconciliation is triggered synchronously on upload rather than as a
  background job, so large batches block the request; fine for a demo,
  wrong for production (see Next Steps).
- Chunking is purely char-budget-based, not layout-aware; tables sometimes
  extract as run-on text, which is the main source of the extraction
  failures you'll see flagged.

## Limitations and Next Steps

**Does not work yet / known weak points:**
- Table-heavy pages (e.g. the RBI macroeconomic appendix, financial
  statement notes) sometimes flatten into text pdfplumber can't cleanly
  linearize, producing low-confidence or ungrounded facts. A
  layout-aware table extractor (e.g. `camelot`/`pdfplumber` table mode) run
  as a second, specialized extraction pass would help.
- The reconciliation cluster cap (`FKL_MAX_CLUSTER_SIZE`) means a fact with
  many true corroborators can lose some to the cap; a two-stage approach
  (cheap pairwise pre-filter, then a larger single judgment call) would
  scale better.
- No de-duplication of near-identical facts extracted twice from
  overlapping chunk windows — currently harmless (they just become an
  extra CORROBORATED relationship) but noisy at scale.
- No auth/rate limiting on the API — fine for local grading, not for
  anything public.

**Next steps if I kept building:**
- Background task queue (e.g. `arq`/Celery) for ingestion + reconciliation
  so upload requests return immediately and processing status streams to
  the UI.
- Swap the OpenAI SDK calls for `litellm.completion`/`litellm.embedding` to
  make the model provider fully pluggable per the assignment's "any LLM"
  allowance.
- A confidence-weighted "fact freshness" view: when two CORROBORATED facts
  have different `time_period`s, surface the more recent one as the
  current best-known value.
- Batch/async the extraction LLM calls per document (currently sequential
  per chunk) — the largest lever for the "large PDFs without significant
  performance issues" brownie point.

## Additional Notes

- `data/` is gitignored; delete it to reset the knowledge layer from
  scratch (fresh SQLite DB + fresh Chroma index).
- Every config knob (models, chunk size, similarity threshold, cluster cap)
  lives in `backend/config.py` and is overridable via `.env` — see
  `.env.example`.
- The system was designed so that uploading a brand-new, previously-unseen
  PDF (i.e. what the graders will test with) only requires it to look
  vaguely like a document with checkable claims — no schema, filename, or
  section-name assumptions anywhere in the code path.
