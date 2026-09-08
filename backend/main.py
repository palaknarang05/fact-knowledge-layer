"""
FastAPI entry point.

Endpoints:
  POST /documents/upload   - upload one or more PDFs, run full pipeline
  GET  /documents          - list ingested documents
  GET  /facts               - list facts, optional ?document_id=&entity=
  GET  /facts/{fact_id}     - single fact
  GET  /failures            - list extraction failures, optional ?document_id=
  GET  /relationships        - list relationships, optional ?relationship_type=
  POST /reconcile/rerun      - re-run reconciliation over ALL stored facts
                                (useful after tuning thresholds, or to
                                backfill relationships for facts ingested
                                before reconciliation existed)
  GET  /health

Nothing here hard-codes a document name or schema: uploading any PDF runs
the same generic pipeline, and the API shape does not change based on what
kind of facts come back.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import config, db
from .extractor import process_document_async
from .models import DocumentRecord, ExtractionFailure, Fact, FactRelationship, RelationshipType, new_id
from .reconciler import run_reconciliation

app = FastAPI(title="Fact Knowledge Layer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local prototype: the Streamlit UI runs on a different port
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    db.init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
async def _save_upload(upload: UploadFile, document_id: str):
    """Reads and writes the file off the event loop's synchronous path so
    one large upload doesn't block other concurrently-arriving uploads."""
    content = await upload.read()
    dest_path = config.PDF_STORE_DIR / f"{document_id}_{upload.filename}"
    await asyncio.to_thread(dest_path.write_bytes, content)
    return dest_path


async def _extract_one(upload: UploadFile, document_id: str, semaphore: asyncio.Semaphore) -> dict:
    """Phase 1 for a single file: save + extract only, bounded by
    DOCUMENT_CONCURRENCY. No DB writes and no reconciliation here - those
    stay sequential in phase 2 so the existing dedupe/error-handling logic
    is untouched by running multiple files' extraction concurrently.

    Never raises: every failure mode (bad extension, unreadable PDF,
    unexpected error) is captured into the returned dict's `error` field so
    a bad file can never take the rest of the batch down when awaited via
    asyncio.gather."""
    if not upload.filename.lower().endswith(".pdf"):
        return {
            "document_id": document_id,
            "filename": upload.filename,
            "ok": False,
            "error": "Not a PDF file.",
            "facts": [],
            "failures": [
                ExtractionFailure(
                    document_id=document_id,
                    source_filename=upload.filename,
                    page_number=-1,
                    chunk_id="n/a",
                    reason="Rejected: file is not a .pdf.",
                )
            ],
            "page_count": 0,
        }

    async with semaphore:
        try:
            dest_path = await _save_upload(upload, document_id)
            facts, failures, page_count = await process_document_async(str(dest_path), upload.filename, document_id)
        except Exception as exc:  # noqa: BLE001 - last-resort net; process_document_async itself shouldn't raise
            return {
                "document_id": document_id,
                "filename": upload.filename,
                "ok": False,
                "error": str(exc),
                "facts": [],
                "failures": [
                    ExtractionFailure(
                        document_id=document_id,
                        source_filename=upload.filename,
                        page_number=-1,
                        chunk_id="n/a",
                        reason=f"Unexpected error while processing this file: {exc}",
                    )
                ],
                "page_count": 0,
            }

    return {
        "document_id": document_id,
        "filename": upload.filename,
        "ok": True,
        "error": None,
        "facts": facts,
        "failures": failures,
        "page_count": page_count,
    }


@app.post("/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Accepts any number of PDFs in one request.

    Phase 1 (concurrent): every file's save+extract runs at once, bounded
    by DOCUMENT_CONCURRENCY, via asyncio.gather - this is where the wall
    clock win is, since extraction is the slow, LLM-bound step and each
    file is otherwise independent.

    Phase 2 (sequential): facts are inserted and reconciled one document at
    a time, in upload order, against the full existing fact store (which
    now includes every prior document AND every earlier file in this same
    batch) - so uploading document #4 in a 4-file batch still correctly
    links back to facts found in documents #1-3 of that same batch. Doing
    this phase sequentially keeps the dedupe/error-handling logic exactly
    as it was before this concurrency change."""
    if not config.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured on the server.")

    document_ids = [new_id("doc") for _ in files]
    semaphore = asyncio.Semaphore(config.DOCUMENT_CONCURRENCY)
    extraction_results = await asyncio.gather(
        *(_extract_one(upload, doc_id, semaphore) for upload, doc_id in zip(files, document_ids))
    )

    summaries = []
    for result in extraction_results:
        document_id = result["document_id"]
        filename = result["filename"]
        facts: List[Fact] = result["facts"]
        failures: List[ExtractionFailure] = result["failures"]
        page_count = result["page_count"]

        if not result["ok"]:
            db.upsert_document(
                DocumentRecord(document_id=document_id, filename=filename, page_count=0, status="failed")
            )
            db.insert_failures(failures)
            summaries.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "status": "failed",
                    "error": result["error"],
                    "page_count": 0,
                    "facts_extracted": 0,
                    "extraction_failures": len(failures),
                    "new_relationships": 0,
                }
            )
            continue

        db.insert_facts(facts)
        db.insert_failures(failures)
        # A document that produced zero facts (e.g. scanned/image-only PDF)
        # is marked failed even though nothing technically raised, so the
        # UI surfaces it as an extraction failure rather than a silent
        # empty success.
        status = "indexed" if facts else "failed"
        db.upsert_document(
            DocumentRecord(
                document_id=document_id,
                filename=filename,
                page_count=page_count,
                fact_count=len(facts),
                status=status,
            )
        )

        existing_facts = db.list_facts()  # includes the ones just inserted, plus earlier files in this batch
        new_relationships = await run_reconciliation(facts, existing_facts) if facts else []

        summaries.append(
            {
                "document_id": document_id,
                "filename": filename,
                "status": status,
                "error": None if facts else "No facts could be extracted from this document.",
                "page_count": page_count,
                "facts_extracted": len(facts),
                "extraction_failures": len(failures),
                "new_relationships": len(new_relationships),
            }
        )

    return {"processed": summaries}


# ---------------------------------------------------------------------------
# Read endpoints
# ---------------------------------------------------------------------------
@app.get("/documents")
def get_documents():
    return db.list_documents()


@app.get("/facts", response_model=List[Fact])
def get_facts(document_id: Optional[str] = None, entity: Optional[str] = None):
    return db.list_facts(document_id=document_id, entity=entity)


@app.get("/facts/{fact_id}", response_model=Fact)
def get_fact(fact_id: str):
    matches = db.get_facts_by_ids([fact_id])
    if not matches:
        raise HTTPException(status_code=404, detail="Fact not found.")
    return matches[0]


@app.get("/failures")
def get_failures(document_id: Optional[str] = None):
    return db.list_failures(document_id=document_id)


@app.get("/relationships")
def get_relationships(relationship_type: Optional[str] = None):
    rels = db.list_relationships(relationship_type=relationship_type)
    # Hydrate with the actual facts so the UI doesn't need a second round trip.
    out = []
    for rel in rels:
        facts = db.get_facts_by_ids(rel.fact_ids)
        out.append({**rel.model_dump(), "facts": [f.model_dump() for f in facts]})
    return out


@app.post("/reconcile/rerun")
async def rerun_reconciliation():
    """Re-run clustering + judgment over every fact currently stored.
    Existing relationships are skipped via cluster_key dedup, so this is
    safe to call repeatedly and only produces genuinely new relationships."""
    all_facts = db.list_facts()
    new_relationships = await run_reconciliation(all_facts, all_facts)
    return {"facts_considered": len(all_facts), "new_relationships": len(new_relationships)}
