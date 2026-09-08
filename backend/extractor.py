"""
Extraction pipeline: PDF -> pages -> chunks -> grounded Fact objects.

Design choices worth calling out:

- Chunking is page-aware. Each chunk remembers which page(s) it was built
  from, so when the LLM returns a fact we can (a) verify the exact_quote
  actually appears in the source text we sent it, and (b) attribute the
  fact to the single most likely page rather than a page range.
- Grounding is enforced in code, not just prompted for. If exact_quote
  cannot be found verbatim (allowing for whitespace/hyphenation noise) in
  the chunk we sent, the fact is kept but flagged `grounded=False` and also
  logged as an ExtractionFailure - this is required case #4 in the brief.
- Nothing here is document-specific: no filenames, section names, or
  regexes tied to Delhivery/RBI/IMF wording. The only assumptions are
  generic PDF structure (pages, text) and the LLM's own judgment of what
  counts as a fact.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import List, Tuple

import pdfplumber
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from . import config
from .models import ExtractionFailure, ExtractionResult, Fact, FactLLM, new_id

_client: AsyncOpenAI | None = None

# Caps how many chunks within ONE document are sent to the extraction LLM
# concurrently. Chunk-level extraction calls are independent (no shared
# state), so this is where most of the wall-clock win comes from for large
# PDFs - a 40-chunk document that used to run ~40 calls sequentially now
# runs in ceil(40 / concurrency) batches instead.
CHUNK_CONCURRENCY = config.CHUNK_CONCURRENCY


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    return _client


@dataclass
class PageText:
    page_number: int
    text: str


@dataclass
class Chunk:
    chunk_id: str
    text: str
    pages: List[PageText]  # pages contributing to this chunk, in order


# ---------------------------------------------------------------------------
# PDF -> pages
# ---------------------------------------------------------------------------
def extract_pages(pdf_path: str) -> List[PageText]:
    """Pull raw text per page. Tables are flattened to text via pdfplumber's
    default extraction, which is sufficient for fact-bearing sentences and
    keeps this generic across prospectuses, annual reports, and slide decks."""
    pages: List[PageText] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(PageText(page_number=i, text=text))
    return pages


# ---------------------------------------------------------------------------
# pages -> chunks
# ---------------------------------------------------------------------------
def chunk_pages(pages: List[PageText]) -> List[Chunk]:
    """Greedily pack consecutive pages into ~CHUNK_MAX_CHARS chunks with a
    small character overlap so facts that straddle a page boundary aren't
    lost. Skips pages that are empty (e.g. divider/cover pages)."""
    chunks: List[Chunk] = []
    current_pages: List[PageText] = []
    current_len = 0
    overlap_tail = ""

    def flush():
        nonlocal current_pages, current_len, overlap_tail
        if not current_pages:
            return
        text = overlap_tail + "\n".join(p.text for p in current_pages)
        chunks.append(Chunk(chunk_id=new_id("chunk"), text=text, pages=list(current_pages)))
        overlap_tail = text[-config.CHUNK_OVERLAP_CHARS:]
        current_pages = []
        current_len = 0

    for page in pages:
        if not page.text.strip():
            continue
        if current_len + len(page.text) > config.CHUNK_MAX_CHARS and current_pages:
            flush()
        current_pages.append(page)
        current_len += len(page.text)

    flush()
    return chunks


# ---------------------------------------------------------------------------
# chunk -> facts (LLM call)
# ---------------------------------------------------------------------------
EXTRACTION_SYSTEM_PROMPT = """You are a meticulous fact-extraction engine for a fact knowledge layer.

Given a chunk of text from a source document, extract every discrete, checkable
fact: numeric figures, dates, named entities and their roles/status, claims
about performance, appointments, or events. Do NOT invent a fixed schema of
"types of facts" - let the document's own content dictate what facts exist.

Rules:
- Every fact's `exact_quote` MUST be copied character-for-character from the
  provided text (you may trim surrounding words but must not paraphrase the
  quoted span itself). This is the single most important rule: a fact whose
  quote cannot be found verbatim in the source is useless for this system.
- Prefer several small, precise facts over one vague fact.
- Skip boilerplate (headers, page numbers, disclaimers) unless it states a
  checkable claim.
- If something looks fact-like but is too ambiguous, garbled, or
  table-mangled to extract confidently, do NOT guess - instead add a short
  note about it to `extraction_issues` and omit it from `facts`.
- Use `extra` for any attributes that matter for this specific fact but
  don't fit entity/metric/value/unit/time_period (e.g. currency basis,
  standalone-vs-consolidated, segment name, source table name).
- Set `confidence` honestly; a fact from a clean sentence deserves high
  confidence, a fact inferred from a messy table deserves lower confidence.

You must output a valid JSON object matching this exact schema:
{
  "facts": [
    {
      "entity": "string",
      "metric_or_claim": "string",
      "value": "string or null",
      "unit": "string or null",
      "time_period": "string or null",
      "context": "string or null",
      "exact_quote": "string",
      "confidence": 0.0 to 1.0,
      "extra": {}
    }
  ],
  "extraction_issues": ["string"]
}
"""


def _build_user_prompt(chunk_text: str, filename: str) -> str:
    return (
        f"Source document: {filename}\n\n"
        f"Text chunk:\n\"\"\"\n{chunk_text}\n\"\"\"\n\n"
        "Extract all facts from this chunk following the system instructions as JSON."
    )


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(4))
async def _call_extraction_llm(chunk_text: str, filename: str) -> ExtractionResult:
    client = get_client()
    completion = await client.chat.completions.create(
        model=config.EXTRACTION_MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(chunk_text, filename)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty response")
    return ExtractionResult.model_validate_json(content)


def _normalize(s: str) -> str:
    """Loose normalization for grounding checks: collapse whitespace and
    line-break hyphenation so a quote spanning a PDF line wrap still matches."""
    s = re.sub(r"-\s*\n\s*", "", s)  # de-hyphenate across line breaks
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def _locate_page_for_quote(quote: str, pages: List[PageText]) -> int:
    """Best-effort: find which single page within the chunk contains the
    quote; falls back to the chunk's first page if no exact match is found."""
    norm_quote = _normalize(quote)
    if norm_quote:
        for p in pages:
            if norm_quote in _normalize(p.text):
                return p.page_number
    return pages[0].page_number if pages else -1


def _is_grounded(quote: str, chunk_text: str) -> bool:
    return bool(quote.strip()) and _normalize(quote) in _normalize(chunk_text)


async def extract_facts_from_chunk(chunk: Chunk, document_id: str, filename: str) -> Tuple[List[Fact], List[ExtractionFailure]]:
    facts: List[Fact] = []
    failures: List[ExtractionFailure] = []

    try:
        result = await _call_extraction_llm(chunk.text, filename)
    except Exception as exc:  # noqa: BLE001 - surfaced as a failure record, not a crash
        failures.append(
            ExtractionFailure(
                document_id=document_id,
                source_filename=filename,
                page_number=chunk.pages[0].page_number if chunk.pages else -1,
                chunk_id=chunk.chunk_id,
                reason=f"LLM extraction call failed: {exc}",
                raw_excerpt=chunk.text[:300],
            )
        )
        return facts, failures

    for note in result.extraction_issues:
        failures.append(
            ExtractionFailure(
                document_id=document_id,
                source_filename=filename,
                page_number=chunk.pages[0].page_number if chunk.pages else -1,
                chunk_id=chunk.chunk_id,
                reason=note,
            )
        )

    for f in result.facts:
        grounded = _is_grounded(f.exact_quote, chunk.text)
        page_number = _locate_page_for_quote(f.exact_quote, chunk.pages)
        fact = Fact(
            document_id=document_id,
            source_filename=filename,
            page_number=page_number,
            chunk_id=chunk.chunk_id,
            entity=f.entity,
            metric_or_claim=f.metric_or_claim,
            value=f.value,
            unit=f.unit,
            time_period=f.time_period,
            context=f.context,
            exact_quote=f.exact_quote,
            confidence=f.confidence,
            grounded=grounded,
            extra=f.extra,
        )
        facts.append(fact)
        if not grounded:
            failures.append(
                ExtractionFailure(
                    document_id=document_id,
                    source_filename=filename,
                    page_number=page_number,
                    chunk_id=chunk.chunk_id,
                    reason=(
                        f"Fact '{fact.entity} / {fact.metric_or_claim}' kept but flagged ungrounded: "
                        "exact_quote was not found verbatim in the source chunk (likely paraphrase or "
                        "OCR/whitespace mismatch)."
                    ),
                    raw_excerpt=f.exact_quote,
                )
            )

    return facts, failures


async def process_document_async(pdf_path: str, filename: str, document_id: str) -> Tuple[List[Fact], List[ExtractionFailure], int]:
    """Full pipeline for one PDF. Returns (facts, failures, page_count).

    Deliberately never raises: an unreadable/corrupt/encrypted PDF, or a page
    that pdfplumber can't parse, must degrade to an ExtractionFailure record
    (required case #4) rather than crash the request and take the rest of a
    batch upload down with it. main.py relies on this guarantee.

    Chunks within this document are extracted CONCURRENTLY (bounded by
    CHUNK_CONCURRENCY) rather than one LLM call at a time - this is the
    main lever for "large PDFs without significant performance issues":
    a 40-chunk document now takes roughly (40 / CHUNK_CONCURRENCY) call
    latencies instead of 40."""
    try:
        pages = await asyncio.to_thread(extract_pages, pdf_path)
    except Exception as exc:  # noqa: BLE001 - any pdfplumber/PDF-structure error lands here
        failure = ExtractionFailure(
            document_id=document_id,
            source_filename=filename,
            page_number=-1,
            chunk_id="n/a",
            reason=f"PDF could not be opened or parsed: {exc}",
        )
        return [], [failure], 0

    if not pages:
        failure = ExtractionFailure(
            document_id=document_id,
            source_filename=filename,
            page_number=-1,
            chunk_id="n/a",
            reason="PDF opened but contained no pages.",
        )
        return [], [failure], 0

    chunks = chunk_pages(pages)
    if not chunks:
        failure = ExtractionFailure(
            document_id=document_id,
            source_filename=filename,
            page_number=-1,
            chunk_id="n/a",
            reason="PDF had pages but no extractable text (likely scanned/image-only pages with no OCR layer).",
        )
        return [], [failure], len(pages)

    semaphore = asyncio.Semaphore(CHUNK_CONCURRENCY)

    async def _bounded(chunk: Chunk):
        async with semaphore:
            return await extract_facts_from_chunk(chunk, document_id, filename)

    chunk_results = await asyncio.gather(*(_bounded(c) for c in chunks))

    all_facts: List[Fact] = []
    all_failures: List[ExtractionFailure] = []
    for facts, failures in chunk_results:
        all_facts.extend(facts)
        all_failures.extend(failures)

    return all_facts, all_failures, len(pages)