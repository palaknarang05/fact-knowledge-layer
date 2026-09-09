from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import List, Tuple

import pdfplumber
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from . import config
from .models import ExtractionFailure, ExtractionResult, Fact, new_id

_groq_client: AsyncOpenAI | None = None

def get_groq_client() -> AsyncOpenAI:
    global _groq_client
    if _groq_client is None:
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing from .env")
        _groq_client = AsyncOpenAI(
            api_key=config.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
    return _groq_client

@dataclass
class PageText:
    page_number: int
    text: str

@dataclass
class Chunk:
    chunk_id: str
    text: str
    pages: List[PageText]

def extract_pages(pdf_path: str) -> List[PageText]:
    pages: List[PageText] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(PageText(page_number=i, text=text))
    return pages

def chunk_pages(pages: List[PageText]) -> List[Chunk]:
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

EXTRACTION_SYSTEM_PROMPT = """You are a meticulous fact-extraction engine for a fact knowledge layer.

Given a chunk of text from a source document, extract every discrete, checkable
fact: numeric figures, dates, named entities and their roles/status, claims
about performance, appointments, or events. Do NOT invent a fixed schema of
"types of facts" - let the document's own content dictate what facts exist.

Rules:
- Every fact's `exact_quote` MUST be copied character-for-character from the provided text.
- Prefer several small, precise facts over one vague fact.
- Skip boilerplate unless it states a checkable claim.
- If something looks fact-like but is too ambiguous, add a short note to `extraction_issues` and omit it from `facts`.
- Use `extra` for any attributes that matter for this specific fact.
- Set `confidence` honestly.

You must output a valid JSON object matching this exact schema. Output ONLY
the raw JSON object - no reasoning, no explanation, no markdown code fences,
nothing before or after the JSON. If a page has no extractable facts
(cover pages, addresses, letterheads), output {"facts": [], "extraction_issues": []}.
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
      "confidence": 0.8,
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


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(4), reraise=True)
async def _call_extraction_llm(chunk_text: str, filename: str) -> ExtractionResult:
    client = get_groq_client()

    try:
        completion = await client.chat.completions.create(
            model=config.EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(chunk_text, filename)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            reasoning_effort="low",  # gpt-oss models: keeps hidden reasoning short so it's
                                      # less likely to leak non-JSON text and trip Groq's
                                      # server-side json_object validator (json_validate_failed)
        )
    except Exception as exc:
        # A page that's just boilerplate/addresses (cover pages, letterhead)
        # can legitimately produce no valid facts and trip the model's JSON
        # formatting - treat this chunk as empty rather than blowing up the
        # whole document. This IS required case #4 (a real extraction
        # failure), not a bug to paper over.
        if "json_validate_failed" in str(exc):
            return ExtractionResult(facts=[], extraction_issues=[f"Model failed to produce valid JSON for this chunk: {exc}"])
        raise

    content = completion.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty response")

    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    return ExtractionResult.model_validate_json(content)

def _normalize(s: str) -> str:
    s = re.sub(r"-\s*\n\s*", "", s) 
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def _locate_page_for_quote(quote: str, pages: List[PageText]) -> int:
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
    except Exception as exc:
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
        failures.append(ExtractionFailure(document_id=document_id, source_filename=filename, page_number=chunk.pages[0].page_number if chunk.pages else -1, chunk_id=chunk.chunk_id, reason=note))

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
            failures.append(ExtractionFailure(document_id=document_id, source_filename=filename, page_number=page_number, chunk_id=chunk.chunk_id, reason="Fact flagged ungrounded.", raw_excerpt=f.exact_quote))

    return facts, failures

async def process_document_async(pdf_path: str, filename: str, document_id: str) -> Tuple[List[Fact], List[ExtractionFailure], int]:
    try:
        pages = await asyncio.to_thread(extract_pages, pdf_path)
    except Exception as exc:
        return [], [ExtractionFailure(document_id=document_id, source_filename=filename, page_number=-1, chunk_id="n/a", reason=f"PDF error: {exc}")], 0

    if not pages:
        return [], [ExtractionFailure(document_id=document_id, source_filename=filename, page_number=-1, chunk_id="n/a", reason="No pages.")], 0

    chunks = chunk_pages(pages)
    if not chunks:
        return [], [ExtractionFailure(document_id=document_id, source_filename=filename, page_number=-1, chunk_id="n/a", reason="No text.")], len(pages)

    semaphore = asyncio.Semaphore(config.CHUNK_CONCURRENCY)

    async def _bounded(chunk: Chunk):
        async with semaphore:
            await asyncio.sleep(1.0)
            return await extract_facts_from_chunk(chunk, document_id, filename)

    chunk_results = await asyncio.gather(*(_bounded(c) for c in chunks))

    all_facts, all_failures = [], []
    for facts, failures in chunk_results:
        all_facts.extend(facts)
        all_failures.extend(failures)

    return all_facts, all_failures, len(pages)