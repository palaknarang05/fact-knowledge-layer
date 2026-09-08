"""
Pydantic schemas.

Two layers of schema live here on purpose:

1. "LLM-facing" models (FactLLM, ExtractionResult, RelationshipLLM) are the
   strict JSON contracts we hand to the model via structured outputs. They
   stay intentionally small and generic so the extractor never assumes a
   document type.
2. "Storage-facing" models (Fact, FactRelationship, Document) add the IDs,
   provenance, and bookkeeping fields the backend fills in after the LLM
   responds. These are what get persisted and served over the API.

The `extra` dict on Fact is the "schema evolves dynamically" mechanism: the
LLM can attach arbitrary additional key/value attributes it discovers in a
document (e.g. "warehouse_count", "director_status") without us ever having
to redeploy a new Pydantic model or migrate a database column.
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# LLM-facing: fact extraction
# ---------------------------------------------------------------------------
class FactLLM(BaseModel):
    """One atomic fact as the extraction model returns it, pre-storage."""

    entity: str = Field(..., description="The primary subject of the fact, e.g. 'Delhivery', 'India CPI inflation', 'Mr. Sahil Barua'.")
    metric_or_claim: str = Field(..., description="What is being stated about the entity, e.g. 'FY24 revenue from operations', 'Director since', 'appointed as CEO'.")
    value: Optional[str] = Field(None, description="The numeric or categorical value stated, kept as written (e.g. '7,225 crore', '5.1%', 'resigned').")
    unit: Optional[str] = Field(None, description="Unit or currency of the value if any, e.g. 'INR crore', '%', 'USD million'.")
    time_period: Optional[str] = Field(None, description="The period, date, or fiscal year the fact applies to, e.g. 'FY24', 'Q4 FY24', 'as of 31 March 2025'.")
    context: str = Field(..., description="One or two sentences of plain-language context making the fact self-contained without re-reading the source.")
    exact_quote: str = Field(..., description="A verbatim substring copied exactly from the source text that supports this fact. Must be copyable character-for-character from the provided chunk.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model's confidence that this fact was extracted correctly and is grounded in the quote.")
    extra: Dict[str, str] = Field(default_factory=dict, description="Optional additional attributes specific to this fact's domain that don't fit the fixed fields above.")


class ExtractionResult(BaseModel):
    """Structured-output envelope returned by the extraction LLM call."""

    facts: List[FactLLM]
    extraction_issues: List[str] = Field(
        default_factory=list,
        description="Free-text notes on anything in this chunk that looked fact-like but could not be confidently extracted (ambiguous tables, garbled OCR, unresolved references, etc.).",
    )


# ---------------------------------------------------------------------------
# Storage-facing: fact
# ---------------------------------------------------------------------------
class Fact(BaseModel):
    fact_id: str = Field(default_factory=lambda: new_id("fact"))
    document_id: str
    source_filename: str
    page_number: int
    chunk_id: str
    entity: str
    metric_or_claim: str
    value: Optional[str] = None
    unit: Optional[str] = None
    time_period: Optional[str] = None
    context: str
    exact_quote: str
    confidence: float
    grounded: bool = True  # False if exact_quote could not be verified verbatim in source
    extra: Dict[str, str] = Field(default_factory=dict)

    def canonical_text(self) -> str:
        """Text used for embedding + display: what the fact *says*."""
        bits = [self.entity, self.metric_or_claim]
        if self.value:
            bits.append(f"{self.value} {self.unit or ''}".strip())
        if self.time_period:
            bits.append(f"({self.time_period})")
        bits.append(self.context)
        return " | ".join(b for b in bits if b)


class ExtractionFailure(BaseModel):
    failure_id: str = Field(default_factory=lambda: new_id("fail"))
    document_id: str
    source_filename: str
    page_number: int
    chunk_id: str
    reason: str
    raw_excerpt: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM-facing: reconciliation
# ---------------------------------------------------------------------------
class RelationshipType(str, Enum):
    CORROBORATED = "CORROBORATED"
    CONTRADICTION = "CONTRADICTION"
    RESOLVED_BY_CONTEXT = "RESOLVED_BY_CONTEXT"
    EXTRACTION_FAILURE = "EXTRACTION_FAILURE"
    UNRELATED = "UNRELATED"  # escape hatch: embeddings put these near each other but the LLM disagrees


class RelationshipLLM(BaseModel):
    relationship_type: RelationshipType
    involved_fact_indices: List[int] = Field(
        ..., description="Indices (0-based, into the fact list given in the prompt) of the facts this judgment covers. Usually 2, but can be more."
    )
    explanation: str = Field(..., description="Plain-language reasoning citing what in each fact's context/time_period/unit/value led to this classification.")
    resolution_context: Optional[str] = Field(
        None, description="For RESOLVED_BY_CONTEXT only: the specific differentiator (e.g. 'standalone vs consolidated', 'FY23 vs FY24', 'crore vs million') that explains the apparent conflict."
    )


class ReconciliationResult(BaseModel):
    relationships: List[RelationshipLLM]


# ---------------------------------------------------------------------------
# Storage-facing: relationship + document
# ---------------------------------------------------------------------------
class FactRelationship(BaseModel):
    relationship_id: str = Field(default_factory=lambda: new_id("rel"))
    relationship_type: RelationshipType
    fact_ids: List[str]
    explanation: str
    resolution_context: Optional[str] = None
    cluster_key: str  # sorted, joined fact_ids - used to dedupe re-runs


class DocumentRecord(BaseModel):
    document_id: str = Field(default_factory=lambda: new_id("doc"))
    filename: str
    page_count: int
    fact_count: int = 0
    status: str = "processing"  # processing | indexed | failed
