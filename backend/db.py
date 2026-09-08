"""
Thin persistence layer over SQLite.

Why plain sqlite3 instead of an ORM: the schema is small, every table maps
1:1 onto a Pydantic model already, and keeping this file dependency-free
makes the whole project runnable with nothing but the Python stdlib for
storage. Facts and relationships round-trip through JSON columns for the
flexible bits (extra attrs, fact_id lists) so the "schema evolves
dynamically" requirement never needs a migration.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Iterator, List, Optional

from . import config
from .models import DocumentRecord, ExtractionFailure, Fact, FactRelationship, RelationshipType

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    fact_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'processing'
);

CREATE TABLE IF NOT EXISTS facts (
    fact_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    chunk_id TEXT NOT NULL,
    entity TEXT NOT NULL,
    metric_or_claim TEXT NOT NULL,
    value TEXT,
    unit TEXT,
    time_period TEXT,
    context TEXT NOT NULL,
    exact_quote TEXT NOT NULL,
    confidence REAL NOT NULL,
    grounded INTEGER NOT NULL DEFAULT 1,
    extra_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS extraction_failures (
    failure_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    chunk_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    raw_excerpt TEXT
);

CREATE TABLE IF NOT EXISTS relationships (
    relationship_id TEXT PRIMARY KEY,
    relationship_type TEXT NOT NULL,
    fact_ids_json TEXT NOT NULL,
    explanation TEXT NOT NULL,
    resolution_context TEXT,
    cluster_key TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_facts_document ON facts(document_id);
CREATE INDEX IF NOT EXISTS idx_facts_entity ON facts(entity);
"""


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)



# Documents

def upsert_document(doc: DocumentRecord) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO documents (document_id, filename, page_count, fact_count, status)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(document_id) DO UPDATE SET
                 fact_count=excluded.fact_count, status=excluded.status""",
            (doc.document_id, doc.filename, doc.page_count, doc.fact_count, doc.status),
        )


def list_documents() -> List[DocumentRecord]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY rowid DESC").fetchall()
    return [DocumentRecord(**dict(r)) for r in rows]


# Facts

def insert_facts(facts: List[Fact]) -> None:
    if not facts:
        return
    with get_conn() as conn:
        conn.executemany(
            """INSERT INTO facts (fact_id, document_id, source_filename, page_number, chunk_id,
                entity, metric_or_claim, value, unit, time_period, context, exact_quote,
                confidence, grounded, extra_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    f.fact_id, f.document_id, f.source_filename, f.page_number, f.chunk_id,
                    f.entity, f.metric_or_claim, f.value, f.unit, f.time_period, f.context,
                    f.exact_quote, f.confidence, int(f.grounded), json.dumps(f.extra),
                )
                for f in facts
            ],
        )


def _row_to_fact(r: sqlite3.Row) -> Fact:
    d = dict(r)
    d["extra"] = json.loads(d.pop("extra_json") or "{}")
    d["grounded"] = bool(d["grounded"])
    return Fact(**d)


def list_facts(document_id: Optional[str] = None, entity: Optional[str] = None) -> List[Fact]:
    query = "SELECT * FROM facts WHERE 1=1"
    params: list = []
    if document_id:
        query += " AND document_id = ?"
        params.append(document_id)
    if entity:
        query += " AND entity LIKE ?"
        params.append(f"%{entity}%")
    query += " ORDER BY rowid DESC"
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_row_to_fact(r) for r in rows]


def get_facts_by_ids(fact_ids: List[str]) -> List[Fact]:
    if not fact_ids:
        return []
    placeholders = ",".join("?" for _ in fact_ids)
    with get_conn() as conn:
        rows = conn.execute(f"SELECT * FROM facts WHERE fact_id IN ({placeholders})", fact_ids).fetchall()
    by_id = {r["fact_id"]: _row_to_fact(r) for r in rows}
    return [by_id[fid] for fid in fact_ids if fid in by_id]



# Extraction failures

def insert_failures(failures: List[ExtractionFailure]) -> None:
    if not failures:
        return
    with get_conn() as conn:
        conn.executemany(
            """INSERT INTO extraction_failures (failure_id, document_id, source_filename,
                page_number, chunk_id, reason, raw_excerpt)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (f.failure_id, f.document_id, f.source_filename, f.page_number, f.chunk_id, f.reason, f.raw_excerpt)
                for f in failures
            ],
        )


def list_failures(document_id: Optional[str] = None) -> List[ExtractionFailure]:
    query = "SELECT * FROM extraction_failures WHERE 1=1"
    params: list = []
    if document_id:
        query += " AND document_id = ?"
        params.append(document_id)
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [ExtractionFailure(**dict(r)) for r in rows]



# Relationships

def relationship_exists(cluster_key: str) -> bool:
    with get_conn() as conn:
        row = conn.execute("SELECT 1 FROM relationships WHERE cluster_key = ?", (cluster_key,)).fetchone()
    return row is not None


def insert_relationship(rel: FactRelationship) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO relationships
               (relationship_id, relationship_type, fact_ids_json, explanation, resolution_context, cluster_key)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                rel.relationship_id, rel.relationship_type.value, json.dumps(rel.fact_ids),
                rel.explanation, rel.resolution_context, rel.cluster_key,
            ),
        )


def list_relationships(relationship_type: Optional[str] = None) -> List[FactRelationship]:
    query = "SELECT * FROM relationships WHERE 1=1"
    params: list = []
    if relationship_type:
        query += " AND relationship_type = ?"
        params.append(relationship_type)
    query += " ORDER BY rowid DESC"
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["fact_ids"] = json.loads(d.pop("fact_ids_json"))
        d["relationship_type"] = RelationshipType(d["relationship_type"])
        out.append(FactRelationship(**d))
    return out
