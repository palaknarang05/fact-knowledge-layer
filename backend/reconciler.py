"""
Reconciliation engine: embed facts, find semantically-close candidates
across documents, and ask an LLM to judge the relationship.

Pipeline per batch of new facts:
  1. Embed each new fact's canonical text, upsert into a persistent Chroma
     collection (persisted to disk => new PDFs never require rebuilding the
     index for existing documents - this is the incremental-ingestion
     brownie point).
  2. For each new fact, query Chroma for its nearest neighbours among ALL
     previously indexed facts (not just this document).
  3. Build an undirected graph over facts connected by a
     similarity-above-threshold edge, and take connected components as
     candidate clusters. This lets a single relationship call cover more
     than two corroborating facts at once, instead of just pairs.
  4. Send each cluster's fact list (with quotes + provenance) to the
     reconciliation LLM, which classifies it as CORROBORATED,
     CONTRADICTION, RESOLVED_BY_CONTEXT, UNRELATED (embedding false
     positive), or EXTRACTION_FAILURE (the LLM itself can't make sense of
     what was extracted - a second, independent failure signal beyond the
     grounding check in extractor.py).
  5. Relationships are deduped by a stable `cluster_key` (sorted fact_ids)
     so re-running reconciliation after adding new documents only proposes
     genuinely new relationships, never duplicates old ones.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Set, Tuple

import chromadb
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from . import config
from .models import Fact, FactRelationship, ReconciliationResult, RelationshipType

_chroma_client = None
_openai_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def get_collection():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _chroma_client.get_or_create_collection(
        name=config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(4))
async def embed_texts(texts: List[str]) -> List[List[float]]:
    client = get_openai_client()
    resp = await client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


# ---------------------------------------------------------------------------
# Step 1: index
# ---------------------------------------------------------------------------
async def index_facts(facts: List[Fact]) -> None:
    if not facts:
        return
    collection = get_collection()
    vectors = await embed_texts([f.canonical_text() for f in facts])
    # chromadb's client is sync/blocking; push it to a worker thread so it
    # doesn't stall the event loop while other documents' async LLM calls
    # are in flight concurrently.
    await asyncio.to_thread(
        collection.upsert,
        ids=[f.fact_id for f in facts],
        embeddings=vectors,
        metadatas=[
            {"document_id": f.document_id, "entity": f.entity, "source_filename": f.source_filename}
            for f in facts
        ],
        documents=[f.canonical_text() for f in facts],
    )


# ---------------------------------------------------------------------------
# Steps 2-3: candidate clusters via connected components
# ---------------------------------------------------------------------------
async def _find_neighbors(fact: Fact) -> List[Tuple[str, float]]:
    """Returns [(neighbour_fact_id, similarity), ...] excluding self."""
    collection = get_collection()
    vectors = await embed_texts([fact.canonical_text()])
    result = await asyncio.to_thread(
        collection.query, query_embeddings=vectors, n_results=config.CANDIDATES_PER_FACT + 1
    )
    neighbours = []
    ids = result["ids"][0]
    distances = result["distances"][0]  # cosine distance = 1 - cosine similarity
    for nid, dist in zip(ids, distances):
        if nid == fact.fact_id:
            continue
        similarity = 1 - dist
        if similarity >= config.SIMILARITY_THRESHOLD:
            neighbours.append((nid, similarity))
    return neighbours


async def build_candidate_clusters(new_facts: List[Fact], all_facts_by_id: Dict[str, Fact]) -> List[List[str]]:
    """Union-find over new_facts <-> their neighbours (new or old) to form
    connected components, capped at MAX_CLUSTER_SIZE so LLM prompts stay
    small and judgments stay focused.

    Neighbour lookups for every new fact are fired concurrently
    (asyncio.gather) since they're independent embedding + vector-search
    calls - this is the main scaling win when a document adds many facts
    at once."""
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    neighbor_lists = await asyncio.gather(*(_find_neighbors(f) for f in new_facts))

    edges_added: Set[Tuple[str, str]] = set()
    for fact, neighbours in zip(new_facts, neighbor_lists):
        parent.setdefault(fact.fact_id, fact.fact_id)
        for neighbour_id, _sim in neighbours:
            if neighbour_id not in all_facts_by_id and neighbour_id not in {f.fact_id for f in new_facts}:
                continue  # neighbour not resolvable (shouldn't happen, but stay defensive)
            key = tuple(sorted((fact.fact_id, neighbour_id)))
            if key in edges_added:
                continue
            edges_added.add(key)
            union(fact.fact_id, neighbour_id)

    groups: Dict[str, List[str]] = {}
    involved_ids = {n for edge in edges_added for n in edge}
    for fid in involved_ids:
        groups.setdefault(find(fid), []).append(fid)

    clusters = [sorted(set(members)) for members in groups.values() if len(members) >= 2]
    # cap oversized clusters by keeping the fact plus its strongest edges' partners
    capped = []
    for cluster in clusters:
        capped.append(cluster[: config.MAX_CLUSTER_SIZE])
    return capped


# ---------------------------------------------------------------------------
# Step 4: LLM judgment per cluster
# ---------------------------------------------------------------------------
RECONCILIATION_SYSTEM_PROMPT = """You are a careful fact-reconciliation judge for a fact knowledge layer.

You will be given a small group of facts that an embedding search flagged as
semantically related. Some may genuinely agree, genuinely conflict, only
appear to conflict because of missing context (different time periods,
units, currency basis, standalone vs consolidated scope, different named
sub-entities, etc.), or simply be unrelated despite surface similarity.

Before you ever label something CONTRADICTION, you MUST first check, in
this order, whether the difference is explained by:
  1. Time period - different fiscal years, quarters, or as-of dates.
  2. Unit or currency basis - crore vs million vs absolute rupees/dollars,
     percentage vs absolute, nominal vs real.
  3. Scope - standalone vs consolidated, one segment/subsidiary vs the
     whole entity, domestic vs global.
  4. Entity granularity - a named individual/subsidiary vs the parent
     organization, or two similarly-named but distinct entities.
If ANY of these differs between the facts and plausibly accounts for the
discrepancy, classify as RESOLVED_BY_CONTEXT, not CONTRADICTION - even if
neither fact explicitly says "this is why we differ from the other
source". Only use CONTRADICTION when the facts share the same time period,
unit/basis, scope, and entity, and still state incompatible values or
claims.

For each distinct relationship you find among these facts, output one entry:
- CORROBORATED: facts state the same underlying truth, even if phrased or
  rounded differently.
- CONTRADICTION: facts make incompatible claims about the same entity,
  metric, scope, unit, and time period - checked against the four factors
  above - with no evident contextual explanation.
- RESOLVED_BY_CONTEXT: facts look contradictory at first glance, but one of
  the four factors above explains it. You MUST name that specific
  differentiator in `resolution_context` (e.g. "FY23 vs FY24", "standalone
  vs consolidated", "INR crore vs USD million").
- EXTRACTION_FAILURE: the facts as given are too vague, mismatched, or
  malformed to responsibly judge - say why in `explanation`.
- UNRELATED: the embedding match was a false positive; these facts do not
  actually bear on the same real-world claim. Use this rather than forcing
  a judgment on unrelated content.

In `explanation`, always state explicitly which of the four factors you
checked and whether each matched or differed - do not skip straight to a
verdict.

Only reference facts by their given index. If a group of >2 facts contains
multiple independent relationships (e.g. facts 0+1 corroborate, fact 2 is
unrelated), return multiple entries rather than forcing one verdict.
"""


def _format_cluster_for_prompt(facts: List[Fact]) -> str:
    lines = []
    for i, f in enumerate(facts):
        lines.append(
            f"[{i}] Source: {f.source_filename} (page {f.page_number})\n"
            f"    Entity: {f.entity}\n"
            f"    Metric/claim: {f.metric_or_claim}\n"
            f"    Value: {f.value or 'n/a'} {f.unit or ''}\n"
            f"    Time period: {f.time_period or 'n/a'}\n"
            f"    Context: {f.context}\n"
            f"    Exact quote: \"{f.exact_quote}\"\n"
            f"    Extra: {f.extra or '{}'}"
        )
    return "\n\n".join(lines)


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(4))
async def _call_reconciliation_llm(facts: List[Fact]) -> ReconciliationResult:
    client = get_openai_client()
    prompt = f"Facts under review:\n\n{_format_cluster_for_prompt(facts)}\n\nJudge the relationship(s) among these facts."
    completion = await client.beta.chat.completions.parse(
        model=config.RECONCILIATION_MODEL,
        messages=[
            {"role": "system", "content": RECONCILIATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format=ReconciliationResult,
        temperature=0.0,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("LLM returned no parsed structured output")
    return parsed


async def reconcile_cluster(cluster_fact_ids: List[str], facts_by_id: Dict[str, Fact]) -> List[FactRelationship]:
    facts = [facts_by_id[fid] for fid in cluster_fact_ids if fid in facts_by_id]
    if len(facts) < 2:
        return []

    result = await _call_reconciliation_llm(facts)
    relationships: List[FactRelationship] = []
    for judgment in result.relationships:
        if judgment.relationship_type == RelationshipType.UNRELATED:
            continue  # not persisted - it's a deliberate "no relationship" signal
        involved_ids = [
            facts[i].fact_id for i in judgment.involved_fact_indices if 0 <= i < len(facts)
        ]
        if len(involved_ids) < 2:
            continue
        cluster_key = "|".join(sorted(involved_ids))
        relationships.append(
            FactRelationship(
                relationship_type=judgment.relationship_type,
                fact_ids=involved_ids,
                explanation=judgment.explanation,
                resolution_context=judgment.resolution_context,
                cluster_key=cluster_key,
            )
        )
    return relationships


# ---------------------------------------------------------------------------
# Step 5: orchestration entry point
# ---------------------------------------------------------------------------
async def run_reconciliation(new_facts: List[Fact], all_facts: List[Fact]) -> List[FactRelationship]:
    from . import db  # local import to avoid a circular import at module load

    if not new_facts:
        return []

    await index_facts(new_facts)

    all_facts_by_id = {f.fact_id: f for f in all_facts}
    for f in new_facts:
        all_facts_by_id[f.fact_id] = f

    clusters = await build_candidate_clusters(new_facts, all_facts_by_id)

    # Judge clusters concurrently (bounded) - this is independent LLM work
    # per cluster, so it's the other big lever alongside chunk-level
    # extraction concurrency. Dedupe + persistence stays sequential and
    # untouched below so the exact-once-per-cluster_key guarantee is
    # unaffected by concurrency.
    semaphore = asyncio.Semaphore(config.RECONCILE_CONCURRENCY)

    async def _bounded(cluster: List[str]) -> List[FactRelationship]:
        async with semaphore:
            return await reconcile_cluster(cluster, all_facts_by_id)

    cluster_results = await asyncio.gather(*(_bounded(c) for c in clusters))

    new_relationships: List[FactRelationship] = []
    for rels in cluster_results:
        for rel in rels:
            if db.relationship_exists(rel.cluster_key):
                continue  # already judged in a previous run - incremental, no duplicate work
            db.insert_relationship(rel)
            new_relationships.append(rel)

    return new_relationships
