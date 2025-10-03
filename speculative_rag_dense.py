from __future__ import annotations

import asyncio
from collections import defaultdict
from statistics import mean
from typing import Any, Protocol

import httpx
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from tiktoken import encoding_for_model, get_encoding

from vllm_client import VLLMClient


# -----------------------------
# Prompts and response schemas
# -----------------------------

# rag_drafting_prompt: str = """Response to the instruction. Also provide rationale for your response.
# ## Instruction: {instruction}

# ## Evidence: {evidence}"""

rag_drafting_prompt: str = """Response to the instruction with exact answer string (no full sentences). Then provide rationale separately.

## Instruction: {instruction}

## Evidence: {evidence}"""


class RagDraftingResponse(BaseModel):
    rationale: str = Field(description="Response rationale.")
    response: str = Field(description="Response to the instruction.")


rag_verifier_prompt: str = """## Instruction: {instruction}

## Response: {response}

## Rationale: {rationale}

Is the rationale good enough to support the answer? (Yes or No)"""


# -----------------------------
# Dense Retriever Client
# -----------------------------

class RetrievedDocument(BaseModel):
    """Document returned from dense retriever."""
    title: str
    paragraph_text: str
    score: float
    doc_embedding: list[float] | None = None

    @property
    def vector(self) -> list[float]:
        """Return embedding vector if available, otherwise use score as 1D vector."""
        if self.doc_embedding is not None:
            return self.doc_embedding
        # Fallback to score as 1D vector for clustering
        return [self.score]

    @property
    def payload(self) -> dict[str, Any]:
        """Mimic Qdrant payload structure."""
        return {"content": self.paragraph_text}


class DenseRetrieverClient:
    """Client for Dense Retriever API (FAISS/BGE)."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def search(
        self,
        query: str,
        corpus_name: str = "wiki",
        max_hits: int = 10,
        return_embeddings: bool = True,
    ) -> list[RetrievedDocument]:
        """Search documents using dense retrieval."""
        url = f"{self.base_url}/retrieve/"
        payload = {
            "retrieval_method": "vector_based_similarity_search",
            "corpus_name": corpus_name,
            "query_text": query,
            "max_hits_count": max_hits,
            "return_embeddings": return_embeddings,
        }
        logger.debug(f"Retriever POST {url} payload={payload}")
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        logger.debug(
            "Retriever response parsed. Keys={} has 'retrieval'={}",
            list(data.keys()),
            "retrieval" in data,
        )
        retrieval = data.get("retrieval", [])
        return [RetrievedDocument(**doc) for doc in retrieval]

    async def close(self):
        await self.client.aclose()


# -----------------------------
# Client Protocol
# -----------------------------

class AsyncOpenAIProtocol(Protocol):
    """Protocol for OpenAI-compatible client interface."""
    beta: Any
    chat: Any


# -----------------------------
# Core building blocks
# -----------------------------

def multi_perspective_sampling(
    k: int,
    retrieved_points: list[RetrievedDocument],
    seed: int = 1399,
    m: int = 5,
) -> list[list[str]]:
    """Cluster retrieved documents into k groups and sample M subsets with replacement.

    Args:
        k: Number of clusters
        retrieved_points: List of retrieved documents
        seed: Random seed
        m: Number of subsets to create (with replacement), M in the paper

    Returns:
        List of M document subsets, each containing k documents (one from each cluster)
    """

    logger.info(f"Finding {k} clusters from {len(retrieved_points)} documents.")

    # Extract vectors for clustering
    _vectors = np.asarray([point.vector for point in retrieved_points], dtype=float)

    # Check if we're using real embeddings or just scores
    has_embeddings = len(retrieved_points) > 0 and retrieved_points[0].doc_embedding is not None
    if has_embeddings:
        logger.info(f"Using real embeddings ({_vectors.shape[1]}D) for clustering.")
    else:
        logger.warning("No embeddings available, using scores (1D) for clustering.")

    if len(retrieved_points) < k:
        k = len(retrieved_points)
        logger.warning(f"Not enough documents for {k} clusters, using {len(retrieved_points)} instead.")

    algo = KMeans(n_clusters=k, random_state=seed, n_init=10)
    clusters: list[int] = list(algo.fit_predict(X=_vectors))

    # Cluster membership map
    cluster_dict: defaultdict[int, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        cluster_dict[cluster].append(index)
    logger.info(f"Clusters distribution: {dict(cluster_dict)}")

    # Generate M subsets with replacement
    logger.info(f"{m} document subsets will be created (with replacement).")

    np.random.seed(seed=seed)
    subsets: list[list[str]] = []
    unique_clusters = list(cluster_dict.keys())

    for _ in range(m):
        sampled_indices: list[int] = []
        for c in unique_clusters:
            chosen_idx: int = int(np.random.choice(cluster_dict[c]))
            sampled_indices.append(chosen_idx)
            # No removal - sampling with replacement

        subset_documents = [
            retrieved_points[idx].paragraph_text for idx in sampled_indices
        ]
        subsets.append(subset_documents)

    return subsets


async def rag_drafting_generator(
    client: AsyncOpenAIProtocol,
    model_name: str,
    instruction: str,
    evidence: str,
    **kwargs: Any,
) -> tuple[RagDraftingResponse, float]:
    """Generate a structured draft (rationale + response) and return (draft, p_draft)."""

    completion: Any = await client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": rag_drafting_prompt.format(
                    instruction=instruction, evidence=evidence
                ),
            }
        ],
        response_format=RagDraftingResponse,
        temperature=0.0,
        logprobs=True,
        max_tokens=16384,
        **kwargs,
    )

    # Compute p_draft
    p_draft: float = np.exp(
        mean(token.logprob for token in completion.choices[0].logprobs.content)
    )

    return completion.choices[0].message.parsed, float(p_draft)


async def rag_verifier_generator(
    client: AsyncOpenAIProtocol,
    model_name: str,
    instruction: str,
    evidence: str,
    response: str,
    rationale: str,
    **kwargs: Any,
) -> tuple[str, float]:
    """Verify the draft and return (verdict_text, p_yes)."""

    try:
        encoder = encoding_for_model(model_name)
    except Exception:
        encoder = get_encoding("cl100k_base")

    completion: Any = await client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": rag_verifier_prompt.format(
                    instruction=instruction,
                    evidence=evidence,
                    response=response,
                    rationale=rationale,
                ),
            }
        ],
        temperature=0.0,
        logprobs=True,
        max_tokens=2,
        **kwargs,
    )

    verdict: str | None = completion.choices[0].message.content
    verdict = verdict or ""

    # Check if verdict contains "yes" (case-insensitive, more robust)
    verdict_normalized = verdict.lower().strip().rstrip('.')
    cond: bool = "yes" in verdict_normalized

    # Calculate p_yes based on logprobs
    p_yes: float = (
        float(
            np.exp(mean(token.logprob for token in completion.choices[0].logprobs.content))
        )
        if cond
        else 0.0
    )

    return verdict, p_yes


# -----------------------------
# End-to-end pipeline
# -----------------------------

async def speculative_rag(
    *,
    query: str,
    corpus_name: str,
    retriever_url: str,
    k: int,
    seed: int,
    client: AsyncOpenAIProtocol,
    m_drafter: str,
    m_verifier: str,
    max_hits: int = 15,
    m: int = 5,
    verbose: bool = False,
) -> str:
    """Speculative RAG pipeline with Dense Retriever.

    Steps:
      1) Dense retrieval (FAISS/BGE)
      2) Multi-perspective sampling via KMeans (with replacement)
      3) Parallel drafting (small LM)
      4) Parallel verification (large LM)
      5) Select argmax(p_draft * p_self)

    Args:
        query: User query
        corpus_name: Corpus to search in
        retriever_url: Dense retriever API URL
        k: Number of clusters (hyperparameter)
        seed: Random seed
        client: OpenAI client (or mock)
        m_drafter: Drafter model name
        m_verifier: Verifier model name
        max_hits: Number of documents to retrieve (hyperparameter)
        m: Number of subsets to create with replacement (hyperparameter, M in the paper)
    """

    _start = asyncio.get_event_loop().time()

    # 1) Dense Retrieval
    logger.info("Fetching relevant documents from Dense Retriever...")
    if verbose:
        logger.debug(
            "query=%r corpus=%s retriever_url=%s max_hits=%d k=%d m=%d seed=%d",
            query,
            corpus_name,
            retriever_url,
            max_hits,
            k,
            m,
            seed,
        )
    _now = asyncio.get_event_loop().time()
    retriever = DenseRetrieverClient(base_url=retriever_url)

    try:
        retrieved_docs = await retriever.search(
            query=query,
            corpus_name=corpus_name,
            max_hits=max_hits,
        )
        elapsed = asyncio.get_event_loop().time() - _now
        logger.info(f"Retrieved {len(retrieved_docs)} documents in {elapsed:.4f} seconds.")
        if verbose and retrieved_docs:
            sample_preview = [
                {
                    "title": d.title,
                    "score": d.score,
                    "has_embed": d.doc_embedding is not None,
                    "snippet": (d.paragraph_text[:120] + "...") if len(d.paragraph_text) > 120 else d.paragraph_text,
                }
                for d in retrieved_docs[:min(5, len(retrieved_docs))]
            ]
            logger.debug(f"Top retrieved docs preview: {sample_preview}")
    finally:
        await retriever.close()

    if not retrieved_docs:
        logger.error("No documents retrieved!")
        return "Unable to retrieve relevant documents."

    # 2) Multi-perspective sampling
    logger.info("Doing Multi Perspective Sampling...")
    _now = asyncio.get_event_loop().time()
    sampled_docs: list[list[str]] = multi_perspective_sampling(
        k=k, retrieved_points=retrieved_docs, seed=seed, m=m
    )
    logger.info(
        f"Multi Perspective Sampling done in {asyncio.get_event_loop().time() - _now:.4f} seconds."
    )

    # 3) Drafting
    logger.info("Doing RAG Drafting...")
    _now = asyncio.get_event_loop().time()
    rag_drafts_raw = await asyncio.gather(
        *[
            rag_drafting_generator(
                client=client,
                model_name=m_drafter,
                instruction=query,
                evidence="\n".join([f"[{idx}] {doc}" for idx, doc in enumerate(subset, start=1)]),
            )
            for subset in sampled_docs
        ],
        return_exceptions=True,
    )
    rag_drafts: list[tuple[RagDraftingResponse, float]] = []
    for i, item in enumerate(rag_drafts_raw):
        if isinstance(item, Exception):
            if verbose:
                logger.opt(exception=item).error(
                    "Drafting failed for subset {}: {}", i, item
                )
            rag_drafts.append((RagDraftingResponse(rationale="", response=""), 0.0))
        else:
            rag_drafts.append(item)
    logger.info(f"RAG Drafting done in {asyncio.get_event_loop().time() - _now:.4f} seconds.")
    if verbose:
        logger.debug(
            "Drafts: {}",
            [
                {
                    "response": d.response,
                    "rationale_len": len(d.rationale),
                    "p_draft": round(p, 6),
                }
                for d, p in rag_drafts
            ],
        )

    # 4) Verification
    logger.info("Doing RAG Verification...")
    _now = asyncio.get_event_loop().time()
    rag_verifications_raw = await asyncio.gather(
        *[
            rag_verifier_generator(
                client=client,
                model_name=m_verifier,
                instruction=query,
                evidence="\n".join([f"[{idx}] {doc}" for idx, doc in enumerate(subset, start=1)]),
                response=rag_drafting_response.response,
                rationale=rag_drafting_response.rationale,
            )
            for subset, (rag_drafting_response, _) in zip(sampled_docs, rag_drafts)
        ],
        return_exceptions=True,
    )
    rag_verifications: list[tuple[str, float]] = []
    for i, item in enumerate(rag_verifications_raw):
        if isinstance(item, Exception):
            if verbose:
                logger.opt(exception=item).error(
                    "Verification failed for subset {}: {}", i, item
                )
            rag_verifications.append(("", 0.0))
        else:
            rag_verifications.append(item)
    logger.info(f"RAG Verification done in {asyncio.get_event_loop().time() - _now:.4f} seconds.")
    if verbose:
        logger.debug(
            "Verifications: {}",
            [
                {"verdict": v, "p_yes": round(p, 6)} for v, p in rag_verifications
            ],
        )

    # 5) Select best draft
    scores = [p_draft * p_self for (_, p_draft), (_, p_self) in zip(rag_drafts, rag_verifications)]
    best_idx: int = int(np.argmax(scores)) if scores else 0
    if verbose:
        logger.debug(
            "Scores={} best_idx={} best_response={!r}",
            [round(s, 6) for s in scores],
            best_idx,
            rag_drafts[best_idx][0].response if rag_drafts else None,
        )

    logger.info(f"Entire process done in {asyncio.get_event_loop().time() - _start:.4f} seconds.")
    return rag_drafts[best_idx][0].response


__all__ = [
    "RagDraftingResponse",
    "RetrievedDocument",
    "DenseRetrieverClient",
    "multi_perspective_sampling",
    "rag_drafting_generator",
    "rag_verifier_generator",
    "speculative_rag",
]
