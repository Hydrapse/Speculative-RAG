from __future__ import annotations

import asyncio
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np
# Optional logging fallback
try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover - fallback logger
    class _Logger:
        def info(self, msg: str, **kwargs: Any) -> None:
            try:
                print(msg.format(**kwargs))
            except Exception:
                print(msg)

    logger = _Logger()  # type: ignore

# Optional pydantic fallback
try:
    from pydantic import BaseModel, Field  # type: ignore
except Exception:  # pragma: no cover - lightweight fallback
    from dataclasses import dataclass as _dataclass

    def Field(*args: Any, **kwargs: Any):  # type: ignore
        return None

    class BaseModel:  # type: ignore
        pass

    _USE_PYDANTIC_FALLBACK = True
else:
    _USE_PYDANTIC_FALLBACK = False

# Optional tiktoken fallback
try:
    from tiktoken import encoding_for_model, get_encoding  # type: ignore
except Exception:  # pragma: no cover - fallback to string comparison
    encoding_for_model = None  # type: ignore
    get_encoding = None  # type: ignore


# -----------------------------
# Prompts and response schemas
# -----------------------------

rag_drafting_prompt: str = """Response to the instruction. Also provide rationale for your response.
## Instruction: {instruction}

## Evidence: {evidence}"""


if _USE_PYDANTIC_FALLBACK:
    # dataclass fallback for environments without pydantic
    from dataclasses import dataclass

    @dataclass
    class RagDraftingResponse(BaseModel):  # type: ignore
        rationale: str
        response: str
else:
    class RagDraftingResponse(BaseModel):  # type: ignore
        rationale: str = Field(description="Response rationale.")
        response: str = Field(description="Response to the instruction.")


rag_verifier_prompt: str = """## Instruction: {instruction}

## Response: {response} \

## Rationale: {rationale}

Is the rationale good enough to support the answer? (Yes or No)"""


# -----------------------------
# Core building blocks
# -----------------------------

def multi_perspective_sampling(
    k: int,
    retrieved_points: Sequence[Any],
    seed: int = 1399,
) -> list[list[str]]:
    """Cluster retrieved vectors into k groups and sample M subsets without replacement.

    Returns a list of M document subsets; each subset is a list of document contents (strings),
    one sampled from each cluster.
    """

    logger.info("Finding {k} clusters.", k=k)
    _vectors = np.asarray([point.vector for point in retrieved_points], dtype=float)

    # Try sklearn if available; else use a simple KMeans fallback
    try:  # pragma: no cover - optional dependency
        from sklearn.cluster import KMeans  # type: ignore

        algo = KMeans(n_clusters=k, random_state=seed)
        clusters: list[int] = list(algo.fit_predict(X=_vectors))
    except Exception:
        # Lightweight KMeans fallback (few iterations of Lloyd's algorithm)
        rng = np.random.default_rng(seed)
        n, d = _vectors.shape
        if n < k:
            # degenerate case: assign sequentially
            clusters = [i % k for i in range(n)]
        else:
            centers = _vectors[rng.choice(n, size=k, replace=False)]
            for _ in range(5):
                # assign
                dists = ((
                    ((_vectors[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                ))  # (n, k)
                labels = dists.argmin(axis=1)
                # update
                new_centers = []
                for ci in range(k):
                    mask = labels == ci
                    if not mask.any():
                        new_centers.append(centers[ci])
                    else:
                        new_centers.append(_vectors[mask].mean(axis=0))
                centers = np.vstack(new_centers)
            clusters = labels.tolist()

    # Cluster membership map
    cluster_dict: defaultdict[int, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        cluster_dict[cluster].append(index)
    logger.info("Clusters distribution: {dist}", dist=dict(cluster_dict))

    # Determine M (min cluster size)
    m: int = min(len(indices) for indices in cluster_dict.values())
    logger.info("{m} document subsets will be created.", m=m)

    # Generate M unique subsets without replacement
    np.random.seed(seed=seed)
    subsets: list[list[str]] = []
    unique_clusters = list(cluster_dict.keys())

    for _ in range(m):
        sampled_indices: list[int] = []
        for c in unique_clusters:
            chosen_idx: int = int(np.random.choice(cluster_dict[c]))
            sampled_indices.append(chosen_idx)
            cluster_dict[c].remove(chosen_idx)

        subset_documents = [
            retrieved_points[idx].payload.get("content") for idx in sampled_indices
        ]
        subsets.append(subset_documents)

    return subsets


async def rag_drafting_generator(
    client: AsyncOpenAI,
    model_name: str,
    instruction: str,
    evidence: str,
    **kwargs: Any,
) -> tuple[RagDraftingResponse, float]:
    """Generate a structured draft (rationale + response) and return (draft, p_draft).

    p_draft is computed as exp(mean(logprob)) across output tokens (naive approximation).
    """

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
        max_tokens=512,
        **kwargs,
    )

    # Compute p_draft
    p_draft: float = np.exp(
        mean(token.logprob for token in completion.choices[0].logprobs.content)
    )

    return completion.choices[0].message.parsed, float(p_draft)


async def rag_verifier_generator(
    client: AsyncOpenAI,
    model_name: str,
    instruction: str,
    evidence: str,
    response: str,
    rationale: str,
    **kwargs: Any,
) -> tuple[str, float]:
    """Verify the draft and return (verdict_text, p_yes).

    Verdict expected to be "Yes"/"No". p_yes is set to exp(mean(logprob)) when verdict == Yes, else 0.0.
    This mirrors the simplified logic used in the original notebook.
    """

    # Prepare tokenizer if available; otherwise use string-match fallback
    encoder = None
    if encoding_for_model and get_encoding:
        try:
            encoder = encoding_for_model(model_name)  # type: ignore[misc]
        except Exception:
            try:
                encoder = get_encoding("cl100k_base")  # type: ignore[misc]
            except Exception:
                encoder = None
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
    verdict = (verdict or "").strip().lower()
    if encoder is not None:  # token-based equality
        cond = encoder.encode(text=verdict) == encoder.encode(text="yes")  # type: ignore[operator]
    else:  # string fallback
        cond = verdict in {"yes", "yes."}
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
    embedding_model: str,
    collection_name: str,
    k: int,
    seed: int,
    client: AsyncOpenAI,
    qdrant_client: AsyncQdrantClient,
    m_drafter: str,
    m_verifier: str,
) -> str:
    """Speculative RAG pipeline.

    Steps:
      1) Embed query -> vector
      2) Qdrant vector search
      3) Multi-perspective sampling via KMeans
      4) Parallel drafting (small LM)
      5) Parallel verification (large LM)
      6) Select argmax(p_draft * p_self)
    """

    _start = asyncio.get_event_loop().time()

    # 1) Query embedding
    logger.info("Generating query vector...")
    _now = asyncio.get_event_loop().time()
    query_vector_obj = await client.embeddings.create(input=query, model=embedding_model)
    query_vector: list[float] = query_vector_obj.data[0].embedding
    logger.info("Query vector generated in {s:.4f} seconds.", s=asyncio.get_event_loop().time() - _now)

    # 2) Vector search
    logger.info("Fetching relevant documents...")
    _now = asyncio.get_event_loop().time()
    out: list[models.ScoredPoint] = await qdrant_client.search(
        collection_name=collection_name, query_vector=query_vector, with_vectors=True
    )
    logger.info("Documents retrieved in {s:.4f} seconds.", s=asyncio.get_event_loop().time() - _now)

    # 3) Multi-perspective sampling
    logger.info("Doing Multi Perspective Sampling...")
    _now = asyncio.get_event_loop().time()
    sampled_docs: list[list[str]] = multi_perspective_sampling(
        k=k, retrieved_points=out, seed=seed
    )
    logger.info(
        "Multi Perspective Sampling done in {s:.4f} seconds.", s=asyncio.get_event_loop().time() - _now
    )

    # 4) Drafting
    logger.info("Doing RAG Drafting...")
    _now = asyncio.get_event_loop().time()
    rag_drafts: list[tuple[RagDraftingResponse, float]] = await asyncio.gather(
        *[
            rag_drafting_generator(
                client=client,
                model_name=m_drafter,
                instruction=query,
                evidence="\n".join([f"[{idx}] {doc}" for idx, doc in enumerate(subset, start=1)]),
            )
            for subset in sampled_docs
        ]
    )
    logger.info("RAG Drafting done in {s:.4f} seconds.", s=asyncio.get_event_loop().time() - _now)

    # 5) Verification
    logger.info("Doing RAG Verification...")
    _now = asyncio.get_event_loop().time()
    rag_verifications: list[tuple[str, float]] = await asyncio.gather(
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
        ]
    )
    logger.info("RAG Verification done in {s:.4f} seconds.", s=asyncio.get_event_loop().time() - _now)

    # 6) Select best draft
    scores = [p_draft * p_self for (_, p_draft), (_, p_self) in zip(rag_drafts, rag_verifications)]
    best_idx: int = int(np.argmax(scores)) if scores else 0

    logger.info("Entire process done in {s:.4f} seconds.", s=asyncio.get_event_loop().time() - _start)
    return rag_drafts[best_idx][0].response


# -----------------------------
# Convenience factory (optional)
# -----------------------------

def default_clients(qdrant_path: str | Path = "qdrant_client") -> tuple[AsyncOpenAI, AsyncQdrantClient]:
    """Return default AsyncOpenAI and local AsyncQdrantClient (disk path) clients."""
    return AsyncOpenAI(), AsyncQdrantClient(path=Path(qdrant_path))


__all__ = [
    "RagDraftingResponse",
    "multi_perspective_sampling",
    "rag_drafting_generator",
    "rag_verifier_generator",
    "speculative_rag",
    "default_clients",
]
