from __future__ import annotations

import argparse
import asyncio
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List

import numpy as np

from speculative_rag import (
    RagDraftingResponse,
    default_clients,
    speculative_rag,
)


# -----------------------------
# Mocks for offline testing
# -----------------------------

class MockOpenAI:
    """Minimal AsyncOpenAI mock that supports:
    - embeddings.create
    - chat.completions.create
    - beta.chat.completions.parse
    """

    def __init__(self, seed: int = 1399, embed_dim: int = 16):
        self._rnd = random.Random(seed)
        self._np_rnd = np.random.default_rng(seed)
        self._embed_dim = embed_dim

        async def _parse_impl(**kwargs: Any):
            instruction = _extract_kw(kwargs, "messages", default=[{"content": ""}])[0][
                "content"
            ]
            # naive parse for instruction & evidence content
            rationale = "Mock rationale: evidence considered."
            response = f"Mock response to: {instruction[:48]}"
            parsed = RagDraftingResponse(rationale=rationale, response=response)
            # produce ~10 tokens with moderate confidence
            tokens = [SimpleNamespace(logprob=-0.12) for _ in range(10)]
            choice = SimpleNamespace(
                message=SimpleNamespace(parsed=parsed),
                logprobs=SimpleNamespace(content=tokens),
            )
            return SimpleNamespace(choices=[choice])

        class _Chat:
            class _Completions:
                async def create(inner_self, **kwargs: Any):
                    # dispatcher to create()
                    return await _create_impl(**kwargs)

            def __init__(inner_self):
                inner_self.completions = _Chat._Completions()

        class _Beta:
            def __init__(inner_self):
                class _ChatContainer:
                    def __init__(inner_inner_self):
                        class _CompletionsContainer:
                            async def parse(inner_self, **kwargs: Any):
                                return await _parse_impl(**kwargs)

                        inner_inner_self.completions = _CompletionsContainer()

                inner_self.chat = _ChatContainer()

        async def _create_impl(**kwargs: Any):
            # Return Yes with good confidence
            tokens = [SimpleNamespace(logprob=-0.05)]
            choice = SimpleNamespace(
                message=SimpleNamespace(content="Yes"),
                logprobs=SimpleNamespace(content=tokens),
            )
            return SimpleNamespace(choices=[choice])

        self.chat = _Chat()
        self.beta = _Beta()

    class embeddings:
        @staticmethod
        async def create(input: str, model: str):  # type: ignore[override]
            # produce fixed-size random embedding
            vec = np.random.default_rng(abs(hash((input, model))) % (2**32)).normal(0, 1, size=16).tolist()
            return SimpleNamespace(data=[SimpleNamespace(embedding=vec)])


def _extract_kw(kwargs: dict, key: str, default: Any = None):
    return kwargs.get(key, default)


@dataclass
class _MockPoint:
    vector: List[float]
    payload: dict


class MockQdrant:
    def __init__(self, seed: int = 1399, dim: int = 16, n: int = 10):
        rng = np.random.default_rng(seed)
        self._points = [
            _MockPoint(
                vector=rng.normal(0, 1, size=dim).tolist(),
                payload={
                    "content": f"Mock document {i+1}: This is synthetic evidence paragraph about topic {i%3}."
                },
            )
            for i in range(n)
        ]

    async def search(self, **kwargs: Any):  # signature-compatible
        return self._points


# -----------------------------
# CLI
# -----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Speculative RAG pipeline (live or mock)")
    p.add_argument("--query", required=True, help="User query/question")
    p.add_argument("--embedding-model", default="text-embedding-3-small")
    p.add_argument("--collection", default="speculative_rag")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--seed", type=int, default=1399)
    p.add_argument("--m-drafter", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--m-verifier", default="gpt-4o-2024-08-06")
    p.add_argument("--qdrant-path", default="qdrant_client", help="Local Qdrant path (live mode)")
    p.add_argument("--mock", action="store_true", help="Use offline mocks (no network/API)")
    return p


async def main_async(args: argparse.Namespace) -> None:
    if args.mock:
        client = MockOpenAI(seed=args.seed)
        qdrant = MockQdrant(seed=args.seed)
    else:
        client, qdrant = default_clients(args.qdrant_path)

    result = await speculative_rag(
        query=args.query,
        embedding_model=args.embedding_model,
        collection_name=args.collection,
        k=args.k,
        seed=args.seed,
        client=client,  # type: ignore[arg-type]
        qdrant_client=qdrant,  # type: ignore[arg-type]
        m_drafter=args.m_drafter,
        m_verifier=args.m_verifier,
    )

    print("\nQuestion:\n------\n" + args.query + "\n")
    print("Response:\n------\n" + result)


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
