#!/usr/bin/env python3
"""
Verbose evaluation with detailed debugging output.
Shows intermediate results at each step of Speculative RAG.
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger
from evaluate_rag import (
    DATASETS,
    normalize_answer,
    accuracy,
    exact_match,
    f1_score,
    load_dataset,
)
from metrics_drop_eval import get_metrics as drop_get_metrics
from speculative_rag_dense import (
    DenseRetrieverClient,
    multi_perspective_sampling,
    rag_drafting_generator,
    rag_verifier_generator,
)
from vllm_client import VLLMClient
import numpy as np


async def evaluate_sample_verbose(
    client: VLLMClient,
    sample: dict[str, Any],
    corpus_name: str,
    dataset_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate a single sample with verbose debugging output."""
    question_id = sample["question_id"]
    question = sample["question"]
    gt_answers = sample["answers"]

    print("\n" + "=" * 80)
    print(f"[{question_id}] Question: {question}")
    print(f"Ground Truth: {gt_answers}")
    print("=" * 80)

    try:
        # Step 1: Retrieval
        print("\n[STEP 1] Dense Retrieval...")
        retriever = DenseRetrieverClient(base_url=args.retriever_url)
        try:
            retrieved_docs = await retriever.search(
                query=question,
                corpus_name=corpus_name,
                max_hits=args.max_hits,
            )
            print(f"  ✓ Retrieved {len(retrieved_docs)} documents")
            for i, doc in enumerate(retrieved_docs[:5], 1):  # Show top 5
                print(f"    [{i}] {doc.title[:60]}... (score: {doc.score:.4f})")
                print(f"        {doc.paragraph_text[:150]}...")
        finally:
            await retriever.close()

        if not retrieved_docs:
            print("  ✗ No documents retrieved!")
            return {
                "question_id": question_id,
                "question": question,
                "prediction": "No documents retrieved",
                "ground_truth": gt_answers,
                "em": 0.0,
                "f1": 0.0,
                "success": False,
            }

        # Step 2: Clustering & Sampling
        print(f"\n[STEP 2] Multi-Perspective Sampling (k={args.k}, m={args.m})...")
        sampled_docs = multi_perspective_sampling(
            k=args.k,
            retrieved_points=retrieved_docs,
            seed=args.seed,
            m=args.m,
        )
        print(f"  ✓ Created {len(sampled_docs)} document subsets")
        for i, subset in enumerate(sampled_docs, 1):
            print(f"    Subset {i}: {len(subset)} documents")

        # Step 3: Drafting
        print(f"\n[STEP 3] RAG Drafting ({len(sampled_docs)} drafts in parallel)...")
        rag_drafts = await asyncio.gather(*[
            rag_drafting_generator(
                client=client,
                model_name=args.m_drafter,
                instruction=question,
                evidence="\n".join([f"[{idx}] {doc}" for idx, doc in enumerate(subset, start=1)]),
            )
            for subset in sampled_docs
        ])

        print(f"  ✓ Generated {len(rag_drafts)} drafts")
        for i, (draft, p_draft) in enumerate(rag_drafts, 1):
            print(f"\n    Draft {i}:")
            print(f"      Response: {draft.response}")
            print(f"      Rationale: {draft.rationale[:200]}...")
            print(f"      p_draft: {p_draft:.6f}")

        # Step 4: Verification
        print(f"\n[STEP 4] RAG Verification ({len(rag_drafts)} verifications in parallel)...")
        rag_verifications = await asyncio.gather(*[
            rag_verifier_generator(
                client=client,
                model_name=args.m_verifier,
                instruction=question,
                evidence="\n".join([f"[{idx}] {doc}" for idx, doc in enumerate(subset, start=1)]),
                response=draft.response,
                rationale=draft.rationale,
            )
            for subset, (draft, _) in zip(sampled_docs, rag_drafts)
        ])

        print(f"  ✓ Completed {len(rag_verifications)} verifications")
        for i, (verdict, p_yes) in enumerate(rag_verifications, 1):
            print(f"    Verification {i}: verdict='{verdict}', p_yes={p_yes:.6f}")

        # Step 5: Selection
        print(f"\n[STEP 5] Selecting Best Draft...")
        scores = [p_draft * p_yes for (_, p_draft), (_, p_yes) in zip(rag_drafts, rag_verifications)]
        best_idx = int(np.argmax(scores)) if scores else 0

        print(f"  Scores (p_draft × p_yes):")
        for i, score in enumerate(scores, 1):
            marker = " ← SELECTED" if i == best_idx + 1 else ""
            print(f"    Draft {i}: {scores[i-1]:.6f}{marker}")

        prediction = rag_drafts[best_idx][0].response

        # Step 6: Evaluation
        print(f"\n[STEP 6] Evaluation...")
        acc = accuracy(prediction, gt_answers)
        em = exact_match(prediction, gt_answers, dataset_name)
        f1 = f1_score(prediction, gt_answers, dataset_name)

        print(f"  Prediction: {prediction}")
        print(f"  Ground Truth: {gt_answers}")
        print(f"  Normalized Prediction: {normalize_answer(prediction)}")
        print(f"  Normalized GT: {[normalize_answer(gt) for gt in gt_answers]}")
        print(f"  Accuracy: {acc:.2f} | EM: {em:.2f} | F1: {f1:.4f}")
        if acc > em:
            print(f"  ℹ️  Accuracy > EM: prediction contains correct answer but with extra text")

        result = {
            "question_id": question_id,
            "question": question,
            "prediction": prediction,
            "ground_truth": gt_answers,
            "accuracy": acc,
            "em": em,
            "f1": f1,
            "success": True,
            # Debugging info
            "debug": {
                "num_retrieved": len(retrieved_docs),
                "num_drafts": len(rag_drafts),
                "drafts": [
                    {
                        "response": draft.response,
                        "rationale": draft.rationale,
                        "p_draft": float(p_draft),
                    }
                    for draft, p_draft in rag_drafts
                ],
                "verifications": [
                    {
                        "verdict": verdict,
                        "p_yes": float(p_yes),
                    }
                    for verdict, p_yes in rag_verifications
                ],
                "scores": [float(s) for s in scores],
                "best_idx": best_idx,
            },
        }

    except Exception as e:
        logger.error(f"Failed on {question_id}: {e}")
        import traceback
        traceback.print_exc()

        result = {
            "question_id": question_id,
            "question": question,
            "prediction": "",
            "ground_truth": gt_answers,
            "em": 0.0,
            "f1": 0.0,
            "success": False,
            "error": str(e),
        }

    return result


async def main_async(args: argparse.Namespace) -> None:
    dataset_config = DATASETS[args.dataset]
    data_path = args.data_path or dataset_config["path"]
    corpus_name = args.corpus or dataset_config["corpus"]

    print("=" * 80)
    print(f"VERBOSE EVALUATION: {args.dataset.upper()}")
    print("=" * 80)
    print(f"Data: {data_path}")
    print(f"Corpus: {corpus_name}")
    print(f"Parameters: k={args.k}, m={args.m}, max_hits={args.max_hits}")
    print(f"Samples: {args.limit if args.limit > 0 else 'ALL'}")
    print("=" * 80)

    # Load data
    test_data = load_dataset(data_path, dataset_config)
    if args.limit > 0:
        test_data = test_data[:args.limit]
    print(f"\nLoaded {len(test_data)} samples\n")

    # Initialize client
    client = VLLMClient(base_url=args.vllm_url, model_name="Qwen3-32B")

    # Evaluate
    results = []
    try:
        for i, sample in enumerate(test_data, 1):
            print(f"\n{'#' * 80}")
            print(f"# Sample {i}/{len(test_data)}")
            print(f"{'#' * 80}")

            result = await evaluate_sample_verbose(client, sample, corpus_name, args.dataset, args)
            results.append(result)

            # Summary
            current_acc = sum(r["accuracy"] for r in results) / len(results)
            current_em = sum(r["em"] for r in results) / len(results)
            current_f1 = sum(r["f1"] for r in results) / len(results)
            print(f"\n>>> Running Average: Acc={current_acc:.4f}, EM={current_em:.4f}, F1={current_f1:.4f}")

    finally:
        await client.close()

    # Save results
    output_file = args.output or f"{args.dataset}_verbose_results.json"
    avg_acc = sum(r["accuracy"] for r in results) / len(results) if results else 0
    avg_em = sum(r["em"] for r in results) / len(results) if results else 0
    avg_f1 = sum(r["f1"] for r in results) / len(results) if results else 0

    with open(output_file, 'w') as f:
        json.dump({
            "dataset": args.dataset,
            "args": vars(args),
            "summary": {
                "total": len(results),
                "successful": sum(1 for r in results if r["success"]),
                "accuracy": avg_acc,
                "em": avg_em,
                "f1": avg_f1,
            },
            "results": results,
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total: {len(results)}")
    print(f"Accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    print(f"EM:       {avg_em:.4f} ({avg_em*100:.2f}%)")
    print(f"F1:       {avg_f1:.4f} ({avg_f1*100:.2f}%)")
    if avg_acc > avg_em:
        print(f"\nℹ️  Accuracy > EM: predictions contain correct answers but with extra text")
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Verbose evaluation with detailed debugging output",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASETS.keys()),
        help="Dataset to evaluate",
    )
    parser.add_argument("--data-path", type=str, help="Override dataset path")
    parser.add_argument("--corpus", type=str, help="Override corpus name")
    parser.add_argument("--retriever-url", default="http://10.136.201.128:8001")
    parser.add_argument("--vllm-url", default="http://localhost:8010")
    parser.add_argument("--k", type=int, default=2, help="Number of clusters")
    parser.add_argument("--max-hits", type=int, default=15, help="Retrieved docs")
    parser.add_argument("--m", type=int, default=5, help="Number of subsets")
    parser.add_argument("--seed", type=int, default=1399)
    parser.add_argument("--m-drafter", default="Qwen3-32B")
    parser.add_argument("--m-verifier", default="Qwen3-32B")
    parser.add_argument("--limit", type=int, default=3, help="Limit samples (default: 3)")
    parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
