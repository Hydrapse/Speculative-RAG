#!/usr/bin/env python3
"""
Evaluate Speculative RAG on multiple datasets.
Computes EM (Exact Match), F1, and Accuracy metrics.

Supported datasets: bamboogle, hotpotqa, musique, 2wikimultihopqa, nq
"""

import argparse
import sys
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from speculative_rag_dense import speculative_rag
from vllm_client import VLLMClient
from metrics.drop_answer_em_f1 import DropAnswerEmAndF1
from metrics.squad_answer_em_f1 import SquadAnswerEmF1Metric


# Dataset configurations
DATASETS = {
    "bamboogle": {
        "path": "/home/gangda/workspace/Adaptive-RAG/processed_data/bamboogle/test.jsonl",
        "corpus": "wiki",
        "question_field": "question_text",
        "answer_field": "answers_objects",
        "id_field": "question_id",
    },
    "hotpotqa": {
        "path": "/home/gangda/workspace/Adaptive-RAG/processed_data/hotpotqa/test_subsampled.jsonl",
        "corpus": "wiki",
        "question_field": "question_text",
        "answer_field": "answers_objects",
        "id_field": "question_id",
    },
    "musique": {
        "path": "/home/gangda/workspace/Adaptive-RAG/processed_data/musique/test_subsampled.jsonl",
        "corpus": "wiki",
        "question_field": "question_text",
        "answer_field": "answers_objects",
        "id_field": "question_id",
    },
    "2wikimultihopqa": {
        "path": "/home/gangda/workspace/Adaptive-RAG/processed_data/2wikimultihopqa/test_subsampled.jsonl",
        "corpus": "wiki",
        "question_field": "question_text",
        "answer_field": "answers_objects",
        "id_field": "question_id",
    },
    "nq": {
        "path": "/home/gangda/workspace/Adaptive-RAG/processed_data/nq/test_subsampled.jsonl",
        "corpus": "wiki",
        "question_field": "question_text",
        "answer_field": "answers_objects",
        "id_field": "question_id",
    },
}


def answer_extractor(potentially_cot: str) -> str:
    """Extract answer from CoT response if present."""
    if potentially_cot.startswith('"') and potentially_cot.endswith('"'):
        potentially_cot = potentially_cot[1:-1]

    cot_regex = re.compile(".* answer is:? (.*)\\.?")
    match = cot_regex.match(potentially_cot)
    if match:
        output = match.group(1)
        if output.endswith("."):
            output = output[:-1]
    else:
        output = potentially_cot

    return output


def load_dataset(data_path: str, dataset_config: dict[str, str]) -> list[dict[str, Any]]:
    """Load test data from JSONL file with dataset-specific field mappings."""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())

            # Normalize to common format
            normalized = {
                "question_id": item.get(dataset_config["id_field"], "unknown"),
                "question": item.get(dataset_config["question_field"], ""),
                "answers": item.get(dataset_config["answer_field"], [{}])[0].get("spans", []),
                "_original": item,  # Keep original for debugging
            }
            data.append(normalized)
    return data


async def evaluate_sample(
    client: VLLMClient,
    sample: dict[str, Any],
    corpus_name: str,
    dataset_name: str,
    metric: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate a single sample."""
    question = sample["question"]
    gt_answers = sample["answers"]

    try:
        # Run Speculative RAG
        if args.verbose:
            logger.debug(
                "Eval sample {} | question={!r} | gt={}",
                sample["question_id"],
                question,
                gt_answers,
            )
        prediction = await speculative_rag(
            query=question,
            corpus_name=corpus_name,
            retriever_url=args.retriever_url,
            k=args.k,
            seed=args.seed,
            client=client,  # type: ignore
            m_drafter=args.m_drafter,
            m_verifier=args.m_verifier,
            max_hits=args.max_hits,
            m=args.m,
            verbose=args.verbose,
        )
        # Extract answer if CoT format
        prediction = answer_extractor(prediction)
        if args.verbose:
            logger.debug(
                "Eval sample {} | prediction={!r}",
                sample["question_id"],
                prediction,
            )

        # Call metric with appropriate format
        if isinstance(metric, DropAnswerEmAndF1):
            # DROP metric expects: List[str] and List[List[str]]
            metric([prediction], [gt_answers])
        else:
            # Squad metric expects: str and List[str]
            metric(prediction, gt_answers)

        result = {
            "question_id": sample["question_id"],
            "question": question,
            "prediction": prediction,
            "ground_truth": gt_answers,
            "success": True,
        }

    except Exception as e:
        if args.verbose:
            logger.exception(f"Failed on {sample['question_id']}: {e}")
        else:
            logger.error(f"Failed on {sample['question_id']}: {e}")
        # Still call metric with empty prediction to count this sample
        if isinstance(metric, DropAnswerEmAndF1):
            metric([""], [gt_answers])
        else:
            metric("", gt_answers)

        result = {
            "question_id": sample["question_id"],
            "question": question,
            "prediction": "",
            "ground_truth": gt_answers,
            "success": False,
            "error": str(e),
        }

    return result


async def evaluate_dataset(
    dataset_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate a single dataset."""
    dataset_config = DATASETS[dataset_name]
    data_path = args.data_path or dataset_config["path"]
    corpus_name = args.corpus or dataset_config["corpus"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {dataset_name.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Corpus: {corpus_name}")

    # Load test data
    logger.info(f"Loading data from {data_path}")
    test_data = load_dataset(data_path, dataset_config)
    logger.info(f"Loaded {len(test_data)} test samples")

    # Limit samples if specified
    if args.limit > 0:
        test_data = test_data[:args.limit]
        logger.info(f"Limiting to first {args.limit} samples")

    # Initialize metric based on dataset
    if dataset_name in ["hotpotqa", "2wikimultihopqa", "musique", "iirc", "nq", "popqa", "bamboogle", "unknown"]:
        metric = DropAnswerEmAndF1()
    else:
        metric = SquadAnswerEmF1Metric()

    # Initialize vLLM client
    client = VLLMClient(base_url=args.vllm_url, model_name="Qwen3-32B")

    # Evaluate all samples
    results = []
    try:
        for sample in tqdm(test_data, desc=f"Evaluating {dataset_name}"):
            result = await evaluate_sample(client, sample, corpus_name, dataset_name, metric, args)
            results.append(result)

            # Log progress
            if len(results) % 10 == 0:
                current_metrics = metric.get_metric()
                logger.info(f"Progress: {len(results)}/{len(test_data)} | " +
                          " | ".join([f"{k}: {v:.3f}" for k, v in current_metrics.items() if k != 'count']))

    finally:
        await client.close()

    # Get final metrics from metric object
    final_metrics = metric.get_metric()
    total = len(results)
    successful = sum(1 for r in results if r["success"])

    # Print results
    print("\n" + "=" * 60)
    print(f"{dataset_name.upper()} Evaluation Results")
    print("=" * 60)
    print(f"Total samples: {total}")
    print(f"Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"Failed: {total - successful}")
    print(f"\nMetrics:")
    for metric_name, value in final_metrics.items():
        if metric_name != 'count':
            print(f"  {metric_name.upper():<12}: {value:.4f} ({value*100:.2f}%)")
    print("=" * 60)

    # Save results (default to logs/ directory)
    default_output_path = Path("logs") / f"{dataset_name}_results.json"
    output_file = Path(args.output) if args.output else default_output_path
    # Ensure parent directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "dataset": dataset_name,
        "args": vars(args),
        "summary": {
            "total": total,
            "successful": successful,
            **final_metrics,
        },
        "results": results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to {output_file}")

    return output_data


async def evaluate_multiple_datasets(
    dataset_names: list[str],
    args: argparse.Namespace,
) -> None:
    """Evaluate multiple datasets and generate combined report."""
    all_results = {}

    for dataset_name in dataset_names:
        result = await evaluate_dataset(dataset_name, args)
        all_results[dataset_name] = result

    # Generate combined summary
    print("\n" + "=" * 100)
    print("COMBINED RESULTS ACROSS ALL DATASETS")
    print("=" * 100)

    # Determine which metrics are available (some datasets have accuracy, some don't)
    has_accuracy = any('accuracy' in result['summary'] for result in all_results.values())

    if has_accuracy:
        print(f"{'Dataset':<20} {'Total':<10} {'Success':<10} {'EM (%)':<12} {'F1 (%)':<12} {'Acc (%)':<12}")
    else:
        print(f"{'Dataset':<20} {'Total':<10} {'Success':<10} {'EM (%)':<12} {'F1 (%)':<12}")
    print("-" * 100)

    for dataset_name, result in all_results.items():
        summary = result["summary"]
        if has_accuracy:
            acc = summary.get('accuracy', 0.0)
            print(f"{dataset_name:<20} {summary['total']:<10} {summary['successful']:<10} "
                  f"{summary['em']*100:<12.2f} {summary['f1']*100:<12.2f} {acc*100:<12.2f}")
        else:
            print(f"{dataset_name:<20} {summary['total']:<10} {summary['successful']:<10} "
                  f"{summary['em']*100:<12.2f} {summary['f1']*100:<12.2f}")

    print("=" * 100)

    # Save combined results (default to logs/ directory)
    combined_output = Path(args.combined_output)
    combined_output.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_output, 'w') as f:
        json.dump({
            "datasets": list(all_results.keys()),
            "args": vars(args),
            "results": all_results,
        }, f, indent=2)

    logger.info(f"Combined results saved to {combined_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Speculative RAG on multiple datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets:
  {', '.join(DATASETS.keys())}

Examples:
  # Evaluate single dataset
  python evaluate_rag.py --dataset bamboogle --limit 10

  # Evaluate multiple datasets
  python evaluate_rag.py --datasets bamboogle hotpotqa musique

  # Custom parameters
  python evaluate_rag.py --dataset nq --k 3 --m 5 --max-hits 20
        """
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()),
        help="Single dataset to evaluate",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=list(DATASETS.keys()),
        help="Multiple datasets to evaluate",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Override default data path for dataset",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        help="Override default corpus name for dataset",
    )

    # Server URLs
    parser.add_argument(
        "--retriever-url",
        default="http://10.136.201.128:8001",
        help="Dense Retriever URL",
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8010",
        help="vLLM server URL",
    )

    # Speculative RAG parameters
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Number of clusters",
    )
    parser.add_argument(
        "--max-hits",
        type=int,
        default=15,
        help="Number of documents to retrieve",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=5,
        help="Number of subsets (with replacement)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1399,
        help="Random seed",
    )
    parser.add_argument(
        "--m-drafter",
        default="Qwen3-32B",
        help="Drafter model",
    )
    parser.add_argument(
        "--m-verifier",
        default="Qwen3-32B",
        help="Verifier model",
    )

    # Evaluation control
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of samples per dataset (0 = all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for single dataset results",
    )
    parser.add_argument(
        "--combined-output",
        type=str,
        default=str(Path("logs") / "combined_results.json"),
        help="Output file for combined results (when using --datasets)",
    )

    args = parser.parse_args()

    # Configure logging verbosity
    if args.verbose:
        try:
            logger.remove()
        except Exception:
            pass
        logger.add(sys.stderr, level="DEBUG")

    # Validate dataset selection
    if not args.dataset and not args.datasets:
        parser.error("Must specify either --dataset or --datasets")

    if args.dataset and args.datasets:
        parser.error("Cannot specify both --dataset and --datasets")

    # Run evaluation
    if args.dataset:
        asyncio.run(evaluate_dataset(args.dataset, args))
    else:
        asyncio.run(evaluate_multiple_datasets(args.datasets, args))


if __name__ == "__main__":
    main()
