"""
Answer metric -- mostly taken directly from squad_tools of allennlp.
Copied from Adaptive-RAG for consistency.
"""
import re
import string
import collections
from typing import Tuple, List
try:
    import ftfy
except ImportError:
    # Fallback if ftfy not installed
    class ftfy:
        @staticmethod
        def fix_text(s):
            return s


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# Simplified interface for our use case
def get_metrics(prediction: str, ground_truths: List[str]) -> Tuple[float, float]:
    """
    Get EM and F1 metrics using Squad evaluation (simple normalization).

    Returns:
        (em, f1) tuple
    """
    prediction = ftfy.fix_text(prediction) if isinstance(prediction, str) else prediction
    ground_truths = [ftfy.fix_text(gt) for gt in ground_truths]

    em = metric_max_over_ground_truths(compute_exact, prediction, ground_truths)
    f1 = metric_max_over_ground_truths(compute_f1, prediction, ground_truths)

    return float(em), float(f1)
