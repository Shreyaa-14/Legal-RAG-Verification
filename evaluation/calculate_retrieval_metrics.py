
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "evaluation"

INPUT_FILE = EVAL_DIR / "retrieval_results_baseline.csv"
OUTPUT_FILE = EVAL_DIR / "baseline_metrics.json"


def parse_labels(value, question_id):
    if not value or not value.strip():
        return None

    try:
        labels = [
            int(item.strip())
            for item in value.split(",")
        ]
    except ValueError:
        raise ValueError(
            f"{question_id}: Labels must contain only 0 and 1."
        )

    if len(labels) != 5:
        raise ValueError(
            f"{question_id}: Expected exactly 5 labels."
        )

    if any(label not in (0, 1) for label in labels):
        raise ValueError(
            f"{question_id}: Labels must be 0 or 1."
        )

    return labels


def precision_at_5(labels):
    return sum(labels) / 5


def reciprocal_rank(labels):
    for rank, label in enumerate(labels, start=1):
        if label == 1:
            return 1 / rank
    return 0.0


def hit_at_5(labels):
    return 1 if any(labels) else 0


def evaluate():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        newline="",
        encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    precision_scores = []
    reciprocal_scores = []
    hit_scores = []
    missing_questions = []

    print("\nBASELINE RETRIEVAL EVALUATION")
    print("=" * 55)

    for row in rows:
        question_id = row.get("question_id", "Unknown")
        labels = parse_labels(
            row.get("manual_relevance_labels", ""),
            question_id
        )

        if labels is None:
            missing_questions.append(question_id)
            continue

        precision = precision_at_5(labels)
        rr = reciprocal_rank(labels)
        hit = hit_at_5(labels)

        precision_scores.append(precision)
        reciprocal_scores.append(rr)
        hit_scores.append(hit)

        print(
            f"{question_id}: "
            f"P@5={precision:.3f}, "
            f"RR={rr:.3f}, "
            f"Hit@5={hit}"
        )

    if not precision_scores:
        raise ValueError(
            "No labeled questions found. "
            "Fill manual_relevance_labels first."
        )

    count = len(precision_scores)

    metrics = {
        "total_questions": len(rows),
        "evaluated_questions": count,
        "unlabeled_questions": missing_questions,
        "precision_at_5": sum(precision_scores) / count,
        "mrr": sum(reciprocal_scores) / count,
        "hit_rate_at_5": sum(hit_scores) / count,
        "recall_at_5": None,
        "note": (
            "Recall@5 requires an independently established "
            "set of relevant passages for each question. "
            "Metrics reflect manually labeled questions only."
        )
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    print("\n" + "=" * 55)
    print("BASELINE METRICS")
    print("=" * 55)

    print("Evaluated questions:", count)
    print("Precision@5:", round(metrics["precision_at_5"], 4))
    print("MRR:", round(metrics["mrr"], 4))
    print("Hit Rate@5:", round(metrics["hit_rate_at_5"], 4))

    if missing_questions:
        print("Unlabeled questions:", ", ".join(missing_questions))
        print("Results are preliminary until all questions are labeled.")

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    evaluate()
