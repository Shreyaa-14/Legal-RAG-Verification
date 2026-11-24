"""
Evaluation metrics for the Legal RAG Verification project.

This module provides:
1. Precision@k
2. Recall@k
3. Mean Reciprocal Rank (MRR) for a single query
4. Verification status counts
5. A summary of retrieval performance across multiple questions
"""


def precision_at_k(retrieved_ids, relevant_ids, k=5):
    """
    Calculate Precision@k.

    Precision@k = relevant retrieved documents / k

    retrieved_ids:
        IDs of retrieved documents, in ranked order.

    relevant_ids:
        IDs of documents judged relevant by a human evaluator.

    k:
        Number of top results to evaluate.
    """

    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    relevant_retrieved = sum(
        1 for doc_id in top_k
        if doc_id in relevant_ids
    )

    return relevant_retrieved / k


def recall_at_k(retrieved_ids, relevant_ids, k=5):
    """
    Calculate Recall@k.

    Recall@k = relevant retrieved documents / total relevant documents
    """

    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    relevant_retrieved = sum(
        1 for doc_id in top_k
        if doc_id in relevant_ids
    )

    return relevant_retrieved / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    """
    Calculate Reciprocal Rank for one query.

    If the first relevant document is ranked:
        1st -> 1.0
        2nd -> 0.5
        3rd -> 0.333...
        4th -> 0.25

    Returns 0.0 if no relevant document was retrieved.
    """

    if not relevant_ids:
        return 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1 / rank

    return 0.0


def verification_summary(claims):
    """
    Count the verification labels assigned to generated claims.

    Expected labels:
        supported
        contradicted
        unsupported
        insufficient_evidence

    The function also counts unrecognized or missing labels.
    """

    counts = {
        "supported": 0,
        "contradicted": 0,
        "unsupported": 0,
        "insufficient_evidence": 0,
        "other": 0
    }

    for claim in claims:
        status = str(claim.get("status", "")).strip().lower()

        if status in counts and status != "other":
            counts[status] += 1
        else:
            counts["other"] += 1

    total_claims = len(claims)

    counts["total_claims"] = total_claims

    if total_claims > 0:
        counts["supported_percentage"] = (
            counts["supported"] / total_claims
        ) * 100

        counts["contradicted_percentage"] = (
            counts["contradicted"] / total_claims
        ) * 100

        counts["unsupported_percentage"] = (
            counts["unsupported"] / total_claims
        ) * 100

        counts["insufficient_evidence_percentage"] = (
            counts["insufficient_evidence"] / total_claims
        ) * 100
    else:
        counts["supported_percentage"] = 0.0
        counts["contradicted_percentage"] = 0.0
        counts["unsupported_percentage"] = 0.0
        counts["insufficient_evidence_percentage"] = 0.0

    return counts


def evaluate_retrieval_dataset(records, k=5):
    """
    Calculate retrieval metrics across a collection of questions.

    Each record should be a dictionary containing:

        {
            "retrieved_ids": [...],
            "relevant_ids": [...]
        }

    Document IDs must use the same format in both lists.

    Returns average Precision@k, Recall@k, and MRR.
    """

    if not records:
        return {
            "number_of_questions": 0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0
        }

    precision_scores = []
    recall_scores = []
    reciprocal_rank_scores = []

    for record in records:
        retrieved_ids = record.get("retrieved_ids", [])
        relevant_ids = set(record.get("relevant_ids", []))

        precision_scores.append(
            precision_at_k(retrieved_ids, relevant_ids, k)
        )

        recall_scores.append(
            recall_at_k(retrieved_ids, relevant_ids, k)
        )

        reciprocal_rank_scores.append(
            reciprocal_rank(retrieved_ids, relevant_ids)
        )

    number_of_questions = len(records)

    return {
        "number_of_questions": number_of_questions,
        "precision_at_k": sum(precision_scores) / number_of_questions,
        "recall_at_k": sum(recall_scores) / number_of_questions,
        "mrr": sum(reciprocal_rank_scores) / number_of_questions
    }


# ---------------------------------------------------------
# Simple standalone test
# Run this file directly to test the metric functions.
# ---------------------------------------------------------

if __name__ == "__main__":

    retrieved = ["chunk_12", "chunk_7", "chunk_3", "chunk_9", "chunk_2"]
    relevant = {"chunk_7", "chunk_9"}

    print("Precision@5:", precision_at_k(retrieved, relevant, k=5))
    print("Recall@5:", recall_at_k(retrieved, relevant, k=5))
    print("Reciprocal Rank:", reciprocal_rank(retrieved, relevant))

    sample_claims = [
        {"status": "supported"},
        {"status": "unsupported"},
        {"status": "supported"},
        {"status": "insufficient_evidence"}
    ]

    print("\nVerification summary:")
    print(verification_summary(sample_claims))