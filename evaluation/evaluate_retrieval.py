import csv
import sys
from pathlib import Path

# Make the project root importable when running this script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import (
    load_retrieval_components,
    retrieve_documents
)


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

EVALUATION_DIR = Path(__file__).resolve().parent

QUESTIONS_FILE = EVALUATION_DIR / "evaluation_questions.csv"

OUTPUT_FILE = EVALUATION_DIR / "retrieval_results.csv"


# --------------------------------------------------
# 2. Load evaluation questions
# --------------------------------------------------

def load_questions():

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation questions file not found: {QUESTIONS_FILE}"
        )

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)
        questions = list(reader)

    if not questions:
        raise ValueError("The evaluation questions file is empty.")

    return questions


# --------------------------------------------------
# 3. Evaluate retrieval
# --------------------------------------------------

def evaluate_retrieval():

    questions = load_questions()

    print("Loading retrieval components...")

    index, metadata, model = load_retrieval_components()

    output_rows = []

    for item in questions:

        question_id = item["question_id"]
        question = item["question"]

        print("\n" + "=" * 70)
        print(f"{question_id}: {question}")
        print("=" * 70)

        try:

            # Do not apply a threshold in the initial baseline run.
            documents = retrieve_documents(
                question=question,
                index=index,
                metadata=metadata,
                model=model,
                top_k=5,
                min_similarity=None
            )

            if not documents:

                print("No documents retrieved.")

                output_rows.append({
                    "question_id": question_id,
                    "question": question,
                    "expected_category": item["expected_category"],
                    "expected_evidence": item["expected_evidence"],
                    "retrieved_count": 0,
                    "retrieved_titles": "",
                    "retrieved_sections": "",
                    "retrieved_scores": "",
                    "retrieved_texts": "",
                    "manual_relevance_labels": "",
                    "notes": "No documents retrieved"
                })

                continue

            titles = []
            sections = []
            scores = []
            texts = []

            for rank, doc in enumerate(documents, start=1):

                title = doc.get("title") or "Unknown"
                section = doc.get("section_number") or "Unknown"
                score = doc.get("similarity_score", 0.0)
                text = doc.get("text") or ""

                titles.append(f"{rank}. {title}")
                sections.append(f"{rank}. {section}")
                scores.append(f"{rank}. {score:.4f}")
                texts.append(f"RESULT {rank}: {text}")

                print(f"\nResult {rank}")
                print("Title:", title)
                print("Section:", section)
                print("Similarity score:", round(score, 4))
                print("Text:", text[:700])

            output_rows.append({
                "question_id": question_id,
                "question": question,
                "expected_category": item["expected_category"],
                "expected_evidence": item["expected_evidence"],
                "retrieved_count": len(documents),
                "retrieved_titles": "\n".join(titles),
                "retrieved_sections": "\n".join(sections),
                "retrieved_scores": "\n".join(scores),
                "retrieved_texts": "\n\n".join(texts),
                "manual_relevance_labels": "",
                "notes": ""
            })

        except Exception as error:

            print("Evaluation error:", error)

            output_rows.append({
                "question_id": question_id,
                "question": question,
                "expected_category": item["expected_category"],
                "expected_evidence": item["expected_evidence"],
                "retrieved_count": 0,
                "retrieved_titles": "",
                "retrieved_sections": "",
                "retrieved_scores": "",
                "retrieved_texts": "",
                "manual_relevance_labels": "",
                "notes": f"ERROR: {error}"
            })

    # --------------------------------------------------
    # 4. Save results
    # --------------------------------------------------

    fieldnames = [
        "question_id",
        "question",
        "expected_category",
        "expected_evidence",
        "retrieved_count",
        "retrieved_titles",
        "retrieved_sections",
        "retrieved_scores",
        "retrieved_texts",
        "manual_relevance_labels",
        "notes"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(output_rows)

    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION FINISHED")
    print("=" * 70)

    print(f"Questions evaluated: {len(questions)}")
    print(f"Results saved to: {OUTPUT_FILE}")


# --------------------------------------------------
# 5. Run evaluation
# --------------------------------------------------

if __name__ == "__main__":

    evaluate_retrieval()