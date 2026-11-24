import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTOR_DIR = PROJECT_ROOT / "vector_store"

INDEX_FILE = VECTOR_DIR / "legal_index.faiss"

METADATA_FILE = VECTOR_DIR / "legal_metadata.json"


# --------------------------------------------------
# 2. Embedding model
# --------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# 3. Load index, metadata, and embedding model
# --------------------------------------------------

def load_retrieval_components():

    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_FILE}\n"
            "Run embeddings.py first."
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}\n"
            "Run embeddings.py first."
        )

    print("Loading FAISS index...")

    index = faiss.read_index(str(INDEX_FILE))

    print("Loading legal metadata...")

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if index.ntotal != len(metadata):
        raise ValueError(
            "The number of vectors does not match the number "
            "of metadata records."
        )

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Retrieval components loaded successfully.")

    return index, metadata, model


# --------------------------------------------------
# 4. Retrieve relevant legal passages
# --------------------------------------------------
# --------------------------------------------------
# 4. Retrieve relevant legal passages
# --------------------------------------------------

def retrieve_documents(
    question,
    index,
    metadata,
    model,
    top_k=5,
    min_similarity=None
):
    """
    Retrieve legal passages for a user's question.

    Args:
        question: User's legal question.
        index: FAISS index.
        metadata: Legal document metadata.
        model: SentenceTransformer embedding model.
        top_k: Maximum number of passages to retrieve.
        min_similarity: Optional minimum FAISS score.
                        Leave as None until calibrated.

    Returns:
        List of retrieved legal passages.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please enter a non-empty question.")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    if min_similarity is not None and not isinstance(
        min_similarity, (int, float)
    ):
        raise ValueError("min_similarity must be a number or None.")

    if index.ntotal == 0:
        return []

    top_k = min(top_k, index.ntotal)

    # Encode the user's question.
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype=np.float32
    )

    # Search the FAISS index.
    scores, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        score = float(score)

        # Optional threshold filtering.
        if (
            min_similarity is not None
            and score < min_similarity
        ):
            continue

        record = metadata[int(idx)]

        results.append({
            "similarity_score": score,
            "title": record.get("title"),
            "chapter": record.get("chapter"),
            "section_number": record.get("section_number"),
            "section_title": record.get("section_title"),
            "text": record.get("text"),
            "source_url": record.get("source_url"),
            "act_id": record.get("act_id"),
            "chunk_id": record.get("chunk_id")
        })

    return results


# --------------------------------------------------
# 5. Display retrieved passages
# --------------------------------------------------

def display_results(question, results):

    print("\n" + "=" * 70)
    print("USER QUESTION")
    print("=" * 70)

    print(question)

    if not results:
        print("\nNo results were retrieved.")
        return

    print("\n" + "=" * 70)
    print("RETRIEVED LEGAL PASSAGES")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):

        print(f"\nResult {rank}")
        print("-" * 70)

        print("Similarity score:", round(result["similarity_score"], 4))
        print("Title:", result["title"])
        print("Chapter:", result["chapter"])
        print("Section:", result["section_number"])
        print("Section title:", result["section_title"])

        print("\nLegal text:")
        print(result["text"])

        print("\nSource URL:")
        print(result["source_url"])


# --------------------------------------------------
# 6. Main program
# --------------------------------------------------

if __name__ == "__main__":

    index, metadata, model = load_retrieval_components()

    print("\nIndian Legal Document Retrieval System")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nEnter your legal question: ").strip()

        if question.lower() == "exit":
            print("Exiting retrieval system.")
            break

        try:

            results = retrieve_documents(
                question=question,
                index=index,
                metadata=metadata,
                model=model,
                top_k=5
            )

            display_results(question, results)

        except ValueError as error:
            print("Input error:", error)