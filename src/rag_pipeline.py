
"""
Indian Legal RAG Pipeline

Workflow:
1. Validate the user question.
2. Retrieve legal evidence using FAISS.
3. Generate an evidence-grounded answer using Ollama.
4. Verify individual claims using Ollama.
5. Return the answer, sources, and verification results.

This is a research prototype.
Verification does not guarantee legal correctness.
"""

from src.retrieval import (
    load_retrieval_components,
    retrieve_documents,
)

from src.generator import generate_answer
from src.verification import verify_answer


# ==========================================================
# 1. INITIALIZE THE RAG PIPELINE
# ==========================================================

def initialize_pipeline():
    """
    Load the existing FAISS index, metadata,
    and SentenceTransformer embedding model.
    """

    print("\n" + "=" * 60)
    print("INITIALIZING INDIAN LEGAL RAG PIPELINE")
    print("=" * 60)

    index, metadata, model = load_retrieval_components()

    print("FAISS vectors loaded:", index.ntotal)
    print("Metadata records loaded:", len(metadata))

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS vector count does not match metadata count."
        )

    print("Legal RAG pipeline initialized successfully.")

    return index, metadata, model


# ==========================================================
# 2. STANDARD VERIFICATION RESPONSE
# ==========================================================

def verification_not_performed(message):
    """
    Return a consistent failure / skipped-verification result.

    This keeps the structure compatible with Streamlit.
    """

    return {
        "verification_message": message,
        "claims": [],
        "verification_completed": False,
    }


# ==========================================================
# 3. STANDARD PIPELINE RESULT
# ==========================================================

def build_result(
    question,
    answer,
    sources=None,
    verification=None,
    status="unknown",
):
    """
    Return a consistent dictionary for app.py.

    Existing keys:
        question
        answer
        sources
        verification

    Additional key:
        status

    Possible statuses:
        invalid_question
        retrieval_failed
        no_evidence
        generation_failed
        verification_failed
        verification_completed
    """

    if sources is None:
        sources = []

    if verification is None:
        verification = verification_not_performed(
            "Verification was not performed."
        )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "verification": verification,
        "status": status,
    }


# ==========================================================
# 4. PROCESS A USER QUESTION
# ==========================================================

def answer_question(
    question,
    index,
    metadata,
    model,
    top_k=5,
):
    """
    Process a question through the complete RAG workflow.

    Returns a dictionary compatible with the
    existing Streamlit interface.
    """

    # ------------------------------------------------------
    # STEP 1: VALIDATE QUESTION
    # ------------------------------------------------------

    if not isinstance(question, str) or not question.strip():

        return build_result(
            question=question,
            answer="Please enter a valid, non-empty legal question.",
            sources=[],
            verification=verification_not_performed(
                "Verification was not performed because "
                "the question was invalid."
            ),
            status="invalid_question",
        )

    question = question.strip()

    if not isinstance(top_k, int) or isinstance(top_k, bool):
        top_k = 5

    top_k = max(1, min(top_k, 10))

    print("\n" + "=" * 60)
    print("PROCESSING LEGAL QUESTION")
    print("=" * 60)

    print("Question:", question)
    print("Top K:", top_k)

    # ------------------------------------------------------
    # STEP 2: RETRIEVE LEGAL EVIDENCE
    # ------------------------------------------------------

    try:

        print("\nRetrieving relevant legal passages...")

        retrieved_docs = retrieve_documents(
            question=question,
            index=index,
            metadata=metadata,
            model=model,
            top_k=top_k,
        )

        print("Passages retrieved:", len(retrieved_docs))

    except Exception as error:

        print("Retrieval error:", error)

        return build_result(
            question=question,
            answer=(
                "Legal evidence could not be retrieved "
                "because an error occurred in the retrieval "
                "system. Please try again."
            ),
            sources=[],
            verification=verification_not_performed(
                "Verification was skipped because "
                "document retrieval failed."
            ),
            status="retrieval_failed",
        )

    # ------------------------------------------------------
    # STEP 3: HANDLE NO RETRIEVED EVIDENCE
    # ------------------------------------------------------

    if not retrieved_docs:

        return build_result(
            question=question,
            answer=(
                "I could not find relevant legal evidence "
                "in the available knowledge base. "
                "I cannot provide a reliable answer "
                "from the current documents."
            ),
            sources=[],
            verification=verification_not_performed(
                "Verification was skipped because "
                "no legal evidence was retrieved."
            ),
            status="no_evidence",
        )

    # ------------------------------------------------------
    # STEP 4: GENERATE ANSWER USING OLLAMA
    # ------------------------------------------------------

    try:

        print("\nGenerating answer using Ollama...")

        answer = generate_answer(
            question=question,
            retrieved_docs=retrieved_docs,
        )

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                "The language model returned an empty answer."
            )

        answer = answer.strip()

        print("Answer generation completed.")

    except Exception as error:

        print("Ollama generation error:", error)

        return build_result(
            question=question,
            answer=(
                "Legal passages were retrieved successfully, "
                "but I could not generate an answer using "
                "the local language model. "
                "Please review the retrieved sources below."
            ),
            sources=retrieved_docs,
            verification=verification_not_performed(
                "Verification was not performed because "
                "answer generation failed."
            ),
            status="generation_failed",
        )

    # ------------------------------------------------------
    # STEP 5: VERIFY GENERATED ANSWER
    # ------------------------------------------------------

    try:

        print("\nVerifying generated legal claims...")

        verification_result = verify_answer(
            answer=answer,
            retrieved_docs=retrieved_docs,
        )

        if not isinstance(verification_result, dict):
            raise ValueError(
                "The verifier returned an invalid result."
            )

        completed = verification_result.get(
            "verification_completed",
            False,
        )

        if not completed:

            print("Claim verification was not completed.")

            return build_result(
                question=question,
                answer=answer,
                sources=retrieved_docs,
                verification=verification_result,
                status="verification_failed",
            )

        print("Claim-level verification completed.")

    except Exception as error:

        print("Claim verification error:", error)

        return build_result(
            question=question,
            answer=answer,
            sources=retrieved_docs,
            verification=verification_not_performed(
                "Verification failed because the local "
                "verification model encountered an error. "
                "The generated answer remains unverified."
            ),
            status="verification_failed",
        )

    # ------------------------------------------------------
    # STEP 6: RETURN COMPLETE RAG RESULT
    # ------------------------------------------------------

    print("\nRAG pipeline execution completed.")

    return build_result(
        question=question,
        answer=answer,
        sources=retrieved_docs,
        verification=verification_result,
        status="verification_completed",
    )


# ==========================================================
# 5. DISPLAY RETRIEVED LEGAL SOURCES
# ==========================================================

def display_sources(sources):
    """
    Print retrieved legal documents and their metadata.
    """

    print("\n" + "=" * 60)
    print("RETRIEVED LEGAL SOURCES")
    print("=" * 60)

    if not sources:
        print("No legal sources were retrieved.")
        return

    for index, source in enumerate(sources, start=1):

        print(f"\nSOURCE {index}")
        print("-" * 60)

        print("Title:", source.get("title", "Unknown"))
        print("Chapter:", source.get("chapter", "Unknown"))
        print("Section:", source.get("section_number", "Unknown"))

        print(
            "Section title:",
            source.get("section_title", "Unknown"),
        )

        score = source.get("similarity_score")

        if isinstance(score, (int, float)):
            print("Similarity score:", round(score, 4))
        else:
            print("Similarity score: Not available")

        print("Source URL:", source.get("source_url", ""))

        print("\nLegal passage:")
        print(source.get("text", ""))


# ==========================================================
# 6. DISPLAY CLAIM-LEVEL VERIFICATION
# ==========================================================

def display_verification(verification):
    """
    Print claim verification status and results.
    """

    print("\n" + "=" * 60)
    print("CLAIM-LEVEL VERIFICATION")
    print("=" * 60)

    if not isinstance(verification, dict):
        print("Invalid verification result.")
        return

    print(
        "Verification completed:",
        verification.get("verification_completed", False),
    )

    print(
        "Verification message:",
        verification.get(
            "verification_message",
            "No verification message available.",
        ),
    )

    claims = verification.get("claims", [])

    if not claims:
        print("\nNo verified claims available.")
        return

    for index, claim in enumerate(claims, start=1):

        print(f"\nCLAIM {index}")
        print("-" * 60)

        print("Claim:", claim.get("claim", ""))
        print("Status:", claim.get("status", ""))
        print("Evidence:", claim.get("evidence", ""))

        print(
            "Supporting/conflicting sources:",
            claim.get("source_numbers", []),
        )


# ==========================================================
# 7. STANDALONE PIPELINE TEST
# ==========================================================

if __name__ == "__main__":

    try:

        index, metadata, model = initialize_pipeline()

    except Exception as error:

        print("\nPipeline initialization failed.")
        print("Error:", error)

        raise SystemExit(1)

    print("\nIndian Legal RAG Verification System")
    print("Type 'exit' to stop.")

    while True:

        try:

            question = input(
                "\nEnter your legal question: "
            ).strip()

        except (KeyboardInterrupt, EOFError):

            print("\nExiting Legal RAG.")
            break

        if question.lower() in {"exit", "quit"}:
            print("Exiting Legal RAG.")
            break

        result = answer_question(
            question=question,
            index=index,
            metadata=metadata,
            model=model,
            top_k=5,
        )

        print("\n" + "=" * 60)
        print("PIPELINE STATUS")
        print("=" * 60)
        print(result["status"])

        print("\n" + "=" * 60)
        print("GENERATED ANSWER")
        print("=" * 60)
        print(result["answer"])

        display_sources(result["sources"])

        display_verification(result["verification"])
