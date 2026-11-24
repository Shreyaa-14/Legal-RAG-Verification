
"""
Legal RAG Answer Generator
Uses Ollama + Qwen2.5 3B for evidence-grounded legal answers.
"""

from ollama import Client, ResponseError


# ==========================================================
# 1. OLLAMA CONFIGURATION
# ==========================================================

MODEL_NAME = "qwen2.5:3b"
OLLAMA_HOST = "http://localhost:11434"

client = Client(
    host=OLLAMA_HOST,
    timeout=120.0
)


# ==========================================================
# 2. BUILD EVIDENCE-GROUNDED PROMPT
# ==========================================================

def build_prompt(question, retrieved_docs):
    """
    Build a prompt that instructs the model to answer
    using only the supplied Indian legal passages.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please provide a valid legal question.")

    if not retrieved_docs:
        return f"""
You are an Indian legal information assistant.

USER QUESTION:
{question}

No legal evidence was retrieved.

You must state:
"I cannot answer this question reliably because the
available legal documents do not provide sufficient evidence."

Do not invent legal facts, sections, or citations.
"""

    context_parts = []

    for i, doc in enumerate(retrieved_docs, start=1):

        title = doc.get("title") or "Unknown legal document"
        section = doc.get("section_number") or "Not specified"
        section_title = doc.get("section_title") or "Not specified"
        text = doc.get("text") or ""
        source_url = doc.get("source_url") or "Not available"

        context_parts.append(
            f"""
SOURCE {i}
Act: {title}
Section: {section}
Section title: {section_title}
Source URL: {source_url}

LEGAL PASSAGE:
{text}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a careful Indian legal information assistant.

Your task is to answer the user's question using ONLY
the retrieved legal evidence provided below.

STRICT RULES:

1. Use only facts explicitly supported by the supplied passages.
2. Never invent acts, sections, legal provisions, quotations,
   penalties, rights, or citations.
3. Before answering, determine whether the passages are
   actually relevant to the question.
4. If the evidence is unrelated, incomplete, or insufficient,
   clearly say that the available legal evidence is insufficient.
5. Do not answer from your own legal knowledge.
6. Do not assume a passage is relevant merely because it has
   a high similarity score.
7. Cite supporting sources using [Source 1], [Source 2], etc.
8. Mention the Act name and section number when supported.
9. Do not cite sources that do not support the answer.
10. Do not give definitive legal advice.

RETRIEVED LEGAL EVIDENCE:

{context}

USER QUESTION:
{question}

RESPONSE FORMAT:

Answer:
A short, clear answer supported by the evidence.
If insufficient, explicitly state this instead.

Supporting sources:
List only sources that genuinely support the answer,
or write "No sufficient supporting evidence".

Disclaimer:
This is general legal information, not professional legal advice.
"""

    return prompt


# ==========================================================
# 3. GENERATE LEGAL ANSWER
# ==========================================================

def generate_answer(question, retrieved_docs):
    """
    Generate an answer using Ollama.

    Returns:
        str: Generated legal answer.

    Raises:
        RuntimeError: If Ollama or the model fails.

    The RAG pipeline handles these errors.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if not retrieved_docs:
        return (
            "I cannot answer this question reliably because "
            "no relevant legal evidence was retrieved from "
            "the available knowledge base."
        )

    prompt = build_prompt(question, retrieved_docs)

    try:

        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an Indian legal information assistant. "
                        "Answer only from supplied evidence. "
                        "If evidence is insufficient or unrelated, "
                        "say so. Never invent legal citations."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.1,
                "num_predict": 700,
                "num_ctx": 4096
            }
        )

        answer = response.message.content

        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        return answer.strip()

    except ResponseError as error:

        raise RuntimeError(
            f"Ollama model error: {error}"
        ) from error

    except Exception as error:

        raise RuntimeError(
            f"Legal answer generation failed: {error}"
        ) from error


# ==========================================================
# 4. STANDALONE TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING OLLAMA LEGAL ANSWER GENERATOR")
    print("=" * 60)

    question = (
        "What must a member do before taking their seat?"
    )

    sample_docs = [
        {
            "title": "Sample Legal Document",
            "section_number": "9",
            "section_title": "Oath or Affirmation",
            "text": (
                "Members must take an oath or affirmation "
                "before taking their seats."
            ),
            "source_url": "https://example.org/sample"
        }
    ]

    print("\nModel:", MODEL_NAME)
    print("Ollama server:", OLLAMA_HOST)

    try:

        answer = generate_answer(
            question=question,
            retrieved_docs=sample_docs
        )

        print("\nGENERATED ANSWER")
        print("-" * 60)
        print(answer)

        print("\nGENERATOR TEST PASSED")

    except Exception as error:

        print("\nGENERATOR TEST FAILED")
        print("Error:", error)
        print("\nCheck:")
        print("1. Ollama is installed and running.")
        print("2. The qwen2.5:3b model is downloaded.")
        print("3. The Ollama Python package is installed.")
