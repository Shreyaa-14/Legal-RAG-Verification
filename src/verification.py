
"""
Claim-Level Legal Evidence Verification

Backend: Ollama
Model: qwen2.5:3b

Verifies generated legal claims against retrieved evidence.

Verification labels:
    supported
    contradicted
    unsupported
    insufficient_evidence

This is evidence-based verification, not a guarantee
of legal correctness.
"""

import json
import re

from ollama import Client


# ==========================================================
# 1. OLLAMA CONFIGURATION
# ==========================================================

MODEL_NAME = "qwen2.5:3b"
OLLAMA_HOST = "http://localhost:11434"

client = Client(
    host=OLLAMA_HOST,
    timeout=180.0
)

ALLOWED_STATUSES = {
    "supported",
    "contradicted",
    "unsupported",
    "insufficient_evidence"
}


# ==========================================================
# 2. VERIFICATION RULES
# ==========================================================

VERIFICATION_RULES = """
You are a careful claim-level evidence verification
system for an Indian legal information chatbot.

Your task is to identify factual and legal claims
in the generated answer and compare EACH claim
with the supplied retrieved legal evidence.

IMPORTANT:
Use ONLY the supplied evidence.
Do not use outside knowledge or assumptions.

VERIFICATION LABELS:

1. supported
The passage directly supports the claim.

2. contradicted
The passage directly conflicts with the claim.
Missing information is NOT a contradiction.

3. unsupported
The claim is not adequately established by the
retrieved passages. This does not mean it is false.

4. insufficient_evidence
The passages are incomplete, ambiguous, unclear
or otherwise insufficient for a reliable judgment.

STRICT RULES:

- Break the answer into meaningful factual claims.
- Check each factual claim separately.
- Do not invent missing legal information.
- Do not assume that the title of a document
  proves a claim.
- Do not use external legal knowledge.
- Do not label unrelated passages as supporting evidence.
- Do not infer a contradiction from missing information.
- For supported claims, provide the relevant source numbers.
- For contradicted claims, cite directly conflicting sources.
- For unsupported claims, do not invent citations.
- For insufficient evidence, explain the evidence gap.
- Source numbers must correspond to the supplied passages.
- Ignore disclaimers and conversational sentences
  unless they contain factual claims.
- If the answer contains no factual legal claims,
  return an empty claims array and explain why.
- An insufficient-evidence statement is not proof
  that no applicable law exists outside the corpus.
- Return only valid JSON.
- Never include Markdown fences.
"""


# ==========================================================
# 3. BUILD VERIFICATION PROMPT
# ==========================================================

def build_verification_prompt(answer, retrieved_docs):

    evidence_parts = []

    for index, doc in enumerate(retrieved_docs, start=1):

        title = doc.get("title") or "Unknown document"
        section = doc.get("section_number") or "Not specified"
        section_title = doc.get("section_title") or "Not specified"
        source_url = doc.get("source_url") or "Not available"
        text = doc.get("text") or ""

        evidence_parts.append(
            f"""
SOURCE {index}

Act: {title}
Section: {section}
Section title: {section_title}
Source URL: {source_url}

LEGAL TEXT:
{text}
"""
        )

    evidence_context = "\n\n".join(evidence_parts)

    prompt = f"""
{VERIFICATION_RULES}

GENERATED ANSWER:
{answer}

RETRIEVED LEGAL EVIDENCE:
{evidence_context}

Return a JSON object with exactly this structure:

{{
    "verification_message": "Brief verification summary",
    "claims": [
        {{
            "claim": "Individual factual claim",
            "status": "supported",
            "evidence": "Explanation based on supplied legal text",
            "source_numbers": [1]
        }}
    ]
}}

REQUIREMENTS:

- Use one of the four allowed statuses for every claim.
- Include relevant source numbers only.
- Never invent source numbers.
- A supported claim must have at least one source.
- A contradicted claim must have at least one source.
- Use an empty source list when no particular
  evidence passage supports a judgment.
- If there are no factual claims, return "claims": [].
"""

    return prompt


# ==========================================================
# 4. PARSE OLLAMA JSON RESPONSE
# ==========================================================

def parse_verification_response(response_text):

    if not isinstance(response_text, str):
        return None

    response_text = response_text.strip()

    if not response_text:
        return None

    # First try direct parsing.
    try:
        result = json.loads(response_text)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # Remove Markdown code fences if present.
    cleaned_text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        response_text,
        flags=re.IGNORECASE
    ).strip()

    try:
        result = json.loads(cleaned_text)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    return None


# ==========================================================
# 5. VALIDATE VERIFICATION OUTPUT
# ==========================================================

def validate_verification(result, number_of_sources):

    if not isinstance(result, dict):
        return False

    message = result.get("verification_message")

    if not isinstance(message, str) or not message.strip():
        return False

    claims = result.get("claims")

    if not isinstance(claims, list):
        return False

    for claim in claims:

        if not isinstance(claim, dict):
            return False

        claim_text = claim.get("claim")
        status = claim.get("status")
        evidence = claim.get("evidence")
        source_numbers = claim.get("source_numbers")

        if not isinstance(claim_text, str) or not claim_text.strip():
            return False

        if status not in ALLOWED_STATUSES:
            return False

        if not isinstance(evidence, str) or not evidence.strip():
            return False

        if not isinstance(source_numbers, list):
            return False

        for source_number in source_numbers:

            if type(source_number) is not int:
                return False

            if not 1 <= source_number <= number_of_sources:
                return False

        if len(source_numbers) != len(set(source_numbers)):
            return False

        # Positive support or contradiction must cite evidence.
        if status in {"supported", "contradicted"}:
            if not source_numbers:
                return False

    return True


# ==========================================================
# 6. STANDARD VERIFICATION FAILURE
# ==========================================================

def verification_failure(message):

    return {
        "verification_message": message,
        "claims": [],
        "verification_completed": False
    }


# ==========================================================
# 7. VERIFY GENERATED ANSWER
# ==========================================================

def verify_answer(answer, retrieved_docs):

    if not retrieved_docs:

        return verification_failure(
            "Verification was not performed because "
            "no legal evidence was retrieved."
        )

    if not isinstance(answer, str) or not answer.strip():

        return verification_failure(
            "Verification was not performed because "
            "the generated answer was empty."
        )

    prompt = build_verification_prompt(
        answer=answer,
        retrieved_docs=retrieved_docs
    )

    try:

        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict legal evidence verifier. "
                        "Evaluate claims using only supplied passages. "
                        "Return valid JSON only."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format="json",
            options={
                "temperature": 0.0,
                "num_predict": 1200,
                "num_ctx": 8192
            }
        )

        response_text = response.message.content

        result = parse_verification_response(
            response_text
        )

        if not validate_verification(
            result,
            len(retrieved_docs)
        ):

            print(
                "Verifier returned an invalid response:",
                str(response_text)[:1000]
            )

            return verification_failure(
                "The verification model returned an invalid "
                "or incomplete result. The generated answer "
                "has not been successfully verified."
            )

        # A completed request with no extracted claims
        # is not automatically successful verification.
        if not result["claims"]:

            return verification_failure(
                "No factual claims were identified for "
                "verification. The answer has not been "
                "established as evidence-supported."
            )

        result["verification_completed"] = True

        return result

    except Exception as error:

        print("Ollama verification error:", error)

        return verification_failure(
            "Verification could not be completed because "
            "the local Ollama verification service "
            "encountered an error. The answer remains unverified."
        )


# ==========================================================
# 8. STANDALONE TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING OLLAMA CLAIM-LEVEL LEGAL VERIFICATION")
    print("=" * 60)

    sample_answer = (
        "Members must take an oath or affirmation "
        "before taking their seats."
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

    print("Model:", MODEL_NAME)
    print("Ollama server:", OLLAMA_HOST)

    result = verify_answer(
        answer=sample_answer,
        retrieved_docs=sample_docs
    )

    print("\nVERIFICATION RESULT")
    print("-" * 60)

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )

    if result.get("verification_completed"):
        print("\nVERIFICATION TEST COMPLETED")
    else:
        print("\nVERIFICATION TEST NOT COMPLETED")
