import os
from pathlib import Path

import streamlit as st


# --------------------------------------------------
# 1. Configure Hugging Face authentication
# --------------------------------------------------

# First, check for an environment variable.
hf_token = os.getenv("HF_TOKEN")

# If a local Streamlit secrets file exists, read the token.
secrets_file = Path(".streamlit") / "secrets.toml"

if secrets_file.exists():
    try:
        hf_token = hf_token or st.secrets.get("HF_TOKEN")
    except Exception:
        pass

# Make the token available to the Hugging Face client.
if hf_token:
    os.environ["HF_TOKEN"] = hf_token


# --------------------------------------------------
# 2. Import the RAG pipeline
# --------------------------------------------------

from src.rag_pipeline import (
    initialize_pipeline,
    answer_question
)


# --------------------------------------------------
# 3. Configure the Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="Legal RAG Assistant",
    page_icon="⚖️",
    layout="wide"
)


# --------------------------------------------------
# 4. Application heading
# --------------------------------------------------

st.title("Legal RAG Assistant")

st.write(
    "Ask questions about the legal documents in our "
    "knowledge base. The system retrieves relevant "
    "passages and uses a language model to generate "
    "an evidence-based response."
)

st.info(
    "This is an academic prototype for legal information. "
    "It is not a substitute for advice from a qualified "
    "legal professional. Always check the original legal source."
)


# --------------------------------------------------
# 5. Load the RAG components
# --------------------------------------------------

@st.cache_resource
def load_pipeline():
    return initialize_pipeline()


try:
    with st.spinner("Loading the legal knowledge base..."):
        index, metadata, embedding_model = load_pipeline()

    st.success(
        f"Knowledge base loaded: {len(metadata)} legal records."
    )

except Exception as error:
    st.error(
        "The knowledge base could not be loaded. "
        "Check that the vector index and metadata files exist."
    )

    st.exception(error)
    st.stop()


# --------------------------------------------------
# 6. Functions for displaying sources
# --------------------------------------------------

def display_sources(sources):
    """Display the legal passages retrieved for an answer."""

    if not sources:
        st.warning("No legal sources were retrieved.")
        return

    with st.expander("View retrieved legal sources"):

        for i, source in enumerate(sources, start=1):

            st.markdown(f"### Source {i}")

            st.write(
                "**Title:**",
                source.get("title", "Not available")
            )

            st.write(
                "**Section:**",
                source.get("section_number", "Not available")
            )

            st.write(
                "**Section title:**",
                source.get("section_title", "Not available")
            )

            st.write(
                "**Similarity score:**",
                round(source.get("similarity_score", 0), 4)
            )

            st.write(
                "**Legal passage:**",
                source.get("text", "Not available")
            )

            source_url = source.get("source_url")

            if source_url and source_url != "Not available":
                st.markdown(
                    f"[Open original source]({source_url})"
                )

            st.divider()


# --------------------------------------------------
# 7. Function for displaying claim-level verification
# --------------------------------------------------

def display_verification(verification):
    """Display claim-level verification results, if available."""

    st.subheader("Claim-Level Verification")

    # Handle missing or empty verification results.
    if not verification:
        st.warning(
            "Claim-level verification results are not available. "
            "The generated answer should not be treated as verified."
        )
        return

    verification_message = verification.get(
        "verification_message",
        "No verification summary is available."
    )

    st.info(verification_message)

    claims = verification.get("claims", [])

    if not claims:
        st.warning(
            "No individual claim verification results were returned. "
            "Do not treat the generated answer as verified."
        )
        return

    # Display each verified claim.
    for i, claim in enumerate(claims, start=1):

        status = claim.get("status", "unknown")
        formatted_status = status.replace("_", " ").title()

        with st.expander(f"Claim {i}: {formatted_status}"):

            st.write("**Claim:**")
            st.write(claim.get("claim", "Not available"))

            st.write("**Verification status:**")
            st.write(formatted_status)

            st.write("**Evidence explanation:**")
            st.write(claim.get("evidence", "Not available"))

            st.write("**Retrieved source numbers:**")
            st.write(claim.get("source_numbers", []))


# --------------------------------------------------
# 8. Sidebar settings
# --------------------------------------------------

with st.sidebar:

    st.header("Settings")

    top_k = st.slider(
        "Number of passages to retrieve",
        min_value=1,
        max_value=10,
        value=5
    )

    st.caption(
        "More passages may provide additional context, "
        "but can also introduce irrelevant information."
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# 9. Initialize chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# 10. Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display sources for previous assistant messages.
        if message["role"] == "assistant":

            sources = message.get("sources", [])
            verification = message.get("verification", {})

            if sources:
                display_sources(sources)

            # Display verification results if they were saved.
            display_verification(verification)


# --------------------------------------------------
# 11. Accept a new question
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about the legal documents..."
)


# --------------------------------------------------
# 12. Process the question
# --------------------------------------------------

if question:

    # Save and display the user's question.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Generate the assistant's response.
    with st.chat_message("assistant"):

        with st.spinner(
            "Retrieving legal evidence and generating an answer..."
        ):

            try:

                # Run the RAG pipeline.
                result = answer_question(
                    question=question,
                    index=index,
                    metadata=metadata,
                    model=embedding_model,
                    top_k=top_k
                )

                # Extract the returned information.
                answer = result.get(
                    "answer",
                    "Sorry, no answer was returned."
                )

                sources = result.get("sources", [])

                # Verification may be absent until the verification
                # stage is connected to the RAG pipeline.
                verification = result.get("verification", {})

                # Display the generated answer.
                st.markdown(answer)

                # Display retrieved legal sources.
                if sources:
                    display_sources(sources)
                else:
                    st.warning("No legal sources were retrieved.")

                # Display claim-level verification information.
                display_verification(verification)

                # Save the assistant's response and related information.
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "verification": verification
                    }
                )

            except Exception as error:

                st.error(
                    "Something went wrong while processing "
                    "your question."
                )

                st.exception(error)