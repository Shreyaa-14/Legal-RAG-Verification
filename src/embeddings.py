
import json
import os
import shutil
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT / "data" / "validated" / "approved_legal.jsonl"
)

VALIDATION_FILE = (
    PROJECT_ROOT / "evaluation" / "dataset_validation_summary.json"
)

VECTOR_DIR = PROJECT_ROOT / "vector_store"

INDEX_FILE = VECTOR_DIR / "legal_index.faiss"
METADATA_FILE = VECTOR_DIR / "legal_metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ==========================================================
# 2. CHECK DATASET VALIDATION
# ==========================================================

def check_validation_status():

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            "Validation report not found.\n"
            "Run: python -m src.validate_dataset"
        )

    with open(VALIDATION_FILE, "r", encoding="utf-8") as file:
        result = json.load(file)

    if result.get("validation_status") != "PASS":
        raise ValueError(
            "Dataset validation has not passed.\n"
            "Fix the validation errors before building embeddings."
        )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Approved dataset not found: {INPUT_FILE}"
        )

    # Prevent indexing after dataset changes that have not
    # been revalidated.
    if INPUT_FILE.stat().st_mtime > VALIDATION_FILE.stat().st_mtime:
        raise ValueError(
            "Approved dataset changed after validation.\n"
            "Run: python -m src.validate_dataset"
        )

    print("Dataset validation status: PASS")

    return result


# ==========================================================
# 3. LOAD APPROVED LEGAL RECORDS
# ==========================================================

def load_approved_records():

    records = []
    seen_ids = set()

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} is not a JSON object."
                )

            record_id = str(record.get("record_id", "")).strip()
            text = record.get("text")

            if not record_id:
                raise ValueError(
                    f"Missing record ID at line {line_number}."
                )

            if record_id in seen_ids:
                raise ValueError(
                    f"Duplicate record ID: {record_id}"
                )

            seen_ids.add(record_id)

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Invalid text at line {line_number}."
                )

            qc = record.get("quality_control", {})

            if not isinstance(qc, dict):
                raise ValueError(
                    f"Missing quality-control metadata "
                    f"at line {line_number}."
                )

            if qc.get("decision") not in {
                "eligible_after_automated_checks",
                "approved_after_manual_review",
            }:
                raise ValueError(
                    f"Unapproved record found at line {line_number}."
                )

            records.append(record)

    if not records:
        raise ValueError(
            "Approved dataset is empty. Cannot build FAISS index."
        )

    return records


# ==========================================================
# 4. BUILD EMBEDDINGS AND FAISS INDEX
# ==========================================================

def build_vector_index():

    print("\n" + "=" * 60)
    print("QUALITY-CONTROLLED LEGAL EMBEDDING PIPELINE")
    print("=" * 60)

    # Step 1: Check validation
    validation = check_validation_status()

    # Step 2: Load records
    print("\nLoading approved legal records...")

    records = load_approved_records()

    print("Approved records loaded:", len(records))

    expected_count = validation.get("approved_records")

    if expected_count != len(records):
        raise ValueError(
            "Approved record count does not match "
            "the validation report. Revalidate the dataset."
        )

    # Step 3: Prepare legal texts
    texts = [record["text"] for record in records]

    # Step 4: Load embedding model
    print("\nLoading embedding model:", MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    # Step 5: Generate embeddings
    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError("Invalid embedding dimensions.")

    if embeddings.shape[0] != len(records):
        raise ValueError(
            "Embedding count does not match record count."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain invalid numerical values."
        )

    print("Embedding matrix shape:", embeddings.shape)

    # Step 6: Create FAISS cosine similarity index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    if index.ntotal != len(records):
        raise ValueError(
            "FAISS vector count does not match record count."
        )

    print("Vectors indexed:", index.ntotal)

    # Step 7: Save files to temporary paths first
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    temp_index = VECTOR_DIR / "legal_index.tmp.faiss"
    temp_metadata = VECTOR_DIR / "legal_metadata.tmp.json"

    backup_index = VECTOR_DIR / "legal_index.faiss.bak"
    backup_metadata = VECTOR_DIR / "legal_metadata.json.bak"

    try:

        faiss.write_index(index, str(temp_index))

        with open(
            temp_metadata,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                records,
                file,
                ensure_ascii=False,
                indent=2,
            )

        # Verify the saved temporary index and metadata.
        saved_index = faiss.read_index(str(temp_index))

        with open(
            temp_metadata,
            "r",
            encoding="utf-8",
        ) as file:
            saved_metadata = json.load(file)

        if saved_index.ntotal != len(saved_metadata):
            raise ValueError(
                "Saved FAISS index and metadata counts differ."
            )

        # Back up the previous index, if one exists.
        if INDEX_FILE.exists():
            shutil.copy2(INDEX_FILE, backup_index)

        if METADATA_FILE.exists():
            shutil.copy2(METADATA_FILE, backup_metadata)

        # Publish the validated files.
        os.replace(temp_index, INDEX_FILE)
        os.replace(temp_metadata, METADATA_FILE)

    finally:

        # Remove unfinished temporary files.
        for path in [temp_index, temp_metadata]:
            if path.exists():
                path.unlink()

    print("\n" + "=" * 60)
    print("LEGAL VECTOR INDEX CREATED SUCCESSFULLY")
    print("=" * 60)

    print("Records indexed:", index.ntotal)
    print("Embedding dimension:", dimension)
    print("Embedding model:", MODEL_NAME)
    print("Input dataset:", INPUT_FILE)
    print("FAISS index:", INDEX_FILE)
    print("Metadata file:", METADATA_FILE)

    print("\nOnly approved records were indexed.")
    print("=" * 60)


# ==========================================================
# 5. MAIN
# ==========================================================

if __name__ == "__main__":
    build_vector_index()
