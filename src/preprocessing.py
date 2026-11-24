
import json
import re
import hashlib
from collections import Counter
from pathlib import Path


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "legal_sample.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "legal_cleaned.jsonl"

REPORT_FILE = OUTPUT_DIR / "preprocessing_report.json"


# ==========================================================
# 2. TEXT CLEANING
# ==========================================================

def clean_text(text):
    """
    Clean unnecessary formatting while preserving
    meaningful legal paragraph boundaries.
    """

    if not isinstance(text, str):
        return ""

    original_text = text
    transformations = []

    # Normalize Windows and Mac line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if text != original_text:
        transformations.append("normalized_line_endings")

    # Replace non-breaking spaces
    if "\u00a0" in text:
        text = text.replace("\u00a0", " ")
        transformations.append("normalized_nonbreaking_spaces")

    # Remove null characters
    if "\x00" in text:
        text = text.replace("\x00", "")
        transformations.append("removed_null_characters")

    # Process individual lines instead of flattening
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:

        # Normalize repeated horizontal whitespace
        cleaned_line = re.sub(r"[ \t\f\v]+", " ", line)

        # Remove unnecessary spaces at line boundaries
        cleaned_line = cleaned_line.strip()

        cleaned_lines.append(cleaned_line)

    text = "\n".join(cleaned_lines)

    if text != original_text and "normalized_horizontal_whitespace" not in transformations:
        transformations.append("normalized_horizontal_whitespace")

    # Limit excessive blank lines while retaining paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = text.strip()

    return text, transformations


# ==========================================================
# 3. TECHNICAL RECORD IDENTIFIER
# ==========================================================

def generate_record_id(record, record_number):
    """
    Generate a stable technical ID when the original
    dataset does not provide a usable record ID.

    This is NOT a legal section number or legal identifier.
    """

    existing_id = record.get("record_id")

    if existing_id is not None and str(existing_id).strip():
        return str(existing_id)

    identifier_data = "|".join([
        str(record.get("act_id", "")),
        str(record.get("chunk_id", "")),
        str(record.get("title", "")),
        str(record.get("source_url", "")),
        str(record_number),
    ])

    return hashlib.sha256(
        identifier_data.encode("utf-8")
    ).hexdigest()[:20]


# ==========================================================
# 4. PREPROCESSING
# ==========================================================

def preprocess_dataset():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {INPUT_FILE}\n"
            "Run download_dataset.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_records = 0
    saved_records = 0
    skipped_records = 0

    transformation_counts = Counter()
    metadata_counts = Counter()

    temporary_file = OUTPUT_FILE.with_suffix(".jsonl.tmp")

    try:

        with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
             open(temporary_file, "w", encoding="utf-8") as outfile:

            for line_number, line in enumerate(infile, start=1):

                if not line.strip():
                    continue

                total_records += 1

                # Read and validate JSON
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at line {line_number}: {error}"
                    ) from error

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Record at line {line_number} is not a JSON object."
                    )

                # Collect metadata availability statistics
                for key, value in record.items():

                    if key == "text":
                        continue

                    if value is not None and str(value).strip():
                        metadata_counts[key] += 1

                # Clean legal text
                original_text = record.get("text", "")

                cleaned_text, transformations = clean_text(original_text)

                if not cleaned_text:

                    skipped_records += 1
                    transformation_counts["empty_text_skipped"] += 1

                    continue

                # Preserve ALL original fields
                processed_record = dict(record)

                # Replace only the cleaned text field
                processed_record["text"] = cleaned_text

                # Add a technical ID only if missing
                processed_record["record_id"] = generate_record_id(
                    record,
                    total_records
                )

                # Store preprocessing information separately
                processed_record["preprocessing_info"] = {
                    "source_record_number": total_records,
                    "original_text_length": len(original_text)
                    if isinstance(original_text, str) else 0,
                    "cleaned_text_length": len(cleaned_text),
                    "transformations": transformations,
                    "preprocessing_status": "completed",
                }

                for transformation in transformations:
                    transformation_counts[transformation] += 1

                outfile.write(
                    json.dumps(
                        processed_record,
                        ensure_ascii=False
                    ) + "\n"
                )

                saved_records += 1

        # Replace the previous processed output only after success
        temporary_file.replace(OUTPUT_FILE)

    except Exception:

        # Do not leave a partial output file
        if temporary_file.exists():
            temporary_file.unlink()

        raise

    # ======================================================
    # 5. PREPROCESSING REPORT
    # ======================================================

    report = {
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "total_records_examined": total_records,
        "records_saved": saved_records,
        "records_skipped": skipped_records,
        "metadata_field_presence": dict(metadata_counts),
        "transformation_counts": dict(transformation_counts),
        "status": "completed",
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as report_file:

        json.dump(
            report,
            report_file,
            ensure_ascii=False,
            indent=4
        )

    # ======================================================
    # 6. DISPLAY RESULTS
    # ======================================================

    print("\n" + "=" * 55)
    print("LEGAL DATASET PREPROCESSING REPORT")
    print("=" * 55)

    print("Total records examined:", total_records)
    print("Records saved:", saved_records)
    print("Records skipped:", skipped_records)

    print("\nMetadata fields preserved:")

    for field, count in sorted(metadata_counts.items()):
        print(f"{field}: {count}/{total_records}")

    print("\nText transformations:")

    if transformation_counts:
        for transformation, count in sorted(
            transformation_counts.items()
        ):
            print(f"{transformation}: {count}")
    else:
        print("No text transformations were required.")

    print("\nProcessed dataset:", OUTPUT_FILE)
    print("Preprocessing report:", REPORT_FILE)

    print("=" * 55)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")


# ==========================================================
# 7. MAIN
# ==========================================================

if __name__ == "__main__":
    preprocess_dataset()