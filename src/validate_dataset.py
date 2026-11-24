
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_FILE = (
    PROJECT_ROOT / "data" / "processed" / "legal_cleaned.jsonl"
)

APPROVED_FILE = (
    PROJECT_ROOT / "data" / "validated" / "approved_legal.jsonl"
)

QUARANTINED_FILE = (
    PROJECT_ROOT / "data" / "validated" / "quarantined_legal.jsonl"
)

AUDIT_FILE = (
    PROJECT_ROOT / "evaluation" / "legal_quality_report.csv"
)

QC_REPORT_FILE = (
    PROJECT_ROOT / "evaluation" / "quality_control_report.csv"
)

REPORT_FILE = (
    PROJECT_ROOT / "evaluation" / "dataset_validation_report.txt"
)

JSON_REPORT_FILE = (
    PROJECT_ROOT / "evaluation" / "dataset_validation_summary.json"
)


# ==========================================================
# 2. CONFIGURATION
# ==========================================================

REQUIRED_FIELDS = [
    "record_id",
    "act_id",
    "chunk_id",
    "title",
    "text",
    "source_url",
]

APPROVED_DECISIONS = {
    "eligible_after_automated_checks",
    "approved_after_manual_review",
}

QUARANTINED_DECISIONS = {
    "quarantined",
}


# ==========================================================
# 3. HELPER FUNCTIONS
# ==========================================================

def is_blank(value):
    return value is None or str(value).strip() == ""


def normalize_text(text):
    return re.sub(r"\s+", " ", text.casefold()).strip()


def valid_url_format(value):
    if not isinstance(value, str) or not value.strip():
        return False

    try:
        parsed = urlsplit(value.strip())

        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and "." in parsed.hostname
        )
    except ValueError:
        return False


def load_jsonl(path):
    """
    Load JSONL and fail on malformed records.
    Never silently skip corrupted JSON.
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    records = []

    with open(path, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path.name}, line {line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path.name}, line {line_number}: "
                    "Expected a JSON object."
                )

            records.append(record)

    return records


def load_csv(path):

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        return list(csv.DictReader(file))


# ==========================================================
# 4. VALIDATE INDIVIDUAL RECORDS
# ==========================================================

def validate_approved_records(records):
    """
    Validate records eligible for embedding.

    These checks do not certify legal correctness.
    """

    errors = []
    warnings = []

    seen_record_ids = set()
    seen_chunk_keys = set()
    seen_texts = {}

    decision_counts = Counter()

    for index, record in enumerate(records, start=1):

        record_id = str(record.get("record_id", "")).strip()
        label = f"Approved record {index} ({record_id or 'no ID'})"

        # Required fields
        for field in REQUIRED_FIELDS:

            if is_blank(record.get(field)):
                errors.append(
                    f"{label}: Missing required field '{field}'."
                )

        # Duplicate technical IDs
        if record_id:

            if record_id in seen_record_ids:
                errors.append(
                    f"{label}: Duplicate record_id."
                )

            seen_record_ids.add(record_id)

        # Duplicate act/chunk identifiers
        act_id = record.get("act_id")
        chunk_id = record.get("chunk_id")

        if not is_blank(act_id) and not is_blank(chunk_id):

            key = (str(act_id).strip(), str(chunk_id).strip())

            if key in seen_chunk_keys:
                errors.append(
                    f"{label}: Duplicate act_id/chunk_id combination."
                )

            seen_chunk_keys.add(key)

        # Legal text checks
        text = record.get("text")

        if not isinstance(text, str):
            errors.append(
                f"{label}: Text must be a string."
            )

        elif not text.strip():
            errors.append(
                f"{label}: Legal text is empty."
            )

        else:

            if "\ufffd" in text:
                errors.append(
                    f"{label}: Encoding replacement character detected."
                )

            normalized = normalize_text(text)

            if normalized in seen_texts:
                errors.append(
                    f"{label}: Possible duplicate legal text; "
                    f"first seen in record {seen_texts[normalized]}."
                )
            else:
                seen_texts[normalized] = index

        # Source URL format
        if not valid_url_format(record.get("source_url")):
            errors.append(
                f"{label}: Invalid source URL format."
            )

        # Quality-control metadata
        qc = record.get("quality_control")

        if not isinstance(qc, dict):
            errors.append(
                f"{label}: Missing quality_control information."
            )
            continue

        decision = qc.get("decision", "")

        decision_counts[decision] += 1

        if decision not in APPROVED_DECISIONS:
            errors.append(
                f"{label}: Invalid approved decision '{decision}'."
            )

        if decision == "eligible_after_automated_checks":

            if qc.get("audit_flagged") is not False:
                errors.append(
                    f"{label}: Automatically eligible record "
                    "is marked as audit-flagged."
                )

        if decision == "approved_after_manual_review":

            if qc.get("audit_flagged") is not True:
                errors.append(
                    f"{label}: Manual approval has no audit flag."
                )

            warnings.append(
                f"{label}: Manual source-review evidence "
                "should be checked separately."
            )

    return errors, warnings, dict(decision_counts)


# ==========================================================
# 5. VALIDATE DATASET SEPARATION
# ==========================================================

def validate_separation(processed, approved, quarantined):

    errors = []

    processed_ids = [
        str(r.get("record_id", "")).strip()
        for r in processed
    ]

    approved_ids = [
        str(r.get("record_id", "")).strip()
        for r in approved
    ]

    quarantined_ids = [
        str(r.get("record_id", "")).strip()
        for r in quarantined
    ]

    # Count consistency
    if len(processed) != len(approved) + len(quarantined):
        errors.append(
            "Record count mismatch: processed total does not "
            "equal approved + quarantined."
        )

    # Unique identifiers
    if len(processed_ids) != len(set(processed_ids)):
        errors.append(
            "Processed dataset contains duplicate record IDs."
        )

    if len(quarantined_ids) != len(set(quarantined_ids)):
        errors.append(
            "Quarantined dataset contains duplicate record IDs."
        )

    # No record may occur in both datasets
    overlap = set(approved_ids) & set(quarantined_ids)

    if overlap:
        errors.append(
            f"{len(overlap)} record IDs occur in both "
            "approved and quarantined datasets."
        )

    # Every processed record should be accounted for
    if Counter(processed_ids) != Counter(
        approved_ids + quarantined_ids
    ):
        errors.append(
            "Approved and quarantined record IDs do not "
            "exactly account for processed records."
        )

    # Quarantine decisions
    for index, record in enumerate(quarantined, start=1):

        qc = record.get("quality_control", {})

        if not isinstance(qc, dict):
            errors.append(
                f"Quarantined record {index}: Missing QC metadata."
            )
            continue

        if qc.get("decision") not in QUARANTINED_DECISIONS:
            errors.append(
                f"Quarantined record {index}: Invalid QC decision."
            )

    return errors


# ==========================================================
# 6. VALIDATE AGAINST AUDIT AND QC REPORTS
# ==========================================================

def validate_reports(approved, quarantined, audit_rows, qc_rows):

    errors = []

    approved_ids = {
        str(r.get("record_id", "")).strip()
        for r in approved
    }

    quarantined_ids = {
        str(r.get("record_id", "")).strip()
        for r in quarantined
    }

    audit_ids = [
        str(r.get("record_id", "")).strip()
        for r in audit_rows
    ]

    qc_ids = [
        str(r.get("record_id", "")).strip()
        for r in qc_rows
    ]

    if len(audit_ids) != len(set(audit_ids)):
        errors.append("Quality audit contains duplicate record IDs.")

    if len(qc_ids) != len(set(qc_ids)):
        errors.append("Quality-control report contains duplicate IDs.")

    if set(qc_ids) != approved_ids | quarantined_ids:
        errors.append(
            "Quality-control report IDs do not match "
            "the validated datasets."
        )

    # Verify that automated approvals have no audit flags
    audited_ids = set(audit_ids)

    for record in approved:

        qc = record.get("quality_control", {})

        if not isinstance(qc, dict):
            continue

        if qc.get("decision") == "eligible_after_automated_checks":

            record_id = str(record.get("record_id", "")).strip()

            if record_id in audited_ids:
                errors.append(
                    f"Automatically approved record {record_id} "
                    "appears in the audit report."
                )

    # Confirm report decisions match dataset decisions
    qc_lookup = {
        str(row.get("record_id", "")).strip(): row
        for row in qc_rows
    }

    for record in approved + quarantined:

        record_id = str(record.get("record_id", "")).strip()
        qc = record.get("quality_control", {})

        if not isinstance(qc, dict):
            continue

        expected = qc.get("decision")
        report_row = qc_lookup.get(record_id)

        if report_row is None:
            errors.append(
                f"Record {record_id} missing from QC report."
            )
            continue

        actual = report_row.get("quality_control_decision")

        if actual != expected:
            errors.append(
                f"Record {record_id}: QC decision mismatch."
            )

    return errors


# ==========================================================
# 7. SAVE VALIDATION REPORTS
# ==========================================================

def save_reports(summary, errors, warnings):

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(
        JSON_REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    lines = [
        "=" * 60,
        "INDIAN LEGAL DATASET VALIDATION REPORT",
        "=" * 60,
        f"Validation status: {summary['validation_status']}",
        f"Processed records: {summary['processed_records']}",
        f"Approved records: {summary['approved_records']}",
        f"Quarantined records: {summary['quarantined_records']}",
        f"Errors: {len(errors)}",
        f"Warnings: {len(warnings)}",
        "",
        "Note: Technical validation does not prove legal correctness.",
        "",
        "ERRORS:",
    ]

    if errors:
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("None")

    lines.append("")
    lines.append("WARNINGS:")

    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("None")

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("\n".join(lines) + "\n")


# ==========================================================
# 8. MAIN
# ==========================================================

def main():

    print("\nLoading processed dataset...")
    processed = load_jsonl(PROCESSED_FILE)

    print("Loading approved dataset...")
    approved = load_jsonl(APPROVED_FILE)

    print("Loading quarantined dataset...")
    quarantined = load_jsonl(QUARANTINED_FILE)

    print("Loading audit and quality-control reports...")
    audit_rows = load_csv(AUDIT_FILE)
    qc_rows = load_csv(QC_REPORT_FILE)

    print("\nRunning validation checks...")

    errors, warnings, decision_counts = (
        validate_approved_records(approved)
    )

    errors.extend(
        validate_separation(
            processed,
            approved,
            quarantined
        )
    )

    errors.extend(
        validate_reports(
            approved,
            quarantined,
            audit_rows,
            qc_rows
        )
    )

    if not approved:
        errors.append(
            "Approved dataset is empty. Cannot proceed to embedding."
        )

    status = "PASS" if not errors else "FAIL"

    summary = {
        "validation_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "validation_status": status,
        "processed_records": len(processed),
        "approved_records": len(approved),
        "quarantined_records": len(quarantined),
        "audit_flagged_records": len(audit_rows),
        "quality_control_report_records": len(qc_rows),
        "approved_decision_counts": decision_counts,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "note": (
            "This report confirms technical consistency "
            "only. It does not certify legal accuracy, "
            "source authenticity, or current applicability."
        )
    }

    save_reports(summary, errors, warnings)

    print("\n" + "=" * 60)
    print("FINAL DATASET VALIDATION")
    print("=" * 60)

    print("Processed records:", len(processed))
    print("Approved records:", len(approved))
    print("Quarantined records:", len(quarantined))

    print("\nValidation status:", status)
    print("Errors:", len(errors))
    print("Warnings:", len(warnings))

    if errors:
        print("\nFirst validation errors:")
        for error in errors[:10]:
            print("-", error)

    print("\nText report:", REPORT_FILE)
    print("JSON report:", JSON_REPORT_FILE)

    print("=" * 60)

    if errors:
        print("DATASET VALIDATION FAILED")
        return 1

    print("DATASET VALIDATION PASSED")
    print("Approved dataset is ready for the embedding stage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
