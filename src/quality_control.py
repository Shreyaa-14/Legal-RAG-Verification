
import csv
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_cleaned.jsonl"
)

AUDIT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "legal_quality_report.csv"
)

REVIEW_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "legal_manual_review.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "validated"

APPROVED_FILE = OUTPUT_DIR / "approved_legal.jsonl"

QUARANTINE_FILE = OUTPUT_DIR / "quarantined_legal.jsonl"

REPORT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "quality_control_report.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "quality_control_summary.json"
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

APPROVED_REVIEW_STATUSES = {
    "verified_correct",
    "corrected_from_source",
    "retain_missing_metadata",
}


# ==========================================================
# 3. BASIC HELPERS
# ==========================================================

def is_blank(value):

    return value is None or str(value).strip() == ""


def read_jsonl(path):

    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

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
                    f"Record at line {line_number} is not an object."
                )

            records.append(record)

    return records


def read_csv(path):

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        reader = csv.DictReader(file)

        return list(reader)


def valid_url(url):

    if is_blank(url):
        return False

    try:
        parsed = urlsplit(str(url).strip())

        return (
            parsed.scheme in ("http", "https")
            and bool(parsed.netloc)
            and "." in parsed.netloc
        )

    except ValueError:
        return False


# ==========================================================
# 4. CREATE REVIEW LOOKUP
# ==========================================================

def build_review_lookup(review_rows):

    review_lookup = {}

    for row in review_rows:

        review_key = str(
            row.get("review_key", "")
        ).strip()

        record_id = str(
            row.get("record_id", "")
        ).strip()

        # Stage 3 review key format
        if not review_key and record_id:
            review_key = f"record_id::{record_id}"

        if not review_key:
            continue

        review_lookup[review_key] = row

    return review_lookup


# ==========================================================
# 5. CHECK BASIC RECORD VALIDITY
# ==========================================================

def validate_record(record):

    issues = []

    for field in REQUIRED_FIELDS:

        if is_blank(record.get(field)):
            issues.append(f"missing_{field}")

    text = record.get("text", "")

    if not isinstance(text, str):
        issues.append("invalid_text_type")

    elif not text.strip():
        issues.append("empty_text")

    if "\ufffd" in str(text):
        issues.append("encoding_corruption")

    if not is_blank(record.get("source_url")):

        if not valid_url(record["source_url"]):
            issues.append("invalid_source_url")

    return issues


# ==========================================================
# 6. CHECK MANUAL REVIEW DECISION
# ==========================================================

def check_manual_approval(review):

    if not review:

        return False, "No manual review record found."

    status = str(
        review.get("manual_review_status", "")
    ).strip()

    verified_source = str(
        review.get("verified_source", "")
    ).strip()

    reviewer = str(
        review.get("reviewer", "")
    ).strip()

    notes = str(
        review.get("review_notes", "")
    ).strip()

    correction_required = str(
        review.get("correction_required", "")
    ).strip().lower()

    if status not in APPROVED_REVIEW_STATUSES:

        return False, f"Review status is '{status or 'missing'}'."

    # All manual approvals must identify the source and reviewer
    if not verified_source:

        return False, "Verified source URL is missing."

    if not valid_url(verified_source):

        return False, "Verified source URL has invalid format."

    if not reviewer:

        return False, "Reviewer name is missing."

    if not notes:

        return False, "Manual review notes are missing."

    # Corrected records must already have been updated
    # in the processed JSONL before being approved.
    if status == "corrected_from_source":

        if correction_required != "no":

            return False, (
                "Correction is still required or not confirmed."
            )

    if status == "retain_missing_metadata":

        if correction_required == "yes":

            return False, (
                "Record still requires a correction."
            )

    if status == "verified_correct":

        if correction_required == "yes":

            return False, (
                "Record is marked for correction."
            )

    return True, "Manual review requirements satisfied."


# ==========================================================
# 7. DATASET QUALITY CONTROL
# ==========================================================

def run_quality_control(
    records,
    audit_rows,
    review_rows,
):

    review_lookup = build_review_lookup(review_rows)

    # Identify records flagged by the latest audit.
    flagged_ids = {
        str(row.get("record_id", "")).strip()
        for row in audit_rows
        if not is_blank(row.get("record_id"))
    }

    approved_records = []
    quarantined_records = []
    report_rows = []

    seen_ids = set()

    for record in records:

        record_id = str(
            record.get("record_id", "")
        ).strip()

        issues = validate_record(record)

        if record_id:

            if record_id in seen_ids:
                issues.append("duplicate_record_id")

            seen_ids.add(record_id)

        review_key = f"record_id::{record_id}"

        review = review_lookup.get(review_key)

        is_flagged = record_id in flagged_ids

        decision = ""
        review_message = ""

        if issues:

            decision = "quarantined"
            review_message = (
                "Basic record validation failed."
            )

        elif is_flagged:

            approved, review_message = check_manual_approval(
                review
            )

            if approved:
                decision = "approved_after_manual_review"
            else:
                decision = "quarantined"

        else:

            decision = (
                "eligible_after_automated_checks"
            )

            review_message = (
                "No audit flags detected. "
                "This is not legal source verification."
            )

        # Preserve original record fields and add QC information
        output_record = dict(record)

        output_record["quality_control"] = {
            "decision": decision,
            "audit_flagged": is_flagged,
            "manual_review_status": (
                review.get("manual_review_status", "")
                if review else "not_required"
            ),
            "review_message": review_message,
        }

        if decision == "quarantined":

            quarantined_records.append(output_record)

        else:

            approved_records.append(output_record)

        report_rows.append({
            "record_id": record_id,
            "title": record.get("title", ""),
            "act_id": record.get("act_id", ""),
            "chunk_id": record.get("chunk_id", ""),
            "audit_flagged": is_flagged,
            "manual_review_status": (
                review.get("manual_review_status", "")
                if review else "not_required"
            ),
            "quality_control_decision": decision,
            "issues": "; ".join(sorted(set(issues))),
            "review_message": review_message,
        })

    return (
        approved_records,
        quarantined_records,
        report_rows,
    )


# ==========================================================
# 8. SAVE JSONL FILES
# ==========================================================

def save_jsonl(path, records):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                ) + "\n"
            )


# ==========================================================
# 9. SAVE CONTROL REPORT
# ==========================================================

def save_report(report_rows):

    fields = [
        "record_id",
        "title",
        "act_id",
        "chunk_id",
        "audit_flagged",
        "manual_review_status",
        "quality_control_decision",
        "issues",
        "review_message",
    ]

    with open(
        REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(report_rows)


# ==========================================================
# 10. GENERATE SUMMARY
# ==========================================================

def generate_summary(
    total,
    approved,
    quarantined,
    report_rows,
):

    decisions = Counter(
        row["quality_control_decision"]
        for row in report_rows
    )

    summary = {
        "total_records": total,
        "approved_records": len(approved),
        "quarantined_records": len(quarantined),
        "decision_counts": dict(decisions),
        "approval_note": (
            "Eligible records have passed automated checks "
            "or the configured manual-review requirements. "
            "This does not certify legal correctness."
        ),
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return summary


# ==========================================================
# 11. MAIN
# ==========================================================

def main():

    required_files = [
        INPUT_FILE,
        AUDIT_FILE,
        REVIEW_FILE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading processed legal records...")

    records = read_jsonl(INPUT_FILE)

    print("Loading quality audit report...")

    audit_rows = read_csv(AUDIT_FILE)

    print("Loading manual review decisions...")

    review_rows = read_csv(REVIEW_FILE)

    print("\nRunning final quality control...")

    (
        approved,
        quarantined,
        report_rows,
    ) = run_quality_control(
        records,
        audit_rows,
        review_rows,
    )

    save_jsonl(
        APPROVED_FILE,
        approved,
    )

    save_jsonl(
        QUARANTINE_FILE,
        quarantined,
    )

    save_report(report_rows)

    summary = generate_summary(
        len(records),
        approved,
        quarantined,
        report_rows,
    )

    print("\n" + "=" * 60)
    print("LEGAL DATASET QUALITY CONTROL REPORT")
    print("=" * 60)

    print("Total records:", summary["total_records"])
    print("Approved records:", summary["approved_records"])
    print("Quarantined records:", summary["quarantined_records"])

    print("\nDecision breakdown:")

    for decision, count in sorted(
        summary["decision_counts"].items()
    ):

        print(f"{decision}: {count}")

    print("\nApproved dataset:", APPROVED_FILE)
    print("Quarantined dataset:", QUARANTINE_FILE)
    print("Detailed report:", REPORT_FILE)
    print("Summary:", SUMMARY_FILE)

    print("=" * 60)
    print("QUALITY CONTROL COMPLETED")


if __name__ == "__main__":
    main()
