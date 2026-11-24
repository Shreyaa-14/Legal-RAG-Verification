
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Audit the cleaned dataset generated in Stage 1
INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_cleaned.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "evaluation"

OUTPUT_FILE = OUTPUT_DIR / "legal_quality_report.csv"

SUMMARY_FILE = OUTPUT_DIR / "legal_quality_summary.json"


# ==========================================================
# 2. QUALITY CHECK CONFIGURATION
# ==========================================================

REQUIRED_FIELDS = [
    "act_id",
    "chunk_id",
    "title",
    "text",
    "source_url",
]

OPTIONAL_METADATA_FIELDS = [
    "chapter",
    "section_number",
    "section_title",
    "year",
    "jurisdiction",
    "language_code",
]

SECTION_PATTERN = re.compile(
    r"^(?:"
    r"\d+[A-Za-z]?(?:\(\d+[A-Za-z]?\))*"
    r"(?:\([A-Za-zivxlcdm]+\))*"
    r"|Article\s+\d+[A-Za-z]?"
    r"|[IVXLCDM]+"
    r")$",
    re.IGNORECASE,
)

REPORT_FIELDS = [
    "line_number",
    "source_record_number",
    "record_id",
    "chunk_id",
    "act_id",
    "title",
    "chapter",
    "section_number",
    "section_title",
    "year",
    "source_url",
    "issues",
    "severity",
    "recommended_action",
    "review_status",
]


# ==========================================================
# 3. HELPER FUNCTIONS
# ==========================================================

def is_blank(value):
    """Check whether a field is missing or blank."""

    return value is None or str(value).strip() == ""


def normalize_text(text):
    """
    Normalize text for duplicate detection only.

    Original legal text is never modified here.
    """

    text = str(text).casefold()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def valid_url_format(value):
    """
    Check URL structure only.

    This does not verify whether the URL is accessible
    or whether it contains the correct legal provision.
    """

    if is_blank(value):
        return False

    try:
        parsed = urlsplit(str(value).strip())

        return (
            parsed.scheme in ("http", "https")
            and bool(parsed.netloc)
            and "." in parsed.netloc
        )

    except ValueError:
        return False


def detect_text_issues(text):
    """Identify suspicious text and formatting artifacts."""

    issues = []

    if not isinstance(text, str):
        return ["invalid_text_type"]

    # Encoding replacement character
    if "\ufffd" in text:
        issues.append("encoding_corruption")

    # Unexpected control characters
    control_chars = [
        char for char in text
        if ord(char) < 32 and char not in "\n\t\r"
    ]

    if control_chars:
        issues.append("unexpected_control_characters")

    # Possible OCR/text corruption.
    # These patterns are heuristic warning signals,
    # not proof that a legal record is incorrect.

    suspicious_patterns = [
        r"\[[^\]]{0,2}\]",
        r"\{[^}]{0,2}\}",
        r"(?i)\b[a-z]*[0-9][a-z]*[0-9][a-z]*\b",
        r"(?i)\b[a-z]{1,2}[~^`][a-z]+\b",
    ]

    suspicious_count = sum(
        len(re.findall(pattern, text))
        for pattern in suspicious_patterns
    )

    if suspicious_count >= 5:
        issues.append("possible_ocr_or_text_corruption")

    # HTML formatting artifacts
    if re.search(r"<br\s*/?>", text, re.IGNORECASE):
        issues.append("html_break_tag")

    if re.search(r"<[^>]+>", text):
        issues.append("possible_html_artifact")

    # Markdown code fences
    if "```" in text:
        issues.append("markdown_code_fence")

    # Repeated pipe characters
    if re.search(r"\|{3,}", text):
        issues.append("repeated_pipe_characters")

    # Escaped whitespace sequences
    if re.search(r"(\\n|\\r|\\t)", text):
        issues.append("escaped_whitespace_sequence")

    # Repeated punctuation
    if re.search(r"[!?.,;:]{4,}", text):
        issues.append("repeated_punctuation")

    return issues


def get_record_issues(record):
    """Perform individual record-level checks."""

    issues = []

    # Required fields
    for field in REQUIRED_FIELDS:

        if is_blank(record.get(field)):
            issues.append(f"missing_{field}")

    # Optional metadata fields
    for field in OPTIONAL_METADATA_FIELDS:

        if is_blank(record.get(field)):
            issues.append(f"missing_metadata_{field}")

    # Text validation
    text = record.get("text", "")

    if not isinstance(text, str):
        issues.append("invalid_text_type")

    elif is_blank(text):
        issues.append("empty_text")

    else:
        issues.extend(detect_text_issues(text))

    # Section number formatting
    section_number = record.get("section_number")

    if not is_blank(section_number):

        if not SECTION_PATTERN.fullmatch(
            str(section_number).strip()
        ):
            issues.append("malformed_section_label")

    # Year format
    
    # Year format validation
    year = record.get("year")

    if not is_blank(year):

        year_text = str(year).strip()

        try:
            year_number = float(year_text)

            # Accept 1994, "1994", 1994.0 and "1994.0"
            # Reject fractional years and invalid values.
            if (
                not year_number.is_integer()
                or not 1000 <= year_number <= 2100
            ):
                issues.append("invalid_year_format")

        except (ValueError, OverflowError):
            issues.append("invalid_year_format")


    # Source URL format
    source_url = record.get("source_url")

    if not is_blank(source_url):

        if not valid_url_format(source_url):
            issues.append("invalid_source_url_format")

    # Recommend source verification when necessary.
    # This is a recommendation, not a legal correctness judgment.

    verification_triggers = {
        "missing_source_url",
        "missing_section_number",
        "encoding_corruption",
        "unexpected_control_characters",
        "possible_ocr_or_text_corruption",
        "malformed_section_label",
        "invalid_source_url_format",
    }

    if verification_triggers.intersection(issues):
        issues.append("requires_legal_source_verification")

    return sorted(set(issues))


# ==========================================================
# 4. LOAD CLEANED DATASET
# ==========================================================

def load_dataset():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found:\n{INPUT_FILE}\n\n"
            "Run Stage 1 first:\n"
            "python -m src.preprocessing"
        )

    records = []

    with open(
        INPUT_FILE,
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
                    f"Line {line_number} is not a JSON object."
                )

            records.append({
                "line_number": line_number,
                "record": record,
            })

    return records


# ==========================================================
# 5. DUPLICATE DETECTION
# ==========================================================

def find_duplicates(records):

    chunk_locations = defaultdict(list)
    text_locations = defaultdict(list)

    for index, item in enumerate(records):

        record = item["record"]

        chunk_id = record.get("chunk_id")

        if not is_blank(chunk_id):

            # Include act ID to avoid treating identical
            # chunk labels in different acts as duplicates.
            key = (
                str(record.get("act_id", "")).strip(),
                str(chunk_id).strip(),
            )

            chunk_locations[key].append(index)

        text = record.get("text", "")

        if isinstance(text, str) and not is_blank(text):

            normalized = normalize_text(text)

            if normalized:
                text_locations[normalized].append(index)

    return chunk_locations, text_locations


# ==========================================================
# 6. SEVERITY AND RECOMMENDED ACTION
# ==========================================================

CRITICAL_ISSUES = {
    "missing_text",
    "empty_text",
    "invalid_text_type",
    "encoding_corruption",
    "unexpected_control_characters",
    "duplicate_chunk_id",
    "possible_duplicate_text",
}

HIGH_ISSUES = {
    "possible_ocr_or_text_corruption",
    "malformed_section_label",
    "invalid_source_url_format",
    "missing_source_url",
    "requires_legal_source_verification",
}


def classify_severity(issues):

    if CRITICAL_ISSUES.intersection(issues):
        return "high"

    if HIGH_ISSUES.intersection(issues):
        return "medium"

    return "low"


def recommend_action(issues):

    if "encoding_corruption" in issues:
        return "quarantine_until_text_recovered"

    if "empty_text" in issues or "invalid_text_type" in issues:
        return "exclude_unusable_record"

    if "duplicate_chunk_id" in issues:
        return "compare_records_and_review_duplicate_id"

    if "possible_duplicate_text" in issues:
        return "compare_records_before_exclusion"

    if (
        "possible_ocr_or_text_corruption" in issues
        or "malformed_section_label" in issues
        or "invalid_source_url_format" in issues
    ):
        return "manual_source_verification"

    if "requires_legal_source_verification" in issues:
        return "verify_against_reliable_legal_source"

    if any(issue.startswith("missing_metadata_") for issue in issues):
        return "review_metadata_completeness"

    return "manual_review_recommended"


# ==========================================================
# 7. AUDIT DATASET
# ==========================================================

def audit_dataset(records):

    chunk_locations, text_locations = find_duplicates(records)

    report_rows = []

    for index, item in enumerate(records):

        record = item["record"]

        issues = get_record_issues(record)

        # Duplicate chunk ID detection
        chunk_id = record.get("chunk_id")

        if not is_blank(chunk_id):

            key = (
                str(record.get("act_id", "")).strip(),
                str(chunk_id).strip(),
            )

            if len(chunk_locations[key]) > 1:
                issues.append("duplicate_chunk_id")

        # Duplicate text detection
        text = record.get("text", "")

        if isinstance(text, str) and not is_blank(text):

            normalized = normalize_text(text)

            if len(text_locations[normalized]) > 1:
                issues.append("possible_duplicate_text")

        issues = sorted(set(issues))

        if not issues:
            continue

        severity = classify_severity(issues)

        report_rows.append({
            "line_number": item["line_number"],
            "source_record_number": (
                record.get("preprocessing_info", {}).get(
                    "source_record_number", ""
                )
            ),
            "record_id": record.get("record_id", ""),
            "chunk_id": record.get("chunk_id", ""),
            "act_id": record.get("act_id", ""),
            "title": record.get("title", ""),
            "chapter": record.get("chapter", ""),
            "section_number": record.get("section_number", ""),
            "section_title": record.get("section_title", ""),
            "year": record.get("year", ""),
            "source_url": record.get("source_url", ""),
            "issues": "; ".join(issues),
            "severity": severity,
            "recommended_action": recommend_action(issues),
            "review_status": "not_reviewed",
        })

    return report_rows


# ==========================================================
# 8. SAVE QUALITY REPORT
# ==========================================================

def save_report(report_rows):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=REPORT_FIELDS,
        )

        writer.writeheader()
        writer.writerows(report_rows)

    print(f"\nQuality report saved to:\n{OUTPUT_FILE}")


# ==========================================================
# 9. GENERATE SUMMARY
# ==========================================================

def generate_summary(records, report_rows):

    issue_counts = Counter()
    severity_counts = Counter()

    for row in report_rows:

        severity_counts[row["severity"]] += 1

        for issue in row["issues"].split("; "):

            if issue:
                issue_counts[issue] += 1

    summary = {
        "total_records": len(records),
        "records_flagged": len(report_rows),
        "records_without_detected_issues": (
            len(records) - len(report_rows)
        ),
        "severity_counts": dict(severity_counts),
        "issue_counts": dict(issue_counts),
        "audit_note": (
            "Automated checks identify potential data-quality "
            "issues. They do not establish legal correctness "
            "or source authenticity."
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
            ensure_ascii=False,
            indent=4,
        )

    return summary


# ==========================================================
# 10. PRINT RESULTS
# ==========================================================

def print_summary(summary):

    print("\n" + "=" * 60)
    print("LEGAL DATASET QUALITY AUDIT")
    print("=" * 60)

    print("Total records examined:", summary["total_records"])

    print("Records flagged:", summary["records_flagged"])

    print(
        "Records without detected issues:",
        summary["records_without_detected_issues"],
    )

    print("\nSeverity summary:")

    for severity, count in sorted(
        summary["severity_counts"].items()
    ):
        print(f"{severity}: {count}")

    print("\nIssue counts:")

    for issue, count in sorted(
        summary["issue_counts"].items()
    ):
        print(f"{issue}: {count}")

    print("\nDetailed report:", OUTPUT_FILE)
    print("Summary report:", SUMMARY_FILE)

    print("=" * 60)
    print("QUALITY AUDIT COMPLETED")


# ==========================================================
# 11. MAIN
# ==========================================================

def main():

    print("Loading cleaned legal dataset...")

    records = load_dataset()

    print(f"Loaded {len(records)} records.")

    print("Performing quality checks...")

    report_rows = audit_dataset(records)

    save_report(report_rows)

    summary = generate_summary(records, report_rows)

    print_summary(summary)


if __name__ == "__main__":
    main()
