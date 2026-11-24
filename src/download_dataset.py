from datasets import load_dataset
import json
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where we will save the downloaded records
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "legal_sample.jsonl"

# Number of records to collect for our starter dataset
NUMBER_OF_RECORDS = 500


def download_legal_data():

    print("Loading Indian legal legislation dataset...")

    dataset = load_dataset(
        "vaquill/open-india-law",
        "legislation",
        split="train",
        streaming=True
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        for record in dataset.take(NUMBER_OF_RECORDS):

            # Save one record per line
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            count += 1

            if count % 50 == 0:
                print(f"Saved {count} records...")

    print("\nDataset collection completed!")
    print("Total records saved:", count)
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    download_legal_data()