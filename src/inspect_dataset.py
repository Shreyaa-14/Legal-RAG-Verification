import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "legal_cleaned.jsonl"


def inspect_dataset():

    if not DATA_FILE.exists():
        print("Processed dataset not found.")
        print("Run preprocessing.py first.")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as file:

        first_line = file.readline()

    if not first_line:
        print("The processed dataset is empty.")
        return

    record = json.loads(first_line)

    print("First processed legal record")
    print("-" * 50)

    for key, value in record.items():

        if key == "text":
            print(f"{key}: {str(value)[:500]}...")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    inspect_dataset()