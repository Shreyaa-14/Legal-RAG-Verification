from datasets import load_dataset

print("Loading a small sample of Indian legal legislation...")

try:
    dataset = load_dataset(
        "vaquill/open-india-law",
        "legislation",
        split="train",
        streaming=True
    )

    # Read only 5 records for our initial test
    sample = list(dataset.take(5))

    print("\nDataset loaded successfully!")
    print("Number of records:", len(sample))

    for i, record in enumerate(sample, start=1):
        print(f"\nRecord {i}")
        print(record)

except Exception as e:
    print("\nSomething went wrong while loading the dataset.")
    print("Error:", e)