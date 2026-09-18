# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "faiss-cpu",
#   "httpx",
#   "numpy",
# ]
# ///

import json
from pathlib import Path

import faiss
import httpx
import numpy as np


JSONL_FILE = Path("uc_prod_2026-09-15_1147_UTC.jsonl")
INDEX_FILE = Path("uc_metadata.index")
METADATA_FILE = Path("uc_metadata.json")

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text:latest"

BATCH_SIZE = 16
REQUEST_TIMEOUT = 300


def load_records() -> list[dict]:
    if not JSONL_FILE.exists():
        raise FileNotFoundError(
            f"JSONL file not found: {JSONL_FILE.resolve()}"
        )

    records = []

    with JSONL_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            text = record.get("text")

            if not isinstance(text, str) or not text.strip():
                print(
                    f"Skipping line {line_number}: "
                    "missing or empty text field"
                )
                continue

            records.append(record)

    if not records:
        raise RuntimeError("No valid metadata records were found.")

    return records


def create_embeddings(
    records: list[dict],
) -> np.ndarray:
    all_embeddings = []
    total = len(records)

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch = records[start:end]

            texts = [
                record["text"]
                for record in batch
            ]

            response = client.post(
                OLLAMA_URL,
                json={
                    "model": EMBEDDING_MODEL,
                    "input": texts,
                    "truncate": True,
                    "keep_alive": "10m",
                },
            )

            response.raise_for_status()

            payload = response.json()
            embeddings = payload.get("embeddings")

            if not embeddings:
                raise RuntimeError(
                    f"No embeddings returned for records "
                    f"{start + 1} to {end}."
                )

            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Expected {len(batch)} embeddings, "
                    f"but received {len(embeddings)}."
                )

            all_embeddings.extend(embeddings)

            print(
                f"Embedded {end}/{total} records"
            )

    matrix = np.asarray(
        all_embeddings,
        dtype=np.float32,
    )

    if matrix.ndim != 2:
        raise RuntimeError(
            f"Unexpected embedding shape: {matrix.shape}"
        )

    if matrix.shape[0] != total:
        raise RuntimeError(
            f"Expected {total} embedding vectors, "
            f"but created {matrix.shape[0]}."
        )

    return matrix


def build_faiss_index(
    embeddings: np.ndarray,
) -> faiss.Index:
    # Normalize vectors so inner product becomes cosine similarity.
    faiss.normalize_L2(embeddings)

    dimensions = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimensions)
    index.add(embeddings)

    return index


def save_metadata(records: list[dict]) -> None:
    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
        )


def main() -> None:
    print(f"Reading: {JSONL_FILE}")

    records = load_records()

    print(f"Records loaded: {len(records)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Batch size: {BATCH_SIZE}")

    embeddings = create_embeddings(records)

    print(
        f"Embedding matrix: "
        f"{embeddings.shape[0]} records x "
        f"{embeddings.shape[1]} dimensions"
    )

    index = build_faiss_index(embeddings)

    if index.ntotal != len(records):
        raise RuntimeError(
            f"FAISS contains {index.ntotal} vectors, "
            f"but metadata contains {len(records)} records."
        )

    faiss.write_index(
        index,
        str(INDEX_FILE),
    )

    save_metadata(records)

    print("\nIndex creation completed")
    print(f"FAISS vectors: {index.ntotal}")
    print(f"FAISS index: {INDEX_FILE.resolve()}")
    print(f"Metadata file: {METADATA_FILE.resolve()}")


if __name__ == "__main__":
    main()