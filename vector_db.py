import json
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_chunk(file_path: str) -> list[dict]:
    with open(Path(file_path), "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_index(chunks: list[dict], model_name="sergeyzh/LaBSE-ru-turbo"):
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))

    df = pd.DataFrame(chunks)
    df["faiss_idx"] = range(len(df))
    return model, index, df


def retrieve(query, model, index, df, k=5):
    q_emb = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(q_emb.astype("float32"), k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        row = df.iloc[int(idx)]
        results.append({
            "score": float(dist),
            "source": row["source"],
            "chunk_id": row["chunk_id"],
            "chunk_index": int(row["chunk_index"]),
            "start_char": int(row["start_char"]),
            "end_char": int(row["end_char"]),
            "text": row["text"],
        })
    return results


if __name__ == "__main__":
    chunks = load_chunk("./chunks.jsonl")
    print(f"Loaded {len(chunks)} chunks")

    model, index, df = build_index(chunks)
    print(f"Index built: {index.ntotal} vectors")

    faiss.write_index(index, "faiss_index.bin")

    df.to_json(
        "chunks_metadata.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    print("Saved: faiss_index.bin, chunks_metadata.jsonl")