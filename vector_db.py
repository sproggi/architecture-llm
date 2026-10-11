import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd


def load_chunk(file_path: str) -> list[dict]:
    with open(Path(file_path), "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    return chunks


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


def retrieve(query: str, model, index, df, k=5):
    q_emb = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(q_emb.astype("float32"), k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        row = df.iloc[int(idx)]
        results.append(
            {
                "score": float(dist),
                "source": row["source"],
                "chunk_id": row["chunk_id"],
                "chunk_index": row["chunk_index"],
                "start_char": row["start_char"],
                "end_char": row["end_char"],
                "text": row["text"],
            }
        )
    return results


# ===== Запуск =====
if __name__ == "__main__":
    # 1. Чанкинг
    chunks = load_chunk("./chunks.jsonl")
    print(f"Created {len(chunks)} chunks")

    # 2. Индексация
    model, index, df = build_index(chunks)
    print(f"Index built: {index.ntotal} vectors")

    # 3. Сохранение
    faiss.write_index(index, "faiss_index.bin")
    df_metadata = pd.read_json("chunks.jsonl", lines=True, orient="records")

    # 4. Поиск
    results = retrieve("How to search?", model, index, df)
    for r in results[:2]:
        print(f"\n[{r['chunk_id']}] {r['text'][:100]}...")
