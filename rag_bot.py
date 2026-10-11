
import os
from dataclasses import dataclass
from pathlib import Path

import faiss
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sergeyzh/LaBSE-ru-turbo")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MAX_DISTANCE_THRESHOLD = float(os.getenv("MAX_DISTANCE_THRESHOLD", "1.2"))

TOP_K = int(os.getenv("TOP_K", "4"))


@dataclass
class RetrievedChunk:
    score: float
    source: str
    chunk_id: str
    chunk_index: int
    start_char: int
    end_char: int
    text: str

    def citation(self) -> str:
        fname = Path(self.source).name
        return f"[{fname}, фрагмент #{self.chunk_index}, симв. {self.start_char}–{self.end_char}]"



FEW_SHOT_EXAMPLES = [
    {
        "q": "Как зовут сестру Darmeliya",
        "a": (
            "1. Ищу упоминание о сестре Darmeliya в предоставленных фрагментах.\n"
            "2. В документе упоминается, что Darmeliya была подругой Zavira.\n"
            "3. Следовательно, ответ — Zavira."
            "[manual.pdf, фрагмент #12, симв. 340–512]"
        ),
    },
    {
        "q": "в каком году родился Andrew Hosen",
        "a": (
            "1. Ищу упоминание Andrew Hosen.\n"
            "2. В источнике сказано: «Andrew Hosen — это персонаж комиксов, родившийся в Москве в 2327 году».\n"
            "3. Следовательно, ответ — 2327. "
            "[andrew_hosen.txt, фрагмент #0, симв. 30011–31798]"
        ),
    },
]


SYSTEM_PROMPT = """Ты — ассистент, который отвечает ТОЛЬКО на основе предоставленных фрагментов базы знаний.

Правила:
1. Сначала размышляй по шагам (Chain-of-Thought): явно нумеруй шаги 1., 2., 3. …
2. Затем дай итоговый ответ.
3. Каждое утверждение подкрепляй ссылкой на источник в формате [имя_файла, фрагмент #N, симв. A–B].
4. Если во фрагментах нет ответа — напиши: «Я не знаю. В базе знаний нет информации по этому вопросу.»
   Не выдумывай факты и не опирайся на внешние знания.
5. Отвечай на языке пользователя.
"""

USER_PROMPT_TEMPLATE = """Примеры (Few-shot):

{examples}

---

Контекст из базы знаний:
{context}

---

Вопрос пользователя: {question}

Ответ (сначала шаги рассуждения, потом итог, со ссылками на источники):"""


class RAGBot:
    def __init__(
        self,
        index_path: str = "faiss_index.bin",
        metadata_path: str = "chunks.jsonl",
    ):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = faiss.read_index(index_path)
        self.df = pd.read_json(metadata_path, lines=True, orient="records")

        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )

    def retrieve(self, query: str, k: int = TOP_K) -> list[RetrievedChunk]:
        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        distances, indices = self.index.search(q_emb, k)

        results: list[RetrievedChunk] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            row = self.df.iloc[int(idx)]
            results.append(
                RetrievedChunk(
                    score=float(dist),
                    source=str(row["source"]),
                    chunk_id=str(row["chunk_id"]),
                    chunk_index=int(row["chunk_index"]),
                    start_char=int(row["start_char"]),
                    end_char=int(row["end_char"]),
                    text=str(row["text"]),
                )
            )
        return results

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[Фрагмент {i}] {c.citation()}\n{c.text}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _format_examples(examples: list[dict]) -> str:
        return "\n\n".join(
            f"Q: {ex['q']}\nA: {ex['a']}" for ex in examples
        )

    def answer(self, query: str, k: int = TOP_K) -> dict:
        chunks = self.retrieve(query, k=k)

        if not chunks or chunks[0].score > MAX_DISTANCE_THRESHOLD:
            return {
                "answer": "Я не знаю. В базе знаний нет информации по этому вопросу.",
                "sources": [],
                "used_llm": False,
                "best_score": chunks[0].score if chunks else None,
            }

        context = self._format_context(chunks)
        examples = self._format_examples(FEW_SHOT_EXAMPLES)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            examples=examples,
            context=context,
            question=query,
        )

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        answer_text = response.choices[0].message.content

        return {
            "answer": answer_text,
            "sources": [
                {
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "score": c.score,
                }
                for c in chunks
            ],
            "used_llm": True,
            "best_score": chunks[0].score,
        }

