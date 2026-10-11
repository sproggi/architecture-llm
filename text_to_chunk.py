from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    """Один логический фрагмент документа с полной навигацией."""

    chunk_id: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int
    word_count: int
    token_count: int
    text: str
    section: str | None = None


class DocumentChunker:
    """
    Режет документы на чанки 100–300 слов (~500–1000 токенов),
    сохраняя источник и точную позицию в оригинале.
    """

    def __init__(
        self,
        chunk_size_tokens: int = 800,
        chunk_overlap_tokens: int = 100,
        min_words: int = 100,
        max_words: int = 300,
        encoding_name: str = "cl100k_base",
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.encoding_name = encoding_name

        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding_name,
            chunk_size=chunk_size_tokens,
            chunk_overlap=chunk_overlap_tokens,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            keep_separator=True,
        )

    def _count_tokens(self, text: str) -> int:
        import tiktoken

        enc = tiktoken.get_encoding(self.encoding_name)
        return len(enc.encode(text))

    def _find_char_offset(
        self, source_text: str, chunk_text: str, search_from: int
    ) -> int:
        """
        Ищет позицию чанка в исходном тексте, начиная с search_from.
        Fallback — приблизительное совпадение по началу чанка.
        """
        idx = source_text.find(chunk_text, search_from)
        if idx != -1:
            return idx
        probe = chunk_text[:80].strip()
        idx = source_text.find(probe, search_from)
        return idx if idx != -1 else search_from

    def split_document(
        self,
        text: str,
        source: str,
        section: str | None = None,
    ) -> List[Chunk]:
        """Разбивает один документ и возвращает список чанков."""
        pieces = self.splitter.split_text(text)

        chunks: List[Chunk] = []
        cursor = 0
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue

            start = self._find_char_offset(text, piece, cursor)
            end = start + len(piece)
            cursor = end

            wc = len(piece.split())
            tc = self._count_tokens(piece)

            if wc < self.min_words and i != len(pieces) - 1:
                if chunks:
                    prev = chunks[-1]
                    prev.text = f"{prev.text}\n{piece}"
                    prev.end_char = end
                    prev.word_count = len(prev.text.split())
                    prev.token_count = self._count_tokens(prev.text)
                    continue

            chunks.append(
                Chunk(
                    chunk_id=f"{source}::{i}",
                    source=source,
                    chunk_index=i,
                    start_char=start,
                    end_char=end,
                    word_count=wc,
                    token_count=tc,
                    text=piece,
                    section=section,
                )
            )
        return chunks

    def split_many(
        self,
        documents: Iterable[Dict[str, Any]],
    ) -> List[Chunk]:
        """
        documents: iterable of {"text": str, "source": str, "section": str | None}
        """
        out: List[Chunk] = []
        for doc in documents:
            out.extend(
                self.split_document(
                    text=doc["text"],
                    source=doc["source"],
                    section=doc.get("section"),
                )
            )
        return out


if __name__ == "__main__":
    docs = []
    for path in Path("./knowledge_base").glob("*.txt"):
        docs.append(
            {
                "text": path.read_text(encoding="utf-8"),
                "source": str(path.resolve()),
                "section": None,
            }
        )

    chunker = DocumentChunker(chunk_size_tokens=800, chunk_overlap_tokens=100)
    chunks = chunker.split_many(docs)

    for c in chunks[:3]:
        d = asdict(c)
        print(
            f"[{d['chunk_id']}] "
            f"{d['word_count']}w / {d['token_count']}t "
            f"chars {d['start_char']}–{d['end_char']}"
        )
        print(d["text"][:200], "...\n")

    # Экспорт для векторной БД / RAG
    import json

    with open("chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
