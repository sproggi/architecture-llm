
from rag_bot import RAGBot

BANNER = """
==================================================
  RAG-бот готов. Задайте вопрос.
  Команды: /exit  /sources  /debug
==================================================
"""


def main():
    print("Загружаю индекс и модель…")
    bot = RAGBot()
    print(BANNER)

    show_sources = False
    debug = False

    while True:
        try:
            query = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            break

        if not query:
            continue
        if query == "/exit":
            break
        if query == "/sources":
            show_sources = not show_sources
            print(f"[источники: {'вкл' if show_sources else 'выкл'}]")
            continue
        if query == "/debug":
            debug = not debug
            print(f"[debug: {'вкл' if debug else 'выкл'}]")
            continue

        result = bot.answer(query)

        print(f"\nБот:\n{result['answer']}\n")

        if debug:
            print(f"[debug] used_llm={result['used_llm']} "
                  f"best_score={result['best_score']}")
        if show_sources and result["sources"]:
            print("Источники:")
            for s in result["sources"]:
                print(f"  - {s['source']} #{s['chunk_index']} "
                      f"(симв. {s['start_char']}–{s['end_char']}, "
                      f"score={s['score']:.3f})")
            print()


if __name__ == "__main__":
    main()