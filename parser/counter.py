from collections import Counter
import os

dir_list = os.listdir("C:\\DEV\\practicum\\architecture-llm\\files")
for file in dir_list:
    with open(
        f"C:\\DEV\\practicum\\architecture-llm\\files\\{file}", "r", encoding="utf-8"
    ) as f:
        text = f.read()
        print(
            file,
            ":",
            Counter([word for word in text.split() if word.istitle()]).most_common(15),
        )
        print()
